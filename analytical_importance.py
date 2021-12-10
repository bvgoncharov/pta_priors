"""
Runs importance sampling:
1. --result: parameter file to load results with proposal likelihood per pulsar;
2. --target: parameter file to construct target likelihood per pulsar;
3. --prfile: overall parameter file;

Example: python run_importance.py --result "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_snall_to_recycle_20210626.dat" --prfile "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_hpe_unif_prod_lg_A_gamma_set_g1_20211011_1.dat" --target "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_snall_cpl_fixgam_30_nf_to_recycle_20210811.dat"
"""
import os
import copy
import time
import tqdm
#import pickle
import numpy as np
from scipy.interpolate import griddata
from scipy.integrate import simps
from mpmath import mp
import matplotlib.pyplot as plt

import bilby

from enterprise_warp import enterprise_warp, bilby_warp, results

import ppta_dr2_models

import hierarchical_models as hm
import importance_sampling as im

n_psr = 26 # total number of pulsars (to-do: get this from parameters)
opts = hm.parse_commandline()

custom = ppta_dr2_models.PPTADR2Models
configuration = hm.HierarchicalInferenceParams
params = enterprise_warp.Params(opts.prfile,opts=opts,custom_models_obj=configuration)
opts.exclude = params.exclude

# Loading results from sampling proposal likelihood per pulsar (no CP)
#params = enterprise_warp.Params(opts.prfile,opts=opts,custom_models_obj=configuration)
hr = im.ImportanceResult(opts)#, suffix=params.par_suffix)
hr.main_pipeline()
for chain in hr.chains:
  print(chain.keys()[0].split('_')[0],': ',len(chain),'samples available')
# Making opts from hierarchical_models compatible with enterprise_warp
# To make compatible with the next part but incompatible with the previous one
opts_ew = copy.copy(opts)
opts_ew.num = 0
opts_ew.drop = 0
opts_ew.clearcache = 0
opts_ew.mpi_regime = 0
opts_ew.wipe_old_output = 0

label = params.paramfile_label
outdir = hr.outdir_all + label + '/'
if not os.path.exists(outdir):
  raise ValueError('Output directory does not exist: '+outdir)

# Loading target likelihoods to sample (with CP)
#tlo_file = outdir + '/target_likelihood_objs.pkl'
#if os.path.exists(tlo_file):
#  print('Loading target likelihood objects, ', tlo_file)
#  with open(tlo_file,'rb') as tlf:
#    obj_likelihoods_targ = pickle.load(tlf)
#else:
if not os.path.exists(outdir + 'likelihood_on_a_grid.npy'):
  obj_likelihoods_targ = []
  for ii in range(n_psr):
    if ii in hr.excluded_nums:
      print('Excluding PSR ', ii, ' from target likelihoods')
      continue
    opts_ew.num = ii
    params_t = enterprise_warp.Params(opts_ew.target, opts=opts_ew, custom_models_obj=custom)
    pta = enterprise_warp.init_pta(params_t)
    priors = bilby_warp.get_bilby_prior_dict(pta[0])
    for pkey, prior in priors.items():
      if type(prior) == bilby.core.prior.analytical.Normal:
        if 'gamma' in pkey:
          priors[pkey].minimum = 0.
          priors[pkey].maximum = 10.
        elif 'log10_A' in pkey:
          priors[pkey].minimum = -20.
          priors[pkey].maximum = -6.
    parameters = dict.fromkeys(priors.keys())
    obj_likelihoods_targ.append(bilby_warp.PTABilbyLikelihood(pta[0],parameters))
  #with open(tlo_file, 'wb') as tlf:
  #  pickle.dump(obj_likelihoods_targ, tlf, pickle.HIGHEST_PROTOCOL)

# Hyper priors
hp_priors = hm.__dict__['hp_'+params.model](params)

# Priors (for quasi-common process, noise likelihood)
sp = hm.__dict__[params.model](suffix=params.par_suffix)

# Constructing Signal likelihood
if not os.path.exists(outdir + 'likelihood_on_a_grid.npy'):
  is_likelihood = im.__dict__[params.importance_likelihood](hr.chains, obj_likelihoods_targ, sp, hr.log_zs, max_samples=params.max_samples_from_measurement, stl_file=outdir+'precomp_unmarg_targ_lnl.npy', grid_size=params.grid_size, save_iterations=opts.save_iterations) #sp, hr.log_zs, max_samples=2)

if 'mu_lg_A' in hp_priors.keys() and 'sig_lg_A' in hp_priors.keys():
  xx = np.linspace(hp_priors['mu_lg_A'].minimum,hp_priors['mu_lg_A'].maximum,params.grid_size)
  yy = np.linspace(hp_priors['sig_lg_A'].minimum,hp_priors['sig_lg_A'].maximum,params.grid_size)
  X, Y = np.meshgrid(xx,yy)
  X_shape, Y_shape = X.shape, Y.shape
  X_flat, Y_flat = X.flatten(), Y.flatten()
  print('Total samples: ', len(X_flat))
  likelihood_grid_files = [outdir + 'likelihood_on_a_grid_' + str(ii) + '.npy' for ii in range(int(len(X_flat)/opts.n_grid_iter))]
  if (opts.save_iterations < 0) and not os.path.exists(outdir + 'likelihood_on_a_grid.npy') and np.all([os.path.exists(lgf) for lgf in likelihood_grid_files]):
    log_likelihood_flat = np.empty(len(X_flat))
    for ii, lgf in enumerate(likelihood_grid_files):
      log_likelihood_flat[ii*opts.n_grid_iter:(ii+1)*opts.n_grid_iter] = np.load(lgf)
    np.save(outdir + 'likelihood_on_a_grid.npy', log_likelihood_flat)
  if os.path.exists(outdir + 'likelihood_on_a_grid.npy'):
    log_likelihood_flat = np.load(outdir + 'likelihood_on_a_grid.npy')
    zv1 = log_likelihood_flat.reshape(X_shape)
    # Evidence calculation
    xy_limits = ((-17.5,-13.5),(0.02004008, 5.))
    evobj = im.AnalyticalEvidence2D(zv1,(X,Y),xy_limits)
    log_z = evobj.logz()
    zz = evobj.z()

    zv1[np.isnan(zv1)] = np.min(log_likelihood_flat[~np.isnan(log_likelihood_flat)]) # To replace nans by minimum values

    # 2D plot
    plt.imshow(zv1,origin='lower',extent=[-20.,-10.,0.,10.],vmin=1323000, vmax=1327000)
    plt.xlabel('mu_lg_A')
    plt.ylabel('sig_lg_A')
    plt.colorbar(label='log_L')
    plt.tight_layout()
    plt.savefig(outdir + 'logL-noise-1.png')
    plt.close()
    # 1D slice plots
    plt.figure()
    for ii in [0,125,250,325,499]:
      plt.plot(Y[:,ii],zv1[:,ii],linestyle='-',label='mu_lg_A='+str(X[0,ii]))
    plt.xscale('log')
    plt.legend()
    plt.xlabel('sigma_lg_A')
    plt.ylabel('log_L')
    plt.tight_layout()
    plt.savefig(outdir + 'logL-noise-1d.png')
    plt.close()
    # 1D zoomed
    for ii in [125,250,265,325]:
      plt.semilogy(Y[1:,ii],zv1[1:,ii],linestyle='-',label='mu_lg_A='+str(X[0,ii]))
    plt.xlabel('sigma_lg_A')
    plt.ylabel('log_L')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir + 'logL-noise-1d-2.png')
    plt.close()
    # 1D zoomed 2
    for ii in [125,245,250,255]:
      plt.semilogy(Y[1:25,ii],zv1[1:25,ii],linestyle='-',label='mu_lg_A='+str(X[0,ii]))
    plt.xlabel('sigma_lg_A')
    plt.ylabel('log_L')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir + 'logL-noise-1d-3.png')
    plt.close()
    # 1D zoomed 4
    for ii in [334,299]:
      plt.semilogy(Y[1:25,ii],zv1[1:25,ii]/simps(zv1[1:25,ii],x=Y[1:25,ii]),linestyle='-',label='mu_lg_A='+str(X[0,ii]))
    plt.xlabel('sigma_lg_A')
    plt.ylabel('Normalized probability')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir + 'logL-noise-1d-4.png')
    plt.close()

    # To plot unlogged likelihood
    zv1s = zv1.shape
    zv1_flat = zv1.flatten()
    # Create a mask for values that are too large - numerical artifacts (look at previous plots).
    # This is for normalization by the largest value. Values too small will be turned to -inf naturally.
    max_zv_for_norm = np.max([np.max(zv1[1:25,ii]) for ii in [125,245,250,255]])
    zv1_flat[zv1_flat > max_zv_for_norm] = max_zv_for_norm
    mp_zv1 = mp.matrix(zv1_flat)
    exp_zv1_flat = mp.matrix([[mp.e**val for val in mp_zv1]])
    exp_zv1_flat_posterior = exp_zv1_flat*(1/(evobj.xx[0,-1]-evobj.xx[0,0]))*(1/(evobj.yy[-1,0]-evobj.yy[0,0]))/zz # - max(exp_zv1_flat) + 1e6
    exp_zv1_posterior = np.array([float(val) for val in exp_zv1_flat_posterior]).reshape(zv1s)

    # 2D plot posterior probability
    plt.imshow(exp_zv1_posterior,origin='lower',extent=[-20.,-10.,0., 10.])#,vmin=1323000, vmax=1327000)
    plt.xlabel('mu_lg_A')
    plt.ylabel('sig_lg_A')
    plt.colorbar(label='Posterior probability')
    plt.tight_layout()
    plt.savefig(outdir + 'logL-noise-posterior.png')
    plt.close()

    # 2D plot posterior probability - zoomed in at evidence integration boundaries
    exp_zv1_posterior_z = exp_zv1_posterior[:,evobj.mask[0]]
    exp_zv1_posterior_z = exp_zv1_posterior_z[evobj.mask[1],:]
    plt.imshow(exp_zv1_posterior_z,origin='lower',extent=[-17.5,-13.5,0.02004008, 5.])#,vmin=1323000, vmax=1327000)
    plt.xlabel('mu_lg_A')
    plt.ylabel('sig_lg_A')
    plt.colorbar(label='Posterior probability')
    plt.tight_layout()
    plt.savefig(outdir + 'logL-noise-posterior_z.png')
    plt.close()

    # Marginalized posteriors
    mu_marg_over_sig = simps(exp_zv1_posterior,axis=0,x=yy)
    sig_marg_over_mu = simps(exp_zv1_posterior,axis=1,x=xx)

    plt.plot(xx, mu_marg_over_sig)
    plt.xlabel('mu_lg_A')
    plt.ylabel('Marginalized posterior')
    plt.savefig(outdir + 'logL-noise-posterior-mu.png')
    plt.close()

    plt.plot(yy, sig_marg_over_mu)
    plt.xlabel('sig_lg_A')
    plt.ylabel('Marginalized posterior')
    plt.savefig(outdir + 'logL-noise-posterior-sig.png')
    plt.close()

  else:
    if opts.save_iterations >= 0:
      t0 = time.time()
      X_flat_iter = X_flat[opts.save_iterations*opts.n_grid_iter:(opts.save_iterations+1)*opts.n_grid_iter]
      Y_flat_iter = Y_flat[opts.save_iterations*opts.n_grid_iter:(opts.save_iterations+1)*opts.n_grid_iter]
      log_likelihood_iter = np.empty(len(X_flat_iter))
      for ii, XY_ii in tqdm.tqdm(enumerate(zip(X_flat_iter, Y_flat_iter)), total=len(X_flat_iter)):
        is_likelihood.parameters = {
          'mu_lg_A': XY_ii[0], #X_flat[opts.save_iterations],
          'sig_lg_A': XY_ii[1], #Y_flat[opts.save_iterations],
          'low_lg_A': -20.,
          'high_lg_A': -10.,
        }
        #log_likelihood_iter.append( is_likelihood.log_likelihood() )
        log_likelihood_iter[ii] = is_likelihood.log_likelihood()
      t1 = time.time()
      print('Elapsed time: ',t1-t0)
      np.save(outdir + 'likelihood_on_a_grid_' + str(opts.save_iterations) + '.npy', log_likelihood_iter)
      print('Saved ', opts.save_iterations, ' in ', outdir)
    else:
      log_likelihood_flat = np.empty(len(X_flat))
      for ii, XY_ii in tqdm.tqdm(enumerate(zip(X_flat, Y_flat)), total=len(X_flat)):
        is_likelihood.parameters = {
          'mu_lg_A': XY_ii[0],
          'sig_lg_A': XY_ii[1],
          'low_lg_A': -20.,
          'high_lg_A': -10.,
        }
        log_likelihood_flat[ii] = is_likelihood.log_likelihood()
        print(ii,'/',len(X_flat))
      np.save(outdir + 'likelihood_on_a_grid.npy', log_likelihood_flat) 
    exit()
elif 'gw_log10_A' in hp_priors.keys():
  xx = np.linspace(hp_priors['gw_log10_A'].minimum, hp_priors['gw_log10_A'].maximum, params.grid_size)
  likelihood_grid_files = [outdir + 'likelihood_on_a_grid_' + str(ii) + '.npy' for ii in range(len(xx))]
  if (opts.save_iterations < 0) and not os.path.exists(outdir + 'likelihood_on_a_grid.npy') and np.all([os.path.exists(lgf) for lgf in likelihood_grid_files]):
    log_likelihood_flat = np.empty(len(xx))
    for ii, lgf in enumerate(likelihood_grid_files):
      log_likelihood_flat[ii] = np.load(lgf)
    np.save(outdir + 'likelihood_on_a_grid.npy', log_likelihood_flat)
  elif np.any([os.path.exists(lgf) for lgf in likelihood_grid_files]):
    mf = np.array(likelihood_grid_files)[~np.array([os.path.exists(lgf) for lgf in likelihood_grid_files])]
    print('Partial samples of gw_log10_A are found. Missing files: \n', mf)
  if os.path.exists(outdir + 'likelihood_on_a_grid.npy'):

    log_likelihood_flat = np.load(outdir + 'likelihood_on_a_grid.npy')

    # Evidence evaluation
    #log_l_mp = mp.matrix(log_likelihood_flat)
    #l_mp = mp.matrix([[mp.e**val for val in log_l_mp]])
    #int_l = mp.fsum(mp.matrix([[(l_mp[ii]+l_mp[ii-1])/2 for ii in range(1,len(l_mp))]]))
    # Times integration step, times constant uniform prior
    #log_z = mp.log(int_l*(xx[1]-xx[0])/(xx[-1]-xx[0]))
    log_z = im.AnalyticalEvidence1D(log_likelihood_flat,xx).logz()
    print('log_Z = ', float(log_z))

    # For a specific range
    xx1 = xx[53:140] # -13.5 -17.5
    log_l_mp_1 = mp.matrix(log_likelihood_flat[53:140])
    l_mp_1 = mp.matrix([[mp.e**val for val in log_l_mp_1]])
    int_l_1 = mp.fsum(mp.matrix([[(l_mp_1[ii]+l_mp_1[ii-1])/2 for ii in range(1,len(l_mp_1))]]))
    log_z_1 = mp.log(int_l_1*(xx1[1]-xx1[0])/(xx1[-1]-xx1[0]))

    # Loading old results to compare
    result = '/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_snall_wnfix_pe_common_pl_factorized_30_nf_20210126.dat'
    from enterprise_warp.results import parse_commandline
    from make_factorized_posterior import FactorizedPosteriorResult
    psrs_set = '/home/bgonchar/pta_gwb_priors/params/pulsar_set_gx3.dat'
    #psrs_set = '/home/bgonchar/correlated_noise_pta_2020/params/pulsar_set_all.dat'
    opts = type('test', (object,), {})()
    opts.__dict__['logbf'] = True
    opts.__dict__['result'] = result
    opts.__dict__['par'] = ['gw']
    opts.__dict__['name'] = 'all'
    opts.__dict__['load_separated'] = False
    opts.__dict__['info'] = False
    result_obj = FactorizedPosteriorResult(opts, psrs_set=psrs_set)
    result_obj.main_pipeline([-20., -6.], plot_results = False, plot_psrs=False)

    # Likelihood times prior over evidence
    plt.plot(xx,log_likelihood_flat + np.log(1/(xx[-1]-xx[0])) - float(log_z))
    plt.xlabel('gw_log10_A')
    plt.ylabel('log_likelihood')
    plt.savefig(outdir+'logL-signal.png')
    plt.close()
    # zoomed in
    up_idx = 135 # Real data PPTA DR2
    #up_idx = 200 # Simulation
    plt.plot(xx[0:up_idx],log_likelihood_flat[0:up_idx] + np.log(1/(xx[-1]-xx[0])) - float(log_z), label='Signal target posterior')
    plt.plot(result_obj.x_vals, np.log(result_obj.prob_factorized_norm),label='Factorized (ApJL)')
    #plt.axvline(-13.3, color='red', label='Simulated value')
    plt.ylim([-16, 4])
    plt.xlim([-20,-12])
    plt.xlabel('gw_log10_A')
    plt.ylabel('log_likelihood')
    plt.legend()
    plt.savefig(outdir+'logL-signal-2.png')
    plt.close()
  else:
    t0 = time.time()
    if opts.save_iterations >= 0:
      is_likelihood.parameters = {
        'gw_log10_A': xx[opts.save_iterations],
      }
      log_likelihood_iter = is_likelihood.log_likelihood()
      t1 = time.time()
      print('Elapsed time: ',t1-t0)
      np.save(outdir + 'likelihood_on_a_grid_' + str(opts.save_iterations) + '.npy', log_likelihood_iter)
      print('Saved ', opts.save_iterations, ' in ', outdir)
    else:
      log_likelihood_flat = np.empty(len(xx))
      for ii, xx_ii in enumerate(xx):
        is_likelihood.parameters = {
          'gw_log10_A': xx_ii,
        }
        log_likelihood_flat[ii] = is_likelihood.log_likelihood()
        print(ii,'/',len(xx))
        t1 = time.time()
        print('Elapsed time: ',t1-t0)
      np.save(outdir + 'likelihood_on_a_grid.npy', log_likelihood_flat)
    exit()
  pass

import ipdb; ipdb.set_trace()
