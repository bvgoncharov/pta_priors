"""
Runs importance sampling:
1. --result: parameter file to load results with proposal likelihood per pulsar;
2. --target: parameter file to construct target likelihood per pulsar;
3. --prfile: overall parameter file;

Example 1, simple:
python run_importance.py --result "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_snall_to_recycle_20210626.dat" --prfile "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_hpe_unif_prod_lg_A_gamma_set_g1_20211011_1.dat" --target "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_snall_cpl_fixgam_30_nf_to_recycle_20210811.dat"

Example 2, running on singularity and saving iterations:
singularity exec --bind "/fred/oz031/epta_code_image/image_content/:$HOME" /fred/oz031/epta_code_image/EPTA_ENTERPRISE.sif python3 /home/bgonchar/pta_gwb_priors/analytical_importance.py --result "/home/bgonchar/epta_dr3/params/epta_dr3_snall_cpfg_to_recycle_20230314.dat" --target "/home/bgonchar/epta_dr3/params/epta_dr3_snall_cpfg_to_recycle_20230314.dat" --prfile "/home/bgonchar/pta_gwb_priors/params/epta_dr2_is_rn_set_all_cppropl_20230316.dat" --n_grid_iter 1000 --save_iterations 0

When all iterations are saved, run the same command without --save_iterations 0 to proceed to plotting.
"""
import os
import sys
sys.path.insert(0, "/home/celestialsapien/enterprise_warp-dev")
import copy
import time
import tqdm
#import pickle
import numpy as np
from scipy import interpolate
from scipy.interpolate import griddata
from scipy.integrate import simps
from mpmath import mp
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif",
  #"font.serif": ["Palatino"],
})
font = {'family' : 'serif',
        'size'   : 17}
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

def cred_lvl_from_analytical_dist(xx,yy,lvl=[0.159,0.841]):
  yy = yy/simps(yy,x=xx)
  yy_cdf = np.array([simps(yy[0:ii],x=xx[0:ii]) for ii in range(1,len(xx))])
  return [(np.abs(yy_cdf - val)).argmin() for val in lvl]

import bilby

from enterprise_warp import enterprise_warp, bilby_warp, results

import epta_models

import hierarchical_models as hm
import importance_sampling as im

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

opts = hm.parse_commandline()

custom = epta_models.EPTAModels
configuration = hm.HierarchicalInferenceParams
params = enterprise_warp.Params(opts.prfile,opts=opts,custom_models_obj=configuration)
opts.exclude = params.exclude

# Loading results from sampling proposal likelihood per pulsar (no CP)
#params = enterprise_warp.Params(opts.prfile,opts=opts,custom_models_obj=configuration)
hr = im.ImportanceResult(opts)#, suffix=params.par_suffix)
hr.main_pipeline()
for ii, chain in enumerate(hr.chains):
  print(chain.keys()[0].split('_')[0],': ',len(chain),'samples available')
  # Remove pulsar names from recycled sample names
  # that are related to the main target likelihood parameter
  # (later on we need a unique name for all pulsars)
  if params.par_suffix=='red_noise':
    rename_dict = {cn: cn[len(cn.split('_')[0])+1:] \
                   for cn in chain.columns if params.par_suffix in cn}
    print('Renaming posterior parameter names: ', \
          ', '.join([key+' > '+val for key, val in rename_dict.items()]))
    hr.chains[ii] = chain.rename(columns=rename_dict)

n_psr = 25 # 26 # total number of pulsars (to-do: get this from parameters)
print('\n[Warning!] By-hand specification of a total number of pulsar is needed! Current n_psr = ', n_psr, '\n')

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
print('Output directory: ', outdir)
if not os.path.exists(outdir):
  raise ValueError('Output directory does not exist.')

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

    # Remove pulsar names from target parameter name
    # (later on we need a unique name for all pulsars)
    for pp in pta[0].params:
      if params.par_suffix in pp.name and params.par_suffix=='red_noise':
        print('Renaming target likelihood parameter: ',pp.name,' > ',pp.name[len(pp.name.split('_')[0])+1:])
        pp.name = pp.name[len(pp.name.split('_')[0])+1:]

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
  is_likelihood = im.__dict__[params.importance_likelihood](hr.chains, obj_likelihoods_targ, sp, hr.log_zs, max_samples=params.max_samples_from_measurement, stl_file=outdir+'precomp_unmarg_targ_lnl.npy', grid_size=params.grid_size, save_iterations=opts.save_iterations, suffix=params.par_suffix, parname=params.parname, qc_range=params.qc_range) #sp, hr.log_zs, max_samples=2)

save_publ_plots = True
overplot_publ_plots = '/home/celestialsapien/epta_dr2_out/20230314_epta_trim_trim_rcl_cpfg/is_rn_all_20230316/' #None #'/fred/oz031/pta_gwb_priors_out/dr2_timing_20200607/20210125_snall_cpl_fixgam_30_nf_rcl_cpfg/is_rn_all_20220204/' # '/fred/oz031/pta_gwb_priors_out/dr2_timing_20200607/20210125_snall_cpl_fixgam_30_nf_rcl_cpfg/is_rn_xg3_20220112/' # dir or None

#ref_log10_A = -13.3 # simulation
#ref_log10_A = -13.8 # simulation for comments
#ref_log10_A = -14.66
#ref_log10_A = -14.72 # NANOGrav
ref_log10_A = -14.6 # EPTA 25 pulsar full

#ref_sigma_log10_A = 0.5 # simulation
ref_sigma_log10_A = 0.

#color_sequence = ['#F1F1F1','#C5E3EC','#AADDEC','#90D5EC'] # PPTA, grey-to-blue
#color_sequence = ['#ceb5a7', '#e9f7ca', '#f7d488', '#f9a03f'] # NANOGrav, sand-green-orange
color_sequence = ['#F4EEE0','#6D5D6E','#4F4557','#393646']

#lims_2d = [[-14.5,-12.5],[0.02004008,1.7]] # simulation original
#lims_2d = [[-14.5,-12.5],[0.02004008,1.7]] # simulation for comments
lims_2d = [[-16,-14],[0.02004008,1.25]]

ref_gamma = 13/3 # simulation and data

ref_sigma_gamma = 0.

lims_2d_gam = [[2,6],[0.02004008, 2.0]]

if 'mu_lg_A' in hp_priors.keys() and 'sig_lg_A' in hp_priors.keys():
  xx = np.linspace(hp_priors['mu_lg_A'].minimum,hp_priors['mu_lg_A'].maximum,params.grid_size)
  yy = np.linspace(hp_priors['sig_lg_A'].minimum,hp_priors['sig_lg_A'].maximum,params.grid_size)
  X, Y = np.meshgrid(xx,yy)
  X_shape, Y_shape = X.shape, Y.shape
  X_flat, Y_flat = X.flatten(), Y.flatten()
  print('Total samples: ', len(X_flat))
  likelihood_grid_files = np.array([outdir + 'likelihood_on_a_grid_' + str(ii) + '.npy' for ii in range(int(len(X_flat)/opts.n_grid_iter))])
  likelihood_grid_files_exist = np.array([os.path.exists(lgf) for lgf in likelihood_grid_files])
  if (opts.save_iterations < 0) and not os.path.exists(outdir + 'likelihood_on_a_grid.npy') and np.all(likelihood_grid_files_exist):
    log_likelihood_flat = np.empty(len(X_flat))
    for ii, lgf in enumerate(likelihood_grid_files):
      log_likelihood_flat[ii*opts.n_grid_iter:(ii+1)*opts.n_grid_iter] = np.load(lgf)
    np.save(outdir + 'likelihood_on_a_grid.npy', log_likelihood_flat)
  elif np.any(likelihood_grid_files_exist) and not os.path.exists(outdir + 'likelihood_on_a_grid.npy'):
    print('Missing likelihood-on-a-grid files:')
    for lgf in np.array(likelihood_grid_files)[~likelihood_grid_files_exist]:
      print(lgf)
    if opts.incomplete:
      log_likelihood_flat = np.empty(len(X_flat))
      for ii, lgf in enumerate(likelihood_grid_files):
        if os.path.exists(lgf):
          log_likelihood_flat[ii*opts.n_grid_iter:(ii+1)*opts.n_grid_iter] = np.load(lgf)
        else:
          log_likelihood_flat[ii*opts.n_grid_iter:(ii+1)*opts.n_grid_iter] = np.repeat(-100,opts.n_grid_iter)
      np.save(outdir + 'likelihood_on_a_grid.npy', log_likelihood_flat)
  if os.path.exists(outdir + 'likelihood_on_a_grid.npy'):
    log_likelihood_flat = np.load(outdir + 'likelihood_on_a_grid.npy')
    zv1 = log_likelihood_flat.reshape(X_shape)
    # Evidence calculation
    #xy_limits = ((-17.5,-13.5),(0.02004008, 5.)) # real data < 2000 samples
    #xy_limits = ((-15.8,-14.2),(0.02004008, 1.5))
    #xy_limits = ((-18.0,-12.0),(0.02004008, 10.)) # simulation 100 samples
    xy_limits = ((-16.,-12.),(0.02004008, 4.)) # simulation 1600 samples
    evobj = im.AnalyticalEvidence2D(zv1,(X,Y),xy_limits)
    log_z = evobj.logz()
    zz = evobj.z()

    zv1[np.isnan(zv1)] = np.min(log_likelihood_flat[~np.isnan(log_likelihood_flat)]) # To replace nans by minimum values

    # 2D plot
    fig = plt.figure()
    axes = fig.add_subplot(111)
    plt.imshow(zv1,origin='lower',extent=[-20.,-10.,0.,10.],vmin=1323000, vmax=1327000)
    axes.set_xlabel('$\mu_{\log_{10} A}$', fontdict=font)
    axes.set_ylabel('$\sigma_{\log_{10} A}$', fontdict=font)
    plt.colorbar(label='$\log\mathcal{L}$')
    axes.tick_params(axis='y', labelsize = font['size'])
    axes.tick_params(axis='x', labelsize = font['size'])
    plt.tight_layout()
    plt.savefig(outdir + 'logL-noise-1.png')
    plt.close()
    # 1D slice plots
    plt.figure()
    for ii in [0,125,250,325,499]:
      plt.plot(Y[:,ii],zv1[:,ii],linestyle='-',label='$\mu_{lgA}='+str(X[0,ii])+'$')
    plt.xscale('log')
    plt.legend()
    plt.xlabel('$\sigma_{lgA}$')
    plt.ylabel('$\log_L$')
    plt.tight_layout()
    plt.savefig(outdir + 'logL-noise-1d.png')
    plt.close()
    # 1D zoomed
    for ii in [125,250,265,325]:
      plt.semilogy(Y[1:,ii],zv1[1:,ii],linestyle='-',label='$mu_lg_A='+str(X[0,ii])+'$')
    plt.xlabel('$\sigma_{lgA}$')
    plt.ylabel('$\log_L$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir + 'logL-noise-1d-2.png')
    plt.close()
    # 1D zoomed 2
    for ii in [125,245,250,255]:
      plt.semilogy(Y[1:25,ii],zv1[1:25,ii],linestyle='-',label='$mu_lg_A='+str(X[0,ii])+'$')
    plt.xlabel('$\sigma_{lgA}$')
    plt.ylabel('$\log_L$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir + 'logL-noise-1d-3.png')
    plt.close()
    # 1D zoomed 4
    for ii in [334,299]:
      plt.semilogy(Y[1:25,ii],zv1[1:25,ii]/simps(zv1[1:25,ii],x=Y[1:25,ii]),linestyle='-',label='$mu_lg_A='+str(X[0,ii])+'$')
    plt.xlabel('$\sigma_{lgA}$')
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
    #max_zv_for_norm = np.max([np.max(zv1[1:25,ii]) for ii in [125,245,250,255]])
    #zv1_flat[zv1_flat > max_zv_for_norm] = max_zv_for_norm
    mp.prec = 170
    mp_zv1 = mp.matrix(zv1_flat)
    exp_zv1_flat = mp.matrix([[mp.e**val for val in mp_zv1]])
    exp_zv1_flat_posterior = exp_zv1_flat*(1/(evobj.xx[0,-1]-evobj.xx[0,0]))*(1/(evobj.yy[-1,0]-evobj.yy[0,0]))/zz # - max(exp_zv1_flat) + 1e6
    exp_zv1_posterior = np.array([float(val) for val in exp_zv1_flat_posterior]).reshape(zv1s)

    # 1D zoomed 5 - for 1000 samples PPTA DR2
    #idx_cp = 266 # PPTA DR2 results
    idx_cp = 333 # simulation injection -13.3
    sig_5 = Y[1:101,idx_cp]
    post_5 = exp_zv1_posterior[1:101,idx_cp]
    fig = plt.figure()
    axes = fig.add_subplot(111)
    plt.plot(sig_5, post_5,label='$\mu_{\log_{10}A}='+str(X[0,idx_cp])+'$')
    axes.set_xlabel('$\sigma_{\log_{10}A}$', fontdict=font)
    axes.set_ylabel('Posterior probability at fixed $\mu_{\log_{10}A}$', fontdict=font)
    axes.tick_params(axis='y', labelsize = font['size'])
    axes.tick_params(axis='x', labelsize = font['size'])
    plt.xlabel('$\sigma_{\log_{10}A}$')
    plt.ylabel('Posterior probability at fixed $\mu_{\log_{10}A}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir + 'logL-noise-1d-5.png')
    plt.close()

    # 2D plot posterior probability
    fig = plt.figure()
    axes = fig.add_subplot(111)
    im = plt.imshow(exp_zv1_posterior,origin='lower',extent=[params.mu_lg_A[0],params.mu_lg_A[1],params.sig_lg_A[0], params.sig_lg_A[1]], cmap=plt.get_cmap('viridis'))#,vmin=1323000, vmax=1327000)
    ##plt.axvline(ref_log10_A,linestyle='--',color='red')
    ##plt.axhline(2,linestyle='--',color='red')
    cb = plt.colorbar()
    cb.set_label(label='Posterior probability',size=font['size'],family=font['family'])
    cb.ax.tick_params(labelsize=font['size'])
    axes.set_xlabel('$\mu_{\log_{10}A}$', fontdict=font)
    axes.set_ylabel('$\sigma_{\log_{10}A}$', fontdict=font)
    axes.tick_params(axis='y', labelsize = font['size'])
    axes.tick_params(axis='x', labelsize = font['size'])
    plt.tight_layout()
    plt.savefig(outdir + 'logL-noise-posterior.png')
    plt.close()

    plt.imshow(np.log(exp_zv1_posterior),origin='lower',extent=[params.mu_lg_A[0],params.mu_lg_A[1],params.sig_lg_A[0], params.sig_lg_A[1]], cmap=plt.get_cmap('viridis'),vmin=np.max(np.log(exp_zv1_posterior))-10, vmax=np.max(np.log(exp_zv1_posterior)))
    ##plt.axvline(ref_log10_A,linestyle='--',color='red')
    #plt.axhline(2,linestyle='--',color='red')
    plt.xlabel('$\mu_{\log_{10}A}$')
    plt.ylabel('$\sigma_{\log_{10}A}$')
    plt.colorbar(label='Log posterior probability')
    plt.tight_layout()
    plt.savefig(outdir + 'logL-noise-log-posterior.png')
    plt.close()

    # 2D plot posterior probability - zoomed in at evidence integration boundaries
    # zooming in:
    exp_zv1_posterior_z = exp_zv1_posterior[:,evobj.mask[0]]
    exp_zv1_posterior_z = exp_zv1_posterior_z[evobj.mask[1],:]
    # determining credible levels:
    # Credit: https://stackoverflow.com/questions/37890550/python-plotting-percentile-contour-lines-of-a-probability-distribution
    exp_zv1_posterior_z_norm = exp_zv1_posterior_z / exp_zv1_posterior_z.sum()
    nn=1000
    tt = np.linspace(0, exp_zv1_posterior_z_norm.max(), nn)
    integral = ((exp_zv1_posterior_z_norm >= tt[:, None, None]) * exp_zv1_posterior_z_norm).sum(axis=(1,2))
    ff = interpolate.interp1d(integral, tt)
    t_contours = ff(np.array([0.997,0.954,0.682]))
    # plotting:
    fig = plt.figure()
    axes = fig.add_subplot(111)
    img1 = plt.imshow(exp_zv1_posterior_z,origin='lower',extent=np.array(xy_limits).flatten(),cmap=get_continuous_cmap(color_sequence)) #['#F1F1F1','#C5E3EC','#AADDEC','#90D5EC']))
    cb = plt.colorbar(orientation="horizontal", fraction=0.05, pad=0.2)
    #plt.set_cmap('cividis')
    img2 = plt.contour(exp_zv1_posterior_z_norm, t_contours, extent=np.array(xy_limits).flatten(), colors='C0', linewidths=1.5)
    if overplot_publ_plots is not None:
      other_1 = np.loadtxt(overplot_publ_plots + 'logL-noise-posterior_z_1.txt')
      other_2 = np.loadtxt(overplot_publ_plots + 'logL-noise-posterior_z_2.txt')
      img3 = plt.contour(other_1, other_2, extent=np.array(xy_limits).flatten(), colors='C1', linewidths=1.5)
    #plt.xlim(xy_limits[0])
    #plt.ylim(xy_limits[1])
    plt.xlim(lims_2d[0])
    plt.ylim(lims_2d[1])
    cb.set_label(label='Posterior probability',size=font['size'],family=font['family'])
    cb.ax.tick_params(labelsize=font['size'])
    plt.axvline(ref_log10_A,linestyle='--',color='black',linewidth=0.5)
    plt.axhline(ref_sigma_log10_A,linestyle='--',color='black',linewidth=0.5)
    axes.set_xlabel('$\mu_{\log_{10}A}$', fontdict=font)
    axes.set_ylabel('$\sigma_{\log_{10}A}$', fontdict=font)
    axes.set_xticks([-16,-15,-14])
    axes.tick_params(axis='y', labelsize = font['size'])
    axes.tick_params(axis='x', labelsize = font['size'])
    #plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(outdir + 'logL-noise-posterior_z.png')
    plt.savefig(outdir + 'logL-noise-posterior_z.pdf')
    plt.close()

    if save_publ_plots:
      np.savetxt(outdir + 'logL-noise-posterior_z_1.txt', \
                 exp_zv1_posterior_z_norm)
      np.savetxt(outdir + 'logL-noise-posterior_z_2.txt', \
                 t_contours)

    # Marginalized posteriors
    mu_marg_over_sig = simps(exp_zv1_posterior,axis=0,x=yy)
    sig_marg_over_mu = simps(exp_zv1_posterior,axis=1,x=xx)

    fig = plt.figure()
    axes = fig.add_subplot(111)
    plt.plot(xx, mu_marg_over_sig)
    plt.axvline(ref_log10_A,linestyle='--',color='red')
    axes.set_xlabel('$\mu_{\log_{10}A}$', fontdict=font)
    axes.set_ylabel('Marginalised posterior', fontdict=font)
    axes.tick_params(axis='y', labelsize = font['size'])
    axes.tick_params(axis='x', labelsize = font['size'])
    plt.savefig(outdir + 'logL-noise-posterior-mu.png')
    plt.close()

    fig = plt.figure()
    axes = fig.add_subplot(111)
    plt.plot(yy[1:], sig_marg_over_mu[1:])
    plt.axvline(ref_sigma_log10_A,linestyle='--',color='red')
    axes.set_xlabel('$\sigma_{\log_{10}A}$', fontdict=font)
    axes.set_ylabel('Marginalised posterior', fontdict=font)
    axes.tick_params(axis='y', labelsize = font['size'])
    axes.tick_params(axis='x', labelsize = font['size'])
    plt.savefig(outdir + 'logL-noise-posterior-sig.png')
    plt.close()

    print('Savage-Dickey Bayes factor for zero sigma, unzoomed: ', sig_marg_over_mu[1]/(yy[-1]-yy[1]))

    # Marginalized zoomed posteriors
    mu_marg_over_sig_z = simps(exp_zv1_posterior_z,axis=0,x=yy[evobj.mask[1]])
    sig_marg_over_mu_z = simps(exp_zv1_posterior_z,axis=1,x=xx[evobj.mask[0]])

    # Savage-Dickey odds ratio
    bf_zero_over_nonzero = sig_marg_over_mu_z[0]/(yy[evobj.mask[1]][-1]-yy[evobj.mask[1]][0])
    print('Savage-Dickey Bayes factor for zero sigma, zoomed: ', bf_zero_over_nonzero)

    lvl = cred_lvl_from_analytical_dist(xx[evobj.mask[0]], mu_marg_over_sig_z)
    print('Maximum-aposteriori value of sigma_log10_A: ', xx[evobj.mask[0]][np.argmax(mu_marg_over_sig_z)])
    print('One-sigma credible levels: ', xx[evobj.mask[0]][lvl[0]], xx[evobj.mask[0]][lvl[1]])
    fig = plt.figure()
    axes = fig.add_subplot(111)
    plt.plot(xx[evobj.mask[0]], mu_marg_over_sig_z)
    plt.axvline(ref_log10_A,linestyle='--',color='red')
    for lv in lvl:
      plt.axvline(xx[evobj.mask[0]][lv],linewidth=0.5,color='black')
    axes.set_xlabel('$\mu_{\log_{10}A}$', fontdict=font)
    axes.set_ylabel('Marginalised posterior', fontdict=font)
    axes.tick_params(axis='y', labelsize = font['size'])
    axes.tick_params(axis='x', labelsize = font['size'])
    plt.savefig(outdir + 'logL-noise-posterior-mu-z.png')
    plt.close()

    fig = plt.figure()
    axes = fig.add_subplot(111)
    plt.plot(yy[evobj.mask[1]], sig_marg_over_mu_z, color='C0', label='DR2full')
    if overplot_publ_plots is not None:
      # Overplotting other result (e.g. with fewer pulsars)
      other = np.loadtxt(overplot_publ_plots + 'logL-noise-posterior-sig-z.txt')
      plt.plot(other[:,0], other[:,1], color='C1', linewidth=1.5, label='DR2new')
      plt.tight_layout()
    plt.axvline(ref_sigma_log10_A,color='black',linestyle='--',linewidth=1.5)
    if np.log(bf_zero_over_nonzero) < -3:
      lvl = cred_lvl_from_analytical_dist(yy[evobj.mask[1]], sig_marg_over_mu_z)
      print('Maximum-aposteriori value of sigma_log10_A: ', yy[evobj.mask[1]][np.argmax(sig_marg_over_mu_z)])
      print('One-sigma credible levels: ', yy[evobj.mask[1]][lvl[0]], yy[evobj.mask[1]][lvl[1]])
      mask_fill = (yy[evobj.mask[1]] >= yy[evobj.mask[1]][lvl[0]]) * (yy[evobj.mask[1]] <= yy[evobj.mask[1]][lvl[1]])
      #plt.fill_between(yy[evobj.mask[1]], sig_marg_over_mu_z, where=mask_fill, color=color_sequence[-1])
      #for lv in lvl:
      #  plt.axvline(yy[evobj.mask[1]][lv],linestyle='--',linewidth=0.5,color='black')
    else:
      lvl = cred_lvl_from_analytical_dist(yy[evobj.mask[1]], sig_marg_over_mu_z, lvl=[0.95])
      print('Upper limit on sigma_log10_A at 95% credibility: ', yy[evobj.mask[1]][lvl[0]])
      mask_fill = (yy[evobj.mask[1]] >= 0) * (yy[evobj.mask[1]] <= yy[evobj.mask[1]][lvl[0]])
      #plt.fill_between(yy[evobj.mask[1]], sig_marg_over_mu_z, where=mask_fill, color=color_sequence[-1])
      plt.axvline(yy[evobj.mask[1]][lvl[0]],linestyle='--',linewidth=1.5,color='black')
    axes.set_xlabel('$\sigma_{\log_{10}A}$', fontdict=font)
    axes.set_ylabel('Marginalised posterior', fontdict=font)
    axes.tick_params(axis='y', labelsize = font['size'])
    axes.tick_params(axis='x', labelsize = font['size'])
    plt.xlim(lims_2d[1])
    plt.ylim([0,np.max(sig_marg_over_mu_z)+0.2])
    prop = font_manager.FontProperties(**font)
    axes.legend(prop=prop)
    plt.tight_layout()
    plt.savefig(outdir + 'logL-noise-posterior-sig-z.png')
    plt.savefig(outdir + 'logL-noise-posterior-sig-z.pdf')
    plt.close()

    if save_publ_plots:
      np.savetxt(outdir + 'logL-noise-posterior-sig-z.txt', \
                 np.array([yy[evobj.mask[1]],sig_marg_over_mu_z]).T)

    # Effective number of samples
    #for ii, XY_ii in tqdm.tqdm(enumerate(zip(X_flat, Y_flat)), total=len(X_flat)):
    #  is_likelihood.parameters = {
    #    'mu_lg_A': XY_ii[0],
    #    'sig_lg_A': XY_ii[1],
    #    'low_lg_A': -20.,
    #    'high_lg_A': -10.,
    #  }
    #  #log_likelihood_flat[ii] = is_likelihood.log_likelihood()
    #  import ipdb; ipdb.set_trace()

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
      np.save(outdir + 'likelihood_on_a_grid.npy', log_likelihood_flat) 
    exit()

if 'mu_gam' in hp_priors.keys() and 'sig_gam' in hp_priors.keys():
  xx = np.linspace(hp_priors['mu_gam'].minimum,hp_priors['mu_gam'].maximum,params.grid_size)
  yy = np.linspace(hp_priors['sig_gam'].minimum,hp_priors['sig_gam'].maximum,params.grid_size)
  X, Y = np.meshgrid(xx,yy)
  X_shape, Y_shape = X.shape, Y.shape
  X_flat, Y_flat = X.flatten(), Y.flatten()
  print('Total samples: ', len(X_flat))
  likelihood_grid_files = np.array([outdir + 'likelihood_on_a_grid_' + str(ii) + '.npy' for ii in range(int(len(X_flat)/opts.n_grid_iter))])
  likelihood_grid_files_exist = np.array([os.path.exists(lgf) for lgf in likelihood_grid_files])
  if (opts.save_iterations < 0) and not os.path.exists(outdir + 'likelihood_on_a_grid.npy') and np.all(likelihood_grid_files_exist):
    log_likelihood_flat = np.empty(len(X_flat))
    for ii, lgf in enumerate(likelihood_grid_files):
      log_likelihood_flat[ii*opts.n_grid_iter:(ii+1)*opts.n_grid_iter] = np.load(lgf)
    np.save(outdir + 'likelihood_on_a_grid.npy', log_likelihood_flat)
  elif np.any(likelihood_grid_files_exist) and not os.path.exists(outdir + 'likelihood_on_a_grid.npy'):
    print('Missing likelihood-on-a-grid files:')
    for lgf in np.array(likelihood_grid_files)[~likelihood_grid_files_exist]:
      print(lgf)
    if opts.incomplete:
      log_likelihood_flat = np.empty(len(X_flat))
      for ii, lgf in enumerate(likelihood_grid_files):
        if os.path.exists(lgf):
          log_likelihood_flat[ii*opts.n_grid_iter:(ii+1)*opts.n_grid_iter] = np.load(lgf)
        else:
          log_likelihood_flat[ii*opts.n_grid_iter:(ii+1)*opts.n_grid_iter] = np.repeat(-100,opts.n_grid_iter)
      np.save(outdir + 'likelihood_on_a_grid.npy', log_likelihood_flat)
  if os.path.exists(outdir + 'likelihood_on_a_grid.npy'):
    log_likelihood_flat = np.load(outdir + 'likelihood_on_a_grid.npy')
    zv1 = log_likelihood_flat.reshape(X_shape)
    # Evidence calculation
    #xy_limits = ((-17.5,-13.5),(0.02004008, 5.)) # real data < 2000 samples
    #xy_limits = ((-15.8,-14.2),(0.02004008, 1.5))
    #xy_limits = ((-18.0,-12.0),(0.02004008, 10.)) # simulation 100 samples
    xy_limits = ((1.,9.),(0.02004008, 4.)) # simulation 1600 samples
    evobj = im.AnalyticalEvidence2D(zv1,(X,Y),xy_limits)
    log_z = evobj.logz()
    zz = evobj.z()

    zv1[np.isnan(zv1)] = np.min(log_likelihood_flat[~np.isnan(log_likelihood_flat)]) # To replace nans by minimum values

    # To plot unlogged likelihood
    zv1s = zv1.shape
    zv1_flat = zv1.flatten()
    # Create a mask for values that are too large - numerical artifacts (look at previous plots).
    # This is for normalization by the largest value. Values too small will be turned to -inf naturally.
    #max_zv_for_norm = np.max([np.max(zv1[1:25,ii]) for ii in [125,245,250,255]])
    #zv1_flat[zv1_flat > max_zv_for_norm] = max_zv_for_norm
    mp.prec = 170
    mp_zv1 = mp.matrix(zv1_flat)
    exp_zv1_flat = mp.matrix([[mp.e**val for val in mp_zv1]])
    exp_zv1_flat_posterior = exp_zv1_flat*(1/(evobj.xx[0,-1]-evobj.xx[0,0]))*(1/(evobj.yy[-1,0]-evobj.yy[0,0]))/zz # - max(exp_zv1_flat) + 1e6
    exp_zv1_posterior = np.array([float(val) for val in exp_zv1_flat_posterior]).reshape(zv1s)

    # 2D plot posterior probability - zoomed in at evidence integration boundaries
    # zooming in:
    exp_zv1_posterior_z = exp_zv1_posterior[:,evobj.mask[0]]
    exp_zv1_posterior_z = exp_zv1_posterior_z[evobj.mask[1],:]
    # determining credible levels:
    # Credit: https://stackoverflow.com/questions/37890550/python-plotting-percentile-contour-lines-of-a-probability-distribution
    exp_zv1_posterior_z_norm = exp_zv1_posterior_z / exp_zv1_posterior_z.sum()
    nn=1000
    tt = np.linspace(0, exp_zv1_posterior_z_norm.max(), nn)
    integral = ((exp_zv1_posterior_z_norm >= tt[:, None, None]) * exp_zv1_posterior_z_norm).sum(axis=(1,2))
    ff = interpolate.interp1d(integral, tt)
    t_contours = ff(np.array([0.997,0.954,0.682]))
    # plotting:
    fig = plt.figure()
    axes = fig.add_subplot(111)
    img1 = plt.imshow(exp_zv1_posterior_z,origin='lower',extent=np.array(xy_limits).flatten(),cmap=get_continuous_cmap(color_sequence)) #['#F1F1F1','#C5E3EC','#AADDEC','#90D5EC']))
    cb = plt.colorbar()
    #plt.set_cmap('cividis')
    img2 = plt.contour(exp_zv1_posterior_z_norm, t_contours, extent=np.array(xy_limits).flatten(), colors='black', linewidths=0.5)
    #plt.xlim(xy_limits[0])
    #plt.ylim(xy_limits[1])
    plt.xlim(lims_2d_gam[0])
    plt.ylim(lims_2d_gam[1])
    cb.set_label(label='Posterior probability',size=font['size'],family=font['family'])
    cb.ax.tick_params(labelsize=font['size'])
    plt.axvline(ref_gamma,linestyle='--',color='black',linewidth=0.5)
    plt.axhline(ref_sigma_gamma,linestyle='--',color='black',linewidth=0.5)
    axes.set_xlabel('$\mu_{\log_{10}A}$', fontdict=font)
    axes.set_ylabel('$\sigma_{\log_{10}A}$', fontdict=font)
    axes.tick_params(axis='y', labelsize = font['size'])
    axes.tick_params(axis='x', labelsize = font['size'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(outdir + 'logL-noise-posterior_z.png')
    plt.savefig(outdir + 'logL-noise-posterior_z.pdf')
    plt.close()

    # Marginalized posteriors
    mu_marg_over_sig = simps(exp_zv1_posterior,axis=0,x=yy)
    sig_marg_over_mu = simps(exp_zv1_posterior,axis=1,x=xx)

    fig = plt.figure()
    axes = fig.add_subplot(111)
    plt.plot(xx, mu_marg_over_sig)
    plt.axvline(ref_gamma,linestyle='--',color='red')
    axes.set_xlabel('$\mu_{\log_{10}A}$', fontdict=font)
    axes.set_ylabel('Marginalized posterior', fontdict=font)
    axes.tick_params(axis='y', labelsize = font['size'])
    axes.tick_params(axis='x', labelsize = font['size'])
    plt.savefig(outdir + 'logL-noise-posterior-mu.png')
    plt.close()

    fig = plt.figure()
    axes = fig.add_subplot(111)
    plt.plot(yy[1:], sig_marg_over_mu[1:])
    plt.axvline(ref_sigma_gamma,linestyle='--',color='red')
    axes.set_xlabel('$\sigma_{\log_{10}A}$', fontdict=font)
    axes.set_ylabel('Marginalized posterior', fontdict=font)
    axes.tick_params(axis='y', labelsize = font['size'])
    axes.tick_params(axis='x', labelsize = font['size'])
    plt.savefig(outdir + 'logL-noise-posterior-sig.png')
    plt.close()

    print('Savage-Dickey Bayes factor for zero sigma, unzoomed: ', sig_marg_over_mu[1]/(yy[-1]-yy[1]))

    # Marginalized zoomed posteriors
    mu_marg_over_sig_z = simps(exp_zv1_posterior_z,axis=0,x=yy[evobj.mask[1]])
    sig_marg_over_mu_z = simps(exp_zv1_posterior_z,axis=1,x=xx[evobj.mask[0]])

    # Savage-Dickey odds ratio
    bf_zero_over_nonzero = sig_marg_over_mu_z[0]/(yy[evobj.mask[1]][-1]-yy[evobj.mask[1]][0])
    print('Savage-Dickey Bayes factor for zero sigma, zoomed: ', bf_zero_over_nonzero)

    lvl = cred_lvl_from_analytical_dist(xx[evobj.mask[0]], mu_marg_over_sig_z)
    print('Maximum-aposteriori value of sigma_log10_A: ', xx[evobj.mask[0]][np.argmax(mu_marg_over_sig_z)])
    print('One-sigma credible levels: ', xx[evobj.mask[0]][lvl[0]], xx[evobj.mask[0]][lvl[1]])
    fig = plt.figure()
    axes = fig.add_subplot(111)
    plt.plot(xx[evobj.mask[0]], mu_marg_over_sig_z)
    plt.axvline(ref_gamma,linestyle='--',color='red')
    for lv in lvl:
      plt.axvline(xx[evobj.mask[0]][lv],linewidth=0.5,color='black')
    axes.set_xlabel('$\mu_{\log_{10}A}$', fontdict=font)
    axes.set_ylabel('Marginalized posterior', fontdict=font)
    axes.tick_params(axis='y', labelsize = font['size'])
    axes.tick_params(axis='x', labelsize = font['size'])
    plt.savefig(outdir + 'logL-noise-posterior-mu-z.png')
    plt.close()

    fig = plt.figure()
    axes = fig.add_subplot(111)
    plt.plot(yy[evobj.mask[1]], sig_marg_over_mu_z, color=color_sequence[-1])
    plt.axvline(ref_sigma_gamma,linestyle='--',color='black',linewidth=0.5)
    if np.log(bf_zero_over_nonzero) < -3:
      lvl = cred_lvl_from_analytical_dist(yy[evobj.mask[1]], sig_marg_over_mu_z)
      print('Maximum-aposteriori value of sigma_log10_A: ', yy[evobj.mask[1]][np.argmax(sig_marg_over_mu_z)])
      print('One-sigma credible levels: ', yy[evobj.mask[1]][lvl[0]], yy[evobj.mask[1]][lvl[1]])
      mask_fill = (yy[evobj.mask[1]] >= yy[evobj.mask[1]][lvl[0]]) * (yy[evobj.mask[1]] <= yy[evobj.mask[1]][lvl[1]])
      plt.fill_between(yy[evobj.mask[1]], sig_marg_over_mu_z, where=mask_fill, color=color_sequence[-1])
      #for lv in lvl:
      #  plt.axvline(yy[evobj.mask[1]][lv],linestyle='--',linewidth=0.5,color='black')
    else:
      lvl = cred_lvl_from_analytical_dist(yy[evobj.mask[1]], sig_marg_over_mu_z, lvl=[0.95])
      print('Upper limit on sigma_log10_A at 95% credibility: ', yy[evobj.mask[1]][lvl[0]])
      mask_fill = (yy[evobj.mask[1]] >= 0) * (yy[evobj.mask[1]] <= yy[evobj.mask[1]][lvl[0]])
      plt.fill_between(yy[evobj.mask[1]], sig_marg_over_mu_z, where=mask_fill, color=color_sequence[-1])
      plt.axvline(yy[evobj.mask[1]][lvl[0]],linestyle='--',linewidth=0.5,color='black')
    axes.set_xlabel('$\sigma_{\log_{10}A}$', fontdict=font)
    axes.set_ylabel('Marginalised posterior', fontdict=font)
    axes.tick_params(axis='y', labelsize = font['size'])
    axes.tick_params(axis='x', labelsize = font['size'])
    plt.xlim(lims_2d_gam[1])
    plt.ylim([0,np.max(sig_marg_over_mu_z)+0.2])
    plt.tight_layout()
    plt.savefig(outdir + 'logL-noise-posterior-sig-z.png')
    plt.savefig(outdir + 'logL-noise-posterior-sig-z.pdf')
    plt.close()

  else:
    if opts.save_iterations >= 0:
      t0 = time.time()
      X_flat_iter = X_flat[opts.save_iterations*opts.n_grid_iter:(opts.save_iterations+1)*opts.n_grid_iter]
      Y_flat_iter = Y_flat[opts.save_iterations*opts.n_grid_iter:(opts.save_iterations+1)*opts.n_grid_iter]
      log_likelihood_iter = np.empty(len(X_flat_iter))
      for ii, XY_ii in tqdm.tqdm(enumerate(zip(X_flat_iter, Y_flat_iter)), total=len(X_flat_iter)):
        is_likelihood.parameters = {
          'mu_gam': XY_ii[0], #X_flat[opts.save_iterations],
          'sig_gam': XY_ii[1], #Y_flat[opts.save_iterations],
          'low_gam': 0.,
          'high_gam': 10.,
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
          'mu_gam': XY_ii[0],
          'sig_gam': XY_ii[1],
          'low_gam': 0.,
          'high_gam': 10.,
        }
        log_likelihood_flat[ii] = is_likelihood.log_likelihood()
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
    fig = plt.figure()
    axes = fig.add_subplot(111)
    plt.plot(xx,log_likelihood_flat + np.log(1/(xx[-1]-xx[0])) - float(log_z))
    plt.xlabel('gw log10 A')
    plt.ylabel('log likelihood')
    plt.savefig(outdir+'logL-signal.png')
    plt.close()
    # zoomed in
    up_idx = 135 # Real data PPTA DR2
    #up_idx = 200 # Simulation

    fig = plt.figure()
    axes = fig.add_subplot(111)
    plt.plot(xx[0:up_idx],log_likelihood_flat[0:up_idx] + np.log(1/(xx[-1]-xx[0])) - float(log_z), label='Signal target posterior')
    plt.plot(result_obj.x_vals, np.log(result_obj.prob_factorized_norm),label='Factorized (ApJL)')
    #plt.axvline(ref_log10_A, color='red', label='Simulated value')
    plt.ylim([-16, 4])
    plt.xlim([-20,-12])
    axes.set_xlabel('$\log_{10}A$', fontdict=font)
    axes.set_ylabel('Posterior probability', fontdict=font)
    axes.tick_params(axis='y', labelsize = font['size'])
    axes.tick_params(axis='x', labelsize = font['size'])
    #plt.legend()
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

# import ipdb; ipdb.set_trace()
