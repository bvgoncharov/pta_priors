"""
Runs importance sampling:
1. --result: parameter file to load results with proposal likelihood per pulsar;
2. --target: parameter file to construct target likelihood per pulsar;
3. --prfile: overall parameter file;

Example: python run_importance.py --result "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_snall_to_recycle_20210626.dat" --prfile "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_hpe_unif_prod_lg_A_gamma_set_g1_20211011_1.dat" --target "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_snall_cpl_fixgam_30_nf_to_recycle_20210811.dat"
"""
import os
import copy
#import pickle

import bilby

from enterprise_warp import enterprise_warp, bilby_warp, results

import epta_models

import hierarchical_models as hm
import importance_sampling as im

n_psr = 26 # total number of pulsars (to-do: get this from parameters)
opts = hm.parse_commandline()

custom = epta_models.EPTAModels
configuration = hm.HierarchicalInferenceParams
params = enterprise_warp.Params(opts.prfile,opts=opts,custom_models_obj=configuration)
opts.exclude = params.exclude

# Loading results from sampling proposal likelihood per pulsar (no CP)
#params = enterprise_warp.Params(opts.prfile,opts=opts,custom_models_obj=configuration)
hr = im.ImportanceResult(opts)#, suffix=params.par_suffix)
hr.main_pipeline()

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
if True:
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
is_likelihood = im.__dict__[params.importance_likelihood](hr.chains, obj_likelihoods_targ, sp, hr.log_zs, max_samples=params.max_samples_from_measurement, stl_file=outdir+'precomp_unmarg_targ_lnl.npy', grid_size=params.grid_size) #sp, hr.log_zs, max_samples=2)

result = bilby.core.sampler.run_sampler(
     likelihood=is_likelihood, priors=hp_priors,
     use_ratio=False, outdir=outdir, label=params.paramfile_label,
     verbose=True, clean=True, sampler=params.sampler, soft_init=True, **params.sampler_kwargs)

result.plot_corner()
