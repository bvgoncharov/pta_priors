"""
Runs importance sampling:
1. --result: parameter file to load results with proposal likelihood per pulsar;
2. --target: parameter file to construct target likelihood per pulsar;
3. --prfile: overall parameter file;

Example: python run_importance.py --result "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_snall_to_recycle_20210626.dat" --prfile "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_hpe_unif_prod_lg_A_gamma_set_g1_20211011_1.dat" --target "/home/bgonchar/pta_gwb_priors/params/ppta_dr2_snall_cpl_fixgam_30_nf_to_recycle_20210811.dat"
"""
import copy

import bilby

from enterprise_warp import enterprise_warp, bilby_warp, results

import ppta_dr2_models

import hierarchical_models as hm
import importance_sampling as im

n_psr = 26 # total number of pulsars (to-do: get this from parameters)
opts = hm.parse_commandline()

custom = ppta_dr2_models.PPTADR2Models
configuration = hm.HierarchicalInferenceParams

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

# Loading target likelihoods to sample (with CP)
obj_likelihoods_targ = []
for ii in range(1,2):#n_psr):
  opts_ew.num = ii
  params = enterprise_warp.Params(opts_ew.target, opts=opts_ew, custom_models_obj=custom)
  pta = enterprise_warp.init_pta(params)
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

# Constructing Signal likelihood
params = enterprise_warp.Params(opts.prfile,opts=opts,custom_models_obj=configuration)
sp = hm.__dict__[params.model](suffix=params.par_suffix)
is_likelihood = im.ImportanceLikelihoodSignal(hr.chains, obj_likelihoods_targ, sp, hr.log_zs, max_samples=2)

hp_priors = hm.__dict__['hp_'+params.model](params)

result = bilby.core.sampler.run_sampler(
     likelihood=hp_likelihood, priors=hp_priors,
     use_ratio=False, outdir=outdir, label=params.paramfile_label,
     verbose=True, clean=True, sampler=params.sampler, **params.sampler_kwargs)

result.plot_corner()

import ipdb; ipdb.set_trace()
