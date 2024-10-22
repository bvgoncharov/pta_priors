import os
import bilby

from .utils import hierarchical_models as hm

def main():

  opts = hm.parse_commandline()
  
  configuration = hm.HierarchicalInferenceParams
  params = enterprise_warp.Params(opts.prfile,opts=opts, \
          custom_models_obj=configuration)
  
  hr = hm.HyperResult(opts, suffix=params.par_suffix)
  
  hr.main_pipeline()
  
  label = params.paramfile_label
  outdir = hr.outdir_all + label + '/'
  if not os.path.exists(outdir):
    raise ValueError('Output directory does not exist: '+outdir)
  
  print('Output directory: ', outdir)
  
  sp = hm.__dict__[params.model](suffix=params.par_suffix)
  
  hp_likelihood = bilby.hyper.likelihood.HyperparameterLikelihood(
      posteriors=hr.chains, 
      hyper_prior=sp, #hm.__dict__[params.model](suffix=params.par_suffix),
      #sampling_prior=run_prior, 
      log_evidences=hr.log_zs, max_samples=params.max_samples_from_measurement)
  
  hp_priors = hm.__dict__['hp_'+params.model](params)
  
  result = bilby.core.sampler.run_sampler(
       likelihood=hp_likelihood, priors=hp_priors,
       use_ratio=False, outdir=outdir, label=params.paramfile_label,
       verbose=True, clean=True, sampler=params.sampler, **params.sampler_kwargs)
  
  result.plot_corner()

if __name__=='__main__':
  main()
