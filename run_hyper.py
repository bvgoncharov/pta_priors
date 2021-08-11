#!/bin/python

import pandas as pd
import numpy as np
import sys
import os
import bilby
import inspect
import optparse
from enterprise_warp import enterprise_warp
from enterprise_warp import bilby_warp
from enterprise_warp.enterprise_warp import get_noise_dict
from enterprise_warp import results
from enterprise_extensions import hypermodel

import ppta_dr2_models
import hierarchical_models as hm

def parse_commandline():
  """
  Parsing command line arguments for action on results
  """

  parser = optparse.OptionParser()

  parser.add_option("-p", "--prfile", help="Parameter file", type=str)

  parser.add_option("-r", "--result", help="Output directory or a parameter \
                    file. In case of individual pulsar analysis, specify a \
                    directory that contains subdirectories with individual \
                    pulsar results. In case of an array analysis, specify a \
                    directory with result files.", \
                    default=None, type=str)

  # Parameters below are not important, added for compatibility with old code:

  parser.add_option("-i", "--info", help="Print information about all results. \
                    In case \"-n\" is specified, print an information about \
                    results for a specific pulsar.", \
                    default=0, type=int)

  parser.add_option("-n", "--name", help="Pulsar name or number (or \"all\")", \
                    default="all", type=str)

  parser.add_option("-a", "--par", help="Include only model parameters that \
                    contain \"par\" (more than one could be added)",
                    action="append", default=None, type=str)

  parser.add_option("-s", "--load_separated", help="Attempt to load separated \
                    chain files with names chain_DATETIME(14-symb)_PARS.txt. \
                    If --par are supplied, load only files with --par \
                    columns.", default=0, type=int)

  parser.add_option("-m", "--mpi_regime", \
                    help="Here only for compatibility, set as 2.",
                    default=2, type=int)

  opts, args = parser.parse_args()

  return opts

class HyperResult(results.BilbyWarpResult):

  def __init__(self, opts):
    super(HyperResult, self).__init__(opts)
    self.results = []
    self.chains = []
    self.log_zs = []

  def main_pipeline(self):
    for psr_dir in self.psr_dirs:

      self.psr_dir = psr_dir
      success = self._scan_psr_output()
      if not success:
        continue

      success = self.load_chains()
      if not success:
        continue

      self.standardize_chain_for_rn_hyper_pe()

      self.results.append(self.result)
      self.chains.append(self.result.posterior)
      self.log_zs.append(self.result.log_evidence)

  def standardize_chain_for_rn_hyper_pe(self):
    for key in self.chain.keys():
      if 'red_noise' not in key and \
         'log_likelihood' not in key and \
         'log_prior' not in key and \
         'prior' not in key:
        del self.chain[key]
      elif 'red_noise_gamma' in key:
        self.chain['red_noise_gamma'] = self.chain.pop(key)
      elif 'red_noise_log10_A' in key:
        self.chain['red_noise_log10_A'] = self.chain.pop(key)
        

opts = parse_commandline()

configuration = hm.HierarchicalInferenceParams
params = enterprise_warp.Params(opts.prfile,opts=opts,custom_models_obj=configuration)

hr = HyperResult(opts)

hr.main_pipeline()

label = params.paramfile_label
outdir = hr.outdir_all + label + '/'
if not os.path.exists(outdir):
  raise ValueError('Output directory does not exist: '+outdir)

print('Output directory: ', outdir)

hp_likelihood = bilby.hyper.likelihood.HyperparameterLikelihood(
    posteriors=hr.chains, hyper_prior=hm.__dict__[params.model],
    #sampling_prior=run_prior, 
    log_evidences=hr.log_zs, max_samples=params.max_samples_from_measurement)

hp_priors = hm.__dict__['hp_'+params.model](params)

result = bilby.core.sampler.run_sampler(
     likelihood=hp_likelihood, priors=hp_priors,
     use_ratio=False, outdir=outdir, label=params.paramfile_label,
     verbose=True, clean=True, sampler=params.sampler, **params.sampler_kwargs)

result.plot_corner()

# hp_likelihood.parameters['lg_A'] = -30.0
# hp_likelihood.parameters['gam'] = 0.
# hp_likelihood.log_likelihood()
