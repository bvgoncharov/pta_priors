
import os.path
from multiprocessing import Pool

from scipy.integrate import simps
import numpy as np

from enterprise_warp import results

from bilby.hyper.model import Model
from bilby.core.likelihood import Likelihood

class ImportanceLikelihoodSignal(Likelihood):
  """
  posteriors: list with pandas.DataFrame, bilby output with proposal likelihoods
  obj_likelihoods_targ: list of bilby_warp.PTABilbyLikelihood with target likelihoods
  log_evidences: list of float, log evidence for proposal likelihoods
  """
  def __init__(self, posteriors, obj_likelihoods_targ, prior, 
               log_evidences, multiproc=False, npool=2, max_samples=1e100,
               stl_file="", grid_size=300):

    #if not isinstance(prior, Model):
    #  prior = Model([prior])

    super(ImportanceLikelihoodSignal, self).__init__({})#prior.parameters)

    self.multiproc = multiproc
    if multiproc:
      self.pool = Pool(npool)

    self.prior = prior
    self.posteriors = [posterior.sample(max_samples) for posterior in posteriors]
    self.obj_likelihoods_targ = obj_likelihoods_targ

    # self.data are cleaned posteriors turned into a list n_psr x n_samples with
    # elements dict with posterior samples
    self.data = []
    self.log_likelihoods_proposal = []
    for posterior in self.posteriors:
      self.data.append( posterior.to_dict(orient='records') )
      self.log_likelihoods_proposal.append( posterior['log_likelihood'].tolist() )
      for ii in range(len(self.data[-1])):
        del self.data[-1][ii]['log_likelihood']
        del self.data[-1][ii]['log_prior']
        #self.data[ii]['gw_log10_A'] = -15.0
    self.log_likelihoods_proposal = np.array(self.log_likelihoods_proposal, dtype=np.longdouble)
    self.data = np.array(self.data)
    self.data_shape = self.data.shape
    self.n_psrs, self.n_posteriors = self.data.shape
    self.flat_data = self.data.flatten()

    self.log_likelihoods_target = np.empty(self.data_shape, dtype=np.longdouble)

    self.log_evidence_factor = np.sum(log_evidences)

  def evaluate_target_likelihood(self):
    for psr_ii in range(self.n_psrs): #zip(self.obj_likelihoods_targ, self.data):
      for posterior_jj in range(self.n_posteriors): #self.data[psr_ii]:
        self.obj_likelihoods_targ[psr_ii].parameters = self.data[psr_ii,posterior_jj]
        self.log_likelihoods_target[psr_ii,posterior_jj] = self.obj_likelihoods_targ[psr_ii].log_likelihood()

  def log_likelihood_ratio(self):
    # First sum for likelihood ratios over self.n_posteriors, outter sum for log ratios over self.n_psrs
    return float(np.sum(np.log(np.sum(np.exp(self.log_likelihoods_target - self.log_likelihoods_proposal), axis=1) / self.n_posteriors)))

  def log_likelihood_ratio_wrapper(self):
    self.update_parameter_samples(self.parameters)
    self.evaluate_target_likelihood()
    return self.log_likelihood_ratio()

  def noise_log_likelihood(self):
    return self.log_evidence_factor

  def log_likelihood(self):
    return self.noise_log_likelihood() + self.log_likelihood_ratio_wrapper()

  def update_parameter_samples(self, sample, mpi=False):
    # This loop can be run through multiprocessing
    for item in self.flat_data:
      item.update(sample)
    self.data = self.flat_data.reshape(self.data_shape)

class ImportanceLikelihoodNoise(ImportanceLikelihoodSignal):
  def __init__(self, posteriors, obj_likelihoods_targ, prior,
               log_evidences, multiproc=False, npool=2, max_samples=1e100,
               stl_file="", grid_size=300):

    super(ImportanceLikelihoodNoise, self).__init__(posteriors, obj_likelihoods_targ, prior, log_evidences, max_samples=max_samples, stl_file=stl_file, grid_size=grid_size)
    self.qc_samples = np.linspace(-20., -10., grid_size) #100+1) # A_qc (gamma_qc) sample grid
    self.prior_for_qc_samples = np.empty(len(self.qc_samples))
    self.log_likelihoods_target_unmarginalized = np.empty(self.data_shape + (len(self.qc_samples),), dtype=np.longdouble)

    # Pre-computing target likelihood at a grid of qc samples
    if stl_file!="" and os.path.exists(stl_file):
      self.log_likelihoods_target_unmarginalized = np.load(stl_file)
    else:
      print('Pre-computing target likelihood, total samples: ', len(self.qc_samples))
      for qc_sample_kk, qc_sample in enumerate(self.qc_samples):
        print('Sample ', qc_sample_kk, '/', len(self.qc_samples))
        self.update_parameter_samples({'gw_log10_A': qc_sample})
        self.evaluate_target_likelihood()
        self.log_likelihoods_target_unmarginalized[:,:,qc_sample_kk] = self.log_likelihoods_target
      if stl_file is not None:
        np.save(stl_file, self.log_likelihoods_target_unmarginalized)
        print('Pre-computed likelihood, exiting')
        exit()

  def log_likelihood_ratio_wrapper(self):
    self.evaluate_prior_for_qc_samples()
    # marginalizing target likelihood over sampled prior
    self.log_likelihoods_target = simps(self.log_likelihoods_target_unmarginalized * self.prior_for_qc_samples, x=self.qc_samples, axis=2)
    logl_ratio = self.log_likelihood_ratio()
    if logl_ratio != np.inf:
      return self.log_likelihood_ratio()
    else:
      return 1e100

  def evaluate_prior_for_qc_samples(self):
    self.prior_for_qc_samples = self.prior({'gw_log10_A': self.qc_samples}, **self.parameters)

# ================================================== #

class ImportanceResult(results.BilbyWarpResult):

  def __init__(self, opts, suffix='red_noise'):
    super(ImportanceResult, self).__init__(opts)
    self.suffix = suffix
    self.results = []
    self.chains = []
    self.log_zs = []

  def main_pipeline(self):
    self.psr_nums = [int(pd.split('_')[0]) for pd in self.psr_dirs]
    self.excluded_nums = []
    for pn, psr_dir in sorted(zip(self.psr_nums, self.psr_dirs)):

      if psr_dir.split('_')[1] in self.opts.exclude:
        self.excluded_nums.append(pn)
        print('Excluding pulsar ', psr_dir)
        continue

      self.psr_dir = psr_dir
      success = self._scan_psr_output()
      if not success:
        continue

      success = self.load_chains()
      if not success:
        continue

      #self.standardize_chain_for_rn_hyper_pe()

      self.results.append(self.result)
      self.chains.append(self.result.posterior)
      self.log_zs.append(self.result.log_evidence)

  def standardize_chain_for_rn_hyper_pe(self):
    for key in self.result.posterior.keys():
      if self.suffix not in key and \
         'log_likelihood' not in key and \
         'log_prior' not in key and \
         'prior' not in key:
        del self.result.posterior[key]
      elif self.suffix+'_gamma' in key:
        self.result.posterior[self.suffix+'_gamma'] = \
                              self.result.posterior.pop(key)
      elif self.suffix+'_log10_A' in key:
        self.result.posterior[self.suffix+'_log10_A'] = \
                              self.result.posterior.pop(key)

# =================================================  #


