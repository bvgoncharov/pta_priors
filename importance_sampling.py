
from multiprocessing import Pool

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
  def __init__(self, posteriors, obj_likelihoods_targ, 
               hyper_prior, log_evidences, multiproc=False, npool=2, max_samples=1e100):

    if not isinstance(hyper_prior, Model):
      hyper_prior = Model([hyper_prior])

    super(ImportanceLikelihoodSignal, self).__init__(hyper_prior.parameters)

    self.multiproc = multiproc
    if multiproc:
      self.pool = Pool(npool)

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
    self.log_likelihoods_proposal = np.array(self.log_likelihoods_proposal)#, dtype=np.float128)
    self.data = np.array(self.data)
    self.data_shape = self.data.shape
    self.n_psrs, self.n_posteriors = self.data.shape
    self.flat_data = self.data.flatten()

    self.log_likelihoods_target = np.empty(self.data_shape)#, dtype=np.float128)

    self.evidence_factor = np.sum(log_evidences)

  def log_likelihood_ratio(self):
    self.update_parameter_samples()
    # This loop can be run through multiprocessing
    for psr_ii in range(self.n_psrs): #zip(self.obj_likelihoods_targ, self.data):
      for posterior_jj in range(self.n_posteriors): #self.data[psr_ii]:
        self.obj_likelihoods_targ[psr_ii].parameters = self.data[psr_ii,posterior_jj]
        self.log_likelihoods_target[psr_ii,posterior_jj] = self.obj_likelihoods_targ[psr_ii].log_likelihood()
    # First sum for likelihood ratios over self.n_posteriors, outter sum for log ratios over self.n_psrs
    return np.sum(np.log(np.sum(np.exp(self.log_likelihoods_target - self.log_likelihoods_proposal), axis=1) / self.n_posteriors))

  def noise_log_likelihood(self):
    return self.log_evidence_factor

  def log_likelihood(self):
    return self.noise_log_likelihood() + self.log_likelihood_ratio()

  def update_parameter_samples(self, mpi=False):
    # This loop can be run through multiprocessing
    import ipdb; ipdb.set_trace()
    self.flat_data = [item.update(self.parameters) for item in self.flat_data]
    self.data = self.flat_data.reshape(self.data_shape)

  def resample_posteriors(self, max_samples=None):
    """
    Convert list of pandas DataFrame object to dict of arrays.

    Parameters
    ==========
    max_samples: int, opt
        Maximum number of samples to take from each posterior,
        default is length of shortest posterior chain.
    Returns
    =======
    data: dict
        Dictionary containing arrays of size (n_posteriors, max_samples)
        There is a key for each shared key in self.posteriors.
    """
    if max_samples is not None:
        self.max_samples = max_samples
    for posterior in self.posteriors:
        self.max_samples = min(len(posterior), self.max_samples)
    data = {key: [] for key in self.posteriors[0]}
    if 'log_prior' in data.keys():
        data.pop('log_prior')
    if 'prior' not in data.keys():
        data['prior'] = []
    for posterior in self.posteriors:
        temp = posterior.sample(self.max_samples)
        if 'log_prior' in temp.keys():
            temp['prior'] = np.exp(temp['log_prior'])
        import ipdb; ipdb.set_trace()
        for key in data:
            data[key].append(temp[key])
    for key in data:
        data[key] = np.array(data[key])
    return data


class ImportanceLikelihoodNoise(ImportanceLikelihoodSignal):
  def __init__(self, posteriors, log_likelihoods_prop, log_likelihoods_targ,
               log_evidences=None, max_samples=1e100):

    super(ImportanceLikelihoodNoise, self).__init__(hyper_prior.parameters)
    pass

  def log_likelihood_ratio(self):
    return

# ================================================== #

class ImportanceResult(results.BilbyWarpResult):

  def __init__(self, opts, suffix='red_noise'):
    super(ImportanceResult, self).__init__(opts)
    self.suffix = suffix
    self.results = []
    self.chains = []
    self.log_zs = []

  def main_pipeline(self):
    for psr_dir in sorted(self.psr_dirs):
      if psr_dir.split('_')[1] in self.opts.exclude:
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


