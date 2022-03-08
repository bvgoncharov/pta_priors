import tqdm
import time
import os.path

from scipy.integrate import simps
from scipy.special import logsumexp
import numpy as np

from enterprise_warp import results

from bilby.hyper.model import Model
from bilby.core.likelihood import Likelihood

class ImportanceLikelihoodSignal(Likelihood):
  """
  posteriors: list with pandas.DataFrame, bilby output with proposal likelihoods
  obj_likelihoods_targ: list of bilby_warp.PTABilbyLikelihood with target likelihoods
  log_evidences: list of float, log evidence for proposal likelihoods

  post_draw_rs: int or other, for Pandas, to draw the same posterior samples for
  parallel runs. Without it, different samples will scramble the results.
  """
  def __init__(self, posteriors, obj_likelihoods_targ, prior, 
               log_evidences, max_samples=1e100, post_draw_rs=777,
               stl_file="", grid_size=300, save_iterations=-1, suffix='gw'):

    #if not isinstance(prior, Model):
    #  prior = Model([prior])

    super(ImportanceLikelihoodSignal, self).__init__({})#prior.parameters)

    self.prior = prior
    self.suffix = suffix
    #self.posteriors = [posterior.sample(max_samples) for posterior in posteriors]
    self.posteriors = []
    for posterior in posteriors:
      print('N posterior samples: ', len(posterior), '; max: ', max_samples)
      if len(posterior)>=max_samples:
        self.posteriors.append(posterior.sample(max_samples, random_state=post_draw_rs))
      else:
        self.posteriors.append(posterior)
    self.obj_likelihoods_targ = obj_likelihoods_targ

    # Make sure the required log10_A parameter is in the target likelihood
    for olt in self.obj_likelihoods_targ:
      if not suffix+'_log10_A' in olt.parameters.keys():
        error_str = 'Parameter '+suffix+\
                    '_log10_A is not in the target likelihood parameters: '+\
                    ','.join(olt.parameters.keys())
        raise ValueError(error_str)

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
    #return float(np.sum(np.log(np.sum(np.exp(self.log_likelihoods_target - self.log_likelihoods_proposal), axis=1) / self.n_posteriors)))
    #scaling_factor - what it should be? https://lips.cs.princeton.edu/computing-log-sum-exp/
    return float(np.sum(logsumexp(self.log_likelihoods_target - self.log_likelihoods_proposal,axis=1) - np.log(self.n_posteriors)))

  def n_eff(self):
    self.update_parameter_samples(self.parameters)
    weight = np.exp(self.log_likelihoods_target - self.log_likelihoods_proposal)
    return np.sum(weight,axis=1)**2/np.sum(weight**2,axis=1)

  def log_likelihood_ratio_wrapper(self):
    #self.log_likelihood() # Why it was here? It creates an infinite recursion!
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
               log_evidences, max_samples=1e100, post_draw_rs=777,
               stl_file="", grid_size=300, save_iterations=-1, suffix='gw'):

    super(ImportanceLikelihoodNoise, self).__init__(posteriors, obj_likelihoods_targ, prior, log_evidences, max_samples=max_samples, post_draw_rs=post_draw_rs, stl_file=stl_file, grid_size=grid_size, save_iterations=save_iterations, suffix=suffix)
    self.qc_samples = np.linspace(-20., -10., grid_size) #100+1) # A_qc (gamma_qc) sample grid
    self.prior_for_qc_samples = np.empty(len(self.qc_samples))
    self.log_likelihoods_target_unmarginalized = np.empty(self.data_shape + (len(self.qc_samples),), dtype=np.longdouble)

    # Pre-computing target likelihood at a grid of qc samples
    stl_iter_files = [stl_file[:-4]+'_'+str(ii)+'.npy' for ii in range(len(self.qc_samples))]
    if stl_file!="" and os.path.exists(stl_file):
      self.log_likelihoods_target_unmarginalized = np.load(stl_file)
    elif stl_file!="" and np.all([os.path.exists(ff) for ff in stl_iter_files]):
      for qc_sample_kk, ff in enumerate(stl_iter_files):
        self.log_likelihoods_target_unmarginalized[:,:,qc_sample_kk] = np.load(ff)
      np.save(stl_file, self.log_likelihoods_target_unmarginalized)
    else:
      if np.any([os.path.exists(ff) for ff in stl_iter_files]):
        mf = np.array(stl_iter_files)[~np.array([os.path.exists(lgf) for lgf in stl_iter_files])]
        print('Partial samples of gw_log10_A_qc are found. Missing files: \n', mf)
      if save_iterations < 0:
        print('Pre-computing target likelihood, total samples: ', len(self.qc_samples))
        for qc_sample_kk, qc_sample in tqdm.tqdm(enumerate(self.qc_samples),total=len(self.qc_samples)):
          #print('Sample ', qc_sample_kk, '/', len(self.qc_samples))
          self.update_parameter_samples({self.suffix+'_log10_A': qc_sample})
          self.evaluate_target_likelihood()
          self.log_likelihoods_target_unmarginalized[:,:,qc_sample_kk] = self.log_likelihoods_target
        if stl_file is not None:
          np.save(stl_file, self.log_likelihoods_target_unmarginalized)
          print('Pre-computed likelihood, exiting')
          exit()
      elif stl_file is not None:
        t0 = time.time()
        self.update_parameter_samples({self.suffix+'_log10_A': self.qc_samples[save_iterations]})
        self.evaluate_target_likelihood()
        t1 = time.time()
        print('Elapsed time: ',t1-t0)
        np.save(stl_iter_files[save_iterations], self.log_likelihoods_target)
        exit()

  def log_likelihood_ratio_wrapper(self):
    self.evaluate_prior_for_qc_samples()
    # marginalizing target likelihood over sampled prior
    self.log_likelihoods_target = logsumexp(self.log_likelihoods_target_unmarginalized + np.log(self.prior_for_qc_samples)[np.newaxis,np.newaxis,:], axis=2) - np.log(self.qc_samples[1]-self.qc_samples[0])
    # Below is incorrect
    #self.log_likelihoods_target = simps(self.log_likelihoods_target_unmarginalized * self.prior_for_qc_samples, x=self.qc_samples, axis=2)
    logl_ratio = self.log_likelihood_ratio()
    if logl_ratio != np.inf:
      return self.log_likelihood_ratio()
    else:
      return 1e100

  def evaluate_prior_for_qc_samples(self):
    self.prior_for_qc_samples = self.prior({self.suffix+'_log10_A': self.qc_samples}, **self.parameters)

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

from mpmath import mp
#mp.prec = 170

class AnalyticalEvidence1D(object):
  """
  Evaluate Bayesian evidence for 1D likelihood by explicit numerical 
  integration, without nested sampling. Assuming a uniform prior.

  limits: ((x_low, x_high)), double brackets for compatibility with the child class
  """
  def __init__(self, log_likelihood=None, x_vals=None, limits=None, method='mpmath'):
    self.log_likelihood = log_likelihood
    self.x_vals = x_vals
    self.limits = limits
    if x_vals is not None and limits is not None:
      self.mask_vals((x_vals))
      self.x_vals = self.x_vals[self.mask[0]]
      self.log_likelihood = self.log_likelihood[self.mask[0]]
    self.logz = getattr(self,'logz_'+method)
    self.z = getattr(self,'z_'+method)

  def z_mpmath(self):
    log_l_mp = mp.matrix(self.log_likelihood)
    l_mp = mp.matrix([[mp.e**val for val in log_l_mp]])
    int_l = mp.fsum(mp.matrix([[(l_mp[ii]+l_mp[ii-1])/2 for ii in range(1,len(l_mp))]]))
    # Times integration step, times constant uniform prior
    return int_l*(self.x_vals[1]-self.x_vals[0])/\
                 (self.x_vals[-1]-self.x_vals[0])

  def logz_mpmath(self):
    return mp.log(self.z_mpmath())

  def mask_vals(self, vals):
    self.mask = []
    for vv, (low, high) in zip(vals, self.limits):
      self.mask.append( (vv >= low) * (vv <= high) )

class AnalyticalEvidence2D(AnalyticalEvidence1D):
  """
  log_likelihood_xy: 2D array with log-likelihood, x columns and y rows
  xy_vals: 2D array (meshgrid) with x columns and y rows
  limits: ((x_low, x_high),(y_low, y_high)), if need to truncated uniform prior
  """
  def __init__(self, log_likelihood_xy=None, xy_vals=None, limits=None, method='mpmath'):
    super(AnalyticalEvidence2D, self).__init__(None,None,None,method)
    self.limits = limits
    if xy_vals is not None and limits is not None:
      self.mask_vals((xy_vals[0][0,:], xy_vals[1][:,0]))
      xy_vals = (xy_vals[0][:,self.mask[0]], xy_vals[1][:,self.mask[0]])
      xy_vals = (xy_vals[0][self.mask[1],:], xy_vals[1][self.mask[1],:])
      log_likelihood_xy = log_likelihood_xy[:,self.mask[0]]
      log_likelihood_xy = log_likelihood_xy[self.mask[1]]
    self.xx, self.yy = xy_vals
    self.log_likelihood_xy = log_likelihood_xy
    self.z = getattr(self, 'z_2d_'+method)
    self.logz = getattr(self, 'logz_2d_'+method)
    self.logz_1d = getattr(self,'logz_'+method)
    self.z_1d = getattr(self,'z_'+method)

  def z_2d_mpmath(self):
    """
    To-do: introduce a possibility to integrate directly likelihood, without log.
    And pass likelihood to self, not log likelihood.
    """
    log_z_at_y = mp.matrix(1,len(self.yy[:,0]))
    for ii, yy_ii in tqdm.tqdm(enumerate(self.yy[:,0]), total=len(self.yy[:,0])):
      self.x_vals = self.xx[ii,:]
      self.log_likelihood = self.log_likelihood_xy[ii,:]
      log_z_at_y[ii] = self.logz_1d()
    self.x_vals = self.yy[:,0]
    self.log_likelihood = log_z_at_y
    return self.z_1d()

  def logz_2d_mpmath(self):
    return mp.log(self.z_2d_mpmath())
