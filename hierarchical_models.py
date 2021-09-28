import optparse
import numpy as np
from scipy.stats import multivariate_normal
from bilby.core.prior import Uniform, DeltaFunction

from enterprise_warp.enterprise_models import StandardModels
from enterprise_warp import results

# ---------------------------------------------------------------------------- #
# Models of prior distributions, parameters of which we would like to measure
# ---------------------------------------------------------------------------- #

# Joint hyper-priors, names to be used for "model: " in parameter files

class Norm_biv_trunc_lg_A_gamma(object):
  """ Joint normal prior with covariance between A and gamma """
  def __init__(self, suffix='red_noise'):
    self.suffix = suffix

  def __call__(self, dataset, mu_lg_A, sig_lg_A, mu_gam, sig_gam, rho_cov, \
               low_lg_A, high_lg_A, low_gam, high_gam):
    return norm_biv_trunc_lg_A_gamma(dataset, mu_lg_A, sig_lg_A, mu_gam, \
                                     sig_gam, rho_cov, low_lg_A, high_lg_A, \
                                     low_gam, high_gam, suffix=self.suffix)

class Norm_biv_lg_A_gamma(object):
  """ Joint normal prior with covariance between A and gamma """
  def __init__(self, suffix='red_noise'):
    self.suffix = suffix

  def __call__(self, dataset, mu_lg_A, sig_lg_A, mu_gam, sig_gam, rho_cov):
    return norm_biv_lg_A_gamma(dataset, mu_lg_A, sig_lg_A, mu_gam, sig_gam, \
           rho_cov, suffix=self.suffix)

class Norm_prod_lg_A_gamma(object):
  """ Assuming independent priors, no covariance between A and gamma """
  def __init__(self, suffix='red_noise'):
    self.suffix = suffix

  def __call__(self, dataset, mu_lg_A, sig_lg_A, mu_gam, sig_gam):
    return norm_lg_A(dataset, mu_lg_A, sig_lg_A, suffix = self.suffix) * \
           norm_gamma(dataset, mu_gam, sig_gam, suffix = self.suffix)

class Norm_lg_A(object):
  def __init__(self, suffix='red_noise'):
    self.suffix = suffix

  def __call__(self, dataset, mu_lg_A, sig_lg_A):
    return norm_lg_A(dataset, mu_lg_A, sig_lg_A, suffix = self.suffix)

class Unif_prod_lg_A_gamma(object):
  def __init__(self, suffix='red_noise'):
    self.suffix = suffix

  def __call__(self, dataset, low_lg_A, high_lg_A, low_gam, high_gam):
    return unif_lg_A(dataset, low_lg_A, high_lg_A, suffix=self.suffix) * \
           unif_gamma(dataset, low_gam, high_gam, suffix=self.suffix)

class Unif_lg_A(object):
  def __init__(self, suffix='red_noise'):
    self.suffix = suffix

  def __call__(self, dataset, low_lg_A, high_lg_A):
    return unif_lg_A(dataset, low_lg_A, high_lg_A, suffix=self.suffix)

class DeltaFunc_lg_A_gamma(object):
  def __init__(self, suffix='red_noise'):
    self.suffix = suffix

  def __call__(self, dataset, lg_A, gam):
    return deltafunc_lg_A(dataset, lg_A, suffix=self.suffix) * \
         deltafunc_gamma(dataset, gam, suffix=self.suffix)

class DeltaFunc_lg_A(object):
  def __init__(self, suffix='red_noise'):
    self.suffix = suffix

  def __call__(self, dataset, lg_A):
    return deltafunc_lg_A(dataset, lg_A, suffix=self.suffix)

# Models for a normal prior, fitting for mean and/or variance

def norm(dataset, key, mu, sigma):
  return np.exp(- (dataset[key] - mu)**2 / (2 * sigma**2)) /\
         (2 * np.pi * sigma**2)**0.5

def norm_lg_A(dataset, mu_lg_A, sig_lg_A, suffix='red_noise'):
  return norm(dataset, suffix+'_log10_A', mu_lg_A, sig_lg_A)

def norm_gamma(dataset, mu_gam, sig_gam, suffix='red_noise'):
  return norm(dataset, suffix+'_gamma', mu_gam, sig_gam)

def norm_biv_lg_A_gamma(dataset, mu_lg_A, sig_lg_A, mu_gam, sig_gam, rho_cov, \
                        suffix='red_noise'):
  return (2*np.pi*sig_lg_A*sig_gam*np.sqrt(1 - rho_cov**2))**(-1) * \
         np.exp(-1./2./(1-rho_cov**2)*( \
         inorm(dataset, suffix+'_log10_A', mu_lg_A, sig_lg_A)**2 - 2*rho_cov* \
         inorm(dataset, suffix+'_log10_A', mu_lg_A, sig_lg_A) * \
         inorm(dataset, suffix+'_gamma', mu_gam, sig_gam) + \
         inorm(dataset, suffix+'_gamma', mu_gam, sig_gam)**2 ) )

def norm_biv_trunc_lg_A_gamma(dataset, mu_lg_A, sig_lg_A, mu_gam, sig_gam, \
                              rho_cov, low_lg_A, high_lg_A, low_gam, high_gam, \
                              suffix='red_noise'):
  # Truncating the bivariate normal distribution that we fit for by creating a mask for posterior samples passed to the prior here. If the data point is outside of prior boundaries, we suggest it as "impossible", and set the probability for this point to be zero. 
  #mask_low_lg_A = dataset[suffix+'_log10_A'] > low_lg_A
  #mask_high_lg_A = dataset[suffix+'_log10_A'] < high_lg_A
  #mask_low_gam = dataset[suffix+'_gamma'] > low_gam
  #mask_high_gam = dataset[suffix+'_gamma'] < high_gam
  # This should be true by default, because we do not have any posterior samples outside of boundaries. So, we comment out this code and only renormalize probability based on boundaries.
  # P. S. Hyper-priors should also have the same boundaries.

  # Normalization
  cov_term = rho_cov*sig_lg_A*sig_gam

  # Handling some internal Bilby samples that returned as single-value arrays
  mu_lg_A = float(mu_lg_A)
  mu_gam = float(mu_gam)
  sig_lg_A = float(sig_lg_A)
  cov_term = float(cov_term)
  sig_gam = float(sig_gam)

  rv = multivariate_normal([mu_lg_A, mu_gam], \
                           [[sig_lg_A**2, cov_term],[cov_term, sig_gam**2]], \
                           allow_singular = True)
  vhh, vlh, vhl, vll = rv.cdf(np.array([[[high_lg_A, high_gam],[low_lg_A, high_gam],[high_lg_A, low_gam],[low_lg_A, low_gam]]]))
  area_truncated = vhh - vlh - vhl + vll

  #if np.sum(norm_biv_lg_A_gamma(dataset, mu_lg_A, sig_lg_A, mu_gam, sig_gam, rho_cov, suffix=suffix) * mask_low_lg_A * mask_high_lg_A * mask_low_gam * mask_high_gam / area_truncated == 0) > 0:
  #  import ipdb; ipdb.set_trace()
  # Total
  return norm_biv_lg_A_gamma(dataset, mu_lg_A, sig_lg_A, mu_gam, sig_gam, rho_cov, suffix=suffix) / area_truncated # * mask_low_lg_A * mask_high_lg_A * mask_low_gam * mask_high_gam

def inorm(dataset, key, mu, sigma):
  """ Helper function for norm_biv() """
  return (dataset[key] - mu) / sigma

#def norm_prod_lg_A_gamma(dataset, mu_lg_A, sig_lg_A, mu_gam, sig_gam, \
#                         suffix = 'red_noise'):
#  """ Assuming independent priors, no covariance between A and gamma """
#  return norm_lg_A(dataset, mu_lg_A, sig_lg_A, suffix = suffix) * \
#         norm_gamma(dataset, mu_gam, sig_gam, suffix = suffix)

# Models for a uniform prior, fitting for lower and upper bounds

def unif(dataset, key, low, high):
  return ((dataset[key] < high) * (dataset[key] > low)).astype(float)

def unif_lg_A(dataset, low_lg_A, high_lg_A, suffix='red_noise'):
  return unif(dataset, suffix+'_log10_A', low_lg_A, high_lg_A)

def unif_gamma(dataset, low_gam, high_gam, suffix='red_noise'):
  return unif(dataset, suffix+'_gamma', low_gam, high_gam)

#def unif_prod_lg_A_gamma(dataset, low_lg_A, high_lg_A, low_gam, high_gam, \
#                         suffix='red_noise'):
#  return unif_lg_A(dataset, low_lg_A, high_lg_A, suffix=suffix) * \
#         unif_gamma(dataset, low_gam, high_gam, suffix=suffix)

# Models for delta-function prior, fitting for position

def deltafunc(dataset, key, val, tol=2.0):
  """ Parameter tol is width of a delta function. It is not zero for
  computational reasons. """
  #return float(bool()) True if dataset[key]==val else False
  #return np.logical_not((dataset[key] - val).astype(bool)).astype(float)
  return (np.abs(dataset[key] - val) < tol).astype(float)

def deltafunc_lg_A(dataset, lg_A, suffix='red_noise'):
  return deltafunc(dataset, suffix+'_log10_A', lg_A, tol=2.0)

def deltafunc_gamma(dataset, gam, suffix='red_noise'):
  return deltafunc(dataset, suffix+'_gamma', gam, tol=2.0)

# ---------------------------------------------------------------------------- #
# (Hyper-)priors for prior parameters that we fit for
# ---------------------------------------------------------------------------- #

# For normal priors we use uniform hyper-priors

def hp_Norm_biv_trunc_lg_A_gamma(hip):
  """ hip is instance of HierarchicalInferenceParams """
  truncation_bounds = dict(low_lg_A = DeltaFunction(hip.low_lg_A, \
                           'low_lg_A', '$\log_{10}A_\mathrm{low}$'), \
                           high_lg_A = DeltaFunction(hip.high_lg_A, \
                           'high_lg_A', '$\log_{10}A_\mathrm{high}$'), \
                           low_gam = DeltaFunction(hip.low_gam, \
                           'low_gam', '$\gamma_\mathrm{low}$'), \
                           high_gam = DeltaFunction(hip.high_gam, \
                           'high_gam', '$\gamma_\mathrm{high}$'))
  return {**hp_Norm_biv_lg_A_gamma(hip), **truncation_bounds}

def hp_Norm_biv_lg_A_gamma(hip):
  """ hip is instance of HierarchicalInferenceParams """
  rho_cov = dict(rho_cov = Uniform(hip.rho_cov[0], hip.rho_cov[1], \
              'rho_cov', r'$\rho$'))
  return {**hp_Norm_lg_A(hip), **hp_Norm_gamma(hip), **rho_cov}

def hp_Norm_lg_A(hip):
  """ hip is instance of HierarchicalInferenceParams """
  return dict(mu_lg_A = Uniform(hip.mu_lg_A[0], hip.mu_lg_A[1], \
              'mu_lg_A', '$\mu_{lgA}$'),
              sig_lg_A = Uniform(hip.sig_lg_A[0], hip.sig_lg_A[1], \
              'sig_lg_A', '$\sigma_{lgA}$'))

def hp_Norm_gamma(hip):
  return dict(mu_gam = Uniform(hip.mu_gam[0], hip.mu_gam[1], \
              'mu_gam', '$\mu_{\gamma}$'),
              sig_gam = Uniform(hip.sig_gam[0], hip.sig_gam[1], \
              'sig_gam', '$\sigma_{\gamma}$'))

def hp_Norm_prod_lg_A_gamma(hip):
  return {**hp_Norm_lg_A(hip), **hp_Norm_gamma(hip)}

# For uniform priors we use uniform hyper-priors

def hp_Unif_lg_A(hip):
  return dict(low_lg_A = Uniform(hip.low_lg_A[0], hip.low_lg_A[1], \
              'low_lg_A', '$low_{lgA}$'),
              high_lg_A = Uniform(hip.high_lg_A[0], hip.high_lg_A[1], \
              'high_lg_A', '$high_{lgA}$'))

def hp_Unif_gamma(hip):
  return dict(low_gam = Uniform(hip.low_gam[0], hip.low_gam[1], \
              'low_gam', '$low_{\gamma}$'),
              high_gam = Uniform(hip.high_gam[0], hip.high_gam[1], \
              'high_gam', '$high_{\gamma}$'))

def hp_Unif_prod_lg_A_gamma(hip):
  return {**hp_Unif_lg_A(hip), **hp_Unif_gamma(hip)}

# For delta-function priors we use either uniform or delta-function hyper-priors

def hp_DeltaFunc_lg_A(hip):
  if type(hip.lg_A) is list:
    return dict(lg_A = Uniform(hip.lg_A[0], hip.lg_A[1], 'lg_A', '$lgA$'))
  elif type(hip.lg_A) is float:
    return dict(lg_A = DeltaFunction(hip.lg_A, 'lg_A', '$lgA$'))

def hp_DeltaFunc_gamma(hip):
  if type(hip.gam) is list:
    return dict(gam = Uniform(hip.gam[0], hip.gam[1], 'gam', '$\gamma$'))
  elif type(hip.gam) is float:
    return dict(gam = DeltaFunction(hip.gam, 'gam', '$\gamma$'))

def hp_DeltaFunc_lg_A_gamma(hip):
  return {**hp_DeltaFunc_lg_A(hip), **hp_DeltaFunc_gamma(hip)}

# ---------------------------------------------------------------------------- #
# Parameter object with default values to read from a parameter file
# ---------------------------------------------------------------------------- #

class HierarchicalInferenceParams(StandardModels):
  """
  Configuration parameters for hierarchical inference: priors, models, etc.
  """
  def __init__(self,psr=None,params=None):
    super(HierarchicalInferenceParams, self).__init__(psr=psr,params=params)
    self.priors.update({
      "max_samples_from_measurement": 500,
      "model": "norm_prod_lg_A_gamma",
      "par_suffix": "red_noise",
      "mu_lg_A": [-20., -10.],
      "sig_lg_A": [0., 10.],
      "mu_gam": [0., 10.],
      "sig_gam": [0., 10.],
      "rho_cov": [0., 1.],
      "low_lg_A": [-20., -10.],
      "high_lg_A": [-20., -10.],
      "low_gam": [0., 10.],
      "high_gam": [0., 10.],
      "lg_A": [-20., -10.],
      "gam": [0., 10.],
    })

# ---------------------------------------------------------------------------- #
# An enterprise_war.results-derived object that loads results from separate
# pulsars to run a population analysis. Also, a relevant command line parser.
# ---------------------------------------------------------------------------- #

class HyperResult(results.BilbyWarpResult):

  def __init__(self, opts, suffix='red_noise'):
    super(HyperResult, self).__init__(opts)
    self.suffix = suffix
    self.results = []
    self.chains = []
    self.log_zs = []

  def main_pipeline(self):
    for psr_dir in sorted(self.psr_dirs):

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
