import sys
sys.path.insert(0, "/home/celestialsapien/enterprise_warp-dev")
import optparse
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm as scipy_norm
from bilby.core.prior import Uniform, Normal, DeltaFunction

from enterprise_warp.enterprise_models import StandardModels
import enterprise.constants as const
from enterprise_warp import results
import enterprise.constants as const


import psd_models as pm

# ---------------------------------------------------------------------------- #
# Models of prior distributions, parameters of which we would like to measure
# ---------------------------------------------------------------------------- #

# Joint hyper-priors, names to be used for "model: " in parameter files

class Mix_double_norm_biv_trunc(object):
  """
  Mixture of (1) a joint normal truncated prior with covariance between A and
  gamma and (2) same prior centred around the alleged common-spectrum process.
  """
  def __init__(self, suffix='red_noise'):
    self.suffix = suffix

  def __call__(self, dataset, mu_lg_A, sig_lg_A, mu_gam, sig_gam, rho_cov, \
               mu_lg_A_2, sig_lg_A_2, mu_gam_2, sig_gam_2, rho_cov_2,
               low_lg_A, high_lg_A, low_gam, high_gam, fnorm):
    return fnorm * norm_biv_trunc_lg_A_gamma(dataset, mu_lg_A, sig_lg_A, \
           mu_gam, sig_gam, rho_cov, low_lg_A, high_lg_A, low_gam, high_gam, \
           suffix=self.suffix) + (1 - fnorm) * \
           norm_biv_trunc_lg_A_gamma(dataset, mu_lg_A_2, sig_lg_A_2, \
           mu_gam_2, sig_gam_2, rho_cov_2, low_lg_A, high_lg_A, low_gam, \
           high_gam, suffix=self.suffix)

class Mix_norm_biv_trunc_and_deltaf(object):
  """
  Mixture of (1) a joint normal truncated prior with covariance between A and
  gamma and (2) a Dirac's delta function.
  """
  def __init__(self, suffix='red_noise'):
    self.suffix = suffix

  def __call__(self, dataset, mu_lg_A, sig_lg_A, mu_gam, sig_gam, rho_cov, \
               low_lg_A, high_lg_A, low_gam, high_gam, lg_A, gam, fnorm):
    return fnorm * norm_biv_trunc_lg_A_gamma(dataset, mu_lg_A, sig_lg_A, \
           mu_gam, sig_gam, rho_cov, low_lg_A, high_lg_A, low_gam, high_gam, \
           suffix=self.suffix) + \
           (1 - fnorm) * deltafunc_lg_A(dataset, lg_A, suffix=self.suffix) * \
           deltafunc_gamma(dataset, gam, suffix=self.suffix)

class Mix_norm_biv_trunc_and_unif(object):
  """
  Mixture of (1) a joint normal truncated prior with covariance between A and 
  gamma and (2) a uniform prior.
  """
  def __init__(self, suffix='red_noise'):
    self.suffix = suffix

  def __call__(self, dataset, mu_lg_A, sig_lg_A, mu_gam, sig_gam, rho_cov, \
               low_lg_A, high_lg_A, low_gam, high_gam, fnorm):
    return fnorm * norm_biv_trunc_lg_A_gamma(dataset, mu_lg_A, sig_lg_A, \
           mu_gam, sig_gam, rho_cov, low_lg_A, high_lg_A, low_gam, high_gam, \
           suffix=self.suffix) + \
           (1 - fnorm) * unif_lg_A(dataset, low_lg_A, high_lg_A, \
           suffix=self.suffix) * unif_gamma(dataset, low_gam, high_gam, \
           suffix=self.suffix)

class Norm_biv_trunc_lg_A_gamma(object):
  """ Joint normal truncated prior with covariance between A and gamma """
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

class Norm_trunc_lg_A(object):
  def __init__(self, suffix='red_noise'):
    self.suffix = suffix

  def __call__(self, dataset, mu_lg_A, sig_lg_A, low_lg_A, high_lg_A):
    return norm_trunc_lg_A(dataset, mu_lg_A, sig_lg_A, low_lg_A, \
                           high_lg_A, suffix = self.suffix)

class Norm_trunc_gamma(object):
  def __init__(self, suffix='red_noise'):
    self.suffix = suffix

  def __call__(self, dataset, mu_gam, sig_gam, low_gam, high_gam):
    return norm_trunc_gamma(dataset, mu_gam, sig_gam, low_gam, \
                           high_gam, suffix = self.suffix)

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

def norm_area(mu, sigma, low, high):
  return np.abs(scipy_norm.cdf(high, mu, sigma) - \
                scipy_norm.cdf(low, mu, sigma))

def norm_lg_A(dataset, mu_lg_A, sig_lg_A, suffix='red_noise'):
  return norm(dataset, suffix+'_log10_A', mu_lg_A, sig_lg_A)

def norm_trunc_lg_A(dataset, mu_lg_A, sig_lg_A, low_lg_A, \
                    high_lg_A, suffix='red_noise'):

  # Setting zero probability to "impossible" values
  mask_low_lg_A = dataset[suffix+'_log10_A'] > low_lg_A
  mask_high_lg_A = dataset[suffix+'_log10_A'] < high_lg_A

  return norm(dataset, suffix+'_log10_A', mu_lg_A, sig_lg_A)/\
         norm_area(mu_lg_A, sig_lg_A, low_lg_A, high_lg_A)*\
         mask_low_lg_A * mask_high_lg_A

def norm_trunc_gamma(dataset, mu_gam, sig_gam, low_gam, \
                    high_gam, suffix='red_noise'):

  # Setting zero probability to "impossible" values
  mask_low_gam = dataset[suffix+'_gamma'] > low_gam
  mask_high_gam = dataset[suffix+'_gamma'] < high_gam

  return norm(dataset, suffix+'_gamma', mu_gam, sig_gam)/\
         norm_area(mu_gam, sig_gam, low_gam, high_gam)*\
         mask_low_gam * mask_high_gam

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
  """
  # Truncating the bivariate normal distribution that we fit for by creating a mask for posterior samples passed to the prior here. If the data point is outside of prior boundaries, we suggest it as "impossible", and set the probability for this point to be zero. 

  When boundaries are fixed and same as for recycled posterior samples: these should be true by default, because we do not have any posterior samples outside of boundaries.

  """
  # Preparation
  cov_term = rho_cov*sig_lg_A*sig_gam

  # Handling some internal Bilby samples that returned as single-value arrays
  mu_lg_A = float(mu_lg_A)
  mu_gam = float(mu_gam)
  sig_lg_A = float(sig_lg_A)
  cov_term = float(cov_term)
  sig_gam = float(sig_gam)
  low_lg_A = float(low_lg_A)
  low_gam = float(low_gam)
  high_lg_A = float(high_lg_A)
  high_gam = float(high_gam)

  # Truncation
  mask_low_lg_A = dataset[suffix+'_log10_A'] > low_lg_A
  mask_high_lg_A = dataset[suffix+'_log10_A'] < high_lg_A
  mask_low_gam = dataset[suffix+'_gamma'] > low_gam
  mask_high_gam = dataset[suffix+'_gamma'] < high_gam

  # Normalization factor: volume of a 2-parameter Gaussian within boundaries
  rv = multivariate_normal([mu_lg_A, mu_gam], \
                           [[sig_lg_A**2, cov_term],[cov_term, sig_gam**2]], \
                           allow_singular = True)

  vhh, vlh, vhl, vll = rv.cdf(np.array([[[high_lg_A, high_gam],\
                                         [low_lg_A, high_gam],\
                                         [high_lg_A, low_gam],\
                                         [low_lg_A, low_gam]]]))
  vol_truncated = vhh - vlh - vhl + vll

  # Total
  return norm_biv_lg_A_gamma(dataset, mu_lg_A, sig_lg_A, mu_gam, sig_gam, \
         rho_cov, suffix=suffix) / vol_truncated * mask_low_lg_A * \
         mask_high_lg_A * mask_low_gam * mask_high_gam

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
  return ((dataset[key] < high) * (dataset[key] > low)).astype(float) / \
         (high - low) # Added on 2021-10-02 to normalize priors

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
  return (np.abs(dataset[key] - val) < tol).astype(float)/(2*tol)

def deltafunc_lg_A(dataset, lg_A, suffix='red_noise'):
  return deltafunc(dataset, suffix+'_log10_A', lg_A, tol=2.0)

def deltafunc_gamma(dataset, gam, suffix='red_noise'):
  return deltafunc(dataset, suffix+'_gamma', gam, tol=2.0)

# ---------------------------------------------------------------------------- #
# Models of prior distributions for importance sampling
# ---------------------------------------------------------------------------- #

# These are not used in signal likelihood
class Unif_lg_A_cp(object):
  def __init__(self, suffix='red_noise'):
    self.suffix = suffix

  def __call__(self, dataset, lg_A):
    raise ValueError('We do not do that here')

# ---------------------------------------------------------------------------- #
# (Hyper-)priors for prior parameters that we fit for
# ---------------------------------------------------------------------------- #

def hyper_prior_type(val, name, mathname, hp_type="uniform"):
  """
  Return Bilby prior object DeltaFunction(val) if val is float or int,
  otherwise return Uniform(val) if val is list of two values - prior boundaries.
  """
  if type(val) is float or type(val) is int:
    return DeltaFunction(val, name, mathname)
  elif type(val) is list and len(val)==2:
    if hp_type == "uniform":
      return Uniform(val[0], val[1], name, mathname)
    elif hp_type == "normal":
      return Normal(val[0], val[1], name, mathname)
  else:
    raise ValueError('Unknown prior parameters in hierarchical_models.py: ', \
                     name, val)

# For normal priors we use uniform hyper-priors

def hp_Mix_double_norm_biv_trunc(hip):
  """
  Here we mix a wide Gaussian and a narrow Gaussian, for which a mean we center
  at the alleged common-spectrum process, hence a Normal hyper-prior, with a 
  variance corresponding to the CP measurement uncertainty. A variance of a 
  second Gaussian is then the intrinsic width of a CP.

  hip is instance of HierarchicalInferenceParams
  """
  fnorm = dict(fnorm = Uniform(0, 1, 'nfrac', r'$\nu_\mathcal{N}$'))
  lg_A_2 = dict(mu_lg_A_2 = Normal(hip.mu_lg_A_2[0], hip.mu_lg_A_2[1], \
                'mu_lg_A_2', '$\mu_{lgA,2}$'),
                sig_lg_A_2 = Uniform(hip.sig_lg_A_2[0], hip.sig_lg_A_2[1], \
                'sig_lg_A_2', '$\sigma_{lgA,2}$'))
  gam_2 = dict(mu_gam_2 = Normal(hip.mu_gam_2[0], hip.mu_gam_2[1], \
               'mu_gam_2', '$\mu_{\gamma,2}$'),
                sig_gam_2 = Uniform(hip.sig_gam_2[0], hip.sig_gam_2[1], \
               'sig_gam_2', '$\sigma_{\gamma,2}$'))
  rho_cov_2 = dict(rho_cov_2 = hyper_prior_type(hip.rho_cov_2, \
                                                    'rho_cov_2', r'$\rho_2$'))
  return {**hp_Norm_biv_trunc_lg_A_gamma(hip), **lg_A_2, **gam_2, **rho_cov_2, **fnorm}

def hp_Mix_norm_biv_trunc_and_deltaf(hip):
  """ hip is instance of HierarchicalInferenceParams """
  fnorm = dict(fnorm = Uniform(0, 1, 'nfrac', r'$\nu_\mathcal{N}$'))
  return {**hp_Norm_biv_trunc_lg_A_gamma(hip), **hp_DeltaFunc_lg_A_gamma(hip), \
          **fnorm}

def hp_Mix_norm_biv_trunc_and_unif(hip):
  """ hip is instance of HierarchicalInferenceParams """
  fnorm = dict(fnorm = Uniform(0, 1, 'nfrac', r'$\nu_\mathcal{N}$'))
  return {**hp_Norm_biv_trunc_lg_A_gamma(hip), **fnorm}

def hp_Norm_biv_trunc_lg_A_gamma(hip):
  """ hip is instance of HierarchicalInferenceParams """
  return {**hp_Norm_biv_lg_A_gamma(hip), **hp_Unif_lg_A(hip), \
          **hp_Unif_gamma(hip)}

def hp_Norm_biv_lg_A_gamma(hip):
  """ hip is instance of HierarchicalInferenceParams """
  rho_cov = dict(rho_cov = hyper_prior_type(hip.rho_cov, \
              'rho_cov', r'$\rho$'))
  return {**hp_Norm_lg_A(hip), **hp_Norm_gamma(hip), **rho_cov}

def hp_Norm_lg_A(hip):
  """ hip is instance of HierarchicalInferenceParams """
  return dict(mu_lg_A = hyper_prior_type(hip.mu_lg_A, \
              'mu_lg_A', '$\mu_{lgA}$', hp_type=hip.mu_lg_A_type),
              sig_lg_A = Uniform(hip.sig_lg_A[0], hip.sig_lg_A[1], \
              'sig_lg_A', '$\sigma_{lgA}$'))

def hp_Norm_trunc_lg_A(hip):
  return {**hp_Norm_lg_A(hip), **hp_Unif_lg_A(hip)}

def hp_Norm_trunc_gamma(hip):
  return {**hp_Norm_gamma(hip), **hp_Unif_gamma(hip)}

def hp_Norm_gamma(hip):
  return dict(mu_gam = hyper_prior_type(hip.mu_gam, \
              'mu_gam', '$\mu_{\gamma}$', hp_type=hip.mu_gam_type),
              sig_gam = Uniform(hip.sig_gam[0], hip.sig_gam[1], \
              'sig_gam', '$\sigma_{\gamma}$'))

def hp_Norm_prod_lg_A_gamma(hip):
  return {**hp_Norm_lg_A(hip), **hp_Norm_gamma(hip)}

# For uniform priors we use uniform hyper-priors

def hp_Unif_lg_A(hip):
  return dict(low_lg_A = hyper_prior_type(hip.low_lg_A, \
              'low_lg_A', '$\log_{10}A_\mathrm{low}$'), \
              high_lg_A = hyper_prior_type(hip.high_lg_A, \
              'high_lg_A', '$\log_{10}A_\mathrm{high}$'))

def hp_Unif_gamma(hip):
  return dict(low_gam = hyper_prior_type(hip.low_gam, \
              'low_gam', '$\gamma_\mathrm{low}$'), \
              high_gam = hyper_prior_type(hip.high_gam, \
              'high_gam', '$\gamma_\mathrm{high}$'))

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
# (Hyper-)priors for target likelihoods in importance sampling
# ---------------------------------------------------------------------------- #

# Signal likelihood (CP), parameter names should match those in enterprise
def hp_Unif_lg_A_cp(hip):
  return dict(gw_log10_A = Uniform(hip.gwb_lgA[0], hip.gwb_lgA[1], \
              'gw_log10_A', '$\log_{10}A_\mathrm{CP}$'))

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
      "parname": "log10_A",
      "exclude": [""],
      "qc_range": [-20., -10.],
      "mu_lg_A": [-20., -10.],
      "mu_lg_A_type": "uniform",
      "sig_lg_A": [0., 10.],
      "mu_gam": [0., 10.],
      "mu_gam_type": "uniform",
      "sig_gam": [0., 10.],
      "rho_cov": [0., 1.],
      "mu_lg_A_2": [-14.55, 0.20],
      "sig_lg_A_2": [0., 10.],
      "mu_gam_2": [4.11, 0.52],
      "sig_gam_2": [0., 10.],
      "rho_cov_2": [-1., 1.],
      "low_lg_A": [-20., -10.],
      "high_lg_A": [-20., -10.],
      "low_gam": [0., 10.],
      "high_gam": [0., 10.],
      "lg_A": [-20., -10.],
      "gam": [0., 10.],
      # Importance sampling only:
      "importance_likelihood": "ImportanceLikelihoodSignal",
      "grid_size": 300,
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

      self.standardize_chain_for_rn_hyper_pe()
      self.convert_log10_A_to_pow()

      self.results.append(self.result)
      self.chains.append(self.result.posterior)
      self.log_zs.append(self.result.log_evidence)

    if self.opts.plots:
      self.make_plots_and_exit()


  def convert_log10_A_to_pow(self, cadence=60):
    """
    cadence: a default cadence to determine high frequency of red PSD
    	if not given in the noise model file for a given pulsar term.
    """
    log10_As = self.result.posterior[self.suffix+'_log10_A']
    
    gammas = self.result.posterior[self.suffix+'_gamma']

    psr = self.psr_dir.split('_')[1]

    noise_name_dict = {
      "red_noise": "spin_noise",
      "dm_gp": "dm_noise"
    }
    flow = 1./pm.PulsarEqualPSDLines().tobs[psr]*const.fyr

    psr_noise_model = self.params.models[0].noisemodel[psr]
    psr_noise_term = psr_noise_model[noise_name_dict[self.suffix]]
    if len(psr_noise_term.split('_'))==1:
      tobs = 1./flow
      nf = int(np.round((1./cadence/const.day - 1/tobs)/(1/tobs)))
    elif len(psr_noise_term.split('_'))==3:
      nf = float(psr_noise_term.split('_')[1])
    else:
      raise ValueError('A new special case of the frequency bin handling is required')
    
    fhigh = flow * nf

    self.result.posterior[self.suffix+'_pow'] = \
                          pm.powerlaw_power(log10_As, gammas, flow, fhigh)

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

  def make_plots_and_exit(self):
    self.total = {self.suffix+'_log10_A': [], self.suffix+'_gamma': [], \
                  self.suffix+'_pow': [],'logz': []}
    for ii, cc in enumerate(self.chains):
      for key in ['_log10_A', '_gamma', '_pow']:
        self.total[self.suffix+key] += cc[self.suffix+key].tolist()
      for jj in range(len(cc[self.suffix+key])):
        self.total['logz'].append(self.log_zs[ii])

    gamma_arr = np.linspace(0, 10, 101)
    pepl = pm.PulsarEqualPSDLines()
    psrs = ['J0437-4715', 'J1832-0836', 'J1824-2452A', 'J1909-3744', \
            'J1939+2134']
    lgA = {psr: pepl.get_equipower_log10_A(psr, gamma_arr) for psr in psrs}
    lgA_psd = {psr: pepl.get_equipsd_log10_A(psr, gamma_arr) for psr in psrs}

    plt.scatter(self.total[self.suffix+'_pow'], \
                self.total[self.suffix+'_gamma'], c = self.total['logz'], \
                s = 1, alpha = 0.1, label='Posterior samples')
    plt.xscale('log')
    plt.xlabel('Noise power')
    plt.ylabel('gamma')
    plt.colorbar(label='log_z')
    plt.legend()
    plt.tight_layout()
    plt.savefig(self.outdir_all + 'population_samples_scatter_pow.png')
    plt.close()

    plt.scatter(self.total[self.suffix+'_log10_A'], \
                self.total[self.suffix+'_gamma'], c = self.total['logz'], \
                s = 1, alpha = 0.1, label='Posterior samples')
    for psr in psrs:
      plt.plot(lgA[psr], gamma_arr, label=psr+': equal power')
    plt.xlabel('log10_A')
    plt.ylabel('gamma')
    plt.colorbar(label='log_z')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(self.outdir_all + 'population_samples_scatter.png')
    plt.close()

    plt.scatter(self.total[self.suffix+'_log10_A'], \
                self.total[self.suffix+'_gamma'], c = self.total['logz'], \
                s = 1, alpha = 0.1, label='Posterior samples')
    for psr in psrs:
      plt.plot(lgA_psd[psr], gamma_arr, label=psr+': equal PSD at 1/Tobs')
    plt.xlabel('log10_A')
    plt.ylabel('gamma')
    plt.colorbar(label='log_z')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(self.outdir_all + 'population_samples_scatter_psd.png')
    plt.close()

    plt.hist(self.total[self.suffix+'_pow'], bins=100)
    plt.xlabel('Power between 1/Tobs_psr and nf_psr/Tobs_psr')
    plt.ylabel('Probability')
    plt.savefig(self.outdir_all + 'population_samples_power.png')
    plt.close()

    plt.hist(self.total[self.suffix+'_log10_A'], bins=100)
    plt.xlabel('log10_A')
    plt.ylabel('Probability')
    plt.savefig(self.outdir_all + 'population_samples_log10_A.png')
    plt.close()

    plt.hist(self.total[self.suffix+'_gamma'], bins=100)
    plt.xlabel('gamma')
    plt.ylabel('Probability')
    plt.savefig(self.outdir_all + 'population_samples_gamma.png')
    plt.close()

    print('Plots are saved at ', self.outdir_all)
    exit()

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

  parser.add_option("-t", "--target", help="Parameter file for the target \
                    distribution (importance sampling)", default=None, type=str)

  parser.add_option("-P", "--plots", help="Make plots and quit", type=int)

  parser.add_option("-e", "--exclude", help="Exclude PSR", action="append", \
                    default=[], type=str)

  # For analytical importance sampling
  parser.add_option("-I", "--save_iterations", default=-1, help="Save \
                    likelihood for each grid point. -1: off, >=0: \
                    grid point", type=int)
  parser.add_option("-N", "--n_grid_iter", default=1, help="Number of \
                    likelihood samples to save in one file, for analytical \
                    sampling of the likelihood.", type=int)
  parser.add_option("-C", "--incomplete", default=0, help="Proceed with \
                    incomplete grid likelihood evaluation.", type=int)

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

  parser.add_option("-x", "--extra_model_terms", \
                    help="Extra noise terms to add to the .json noise model \
                          file, a string that will be converted to dict. \
                          E.g. {'J0437-4715': {'system_noise': \
                          'CPSR2_20CM'}}. Extra terms are applied either on \
                          the only model, or the second model.", \
                    default='None', type=str)

  opts, args = parser.parse_args()

  return opts
