import numpy as np
from bilby.core.prior import Uniform, DeltaFunction

from enterprise_warp.enterprise_models import StandardModels

# ---------------------------------------------------------------------------- #
# Models of prior distributions, parameters of which we would like to measure
# ---------------------------------------------------------------------------- #

# Models for a normal prior, fitting for mean and/or variance

def norm(dataset, key, mu, sigma):
  return np.exp(- (dataset[key] - mu)**2 / (2 * sigma**2)) /\
         (2 * np.pi * sigma**2)**0.5

def norm_lg_A(dataset, mu_lg_A, sig_lg_A):
  return norm(dataset, 'red_noise_log10_A', mu_lg_A, sig_lg_A)

def norm_gamma(dataset, mu_gam, sig_gam):
  return norm(dataset, 'red_noise_gamma', mu_gam, sig_gam)

def norm_prod_lg_A_gamma(dataset, mu_lg_A, sig_lg_A, mu_gam, sig_gam):
  """ Assuming independent priors, no covariance between A and gamma """
  return norm_lg_A(dataset, mu_lg_A, sig_lg_A) * \
         norm_gamma(dataset, mu_gam, sig_gam)

# Models for a uniform prior, fitting for lower and upper bounds

def unif(dataset, key, low, high):
  return ((dataset[key] < high) * (dataset[key] > low)).astype(float)

def unif_lg_A(dataset, low_lg_A, high_lg_A):
  return unif(dataset, 'red_noise_log10_A', low_lg_A, high_lg_A)

def unif_gamma(dataset, low_gam, high_gam):
  return unif(dataset, 'red_noise_gamma', low_gam, high_gam)

def unif_prod_lg_A_gamma(dataset, low_lg_A, high_lg_A, low_gam, high_gam):
  return unif_lg_A(dataset, low_lg_A, high_lg_A) * \
         unif_gamma(dataset, low_gam, high_gam)

# Models for delta-function prior, fitting for position

def deltafunc(dataset, key, val, tol=2.0):
  """ Parameter tol is width of a delta function. It is not zero for
  computational reasons. """
  #return float(bool()) True if dataset[key]==val else False
  #return np.logical_not((dataset[key] - val).astype(bool)).astype(float)
  return (np.abs(dataset[key] - val) < tol).astype(float)

def deltafunc_lg_A(dataset, lg_A):
  return deltafunc(dataset, 'red_noise_log10_A', lg_A, tol=2.0)

def deltafunc_gamma(dataset, gam):
  return deltafunc(dataset, 'red_noise_gamma', gam, tol=2.0)

def deltafunc_lg_A_gamma(dataset, lg_A, gam):
  return deltafunc_lg_A(dataset, lg_A) * deltafunc_gamma(dataset, gam)

# ---------------------------------------------------------------------------- #
# (Hyper-)priors for prior parameters that we fit for
# ---------------------------------------------------------------------------- #

# For normal priors we use uniform hyper-priors

def hp_norm_lg_A(hip):
  """ hip is instance of HierarchicalInferenceParams """
  return dict(mu_lg_A = Uniform(hip.mu_lg_A[0], hip.mu_lg_A[1], \
              'mu_lg_A', '$\mu_{lgA}$'),
              sig_lg_A = Uniform(hip.sig_lg_A[0], hip.sig_lg_A[1], \
              'sig_lg_A', '$\sigma_{lgA}$'))

def hp_norm_gamma(hip):
  return dict(mu_gam = Uniform(hip.mu_gam[0], hip.mu_gam[1], \
              'mu_gam', '$\mu_{\gamma}$'),
              sig_gam = Uniform(hip.sig_gam[0], hip.sig_gam[1], \
              'sig_gam', '$\sigma_{\gamma}$'))

def hp_norm_prod_lg_A_gamma(hip):
  return {**hp_norm_lg_A(hip), **hp_norm_gamma(hip)}

# For uniform priors we use uniform hyper-priors

def hp_unif_lg_A(hip):
  return dict(low_lg_A = Uniform(hip.low_lg_A[0], hip.low_lg_A[1], \
              'low_lg_A', '$low_{lgA}$'),
              high_lg_A = Uniform(hip.high_lg_A[0], hip.high_lg_A[1], \
              'high_lg_A', '$high_{lgA}$'))

def hp_unif_gamma(hip):
  return dict(low_gam = Uniform(hip.low_gam[0], hip.low_gam[1], \
              'low_gam', '$low_{\gamma}$'),
              high_gam = Uniform(hip.high_gam[0], hip.high_gam[1], \
              'high_gam', '$high_{\gamma}$'))

def hp_unif_prod_lg_A_gamma(hip):
  return {**hp_unif_lg_A(hip), **hp_unif_gamma(hip)}

# For delta-function priors we use either uniform or delta-function hyper-priors

def hp_deltafunc_lg_A(hip):
  if type(hip.lg_A) is list:
    return dict(lg_A = Uniform(hip.lg_A[0], hip.lg_A[1], 'lg_A', '$lgA$'))
  elif type(hip.lg_A) is float:
    return dict(lg_A = DeltaFunction(hip.lg_A, 'lg_A', '$lgA$'))

def hp_deltafunc_gamma(hip):
  if type(hip.gam) is list:
    return dict(gam = Uniform(hip.gam[0], hip.gam[1], 'gam', '$\gamma$'))
  elif type(hip.gam) is float:
    return dict(gam = DeltaFunction(hip.gam, 'gam', '$\gamma$'))

def hp_deltafunc_lg_A_gamma(hip):
  return {**hp_deltafunc_lg_A(hip), **hp_deltafunc_gamma(hip)}

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
      "mu_lg_A": [-20., -10.],
      "sig_lg_A": [0., 10.],
      "mu_gam": [0., 10.],
      "sig_gam": [0., 10.],
      "low_lg_A": [-20., -10.],
      "high_lg_A": [-20., -10.],
      "low_gam": [0., 10.],
      "high_gam": [0., 10.],
      "lg_A": [-20., -10.],
      "gam": [0., 10.],
    })
