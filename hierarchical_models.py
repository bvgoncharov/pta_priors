import numpy as np
from bilby.core.prior import Uniform

from enterprise_warp.enterprise_models import StandardModels

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


class HierarchicalInferenceParams(StandardModels):
  """
  Configuration parameters for hierarchical inference: priors, models, etc.
  """
  def __init__(self,psr=None,params=None):
    super(HierarchicalInferenceParams, self).__init__(psr=psr,params=params)
    self.priors.update({
      "model": "norm_prod_lg_A_gamma",
      "mu_lg_A": [-20., -10.],
      "sig_lg_A": [0., 10.],
      "mu_gam": [0., 10.],
      "sig_gam": [0., 10.]
    })
