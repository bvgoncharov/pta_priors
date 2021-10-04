import numpy as np
import enterprise.constants as const

def powerlaw_psd(log10_A, gamma, f):
  return (10 ** log10_A) ** 2 / 12.0 / np.pi ** 2 * const.yr ** 3 * \
         (f * const.yr) ** (-gamma)

def powerlaw_power(log10_A, gamma, f_low, f_high):
  """ PSD integrated from f_low to f_high. I.e., from 1/Tobs to 30/Tobs. """
  return (10 ** log10_A) ** 2 / 12.0 / np.pi ** 2 * const.yr ** 3 * \
         const.yr ** (-gamma) / (1 - gamma) * \
         (f_high ** (1 - gamma) - f_low ** (1 - gamma))

def log10_A_of_gamma_at_psd(gamma, psd, f):
  """ This is powerlaw_psd() changed to calculate log10_A from psd, gamma """
  return np.log10(np.sqrt(psd * 12.0 * np.pi ** 2 * const.fyr ** 3 * \
                          (f * const.yr) ** gamma))

def log10_A_of_gamma_at_power(gamma, p0, f_low, f_high):
  """ This is powerlaw_power() changed to calculate log10_A from p0, gamma """
  return np.log10(np.sqrt(p0 * 12.0 * np.pi ** 2 * const.fyr ** 3 * (1 - gamma) * const.yr ** gamma / (f_high ** (1 - gamma) - f_low ** (1 - gamma))))

class PulsarEqualPSDLines(object):

  def __init__(self):
    self.tobs = {
      'J0437-4715': 15.0,
      'J0613-0200': 14.2,
      'J0711-6830': 14.2,
      'J1017-7156': 7.8,
      'J1022+1001': 14.2,
      'J1024-0719': 14.1,
      'J1045-4509': 14.2,
      'J1125-6014': 12.3,
      'J1446-4701': 17.4,
      'J1545-4550': 7.0,
      'J1600-3053': 14.2,
      'J1603-7202': 14.2,
      'J1643-1224': 14.2,
      'J1713+0747': 14.2,
      'J1730-2304': 14.2,
      'J1732-5049': 7.2,
      'J1744-1134': 14.2,
      'J1824-2452A': 13.8,
      'J1832-0836': 5.4,
      'J1857+0943': 14.2,
      'J1909-3744': 14.2,
      'J1939+2134': 14.1,
      'J2124-3358': 14.2,
      'J2129-5721': 13.9,
      'J2145-0750': 14.1,
      'J2241-5236': 8.2,
    }
    # Below values are from Goncharov, Reardon, Shannon, Zhu, Thrane (2020)
    self.log10_A = {
      'J0437-4715': -14.56,
      'J1832-0836': -12.5, # fiducial
      'J1824-2452A': -13.26,
      'J1909-3744': -14.74,
      'J1939+2134': -14.33,
    }
    self.gamma = {
      'J0437-4715': 2.99,
      'J1832-0836': 2.0, # fiducial
      'J1824-2452A': 5.02,
      'J1909-3744': 4.05,
      'J1939+2134': 5.39,
    }

  def get_psd(self, psr_name):
    return powerlaw_psd(self.log10_A[psr_name], self.gamma[psr_name], 1/self.tobs[psr_name]/const.yr)

  def get_p0(self, psr_name):
    return powerlaw_power(self.log10_A[psr_name], self.gamma[psr_name], \
           1/self.tobs[psr_name]/const.yr, 30/self.tobs[psr_name]/const.yr)

  def get_equipsd_log10_A(self, psr_name, gamma_arr):
    psd = self.get_psd(psr_name)
    return log10_A_of_gamma_at_psd(gamma_arr, psd, 1/self.tobs[psr_name]/const.yr)

  def get_equipower_log10_A(self, psr_name, gamma_arr):
    p0 = self.get_p0(psr_name)
    return log10_A_of_gamma_at_power(gamma_arr, p0, \
                                     1/self.tobs[psr_name]/const.yr, \
                                     30/self.tobs[psr_name]/const.yr)
