import os
from datetime import datetime
import numpy as np
from scipy.stats import truncnorm
import libstempo as T
import libstempo.plot as LP, libstempo.toasim as LT
from matplotlib import pyplot as plt

from astropy.stats import LombScargle
from enterprise_warp.libstempo_warp import red_psd

day = 24 * 3600
year = 365.25 * day

def enterprise_to_tempo2_A_gw(a_gw):
    return a_gw**2 * year**3 / 12 / np.pi**2

def add_rednoise_nostoch(psr, A, gamma, components=10, seed=None):
    """
    Add red noise without stochastic realization fluctuations
    (pure power law)
    """

    if seed is not None:
        np.random.seed(seed)

    t = psr.toas()
    minx, maxx = np.min(t), np.max(t)
    x = (t - minx) / (maxx - minx)
    T = (day / year) * (maxx - minx)

    size = 2 * components
    F = np.zeros((psr.nobs, size), "d")
    f = np.zeros(size, "d")

    for i in range(components):
        F[:, 2 * i] = np.cos(2 * np.pi * (i + 1) * x)
        F[:, 2 * i + 1] = np.sin(2 * np.pi * (i + 1) * x)

        f[2 * i] = f[2 * i + 1] = (i + 1) / T

    norm = A ** 2 * year ** 2 / (12 * np.pi ** 2 * T)
    prior = norm * f ** (-gamma)

    y = np.sqrt(prior)# * np.random.randn(size)
    psr.stoas[:] += (1.0 / day) * np.dot(F, y)

#data = '/home/bgonchar/pta_gwb_priors/data/dr2_timing_20200607/'
data = '/fred/oz031/pta_gwb_priors_out/sim_data/'
out = '/fred/oz031/pta_gwb_priors_out/sim_data/'
n_psrs = 26

# ==== Red noise population model (comment/uncomment) ==== #

add_simple_red = True

# qCP, from Gaussian truncated https://stackoverflow.com/a/37338391
# a = (lower - mu) / sigma, b = (upper - mu) / sigma
gamma = truncnorm.rvs(a=(0.-5.)/2., b=(10.-5.)/2., loc=5., scale=2., size=n_psrs)
log10_AA = truncnorm.rvs((-20-(-14.3))/1.3, (-10-(-14.3))/1.3, loc=-14.3, scale=1.3, size=n_psrs)
#name = 'qcp'

# qCP2, gamma fixed
# a = (lower - mu) / sigma, b = (upper - mu) / sigma
#gamma = np.repeat(13/3, n_psrs)
#log10_AA = truncnorm.rvs((-20-(-13.3))/2, (-10-(-13.3))/2, loc=-13.3, scale=2, size=n_psrs)
#name = 'qcp2'

# CP, delta function
#gamma = np.repeat(13/3, n_psrs)
#log10_AA = np.repeat(-13.3, n_psrs)
#name = 'cp'

# ===== Second red noise term in the pulsar ===== #

two_rn_terms = True

# qCP2, gamma fixed
# a = (lower - mu) / sigma, b = (upper - mu) / sigma
gamma_2 = np.repeat(13/3, n_psrs)
log10_AA_2 = truncnorm.rvs((-20-(-13.8))/0.4, (-10-(-13.8))/0.4, loc=-13.8, scale=0.4, size=n_psrs)
name = 'rnqcp'

# ====== GWB ===== #

add_gwb = False
f_low = 4.5388525780680866447e-09 # Check before running, 1/day/(psr.stoas[-1]-psr.stoas[0])

#log10_A_gw = -13.3
#gwamp_tempo2 = 10**(log10_A_gw) # enterprise_to_tempo2_A_gw(10**log10_A_gw)
#print('GW amplitude in tempo2 convention: ', gwamp_tempo2)
#
#gwb = LT.GWB(gwAmp=gwamp_tempo2,flow=f_low)
#name = 'rngwb'

plot_psd = True
psr_psds = []
psr_fs = []
psr_f = np.arange(f_low, 30*f_low, f_low)



# ======================================================== #

# Creating a unique name for the dataset
unique_id = datetime.now().strftime("%y%m%d_%H%M%S")
outdir = out + name + '_' + unique_id + '/'
os.mkdir(outdir)

inj_params = np.concatenate([np.arange(0,n_psrs,1)[:,np.newaxis], \
                             log10_AA[:,np.newaxis], \
                             gamma[:,np.newaxis]], axis=1)
fmt = ['%0.0i','%.18f','%.18f']

if two_rn_terms:
  inj_params = np.concatenate([inj_params,log10_AA_2[:,np.newaxis],gamma_2[:,np.newaxis]],axis=1)
  fmt += ['%.18f','%.18f']

np.savetxt(outdir+'inj_params.txt', inj_params, fmt = fmt)

#plt.hist(log10_AA,density=True,label='samples')
#log10_A_xvals = np.linspace(-10,-20,100)
#log10_A_probvals = truncnorm((-20-(-13.3))/2, (-10-(-13.3))/2, loc=-13.3, scale=2).pdf(log10_A_xvals)
#plt.plot(log10_A_xvals, log10_A_probvals)
#plt.xlabel('log10_A')
#plt.ylabel('Probability')
#plt.legend()
#plt.savefig(outdir+'log10_A.png')
#plt.close()

for ii in range(n_psrs):
  # Take one PPTA pulsar as a template (less ToAs, for speed)
  #psr = T.tempopulsar(parfile = data + 'J1832-0836_template.par',
  #                    timfile = data + 'J1832-0836_template.tim')

  #LT.make_ideal(psr)
  #psr = LT.fakepulsar(parfile = data + 'J1832-0836_template.par',
  #                    obstimes=np.arange(53000,55555,30),
  #                    toaerr=0.1)
  psr = LT.fakepulsar(parfile = '/fred/oz031/pta_gwb_priors_out/sim_data/tempo2_fake_test/testfake.par', obstimes=np.arange(53000,55555,30), toaerr=0.1)

  LT.add_efac(psr,efac=1.1)
  #LT.add_equad(psr,equad=1e-6)

  if add_simple_red:  
    LT.add_rednoise(psr,10**log10_AA[ii],gamma[ii],components=30)
    #add_rednoise_nostoch(psr,10**log10_AA[ii],gamma[ii],components=30)

  if two_rn_terms:
    LT.add_rednoise(psr,10**log10_AA_2[ii],gamma_2[ii],components=30)
    #add_rednoise_nostoch(psr,10**log10_AA_2[ii],gamma_2[ii],components=30)

  if add_gwb:
    gwb.add_gwb(psr)

  # Do or not to do...
  psr.fit() # Does not work for fakepulsar

  # Rename pulsar
  psr.name = 'J' + "{0:09.4f}".format(ii/10000).replace('.','-')

  # Saving
  psr.savepar(outdir + 'psr_' + str(ii) + '.par')
  psr.savetim(outdir + 'psr_' + str(ii) + '.tim')

  if plot_psd:
    ls_psr = LombScargle(psr.toas().astype('float64')*day, \
        psr.residuals().astype('float64'), \
        normalization='psd')
    psr_pow = ls_psr.power(psr_f)
    psr_psd = psr_pow/np.sqrt(psr_f)
    fsamp_approx = 1/(2*7*24*60*60)
    psr_psds.append(psr_psd)

#
  #  plt.loglog(psr_f, psr_psd,label='Total AstroPy')
  #  plt.plot(psr_f,red_psd(psr_f,10**log10_AA[ii],gamma[ii]),label='Red noise')
  #  if two_rn_terms:
  #    plt.plot(psr_f,red_psd(psr_f,10**log10_AA_2[ii],gamma_2[ii]),label='(Quasi-)Common red noise')
  #  if add_gwb:
  #    plt.plot(psr_f,red_psd(psr_f,gwamp_tempo2,13/3),label='GWB')
  #  plt.legend()
  #  plt.xlabel('Frequency [Hz]')
  #  plt.ylabel('PSD [s^3]')
  #  plt.savefig(outdir+'psd_'+str(ii)+'.png')
  #  plt.close()

if plot_psd:
  # A weird number is an empirical factor of 339.4738902672604 is for normalizing AstroPy LombScargle
  red_analytical_psds = []
  for ii in range(n_psrs):
    plt.loglog(psr_f, psr_psds[ii]*339.4738902672604,color='black',alpha=0.5,label='Total AstroPy')
    if add_simple_red:
      red_psd_ii = red_psd(psr_f,10**log10_AA[ii],gamma[ii])
      red_analytical_psds.append(red_psd_ii)
    #  plt.plot(psr_f,red_psd_ii,color='red',alpha=0.5,linewidth=0.3,label='Red noise')
    if two_rn_terms:
      plt.plot(psr_f,red_psd(psr_f,10**log10_AA_2[ii],gamma_2[ii]),label='(Quasi-)Common red noise')
  if add_gwb:
    plt.plot(psr_f,red_psd(psr_f,gwamp_tempo2,13/3),label='GWB')
  if add_simple_red:
    red_analytical_total = np.sum(red_analytical_psds,axis=0)/n_psrs
    plt.plot(psr_f,red_psd_ii,color='red',linewidth=2,label='Red noise mean')
  mean_psd_eval = np.sum(psr_psds,axis=0)/n_psrs*339.4738902672604
  plt.loglog(psr_f, mean_psd_eval, color='green',linewidth=2)
  #plt.ylim([np.min(mean_psd_eval),np.max(mean_psd_eval)])
  plt.xlabel('Frequency [Hz]')
  plt.ylabel('PSD [s^3]')
  plt.savefig(outdir+'psd.png')
  plt.close()

import ipdb; ipdb.set_trace()

print('Saved simulated data in ', outdir)
