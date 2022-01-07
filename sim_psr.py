import os
from datetime import datetime
import numpy as np
from scipy.stats import truncnorm
import libstempo as T
import libstempo.plot as LP, libstempo.toasim as LT
from matplotlib import pyplot as plt

day = 24 * 3600
year = 365.25 * day

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
np.psrs = 26

# ==== Red noise population model (comment/uncomment) ==== #

# qCP, from Gaussian truncated https://stackoverflow.com/a/37338391
# a = (lower - mu) / sigma, b = (upper - mu) / sigma
#gamma = truncnorm.rvs(a=(0-13/3)/2, b=(10-13/3)/2, loc=13/3, scale=2, size=np.psrs)
#log10_AA = truncnorm.rvs((-20-(-13.3))/2, (-10-(-13.3))/2, loc=-13.3, scale=2, size=np.psrs)
#name = 'qcp'

# qCP2, gamma fixed
# a = (lower - mu) / sigma, b = (upper - mu) / sigma
gamma = np.repeat(13/3, np.psrs)
log10_AA = truncnorm.rvs((-20-(-13.3))/2, (-10-(-13.3))/2, loc=-13.3, scale=2, size=np.psrs)
name = 'qcp2'

# CP, delta function
#gamma = np.repeat(13/3, np.psrs)
#log10_AA = np.repeat(-13.3, np.psrs)
#name = 'cp'

# ======================================================== #

# Creating a unique name for the dataset
unique_id = datetime.now().strftime("%y%m%d_%H%M%S")
outdir = out + name + '_' + unique_id + '/'
os.mkdir(outdir)

inj_params = np.concatenate([np.arange(0,np.psrs,1)[:,np.newaxis], \
                             log10_AA[:,np.newaxis], \
                             gamma[:,np.newaxis]], axis=1)

np.savetxt(outdir+'inj_params.txt', inj_params, fmt = ['%0.0i','%.18f','%.18f'])

#plt.hist(log10_AA,density=True,label='samples')
#log10_A_xvals = np.linspace(-10,-20,100)
#log10_A_probvals = truncnorm((-20-(-13.3))/2, (-10-(-13.3))/2, loc=-13.3, scale=2).pdf(log10_A_xvals)
#plt.plot(log10_A_xvals, log10_A_probvals)
#plt.xlabel('log10_A')
#plt.ylabel('Probability')
#plt.legend()
#plt.savefig(outdir+'log10_A.png')
#plt.close()

for ii in range(np.psrs):
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
  
  #LT.add_rednoise(psr,10**log10_AA[ii],gamma[ii],components=30)
  add_rednoise_nostoch(psr,10**log10_AA[ii],gamma[ii],components=30)

  # Do or not to do...
  psr.fit() # Does not work for fakepulsar

  # Rename pulsar
  psr.name = 'J' + "{0:09.4f}".format(ii/10000).replace('.','-')

  # Saving
  psr.savepar(outdir + 'psr_' + str(ii) + '.par')
  psr.savetim(outdir + 'psr_' + str(ii) + '.tim')

print('Saved simulated data in ', outdir)
