import os
from datetime import datetime
import numpy as np
from scipy.stats import truncnorm
import libstempo as T
import libstempo.plot as LP, libstempo.toasim as LT

#data = '/home/bgonchar/pta_gwb_priors/data/dr2_timing_20200607/'
data = '/fred/oz031/pta_gwb_priors_out/sim_data/'
out = '/fred/oz031/pta_gwb_priors_out/sim_data/'
N_psrs = 26

# ==== Red noise population model (comment/uncomment) ==== #

# qCP, from Gaussian truncated https://stackoverflow.com/a/37338391
# a = (lower - mu) / sigma, b = (upper - mu) / sigma
#gamma = truncnorm.rvs(a=(0-13/3)/2, b=(10-13/3)/2, loc=13/3, scale=2, size=N_psrs)
#log10_AA = truncnorm.rvs((-20-(-13.3))/2, (-10-(-13.3))/2, loc=-13.3, scale=2, size=N_psrs)
#name = 'qcp'

# qCP2, gamma fixed
# a = (lower - mu) / sigma, b = (upper - mu) / sigma
#gamma = np.repeat(13/3, N_psrs)
#log10_AA = truncnorm.rvs((-20-(-13.3))/2, (-10-(-13.3))/2, loc=-13.3, scale=2, size=N_psrs)
#name = 'qcp2'

# CP, delta function
#gamma = np.repeat(13/3, N_psrs)
#log10_AA = np.repeat(-13.3, N_psrs)
#name = 'cp'

# ======================================================== #

# Creating a unique name for the dataset
unique_id = datetime.now().strftime("%y%m%d_%H%M%S")
outdir = out + name + '_' + unique_id + '/'
os.mkdir(outdir)

inj_params = np.concatenate([np.arange(0,N_psrs,1)[:,np.newaxis], \
                             log10_AA[:,np.newaxis], \
                             gamma[:,np.newaxis]], axis=1)

np.savetxt(outdir+'inj_params.txt', inj_params, fmt = ['%0.0i','%.18f','%.18f'])

for ii in range(N_psrs):
  # Take one PPTA pulsar as a template (less ToAs, for speed)
  psr = T.tempopulsar(parfile = data + 'J1832-0836_template.par',
                      timfile = data + 'J1832-0836_template.tim')

  #LT.make_ideal(psr)
  #psr = LT.fakepulsar(parfile = data + 'J1832-0836_template.par',
  #                    obstimes=np.arange(53000,55555,30),
  #                    toaerr=0.1)


  LT.add_efac(psr,efac=0.8)
  #LT.add_equad(psr,equad=1e-6)
  
  LT.add_rednoise(psr,10**log10_AA[ii],gamma[ii])

  # Do or not to do...
  #psr.fit() # Does not work for fakepulsar

  # Rename pulsar
  psr.name = 'J' + "{0:09.4f}".format(ii/10000).replace('.','-')

  # Saving
  psr.savepar(outdir + 'psr_' + str(ii) + '.par')
  psr.savetim(outdir + 'psr_' + str(ii) + '.tim')

print('Saved simulated data in ', outdir)
