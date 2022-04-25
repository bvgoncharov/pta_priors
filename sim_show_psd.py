import numpy as np
from matplotlib import pyplot as plt
from enterprise_warp.libstempo_warp import red_psd

#data = '/fred/oz031/pta_gwb_priors_out/sim_data/rnqcp_220308_075224/' # GH
data = '/fred/oz031/pta_gwb_priors_out/sim_data/rnqcp_220208_073559/'

f_low = 4.5388525780680866447e-09
psr_f = np.arange(f_low, 30*f_low, f_low)

params = np.loadtxt(data+'inj_params.txt')

for ii in range(len(params)):
  spin_psd = red_psd(psr_f, 10**params[ii,1], params[ii,2])
  qcp_psd = red_psd(psr_f, 10**params[ii,3], params[ii,4])
  plt.loglog(psr_f, spin_psd, color='red')
  plt.loglog(psr_f, qcp_psd, color='blue')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [s^3]')
plt.tight_layout()
plt.savefig(data + 'sim_psd.png')
plt.close()

import ipdb; ipdb.set_trace()
