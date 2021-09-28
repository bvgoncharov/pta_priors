import json
import numpy as np
from scipy.stats import kde
from matplotlib import pyplot as plt

import hierarchical_models as hm

def draw_from_normal(mu_x, sigma_x, mu_y, sigma_y):
  x = np.random.normal(loc = mu_x, scale = sigma_x, size=1000)
  y = np.random.normal(loc = mu_y, scale = sigma_y, size=1000)
  return x, y

def normal_kde(xx, yy, x_min, x_max, y_min, y_max):
  k = kde.gaussian_kde([xx,yy])
  nbins = 300
  xi, yi = np.mgrid[x_min:x_max:nbins*1j, y_min:y_max:nbins*1j]
  zi = k(np.vstack([xi.flatten(), yi.flatten()]))
  return xi, yi, zi

outdir = '/fred/oz031/pta_gwb_priors_out/'
vals_dir = '/home/bgonchar/soft/ppta_dr2_noise_analysis/reproduce_figures/'
with open(vals_dir + 'vals_spin.json', 'r') as jf:
  sn_vals = json.load(jf)
with open(vals_dir + 'vals_band_system.json', 'r') as jf:
  bn_gn_vals = json.load(jf)

# Plotting density of inferred Gaussian SN prior

xx, yy = draw_from_normal(-13.98, 0.61, 2.84, 1.33)
xi, yi, zi = normal_kde(xx, yy, -20, -10, 0, 10)
import ipdb; ipdb.set_trace()
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.Greens_r)

for key, dct in {'SN': sn_vals, 'BN/GN': bn_gn_vals}.items():

  dct['log10_A'] = np.array(dct['log10_A'])
  dct['gamma'] = np.array(dct['gamma'])

  plt.errorbar(dct['log10_A'][:,0], dct['gamma'][:,0], xerr = dct['log10_A'][:,1:].T, yerr = dct['gamma'][:,1:].T, label = key, fmt = '.', marker='o', markeredgecolor='none')

plt.xlabel('$\log_{10}A$')
plt.ylabel('$\gamma$')
plt.legend()
plt.savefig(outdir+'normal_estimated.png')
plt.close()

import ipdb; ipdb.set_trace()

# Estimating KDE of SN parameters
