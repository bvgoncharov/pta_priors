"""
Construct a KDE-based prior object
"""
import json
import pickle
import numpy as np
from matplotlib import pyplot as plt
from chainconsumer import ChainConsumer
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import QuantileTransformer

def get_inv_sq_errors(list_val_low_high, prior_width):
  """
  Takes a list with value, low spread (i.e. -1 sigma), 
  and high spread (i.e. 1 sigma)
  """
  return (list_val_low_high[1] + list_val_low_high[2]) / prior_width

def evaluate_prob(kde, x_min, x_max, y_min, y_max):
  nbins = 300
  xi, yi = np.mgrid[x_min:x_max:nbins*1j, y_min:y_max:nbins*1j]
  zi = kde.score_samples(np.vstack([xi.flatten(), yi.flatten()]).T)
  return xi, yi, zi

def plot_sn_bn_vals():
  for key, dct in {'SN': sn_vals, 'BN/GN': bn_gn_vals}.items():
  
    dct['log10_A'] = np.array(dct['log10_A'])
    dct['gamma'] = np.array(dct['gamma'])
  
    plt.errorbar(dct['log10_A'][:,0], dct['gamma'][:,0], xerr = dct['log10_A'][:,1:].T, yerr = dct['gamma'][:,1:].T, label = key, fmt = '.', marker='o', markeredgecolor='none')
  plt.xlabel('$\log_{10}A$')
  plt.ylabel('$\gamma$')
  plt.legend()

outdir = '/fred/oz031/pta_gwb_priors_out/'
vals_dir = '/home/bgonchar/soft/ppta_dr2_noise_analysis/reproduce_figures/'
with open(vals_dir + 'vals_spin.json', 'r') as jf:
  sn_vals = json.load(jf)
with open(vals_dir + 'vals_band_system.json', 'r') as jf:
  bn_gn_vals = json.load(jf)

# Let's make weights proportional to inverse squared of relative errors
sn = {}
snw = {}
prior_width = 10 # Prior range is 10 for both lgA (-20,-10) and gamma (0,10)
for par in sn_vals.keys():
  sn[par] = [triple[0] for triple in sn_vals[par]]
  snw[par] = [get_inv_sq_errors(triple, prior_width) for triple in sn_vals[par]]
bn = {}
bnw = {}
for par in bn_gn_vals.keys():
  bn[par] = [triple[0] for triple in bn_gn_vals[par]]
  bnw[par] = [get_inv_sq_errors(triple, prior_width) for triple in bn_gn_vals[par]]

# Let's join A and gamma for the right formatting
sn = np.array([sn['log10_A'], sn['gamma']]).T
bn = np.array([bn['log10_A'], bn['gamma']]).T
snw = np.array([np.sqrt(aa**(-2)+gg**(-2)) for aa, gg in zip(snw['log10_A'],snw['gamma'])])
bnw = np.array([np.sqrt(aa**(-2)+gg**(-2)) for aa, gg in zip(bnw['log10_A'],bnw['gamma'])])
snbn = np.concatenate([sn, bn])
snbnw = np.concatenate([snw, bnw])

# Let's fit a KDE
kdesn = KernelDensity(bandwidth=0.1)
kdesn.fit(sn)#, sample_weight=snw)
kdebn = KernelDensity(bandwidth=0.1)
kdebn.fit(bn)#, sample_weight=bnw)
kdesnbn = KernelDensity(bandwidth=0.1)
kdesnbn.fit(snbn)#, sample_weight=snbnw)

# Let's plot KDE priors
kdesn_samples = kdesn.sample(1000)
xi, yi, zi = evaluate_prob(kdesn, -20, -10, 0, 10)
plt.pcolormesh(xi, yi, np.exp(zi.reshape(xi.shape)), shading='auto', cmap=plt.cm.Greens_r)
plot_sn_bn_vals()
plt.savefig(outdir+'kde_sn.png')
plt.close()

kdebn_samples = kdebn.sample(1000)
xi, yi, zi = evaluate_prob(kdebn, -20, -10, 0, 10)
plt.pcolormesh(xi, yi, np.exp(zi.reshape(xi.shape)), shading='auto', cmap=plt.cm.Greens_r)
plot_sn_bn_vals()
plt.savefig(outdir+'kde_bngn.png')
plt.close()

kdesnbn_samples = kdesnbn.sample(1000)
xi, yi, zi = evaluate_prob(kdesnbn, -20, -10, 0, 10)
plt.pcolormesh(xi, yi, np.exp(zi.reshape(xi.shape)), shading='auto', cmap=plt.cm.Greens_r)
plot_sn_bn_vals()
plt.savefig(outdir+'kde_snbngn.png')
plt.close()

with open(outdir + 'kdesn_bw0.1.pkl', 'wb') as outf:
  pickle.dump(kdesn, outf)

# ================================== EXTRAS ================================== #

## For the useful reference, an example if a KDE distribution had no covariance.
## ([!!!] all covariance information is destroyed [!!!])
## Now, train the quantile transformer, which can map uniform prior to the
## KDE prior for nested sampling (from "unit cube")
## More information: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html#sklearn.preprocessing.QuantileTransformer.inverse_transform
#N_TRAIN_SAMPLES = 1000000000
#random_state = 100500
##qt = QuantileTransformer(n_quantiles=1000000, output_distribution='uniform',
##                         random_state=random_state, subsample=1000000)
#qt = QuantileTransformer(n_quantiles=500000, output_distribution='uniform',
#                         random_state=random_state, subsample=1000000)
#
#kdesn_train_samples = kdesn.sample(N_TRAIN_SAMPLES)
#trans_qt = qt.fit(kdesn_train_samples)
## Later on: samples_from_prior = trans_qt.inverse_transform(unit_sq_samples)
#
##with open(outdir + 'kdesn_bw1_qtrans.pkl', 'wb') as outf:
##  pickle.dump(trans_qt, outf)
#
## Verifying by eye samples from the unit square to match the KDE distribution
## (Statistical consistency test might be useful too)
#cc = ChainConsumer()
#kdesamples = kdesn.sample(1000000)
#nested = trans_qt.inverse_transform(np.random.uniform(size=(1000000,2)))
#cc.add_chain(kdesamples, parameters=["A", "gamma"], name='KDE Prior')
#cc.add_chain(nested, parameters=["A", "gamma"], name='Nested samples')
#cc.configure(usetex=False)
#fig = cc.plotter.plot(filename = outdir + 'test.png')
