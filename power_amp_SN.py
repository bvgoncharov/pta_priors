#!/bin/python
import sys
sys.path.insert(0, "/home/celestialsapien/enterprise_warp-dev")
import pandas as pd
import numpy as np
import os

from scipy import interpolate
from scipy.stats import gaussian_kde
import random

import bilby

from enterprise_warp import enterprise_warp
from enterprise_warp import bilby_warp
from enterprise_warp.enterprise_warp import get_noise_dict
from enterprise_extensions import hypermodel
import hierarchical_models as hm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


opts = hm.parse_commandline()

configuration = hm.HierarchicalInferenceParams
params = enterprise_warp.Params(opts.prfile,opts=opts,custom_models_obj=configuration)

hr = hm.HyperResult(opts, suffix=params.par_suffix)
hr.main_pipeline()

# print(hr.results[1].posterior.keys())
# pow_values = np.array(hr.result.posterior['red_noise_pow'])
# print(pow_values)

# access json files
outdir = '/home/celestialsapien/epta_dr3_out/20240616_epta_trim_rcl_cpfg/'

XR_arr = []
YR_arr = []
ZR_joint_norm_arr = []
sigma_R_arr = []
pulsar = []

for i, (root, dirs, files) in enumerate(os.walk(outdir)):
    dirs.sort()
    if 'epta_dr3_out_result.json' in files:
        out = os.path.join(root, 'epta_dr3_out_result.json')
        result = bilby.result.read_in_result(out)
        dir_name = os.path.basename(root)
        import ipdb; ipdb.set_trace()
        
        # reduce data into np arrays
        rngma = np.array(result.posterior[f'{dir_name[-10:]}_red_noise_gamma'].values)
        pow_values = np.array(hr.results[i-1].posterior['red_noise_pow'])

        points = 1000
        xr = np.linspace(min(rngma), max(rngma), points)
        yr = np.linspace(min(pow_values), max(pow_values), points)
        XR, YR = np.meshgrid(xr, yr)

        # data arrays
        data_R = np.array([rngma, pow_values])
        kde_joint_R = gaussian_kde(data_R)(np.array([XR.ravel(), YR.ravel()]))

        # this is the z axis values that should be plotted over the grid
        ZR_joint = np.reshape(kde_joint_R, (points, points))

        # Normalization
        ZR_joint_norm = ZR_joint / ZR_joint.sum()

        def get_index_of_sigma(data_norm):
            cdf = np.cumsum(data_norm.ravel())
            arr = cdf - 0.6827
            pos_arr = arr[arr >= 0]
            min_pos_val = pos_arr.min()
            sigma_index = np.where(arr == min_pos_val)[0]
            return data_norm.ravel()[sigma_index]

        sigma_R = get_index_of_sigma(ZR_joint_norm)

        XR_arr.append(XR)
        YR_arr.append(YR)

        sigma_R_arr.append(sigma_R)
        ZR_joint_norm_arr.append(ZR_joint_norm)
        pulsar.append(dir_name[-10:])
        print('done! for pulsar no.', i)

# Plot
for i in range (len(pulsar)):
    # plt.contourf(XR_arr[i], YR_arr[i], ZR_joint_norm_arr[i], levels=50, cmap='Blues')
    plt.contour(XR_arr[i], YR_arr[i], ZR_joint_norm_arr[i], levels=[sigma_R_arr[i][0]], colors= 'black', linewidths=0.7)

plt.xlabel('$\gamma$ (Spectral Index)')
plt.ylabel('log10_power')
plt.yscale('log')
# plt.ylim(0,  7*pow(10, -11))
plt.savefig('GvP_SN(log).png', format = 'png', dpi=300)
plt.savefig('GvP_SN(log).pdf', format = 'pdf')
plt.close()