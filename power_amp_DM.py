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

# import ipdb; ipdb.set_trace()

# print(hr.results[1].posterior.keys())
# pow_values = np.array(hr.result.posterior['dm_gp_pow'])
# print(pow_values)

# access json files
outdir = '/home/celestialsapien/epta_dr3_out/20240616_epta_trim_rcl_cpfg/'

XD_arr = []
YD_arr = []
ZD_joint_norm_arr = []
sigma_D_arr = []
pulsar = []

for i, (root, dirs, files) in enumerate(os.walk(outdir)):
    dirs.sort()
    if 'epta_dr3_out_result.json' in files:
        out = os.path.join(root, 'epta_dr3_out_result.json')
        result = bilby.result.read_in_result(out)
        dir_name = os.path.basename(root)
        
        # reduce data into np arrays
        dmgma = np.array(result.posterior[f'{dir_name[-10:]}_dm_gp_gamma'].values)
        pow_values = np.array(hr.results[i-1].posterior['dm_gp_pow'])

        points = 1000
        xd = np.linspace(min(dmgma), max(dmgma), points)
        yd = np.linspace(min(pow_values), max(pow_values), points)
        XD, YD = np.meshgrid(xd, yd)

        # data arrays
        data_D = np.array([dmgma, pow_values])
        kde_joint_D = gaussian_kde(data_D)(np.array([XD.ravel(), YD.ravel()]))

        # this is the z axis values that should be plotted over the grid
        ZD_joint = np.reshape(kde_joint_D, (points, points))

        # Normalization
        ZD_joint_norm = ZD_joint / ZD_joint.sum()

        def get_index_of_sigma(data_norm):
            cdf = np.cumsum(data_norm.ravel())
            arr = cdf - 0.6827
            pos_arr = arr[arr >= 0]
            min_pos_val = pos_arr.min()
            sigma_index = np.where(arr == min_pos_val)[0]
            return data_norm.ravel()[sigma_index]

        sigma_D = get_index_of_sigma(ZD_joint_norm)

        XD_arr.append(XD)
        YD_arr.append(YD)

        sigma_D_arr.append(sigma_D)
        ZD_joint_norm_arr.append(ZD_joint_norm)
        pulsar.append(dir_name[-10:])
        print('done! for pulsar no.', i)

# Plot
for i in range (len(pulsar)):
    # plt.contourf(XD_arr[i], YD_arr[i], ZD_joint_norm_arr[i], levels=50, cmap='Blues')
    plt.contour(XD_arr[i], YD_arr[i], ZD_joint_norm_arr[i], levels=[sigma_D_arr[i][0]], colors= 'black', linewidths=0.7)

plt.xlabel('$\gamma$ (Spectral Index)')
plt.ylabel('log10_power')
plt.yscale('log')
# plt.ylim(0,  7*pow(10, -11))
plt.savefig('GvP_DM(log).png', format = 'png', dpi=300)
plt.savefig('GvP_DM(log).pdf', format = 'pdf')
plt.close()
