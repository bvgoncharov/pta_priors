import os
import bilby
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import random
import pandas as pd
import matplotlib.patches as pat
import json


# output directory
outdir = '/home/celestialsapien/epta_dr2_out__/'

# setting up arrays to ignore pulsars with no dm or sn
psrs_no_dm = ['J0030+0451', 'J1455-3330', 'J1738+0333', 'J2322+2057']
psrs_no_sn = ['J0751+1807', 'J1600-3053', 'J1024-0719', 'J1640+2224', 'J1730-2304', 'J1751-2857', 'J1801-1417', 'J1804-2717', 'J1843-1113', 'J1857+0943', 'J1910+1256', 'J1911+1347', 'J1918-0642', 'J2124-3358', 'J2322+2057']+ ['J1909-3744']

# setting up arrays to append veles of meshgrid values, Pulsar names etc...
XD_arr = []
YD_arr = []
XR_arr = []
YR_arr = []

H_D_arr = []
H_R_arr = []

sigma_D_arr = []
sigma_R_arr = []

pulsar = []

for root, dirs, files in os.walk(outdir):
    if 'chain_1.txt' in files:
        dir_name = os.path.basename(root)
        print(f'start for {dir_name}')

        out = os.path.join(root, 'chain_1.txt')
        data = np.loadtxt(f'epta_dr2_out__/{dir_name}/chain_1.txt')
        pars = np.loadtxt(f'epta_dr2_out__/{dir_name}/pars.txt', dtype='str')
        
        data_pd = pd.DataFrame(data, columns=np.hstack([pars,np.array(['logl','logpr','ar','ptar'])]))
        
        rngma = data_pd[dir_name[-10:]+'_red_noise_gamma']
        rnlg10A = data_pd[dir_name[-10:]+'_red_noise_log10_A']

        dmgma = data_pd[dir_name[-10:]+'_dm_gp_gamma']
        dmlg10A = data_pd[dir_name[-10:]+'_dm_gp_log10_A']

        hist_d, xedges_d, yedges_d = np.histogram2d(dmgma, dmlg10A, bins=100)
        hist_r, xedges_r, yedges_r = np.histogram2d(rngma, rnlg10A, bins=100)

        xcenters_d = np.linspace(min(xedges_d), max(xedges_d), 100)
        ycenters_d = np.linspace(min(yedges_d), max(yedges_d), 100)
        xcenters_r = np.linspace(min(xedges_r), max(xedges_r), 100)
        ycenters_r = np.linspace(min(yedges_r), max(yedges_r), 100)

        XR, YR = np.meshgrid(xcenters_r, ycenters_r)
        XD, YD = np.meshgrid(xcenters_d, ycenters_d)

        hist_d_flat = hist_d.flatten()
        inds_d = np.argsort(hist_d_flat)[::-1]
        hist_d_flat = hist_d_flat[inds_d]
        hist_r_flat = hist_r.flatten()
        inds_r = np.argsort(hist_r_flat)[::-1]
        hist_r_flat = hist_r_flat[inds_r]
        
        sm_d = np.cumsum(hist_d_flat)
        sm_r = np.cumsum(hist_r_flat)
        
        sm_d /=np.max(sm_d)
        sm_r /=np.max(sm_r)
        
        threshold_index_d = hist_d_flat[sm_d<=0.393][-1]
        threshold_index_r = hist_r_flat[sm_r<=0.393][-1]

        XD_arr.append(XD)
        YD_arr.append(YD)
        XR_arr.append(XR)
        YR_arr.append(YR)
        H_D_arr.append(hist_d)
        H_R_arr.append(hist_r)
        sigma_D_arr.append(threshold_index_d)
        sigma_R_arr.append(threshold_index_r)
        pulsar.append(dir_name[-10:])

        print(f'end for {dir_name}')



XD_arr = [arr.tolist() for arr in XD_arr]
YD_arr = [arr.tolist() for arr in YD_arr]
XR_arr = [arr.tolist() for arr in XR_arr]
YR_arr = [arr.tolist() for arr in YR_arr]

H_D_arr = [arr.tolist() for arr in H_D_arr]
H_R_arr = [arr.tolist() for arr in H_R_arr]

data = {
    'XD_arr': XD_arr,
    'YD_arr': YD_arr,
    'XR_arr': XR_arr,
    'YR_arr': YR_arr,
    'H_D_arr': H_D_arr, 
    'H_R_arr': H_R_arr,
    'sigma_D_arr': sigma_D_arr,
    'sigma_R_arr': sigma_R_arr,
    'pulsar': pulsar
}

with open('pta_gwb_priors/sigma_one/epta_output.json', 'w') as file:
    json.dump(data, file, indent=4)
