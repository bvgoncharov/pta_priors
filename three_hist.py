import bilby
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


plt.rcParams.update({
  "text.usetex": False,
  "font.family": "serif",
  #"font.serif": ["Palatino"],
})
font = {'family' : 'serif',
        'size'   : 17}

out_1 = '/home/celestialsapien/epta_dr2_out/20240616_epta_rcl_cpfg/DMixNgauss_sn_full_2024/DMixNgauss_sn_full_2024_result.json'
result_1 = bilby.result.read_in_result(out_1)
fnorm_arr_1 = result_1.posterior["fnorm"]
sig_gam_2_arr_1 = result_1.posterior["sig_gam_2"]
sig_lg10A_2_arr_1 = result_1.posterior["sig_lg_A_2"]

out_2 = '/home/celestialsapien/epta_dr2_out_rec/20240616_epta_rcl_cpfg/DMixNgauss_sn_full_2024/DMixNgauss_sn_full_2024_result.json'
result_2 = bilby.result.read_in_result(out_2)
fnorm_arr_2 = result_2.posterior["fnorm"]
sig_gam_2_arr_2 = result_2.posterior["sig_gam_2"]
sig_lg10A_2_arr_2 = result_2.posterior["sig_lg_A_2"]

df = pd.DataFrame({'Fnorm_1': fnorm_arr_1,
                   'Fnorm_2': fnorm_arr_2,
                   'sig_gam_2_1': sig_gam_2_arr_1,
                   'sig_gam_2_2': sig_gam_2_arr_2, 
                   'sig_lg_A_2_1': sig_lg10A_2_arr_1,
                   'sig_lg_A_2_2': sig_lg10A_2_arr_2})

# Plot using sns.kdeplot
fig = plt.figure()
axes = fig.add_subplot(111)
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x="Fnorm_1", fill=True, palette="crest", color='#120309', alpha=.5, linewidth=2, common_norm=True, common_grid=True, label='with common red noise filter',bw_adjust=0.05,bw_method='silverman')
sns.kdeplot(data=df, x="Fnorm_2", fill=True, palette="crest", color='#2e0f15', alpha=.5, linewidth=2, common_norm=True, common_grid=True, label='without common red noise filter',bw_adjust=0.05,linestyle='--')

# plt.xlabel(r'$\nu_{\mathrm{NORM}}$')
# plt.ylabel('Posterior Density')
# plt.title(r'$\nu_{\mathrm{NORM}}$ Plot')
axes.set_xlabel('', fontdict=font)
axes.set_ylabel('.........', fontdict=font) 
axes.tick_params(axis='y', labelsize = font['size'])
axes.tick_params(axis='x', labelsize = font['size'])
plt.xlim([0, 1])
# plt.ylim([1e-2, 1e1])
plt.tight_layout()
plt.savefig('/home/celestialsapien/pta_gwb_priors/sigma_one/Histogram/Fnorm.png', format='png', dpi=300)
plt.close()

plt.figure(figsize=(10, 6))

sns.kdeplot(data=df, x="sig_gam_2_1", fill=True, palette="crest", color='#120309', alpha=.5, linewidth=2, common_norm=True, common_grid=True, label='with common red noise filter',bw_adjust=0.05,bw_method='silverman')
sns.kdeplot(data=df, x="sig_gam_2_2", fill=True, palette="crest", color='#2e0f15', alpha=.5, linewidth=2, common_norm=True, common_grid=True, label='without common red noise filter',bw_adjust=0.05,linestyle='--')
# plt.xlabel(r'$\sigma_{\gamma 2}$')
# plt.ylabel('Posterior Density')
# plt.title(r'$\sigma_{\gamma 2}$ Histogram')
axes.set_xlabel('........', fontdict=font)
axes.set_ylabel('.........', fontdict=font) 
axes.tick_params(axis='y', labelsize = font['size'])
axes.tick_params(axis='x', labelsize = font['size'])
plt.xlim([0, 10])
plt.tight_layout()
plt.savefig('/home/celestialsapien/pta_gwb_priors/sigma_one/Histogram/Sigma_Gamma_2.png', format='png', dpi=300)
plt.close()

#bins = 10**(np.linspace(-3,1,500))
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x="sig_lg_A_2_1", fill=True, palette="crest", color='#120309', alpha=.5, linewidth=2, common_norm=True, common_grid=True, label='with common red noise filter',bw_adjust=0.05,bw_method='silverman')
sns.kdeplot(data=df, x="sig_lg_A_2_2", fill=True, palette="crest", color='#2e0f15', alpha=.5, linewidth=2, common_norm=True, common_grid=True, label='without common red noise filter',bw_adjust=0.05,linestyle='--')
#plt.hist(df['sig_lg_A_2_1'],bins=1000,density=True,log=True,alpha=0.5)
# sns.kdeplot(data=df, x="sig_lg_A_2_2", fill=False, palette="crest", alpha=1, linewidth=0.5, linestyle='--',common_norm=True, common_grid=True, color='black',label='without common red noise filter',)
# plt.xlabel(r'$\sigma_{\log_{10} A 2}$')
# plt.ylabel('Posterior Density')
# plt.title(r'$\sigma_{\log_{10} A 2}$ Plot')
axes.set_xlabel(r'$$\\sigma\_{\\log_{10} A^2}$$', fontdict=font)
#axes.set_ylabel('.........', fontdict=font) 
axes.tick_params(axis='y', labelsize = font['size'])
axes.tick_params(axis='x', labelsize = font['size'])
plt.xlim([0, 5])
plt.ylim([5e-2, 1.6])
plt.yscale('log')
plt.tight_layout()
plt.savefig('/home/celestialsapien/pta_gwb_priors/sigma_one/Histogram/Sigma_Log10_A_2.png', format='png', dpi=300)
plt.close()

# 