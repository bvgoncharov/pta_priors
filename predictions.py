import numpy as np
import random
from  matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker

from corner.core import quantile
# quantile(samples, (0.16, 0.84))

dict = {
    'Kulier et al., 2015': [1.58e-15, 2.51e-15],
    'Simon, 2023': [1.46e-15, 2.26e-15],
    'McWilliams et al., 2014': [1.07e-15, 1.51e-14],
    'Ravi et al., 2014': [6.51e-16, 2.10e-15],
    'Bonetti et al., 2018': [5.83e-16, 1.01e-15],
    'Ryu et al., 2018': [5.30e-16, 7.00e-16],
    'Ravi et al., 2015': [5.10e-16, 2.40e-15],
    'Wyithe et al., 2003': [4.77e-16, 8.84e-16],
    'Enoki et al., 2004': [4.70e-16, 1.25e-15],
    'Roebber et al., 2016': [4.00e-16, 7.23e-16],
    'Sesana, 2013': [3.50e-16, 1.50e-15],
    'Sesana et al., 2009': [2.79e-16, 8.21e-16],
    'Siwek et al., 2020': [2.50e-16, 1.00e-15],
    'Sesana et al., 2016': [2.15e-16, 7.08e-16],
    'Rosado et al., 2015': [1.91e-16, 2.01e-15],
    'Sesana et al., 2008': [1.15e-16, 2.88e-15],
    'Chen et al., 2019': [1.04e-16, 1.05e-15],
    'Kelley et al., 2017': [1.00e-16, 6.00e-16],
    'Rajagopal et al., 1995': [9.32e-17, 2.41e-16],
    'Rasskazov et al., 2017': [8.74e-17, 6.57e-16],
    'Jaffe et al., 2003': [8.10e-17, 1.50e-16],
    'Zhu et al., 2019': [6.10e-17, 2.40e-15],
    'Chen et al., 2020': [6.10e-17, 5.40e-16],
    # 'Dvorkin et al., 2017': [8.74e-17, 6.57e-16]
    'Dvorkin et al., 2017': [10**(-17.2), 10**(-16)]
}


dist_1 = np.random.normal((-14.3), (0.4), size=1000)
dist_2 = np.random.normal((-14), (0.3), size=1000)
df = pd.DataFrame({'dist_1': dist_1, 'dist_2': dist_2})
cut_1 = quantile(dist_1, (0.16, 0.84))
cut_2 = quantile(dist_2, (0.16, 0.84))

fig = plt.figure(figsize=(7, 8))
gs = fig.add_gridspec(2, 1, height_ratios=[0.49, 2])

axes1 = fig.add_subplot(gs[0])

sns.kdeplot(data=df, x="dist_1", fill=False, palette="crest", color='green', alpha=0.7, linewidth=1, common_norm=True, common_grid=True, bw_adjust=1, bw_method='silverman', ax=axes1, linestyle='--', dashes=[5, 2], zorder=-1)

sns.kdeplot(data=df, x="dist_1", fill=True, palette="crest", color='green', alpha=0.1, linewidth=0, common_norm=True, common_grid=True, bw_adjust=1, bw_method='silverman', ax=axes1, clip=cut_1, zorder=-1)


sns.kdeplot(data=df, x="dist_2", fill=False, palette="crest", color='0.3', alpha=0.7, linewidth=1, common_norm=True, common_grid=True, bw_adjust=1, bw_method='silverman', ax=axes1, zorder=-1)
sns.kdeplot(data=df, x="dist_2", fill=True, palette="crest", color='0.3', alpha=0.1, linewidth=0, common_norm=True, common_grid=True, bw_adjust=1, bw_method='silverman', ax=axes1, clip=cut_2, zorder=-1)

axes1.grid(True, which='both', axis='x', color='gray', alpha=0.05, linewidth=1.7, zorder=-2)
axes1.set_xlabel('')
axes1.set_ylabel('')
axes1.set_xlim([-17.2, -13.5])
axes1.set_yticks([])
axes1.set_xticklabels([''])
axes1.set_ylim([0, 3])


axes2 = fig.add_subplot(gs[1])
names = list(dict.keys())

lower_values = [np.log10(dict[name][0]) for name in names]
upper_values = [np.log10(dict[name][1]) for name in names]

bar_lengths = [upper - lower for upper, lower in zip(upper_values, lower_values)]

colors = plt.cm.Greens(np.linspace(0.45, 0.75, len(names)))

# Plot each bar with gradient
for i in range(len(names)):
    axes2.barh(names[i], bar_lengths[i], left=lower_values[i], color=colors[i], alpha=1)

y = np.linspace(-1, 30, 1000)
x11 = cut_1[0]
x12 = cut_1[1]
x21 = cut_2[0]
x22 = cut_2[1]

axes2.fill_betweenx(y, x11, x12, color='green', alpha=0.1, linewidth=0, zorder=-1)
axes2.fill_betweenx(y, x21, x22, color='0.3', alpha=0.1, linewidth=0, zorder=-1)
axes2.set_ylim([-0.7, 23.7])
axes2.invert_yaxis()
axes2.set_xlim([-17.2, -13.5])
axes2.grid(True, which='both', axis='x', color='gray', alpha=0.05, linewidth=1.7, zorder=-2)
axes2.grid(True, which='both', axis='y', color='gray', alpha=0.05, linewidth=1.7, linestyle='--', zorder=-2)
axes2.yaxis.tick_right()
axes2.yaxis.set_label_position('right')
axes2.tick_params(axis='y', colors='green', length=0)
axes2.set_xlabel(r'$\log_{10}(A_{\text{yr}})$')
plt.tight_layout(pad=2, h_pad=.7)
plt.show()