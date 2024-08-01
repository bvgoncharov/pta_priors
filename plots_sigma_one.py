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
from scipy.ndimage import gaussian_filter
import matplotlib.patches as patches
from matplotlib.patches import Ellipse, PathPatch
from matplotlib.path import Path


plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif",
  #"font.serif": ["Palatino"],
})
font = {'family' : 'serif',
        'size'   : 17}


# Ellipse
def plot_ellipse(mu, sigma_x, sigma_y, rho, edgecolor, facecolor, alpha, linestyle='none', zorder=0, linewidth=0.7):
    # Covariance matrix
    cov_matrix = np.array([[sigma_x**2, rho * sigma_x * sigma_y], [rho * sigma_x * sigma_y, sigma_y**2]])
    # Eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    # Sort the eigenvalues and eigenvectors
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    # Calculate the angle of rotation
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    # Create the ellipse
    ellipse = plt.matplotlib.patches.Ellipse(mu, 2*np.sqrt(eigvals[0]), 2*np.sqrt(eigvals[1]),
                                             angle=angle, edgecolor=edgecolor, facecolor=facecolor, alpha=alpha, linestyle=linestyle, zorder=zorder, linewidth=linewidth)
    # Add ellipse to plot
    ax.add_patch(ellipse)

    return ellipse

# def color_rect(Ah, Al, Gh, Gl, color_select):
#     rect_1 = pat.Rectangle((Gl[0] + Gl[2], Al[0] + Al[2]), Gl[1]-Gl[2], Ah[0]+Ah[1]-(Al[0]+Al[2]), color=color_select, alpha=alpha_select, linestyle='none')
#     rect_2 = pat.Rectangle((Gh[0]+Gh[2], Al[0] + Al[2]), Gh[1]-Gh[2], Ah[0]+Ah[1]-(Al[0]+Al[2]), color=color_select, alpha=alpha_select, linestyle='none')
#     rect_3 = pat.Rectangle((Gl[0]+Gl[1], Al[0] + Al[2]), Gh[0]+Gh[2]-(Gl[0]+Gl[1]), Al[1]-Al[2], color=color_select, alpha=alpha_select, linestyle='none')
#     rect_4 = pat.Rectangle((Gl[0]+Gl[1], Ah[0]+Ah[2]), (Gh[0]+Gh[2]-(Gl[0]+Gl[1])), Ah[1]-Ah[2], color=color_select, alpha=alpha_select, linestyle='none')

#     square = [(Gl[0], Al[0]), (Gh[0], Al[0]), (Gh[0], Ah[0]), (Gl[0], Ah[0])]
#     x_coords = [point[0] for point in square]
#     y_coords = [point[1] for point in square]
#     x_coords.append(x_coords[0])
#     y_coords.append(y_coords[0])

#     return [rect_1, rect_2, rect_3, rect_4, x_coords, y_coords]

def rect(Ah, Al, Gh, Gl):
    square = [(Gl, Al), (Gh, Al), (Gh, Ah), (Gl, Ah)]
    x_coords = [point[0] for point in square]
    y_coords = [point[1] for point in square]
    x_coords.append(x_coords[0])
    y_coords.append(y_coords[0])

    return [x_coords, y_coords]

def get_ellipse_vertices(ellipse, num_vertices=100):
    """
    Get the vertices of a matplotlib Ellipse.

    Parameters:
    - ellipse: matplotlib.patches.Ellipse instance.
    - num_vertices: Number of vertices to compute along the boundary of the ellipse.

    Returns:
    - vertices: A 2D numpy array of shape (num_vertices, 2) containing the x and y coordinates of the vertices.
    """
    angle = np.deg2rad(ellipse.angle)
    theta = np.linspace(0, 2 * np.pi, num_vertices)
    x = ellipse.width / 2 * np.cos(theta)
    y = ellipse.height / 2 * np.sin(theta)
   
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
   
    vertices = np.dot(np.column_stack((x, y)), R.T) + np.array([ellipse.center])
    return vertices


psrs_no_dm = ['J0030+0451', 'J1455-3330', 'J1738+0333', 'J2322+2057']
psrs_no_sn = ['J0751+1807', 'J1600-3053', 'J1024-0719', 'J1640+2224', 'J1730-2304', 'J1751-2857', 'J1801-1417', 'J1804-2717', 'J1843-1113', 'J1857+0943', 'J1910+1256', 'J1911+1347', 'J1918-0642', 'J2124-3358', 'J2322+2057']
psr_nogf_dm = ['J1909-3744', 'J1843-1113', 'J1804-2717', 'J1801-1417', 'J1713+0747']
psr_nogf_sn = ['J0900-3144', 'J1012+5307']

with open('/home/celestialsapien/image_content/pta_gwb_priors/sigma_one/epta_output.json', 'r') as file:
    data = json.load(file)


XD_arr = data['XD_arr'] 
YD_arr = data['YD_arr']
XR_arr = data['XR_arr']
YR_arr = data['YR_arr']
H_D_arr = data['H_D_arr']
H_R_arr = data['H_R_arr']
sigma_D_arr = data['sigma_D_arr']
sigma_R_arr = data['sigma_R_arr']
pulsar = data['pulsar']

XD_arr = [np.array(arr) for arr in data['XD_arr']]
YD_arr = [np.array(arr) for arr in data['YD_arr']]
XR_arr = [np.array(arr) for arr in data['XR_arr']]
YR_arr = [np.array(arr) for arr in data['YR_arr']]

H_D_arr = [np.array(arr) for arr in data['H_D_arr']]
H_R_arr = [np.array(arr) for arr in data['H_R_arr']]

# DM gaussian
for i in range(len(pulsar)):
    if pulsar[i] not in psr_nogf_dm:
        H_D_arr[i] = gaussian_filter(H_D_arr[i], 0.7)

    if pulsar[i]=='J1909-3744': # special filters for pulsars with large measurement uncertainty
        H_R_arr[i] = gaussian_filter(H_R_arr[i], 1.0)

    if pulsar[i]=='J1744-1134':
        H_R_arr[i] = gaussian_filter(H_R_arr[i], 1.37)

    if pulsar[i] not in psr_nogf_sn:
        H_R_arr[i] = gaussian_filter(H_R_arr[i], 0.7)

# for i in range(len(pulsar)):
#     fig = plt.figure(figsize=(20, 11))
    
#     axes1 = fig.add_subplot(121)
#     axes1.contour(XD_arr[i], YD_arr[i], H_D_arr[i].T, levels=[sigma_D_arr[i]], colors='black', linewidths=0.5)
#     axes1.set_title('DM')
#     if pulsar[i] in psrs_no_dm:
#         axes1.set_xlabel('....not is paper so excluded....')
#         axes1.set_ylabel('....not is paper so excluded....')
#     else:
#         axes1.set_xlabel('........')
#         axes1.set_ylabel('........')
        
#     axes1.tick_params(axis='y')
#     axes1.tick_params(axis='x')

#     axes2 = fig.add_subplot(122)
#     axes2.contour(XR_arr[i], YR_arr[i], H_R_arr[i].T, levels=[sigma_R_arr[i]], colors='black', linewidths=0.5)
#     axes2.set_title('SN')
#     if pulsar[i] in psrs_no_sn:
#         axes2.set_xlabel('....not is paper so excluded....')
#         axes2.set_ylabel('....not is paper so excluded....')
#     else:
#         axes2.set_xlabel('........')
#         axes2.set_ylabel('........')

#     axes2.tick_params(axis='y')
#     axes2.tick_params(axis='x')
#     plt.tight_layout()
#     plt.savefig(f'/home/celestialsapien/pta_gwb_priors/sigma_one/{pulsar[i]}.png', format = 'png', dpi=300)
#     plt.close()
#     print('done! for pulsar:', f'{i+1}.', pulsar[i])



# Selection
color_select_sn = '0.1'
color_select_dm= '0.1' 
arrow_color = '#000000'

color_map_sn = 'YlOrBr'
color_map_dm = 'YlGn'
alpha_select = 0.5



# dm noise
Ah_d = -12.61
Al_d = -13.83
Gh_d = 3.46
Gl_d = 0.60

fig = plt.figure()
ax = fig.add_subplot(111)
v_rect_out_dm = np.array([[0.60-0.28, -13.83-0.09], [0.60-0.28, -12.61+0.08], [3.46+0.31, -12.61+0.08], [3.46+0.31, -13.83-0.09]])
v_rect_in_dm = np.array([[0.60+0.23, -13.83+0.08], [0.60+0.23, -12.61-0.06], [3.46-0.28, -12.61-0.06], [3.46-0.28, -13.83+0.08]])

# We need to close the path, so add the first point of the inner rectangle at the end
v_rect_in_dm = np.vstack([v_rect_in_dm, v_rect_in_dm[0]])
v_rect_out_dm = np.vstack([v_rect_out_dm, v_rect_out_dm[0]])

# Create the path
vertices_dm = np.concatenate([v_rect_out_dm, v_rect_in_dm[::-1]])  # reverse the inner rectangle
codes_dm = np.full(len(vertices_dm), Path.LINETO, dtype=int)
codes_dm[0] = Path.MOVETO  # start outer rectangle
codes_dm[len(v_rect_out_dm)] = Path.MOVETO  # start inner rectangle
codes_dm[-1] = Path.CLOSEPOLY  # close the path

# Create the Path object
path_dm = Path(vertices_dm, codes_dm)

# Create the PathPatch
patch_dm = patches.PathPatch(path_dm, facecolor=color_select_dm, edgecolor='none', alpha=0.5)  # Set the color and transparency as needed

# Plotting
ax.add_patch(patch_dm)
for i in range (len(pulsar)):
    if pulsar[i] in psrs_no_dm:
        plt.contourf(XD_arr[i], YD_arr[i], np.array(H_D_arr[i]).T, levels=50, cmap=color_map_dm, alpha = alpha_select)
        # plt.scatter(XD_arr[i], YD_arr[i], np.array(H_D_arr[i]).T, alpha=0.1)
    else:    
        plt.contour(XD_arr[i], YD_arr[i], np.array(H_D_arr[i]).T, levels=[sigma_D_arr[i]], colors='black', linewidths=0.5)
# ax.set_xlabel('$\gamma_\mathrm{DM}$', fontdict=font)
# ax.set_ylabel('$\lg A_\mathrm{DM}$', fontdict=font)
# ax.tick_params(axis='y', labelsize=font['size'])
# ax.tick_params(axis='x', labelsize=font['size'])
plt.xlim(0, 5)
plt.ylim(-15, -12)
plt.plot(rect(Ah_d, Al_d, Gh_d, Gl_d)[0], rect(Ah_d, Al_d, Gh_d, Gl_d)[1], color = 'black', linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.savefig(f'/home/celestialsapien/image_content/pta_gwb_priors/sigma_one/logA_Gamma_DM_rect.png', format = 'png', dpi=1000)
plt.savefig(f'/home/celestialsapien/image_content/pta_gwb_priors/sigma_one/logA_Gamma_DM_rect.pdf', format = 'pdf')
plt.close()


# spin noise
Ah_s = -12.71
Al_s = -14.92
Gh_s = 2.88
Gl_s = 0.73


fig = plt.figure()
ax = fig.add_subplot(111)

v_rect_out_sn = np.array([[0.73-0.41, -14.92-0.66], [0.73-0.41, -12.71+0.13], [2.88+1.09, -12.71+0.13], [2.88+1.09, -14.92-0.66]])
v_rect_in_sn = np.array([[0.73+0.27, -14.91+0.43], [0.73+0.27, -12.71-0.13], [2.88-1.23, -12.71-0.13], [2.88-1.23, -14.91+0.43]])

# We need to close the path, so add the first point of the inner rectangle at the end
v_rect_in_sn = np.vstack([v_rect_in_sn, v_rect_in_sn[0]])
v_rect_out_sn = np.vstack([v_rect_out_sn, v_rect_out_sn[0]])

# Create the path
vertices_sn = np.concatenate([v_rect_out_sn, v_rect_in_sn[::-1]])  # reverse the inner rectangle
codes_sn = np.full(len(vertices_sn), Path.LINETO, dtype=int)
codes_sn[0] = Path.MOVETO  # start outer rectangle
codes_sn[len(v_rect_out_sn)] = Path.MOVETO  # start inner rectangle
codes_sn[-1] = Path.CLOSEPOLY  # close the path

# Create the Path object
path_sn = Path(vertices_sn, codes_sn)

# Create the PathPatch
patch_sn = patches.PathPatch(path_sn, facecolor=color_select_sn, edgecolor='none', alpha=0.5)  # Set the color and transparency as needed

# Plotting
ax.add_patch(patch_sn)

for i in range (len(pulsar)):
    if pulsar[i] in psrs_no_sn:
        plt.contourf(XR_arr[i], YR_arr[i], np.array(H_R_arr[i]).T, levels=50, cmap=color_map_sn, alpha = alpha_select)
        # plt.scatter(XR_arr[i], YR_arr[i], np.array(H_R_arr[i]).T, alpha=0.1)
    else:    
        plt.contour(XR_arr[i], YR_arr[i], np.array(H_R_arr[i]).T, levels=[sigma_R_arr[i]], colors='black', linewidths=0.5)
ax.set_xlabel('$\gamma_\mathrm{SN}$', fontdict=font)
ax.set_ylabel('$\lg A_\mathrm{SN}$', fontdict=font)

ax.tick_params(axis='y', labelsize=font['size'])
ax.tick_params(axis='x', labelsize=font['size'])
plt.xlim(0, 7)
plt.ylim(-16, -12)
plt.plot(rect(Ah_s, Al_s, Gh_s, Gl_s)[0], rect(Ah_s, Al_s, Gh_s, Gl_s)[1], color = 'black', linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.savefig(f'/home/celestialsapien/image_content/image_content/pta_gwb_priors/sigma_one/logA_Gamma_SN_rect.png', format = 'png', dpi=1000)
plt.savefig(f'/home/celestialsapien/image_content/pta_gwb_priors/sigma_one/logA_Gamma_SN_rect.pdf', format = 'pdf')
plt.close()

print('done plotting rectangle-collective plot')



# spin noise
mu_dev_s_sn = [2.17, -14.21]
mu_s_sn = [2.82, -13.88]
sigma_x_s_sn = 1.02
sigma_y_s_sn = 0.59
rho_s_sn = -0.99

mu_dev_l_sn = [3.43, -13.59]
mu_l_sn = [2.82, -13.88]
sigma_x_l_sn = 2.47
sigma_y_l_sn = 1.36
rho_l_sn = -0.88

mu_m_sn = [2.82, -13.88]
sigma_x_m_sn = 1.52
sigma_y_m_sn = 0.84
rho_m_sn = -0.96

# dm noise
mu_dev_s_dm = [1.63, -13.36]
mu_s_dm = [1.95, -13.27]
sigma_x_s_dm = 0.71
sigma_y_s_dm = 0.31
rho_s_dm = -0.50

mu_dev_l_dm = [2.20, -13.18]
mu_l_dm = [1.95, -13.27]
sigma_x_l_dm = 1.32
sigma_y_l_dm = 0.46
rho_l_dm = 0.08

mu_m_dm = [1.95, -13.27]
sigma_x_m_dm = 0.95
sigma_y_m_dm = 0.37
rho_m_dm = -0.21

# =======================================================================================================================================#
# Spin Noise
fig = plt.figure()
ax = fig.add_subplot(111)
el_out_sn = plot_ellipse(mu_l_sn, sigma_x_l_sn, sigma_y_l_sn, rho_l_sn, edgecolor='none', facecolor=color_select_sn, alpha= alpha_select)
el_in_sn = plot_ellipse(mu_s_sn, sigma_x_s_sn, sigma_y_s_sn, rho_s_sn, edgecolor='none', facecolor='white', alpha = 1)
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111)
v_el_out_sn = get_ellipse_vertices(el_out_sn)
v_el_in_sn = get_ellipse_vertices(el_in_sn)[::-1]
codes_sn=np.full(len(v_el_out_sn) + len(v_el_in_sn), Path.LINETO, dtype=int)
codes_sn[0] = Path.MOVETO  # Starting point for the outer ellipse
codes_sn[len(v_el_out_sn)] = Path.MOVETO  # Starting point for the inner ellipse
codes_sn[-1] = Path.CLOSEPOLY  # Close the path
# Combine vertices of both ellipses
vertices_sn = np.concatenate([v_el_out_sn, v_el_in_sn])
# Create the path object
path_sn = Path(vertices_sn, codes_sn)
# Create a patch from the path
patch_sn = patches.PathPatch(path_sn, facecolor=color_select_sn, edgecolor='none', alpha=alpha_select)
plot_ellipse(mu_m_sn, sigma_x_m_sn, sigma_y_m_sn, rho_m_sn, edgecolor='black', facecolor='none', alpha= 1, linestyle='--', zorder=2)



for i in range (len(pulsar)):
    if pulsar[i] in psrs_no_sn:
        contourf = plt.contourf(XR_arr[i], YR_arr[i], np.array(H_R_arr[i]).T, levels=50, cmap=color_map_sn, alpha = alpha_select)
    else:    
        plt.contour(XR_arr[i], YR_arr[i], np.array(H_R_arr[i]).T, levels=[sigma_R_arr[i]], colors='black', linewidths=0.5)
ax.add_patch(patch_sn)
ax.set_xlabel('$\gamma_\mathrm{SN}$', fontdict=font)
ax.set_ylabel('$\lg A_\mathrm{SN}$', fontdict=font)
ax.tick_params(axis='y', labelsize=font['size'])
ax.tick_params(axis='x', labelsize=font['size'])
plt.arrow(mu_m_sn[0], mu_m_sn[1], mu_dev_s_sn[0] - mu_m_sn[0], 0, color=arrow_color)
plt.arrow(mu_m_sn[0], mu_m_sn[1], mu_dev_l_sn[0] - mu_m_sn[0], 0, color=arrow_color)
plt.arrow(mu_m_sn[0], mu_m_sn[1], 0, mu_dev_s_sn[1] - mu_m_sn[1], color=arrow_color)
plt.arrow(mu_m_sn[0], mu_m_sn[1], 0, mu_dev_l_sn[1] - mu_m_sn[1], color=arrow_color)
plt.xlim(0, 7)
plt.ylim(-16, -12)
plt.tight_layout()
plt.savefig(f'/home/celestialsapien/image_content/pta_gwb_priors/sigma_one/logA_Gamma_SN_ellipse.png', format = 'png', dpi=1000)
plt.savefig(f'/home/celestialsapien/image_content/pta_gwb_priors/sigma_one/logA_Gamma_SN_ellipse.pdf', format = 'pdf')
plt.close()


# DM Noise
fig = plt.figure()
ax = fig.add_subplot(111)
el_out_dm = plot_ellipse(mu_l_dm, sigma_x_l_dm, sigma_y_l_dm, rho_l_dm, edgecolor='none', facecolor=color_select_dm, alpha= alpha_select)
el_in_dm = plot_ellipse(mu_s_dm, sigma_x_s_dm, sigma_y_s_dm, rho_s_dm, edgecolor='none', facecolor='white', alpha = 1)
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111)
v_rect_out_dm = get_ellipse_vertices(el_out_dm)
v_rect_in_dm = get_ellipse_vertices(el_in_dm)[::-1]
codes_dm=np.full(len(v_rect_out_dm) + len(v_rect_in_dm), Path.LINETO, dtype=int)

codes_dm[0] = Path.MOVETO  # Starting point for the outer ellipse
codes_dm[len(v_rect_out_dm)] = Path.MOVETO  # Starting point for the inner ellipse
codes_dm[-1] = Path.CLOSEPOLY  # Close the path

# Combine vertices of both ellipses
vertices_dm = np.concatenate([v_rect_out_dm, v_rect_in_dm])

# import ipdb; ipdb.set_trace()
# Create the path object
path_dm = Path(vertices_dm, codes_dm)

# Create a patch from the path
patch_dm = patches.PathPatch(path_dm, facecolor=color_select_dm, edgecolor='none', alpha=alpha_select)


plot_ellipse(mu_m_dm, sigma_x_m_dm, sigma_y_m_dm, rho_m_dm, edgecolor='black', facecolor='none', alpha= 1, linestyle='--', zorder=2)



for i in range (len(pulsar)):
    if pulsar[i] in psrs_no_dm:
        contourf = plt.contourf(XD_arr[i], YD_arr[i], np.array(H_D_arr[i]).T, levels=50, cmap=color_map_dm, alpha = alpha_select)
    else:    
        plt.contour(XD_arr[i], YD_arr[i], np.array(H_D_arr[i]).T, levels=[sigma_D_arr[i]], colors='black', linewidths=0.5)
ax.add_patch(patch_dm)
ax.set_xlabel('$\gamma_\mathrm{DM}$', fontdict=font)
ax.set_ylabel('$\lg A_\mathrm{DM}$', fontdict=font)
ax.tick_params(axis='y', labelsize=font['size'])
ax.tick_params(axis='x', labelsize=font['size'])
plt.arrow(mu_m_dm[0], mu_m_dm[1], mu_dev_s_dm[0] - mu_m_dm[0], 0, color=arrow_color)
plt.arrow(mu_m_dm[0], mu_m_dm[1], mu_dev_l_dm[0] - mu_m_dm[0], 0, color=arrow_color)
plt.arrow(mu_m_dm[0], mu_m_dm[1], 0, mu_dev_s_dm[1] - mu_m_dm[1], color=arrow_color)
plt.arrow(mu_m_dm[0], mu_m_dm[1], 0, mu_dev_l_dm[1] - mu_m_dm[1], color=arrow_color)
plt.xlim(0, 5)
plt.ylim(-15, -12)
plt.tight_layout()
plt.savefig(f'/home/celestialsapien/image_content/pta_gwb_priors/sigma_one/logA_Gamma_DM_ellipse.png', format = 'png', dpi=1000)
plt.savefig(f'/home/celestialsapien/image_content/pta_gwb_priors/sigma_one/logA_Gamma_DM_ellipse.pdf', format = 'pdf')
plt.close()
print('done plotting ellipse-collective plot')