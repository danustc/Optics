'''
updated by Dan on 12/15
'''

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def IMshow_pupil(pupil_raw, axnum = True, c_scale = None, crop = None, inner_diam= None):
    '''
    display a pupil function in 2D
    '''
    pupil = np.copy(pupil_raw)
    NY, NX = pupil.shape
    ry = int(NY/2.)
    rx = int(NX/2.)
    # print(rx, ry)
    yy = (np.arange(NY)-ry)/ry
    xx = (np.arange(NX)-rx)/rx
    fig = plt.figure(figsize=(3.0,3.5))
    ax = fig.add_subplot(111)
    [MX, MY] = np.meshgrid(xx,yy)
    if crop is None:
        ax.set_ylim([-1, 1])
        ax.set_xlim([-1, 1])
    else:
        ax.set_ylim([-crop, crop])
        ax.set_xlim([-crop, crop])
        cropped = np.sqrt(MX**2+MY**2)> crop
        pupil[cropped] = 0
    if (axnum == False):
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        # fig.axes.get_yaxis().set_visible(False)
    if(c_scale is None):
        pcm = ax.pcolor(MX, MY, pupil, cmap = 'RdYlBu_r')
        cbar = fig.colorbar(pcm, ax = ax, extend = 'max' , orientation= 'horizontal')
        cbar.ax.tick_params(labelsize = 12)
    else:
        pcm = ax.pcolor(MX, MY, pupil, cmap = 'RdYlBu_r',vmin = c_scale[0], vmax = c_scale[1])
        cbar = fig.colorbar(pcm, ax = ax, extend = 'max' , orientation = 'horizontal', pad = 0.05)
        cbar.ax.tick_params(labelsize = 12)

    if(inner_diam is not None):
        cmark = plt.Circle((0,0), inner_diam, color= 'w', ls = '--', linewidth = 2, fill = False)
        ax.add_artist(cmark)
    plt.tight_layout()
    return fig  # return the figure handle



def zern_display(z_coeffs, z0 = 4, ylim = None):
    '''
    zern_display, bar plot
    z_coeffs: the coefficients
    z0: the starting mode (index 4 means 5th zernike mode)
    '''
    zrange = z0 + np.arange(len(z_coeffs[z0:]))+1
    fig = plt.figure(figsize=(6.0, 2.5))
    ax = fig.add_subplot(111)
    ax.bar(zrange, z_coeffs[z0:], width = 0.8, color = 'g')
    ax.set_xlim([z0-0.5, len(z_coeffs)+0.5])
    ax.set_xticks(zrange[::3])
    if ylim is None:
        ax.set_ylim([-1.5, 1.5])
    else:
        ax.set_ylim(ylim)
    ax.set_xlabel('Zernike modes', fontsize = 14)
    ax.set_ylabel('Amplitude/$\lambda$', fontsize = 14)
    fig.tight_layout()
    return fig



def psf_plane(psf_frame, aov = 's', dz=0.500, dr=0.102 ):
    '''
    display psf on a plane
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if aov == 's':
        nz, nr=psf_frame.shape
        range_z = dz*nz
        range_r = dr*nr
        ax.imshow(np.log(psf_frame), cmap = 'Greys_r', extent = [-range_r*0.5, range_r*0.5, -range_z*0.5, range_z*0.5])

    else:
        ny, nx = psf_frame.shape
        range_y = dr*ny
        range_x = dr*nx
        ax.imshow(np.log(psf_frame), cmap = 'Greys_r', extent = [-range_x*0.5, range_x*0.5, -range_y*0.5, range_y*0.5])

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()

    return fig



def pupil_showcross(pf):
    # display the pupil function from top and cross section.
    NY, NX = pf.shape
    ry = int(NY/2.)
    rx = int(NX/2.)
    yy = (np.arange(NY)-ry)/ry
    xx = (np.arange(NX)-rx)/rx
    [MX, MY] = np.meshgrid(xx,yy)
    figp = plt.figure(figsize = (7.9,3.2))
    grid_size = (1,5)
    plt.subplot2grid(grid_size,(0,0), rowspan = 1, colspan = 2)
    plt.subplot2grid(grid_size,(0,2), rowspan = 1, colspan = 3)
    ax1, ax2 = figp.axes
    ax1.pcolor(MX,MY,pf,cmap = 'RdYlBu_r')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    ax2.plot(xx,pf[ry, :], '-r', linewidth = 2, label = 'kx')
    ax2.plot(yy,pf[:,rx], '-g', linewidth = 2, label = 'ky')
    ax2.set_xlabel('k', fontsize = 14)
    ax2.set_ylabel('Wavelength', fontsize = 14)
    ax2.set_ylim([-1,1])
    ax2.legend(['x', 'y'], fontsize=12)
    ax2.set_xticks([-1,-0.5,0,0.5,1] )

    plt.tight_layout()
    return figp
