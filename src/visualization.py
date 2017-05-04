'''
updated by Dan on 12/15
'''

import matplotlib.pyplot as plt
import numpy as np


def IMshow_pupil(pupil, axnum = True):
    '''
    display a pupil function in 2D
    '''
    NY, NX = pupil.shape
    ry = int(NY/2.)
    rx = int(NX/2.)
    # print(rx, ry)
    yy = (np.arange(NY)-ry)/ry
    xx = (np.arange(NX)-rx)/rx
    [MX, MY] = np.meshgrid(xx,yy)
    fig = plt.figure(figsize=(5.0,4.0))
    ax = fig.add_subplot(111)

    ax.set_ylim([-1, 1])
    ax.set_xlim([-1, 1])
    if (axnum == False):
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        # fig.axes.get_yaxis().set_visible(False)
    pcm = ax.pcolor(MX, MY, pupil, cmap = 'RdYlBu_r')
    fig.colorbar(pcm, ax = ax, extend='max')

    return fig  # return the figure handle


def zern_display(z_coeffs, z0 = 4, ylim = None):
    '''
    zern_display, bar plot
    z_coeffs: the coefficients
    z0: the starting mode (index 4 means 5th zernike mode)
    '''
    zrange = z0 + np.arange(len(z_coeffs[z0:]))+1
    fig = plt.figure(figsize=(6.0, 3.0))
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
