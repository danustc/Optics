'''
 Last update by Dan Xie, 04/10/2017
 This file contains all the psf tools: reading, plotting

'''

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy.ndimage import gaussian_filter as gf

def gaussian(z, a, z0, w, b):
    # w = 2 \sigma ^2
    # FWHM: 2.355 \sigma
    return a * np.exp(-(z-z0)**2/w) + b


def cylinder_cutter(dims,c_offset,rad1, rad2 = None):
    """
    return a cylinder-shaped 3D array
    This is kinda useless, but let's still keep it here.
    """
    nz = dims[0]
    ny = dims[1]
    nx = dims[2]

    cy = c_offset[0]
    cx = c_offset[1]

    MX, MY = np.meshgrid(np.arange(nx), np.arange(ny))
    rad = np.sqrt((MX-cx)**2 + (MY-cy)**2) # radius of t
    if(rad2 is None):
        cyl = np.array(nz*[rad<rad1])
    else:
        cyl = np.array(nz*[np.logical_and(rad> rad1, rad<rad2)])
    return cyl

    # done with cylinder

def psf_recenter(stack, r_mask = 40, cy_ext = 1.5):
    '''
    find the center of the psf and
    stack: the raw_psf
    r_mask: the radius of the mask size
    cy_ext: how far the background should extend to the outside
    '''
    nz, ny, nx = stack.shape
    cy, cx = np.unravel_index(np.argmax(gf(stack,2)), (nz,ny,nx))[1:]

    ny_shift = int(ny/2 - cy)
    nx_shift = int(nx/2 - cx)
    PSF = np.roll(stack, ny_shift, axis = 1)
    PSF = np.roll(PSF, nx_shift, axis = 2)

    return PSF
            # Background estimation


def psf_slice(stack, dim_slice = 0, n_slice = None, trunc = 50):
    """
    take out a slice from a stack and return it.
    """

    hy, hx = stack.shape[1:]
    hy/=2
    hx/=2
    if n_slice is None:
        n_slice = stack.shape[dim_slice]/2 # take from the middle

    if (dim_slice == 0):
        # take an xy slice on
        pslice = stack[n_slice,hy-trunc:hy+trunc, hx-trunc:hx+trunc]
    elif(dim_slice == 1):
        # take an xz slice
        pslice = stack[:, n_slice, hx-trunc:hx+trunc]
    else:
        # take an yz slice
        pslice = stack[:,hy-trunc:hy+trunc,n_slice]

    return pslice
    # end of psf_slice




def psf_zplane(stack, dz, w0, de = 1):
    '''
    determine the position of the real focal plane.
    Don't mistake with psf_slice!
    '''
    nz, ny, nx = stack.shape
    cy, cx = np.unravel_index(np.argmax(gf(stack,2)), (nz,ny,nx))[1:]

    zrange = (nz-1)*dz*0.5
    zz = np.linspace(-zrange, zrange, nz)
    center_z = stack[:,cy-de:cy+de+1,cx-de:cx+de+1]
    im_z = center_z.mean(axis=2).mean(axis=1)

    b = np.mean((im_z[0],im_z[-1]))
    a = im_z.max() - b

    p0 = (a,0,w0,b)
    popt = optimize.curve_fit(gaussian, zz, im_z, p0)[0]
    z_offset = popt[1] # The original version is wrong
    return z_offset, zz


def psf_lineplot(stack, cut_range = 2, z_step = 0.3, r_step=0.097):
    """
    cut_range: where to cut off
    axis:    0 --- z-direction
            1 --- y-direction
            2 --- x-direction
    z_step: step in z-direction
    r_step: step in x and y direction
    plot all the three directions
    and fit to Gaussian to give the FWHM
    """
    figv = plt.figure(figsize=(6,4))
    ax = figv.add_subplot(1,1,1)
    ax.set_xlim([-cut_range,cut_range])
    nz, ny, nx = stack.shape
    cz, cy, cx = np.unravel_index(np.argmax(stack), (nz,ny,nx))

    FWHM = np.zeros(3)
    # plot along z-direction
    psf_z = stack[:, cy, cx]
    coord_z = (np.arange(nz)-nz*0.5)*z_step
    b = np.mean((psf_z[0],psf_z[-1]))
    a = psf_z.max() - b
    w0 = 3.00
    pz_0 = (a,0,w0,b)
    popt = optimize.curve_fit(gaussian, coord_z, psf_z, pz_0)[0]
    FWHM[0] = np.sqrt(popt[2]*0.5)* 2.355
    ax.plot(coord_z-popt[1], psf_z, '-ob', linewidth = 2, label = 'z')


    # plot along y-direction
    psf_y = stack[cz,:, cx]
    coord_y = (np.arange(ny)-ny*0.5)*r_step
    b = np.mean((psf_y[0],psf_y[-1]))
    a = psf_y.max() - b
    w0 = 0.50
    py_0 = (a,0,w0,b)
    popt = optimize.curve_fit(gaussian, coord_y, psf_y, py_0)[0]
    FWHM[1] = np.sqrt(popt[2]*0.5)* 2.355
    ax.plot(coord_y-popt[1], psf_y, '->g', linewidth = 2, label = 'y')


    # plot along x-direction
    psf_x = stack[cz, cy, :]
    coord_x = (np.arange(nx)-nx*0.5)*r_step
    b = np.mean((psf_x[0],psf_x[-1]))
    a = psf_x.max() - b
    w0 = 0.50
    px_0 = (a,0,w0,b)
    popt = optimize.curve_fit(gaussian, coord_x, psf_x, px_0)[0]
    FWHM[2] = np.sqrt(popt[2]*0.5)* 2.355
    ax.plot(coord_x-popt[1], psf_x, '-xr', linewidth = 2, label = 'x')

    ax.legend(['z', 'y', 'x'])
    ax.set_xlabel('distance (micron)')



    plt.tight_layout()
    return figv, FWHM
    # done with psf_lineplot


def psf_planeplot(stack, plane = 0, c_pxl = None, argmt = None):
    """
    select one or more planes to display
    argmt: arrangement of multiple plots
    """
    side_0 = 6.0
    nz, ny, nx = stack.shape
    cz, cy, cx = np.unravel_index(np.argmax(stack), (nz,ny,nx))
    centers = [cz, cy, cx]



    if c_pxl is None:
        # plot where the maximum is
        c_pxl = centers[plane]

    if(np.isscalar(c_pxl) == True):
        # If we only plot one frame
        pslice = psf_slice(stack, plane, c_pxl)
        py, px = pslice.shape
        figp = plt.figure(figsize = (side_0,side_0*py/px))
        ax = figp.add_subplot(1,1,1)
        ax.imshow(pslice, cmap = 'Greys_r', interpolation = 'none')
        ax.tick_params(
            axis = 'both',
            which = 'both',
            bottom = 'off',
            top = 'off',
            right = 'off',
            left = 'off',
            labelleft='off',
            labelbottom = 'off')
        # All the units are in pixels, no microns involved
    else:
        # plot multiple frames in one figure
        # if the length of c_pxl is smaller than the arrangement, then stop at c_pxl;
        # otherwise stop when the arrangement is full.
        n_stop = np.min(len(c_pxl), np.prod(argmt))
        ii = 1
        figp = plt.figure(figsize = (side_0, side_0*argmt[0]/argmt[1])) # scale
        for n_slice in c_pxl[:n_stop]:
            # plot one by one
            pslice = psf_slice(stack, plane, n_slice)
            ax = figp.add_subplot(argmt[0], argmt[1], ii)
            ax.imshow(pslice, cmap = 'Greys_r', interpolation = 'none')
            ax.tick_params(
                axis = 'both',
                which = 'both',
                bottom = 'off',
                top = 'off',
                right = 'off',
                left = 'off',
                labelleft='off',
                labelbottom = 'off')


    plt.tight_layout()
    return figp
    # end of psf_planeplot
