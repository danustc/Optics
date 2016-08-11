'''
 Last update by Dan Xie, 08/10/2016
 A systematic simulation of optical systems, based on POPPY package.
 This file contains all the psf tools 

'''

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy.ndimage import gaussian_filter as gf

def gaussian(z, a, z0, w, b):
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


def psf_zplane(stack, dz, w0, de = 1):
    '''
    determine the position of the real focal plane.
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


def psf_visualize(stack, cut_range = 2, axis = 0, z_step = 0.3, r_step=0.097, c_pxl = None):
    """
    cut_range: where to cut off 
    axis:    0 --- z-direction
            1 --- y-direction
            2 --- x-direction
    z_step: step in z-direction
    r_step: step in x and y direction 
    """
    figv = plt.figure(figsize=(6,4))
    ax = figv.add_subplot(1,1,1)
    nz, ny, nx = stack.shape 
    cz, cy, cx = np.unravel_index(np.argmax(stack), (nz,ny,nx))
    dims = [nz, ny, nx]
    centers = [cz, cy, cx]
    steps = [z_step, r_step, r_step]
    
    
    if (c_pxl is None):
        c_pxl = dims[axis]/2
    else:
        c_pxl = centers[axis]
    
    n_cut = int(cut_range/steps[axis])
    
    psf_line = stack[]
    
    
    
    return figv