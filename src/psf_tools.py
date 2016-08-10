'''
 Last update by Dan Xie, 08/08/2016
 A systematic simulation of optical systems, based on POPPY package.
 This file contains all the psf tools 

'''

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.ndimage import gaussian_filter as gf

def gaussian(z, a, z0, w, b):
    return a * np.exp(-(z-z0)**2/w) + b


def cylinder_cutter(dims,c_offset,rad1, rad2 = None):
    """
    return a cylinder-shaped 3D array 
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


def psf_zplane(stack, dz):
    '''
    determine the position of the real focal plane.
    '''
    nz, ny, nx = stack.shape 
    cy, cx = np.unravel_index(np.argmax(gf(stack,2)), (nz,ny,nx))[1:]
    
    zrange = (nz-1)*dz*0.5
    zz = np.linspace(-zrange, zrange, nz)
    im_z = 
    
    
    b = np.mean((im_z[0],im_z[-1]))
            a = im_z.max() - b
            w = l/3.2
            p0 = (a,0,w,b)
            
            # Fit gaussian to axial intensity trace
            popt, pcov = optimize.curve_fit(gaussian, z, im_z, p0)
            # Where we think the emitter is axially located:
    z_offset = popt[1] # The original version is wrong
    
    