'''
 Last update by Dan Xie, 08/08/2016
 A systematic simulation of optical systems, based on POPPY package.
 This file contains all the psf tools 

'''

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.ndimage import gaussian_filter as gf


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
    cz, cx, cy = np.unravel_index(np.argmax(gf(stack*r_mask,2)), (nz,ny,nx))
    print( "Center found at: ", cz,cx,cy)
            # We laterally center the stack at the brightest pixel
    cut = stack[:,cx-r_mask:cx+r_mask,cy-r_mask:cy+r_mask]
    PSF = np.zeros((nz,nx,ny))
    PSF[:,nx/2-r_mask:nx/2+r_mask,ny/2-r_mask:ny/2+r_mask] = cut
    
    c_cyl = cylinder_cutter([nz, ny, nx], [cy, cx], r_mask)
    b_cyl = cylinder_cutter([nz, ny, nx], [cy, cx], r_mask, r_mask*cy_ext)
    background = np.mean(stack[b_cyl])
    
    PSF[np.logical_not(c_cyl)] = background
    return PSF
            # Background estimation
