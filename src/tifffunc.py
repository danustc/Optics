"""
Edited on 08/07/2016 by Dan. 
A wrapper designed for Dan's image processing. 
Based on Christoph Gohlke (UCI)'s tifffile module.
"""

from tifffile import TiffFile
from tifffile import imsave
import numpy as np
import glob
import os 

# read a tiff stack
def read_tiff_singles(dph, tflags, spl = 1, srt = 'name'):
    # read a series of single tiffs and output a stack
   
    if (srt =='name'):
        tif_list = sorted(glob.glob(dph+'*'+tflags+'*.tif'))
    elif(srt == 'time'):
        tif_list = glob.glob(dph+'*'+tflags+'*.tif')
        tif_list.sort(key = os.path.getmtime)
    print(tif_list)
    
    n_tif = len(tif_list)
    print(n_tif)
    with TiffFile(tif_list[0]) as tif:
        im0 = tif.asarray()
        ny, nx = im0.shape
        # from here, assume that all the figures with the same tflags share 
    nz = int(n_tif/spl)
    istack = np.zeros([nz, ny, nx])
    istack[0] = np.copy(im0)
    
    
    for ii in np.arange(1, nz):
        ia = ii*spl
        fname = tif_list[ia]
        with TiffFile(fname) as tif:
            im0 = tif.asarray()
            istack[ii] = np.copy(im0)

    return istack




def read_tiff(fname):
    # the fname should include the absolute path.
    with TiffFile(fname + '.tif') as tif:
        istack = tif.asarray()
    return istack

def intp_tiff(istack, ns1, ns2, nint = 1):
    # linear interpolation of slices between
    int_stack = np.zeros(shape = (nint,)+ istack.shape[1:]) 
    for ii in np.arange(nint + 2):
        alpha = ii/(nint + 1.)
        int_stack[ii] = istack[ns1]*(1-alpha) + istack[ns2]*alpha
    return int_stack.astype('uint16')  # return as unint16, tiff

def write_tiff(imstack, fname):
    imsave(fname+'.tif', imstack)
    
