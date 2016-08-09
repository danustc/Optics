# A dumb test of    everything stupid and uncertain.
# Later this will be shaped into a good simulator of optics system. 

import os
from Phase_retrieval import PSF_PF, DM_simulate
from skimage.restoration import unwrap_phase
import glob
import numpy as np
from matplotlib import pyplot as plt
import tifffunc
from psf_tools import psf_recenter


path = '/home/sillycat/Programming/Python/Optics/bin2tiff/'


def main():
    tflag = 'psfStack'
    psf_raw = tifffunc.read_tiff_singles(path, tflag, spl = 1).astype('float64')
    psf = psf_recenter(psf_raw, r_mask = 150, cy_ext = 1.5)
    
    n_trunc = 15
    psf_mid = psf[n_trunc:-n_trunc]
    
    tifffunc.write_tiff(psf.astype('uint16'), 'psf_stack')
    
    
    Retrieve = PSF_PF(psf_mid, dx=0.078, dz=0.10, ld=0.530, nrefrac=1.515, NA=1.42, fl=3000, nIt=10)
#     
#     
#     
    PF = unwrap_phase(Retrieve.retrievePF(bscale=1.00, psf_diam= 150, resample = 8).phase)
#     Strehl = Retrieve.Strehl_ratio()
#     dia_PF = 47
#     pattern = PF[128-dia_PF:128+dia_PF, 128-dia_PF:128+dia_PF]
    plt.figure(figsize=(5,4))
    im = plt.imshow(PF, cmap = 'RdBu')
#     plt.tick_params(
#             axis = 'both',
#             which = 'both',
#             bottom = 'off',
#             top = 'off',
#             right = 'off',
#             left = 'off',
#             labelleft='off',
#             labelbottom = 'off')
    plt.colorbar(im)
    plt.show()
#     print('Strehl ratio: ',Strehl)
    
    
if __name__ == "__main__":
    main()    