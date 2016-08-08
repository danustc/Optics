# A dumb test of    everything stupid and uncertain.
# Later this will be shaped into a good simulator of optics system. 

import os
from Phase_retrieval import PSF_PF, DM_simulate
from skimage.restoration import unwrap_phase
import glob
import numpy as np
from matplotlib import pyplot as plt
import tifffunc

path = '/home/sillycat/Programming/Python/Optics/bin2tiff/'


def main():
    tflag = 'psfStack'
    psf = tifffunc.read_tiff_singles(path, tflag, spl = 2).astype('float64')
    
    
    
#     tifffunc.write_tiff(psf.astype('int64'), 'psf_stack')
    
    
    Retrieve = PSF_PF(psf, dx=0.078, dz=0.20, ld=0.530, nrefrac=1.33, NA=1.42, fl=3000, nIt=8)
#     
#     
#     
    PF = unwrap_phase(Retrieve.retrievePF(bscale=5.0).phase)
#     Strehl = Retrieve.Strehl_ratio()
#     dia_PF = 47
#     pattern = PF[128-dia_PF:128+dia_PF, 128-dia_PF:128+dia_PF]
#     plt.figure(figsize=(5,4))
#     im = plt.imshow(pattern, cmap = 'RdBu')
#     plt.tick_params(
#             axis = 'both',
#             which = 'both',
#             bottom = 'off',
#             top = 'off',
#             right = 'off',
#             left = 'off',
#             labelleft='off',
#             labelbottom = 'off')
#     plt.colorbar(im)
#     plt.show()
#     print('Strehl ratio: ',Strehl)
    
    
if __name__ == "__main__":
    main()    