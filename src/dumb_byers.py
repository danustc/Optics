import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase
from psf_tools import *
from Phase_retrieval import PSF_PF

 
def dumb_byers(): 

#     path = '/home/sillycat/Documents/Light_sheet/Data/Sep07/' 
    path = 'D:\Data\Dan\Sep22\\'
    psf_list = glob.glob(path+"T33_mod*.npy")
    pupil_list = glob.glob(path+'Pupil/*phase.npy')
    psf_list.sort(key = os.path.getmtime) 
    pupil_list.sort(key = os.path.getmtime)
    zfit_list = glob.glob(path+'Pupil/*zfit.npy')
    
    Strehl = np.zeros(len(psf_list))
    ii = 0
    
    for fname in psf_list:
        print(ii)
        session_name = os.path.split(fname)[-1][:-4]
        print(session_name)
#         psf_name = session_name + '_psf'
        psf_stack = np.load(fname)
        
        PR = PSF_PF(psf_stack, dx=0.097, dz=0.30, ld=0.525, nrefrac=1.33, NA=1.0, fl=9000, nIt=10)
        PR.retrievePF(bscale = 1.00, psf_diam = 60, resample = False)
        Strehl[ii] = PR.Strehl_ratio()
        ii+=1
        
        
    fig = plt.figure(figsize = (8,5))
    ax = fig.add_subplot(1,1,1)
    ax.plot(Strehl, '-x', linewidth = 2)
    ax.set_xlabel('Iterations', fontsize = 14)
    ax.set_ylabel('Strehl ratio', fontsize = 14)
    fig.savefig(path+ 'Strehl')
    
    plt.show()
#         psf_yz = psf_slice(psf_stack, dim_slice=1, trunc = 40)
#         fig = plt.figure()
#         ax = fig.add_subplot(1,1,1)
#         ax.imshow(np.log(psf_yz), cmap = 'Greys_r')
#         ax.set_title(psf_name)
#         plt.tight_layout()
#         plt.axis('off')
#         fig.savefig(path+psf_name+'_yz')
# #         fig_v.savefpig(pupil_name+'_cr')
#         
#     pr = 50 
#         
#     for pname in pupil_list:
#         session_name = os.path.split(pname)[-1][:-4]
#         pupil_name = session_name + '_pupil'
#         pupil = unwrap_phase(np.load(pname))
#         py, px = pupil.shape
#         py/=2 
#         px/=2 
#         
#         kk = (np.arange(-pr,pr)+0.5)/47
#         
#     
#     
#         fig = plt.figure()
#         ax1 = fig.add_subplot(1,2,1)
#         ax1.imshow(pupil[py-pr:py+pr, px-pr:px+pr], cmap = 'RdBu_r')
#         plt.axis('off')
#         ax2 = fig.add_subplot(1,2,2)
#         ax2.plot(kk,pupil[py, px-pr:px+pr], '-r', linewidth = 2, label = 'ky')
#         ax2.plot(kk,pupil[py-pr:py+pr, px], '-g', linewidth = 2, label = 'kx')
#         plt.tight_layout()
#         fig.savefig(path+pupil_name)
#         plt.clf()
#         
#     plt.close('all')
#     
#     for zname in zfit_list:
#         zfit = np.load(zname)
#         ny, nx = zfit.shape
#         
#         ry = ny/2 
#         rx = nx/2
#         
#         my = np.arange(-ry, ry)+0.5
#         mx = np.arange(-rx, rx)+0.5
#         
#         [MY,MX] = np.meshgrid(my, mx)
#         
#         mask = (MY**2 + MX**2)> ry**2
#         zfit[mask] = 0
#         
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         ax.imshow(zfit, cmap = 'RdBu_r')
#         plt.axis('off')
#         plt.tight_layout()
#         plt.savefig(zname[:-4])
#         plt.clf()
#         
#         plt.close()
        
    
        
    
if __name__ == '__main__':
    dumb_byers()