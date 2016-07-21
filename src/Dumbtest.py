# A dumb test of everything stupid and uncertain.
# Later this will be shaped into a good simulator of optics system. 

import os
from Phase_retrieval import PSF_PF
from skimage.restoration import unwrap_phase
import glob
import numpy as np
from matplotlib import pyplot as plt

path = '/home/sillycat/Documents/Light_sheet/Data/Jul14/'
# path2 = '/home/sillycat/Documents/Light_sheet/Data/May17/'

# PSF1 = np.load(path1+'T0_modnone.npy')
# PSF2 = np.load(path2+ 'psfm01.npy')

def main():
    psf_list = glob.glob(path+"T*mod*.npy")
    psf_list.sort(key = os.path.getmtime) 
    RMSE = np.zeros(len(psf_list))
    ii = 0
    session_list = []
    for psf_name in psf_list:
        session_name = os.path.split(psf_name)[-1][:-4]
        session_list.append(session_name)
        print(psf_name)
        psf = np.load(psf_name)
        Retrieve = PSF_PF(psf)
    
        PF = unwrap_phase(Retrieve.retrievePF())
        PF_core = PF[64:192, 64:192]
        plt.figure(figsize=(5,4))
        im = plt.imshow(PF_core, cmap = 'RdBu')
        plt.tick_params(
                        axis = 'both',
                        which = 'both',
                        bottom = 'off',
                        top = 'off',
                        right = 'off',
                        left = 'off',
                        labelleft='off',
                        labelbottom = 'off')
        plt.colorbar(im)
        plt.savefig(path+session_name)
        plt.clf()
        
        plt.figure(figsize = (5,3.5))
        
        sk = (np.arange(128)-64)/47.
        ax=plt.gca()
        plt.plot(sk,PF_core[:,64], '-b', linewidth = 2)
        plt.plot(sk,PF_core[64,:], '-g', linewidth = 2)
        plt.xlabel('k')
        ax.set_ylim([-1.0,1.2])
        ax.set_yticks([-0.5, 0, 0.5, 1.0])
        
        plt.savefig(path+session_name + '_cs')
        plt.clf()
        plt.close()
        
        
        RMSE[ii] = np.sqrt(np.var(PF_core))
        ii +=1

    plt.figure(figsize=(7,4))
    lp = len(psf_list)
    plt.plot(np.arange(lp-1),RMSE[:-1], 'b-x', linewidth=2)
    ax = plt.gca()
    ax.set_xticks(np.arange(1,lp-1,3))
    ax.set_xticklabels(session_list[1:lp-1:3], rotation = 20)
    plt.savefig(path+'RMSE')
    plt.show()
    
if __name__ == "__main__":
    main()    