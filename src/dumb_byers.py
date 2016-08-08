import os
import glob
import numpy as np
import matplotlib.pyplot as plt
 
 
def dumb_byers(path): 
 
    psf_list = glob.glob(path+"*.npy")
    psf_list.sort(key = os.path.getmtime) 
    ii = 0
    session_list = []
    for psf_name in psf_list:
        session_name = os.path.split(psf_name)[-1][:-4]+'pupil'
        session_list.append(session_name)
        print(psf_name)
        psf = np.load(psf_name)
        Retrieve = PSF_PF(psf)
    
        PF = unwrap_phase(Retrieve.retrievePF().phase)
        Strehl[ii] = Retrieve.Strehl_ratio()
        
        print(Strehl[ii])
        PF_core = PF[64:192, 64:192]
        dia_PF = 47
        pattern = PF[128-dia_PF:128+dia_PF, 128-dia_PF:128+dia_PF]
        plt.figure(figsize=(5,4))
        im = plt.imshow(pattern, cmap = 'RdBu')
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
        
        
        DM = DM_simulate(12, 256, pattern)
        segs = DM.findSeg()
        im = plt.imshow(segs, cmap = 'RdBu', interpolation='none')
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
        plt.savefig(path+session_name + '_seg')
        
        
        
        
        plt.clf()
        
        plt.figure(figsize = (5,3.5))
        
        sk = (np.arange(128)-64)/47.
        ax=plt.gca()
        plt.plot(sk,PF_core[:,64], '-b', linewidth = 2)
        plt.plot(sk,PF_core[64,:], '-g', linewidth = 2)
        plt.xlabel('k')
        ax.set_ylim([-1.3,1.5])
        ax.set_yticks([-0.5, 0, 0.5, 1.0])
        
        plt.savefig(path+session_name + '_cs')
        plt.clf()
        plt.close()
        
        
        
        
        
#         RMSE[ii] = np.sqrt(np.var(PF_core))
        ii +=1

    plt.figure(figsize=(7,4))
    lp = len(psf_list)
    plt.plot(np.arange(lp-1),Strehl[:-1], 'b-x', linewidth=2)
    ax = plt.gca()
    ax.set_xticks(np.arange(1,lp-1,3))
    ax.set_xticklabels(session_list[1:lp-1:3], rotation = 20)
    plt.savefig(path+'Strehl')
    plt.show()