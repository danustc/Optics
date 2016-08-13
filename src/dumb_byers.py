import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from Phase_retrieval import PSF_PF
from skimage.restoration import unwrap_phase
from DM_simulate import DM_simulate
from psf_tools import *
 
def dumb_byers(): 

    path = '/home/sillycat/Documents/Light_sheet/Data/Aug09/' 
    psf_list = glob.glob(path+"T0_mod*.npy")
    psf_list.sort(key = os.path.getmtime) 
    ii = 0
    session_list = []
    
    DS = DM_simulate()
    seg1 = DS.zernSeg(6)
    plt.imshow(DS.pattern)
    plt.tick_params(
        axis = 'both',
        which = 'both', 
        bottom = 'off',
        top = 'off',
        right = 'off',
        left = 'off',
        labelleft='off',
        labelbottom = 'off')
    plt.tight_layout()
    
    plt.savefig('zern6')

    
    seg4 = DS.zernSeg(11)
    plt.imshow(DS.pattern)
    plt.tick_params(
        axis = 'both',
        which = 'both', 
        bottom = 'off',
        top = 'off',
        right = 'off',
        left = 'off',
        labelleft='off',
        labelbottom = 'off')
    plt.tight_layout()
    
    plt.savefig('zern11')

    
    
    
    
    
    for fname in psf_list:
        session_name = os.path.split(fname)[-1][:-4]
        psf_name = session_name + 'psf'
        pupil_name = session_name + 'pupil'
        session_list.append(session_name)
        psf_stack = np.load(fname)
        
        Retrieve = PSF_PF(psf_stack)
        
        fig, FHWM = psf_lineplot(psf_stack, cut_range = 2)
        print(session_name)
        print(FHWM)
        
        pupil = Retrieve.retrievePF()
        fig_amp = plt.figure(figsize = (4.5,4))
        ax = fig_amp.add_subplot(111) 
        ax.imshow(Retrieve.pf_ampli[128-50:128+50, 128-50:128+50])
        fig_amp.savefig(pupil_name+'amp')   
#         fig_p = Retrieve.pupil_display(cross = False)
#         fig_v = Retrieve.pupil_display(cross = True)
        
#         fig_p.savefig(pupil_name+'_pl')
#         fig_v.savefig(pupil_name+'_cr')
          
        plt.close('all')
        
        
    
        
    
if __name__ == '__main__':
    dumb_byers()