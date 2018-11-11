import sys
import os
import scipy.io as sio
import glob
import numpy as np
import matplotlib.pyplot as plt
import libtim.zern as zern
from visualization import  zern_display, IMshow_pupil
from skimage.restoration import unwrap_phase
from psf_tools import *
from Phase_retrieval import PSF_PF
import pandas as pd

def RMSE(pf, mask = 50):
    ny, nx = pf.shape
    MY, MX = np.meshgrid(np.arange(ny), np.arange(nx))
    MR = np.sqrt(MY**2 + MX **2)
    inner = MR<=mask
    ns = len(pf[inner])
    print("inner pixels:", ns)
    rmse = np.sqrt(np.sum((pf[inner]-np.mean(pf[inner]))**2)/ns)
    return rmse


def compare_wl(plist_a, plist_b):
    '''
    compare the pupil function retrieved for plist a and plist b.
    '''
    diff_wl = dict()
    for k, v in plist_a.items():
        pf_a = v
        pf_b = plist_b[k]
        print("Iteration:", k)
        dpf = pf_a-pf_b
        diff_wl[k] = dpf
    return diff_wl


def group_retrieval(path_root, fname_flag, lstp, ext = 'npy'):
    '''
    path_root: the path to the directory containing the data files
    fname_flag: any string that you want to append to the file name while saving figures.
    '''

    psf_list = glob.glob(path_root+fname_flag+  '*.'+ext) # search all the .mat files and make the paths into a list
    psf_list.sort(key = os.path.getmtime)
    print(path_root)
    print(psf_list)
    Strehl = []
    pupil_list = {}

    for fname in psf_list:
        session_name = os.path.split(fname)[-1][:-4] # show the psf filename
        print("psf:", session_name)
        psf_name = session_name
        psf_stack = np.load(fname)

        pr = PSF_PF(psf_stack, dx=0.10200, dz=0.5, ld=0.515, lstep = lstp, nrefrac=1.33, NA=1.00, fl=9000, nIt=11) # phase retrieval core function. 
        k_max = pr.PF.k_pxl # the radius of pupil in the unit of pixel (k-space)
        print("k_max:", k_max)
        pr.retrievePF(bscale = 1.00, psf_diam = 50) # psf_diam should be less than half of your array size 
        pupil_final = pr.get_phase() # grab the phase from the pr class
        Strehl.append(pr.strehl_ratio())
        pf_crop = pupil_final[128-k_max:128+k_max, 128-k_max:128+k_max]
        pupil_list[psf_name] = pf_crop# save the pupil inside the dictionary

    return Strehl, pupil_list






if __name__ == '__main__':

    psf_raw = np.load('psf.npy')
    print(psf_raw.shape)
    psf_stack = psf_raw.astype('float64') + np.random.randn(120,90,90)**2*1.0e-06
    psf_trunc = psf_stack[20:100]
    pr = PSF_PF(psf_trunc, dx=0.17500, dz=0.02, ld=0.700, lstep = [3,0.005], nrefrac=1.54, NA=1.45, fl=2000, nIt=25) # phase retrieval core function. 
    k_max = pr.PF.k_pxl # the radius of pupil in the unit of pixel (k-space)
    print("k_max:", k_max)
    pr.retrievePF(bscale = 1.00, psf_diam = 30) # psf_diam should be less than half of your array size 
    pupil_final = pr.get_phase() # grab the phase from the pr class
    fig = IMshow_pupil(pupil_final)
    fig.savefig('pupil_25.png')
    Strehl= pr.strehl_ratio()
    pf_crop = pupil_final[128-k_max:128+k_max, 128-k_max:128+k_max]
    #comp_flags = ['w1', 'w3_d05', 'w6_d02', 'w11_d01']
    ##comp_flags = ['w1', 'w3_d05', 'w3_d02', 'w3_d01']
    #L = np.zeros([4,2])
    #L[:,0] = np.array([1,3,6,11])
    #L[:,1] = np.array([0.000, 0.005, 0.005, 0.005])
    #Strehl= np.zeros([14, 4])
    #df_ = pd.DataFrame(index = np.arange(14), columns = comp_flags )
    #df_ = df_.fillna(0.0)
    #for ii in range(4):
    #    Strehl[:,ii], pupil_list = group_retrieval(path_root, lstp = L[ii], fname_flag = 'T0_mod')
    #    np.savez(path_root+'plist_'+comp_flags[ii], **pupil_list)
    #    for k, v in pupil_list.items():
    #        print('Key:', k)
    #        rmse = RMSE(v)
    #        try:
    #            idx = int(k.split('_')[-1][3:])
    #            print(idx)
    #            ckey = comp_flags[ii]
    #            df_[ckey][idx] = rmse
    #        except ValueError:
    #            print("Not a number!")
    #            continue
    #print(df_)
    #print(Strehl)
    #df_.to_csv(path_root+'pupil_nw.csv', sep = '\t')
    #np.save(path_root+'Strehl_nw', Strehl)
    #Strehl_nw = np.load(path_root+'Strehl_nw.npy')
    #Strehl_ds = np.load(path_root+'Strehl_steps.npy')

    #fig_nw = plt.figure(figsize=(6,3.5))
    #ax = fig_nw.add_subplot(111)
    #ax.plot(Strehl_nw[:-1], linewidth = 2)
    #ax.legend(['n = 1', 'n = 3', 'n = 6', 'n = 11'])
    #ax.set_xlabel('Iterations', fontsize = 14)
    #ax.set_ylabel('Strehl ratio', fontsize = 14)
    #plt.tight_layout()
    #fig_nw.savefig(path_root+'Strehl_nw')

    #fig_ds= plt.figure(figsize=(6,3.5))
    #ax = fig_ds.add_subplot(111)
    #ax.plot(Strehl_ds[:-1], linewidth = 2)
    #ax.set_xlabel('Iterations', fontsize = 14)
    #ax.set_ylabel('Strehl ratio', fontsize = 14)
    #ax.legend(['single', 'd = 5', 'd = 2', 'd = 1'])
    #ax.set_xlabel('Iterations', fontsize = 14)
    #plt.tight_layout()
    #fig_ds.savefig(path_root+'Strehl_ds')


    for ds in ds_flags:
        plw = np.load(path_root+'plist_'+ds+'.npz')
        for k,v in plw.items():
            fig = IMshow_pupil(v)
            fig.savefig(path_root + ds + k)
            plt.close()


    for k,v in pl_w1.items():
        fig = IMshow_pupil(v)
        fig.savefig(path_root + k)
        plt.close()
