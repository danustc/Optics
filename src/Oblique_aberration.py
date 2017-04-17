import sys
import os
import scipy.io as sio
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase
from psf_tools import *
from Phase_retrieval import PSF_PF


def load_mat(mat_path, fl_convert = True):
    '''
    load matlab data file (*.mat).
    '''
    if os.path.exists(mat_path):
    try:
        im_stack = sio.loadmat(mat_path)
    except IOError:
        print("Could not read the file:", mat_path)
        sys.exit()



def dumb_byers(date_folder, file_flag):

    path_root = '/home/sillycat/Documents/Light_sheet/Data/'
    path = path_root + date_folder + '/'
    psf_list = glob.glob(path+file_flag+"*.npy")
    pupil_list = glob.glob(path+'pupil/*'+file_flag+'*phase.npy')
    zfit_list = glob.glob(path+'pupil/*'+file_flag+'*zfit.npy')
    psf_list.sort(key = os.path.getmtime)
    pupil_list.sort(key = os.path.getmtime)
    Strehl = np.zeros(len(psf_list))
    ii = 0
    FWHM = np.zeros([len(psf_list), 3])

#
    for fname in psf_list:
        print(ii)
        session_name = os.path.split(fname)[-1][:-4]
        print(session_name)
        psf_name = session_name + '_psf'
        psf_stack = np.load(fname)
        figv, FWHM[ii] = psf_lineplot(psf_stack)
        figv.savefig(path+session_name+'_line')

        # pr = psf_pf(psf_stack, dx=0.0975, dz=0.30, ld=0.525, nrefrac=1.33, na=1.0, fl=9000, nit=15)
        # pr.retrievepf(bscale = 1.00, psf_diam = 60, resample = false)
        # strehl[ii] = pr.strehl_ratio()
        psf_yz = psf_slice(psf_stack, dim_slice=2, trunc = 40)
        range_z = 0.3*len(psf_yz)
        range_r = psf_yz.shape[1]*0.097
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.imshow(np.log(psf_yz), cmap = 'Greys_r', extent = [-range_r*0.5, range_r*0.5, -range_z*0.5, range_z*0.5])
        ax.set_title(psf_name)
        plt.tight_layout()
        # plt.axis('off')
        fig.savefig(path+psf_name+'_yz')
        ii+=1
#

    pr = 50
    fig = plt.figure(figsize = (8,4.5))

    for pname in pupil_list:
        session_name = os.path.split(pname)[-1][:-4]
        pupil_name = session_name + '_pupil'
        pupil = unwrap_phase(np.load(pname))*0.550/(2*np.pi) # unit in microns
        py, px = pupil.shape
        py/=2
        px/=2

        kk = (np.arange(-pr,pr)+0.5)/47



        ax1 = fig.add_subplot(1,2,1)
        ax1.imshow(pupil[py-pr:py+pr, px-pr:px+pr], cmap = 'RdBu_r')
        plt.axis('off')
        ax2 = fig.add_subplot(1,2,2)
        ax2.plot(kk,pupil[py, px-pr:px+pr], '-r', linewidth = 2, label = 'ky')
        ax2.plot(kk,pupil[py-pr:py+pr, px], '-g', linewidth = 2, label = 'kx')
        ax2.set_xlabel('k')
        ax2.set_ylabel('microns')
        plt.tight_layout()
        fig.savefig(path+pupil_name)
        plt.cla()

    plt.close('all')
#
    fig = plt.figure(figsize = (8,4.5))
    for zname in zfit_list:
        zfit = np.load(zname)*0.515/(2*np.pi)
        ny, nx = zfit.shape

        ry = ny/2
        rx = nx/2

        my = np.arange(-ry, ry)+0.5
        mx = np.arange(-rx, rx)+0.5

        [MY,MX] = np.meshgrid(my, mx)

        mask = (MY**2 + MX**2)> ry**2
        zfit[mask] = 0


        ax1 = fig.add_subplot(1,2,1)
        ax1.imshow(zfit, cmap = 'RdBu_r')
        plt.axis('off')
        ax2 = fig.add_subplot(1,2,2)
        ax2.plot(mx/rx,zfit[ry,:], linewidth = 2, label = 'ky')
        ax2.plot(my/ry,zfit[:, rx], '-g', linewidth = 2, label = 'kx')
        ax2.set_xlabel('k')
        ax2.set_ylabel('microns')
        ax2.legend(['ky', 'kx'])
        plt.tight_layout()
        fig.savefig(zname[:-4])
        plt.cla()

#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        ax.imshow(zfit, cmap = 'RdBu_r')
#        plt.axis('off')
#        plt.tight_layout()
#        plt.savefig(zname[:-4])
#        plt.clf()

#        plt.close()




if __name__ == '__main__':
    date_folder = 'Mar09_2017'
    file_flag = 'T0'

    dumb_byers(date_folder, file_flag)
    sharpness = glob.glob('/home/sillycat/Documents/Light_sheet/Data/'+date_folder+'/zmet_*round*.npy')
    print(sharpness)
    fig = plt.figure(figsize=(6,3.5))
    ax = fig.add_subplot(1,1,1)
    for sharpfile in sharpness:
        tr_sharp=np.load(sharpfile)
        ax.plot(np.arange(13),tr_sharp[:,0], '-xr', linewidth = 2,label = 'z6')
        ax.plot(np.arange(13),tr_sharp[:,1], '-og', linewidth = 2, label = 'z11')
        ax.plot(np.arange(13),tr_sharp[:,2], '->b', linewidth = 2, label = 'z22')
        ax.legend(['z6', 'z11', 'z22'])
        ax.set_xlabel('iterations', fontsize = 14)
        ax.set_ylabel('sharpness', fontsize = 14)
        fig.savefig(sharpfile[:-4])
        plt.cla()
