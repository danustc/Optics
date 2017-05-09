# processing AO-SPIM data.
# some simple functions.
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase
from psf_tools import *
from visualization import *
from DM_simulate import DM_simulate

path_root = '/home/sillycat/Documents/Light_sheet/Data/'


def pupil_crop(raw_pupil, mask = True, rad = 49):
    '''
    crop pupil function to the size.
    assume that the pupil function is centered.
    '''
    ny, nx = raw_pupil.shape
    hy, hx = int(ny/2), int(nx/2)
    pupil = np.copy(raw_pupil)
    # crop in x,y
    if hy > rad:
        pupil = pupil[hy-rad:hy+rad,:]
        yy = np.arange(-rad,rad)+0.5
    else:
        yy = np.arange(ny)-hy+0.5
    if hx > rad:
        pupil = pupil[:,hx-rad:hx+rad]
        xx = np.arange(-rad,rad)+0.5
    else:
        xx = np.arange(nx)-hx+0.5

    if mask:
        [MY,MX] = np.meshgrid(yy,xx)
        out_pupil= (MX**2+MY**2)>rad**2
        pupil[out_pupil] = 0.0

    return pupil



def load_folder_psf(folder, flag = 'mod', dz = 0.50, dr = 0.102, verbose = False):
    psf_list = glob.glob(folder+ '*'+flag+'*.npy')
    psf_list.sort(key = os.path.getmtime)
    n_psf = len(psf_list)
    FWHM = []

    for fname in psf_list:
        session_name = os.path.basename(fname)[:-4]
        if verbose:
            print(session_name)
        psf_name = session_name+'_psf'
        psf_stack = np.load(fname)
        # the line plot section
        figv, fwhm = psf_lineplot(psf_stack, z_step = dz)
        figv.savefig(folder + session_name+'_line')
        FWHM.append(fwhm)

        psf_yz = psf_slice(psf_stack, dim_slice = 2, trunc = 40)
        range_z = dz*psf_yz.shape[0]
        range_r = dr*psf_yz.shape[1]


        fig_yz = psf_plane(psf_yz,aov = 's', dz = dz, dr=0.102)
        fig_yz.savefig(folder+psf_name+'_yz')
        plt.close()

    return FWHM


def load_folder_pupil(folder, flag = 'mod', radius = 49 ):
    # load and plot all the pupil functions in the chosen folder.
    pupil_list = glob.glob(folder + '*'+ flag+'*.npy')
    pupil_list.sort(key = os.path.getmtime)
    for pname in pupil_list:
        session_name = os.path.basename(pname).split('.')[0]
        pupil_name = session_name + '_pupil'
        print(session_name)
        raw_pupil = unwrap_phase(np.load(pname))/(2*np.pi) # convert to wavelength
        pupil = pupil_crop(raw_pupil,mask = True, rad = radius)
        figp = pupil_showcross(pupil)
        figp.savefig(folder+pupil_name)
        plt.close()


def load_fwhm(folder, flag = 'fwhm'):
    FWHM_list = glob.glob(folder+flag+'*.npy')
    n_fwhm = len(FWHM_list)
    fig = plt.figure(figsize=(6,3.5))
    ax = fig.add_subplot(111)
    for fname in FWHM_list:
        fwhm = np.load(fname)
        fl = fwhm.shape[0]
        fbest = fwhm.min(axis = 0)
        fbest_str = [float('%.2f'% fn) for fn in fbest]
        session_name = os.path.basename(fname)[:-4]
        ax.plot(fwhm,linewidth =2, label = ['z', 'y', 'x'])
        ax.legend(['z', 'y', 'x'], fontsize = 14)
        ax.set_xlabel('Iterations', fontsize = 14)
        ax.set_ylabel('FWHM', fontsize = 14)
        ax.annotate("Optimal:"+str(fbest_str), xytext = (fl*0.2,2.0 ), xy = (fl*0.8, fbest[0]), fontsize = 14)
        plt.tight_layout()
        fig.savefig(folder+session_name)
        plt.cla()

#----------------------The main function-------------


def main():
    date_folder = 'May03_2017/'#T0_second_round/'
    FWHM = load_folder_psf(path_root+date_folder)
    print(FWHM)
    #load_fwhm(path_root+date_folder, flag = 'FWHM')
    #psf_list = glob.glob(path_root+date_folder+'*mod*.npy')
    #psf_list.sort(key = os.path.getmtime)
    #psf_best = np.load(psf_list[-1])
    #nz, ny, nx = psf_best.shape
    #hy = int(ny/2)
    #hx = int(nx/2)
    #trunc = 50
    #ax.imshow(psf_best[zp, hy-trunc:hy+trunc, hx-trunc:hx+trunc], cmap = 'Greys_r', interpolation = 'none')

    #np.save(path_root+date_folder+ 'FWHM_rd3', np.array(FWHM))
    #load_folder_pupil(path_root+date_folder+'phase/', radius = 49)
    #load_folder_pupil(path_root+date_folder+'zfit/', radius = 256)







if __name__ == '__main__':
    main()
