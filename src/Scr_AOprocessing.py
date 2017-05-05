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


def pupil_crop(pupil, mask = True, diag = 51):
    '''
    crop pupil function to the size.
    assume that the pupil function is centered.
    '''
    ny, nx = pupil.shape
    hy, nx = int(ny/2), int(nx/2)



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
        figv, fwhm = psf_lineplot(psf_stack)
        figv.savefig(folder + session_name+'_line')
        FWHM.append(fwhm)

        psf_yz = psf_slice(psf_stack, dim_slice = 2, trunc = 40)
        range_z = dz*psf_yz.shape[0]
        range_r = dr*psf_yz.shape[1]


        fig_yz = psf_plane(psf_yz,aov = 's', dz = dz, dr=0.102)
        fig_yz.savefig(folder+psf_name+'_yz')
        plt.close()

    return FWHM


def load_folder_pupil(folder, flag = 'mod', k_px = 51):
    # load and plot all the pupil functions in the chosen folder.
    pupil_list = glob.glob(folder + '*'+ flag+'*.npy')
    pupil_list.sort(key = os.path.getmtime)
    for pname in pupil_list:
        session_name = os.path.basename(pname).split('.')[0]
        pupil_name = session_name + '_pupil'
        pupil = unwrap_phase(np.load(pname))/(2*np.pi) # convert to wavelength
        figp = pupil_showcross(pupil)
        figp.savefig(folder+pupil_name)
        plt.close()

#----------------------The main function-------------


def main():
    date_folder = 'May02_2017/T0_second_round/'
    #load_folder_psf(path_root+date_folder)
    load_folder_pupil(path_root+date_folder+'zfit/')


if __name__ == '__main__':
    main()
