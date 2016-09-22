"""
Created by Dan Xie on 07/15/2016
Last edit: 09/19/2016 
Class PSF_PF retrieves a the pupil plane from a given PSF measurement
"""

# This should be shaped into an independent module 
# Phase retrieval 

import numpy as np
import libtim.zern
import matplotlib.pyplot as plt
import pupil2device as pupil
from numpy.lib.scimath import sqrt as _msqrt
from skimage.restoration import unwrap_phase
from psf_tools import psf_zplane
from microscope import objective
# a small zernike function


class phase_retrieval(objective):
    """
    retrieve phase from arbitrary image pattern
    """

    def __init__(self, NA, fl, rind, dr):
        """
        phase retrieval 
        """
        self.NA = NA 
        self.fl = fl
        self.n_refrac = rind
        self.dr = dr
        objective.__init__(self, NA, rind, fl)
        
        
        
        
    def retrievePF(self, pattern, nIt = 10):
        """
        Retrieve phase from arbitrary pattern.
        1. load the pattern, recognize dimensions 
        2. Have an initial guess, usually plane wave 
        3.
        """
        
        ny, nx = pattern.shape
        PF = pupil.Simulation(ny, nx, self.dr, self.wl, self.n_refrac, self.NA, self.fl, wavelengths= 1)# initialize a pupil simulation class 
        PF.psf2pf(pattern, zs, mu, A, nIterations, use_pyfftw, resetAmp, symmeterize)
        
        




class PSF_PF(object):
    def __init__(self, pattern, dx=0.097, dz=0.30, ld=0.525, nrefrac=1.33, NA=1.0, fl=9000, nIt=10):
        
        self.dx = dx
        self.dz = dz
        self.l =ld   
        self.n = nrefrac
        self.NA = NA
        self.f = fl
        self.nIt = nIt
        self.pattern = pattern
        self.nz, self.ny, self.nx = self.PSF.shape
        
        print(self.nz, self.ny, self.nx)
        self.PF=pupil.Simulation(self.nx,dx,ld,nrefrac,NA,fl,wavelengths=1) # initialize the pupil function

        
    
    def retrievePF(self, bscale = 1.00, psf_diam = 50, resample = None):
        # an ultrasimplified version
        # comment on 08/12: I am still not convinced of the way of setting background. 
        
        
        z_offset, zz = psf_zplane(self.PSF, self.dz, self.l/3.2) # This should be the reason!!!! >_<
        A = self.PF.plane
        
#         z_offset = -z_offset # To deliberately add something wrong
        
        Mx, My = np.meshgrid(np.arange(self.nx)-self.nx/2., np.arange(self.nx)-self.nx/2.)
        r_pxl = _msqrt(Mx**2 + My**2)
        
        bk_inner = 50
        bk_outer = 61
        
        hcyl = np.array(self.nz*[np.logical_and(r_pxl>=bk_inner, r_pxl<bk_outer)])
        background = np.mean(self.PSF[hcyl])*bscale
        print( "   background = ", background)
        print( "   z_offset = ", z_offset)
        
        if(resample is None):
            PSF_sample = self.PSF
            zs = zz-z_offset
        else:
            PSF_sample = self.PSF[::resample]
            zs = zz[::resample]-z_offset
        
        complex_PF = self.PF.psf2pf(PSF_sample, zs, background, A, self.nIt)
    
    
    def Strehl_ratio(self):
        # this is very raw. Should save the indices for pixels inside the pupil. 
        c_up = (self.pf_ampli.sum())**2
        c_down = (self.pf_ampli**2).sum()*(len(np.where(self.pf_ampli!=0)[0]))
        
        strehl = c_up/c_down
        return strehl
    
    
    def zernike_fitting(self, z_max = 22, head_remove = True):
        """
        Fit self.pf_phase to zernike modes 
        z_max: maximum order of Zernike modes that should be fitted 
        head_remove: remove the first 1 --- 4 order modes, by default true.
        To be fitted later. 
        """ 
    
    
    #-------------------------Visualization part of PSF-PF ------------------------------- 
    
    def pupil_display(self, cross = False):
        # this function plots pupil plane phase in the unit of wavelength(divided by 2pi)
        k_pxl = self.PF.k_pxl+1 # leave some space for the edge
        
        if (cross == False): # display the plane
            phase_block = self.pf_phase[self.ny/2-k_pxl:self.ny/2+k_pxl, self.nx/2-k_pxl:self.nx/2+k_pxl]/(2.*np.pi)
            fig = plt.figure(figsize=(5.5,4.5))
            im = plt.imshow(phase_block, cmap = 'RdBu', extent=(-3,3,3,-3))
            plt.colorbar(im)  
            plt.tick_params(
                axis = 'both',
                which = 'both', 
                bottom = 'off',
                top = 'off',
                right = 'off',
                left = 'off',
                labelleft='off',
                labelbottom = 'off')
        else: # display the cross sections
            fig = plt.figure(figsize = (6.5, 4.0))
            ax = fig.add_subplot(1,1,1)
            pf_crx = self.pf_phase[self.ny/2, self.nx/2-k_pxl:self.nx/2+k_pxl]/(2.*np.pi)
            pf_cry = self.pf_phase[self.ny/2-k_pxl:self.ny/2+k_pxl, self.nx/2]/(2.*np.pi)
            
            # define the plot range
#             lim_up = np.max(np.max(pf_crx), np.max(pf_cry)) 
#             lim_down = np.min(np.min(pf_crx), np.min(pf_cry))
            
            k_coord = (np.arange(-k_pxl, k_pxl).astype('float64')+0.5)/(k_pxl-1.)
            ax.plot(k_coord, pf_crx, '-r', linewidth = 2, label = 'X')
            ax.plot(k_coord, pf_cry, '-g', linewidth = 2, label = 'Y')
            ax.set_xlabel('k')
            ax.set_ylim([-0.5,0.5])
            ax.set_xlim([-1.1,1.1])
            ax.set_xticks([-1.0, 0, 1.0])
            
        plt.tight_layout()
        return fig
        # done with pupil_display
    


