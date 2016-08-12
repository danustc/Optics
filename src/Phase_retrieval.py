"""
Created by Dan Xie on 07/15/2016
Last edit: 08/11/2016
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

# a small zernike function

class PSF_PF(object):
    def __init__(self, PSF, dx=0.097, dz=0.30, ld=0.525, nrefrac=1.33, NA=1.0, fl=9000, nIt=5):
        
        self.dx = dx
        self.dz = dz
        self.l =ld   
        self.n = nrefrac
        self.NA = NA
        self.f = fl
        self.nIt = nIt
        self.PSF = PSF
        self.nz, self.ny, self.nx = self.PSF.shape
        
        print(self.nz, self.ny, self.nx)
        self.PF=pupil.Simulation(self.nx,dx,ld,nrefrac,NA,fl,wavelengths=1) # initialize the pupil function

        
    
    def retrievePF(self, bscale = 0.98, psf_diam = 50, resample = None):
        # an ultrasimplified version
        
        z_offset, zz = psf_zplane(self.PSF, self.dz, self.l/3.2) # This should be the reason!!!! >_<
        A = self.PF.plane
        
        Mx, My = np.meshgrid(np.arange(self.nx)-self.nx/2., np.arange(self.nx)-self.nx/2.)
        r_pxl = _msqrt(Mx**2 + My**2)
        
        bk_inner = psf_diam*1.20
        bk_outer = psf_diam*1.50
        
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
    
          
        Pupil_final = _PupilFunction(complex_PF, self.PF)
        self.pf_complex = Pupil_final.complex
        self.pf_phase = unwrap_phase(Pupil_final.phase)
        self.pf_ampli = Pupil_final.amplitude
        return Pupil_final
        
    
    def Strehl_ratio(self):
        # this is very raw. Should save the indices for pixels inside the pupil. 
        c_up = (self.pf_ampli.sum())**2
        c_down = (self.pf_ampli**2).sum()*(len(np.where(self.pf_ampli>0)[0]))
        
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
            fig = plt.figure(figsize=(6.5,5.6))
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
            fig = plt.figure(figsize = 6.5, 4.0)
            ax = fig.add_subplot(1,1,1)
            pf_crx = self.pf_phase[self.ny/2, self.nx/2-k_pxl:self.nx/2+k_pxl]/(2.*np.pi)
            pf_cry = self.pf_phase[self.ny/2-k_pxl:self.ny/2+k_pxl, self.nx/2]/(2.*np.pi)
            
            # define the plot range
            lim_up = np.max(np.max(pf_crx), np.max(pf_cry)) 
            lim_down = np.min(np.min(pf_crx), np.min(pf_cry))
            
            k_coord = float(np.arange(-k_pxl, k_pxl)+0.5)/(k_pxl-1)
            ax.plot(k_coord, pf_crx, '-r', linewidth = 2)
            ax.plot(k_coord, pf_cry, '-g', linewidth = 2)
            ax.set_xlabel('k')
            ax.set_ylim([lim_down,lim_up])
            ax.set_xticks([-1.0, 0, 1.0])
            
    
        return fig
        # done with pupil_display
    




class _PupilFunction(object):
    '''
    A pupil function that keeps track when when either complex or amplitude/phase
    representation is changed.
    '''
    def __init__(self, cmplx, geometry):
        self.complex = cmplx
        self._geometry = geometry

    @property
    def complex(self):
        return self._complex

    @complex.setter
    def complex(self, new):
        self._complex = new
        self._amplitude = abs(new)
        self._phase = np.angle(new)

    @property
    def amplitude(self):
        return self._amplitude

    @amplitude.setter
    def amplitude(self, new):
        self._amplitude = new
        self._complex = new * np.exp(1j*self._phase)

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, new):
        self._phase = new
        self._complex = self._amplitude * np.exp(1j*new)

    @property
    def zernike_coefficients(self):
        return self._zernike_coefficients

    @zernike_coefficients.setter
    def zernike_coefficients(self, new):
        self._zernike_coefficients = new
        #self._zernike = zernike.basic_set(new, self._geometry.r, self._geometry.theta)
        self._zernike = libtim.zern.calc_zernike(new, self._geometry.nx/2.0)

    @property
    def zernike(self):
        return self._zernike
