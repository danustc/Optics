"""
Created by Dan Xie on 07/15/2016
Last edit: 08/11/2016
"""

# This should be shaped into an independent module 
# Phase retrieval 


import numpy as np
import libtim.zern
import matplotlib.pyplot as plt
import pupil2device as pupil
from numpy.lib.scimath import sqrt as _msqrt
from scipy.ndimage import interpolation
from skimage.restoration import unwrap_phase
from psf_tools import psf_zplane

# a small zernike function
 
class Zernike_func(object):
    def __init__(self,radius, mask=False):
        self.radius = radius
        self.useMask = mask
        self.pattern = []

        
    
    def single_zern(self, mode, amp):
        modes = np.zeros((mode))
        modes[mode-1] = amp
        self.pattern= libtim.zern.calc_zernike(modes, self.radius, mask = self.useMask, zern_data= {})
        
        return self.pattern
    
    
    def multi_zern(self, amps):
        self.pattern = libtim.zern.calc_zernike(amps, self.radius, mask = self.useMask, zern_data = {})
        return self.pattern
#         RMSE[ii] = np.sqrt(np.var(PF_core))
#         ii +=1

    def plot_zern(self):
        plt.imshow(self.pattern, interpolation='none')
        plt.show()
    

# ---------------Below is a simulation of deformable mirror

class DM_simulate(object):
    
    def __init__(self, nseg = 12, nPixels = 256, pattern=None):
        self.nSegments = nseg
        self.nPixels = nPixels
        self.DMsegs = np.zeros((self.nSegments, self.nSegments))
        self.zern = Zernike_func(nPixels/2)
        self.borders = np.linspace(0,self.nPixels,num=self.nSegments+1).astype(int)
        
        
        if pattern is None:
            self.pattern = np.zeros((nPixels,nPixels))
        else: 
            zoom = 256./np.float(pattern.shape[0])
            MOD = interpolation.zoom(pattern,zoom,order=0,mode='nearest')
            self.pattern = MOD
            

    def findSeg(self):
        for ii in np.arange(self.nSegments):
            for jj in np.arange(self.nSegments):
                xStart = self.borders[ii]
                xEnd = self.borders[ii+1]
                yStart = self.borders[jj]
                yEnd = self.borders[jj+1]
                
                av = np.mean(self.pattern[xStart:xEnd, yStart:yEnd])
                self.DMsegs[ii,jj] = av
                
        DMsegs = np.copy(self.DMsegs)
        return DMsegs
        


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
    
    def pupil_display(self):
        # this function plots pupil plane phase in the unit of wavelength(divided by 2pi)
        k_pxl = self.PF.k_pxl+1 # leave some space for the edge
        phase_block = self.pf_phase[self.ny/2-k_pxl:self.ny/2+k_pxl, self.nx/2-k_pxl:self.nx/2+k_pxl]/(2.*np.pi)
        fig = plt.figure(figsize=(6.5,6))
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
    
        return fig
    

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
