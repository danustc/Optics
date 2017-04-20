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
import pupil
from numpy.lib.scimath import sqrt as _msqrt
from skimage.restoration import unwrap_phase
from psf_tools import psf_zplane

# a small zernike function

class PSF_PF(object):
    def __init__(self, PSF, dx, dz, ld, nrefrac=1.33, NA=1.0, fl=9000, nIt=10):
        self.dx = dx
        self.dz = dz
        self.l =ld  #wavelength 
        self.n = nrefrac
        self.NA = NA
        self.f = fl
        self.nIt = nIt
        self.PSF = PSF
        self.nz, self.ny, self.nx = self.PSF.shape
        print(self.nz, self.ny, self.nx)
        self.PF=pupil.Simulation(self.nx,dx,ld,nrefrac,NA,fl,wavelengths=1) # initialize the pupil function
        in_pupil = self.PF.k < self.PF.k_max
        self.NK = in_pupil.sum()
        print("# pixels inside the pupil:", self.NK)

    def retrievePF(self, bscale = 1.00, psf_diam = 50, resample = None):
        # an ultrasimplified version
        # comment on 08/12: I am still not convinced of the way of setting background. 


        z_offset, zz = psf_zplane(self.PSF, self.dz, self.l/3.2) # This should be the reason!!!! >_<
        A = self.PF.plane
#         z_offset = -z_offset # To deliberately add something wrong
        Mx, My = np.meshgrid(np.arange(self.nx)-self.nx/2., np.arange(self.nx)-self.nx/2.)
        r_pxl = _msqrt(Mx**2 + My**2)
        bk_inner = 16
        bk_outer = 20
        hcyl = np.array(self.nz*[np.logical_and(r_pxl>=bk_inner, r_pxl<bk_outer)])
        background = np.mean(self.PSF[hcyl])*bscale
        print( "   background = ", background)
        print( "   z_offset = ", z_offset)
        if(resample == False):
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


    def get_phase(self):
        '''
        return the (unwrapped pupil phase)
        '''
        return self.pf_phase


    def strehl_ratio(self):
        # this is very raw. Should save the indices for pixels inside the pupil. 
        c_up = np.abs(self.pf_complex.sum())**2
        c_down = (self.pf_ampli**2).sum()*self.NK
        strehl = c_up/c_down
        return strehl



class _PupilFunction(object):
    '''
    a pupil function that keeps track when when either complex or amplitude/phase
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
