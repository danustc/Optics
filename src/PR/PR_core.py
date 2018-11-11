"""
Created by Dan Xie on 07/15/2016
Last edit: 05/11/2017
Class PSF_PF retrieves a the pupil plane from a given PSF measurement
Need to use @setter and @property functions to simplify this.
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

class Core(object):
    def __init__(self,data_folder = None):
        self.data_folder = data_folder
        self.PSF = None
        self.PF = None
        self.dx = None
        self.l= None
        self._NA = None
        self._nfrac = None
        self.cf = None
        print("Initialized!")
    # -----------------------Below is a couple of setting functions ---------------
    @property
    def nfrac(self):
        return self._nfrac

    @nfrac.setter
    def nfrac(self,new_nfrac):
        self._nfrac = new_nfrac

    @property
    def NA(self):
        return self._NA

    @NA.setter
    def NA(self, new_NA):
        self._NA = new_NA

    @property
    def lcenter(self):
        return self.l

    @lcenter.setter
    def lcenter(self, new_lcenter):
        # set the central wavelength
        self.l = new_lcenter

    @property
    def pxl(self):
        return self.dx
    @pxl.setter
    def pxl(self, new_pxl):
        self.dx = new_pxl

    @property
    def objf(self):
        return self.cf
    @objf.setter
    def objf(self, new_cf):
        self.cf = new_cf

    def load_psf(self,psf_path):
        '''
        load a psf function
        '''
        PSF = np.load(psf_path)
        nz, ny, nx = PSF.shape
        print(nz, ny, nx)
        self.PSF = PSF
        self.nx = np.min([ny,nx])
        self.nz = nz
        print("psf loaded!")


    def pupil_Simulation(self, n_wave, d_wave):
        # simulate a pupil function using given parameters; update the list
        self.PF=pupil.Simulation(self.nx, self.dx,self.l,self.nfrac,self.NA,self.cf,wavelengths=n_wave, wave_step = d_wave) # initialize the pupil function
        self.n_wave = n_wave
        self.d_wave = d_wave

    def background_reset(self, mask, psf_diam = 50):
        '''
        reset the background of the PSF
        '''
        Mx, My = np.meshgrid(np.arange(self.nx)-self.nx/2., np.arange(self.nx)-self.nx/2.)
        r_pxl = _msqrt(Mx**2 + My**2)
        bk_inner = psf_diam
        bk_outer = mask
        hcyl = np.array(self.nz*[np.logical_and(r_pxl>=bk_inner, r_pxl<bk_outer+1)])
        incyl = np.array(self.nz*[r_pxl<60])
        background = np.mean(self.PSF[hcyl])
        self.PSF[np.logical_not(incyl)] = background

        return background


    def retrievePF(self, dz, psf_diam, nIt):
        z_offset, zz = psf_zplane(self.PSF, dz, self.l/3.2) # This should be the reason!!!! >_<
        A = self.PF.plane
        Mx, My = np.meshgrid(np.arange(self.nx)-self.nx/2., np.arange(self.nx)-self.nx/2.)
        background = self.background_reset(mask = 22, psf_diam = self.PF.k_pxl)
        print( "   background = ", background)
        print( "   z_offset = ", z_offset)
        PSF_sample = self.PSF
        zs = zz-z_offset
        self.cz = int(-zs[0]//dz)
        complex_PF = self.PF.psf2pf(PSF_sample, zs, background, A, nIt)
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

    def shutDown(self):
        '''
        what should I fill here?
        '''
        pass

class _PupilFunction(object):
    '''
    a pupil function that keeps track when either complex or amplitude/phase
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
