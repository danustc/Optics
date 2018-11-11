#!/usr/bin/python
# based on Ryan's inControl package with some minor modification.
# This should be wrapped into a cleaner class.
# Contains a couple of redundant functions which are device-based. Should be removed.
# Last update: 08/11/16 by Dan


import numpy as _np
from scipy import fftpack as _fftpack
from scipy import ndimage
from numpy.lib.scimath import sqrt as _msqrt
import tempfile as _tempfile
import pyfftw

class Pupil(object):
    """
    This contains the pupil definition of the pupil function, which I don't find very useful.
    """

    def __init__(self, l, n, NA, f, d=170):

        self.l = float(l)
        self.n = float(n)
        self.f = float(f)
        self.NA = NA
        self.s_max = f*NA # The maximum radius of pupil function, but it appears no where 
        self.k_max = NA/l # The radius of the pupil in the k space 
        self.d = float(d)


    def unit_disk_to_spatial_radial_coordinate(self, unit_disk):
        # This is the real radius of the pupil plane on the deformable mirror
        return self.s_max * unit_disk



class Simulation(Pupil):

    '''
    Simulates the behaviour of a microscope based on Fourier optics.

    Parameters
    ----------
    nx: int
        The side length of the pupil function or microscope image in pixels.
    dx: float
        The pixel size in the image plane. unit:
    l: float
        Light wavelength in micrometer.
    n: float
        The refractive index of immersion and sample medium. Mismatching media
        are currently not supported.
    NA: float
        The numerical aperture of the microscope objective.
    f: float
        The objective focal length in micrometer.
    '''

    def __init__(self, nx=256, dx=0.1, l=0.68, n=1.33, NA=1.27, f=3333.33, wavelengths=10, wave_step=0.005):

        dx = float(dx)
        self.dx = dx
        l = float(l)
        n = float(n)
        NA = float(NA)
        f = float(f)
        self.nx = nx
        self.ny = nx
        Pupil.__init__(self, l, n, NA, f)

        self.numWavelengths = wavelengths

        dk = 1/(nx*dx)
        self.k_pxl = int(self.k_max/dk)
        print("The pixel radius of pupil:", self.k_pxl)
        # Pupil function pixel grid:
        Mx,My = _np.mgrid[-nx/2.:nx/2.,-nx/2.:nx/2.]+0.5
        self.x_pxl = Mx # pixel grid in x  
        self.y_pxl = My # pixel grid in y
        self.r_pxl = _msqrt(Mx**2+My**2) # why the x,y,r_pxl are dimensionless?
        # Pupil function frequency space: 
        kx = dk*Mx
        ky = dk*My
        self.k = _msqrt(kx**2+ky**2) # This is in the unit of 1/x # this is a 2-D array 
        out_pupil = self.k>self.k_max
        # Axial Fourier space coordinate:
        self.kz = _msqrt((n/l)**2-self.k**2)
        self.kz[out_pupil] = 0
        self.kzs = _np.zeros((self.numWavelengths,self.kz.shape[0],self.kz.shape[1]),dtype=self.kz.dtype)
        ls = _np.linspace(l-wave_step,l+wave_step,self.numWavelengths)
        for i in range(0,self.kzs.shape[0]):
            self.kzs[i] = _msqrt((n/ls[i])**2-self.k**2)
            self.kzs[i,out_pupil] = 0
        # Scaled pupil function radial coordinate:
        self.r = self.k/self.k_max # Should be dimension-less
        self.s = self.unit_disk_to_spatial_radial_coordinate(self.r) # The real radius of the pupil. 

        # Plane wave:
        self.plane = _np.ones((nx,nx))+1j*_np.zeros((nx,nx))
        self.plane[self.k>self.k_max] = 0 # Outside the pupil: set to zero
        self.pupil_npxl = abs(self.plane.sum()) # how many non-zero pixels

        self.kx = kx # This is not used
        self.theta = _np.arctan2(My,Mx) # Polar coordinate: angle


    def pf2psf(self, PF, zs, intensity=True, verbose=False, use_pyfftw=True):
        """
        Computes the point spread function for a given pupil function.

        Parameters
        ----------
        PF: array
            The complex pupil function.
        zs: number or iteratable
            The axial position or a list of axial positions which should be
            computed. Focus is at z=0.
        intensity: bool
            Specifies if the intensity or the complex field should be returned.

        Returns
        -------
        PSF: array or memmap
            The complex PSF. If the memory is to small, a memmap will be
            returned instead of an array.
        """
        nx = self.nx
        if _np.isscalar(zs):
            zs = [zs]
        nz = len(zs)
        kz = self.kz

    # The normalization for ifft2:
        N = _np.sqrt(nx*self.ny)

        # Preallocating memory for PSF:
        try:
            if intensity:
                PSF = _np.zeros((nz,nx,nx))
            else:
                PSF = _np.zeros((nz,nx,nx))+1j*_np.zeros((nz,nx,nx))
        except MemoryError:
            print('Not enough memory for PSF, \
                    using memory map in a temporary file.')
            temp_file = _tempfile.TemporaryFile()
            if intensity:
                temp_type = float
            else:
                temp_type = complex
            PSF = _np.memmap(temp_file, dtype=temp_type, mode='w+',
                shape=(nz,nx,nx))

        for i in range(nz):
            if verbose: print('Calculating PSF slice for z={0}um.'.format(zs[i]))
            if use_pyfftw:
                aligned = pyfftw.n_byte_align(_np.exp(2*_np.pi*1j*kz*zs[i])*PF,16)
                U = N * pyfftw.interfaces.numpy_fft.ifft2(aligned)
            else:
                U = N*_fftpack.ifft2(_np.exp(2*_np.pi*1j*kz*zs[i])*PF)
            for j in range(0,self.kzs.shape[0]):
                if use_pyfftw:
                     aligned = pyfftw.n_byte_align(_np.exp(2*_np.pi*1j*self.kzs[j]*zs[i])*PF,16)
                     U = U + N*pyfftw.interfaces.numpy_fft.ifft2(aligned)
                else:
                     U = U + N*_fftpack.ifft2(_np.exp(2*_np.pi*1j*self.kzs[j]*zs[i])*PF)
            U = U/(1+self.kzs.shape[0])
            _slice_ = _fftpack.ifftshift(U)
            if intensity:
                _slice_ = _np.abs(_slice_)**2
            PSF[i] = _slice_
        if nz == 1:
            PSF = PSF[0]
        return PSF




    def psf2pf(self, PSF, zs, mu, A, nIterations=5, use_pyfftw=True, resetAmp=False,
               symmeterize=False):

        '''
        Retrieves the complex pupil function from an intensity-only
        PSF stack by relative entropy minimization. The algorithm is
        based on Kner et al., 2010, doi:10.1117/12.840943, which in turn
        is based on Deming, 2007, J Opt Soc Am A, Vol 24, No 11, p.3666.

        Parameters
        ---------
        PSF: 3D numpy.array
            An intensity PSF stack. PSF.shape has to be
            (nz, psf_tools.nx, psf_tools.nx), where nz is the arbitrary
            number of z slices.
        dz: float
            The distance between two PSF slices.
        mu: float
            The noise level of the PSF.
        A: 2D numpy.array
            The initial guess for the complex pupil function with shape
            (psf_tools.nx, psf_tools.nx).
        Edited on 07/29: instead of counting all the slices in, we only take the slices adjacent to the focal plane.
        '''
        # z spacing:
        # Number of z slices:
        nz = PSF.shape[0]
        # Noise level:
        mu = float(mu)
        kz = self.kz
        k = self.k # The lateral self.k
        k_max = self.k_max

        # Z position of slices:
        # edit on 08/09: directly pass zs instead of dz and z_offset
    # Normalization for fft2:
        N = _np.sqrt(self.nx*self.ny)

        if use_pyfftw:
            pyfftw.interfaces.cache.enable()

#         mu_purpose = _np.random.randint(1,2, size = (nz, self.ny, self.nx))
#         PSF += mu_purpose # To remove the zero pixel
        Ue = _np.ones_like(PSF).astype(_np.complex128)
        U = _np.ones_like(PSF).astype(_np.complex128)
        Uconj = _np.ones_like(PSF).astype(_np.complex128)
        Ic = _np.ones_like(PSF).astype(_np.complex128)

#         expr1 = "Ue = (PSF/Ic)*U"
#         expr2 = "Ic = mu + (U * Uconj)"

        for ii in range(nIterations):
            # Withing the iteration, A should be masked 
            print( 'Iteration',ii+1)
            # Calculate PSF field from given PF:
            U = self.pf2psf(A, zs, intensity=False)
            # Calculated PSF intensity with noise:
            Uconj = _np.conj(U)
            #weave.blitz(expr2)
            Ic = mu + (U * Uconj) # should I have sqrt here instead of 
            print("min", _np.min(PSF))
            minFunc = _np.mean(PSF*_np.log(PSF/Ic))
            print( 'Relative entropy per pixel:', minFunc)
            #redChiSq = _np.mean((PSF-Ic)**2)
            redChiSq = _np.mean((PSF-Ic)**2)
            print( 'Reduced Chi square:', redChiSq)

            # Comparing measured with calculated PSF by entropy minimization:
            Ue = (PSF/Ic)*U # All are 3 d arrays
            #weave.blitz(expr1)
            # New PF guess:
            A = _np.zeros_like(Ue) + 1j*_np.zeros_like(Ue) # temporarily set A as a 3-D array 
            for i in range(len(zs)):
                #Ue[i] = _fftpack.fftshift(Ue[i])
                if use_pyfftw:
                    Ue_aligned = pyfftw.n_byte_align(_fftpack.fftshift(Ue[i]),16)
                    fted_ue = pyfftw.interfaces.numpy_fft.fft2(Ue_aligned)
                    A[i] = fted_ue/_np.exp(2*_np.pi*1j*kz*zs[i])/N
                else:
                    fted_ue = _fftpack.fft2(_fftpack.fftshift(Ue[i])) # Transform in x-y plane
                    A[i] = fted_ue/_np.exp(2*_np.pi*1j*kz*zs[i])/N # multiply by the phase (exp(-2pi i kz *z ))
                for j in range(0,self.kzs.shape[0]): # what does this mean? # A correction for multi-wavelength
                    A[i] = A[i] + fted_ue/_np.exp(2*_np.pi*1j*self.kzs[j]*zs[i])/N
                A[i] = A[i]/(1+self.kzs.shape[0])
            A = _np.mean(A,axis=0) # Convert A from 3D to 2D; 
            #mean(abs(A))*_np.exp(1j*_np.angle(A))
            # NA restriction:
            A[k>k_max] = 0 # set everything outside k_max as 0 
            if resetAmp:
                amp = ndimage.gaussian_filter(_np.abs(A),15)
                A = amp*_np.nan_to_num(A/_np.abs(A))

            if symmeterize:
                if ii>(nIterations/2):
                    A = 0.5*(A+_np.flipud(A)) # This is to symmetrize across z-direction
                #counts = sum(abs(A))/self.pupil_npxl
                #A = counts*_np.exp(1j*angle(A))
                #A[k>k_max] = 0

        return A

