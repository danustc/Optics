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
        self.k_max = NA/l # Why it is not 2NA / l ?
        self.d = float(d)


    def unit_disk_to_spatial_radial_coordinate(self, unit_disk):
            
        # This is the real radius of the pupil plane on the deformable mirror
        return self.s_max * unit_disk


    def spatial_radial_coordinate_to_optical_angle(self, radial_coordinate):

        alphac = _np.arcsin(radial_coordinate/(self.n*self.f) + 1j*0)
        return _np.real(alphac) - _np.imag(alphac)


    def spatial_radial_coordinate_to_xy_slope(self, s):
        # what does this one do? What's difference between s and self.s? 
        # This is cot instead of tan.
        _mc = _msqrt((self.n*self.f/self.s)**2-1)
        return _np.real(_mc) - _np.imag(_mc)


    def get_pupil_function(self, z, n_photons=1000, coverslip_tilt=0, coverslip_tilt_direction=0):

        '''
        Computes the complex pupil function of single point source.

        Parameters
        ----------
        z:
            Axial coordinate of the point source, where the origin of the
            coordinate system is at focus. Positive values are towards the
            objective.
    n_photons: int
        The number of collected photons.

        Returns
        -------
        PF: 2D array or list of 2D arrays
            If parameter *z* was a single number, a 2D array of the complex
            pupil function is returned. If *z* was an iterable of floats, a list
            of 2D pupil functions is returned.
        '''

        f = self.f
        m = self.m
        l = self.l
        n = self.n
        a = self.alpha
        t = self.theta
        d = self.d # it seems doing nothing here.
        e = coverslip_tilt
        g = 2*_np.pi*coverslip_tilt_direction/360.
        ng = 1.5255 # Coverslip refractive index

        def compute_pupil_function(z):
            phase = (2*_np.pi*n*(_msqrt(f**2*(1+m**2)-z**2)-m*z))/ \
                    (l*_msqrt(1+m**2))
            if coverslip_tilt != 0:
                # Angle between ray and coverslip normal:
                d = _np.arccos(_np.sin(a)*_np.sin(e) * \
                        (_np.cos(t)*_np.cos(g)+_np.sin(t)*_np.sin(g)) + \
                        _np.cos(a)*_np.cos(e))
                # Path length through coverslip:
                p = d/_np.sqrt(1+(n*_np.sin(d)/ng)**2)
                # The correction collar takes care of none tilt based differences:
                p -= d/_np.sqrt(1+(n*_np.sin(a)/ng)**2)
                # Path length to phase conversion:
                p *= ng*2*_np.pi/l
                phase += p
            PF = _np.sqrt(n_photons/self.pupil_npxl)*_np.exp(1j*phase)
            PF = _np.nan_to_num(PF)
            return self.apply_NA_restriction(PF)

            if _np.ndim(z) == 0 or _np.ndim(z) == 2:
                return compute_pupil_function(z)

            elif _np.ndim(z) == 1:
                return _np.array([compute_pupil_function(_) for _ in z])


    def get_sli_pupil_function(self, z0, n_photons, dmf=0, tilt=(0,0)):
        
        '''
        Computes the pupil function of a point source in front of a mirror.

        Parameters
        ----------
        z0: float or list/tuple of floats or array or list/tuple of arrays
            Distance between molecule and mirror.
        d: float
            Distance between sample-side coverslip surface and focal plane.
        tilt: tuple of floats
            Coefficients of the Zernike modes (1,-1) and (1,1) to be added to
            the pupil function of the mirror image. This simulates the effect of
            a tilted mirror.

        Returns
        -------
        PF: 2D array or list of 2D arrays
            The 2D complex pupil function or a list of 2D complex pupil
            functions if z0 was an iterable.
        '''

        dmf = float(dmf)
        tv, th = tilt
#         if tv != 0:
#             vertical_tilt = float(tv)*_zernike.zernike((1,-1), self.r,
#                     self.theta)
#         if th != 0:
#             horizontal_tilt = float(th)*_zernike.zernike((1,1), self.r,
#                     self.theta)

        def compute_sli_pupil_function(z1, z2):
            pf1 = self.get_pupil_function(z1, 0.5*n_photons)
            pf2 = self.get_pupil_function(z2, 0.5*n_photons)
            if tv != 0:
                pass
#                 pf2 = pf2*_np.exp(1j*vertical_tilt)
            if th != 0:
                pass
#                 pf2 = pf2*_np.exp(1j*horizontal_tilt)
            return pf1 - pf2
       
        if type(z0) in (list,tuple):
            z0 = _np.array(z0)

        z1 = z0 - dmf
        z2 = -(z0 + dmf)

        if _np.ndim(z0) in (0,2):
            return compute_sli_pupil_function(z1,z2)
        
        elif _np.ndim(z0) == 1:
            return _np.array(
                    [compute_sli_pupil_function(*_) for _ in zip(z1,z2)])
        


    def get_sli_virtual_focalplane_modulation(self, z, dmf=0, tilt=(0,0), correction=None):

    # Dummy n_photons of 1000:
        return -_np.angle(self.get_sli_pupil_function(z, 1000, dmf, tilt))



    def apply_NA_restriction(self, PF):

        '''
        Sets all values of a given pupil function to zero that are out of NA range.
        '''

        PF[self.r>1] = 0
        return PF

    
    def compute_Fisher_Information(self, PSF, poisson_noise, voxel_size,
            mask=None):

        '''
        Computes the Fisher information matrix as described in Ober et al.,
        Biophysical Journal, 2004, taking into account Poisson background noise.

        Parameters
        ----------
        PSF: array
            The 3D Point Spread Function, where the intensity values are the
        number of photons.
        poisson_noise: float
            The poisson background noise per pixel in photons
        voxel_size: float
            The side length of the PSF voxel in micrometer.
        mask: 2D boolean numpy.array
            Optional: A two dimensional array of the same shape as one PSF
            slice. If mask is set, the Fisher Information is only calculated for
            regions where mask is True.
        '''

        dPSF = _np.gradient(PSF, voxel_size[0], voxel_size[1], voxel_size[2])
        if mask is not None:
            dPSF = [_*mask for _ in dPSF]
            noisy = PSF + poisson_noise

        FI = [[0,0,0] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                FI[i][j] = _np.sum(dPSF[i]*dPSF[j]/noisy,-1).sum(-1)

        return FI


    def compute_CRLB(self, PSF, poisson_noise, voxel_size, mask=None):

        FI = self.compute_Fisher_Information(PSF, poisson_noise, voxel_size,
                mask)
        return [_np.sqrt(1.0/FI[_][_]) for _ in range(3)]



class Geometry:
    # This is a static class for geometry description, which has nothing to do with the operation.
    '''
    A base class for pupil.Experiment which provides basic
    geometrical data of a microscope experiment.

    Parameters
    ----------
    size: tuple
        The pixel size of a device in the pupil plane of the
        microscope.
    cx: float
        The x coordinate of the pupil function center on the
        pupil plane device in pixels.
    cy: float
        The y coordinate (see cx).
    d: float
        The diameter of the pupil function on the pupil device
        in pixels. Ok this is correct. 
    '''
    

    def __init__(self, size, cx, cy, d):
        # cx, cy: the pupil center 
        self.cx = float(cx)
        self.cy = float(cy)
        self.d = float(d)
        self.size = size
        self.nx, self.ny = size
        self.x_pxl, self.y_pxl = _np.meshgrid(_np.arange(self.nx),_np.arange(self.ny))
        self.x_pxl -= cx
        self.y_pxl -= cy
        self.r_pxl = _msqrt(self.x_pxl**2+self.y_pxl**2) # the radial coordinate in the unit of 1 (grid number )
        self.r = 2.0*self.r_pxl/d # the radial coordinate coordinate in the unit of length. Theoretically between 0 and 1.
        self.theta = _np.arctan2(self.y_pxl, self.x_pxl)
        self.x = 2.0*self.x_pxl/d # so self.x and y are in the range of (0,1)
        self.y = 2.0*self.y_pxl/d # But, isn't d the same with nx, ny?
        
    

class Experiment(Pupil):
    '''
    Provides computations for a microscope experiment based on
    Fourier optics.

    Parameters
    ----------

    geometry: pupil.Geometry
        A base object that provides basic geometric data of the
        microscope experiment.
    l: float
        The light wavelength in micrometer.
    n: float
        The refractive index of immersion and sample media.
    NA: float
        The numerical aperture of the microscope objective.
    f: float
        The objective focal length in micrometer.
    '''
    
    def __init__(self, geometry, l, n, NA, f):

        l = float(l) # wavelength 
        n = float(n) # refractive index 
        NA = float(NA)
        f = float(f) # obj focal length
        self.geometry = geometry 
        self.nx = geometry.nx
        self.ny = geometry.ny
        self.theta = geometry.theta
        Pupil.__init__(self, l, n, NA, f) 
        # So this step incorporates all the elements of Pupil into the class of Experiment 

        self.s = self.unit_disk_to_spatial_radial_coordinate(geometry.r)
        #self.s has the dimension of micron. 
        self.alpha = self.spatial_radial_coordinate_to_optical_angle(self.s)
        self.m = self.spatial_radial_coordinate_to_xy_slope(self.alpha)
        self.r = geometry.r # dimensionless, between 0 and 1
        self.size = geometry.size
        self.pupil_npxl = sum(self.r<=1)
        
        # I am so confused. Why this is not used in the whole module?


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

    def __init__(self, nx=256, dx=0.1, l=0.68, n=1.33, NA=1.27, f=3333.33, wavelengths=10, wavelength_halfmax=0.005):

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

        # Frequency sampling:
        dk = 1/(nx*dx) # What should dx be? 
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
        
#         self.k = _msqrt(kx**2+ky**2)# This is in the unit of 1/x # this is a 2-D array 
        out_pupil = self.k>self.k_max
        
        # Axial Fourier space coordinate:
        self.kz = _msqrt((n/l)**2-self.k**2)
        self.kz[out_pupil] = 0
#         self.kz.imag = 0 # brutal force

        
        
        
        print(self.kz.dtype)
        self.kzs = _np.zeros((self.numWavelengths,self.kz.shape[0],self.kz.shape[1]),dtype=self.kz.dtype)
        ls = _np.linspace(l-wavelength_halfmax,l+wavelength_halfmax,self.numWavelengths)
        for i in range(0,self.kzs.shape[0]):
            self.kzs[i] = _msqrt((n/ls[i])**2-self.k**2)
            self.kzs[i,out_pupil] = 0
#             self.kzs.imag = 0 # brutal force
             
        # Scaled pupil function radial coordinate:
        self.r = self.k/self.k_max # Should be dimension-less
        self.s = self.unit_disk_to_spatial_radial_coordinate(self.r) # The real radius of the pupil. 
        self.alpha = self.spatial_radial_coordinate_to_optical_angle(self.s)
        self.m = self.spatial_radial_coordinate_to_xy_slope(self.alpha) # Should this be inverted?  And self.m is not used at all...

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
#            
            U = N*_fftpack.ifft2(_np.exp(2*_np.pi*1j*kz*zs[i])*PF)
            
            
            
            for j in range(0,self.kzs.shape[0]):
#                 if use_pyfftw:
#                     pass
#                     aligned = pyfftw.n_byte_align(_np.exp(2*_np.pi*1j*self.kzs[j]*zs[i])*PF,16)
#                     U = U + N*pyfftw.interfaces.numpy_fft.ifft2(aligned)
#                 else:
                U = U + N*_fftpack.ifft2(_np.exp(2*_np.pi*1j*self.kzs[j]*zs[i])*PF)
            U = U/(1+self.kzs.shape[0])
            _slice_ = _fftpack.ifftshift(U)
            if intensity:
                _slice_ = _np.abs(_slice_)**2
            PSF[i] = _slice_
            
        if nz == 1:
            PSF = PSF[0]
        
        return PSF




    def psf2pf(self, PSF, zs, mu, A, nIterations=5, use_pyfftw=True, resetAmp=True,
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
            pass
#             pyfftw.interfaces.cache.enable()

        mu_purpose = _np.random.randint(1,2, size = (nz, self.ny, self.nx))
        PSF += mu_purpose # To remove the zero pixel
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
        
            
            minFunc = _np.mean(PSF*_np.log(PSF/Ic))
            print( 'Relative entropy per pixel:', minFunc)
            redChiSq = _np.mean((PSF-Ic)**2)
            print( 'Reduced Chi square:', redChiSq)

            # Comparing measured with calculated PSF by entropy minimization:
            Ue = (PSF/Ic)*U # All are 3 d arrays
            #weave.blitz(expr1)
            # New PF guess:
            A = _np.zeros_like(Ue) + 1j*_np.zeros_like(Ue) # temporarily set A as a 3-D array 
            for i in range(len(zs)):
                #Ue[i] = _fftpack.fftshift(Ue[i])
#                 if use_pyfftw:
#                     pass
#                     Ue_aligned = pyfftw.n_byte_align(_fftpack.fftshift(Ue[i]),16)
#                     fted_ue = pyfftw.interfaces.numpy_fft.fft2(Ue_aligned)
#                     A[i] = fted_ue/_np.exp(2*_np.pi*1j*kz*zs[i])/N
                
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


    def sliFocusScan(self,z0,dmfs):

        PSF = []
        SLM = -_np.angle(self.get_sli_pupil_function(z0,0.0))
        for dmf in dmfs:
            PF = self.get_sli_pupil_function(z0,dmf)
            R = abs(PF)
            Phi = _np.angle(PF)-SLM
            PF = R*_np.exp(1j*Phi)
            PSF.append(_np.abs(self.pf2psf(PF,[0.0]))**2)
        return PSF


    def modulation2slipsf(self, modulation, zs, n_photons, dmf=0,
            intensity=True, verbose=True):

        nz = len(zs)
        sliPSF = _np.zeros((nz,self.nx,self.nx))

        for i in range(nz):
            if verbose: print( 'Calculating PSF slice for z={0}um.'.format(zs[i]))

            modPF = self.get_sli_pupil_function(zs[i], n_photons, dmf) * _np.exp(1j*modulation)
            sliPSF[i] = self.pf2psf(modPF, 0, intensity=intensity,
                    verbose=False)

        return sliPSF


    def modulations2sliImages(self, modulations, z0, n_photons):

        nz = modulations.shape[0]
        images = _np.zeros((nz, self.nx, self.nx))
        PF = self.get_sli_pupil_function(z0, n_photons)

        for i in range(nz):
            modPF = PF * _np.exp(1j*modulations[i])
            images[i] = self.pf2psf(modPF, 0, intensity=True, verbose=False)

        return images
