"""
 Last update by Dan Xie, 07/25/2016
 A systematic simulation of optical systems, based on POPPY package.


"""


import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import poppy 
import logging
from poppy import poppy_core


logging.basicConfig(level=logging.DEBUG)


osys  = poppy.OpticalSystem()
osys.add_pupil(poppy.CircularAperture(radius=3))    # pupil radius in meters
osys.add_detector(pixelscale=0.010, fov_arcsec=5.0)  # image plane coordinates in arcseconds

psf = osys.calc_psf(2e-6)                            # wavelength in microns
poppy.display_PSF(psf, title='The Airy Function')


