"""
Created by Dan on 09/16/2016.
Microscope components classes: objective, camera, lens 
"""

import numpy as np


class objective(object):
    """
    this class defines a microscope objective, * means must have: 
    1. Numerical Aperture*
    2. Refractive index of the imaging medium * 
    3. Focal length *
    4. Working distance
#     5. Illuminating wavelength
    """
    def __init__(self, NA, ref_ind, FL, wd = None):
        """
        Initialize the objective. 
        """
        self.NA = NA 
        self.r_ind = ref_ind
        self.fl = FL
        self.wd = wd
        
        


class lens(object):
    """
    this class defines a lens, * means must have:
    1. focal length, unit: mm * 
    2. location in x,y plane (y,x) order * 
    3. Angle of normal direction * 
    4. diameter
    """
    
    def __init__(self, FL, loc = [0.,0.], ang = 0., diameter = 25):
        self.fl = FL 
        self.position = np.array(loc)
        self.angle = ang
        self.dia = diameter 
        # finished initializing 
    
    def pos_update(self):
        pass
    
    
class mirror(object):
    """
    this class defines a mirror. * 
    1. location, unit: mm * 
    2. angle of normal direction *
    3. diameter  
    4. coating specification
    """ 
    
    def __init__(self, loc = [0.,0.], ang = 0, diameter = 25):
        self.position = np.array(loc)
        self.angle = ang
        self.dia = diameter
        