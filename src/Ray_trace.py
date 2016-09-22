"""
Created by Dan on 09/21/2016 for ray-tracing calculation. 
"""

import numpy as np
import matplotlib 
from matplotlib import pyplot as plt
from microscope import objective


class Interface(object):
    """
    interface between two media with different refractive indices.
    """
    def __init__(self, rf_inc, rf_rac):
        """
        rf_inc: refractive index, n1 
        rf_rac: refractive index, n2
        ang_inc: angle of incidence 
        """
        self.n_1= rf_inc
        self.n_2 = rf_rac
        
        
    def refract_fwd(self, ang_inc, fwd = True ):
        """
        calculate angle of refraction, fwd = Forward propagation or not? 
        """
        if(fwd):
            n_inc, n_rac = (self.n_1, self.n_2)
        else:
            n_inc, n_rac = (self.n_2, self.n_1)
            
        sa_rac = np.sin(ang_inc) * n_inc/n_rac 
        ang_rac = np.arcsin(sa_rac)
        
        return ang_rac
    




class Trace(object):
    """
    This class traces one or more optical rays through a lens. 
    The direction can be from the sample side to the camera side or vice versa.
    """
    
    def __init__(self, obj, s_to_c = True):
        self.s2c = s_to_c
        self.fl = - obj.fl # where the opjective is located
        self.z_foc = 0. # initialize the focus
        self.n_immer = obj.r_ind # specify refractive index 
        self.fov = obj.fn / obj.mag
        self.alf = self.NA / self.n_immer # The alpha angle, geometrically 
        # Howabout I don' speficy that self.NA = obj.NA?
#         self.NA = obj.NA 
        # Question: how should this be compatible with the back pupil plane? 
        
        
        self.d_bfp = 2*self.fl*self.alf 

    def single_rt_focused(self, back_dist, r_dist = 0.0, th_dist = None):
        """
        Trace a single ray. Assuming that the light source is on the focal plane;  no obstacle is in between the light source and the objective;
        r_disp: radial distance with respect to the center (within the field of view)
        back_dist: the distance from measuring point to the back pupil plane.
        Also, I need to calculate the 
        """
        beta_divergence = np.arctan(r_dist/self.fl)
        bundle_distance = self.d_bfp + back_dist * r_dist / self.fl
        
        return beta_divergence, bundle_distance
    
    
    def trace_back(self, ):
        
        
        
    
    