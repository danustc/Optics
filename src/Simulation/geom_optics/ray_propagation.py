'''
Reflection of light rays.
'''
import numpy as np
import geometry_elements as ge

class Reflective_system(objective):
    '''
    build a virtual reflective mirror system.
    '''
    def __init__(self, k_init, r_init, e_init):
        self.k_init = k_init
        self.r_init = r_init
        self.e_init = e_init
        self.mirrors = [] #reflective mirrors


    def add_mirror(self,n_vec, r_mirror):
        '''
        Add a mirror into the system.
        n_vec: the normal vector of the reflective plane
        r_mirror: a point that is on the reflective plane.
        '''
        self.mirrors.append(Mirror(n_vec, r_mirror))

    def chain_reflection(self):
        '''
        chain reflection through a list of mirrors
        warning! Spatial hinderance is not included.
        '''
        k_inc = self.k_init
        e_inc = self.e_init
        r_inc = self.r_init
        for M in self.mirrors:
            n_inc = M.n_vec
            r_mirror = M.r_mirror
            r_inter = ge.plane_line_intersect(n_inc, r_mirror, k_inc, r_inc) # intersection between the ray and the mirror plane
            k_ref, e_ref = ge.reflection(n_inc,k_inc,e_inc)
            k_inc = k_ref
            r_inc = r_inter

        k_final, e_final = k_ref, e_ref
        return k_final



class Mirror(object):
    def __init__(self, n_vec, r_mirror):
        self.n_vec = n_vec
        self.r_mirror = r_mirror


    def shift(self, v_shift):
        self.r_mirror-=v_shift


