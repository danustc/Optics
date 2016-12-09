'''
Created by Dan on 12/01/2016. A simple calculation of aberration.
Default unit: micron.
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from geom_optics.geometry_elements import plane_line_intersect, cone_to_plane

'''
small functions
'''
def incident_vec(theta, varphi):
    '''
    The incidental direction vector
    '''
    k_vec = np.array([np.sin(theta)*np.cos(varphi), np.sin(theta)*np.sin(varphi), np.cos(theta)])
    return k_vec



def normal_vec(gamma, varphi):
    '''
    The normal vector of a plane: deviates from z-direction by gamma, then rotate in the x-y plane by varphi.
    '''
    n_vec = np.array([np.sin(gamma)*np.cos(varphi), np.sin(gamma)*np.sin(varphi), np.cos(gamma)])
    return n_vec
    # done with normal_vec


def incident_plane(k_vec, n_vec):
    '''
    k_vec: the directional vector of the incidental ray
    n_vec: the normal vector of the incidental plane
    '''
    s_raw = np.cross(k_vec, n_vec)
    sin_ins  = np.linalg.norm(s_raw) # the sin of incidental angle as well as the norm of s_raw
    s_vec = s_raw/sin_ins # the normalized vector of s-polarization
    p_vec = np.cross(s_vec, k_vec) # no further normalization needed.
    return s_vec, p_vec, sin_ins
    # done with incident plane


def refraction_plane(k_vec, n_vec, n1, n2):
    '''
    k_vec: the directional vector of the incidental ray
    n_vec: the normal vector of the incidental plane
    n1, n2: the refractive indices
    '''
    s_vec, p_vec, sin_ins = incident_plane(k_vec, n_vec)

    sin_ref = n1*sin_ins /n2 # sin(phi')
    a_inc = np.arcsin(sin_ins)
    a_ref = np.arcsin(sin_ref)
    a_diff = a_inc-a_ref
    kr_vec = np.cos(a_diff)*k_vec -np.sin(a_diff)*p_vec
    # next, let's calculate the reflectance
    cos_inc = np.cos(a_inc)
    cos_ref = np.cos(a_ref)
    Rs = ((n1*cos_inc - n2*cos_ref)/(n1*cos_inc + n2*cos_ref))**2
    Rp = ((n1*cos_ref - n2*cos_inc)/(n1*cos_ref + n2*cos_inc))**2

    return kr_vec, Rs, Rp
    # done with refraction_plane


def aberration_coverslide(ki_vec, kr_vec, n_vec, d, n_ind):
    '''
    ki_vec: the directional vector of the incidental ray
    kr_vec: the directional vector of the refracted ray
    n_vec: the normal vector of the incidental plane
    d: the thickness of the coverslide
    n_ind: the refractive indices (n1,n2)
    '''
    n1, n2 = n_ind
    r_plane = -d*n_vec
    r_line = np.array([0.,0.,0.])
    # _plane, r_plane, k_line, r_line
    r_inter = plane_line_intersect(n_vec, r_plane, kr_vec, r_line)
    z_inter = r_inter[2] # the z-coordinate of the intersection
    i_inter = z_inter * ki_vec/ki_vec[2] # the original ki-should end_up
    sin_s = np.sqrt(ki_vec[0]**2 + ki_vec[1]**2)/np.linalg.norm(ki_vec)
    sl = np.linalg.norm(r_inter-i_inter)*sin_s
    print(sl)


    opl_inc = np.linalg.norm(i_inter)*n1
    opl_ref = np.linalg.norm(r_inter)*n2 + sl*n1
    # print(opl_inc, opl_ref)

    opl_diff = opl_ref-opl_inc
    return opl_diff, r_inter, i_inter# divide opl_diff by wavelength, then multiply by 2*pi to get the phase difference



def dummy():
    d = 170 # microns, the thickness of the coverslide
    n1 = 1.33
    n2 = 1.52
    N_dis = 50
    a_max  = np.arcsin(1./n1) # maximum incidental angle, unit: radian

    n_vec = np.array([0.,0.,1.])
    a_inc = np.arange(1,N_dis+1)*a_max/N_dis
    r_max = np.arcsin(np.sin(a_max)*n1/n2)
    s = d*np.tan(r_max)

    k_vec = incident_vec(np.pi-a_inc, 0.)
    r_vec,Rs, Rp = refraction_plane(k_vec[:,-1], n_vec, n1, n2)
    print(k_vec[:,-1],r_vec)
    o_diff, r_inter, i_inter = aberration_coverslide(k_vec[:,-1], r_vec, n_vec, d, [n1,n2])
    print(s, r_inter, i_inter)
    diff_inter = d*(np.tan(a_max)-np.tan(r_max))
    print(diff_inter)
    o_diff_manual = n1*diff_inter*np.sin(a_max) + n2* d/np.cos(r_max) - n1*d/np.cos(a_max)

    print("er", d/np.cos(r_max)-np.linalg.norm(r_inter))
    print("ei", d/np.cos(a_max)-np.linalg.norm(i_inter))

    sl = diff_inter*np.sin(a_max)


    opl_inc = np.linalg.norm(i_inter)*n1
    opl_ref = np.linalg.norm(r_inter)*n2 + sl*n1
    print(o_diff_manual, o_diff, opl_ref-opl_inc)
