'''
This file simulates the planes, lines in the Euclid space, and does the simplest vector analysis.
Created by Dan on 12/04/2016.
Last update: 11/21/2017.
'''

import numpy as np

def plane_line_intersect(n_plane, r_plane, k_line, r_line):
    '''
    n_plane: the normal vector of the plane
    r_plane: the coordinates of a point that the plane contains
    k_line: the directional vector of the line
    r_line: the coordinates of a point that the line contains
    '''
    nx, ny, nz = n_plane
    kx, ky, kz = k_line
    lx, ly, lz = r_line
    '''
    Let me add some steps to prevent singular solution.
    '''
    row_1 = np.array([ky+kz, kz-kx, -(kx+ky)])
    row_2 = np.array([ky-kz, -(kz+kx), kx+ky])
    b_1 = np.dot(row_1, r_line)
    b_2 = np.dot(row_2, r_line)

    M = np.array([row_1, row_2, [nx, ny, nz]]) # determine the coefficient matrix

    b = np.array([b_1, b_2, np.dot(r_plane, n_plane)])
    r_inter = np.linalg.solve(M, b)
    return r_inter



def cone_to_plane(theta, a_max):
    '''
    convert the cone to plane
    '''
    h = 1./np.tan(a_max)
    sl = h*np.tan(theta)
    return sl


def reflection(n_inc, k_inc, e_inc, normalize = False):
    '''
    Inputs:
        n_vec: the normal vector of the reflective plane, unit vector.
        k_inc: the k-vector of the incidental ray, unit vector.
        e-inc: the e-vector of the incidental ray, must be perpendicular to k_vec, unit vector.
    Output:
        k_ref: the k-vector of the reflected ray
        e_ref: the e-vector of the reflected ray
    '''
    if normalize:
        n_inc/=np.linalg.norm(n_inc)
        k_inc/=np.linalg.norm(k_inc)
        e_inc/=np.linalg.norm(e_inc)

    sv_raw = np.cross(k_inc, n_inc) # the raw s-vector, s-k-p forms a right-hand coordinate.
    sin_inc = np.linalg.norm(sv_raw) # np.sin(incidental angle)
    a_inc = np.arcsin(sin_inc) # the incident angle 
    s_inc = sv_raw/sin_inc # the unit vector of s-polarization
    p_inc = np.cross(s_inc, k_inc) # the unit vector of p-polarization
    k_ref = -np.dot(k_inc, n_inc)*n_inc + np.cross(n_inc, sv_raw) #the norm of sv_raw is sin_inc
    sp_inc = np.dot(e_inc,s_inc)*s_inc # the s-polarized component of the E-vector
    pp_inc = np.dot(e_inc,p_inc)*p_inc # the p-polarized component of the E-vector 

    p_ref = np.cross(s_inc, k_ref) # the unit vector of p-polarization in the reflected ray
    pp_ref = np.linalg.norm(pp_inc)*p_ref
    e_ref = sp_inc + pp_ref

    return k_ref, e_ref # OK! Done.

