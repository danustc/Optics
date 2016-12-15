'''
Created by Dan on 12/01/2016. A simple calculation of aberration.
Default unit: micron.
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from geometry_elements import plane_line_intersect, cone_to_plane
import libtim.zern as zern

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
    if(kr_vec[2]>0):
        print(a_diff, k_vec)
    # next, let's calculate the reflectance
    cos_inc = np.cos(a_inc)
    cos_ref = np.cos(a_ref)
    Rs = ((n1*cos_inc - n2*cos_ref)/(n1*cos_inc + n2*cos_ref))**2
    Rp = ((n1*cos_ref - n2*cos_inc)/(n1*cos_ref + n2*cos_inc))**2

    return kr_vec, Rs, Rp
    # done with refraction_plane


def aberration_coverslide_ray(ki_vec, kr_vec, n_vec, d, n_ind):
    '''
    calculates the aberration (phase shift)
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
    # if(z_inter>0):
        # print(kr_vec, z_inter)
    i_inter = z_inter * ki_vec/ki_vec[2] # the original ki-should end_up
    sin_s = np.sqrt(ki_vec[0]**2 + ki_vec[1]**2)/np.linalg.norm(ki_vec)
    sl = np.linalg.norm(r_inter-i_inter)*sin_s



    opl_inc = np.linalg.norm(i_inter)*n1
    opl_ref = np.linalg.norm(r_inter)*n2 + sl*n1
    opl_diff = opl_ref-opl_inc
    return opl_diff, r_inter, i_inter# divide opl_diff by wavelength, then multiply by 2*pi to get the phase difference
    # done with aberration_coverslide


def aberration_coverslide_objective(NA, N_radius, g_tilt, lam = 0.550, n_med = 1.33, n_mis = 1.52):
    '''
    Aberration of the whole objective.
    NA: the numerical aperture of the objective.
    N_radius: the grid number. The total grid number = 2*N_radius + 1.
    g_tilt: tilt angle of the coverslip
    '''
    xx = np.arange([-N_radius, N_radius])
    yy = xx
    [MX, MY] = np.meshgrid(xx,yy)
    a_max  = np.arcsin(NA/n1) # maximum incidental angle, unit: radian
    h = N_radius/np.tan(a_max)

    MR = np.sqrt(MX**2 + MY**2)
    mask = MR<=N_radius
    M_inc = np.arctan(MR/h)
    M_rot = np.arccos(MX/MR)
    M_sl = cone_to_plane(np.pi-M_inc, a_max) # the mapped lateral position

    k_vec = incident_vec(np.pi-M_inc, M_rot)
    nz, ny, nx = k_vec.shape
    k_vec = np.reshape(k_vec, (nz, ny*nx)) # reshape the array
    n_vec = normal_vec(g_tilt, 0)


    n_ind = [n1,n2]
    for ip in np.arange(NN):
        '''
        Iterate through NN vectors
        '''
        nv = n_vec[:,ip]
        reflect_s = np.zeros(NK)
        reflect_p = np.zeros(NK)
        opd_array = np.zeros(NK)
        for ik in np.arange(NK):
            '''
            iterate through the NN vectors of N_vec and NK vectors of K vec
            '''
            kv = k_vec[:,ik]
            if(np.allclose(np.linalg.norm(kv),1.)):
                rv, Rs, Rp = refraction_plane(kv, nv, n1, n2)
                reflect_s[ik] = Rs
                reflect_p[ik] = Rp
                opl_diff = aberration_coverslide_ray(kv, rv, nv, d, n_ind)[0]
                opd_array[ik] = opl_diff
            else:
                print("Error!")

        opd_array = opd_array/lam
        rsm = reflect_s.reshape(ny, nx)
        rpm = reflect_p.reshape(ny, nx)
        rsm[np.logical_not(mask)] = 1.
        rpm[np.logical_not(mask)] = 1.
        RS_mat.append(rsm)
        RP_mat.append(rpm)
        raw_opd = opd_array.reshape(ny, nx)
        raw_opd[np.logical_not(mask)]=0
        z_coeffs = zern.fit_zernike(raw_opd, rad = N_radius+0.5, nmodes = 25, zern_data={})[0]
        print(z_coeffs)
        z_deduct = z_coeffs[0:4]
        deduct_opd = zern.calc_zernike(z_deduct, rad = N_radius+0.5, mask = True)
        OPD_mat.append(raw_opd-deduct_opd)


    RSM  = np.array(RS_mat)
    RPM  = np.array(RP_mat)
    OPD = np.array(OPD_mat)

    T_aver = 1-(RSM+RPM) + 0.5*(RSM**2+RPM**2)
