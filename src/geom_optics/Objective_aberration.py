'''
Created by Dan on 12/01/2016. A simple calculation of aberration.
Default unit: micron.
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from geometry_elements import plane_line_intersect

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

    opl_inc = np.linalg.norm(i_inter)*n1
    opl_ref = np.linalg.norm(r_inter)*n2

    opl_diff = opl_ref-opl_inc
    return opl_diff # divide opl_diff by wavelength, then multiply by 2*pi to get the phase difference



def main():
    d = 170 # microns, the thickness of the coverslide
    n1 = 1.33
    n2 = 1.52
    N_dis = 50
    a_max  = np.arcsin(1./n1) # maximum incidental angle, unit: radian
    a_inc = np.pi-np.arange(1,N_dis+1)*a_max/N_dis # an array of incidental angles. Theta should be larger than pi/2
    g_max = 40*np.pi/180 # the maximum tilt angle
    g_tilt = np.arange(1,N_dis+1)*g_max/N_dis # an array of tilting angles of the coverslide
    a_rot = 2*np.pi*np.arange(2*N_dis)/(2*N_dis) # rotating

    [M_inc, M_rot] = np.meshgrid(a_inc, a_rot) # meshgrid
    k_vec = incident_vec(M_inc, M_rot)
    nz, ny, nx = k_vec.shape
    k_vec = np.reshape(k_vec, (nz, ny*nx)) # reshape the array
    n_vec = normal_vec(g_tilt, 0)

    NK = ny*nx
    NN = len(g_tilt)
    lam = 0.550 # the wavelength
    RS_mat = []
    RP_mat = []
    OPD_mat = []

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
            rv, Rs, Rp = refraction_plane(kv, nv, n1, n2)
            reflect_s[ik] = Rs
            reflect_p[ik] = Rp
            opl_diff = aberration_coverslide(kv, rv, nv, d, n_ind)
            opd_array[ik] = opl_diff


        RS_mat.append(reflect_s.reshape(ny,nx))
        RP_mat.append(reflect_p.reshape(ny,nx))
        OPD_mat.append(opd_array.reshape(ny,nx))
        print(ip)


    RSM  = np.array(RS_mat)
    RPM  = np.array(RP_mat)
    OPD = np.array(OPD_mat)

    results = dict()
    results['rsm'] = RSM
    results['rpm'] = RPM
    results['OPD'] = OPD
    np.savez('results', **results)



if __name__ == '__main__':
    main()
