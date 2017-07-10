'''
main test function.
'''

import numpy as np
import matplotlib.pyplot as plt
import geom_optics.Objective_aberration as Objective_aberration
import geom_optics.geometry_elements
import libtim.zern as zern
from visualization import zern_display, IMshow_pupil

def main():
    d = 170 # microns, the thickness of the coverslide
    n1 = 1.33
    n2 = 1.52
    N_radius = 64
    NA =1.0
    #NA = 0.8
    a_max  = np.arcsin(NA/n1) # maximum incidental angle, unit: radian
    h = N_radius/np.tan(a_max)

    xx = np.arange(-N_radius, N_radius)+0.5
    yy = np.arange(-N_radius, N_radius)+0.5
    [MX, MY] = np.meshgrid(xx,yy)
    MR = np.sqrt(MX**2 + MY**2)
    M_inc = np.arctan(MR/h)
    M_rot = np.arccos(MX/MR)
    #M_rot = (M_rot+np.flipud(M_rot))/2.

    g_tilt = np.array([0.01, 15, 35, 40, 45.0])*np.pi/180. #tilt angles 

    k_vec = Objective_aberration.incident_vec(np.pi-M_inc, M_rot)
    nz, ny, nx = k_vec.shape
    k_vec = np.reshape(k_vec, (nz, ny*nx)) # reshape the array
    n_vec = Objective_aberration.normal_vec(g_tilt, 0)

    NK = ny*nx
    NN = len(g_tilt)
    NA_crop = np.ones(NN)
    NA_crop[-1] = 0.80
    NA_crop[-2] = 0.80
    lam = 0.5150 # the wavelength
    RS_mat = []
    RP_mat = []
    OPD_mat = []

    n_ind = [n1,n2]
    for ip in np.arange(NN):
        '''
        Iterate through NN vectors
        '''
        NA_percent = NA_crop[ip] # the percentage of NA used in zernike fitting 
        ncut = int(NA_percent*N_radius)
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
                rv, Rs, Rp = Objective_aberration.refraction_plane(kv, nv, n1, n2)
                reflect_s[ik] = Rs
                reflect_p[ik] = Rp
                opl_diff = Objective_aberration.aberration_coverslide_ray(kv, rv, nv, d, n_ind)[0]
                opd_array[ik] = opl_diff
            else:
                print("Error!")

        opd_array = opd_array/lam
        rsm = reflect_s.reshape(ny, nx)
        rpm = reflect_p.reshape(ny, nx)
        mask = MR<=(NA_percent*N_radius)
        rsm[np.logical_not(mask)] = 1.
        rpm[np.logical_not(mask)] = 1.
        RS_mat.append(rsm)
        RP_mat.append(rpm)
        raw_opd = opd_array.reshape(ny, nx)
        raw_x = raw_opd[N_radius, N_radius-ncut:N_radius+ncut]
        m, b = np.polyfit(xx[N_radius-ncut:N_radius+ncut],raw_x,1)
        raw_opd = raw_opd-(MX*m+b)
        raw_opd[np.logical_not(mask)] = 0 #crop the OPD 
        OPD_mat.append(raw_opd)
        # end for ip


    RSM  = np.array(RS_mat)
    RPM  = np.array(RP_mat)
    OPD = np.array(OPD_mat)
    T_aver = 1-(RSM+RPM) + 0.5*(RSM**2+RPM**2)

    results = dict()
    results['rsm'] = RSM
    results['rpm'] = RPM
    results['OPD'] = OPD
    np.savez('results_0removed_xtilt', **results)


# ------------------------main program---------------------

if __name__ == '__main__':
    main()
