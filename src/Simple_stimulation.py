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
    a_comp = np.arcsin(0.8/n1)
    h = N_radius/np.tan(a_max)
    r_comp = h*np.tan(a_comp)
    #rx = r_comp*np.cos(2*np.pi*np.arange(101)/100.)
    #ry = r_comp*np.sin(2*np.pi*np.arange(101)/100.)

    xx = np.arange(-N_radius, N_radius+1)+1.0e-07
    [MX, MY] = np.meshgrid(xx,xx)
    MR = np.sqrt(MX**2 + MY**2)
    M_inc = np.arctan(MR/h)
    M_rot = np.arccos(MX/MR)


    g_tilt = np.array([0.01, 15, 30, 40, 45.0])*np.pi/180. #tilt angles 
    M_sl = Objective_aberration.cone_to_plane(np.pi-M_inc, a_max) # the mapped lateral position

    k_vec = Objective_aberration.incident_vec(np.pi-M_inc, M_rot)
    nz, ny, nx = k_vec.shape
    k_vec = np.reshape(k_vec, (nz, ny*nx)) # reshape the array
    n_vec = Objective_aberration.normal_vec(g_tilt, 0)

    NK = ny*nx
    NN = len(g_tilt)
    NA_crop = np.ones(NN)
    NA_crop[-1] = 0.80
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
        raw_opd[np.logical_not(mask)] = 0 #crop the OPD 
        z_coeffs = zern.fit_zernike(raw_opd[N_radius-ncut:N_radius+ncut, N_radius-ncut:N_radius+ncut], rad = ncut, nmodes = 22, zern_data={})[0]

    # remove the tip-tilt only or remove the defocus?
        z_deduct = z_coeffs[0:3]
        deduct_opd = zern.calc_zernike(z_deduct, rad = ncut+0.5, zern_data = {}, mask = True)
        op_base = np.zeros_like(raw_opd)
        op_center = raw_opd[N_radius-ncut:N_radius+ncut+1, N_radius-ncut:N_radius+ncut+1]-deduct_opd
        op_base[N_radius-ncut:N_radius+ncut+1,N_radius-ncut:N_radius+ncut+1] =op_center
        op_base[np.logical_not(mask)] = 0
        OPD_mat.append(op_base)
        # end for ip


    RSM  = np.array(RS_mat)
    RPM  = np.array(RP_mat)
    OPD = np.array(OPD_mat)
    T_aver = 1-(RSM+RPM) + 0.5*(RSM**2+RPM**2)

    results = dict()
    results['rsm'] = RSM
    results['rpm'] = RPM
    results['OPD'] = OPD
    np.savez('results_0-3removed', **results)


# ------------------------main program---------------------

if __name__ == '__main__':
    main()
