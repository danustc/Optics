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
    n2 = 1.50
    N_radius = 64
    NA =1.0
    a_max  = np.arcsin(NA/n1) # maximum incidental angle, unit: radian
    a_comp = np.arcsin(0.8/n1)
    h = N_radius/np.tan(a_max)
    r_comp = h*np.tan(a_comp)
    rx = r_comp*np.cos(2*np.pi*np.arange(101)/100.)
    ry = r_comp*np.sin(2*np.pi*np.arange(101)/100.)

    xx = np.arange(-N_radius, N_radius+1)+1.0e-06
    [MX, MY] = np.meshgrid(xx,xx)
    MR = np.sqrt(MX**2 + MY**2)
    mask = MR<=N_radius
    M_inc = np.arctan(MR/h)
    M_rot = np.arccos(MX/MR)


    g_tilt = np.array([0.01, 15., 30.01, 45.0])*np.pi/180.
    M_sl = Objective_aberration.cone_to_plane(np.pi-M_inc, a_max) # the mapped lateral position

    k_vec = Objective_aberration.incident_vec(np.pi-M_inc, M_rot)
    nz, ny, nx = k_vec.shape
    k_vec = np.reshape(k_vec, (nz, ny*nx)) # reshape the array
    n_vec = Objective_aberration.normal_vec(g_tilt, 0)

    NK = ny*nx
    NN = len(g_tilt)
    lam = 0.5150 # the wavelength
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
        rsm[np.logical_not(mask)] = 1.
        rpm[np.logical_not(mask)] = 1.
        RS_mat.append(rsm)
        RP_mat.append(rpm)
        raw_opd = opd_array.reshape(ny, nx)
        raw_opd[np.logical_not(mask)]=0
        z_coeffs = zern.fit_zernike(raw_opd, rad = N_radius+0.5, nmodes = 22, zern_data={})[0]
        fig_zd = zern_display(z_coeffs, z0=4,ylim = [-1.8,0.8])
        fig_zd.savefig('zfit_'+str(ip*15)+'_NA10.eps', format  = 'eps', dpi = 200)


        z_deduct = z_coeffs[0:4]
        deduct_opd = zern.calc_zernike(z_deduct, rad = N_radius+0.5, mask = True)
        OPD_mat.append(raw_opd-deduct_opd)

    RSM  = np.array(RS_mat)
    RPM  = np.array(RP_mat)
    OPD = np.array(OPD_mat)
    T_aver = 1-(RSM+RPM) + 0.5*(RSM**2+RPM**2)

    results = dict()
    results['rsm'] = RSM
    results['rpm'] = RPM
    results['OPD'] = OPD
    np.savez('results', **results)

#-- Plot... ------------------------g------------------------
    vmn = -8
    vmx = 10

    plt.close('all')


    fig = IMshow_pupil(OPD[0], False)
    fig.tight_layout()
    plt.savefig('T0NA10.eps', format = 'eps', dpi = 200)


    fig = IMshow_pupil(OPD[1], False)
    fig.tight_layout()
    plt.savefig('T15NA10.eps', format = 'eps', dpi = 200)

    fig = IMshow_pupil(OPD[2], False)
    fig.tight_layout()
    plt.savefig('T30NA10.eps', format = 'eps', dpi = 200)

    fig = IMshow_pupil(OPD[3], False)
    fig.tight_layout()
    plt.savefig('T40NA10.eps', format = 'eps', dpi = 200)

    kk = xx[:-1]/N_radius
    secx_0 = OPD[0][N_radius,:-1]
    secy_0 = OPD[0][:-1, N_radius]
    secy_15 = OPD[1][:-1, N_radius]
    secy_30 = OPD[2][:-1, N_radius]
    secy_45 = OPD[3][:-1, N_radius]
    secx_15 = OPD[1][N_radius,:-1]
    secx_30 = OPD[2][N_radius, :-1]
    secx_45 = OPD[3][N_radius, :-1]
    fig = plt.figure(figsize = (7.5, 4))
    plt.plot(kk, secx_0,'-g', linewidth = 2, label = '0 degree')
    plt.plot(kk, secx_15, '-r', linewidth = 2, label = '15 degree')
    plt.plot(kk, secx_30, '-b',linewidth = 2, label = '30 degree')
    plt.plot(kk, secx_45, '--k', linewidth = 2, label = '45 degree')
    plt.plot([-0.8, -0.8], [-10, 10], '-.m', linewidth = 2)
    plt.plot([0.8, 0.8], [-10, 10], '-.m', linewidth = 2)
    plt.legend()
    plt.xlabel('NA')
    plt.ylabel('Aberration')
    plt.ylim([-7,7])
    # ax.plot(xx[:-1]/N_radius, secy_0, '-g')
    plt.savefig('cross_xNA10.eps', format = 'eps', dpi = 200)

    fig = plt.figure(figsize = (7.5, 4))
    plt.plot(kk, secy_0,'-g', linewidth = 2, label = '0 degree')
    plt.plot(kk, secy_15, '-r', linewidth = 2, label = '15 degree')
    plt.plot(kk, secy_30, '-b',linewidth = 2, label = '30 degree')
    plt.plot(kk, secy_45, '--k', linewidth = 2, label = '45 degree')
    plt.plot([-0.8, -0.8], [-7, 7], '-.m', linewidth = 2)
    plt.plot([0.8, 0.8], [-7, 7], '-.m', linewidth = 2)
    plt.legend()
    plt.xlabel('NA')
    plt.ylabel('Aberration')
    plt.ylim([-7,7])
    plt.savefig('cross_yNA10.eps', format = 'eps', dpi = 200)

    #------------------------Reflectance

    fig = IMshow_pupil(T_aver[0], False)
    fig.tight_layout()
    plt.savefig('Trans0NA10.eps', format = 'eps', dpi = 200)

    fig = IMshow_pupil(T_aver[1], False)
    fig.tight_layout()
    plt.savefig('Trans15NA10.eps', format = 'eps', dpi = 200)

    fig = IMshow_pupil(T_aver[2], False)
    fig.tight_layout()
    plt.savefig('Trans30NA10.eps', format = 'eps', dpi = 200)

    fig = IMshow_pupil(T_aver[3], False)
    fig.tight_layout()
    plt.savefig('Trans45NA10.eps', format = 'eps', dpi = 200)



if __name__ == '__main__':
    main()
