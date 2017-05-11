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

#-- Plot... ------------------------g------------------------
    results = np.load('results.npz')
    N_radius = 64
    xx = np.arange(-N_radius, N_radius+1)+1.0e-06
    OPD = results['OPD']
    RSM = results['rsm']
    RPM = results['rpm']
    vmn = -8
    vmx = 10

    T_aver = 1-(RSM+RPM) + 0.5*(RSM**2+RPM**2)
    plt.close('all')
    csa = [-6, 5]
    csb = [-4, 3]

    #fig = IMshow_pupil(OPD[0], False, c_scale = csa, inner_diam = 0.8)
    #fig.tight_layout()
    #plt.savefig('T0NA10.eps', format = 'eps', dpi = 200)


    #fig = IMshow_pupil(OPD[1], False, c_scale = csa, inner_diam = 0.8)
    #fig.tight_layout()
    #plt.savefig('T15NA10.eps', format = 'eps', dpi = 200)

    #fig = IMshow_pupil(OPD[2], False,c_scale = csa, inner_diam = 0.8)
    #fig.tight_layout()
    #plt.savefig('T30NA10.eps', format = 'eps', dpi = 200)

    #fig = IMshow_pupil(OPD[3], False,c_scale = csa, inner_diam = 0.8, crop = 0.8)
    #fig.tight_layout()
    #plt.savefig('T45NA10.eps', format = 'eps', dpi = 200)


    #OPD_base = OPD[0]

    #fig = IMshow_pupil(OPD[1]-OPD_base, False, c_scale = csb, inner_diam = 0.8)
    #fig.tight_layout()
    #plt.savefig('diff_T15NA10.eps', format = 'eps', dpi = 200)

    #fig = IMshow_pupil(OPD[2]-OPD_base, False, c_scale = csb, inner_diam = 0.8)
    #fig.tight_layout()
    #plt.savefig('diff_T30NA10.eps', format = 'eps', dpi = 200)

    #fig = IMshow_pupil(OPD[3]-OPD_base, False, c_scale = csb, inner_diam = 0.8, crop = 0.8)
    #fig.tight_layout()
    #plt.savefig('diff_T45NA10.eps', format = 'eps', dpi = 200)

    alpha_para = -10.0
    kk = xx[:-1]/N_radius
    secx_0 = OPD[0][N_radius,:-1]-OPD[0][N_radius, N_radius]
    secx_15 = OPD[1][N_radius,:-1]-OPD[1][N_radius, N_radius]
    secx_30 = OPD[2][N_radius, :-1]-OPD[2][N_radius, N_radius]
    secx_45 = OPD[3][N_radius, :-1]-OPD[3][N_radius, N_radius]

    psec_corr = alpha_para*(kk+0.1)**2 # the parabolic mirror 
    secy_0 = OPD[0][:-1, N_radius]-OPD[0][N_radius, N_radius]
    secy_15 = OPD[1][:-1, N_radius]-OPD[1][N_radius, N_radius]
    secy_30 = OPD[2][:-1, N_radius]-OPD[2][N_radius, N_radius]
    secy_45 = OPD[3][:-1, N_radius]-OPD[3][N_radius, N_radius]
    dsecx_15 = secx_15 - secx_0
    dsecx_30 = secx_30 - secx_0
    dsecx_45 = secx_45 - secx_0

    psecx_0 = secx_0 # for zero-tilt, no CL correction is needed.
    psecx_15 =secx_15-0.5*psec_corr
    psecx_30 =secx_30-0.5*psec_corr
    psecx_45 =secx_45-2.0*psec_corr
    dpsecx_15 = psecx_15 - psecx_0
    dpsecx_30 = psecx_30 - psecx_0
    dpsecx_45 = psecx_45 - psecx_0

    fig = plt.figure(figsize = (5,3))
    #plt.plot(kk, psecx_0, '-g', linewidth = 2, label = '0 degree')
    plt.plot(kk, psecx_15, '-r', linewidth = 2, label = '15 degree')
    plt.plot(kk, psecx_30, '-b',linewidth = 2, label = '30 degree')
    plt.plot(kk, psecx_45, '--k', linewidth = 2, label = '45 degree')
    plt.plot([-0.8, -0.8], [-10, 10], '-.m', linewidth = 1)
    plt.plot([0.8, 0.8], [-10, 10], '-.m', linewidth = 1)
    plt.legend(fontsize = 12)
    plt.xlabel('NA')
    plt.ylabel('Aberration')
    plt.ylim([-7.5,3.5])
    plt.xlim([-1, 1])
    plt.xticks(np.arange(-1,1.1,0.2),fontsize = 12)

    plt.tight_layout()
    plt.savefig('CL_cross_xNA10.eps', format = 'eps', dpi = 200)


    fig = plt.figure(figsize = (5,3))
    plt.plot(kk, dpsecx_15, '-r', linewidth = 2, label = '15 degree')
    plt.plot(kk, dpsecx_30, '-b',linewidth = 2, label = '30 degree')
    plt.plot(kk, dpsecx_45, '--k', linewidth = 2, label = '45 degree')
    plt.plot([-0.8, -0.8], [-10, 10], '-.m', linewidth = 1)
    plt.plot([0.8, 0.8], [-10, 10], '-.m', linewidth = 1)
    plt.legend(fontsize = 12)
    plt.xlabel('NA')
    plt.ylim([-7.5,3.5])
    plt.xlim([-1, 1])
    plt.xticks(np.arange(-1,1.1,0.2),fontsize = 12)
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position('right')
    plt.ylabel('Relative aberration')
    # ax.plot(xx[:-1]/N_radius, secy_0, '-g')
    plt.tight_layout()
    plt.savefig('DCL_cross_xNA10.eps', format = 'eps', dpi = 200)


    dsecy_15 = secy_15 - secy_0
    dsecy_30 = secy_30 - secy_0
    dsecy_45 = secy_45 - secy_0
    fig = plt.figure(figsize = (5,3))
    plt.plot(kk, dsecx_15, '-r', linewidth = 2, label = '15 degree')
    plt.plot(kk, dsecx_30, '-b',linewidth = 2, label = '30 degree')
    plt.plot(kk, dsecx_45, '--k', linewidth = 2, label = '45 degree')
    plt.plot([-0.8, -0.8], [-10, 10], '-.m', linewidth = 1)
    plt.plot([0.8, 0.8], [-10, 10], '-.m', linewidth = 1)
    plt.legend(fontsize = 12)
    plt.xlabel('NA')
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position('right')
    plt.ylabel('Relative aberration')
    plt.ylim([-7.5,3.5])
    plt.xlim([-1, 1])
    plt.xticks(np.arange(-1,1.1,0.2),fontsize = 12)
    # ax.plot(xx[:-1]/N_radius, secy_0, '-g')
    plt.tight_layout()
    plt.savefig('diff_cross_xNA10.eps', format = 'eps', dpi = 200)

    fig = plt.figure(figsize = (5,3))
    plt.plot(kk, dsecy_15, '-r', linewidth = 2, label = '15 degree')
    plt.plot(kk, dsecy_30, '-b',linewidth = 2, label = '30 degree')
    plt.plot(kk, dsecy_45, '--k', linewidth = 2, label = '45 degree')
    plt.plot([-0.8, -0.8], [-10, 10], '-.m', linewidth = 1)
    plt.plot([0.8, 0.8], [-10, 10], '-.m', linewidth = 1)
    plt.legend(fontsize = 12)
    plt.xlabel('NA')
    plt.ylim([-7.5,3.5])
    plt.xlim([-1, 1])
    plt.xticks(np.arange(-1,1.1,0.2),fontsize = 12)
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position('right')
    plt.ylabel('Relative aberration')
    # ax.plot(xx[:-1]/N_radius, secy_0, '-g')
    plt.tight_layout()
    plt.savefig('diff_cross_yNA10.eps', format = 'eps', dpi = 200)

    fig = plt.figure(figsize = (5,3))
    plt.plot(kk, secx_0,'-g', linewidth = 2, label = '0 degree')
    plt.plot(kk, secx_15, '-r', linewidth = 2, label = '15 degree')
    plt.plot(kk, secx_30, '-b',linewidth = 2, label = '30 degree')
    plt.plot(kk, secx_45, '--k', linewidth = 2, label = '45 degree')
    plt.plot([-0.8, -0.8], [-10, 10], '-.m', linewidth = 1)
    plt.plot([0.8, 0.8], [-10, 10], '-.m', linewidth = 1)
    plt.legend(fontsize = 12)
    plt.xlabel('NA')
    plt.ylabel('Aberration')
    plt.ylim([-7.5,3.5])
    plt.xlim([-1, 1])
    plt.xticks(np.arange(-1,1.1,0.2),fontsize = 12)
    # ax.plot(xx[:-1]/N_radius, secy_0, '-g')
    plt.tight_layout()
    plt.savefig('cross_xNA10.eps', format = 'eps', dpi = 200)

    fig = plt.figure(figsize = (5,3))
    plt.plot(kk, secy_0,'-g', linewidth = 2, label = '0 degree')
    plt.plot(kk, secy_15, '-r', linewidth = 2, label = '15 degree')
    plt.plot(kk, secy_30, '-b',linewidth = 2, label = '30 degree')
    plt.plot(kk, secy_45, '--k', linewidth = 2, label = '45 degree')
    plt.plot([-0.8, -0.8], [-7, 7], '-.m', linewidth = 1)
    plt.plot([0.8, 0.8], [-7, 7], '-.m', linewidth = 1)
    plt.legend(fontsize = 12)
    plt.xlabel('NA', fontsize = 12)
    plt.ylabel('Aberration')
    plt.xticks(np.arange(-1,1.1,0.2), fontsize = 12)
    plt.ylim([-7.5,3.5])
    plt.xlim([-1, 1])
    plt.tight_layout()
    plt.savefig('cross_yNA10.eps', format = 'eps', dpi = 200)

    #------------------------Reflectance
#
#    fig = IMshow_pupil(T_aver[0], False, inner_diam = 0.8)
#    fig.tight_layout()
#    plt.savefig('Trans0NA10.eps', format = 'eps', dpi = 200)
#
#    fig = IMshow_pupil(T_aver[1], False, inner_diam = 0.8)
#    fig.tight_layout()
#    plt.savefig('Trans15NA10.eps', format = 'eps', dpi = 200)
#
#    fig = IMshow_pupil(T_aver[2], False, inner_diam = 0.8)
#    fig.tight_layout()
#    plt.savefig('Trans30NA10.eps', format = 'eps', dpi = 200)
#
#    fig = IMshow_pupil(T_aver[3], False, crop = 0.8, inner_diam = 0.8)
#    fig.tight_layout()
#    plt.savefig('Trans45NA10.eps', format = 'eps', dpi = 200)
#
    fig = plt.figure(figsize = (4,1.8))
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize = 12)
    ax.plot(kk, T_aver[0][N_radius, :-1], '-k', linewidth = 2)
    ax.set_ylim([0.,1.1])
    plt.tight_layout()
    fig.savefig('Tcross0_NA10.eps', format = 'eps', dpi = 200)
    plt.cla()

    ax.plot(kk, T_aver[1][N_radius, :-1], '-k', linewidth = 2)
    ax.set_ylim([0.,1.1])
    plt.tight_layout()
    fig.savefig('Tcross15_NA10.eps', format = 'eps', dpi = 200)
    plt.cla()
    ax.plot(kk, T_aver[2][N_radius, :-1], '-k', linewidth = 2)
    ax.set_ylim([0.,1.1])
    plt.tight_layout()
    fig.savefig('Tcross30_NA10.eps', format = 'eps', dpi = 200)
    plt.cla()
    ax.plot(kk, T_aver[3][N_radius, :-1], '-k', linewidth = 2)
    ax.plot([-0.8, -0.8], [0, 1], '-.m', linewidth = 1)
    ax.plot([0.8, 0.8], [0, 1], '-.m', linewidth = 1)
    ax.set_ylim([0.,1.1])
    plt.tight_layout()
    fig.savefig('Tcross45_NA10.eps', format = 'eps', dpi = 200)

if __name__ == '__main__':
    main()
