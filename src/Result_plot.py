'''
main test function.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import geom_optics.Objective_aberration as Objective_aberration
import geom_optics.geometry_elements
import libtim.zern as zern
from visualization import zern_display, IMshow_pupil

def cross_plot(cr_group, ylim, label_flags, lcolors = ['r', 'g', 'b', 'c'], NA=1.0, fsize = (5,3.5), crop = None):
    '''
    do the cross_section plot
    cr_group: the array of cross_plot
    '''
    nd, nc = cr_group.shape
    N_radius = int(nd/2)
    kk = NA*(np.arange(nd)-N_radius+0.5)/N_radius # the k-axis, from -NA
    figc = plt.figure(figsize = fsize)
    ax=figc.add_subplot(111)
    ax.plot(kk, cr_group, linewidth = 2)
    for i,j in enumerate(ax.lines):
        print(lcolors[i])
        j.set_color(lcolors[i])

    ax.plot([-0.8, -0.8], [-10, 10], '-.k', linewidth = 1)
    ax.plot([0.8, 0.8], [-10, 10], '-.k', linewidth = 1)
    ax.legend(label_flags, fontsize = 12)
    ax.set_xlabel('NA')
    ax.set_ylabel('Aberration')
    ax.set_ylim(ylim)
    if crop is None:
        ax.set_xlim([-1, 1])
        ax.set_xticks(np.arange(-1,1.1,0.2))
    else:
        ax.set_xlim([-crop, crop])
        ax.set_xticks(np.arange(-crop,crop+0.1,0.2))
    ax.tick_params(labelsize = 12)

    plt.tight_layout()

    return figc

#------------------------------the main test program---------------------------

def main():
    color_dic = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    results = np.load('results_0-3removed.npz')
    N_radius = 64
    xx = np.arange(-N_radius, N_radius+1)+1.0e-06
    OPD = results['OPD']
    RSM = results['rsm']
    RPM = results['rpm']
    vmn = -8
    vmx = 10

    T_aver = 1-(RSM+RPM) + 0.5*(RSM**2+RPM**2)
    plt.close('all')
    csa = np.array([-6, 5])
    csb = np.array([-4, 3])
    csc = np.array([-8, 4])
    color_spec = [color_dic['darkorchid'], color_dic['orange'],color_dic['steelblue'], color_dic['limegreen']]



    #fig = IMshow_pupil(OPD[0], False, c_scale = csa, inner_diam = 0.8)
    #fig.tight_layout()
    #plt.savefig('T0NA10.eps', format = 'eps', dpi = 200)


    #fig = IMshow_pupil(OPD[1], False, c_scale = csa, inner_diam = 0.8)
    #fig.tight_layout()
    #plt.savefig('T15NA10.eps', format = 'eps', dpi = 200)

    #fig = IMshow_pupil(OPD[2], False,c_scale = csa, inner_diam = 0.8)
    #fig.tight_layout()
    #plt.savefig('T30NA10.eps', format = 'eps', dpi = 200)

    #fig = IMshow_pupil(OPD[3], False,c_scale = csa, inner_diam = 0.8)
    #fig.tight_layout()
    #plt.savefig('T40NA10.eps', format = 'eps', dpi = 200)

    #fig = IMshow_pupil(OPD[4], False,c_scale = csa, inner_diam = 0.8, crop = 0.8)
    #fig.tight_layout()
    #plt.savefig('T45NA10.eps', format = 'eps', dpi = 200)

    #OPD_base = OPD[0]

    #fig = IMshow_pupil(OPD[1]-OPD_base, False, c_scale = csb, inner_diam = 0.8)
    #fig.tight_layout()
    #plt.savefig('diff_T15NA10.eps', format = 'eps', dpi = 200)

    #fig = IMshow_pupil(OPD[2]-OPD_base, False, c_scale = csb, inner_diam = 0.8)
    #fig.tight_layout()
    #plt.savefig('diff_T30NA10.eps', format = 'eps', dpi = 200)

    #fig = IMshow_pupil(OPD[3]-OPD_base, False, c_scale = csb, inner_diam = 0.8)
    #fig.tight_layout()
    #plt.savefig('diff_T40NA10.eps', format = 'eps', dpi = 200)

    #fig = IMshow_pupil(OPD[4]-OPD_base, False, c_scale = csb, inner_diam = 0.8, crop = 0.8)
    #fig.tight_layout()
    #plt.savefig('diff_T45NA10.eps', format = 'eps', dpi = 200)

    alpha_para = -10.0
    kk = xx[:-1]/N_radius
    psec_corr = alpha_para*(kk+0.1)**2 # the parabolic mirror 
    # construct the cross section 
    secx_00 = OPD[0][N_radius, :-1]#-OPD[0][N_radius, N_radius]
    secx_15 = OPD[1][N_radius, :-1]#-OPD[1][N_radius, N_radius]
    secx_30 = OPD[2][N_radius, :-1]#-OPD[2][N_radius, N_radius]
    secx_40 = OPD[3][N_radius, :-1]#-OPD[3][N_radius, N_radius]
    secx_45 = OPD[4][N_radius, :-1]#-OPD[4][N_radius, N_radius]

    secy_00 = OPD[0][:-1, N_radius]#-OPD[0][N_radius, N_radius]
    secy_15 = OPD[1][:-1, N_radius]#-OPD[1][N_radius, N_radius]
    secy_30 = OPD[2][:-1, N_radius]#-OPD[2][N_radius, N_radius]
    secy_40 = OPD[3][:-1, N_radius]#-OPD[3][N_radius, N_radius]
    secy_45 = OPD[4][:-1, N_radius]#-OPD[4][N_radius, N_radius]


    secy_30[N_radius-1] = np.mean(secy_30[N_radius-3:N_radius+2])
    secy_30[N_radius] = np.mean(secy_30[N_radius-2:N_radius+3])
    secy_30[N_radius+1] = np.mean(secy_30[N_radius-1:N_radius+4])
    secy_40[N_radius-1] = np.mean(secy_40[N_radius-3:N_radius+2])
    secy_40[N_radius] = np.mean(secy_40[N_radius-2:N_radius+3])
    secy_40[N_radius+1] = np.mean(secy_40[N_radius-1:N_radius+4])
    secy_45[N_radius-1] = np.mean(secy_45[N_radius-3:N_radius+2])
    secy_45[N_radius] = np.mean(secy_45[N_radius-2:N_radius+3])
    secy_45[N_radius+1] = np.mean(secy_45[N_radius-1:N_radius+4])

    label_flags = ['0 degree', '15 degree', '30 degree', '45 degree']

    secx_all = np.array([secx_00, secx_15, secx_30, secx_45]).T
    secy_all = np.array([secy_00, secy_15, secy_30, secy_45]).T



    figc_x = cross_plot(secx_all, ylim = csc, lcolors = color_spec, label_flags = label_flags, crop = 0.8)
    figc_y = cross_plot(secy_all, ylim = csb+2, lcolors = color_spec, label_flags = label_flags, crop = 0.8)
    figc_x.savefig('cross_X_crop.eps', format = 'eps', dpi = 200)
    figc_y.savefig('cross_Y_crop.eps', format = 'eps', dpi = 200)

    dsecx_15 = secx_15 - secx_00
    dsecx_30 = secx_30 - secx_00
    dsecx_40 = secx_40 - secx_00
    dsecx_45 = secx_45 - secx_00

    dsecy_15 = secy_15 - secy_00
    dsecy_30 = secy_30 - secy_00
    dsecy_40 = secy_40 - secy_00
    dsecy_45 = secy_45 - secy_00

    dsecx_all = np.array([dsecx_15, dsecx_30, dsecx_45]).T
    dsecy_all = np.array([dsecy_15, dsecy_30, dsecy_45]).T

    figc_x  = cross_plot(dsecx_all, ylim = csc, lcolors = color_spec[1:], label_flags = label_flags[1:], crop = 0.8)
    figc_y  = cross_plot(dsecy_all, ylim = csb+2, lcolors = color_spec[1:], label_flags = label_flags[1:], crop = 0.8)
    figc_x.savefig('diffcross_X_crop.eps', format = 'eps', dpi = 200)
    figc_y.savefig('diffcross_Y_crop.eps', format = 'eps', dpi = 200)

    psecx_00 = secx_00 # for zero-tilt, no CL correction is needed.
    psecx_15 =secx_15-0.5*psec_corr
    psecx_30 =secx_30-0.5*psec_corr
    psecx_40 =secx_40-0.5*psec_corr
    psecx_45 =secx_45-2.0*psec_corr
    dpsecx_15 = psecx_15 - psecx_00
    dpsecx_30 = psecx_30 - psecx_00
    dpsecx_40 = psecx_40 - psecx_00
    dpsecx_45 = psecx_45 - psecx_00

    psecx_all = np.array([psecx_15, psecx_30, psecx_45]).T
    dpsecx_all = np.array([dpsecx_15, dpsecx_30, dpsecx_45]).T

    figc_px = cross_plot(psecx_all, ylim = csa+2.5, lcolors = color_spec[1:], label_flags = label_flags[1:], crop = 0.8)
    figc_dpx = cross_plot(dpsecx_all, ylim = csa+2.5, lcolors = color_spec[1:],label_flags = label_flags[1:], crop = 0.8)
    figc_px.savefig('cl_cross_X_crop.eps', format = 'eps', dpi = 200)
    figc_dpx.savefig('dcl_cross_X_crop.eps', format = 'eps', dpi = 200)

    print('Edit completed! ')


    #------------------------Reflectance

#    fig = IMshow_pupil(T_aver[0], False, c_scale = [0,1], inner_diam = 0.8)
#    fig.tight_layout()
#    plt.savefig('Trans0NA10.eps', format = 'eps', dpi = 200)
#
#    fig = IMshow_pupil(T_aver[1], False, c_scale = [0,1], inner_diam = 0.8)
#    fig.tight_layout()
#    plt.savefig('Trans15NA10.eps', format = 'eps', dpi = 200)
#
#    fig = IMshow_pupil(T_aver[2], False, c_scale = [0,1], inner_diam = 0.8)
#    fig.tight_layout()
#    plt.savefig('Trans30NA10.eps', format = 'eps', dpi = 200)
#
#    fig = IMshow_pupil(T_aver[3], False, c_scale = [0,1], inner_diam = 0.8)
#    fig.tight_layout()
#    plt.savefig('Trans40NA10.eps', format = 'eps', dpi = 200)
#
#    fig = IMshow_pupil(T_aver[4], False, c_scale = [0,1], inner_diam = 0.8, crop = 0.8)
#    fig.tight_layout()
#    plt.savefig('Trans45NA10.eps', format = 'eps', dpi = 200)
#
#    fig = plt.figure(figsize = (4,1.8))
#    ax = fig.add_subplot(111)
#    ax.tick_params(labelsize = 12)
#    ax.plot(kk, T_aver[0][N_radius, :-1], 'steelblue', linewidth = 2)
#    ax.set_ylim([0.,1.1])
#    plt.tight_layout()
#    fig.savefig('Tcross0_NA10.eps', format = 'eps', dpi = 200)
#    plt.cla()
#
#    ax.plot(kk, T_aver[1][N_radius, :-1], 'steelblue', linewidth = 2)
#    ax.set_ylim([0.,1.1])
#    plt.tight_layout()
#    fig.savefig('Tcross15_NA10.eps', format = 'eps', dpi = 200)
#    plt.cla()
#    ax.plot(kk, T_aver[2][N_radius, :-1], 'steelblue', linewidth = 2)
#    ax.set_ylim([0.,1.1])
#    plt.tight_layout()
#    fig.savefig('Tcross30_NA10.eps', format = 'eps', dpi = 200)
#    plt.cla()
#    ax.plot(kk, T_aver[3][N_radius, :-1], 'steelblue', linewidth = 2)
#    ax.set_ylim([0.,1.1])
#    plt.tight_layout()
#    fig.savefig('Tcross40_NA10.eps', format = 'eps', dpi = 200)
#    plt.cla()
#    ax.plot(kk, T_aver[4][N_radius, :-1], 'steelblue', linewidth = 2)
#    ax.plot([-0.8, -0.8], [0, 1], '-.k', linewidth = 1)
#    ax.plot([0.8, 0.8], [0, 1], '-.k', linewidth = 1)
#    ax.set_ylim([0.,1.1])
#    plt.tight_layout()
#    fig.savefig('Tcross45_NA10.eps', format = 'eps', dpi = 200)
#
if __name__ == '__main__':
    main()
