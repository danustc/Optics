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

def tilt_correct(cr, kk = None):
    # fit the cross section with tilt
    if kk is None:
        kk = np.arange(len(cr))
    m,b = np.polyfit(kk, cr,1)
    cr_corr = (cr - m*kk+b)
    return cr_corr, m, b


def cross_plot(cr_group, ylim, krange, label_flags, lcolors = ['r', 'g', 'b', 'c'], NA=1.0, fsize = (5,3.5), crop = None):
    '''
    do the cross_section plot
    cr_group: the array of cross_plot
    '''
    figc = plt.figure(figsize = fsize)
    ax=figc.add_subplot(111)
    ax.plot(krange, cr_group, linewidth = 2)
    for i,j in enumerate(ax.lines):
        print(lcolors[i])
        j.set_color(lcolors[i])

    #ax.legend(label_flags, fontsize = 12)
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
    results = np.load('results_0removed_xtilt.npz')
    N_radius = 64
    xx = np.arange(-N_radius, N_radius)+0.5
    OPD = results['OPD']
    RSM = results['rsm']
    RPM = results['rpm']
    vmn = -8
    vmx = 10

    T_aver = 1-(RSM+RPM) + 0.5*(RSM**2+RPM**2)
    plt.close('all')
    csa = np.array([-6, 5])
    csb = np.array([-4, 3])
    csc = np.array([-7, 4])
    color_spec = [color_dic['darkorchid'], color_dic['orange'],color_dic['steelblue'], color_dic['limegreen']]

    pupil_00 = OPD[0]
    pupil_15 = OPD[1]
    pupil_35 = OPD[2]
    pupil_40 = OPD[3]
    pupil_45 = OPD[4]


    fig = IMshow_pupil(OPD[0], False, c_scale = csa, inner_diam = 0.8)
    fig.tight_layout()
    plt.savefig('T0NA10.eps', format = 'eps', dpi = 200)


    fig = IMshow_pupil(OPD[1], False, c_scale = csa, inner_diam = 0.8)
    fig.tight_layout()
    plt.savefig('T15NA10.eps', format = 'eps', dpi = 200)

    fig = IMshow_pupil(OPD[2], False,c_scale = csa, inner_diam = 0.8)
    fig.tight_layout()
    plt.savefig('T35NA10.eps', format = 'eps', dpi = 200)

    fig = IMshow_pupil(OPD[3], False,c_scale = csa, inner_diam = 0.8, crop = 0.8)
    fig.tight_layout()
    plt.savefig('T40NA10.eps', format = 'eps', dpi = 200)

    fig = IMshow_pupil(OPD[4], False,c_scale = csa, inner_diam = 0.8, crop = 0.8)
    fig.tight_layout()
    plt.savefig('T45NA10.eps', format = 'eps', dpi = 200)

    OPD_base = OPD[0]

    fig = IMshow_pupil(OPD[1]-OPD_base, False, c_scale = csb, inner_diam = 0.8)
    fig.tight_layout()
    plt.savefig('diff_T15NA10.eps', format = 'eps', dpi = 200)

    fig = IMshow_pupil(OPD[2]-OPD_base, False, c_scale = csb, inner_diam = 0.8)
    fig.tight_layout()
    plt.savefig('diff_T35NA10.eps', format = 'eps', dpi = 200)

    fig = IMshow_pupil(OPD[3]-OPD_base, False, c_scale = csb, inner_diam = 0.8, crop = 0.8)
    fig.tight_layout()
    plt.savefig('diff_T40NA10.eps', format = 'eps', dpi = 200)

    fig = IMshow_pupil(OPD[4]-OPD_base, False, c_scale = csb, inner_diam = 0.8, crop = 0.8)
    fig.tight_layout()
    plt.savefig('diff_T45NA10.eps', format = 'eps', dpi = 200)

    alpha_para = -10.0
    kk = xx[1:-1]/N_radius
    kk_crop = kk[int(0.2*N_radius)+1:int(1.8*N_radius)-2]
    psec_corr = alpha_para*(kk+0.07)**2 # the parabolic mirror 
    psec_corr_crop = alpha_para*(kk_crop+0.1)**2
    # construct the cross section 
    secx_00_raw = OPD[0][N_radius, 1:-1]#-OPD[0][N_radius, N_radius]
    secx_15_raw = OPD[1][N_radius, 1:-1]#-OPD[1][N_radius, N_radius]
    secx_35_raw = OPD[2][N_radius, 1:-1]#-OPD[2][N_radius, N_radius]
    secx_40_raw = OPD[3][N_radius, 1:-1]#-OPD[3][N_radius, N_radius]
    secx_45_raw = OPD[4][N_radius, 1:-1]#-OPD[4][N_radius, N_radius]
    secx_40_crop = secx_40_raw[int(0.2*N_radius)+1:int(1.8*N_radius)-2]
    secx_45_crop = secx_45_raw[int(0.2*N_radius)+1:int(1.8*N_radius)-2]

    secy_00_raw = OPD[0][1:-1, N_radius]#-OPD[0][N_radius, N_radius]
    secy_15_raw = OPD[1][1:-1, N_radius]#-OPD[1][N_radius, N_radius]
    secy_35_raw = OPD[2][1:-1, N_radius]#-OPD[2][N_radius, N_radius]
    secy_40_raw = OPD[3][1:-1, N_radius]#-OPD[3][N_radius, N_radius]
    secy_45_raw = OPD[4][1:-1, N_radius]#-OPD[4][N_radius, N_radius]
    secy_40_crop = secy_40_raw[int(0.2*N_radius)+1:int(1.8*N_radius)-2]
    secy_45_crop = secy_45_raw[int(0.2*N_radius)+1:int(1.8*N_radius)-2]

    secx_00 = secx_00_raw - secx_00_raw[N_radius]
    secx_15 = secx_15_raw - secx_15_raw[N_radius]
    secx_35 = secx_35_raw - secx_35_raw[N_radius]
    secx_40 = secx_40_crop - secx_40_raw[N_radius]
    secx_45 = secx_45_crop - secx_45_raw[N_radius]

    secy_00 = secy_00_raw - secy_00_raw[N_radius]
    secy_15 = secy_15_raw - secy_15_raw[N_radius]
    secy_35 = secy_35_raw - secy_35_raw[N_radius]
    secy_40 = secy_40_crop - secy_40_raw[N_radius]
    secy_45 = secy_45_crop - secy_45_raw[N_radius]


    label_flags = [r"$0^\circ$", r"$15^\circ$", r"$35^\circ$", r"$45^\circ$"]

    secx_all = np.array([secx_00, secx_15, secx_35]).T
    secy_all = np.array([secy_00, secy_15, secy_35]).T


    figc_x = cross_plot(secx_all, ylim = csc, krange = kk, lcolors = color_spec, label_flags = label_flags)
    ax = figc_x.get_axes()[0]
    ax.plot(kk_crop, secx_45, color = color_spec[-1])
    ax.legend(label_flags, fontsize = 12)
    ax.plot([-0.8, -0.8], [-10, 10], '-.k', linewidth = 1)
    ax.plot([0.8, 0.8], [-10, 10], '-.k', linewidth = 1)
    figc_x.savefig('cross_X.eps', format = 'eps', dpi = 200)


    figc_y = cross_plot(secy_all, ylim = csb+2, krange = kk, lcolors = color_spec, label_flags = label_flags)
    ay = figc_y.get_axes()[0]
    ay.plot(kk_crop, secy_45, color = color_spec[-1])
    ay.legend(label_flags, fontsize = 12)
    ay.plot([-0.8, -0.8], [-10, 10], '-.k', linewidth = 1)
    ay.plot([0.8, 0.8], [-10, 10], '-.k', linewidth = 1)
    figc_y.savefig('cross_Y.eps', format = 'eps', dpi = 200)

    crop_begin = int(0.2*N_radius)+1
    crop_end = int(1.8*N_radius)-2
    dsecx_15_raw = secx_15_raw - secx_00_raw
    dsecx_35_raw = secx_35_raw - secx_00_raw
    dsecx_40_raw = secx_40_raw - secx_00_raw
    dsecx_45_raw = secx_45_raw - secx_00_raw
    dsecx_40_crop = dsecx_40_raw[int(0.2*N_radius)+1:int(1.8*N_radius)-2]
    dsecx_45_crop = dsecx_45_raw[int(0.2*N_radius)+1:int(1.8*N_radius)-2]


    dsecx_15 = dsecx_15_raw - dsecx_15_raw[N_radius]
    dsecx_35 = dsecx_35_raw - dsecx_35_raw[N_radius]
    dsecx_40 = dsecx_40_crop - dsecx_40_raw[N_radius]
    dsecx_45 = dsecx_45_crop - dsecx_45_raw[N_radius]

    dsecy_15_raw = secy_15_raw - secy_00_raw
    dsecy_35_raw = secy_35_raw - secy_00_raw
    dsecy_40_raw = secy_40_raw - secy_00_raw
    dsecy_45_raw = secy_45_raw - secy_00_raw

    dsecy_15 = dsecy_15_raw - dsecy_15_raw[N_radius]
    dsecy_35 = dsecy_35_raw - dsecy_35_raw[N_radius]
    dsecy_40_crop = dsecy_40_raw[int(0.2*N_radius)+1:int(1.8*N_radius)-2]
    dsecy_45_crop = dsecy_45_raw[int(0.2*N_radius)+1:int(1.8*N_radius)-2]
    dsecy_40 = dsecy_40_crop - dsecy_40_raw[N_radius]
    dsecy_45 = dsecy_45_crop - dsecy_45_raw[N_radius]


    dsecx_all = np.array([dsecx_15, dsecx_35]).T
    dsecy_all = np.array([dsecy_15, dsecy_35]).T

    figc_x  = cross_plot(dsecx_all, ylim = csc, krange = kk, lcolors = color_spec[1:], label_flags = label_flags[1:] )
    ax = figc_x.get_axes()[0]
    ax.plot(kk_crop, dsecx_45, color = color_spec[-1])
    ax.legend(label_flags[1:], fontsize = 12)
    ax.plot([-0.8, -0.8], [-10, 10], '-.k', linewidth = 1)
    ax.plot([0.8, 0.8], [-10, 10], '-.k', linewidth = 1)
    figc_x.savefig('diffcross_X.eps', format = 'eps', dpi = 200)

    figc_y  = cross_plot(dsecy_all, ylim = csb+2, krange = kk, lcolors = color_spec[1:], label_flags = label_flags[1:] )
    ay = figc_y.get_axes()[0]
    ay.plot(kk_crop, dsecy_45, color = color_spec[-1])
    ay.legend(label_flags[1:], fontsize = 12)
    ay.plot([-0.8, -0.8], [-10, 10], '-.k', linewidth = 1)
    ay.plot([0.8, 0.8], [-10, 10], '-.k', linewidth = 1)
    figc_y.savefig('diffcross_Y.eps', format = 'eps', dpi = 200)
    # now, time for cylindrical lens
    psecx_00 = secx_00 # for zero-tilt, no CL correction is needed.
    psecx_15_raw = secx_15-0.5*psec_corr
    psecx_35_raw = secx_35-0.5*psec_corr
    psecx_40_raw = secx_40-0.5*psec_corr_crop
    psecx_45_raw = secx_45-1.0*psec_corr_crop

    psecx_15 = psecx_15_raw - psecx_15_raw[N_radius]
    psecx_35 = psecx_35_raw - psecx_35_raw[N_radius]
    psecx_40 = psecx_40_raw - psecx_40_raw[int(0.8*N_radius)]
    psecx_45 = psecx_45_raw - psecx_45_raw[int(0.8*N_radius)]

    dpsecx_15 = psecx_15 - psecx_00
    dpsecx_35 = psecx_35 - psecx_00
    dpsecx_40 = psecx_40 - psecx_00[int(0.2*N_radius)+1:int(1.8*N_radius)-2]
    dpsecx_45 = psecx_45 - psecx_00[int(0.2*N_radius)+1:int(1.8*N_radius)-2]

    psecx_all = np.array([psecx_15, psecx_35 ]).T
    dpsecx_all = np.array([dpsecx_15, dpsecx_35]).T

    figc_px = cross_plot(psecx_all, ylim = csa, krange = kk, lcolors = color_spec[1:], label_flags = label_flags[1:])
    ax = figc_px.get_axes()[0]
    ax.plot(kk_crop, psecx_45, color = color_spec[-1])
    ax.legend(label_flags[1:], fontsize = 12)
    ax.plot([-0.8, -0.8], [-10, 10], '-.k', linewidth = 1)
    ax.plot([0.8, 0.8], [-10, 10], '-.k', linewidth = 1)
    figc_px.savefig('cl_cross_X.eps', format = 'eps', dpi = 200)

    figc_dpx = cross_plot(dpsecx_all, ylim = csa, krange = kk, lcolors = color_spec[1:],label_flags = label_flags[1:] )
    ax = figc_dpx.get_axes()[0]
    ax.plot(kk_crop, dpsecx_45, color = color_spec[-1])
    ax.legend(label_flags[1:], fontsize = 12)
    ax.plot([-0.8, -0.8], [-10, 10], '-.k', linewidth = 1)
    ax.plot([0.8, 0.8], [-10, 10], '-.k', linewidth = 1)
    figc_dpx.savefig('dcl_cross_X.eps', format = 'eps', dpi = 200)

    print('Edit completed! ')

    plt.close('all') #add this line if you have opened too many figures.

    #------------------------Reflectance--------------------------------------

    fig = IMshow_pupil(T_aver[0], False, c_scale = [0,1], inner_diam = 0.8, cbo = 'vertical')
    fig.tight_layout()
    plt.savefig('Trans0NA10.eps', format = 'eps', dpi = 200)

    fig = IMshow_pupil(T_aver[1], False, c_scale = [0,1], inner_diam = 0.8, cbo = 'vertical')
    fig.tight_layout()
    plt.savefig('Trans15NA10.eps', format = 'eps', dpi = 200)

    fig = IMshow_pupil(T_aver[2], False, c_scale = [0,1], inner_diam = 0.8, cbo = 'vertical')
    fig.tight_layout()
    plt.savefig('Trans35NA10.eps', format = 'eps', dpi = 200)

    fig = IMshow_pupil(T_aver[3], False, c_scale = [0,1], inner_diam = 0.8, cbo = 'vertical')
    fig.tight_layout()
    plt.savefig('Trans40NA10.eps', format = 'eps', dpi = 200)

    fig = IMshow_pupil(T_aver[4], False, c_scale = [0,1], inner_diam = 0.8, crop = 0.8, cbo = 'vertical')
    fig.tight_layout()
    plt.savefig('Trans45NA10.eps', format = 'eps', dpi = 200)

    fig = plt.figure(figsize = (4,1.8))
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize = 12)
    ax.plot(kk, T_aver[0][N_radius, 1:-1], 'steelblue', linewidth = 2)
    ax.set_xlim([-1,1])
    ax.set_ylim([0.,1.1])
    plt.tight_layout()
    fig.savefig('Tcross0_NA10.eps', format = 'eps', dpi = 200)
    plt.cla()

    ax.plot(kk, T_aver[1][N_radius, 1:-1], 'steelblue', linewidth = 2)
    ax.set_xlim([-1,1])
    ax.set_ylim([0.,1.1])
    plt.tight_layout()
    fig.savefig('Tcross15_NA10.eps', format = 'eps', dpi = 200)
    plt.cla()
    ax.plot(kk, T_aver[2][N_radius, 1:-1], 'steelblue', linewidth = 2)
    ax.set_xlim([-1,1])
    ax.set_ylim([0.,1.1])
    plt.tight_layout()
    fig.savefig('Tcross35_NA10.eps', format = 'eps', dpi = 200)
    plt.cla()
    ax.plot(kk_crop, T_aver[3][N_radius, crop_begin:crop_end], 'steelblue', linewidth = 2)
    ax.set_xlim([-1,1])
    ax.set_ylim([0.,1.1])
    plt.tight_layout()
    fig.savefig('Tcross40_NA10.eps', format = 'eps', dpi = 200)
    plt.cla()
    ax.plot(kk_crop, T_aver[4][N_radius, crop_begin:crop_end], 'steelblue', linewidth = 2)
    ax.plot([-0.8, -0.8], [0, 1], '-.k', linewidth = 1)
    ax.plot([0.8, 0.8], [0, 1], '-.k', linewidth = 1)
    ax.set_xlim([-1,1])
    ax.set_ylim([0.,1.1])
    plt.tight_layout()
    fig.savefig('Tcross45_NA10.eps', format = 'eps', dpi = 200)

if __name__ == '__main__':
    main()
