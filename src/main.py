'''
main test function.
'''

import numpy as np
import matplotlib.pyplot as plt
import geom_optics.Objective_aberration as Objective_aberration
import geom_optics.geometry_elements
import libtim.zern as zern


def main():
    d = 170 # microns, the thickness of the coverslide
    n1 = 1.33
    n2 = 1.50
    N_radius = 64
    a_max  = np.arcsin(1./n1) # maximum incidental angle, unit: radian
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


    g_tilt = np.array([0.01, 15., 30.01, 44.9])*np.pi/180.
    M_sl = Objective_aberration.cone_to_plane(np.pi-M_inc, a_max) # the mapped lateral position

    k_vec = Objective_aberration.incident_vec(np.pi-M_inc, M_rot)
    nz, ny, nx = k_vec.shape
    k_vec = np.reshape(k_vec, (nz, ny*nx)) # reshape the array
    n_vec = Objective_aberration.normal_vec(g_tilt, 0)

    NK = ny*nx
    NN = len(g_tilt)
    lam = 0.550 # the wavelength
    RS_mat = []
    RP_mat = []
    OPD_mat = []


    results = dict()
    results['rsm'] = RSM
    results['rpm'] = RPM
    results['OPD'] = OPD
    np.savez('results', **results)

    # azimuths = np.radians(np.linspace(0, 360, 20))
    # zeniths = np.arange(0, 70, 10)

    # r, theta = np.meshgrid(zeniths, azimuths)
    # values = np.random.random((azimuths.size, zeniths.size))

#-- Plot... ------------------------g------------------------
    plt.close('all')
    fig = plt.figure(figsize=((7.5,5.8)))
    ax = fig.add_subplot(111)
    ax.set_ylim([-N_radius, N_radius])
    ax.set_xlim([-N_radius, N_radius])
    pcm = ax.pcolor(MX, MY, OPD[0], cmap = 'RdYlBu_r')
    ax.plot(rx,ry, '--w')
    fig.colorbar(pcm, ax = ax, extend='max')
    plt.savefig('T0.eps', format = 'eps', dpi = 300)


    fig = plt.figure(figsize=((7.5,5.8)))
    ax = fig.add_subplot(111)
    ax.set_ylim([-N_radius, N_radius])
    ax.set_xlim([-N_radius, N_radius])
    pcm = ax.pcolor(MX, MY, OPD[1], cmap = 'RdYlBu_r')
    ax.plot(rx,ry, '--w')
    fig.colorbar(pcm, ax = ax, extend='max')
    plt.savefig('T15.eps', format = 'eps', dpi = 300)

    fig = plt.figure(figsize=((7.5,5.8)))
    ax = fig.add_subplot(111)
    ax.set_ylim([-N_radius, N_radius])
    ax.set_xlim([-N_radius, N_radius])
    pcm = ax.pcolor(MX, MY, OPD[2], cmap = 'RdYlBu_r')
    ax.plot(rx,ry, '--w')
    fig.colorbar(pcm, ax = ax, extend='max')
    plt.savefig('T30.eps', format = 'eps', dpi = 300)


    fig = plt.figure(figsize=(7.5,5.8))
    ax = fig.add_subplot(111)
    ax.set_ylim([-N_radius, N_radius])
    ax.set_xlim([-N_radius, N_radius])
    pcm = ax.pcolor(MX, MY, OPD[3], cmap = 'RdYlBu_r')
    ax.plot(rx,ry, '--w')
    fig.colorbar(pcm, ax = ax, extend='max')
    plt.savefig('T45.eps', format = 'eps', dpi = 300)

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
    plt.legend()
    plt.xlabel('NA')
    plt.ylabel('Aberration')
    plt.ylim([-20,35])
    # ax.set_label(['0', '15', '30', '45'])
    # ax.plot(xx[:-1]/N_radius, secy_0, '-g')
    plt.savefig('cross_x.eps', format = 'eps', dpi = 300)

    fig = plt.figure(figsize = (7.5, 4))
    plt.plot(kk, secy_0,'-g', linewidth = 2, label = '0 degree')
    plt.plot(kk, secy_15, '-r', linewidth = 2, label = '15 degree')
    plt.plot(kk, secy_30, '-b',linewidth = 2, label = '30 degree')
    plt.plot(kk, secy_45, '--k', linewidth = 2, label = '45 degree')
    plt.legend()
    plt.xlabel('NA')
    plt.ylabel('Aberration')
    plt.ylim([-20,35])
    plt.savefig('cross_y.eps', format = 'eps', dpi = 300)

    #------------------------Reflectance

    fig = plt.figure(figsize=(7.5,5.8))
    ax = fig.add_subplot(111)
    ax.set_ylim([-N_radius, N_radius])
    ax.set_xlim([-N_radius, N_radius])
    pcm = ax.pcolor(MX, MY, T_aver[0], cmap = 'RdYlBu_r', vmin = 0, vmax = 1)
    ax.plot(rx,ry, '--w')
    fig.colorbar(pcm, ax = ax, extend='max')
    plt.savefig('Trans0.eps', format = 'eps', dpi = 300)

    fig = plt.figure(figsize=(7.5,5.8))
    ax = fig.add_subplot(111)
    ax.set_ylim([-N_radius, N_radius])
    ax.set_xlim([-N_radius, N_radius])
    pcm = ax.pcolor(MX, MY, T_aver[1], cmap = 'RdYlBu_r', vmin = 0, vmax = 1)
    ax.plot(rx,ry, '--w')
    fig.colorbar(pcm, ax = ax, extend='max')
    plt.savefig('Trans15.eps', format = 'eps', dpi = 300)

    fig = plt.figure(figsize=(7.5,5.8))
    ax = fig.add_subplot(111)
    ax.set_ylim([-N_radius, N_radius])
    ax.set_xlim([-N_radius, N_radius])
    pcm = ax.pcolor(MX, MY, T_aver[2], cmap = 'RdYlBu_r', vmin = 0, vmax = 1)
    ax.plot(rx,ry, '--w')
    fig.colorbar(pcm, ax = ax, extend='max')
    plt.savefig('Trans30.eps', format = 'eps', dpi = 300)

    fig = plt.figure(figsize=(7.5,5.8))
    ax = fig.add_subplot(111)
    ax.set_ylim([-N_radius, N_radius])
    ax.set_xlim([-N_radius, N_radius])
    pcm = ax.pcolor(MX, MY, T_aver[3], cmap = 'RdYlBu_r', vmin = 0, vmax = 1)
    ax.plot(rx,ry, '--w')
    fig.colorbar(pcm, ax = ax, extend='max')
    plt.savefig('Trans45.eps', format = 'eps', dpi = 300)



if __name__ == '__main__':
    main()
