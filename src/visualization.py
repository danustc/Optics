'''
updated by Dan on 12/15
'''

import matplotlib.pyplot as plt
import seaborn

def IMshow_pupil(pupil, axnum = True):
    '''
    display a pupil function in 2D
    '''
    NY, NX = pupil.shape
    ry = int(NY/2.)
    rx = int(NX/2.)
    yy = (np.arange(NY)-ry)/ry
    xx = (np.arange(NX)-rx)/rx
    [MX, MY] = np.meshgrid(xx,yy)
    fig = plt.figure(figsize=(7.5,5.8))
    ax = fig.add_subplot(111)
    ax.set_ylim([-N_radius, N_radius])
    ax.set_xlim([-N_radius, N_radius])
    if (axnum == False):
        ax.get_yaxis.set_visible(False)
        ax.get_xaxis.set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    pcm = ax.pcolor(MX, MY, pupil, cmap = 'RdYlBu_r')
    fig.colorbar(pcm, ax = ax, extend='max')

    return fig  # return the figure handle
