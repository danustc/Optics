"""
Created by Dan on 08/12/16
All visualization tools 
"""

import numpy as np
import matplotlib.pyplot as plt 

def plot_2d(arr_2d, x_side = 5., ax_label = False, xy_range = None, flip_y = False):
    """
    plot an 2d array with automatically figured aspect ratio
    ax_label: if False ,turn off all axes labels.
    xy_range:[xleft, xright, yup, ydown] --- This is quite different from matlab!!!
    """
    ny, nx = arr_2d.shape
    
    if xy_range is None:
        sc_x = x_side
        sc_y = x_side*ny/nx
    else:
        rg_x = xy_range[1]-xy_range[0]
        rg_y = np.abs(xy_range[3] - xy_range[2])
        
        sc_x = x_side
        sc_y = x_side* ny*rg_y/(nx*rg_x)
        
    
    fig, ax = plt.figure(figsize= (sc_x, sc_y))
    ax.imshow(arr_2d, extent = xy_range)
    if (ax_label == False): # turn off all the axis tools
        ax.tick_params(
            axis = 'both',
            which = 'both', 
            bottom = 'off',
            top = 'off',
            right = 'off',
            left = 'off',
            labelleft='off',
            labelbottom = 'off')
    
    
     
    return fig
    