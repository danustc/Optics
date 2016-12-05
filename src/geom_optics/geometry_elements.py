'''
This file simulates the planes, lines in the Euclid space, and does the simplest vector analysis.
Created by Dan on 12/04/2016.
'''

import numpy as np

def plane_line_intersect(n_plane, r_plane, k_line, r_line):
    '''
    n_plane: the normal vector of the plane
    r_plane: the coordinates of a point that the plane contains
    k_line: the directional vector of the line
    r_line: the coordinates of a point that the line contains
    '''
    nx, ny, nz = n_plane
    kx, ky, kz = k_line
    lx, ly, lz = r_line
    M = np.array([[kz, 0., -kx], [0., kz, -ky], [nx, ny, nz]]) # determine the coefficient matrix
    b = np.array([kz*lx - kx*lz, kz*ly - ky*lz, np.dot(r_plane, n_plane)])
    r_inter = np.linalg.solve(M, b)
    return r_inter
