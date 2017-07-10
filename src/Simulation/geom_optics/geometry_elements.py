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
    '''
    Let me add some steps to prevent singular solution.
    '''
    row_1 = np.array([ky+kz, kz-kx, -(kx+ky)])
    row_2 = np.array([ky-kz, -(kz+kx), kx+ky])
    b_1 = np.dot(row_1, r_line)
    b_2 = np.dot(row_2, r_line)

    M = np.array([row_1, row_2, [nx, ny, nz]]) # determine the coefficient matrix

    b = np.array([b_1, b_2, np.dot(r_plane, n_plane)])
    r_inter = np.linalg.solve(M, b)
    return r_inter



def cone_to_plane(theta, a_max):
    '''
    convert the cone to plane
    '''
    h = 1./np.tan(a_max)
    sl = h*np.tan(theta)
    return sl
