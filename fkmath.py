# Author: Victor Augusto Kich
# Github: https://github.com/victorkich
# E-mail: victorkich@yahoo.com.br

import numpy as np

def deg2rad(deg):
    ''' Convert angles from degress to radians
    '''
    return np.pi * deg / 180.0

def rad2deg(rad):
    ''' Converts angles from radians to degress
    '''
    return 180.0 * rad / np.pi

def dh(a, alfa, d, theta):
    ''' Builds the Homogeneous Transformation matrix
        corresponding to each line of the Denavit-Hartenberg
        parameters
    '''
    m = np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alfa),
        np.sin(theta)*np.sin(alfa), a*np.cos(theta)],
        [np.sin(theta), np.cos(theta)*np.cos(alfa),
        -np.cos(theta)*np.sin(alfa), a*np.sin(theta)],
        [0, np.sin(alfa), np.cos(alfa), d],
        [0, 0, 0, 1]
    ])
    return m

def fk(mode, goals):
    ''' Forward Kinematics
    '''
    # Convert angles from degress to radians
    t = [deg2rad(x) for x in goals]
    # Register the DH parameters
    hs = []
    hs.append(dh(0,       -np.pi/2, 4.3,  t[0]))
    if mode >= 2:
        hs.append(dh(0,    np.pi/2, 0.0,  t[1]))
    if mode >= 3:
        hs.append(dh(0,   -np.pi/2, 24.3, t[2]))
    if mode == 4:
        hs.append(dh(27.0, np.pi/2, 0.0,  t[3] - np.pi/2))

    m = np.eye(4)
    for h in hs:
        m = m.dot(h)
    return m
