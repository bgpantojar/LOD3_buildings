#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 01:46:53 2020

This script contains the codes to generate LOD3 models.
The codes are based on "Generation LOD3 models from structure-from-motion and semantic segmentation" 
by Pantoja-Rosero et., al.
https://doi.org/10.1016/j.autcon.2022.104430

This script specifically support LOD3 codes development.
These are based on codes published in:
Solem, J.E., 2012. Programming Computer Vision with Python: Tools and algorithms for analyzing images. " O'Reilly Media, Inc.".

Slightly changes are introduced to addapt to general pipeline


@author: pantoja
"""

from pylab import *
from numpy import *
from scipy import linalg

#function that computes the least squares triangulation of a point pair
def triangulate_point(x1,x2,P1,P2):
    """ Point pair triangulation from 
        least squares solution. """
        
    M = zeros((6,6))
    M[:3,:4] = P1
    M[3:,:4] = P2
    M[:3,4] = -x1
    M[3:,5] = -x2

    U,S,V = linalg.svd(M)
    X = V[-1,:4]

    return X / X[3]

#To triangulate many points, we can add the following convenience function
#This function takes two arrays of points and returns an array of 
#3D coordinates.
def triangulate(x1,x2,P1,P2):
    """    Two-view triangulation of points in 
        x1,x2 (3*n homog. coordinates). """
        
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    X = [ triangulate_point(x1[:,i],x2[:,i],P1,P2) for i in range(n)]
    return array(X).T