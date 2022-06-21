#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:56:02 2020

This script contains the codes to generate LOD3 models.
The codes are based on "Generation LOD3 models from structure-from-motion and semantic segmentation" 
by Pantoja-Rosero et., al.
https://doi.org/10.1016/j.autcon.2022.104430

This script specifically support LOD3 codes development. These are based on codes published in:
Solem, J.E., 2012. Programming Computer Vision with Python: Tools and algorithms for analyzing images. " O'Reilly Media, Inc.".

Slightly changes are introduced to addapt to general pipeline


@author: pantoja
"""

from numpy import *
from scipy import linalg

class Camera(object):
    """ Class for representing pin-hole cameras. """
    def __init__(self,P):
        """ Initialize P = K[R|t] camera model. """
        self.P = P
        self.K = None # calibration matrix
        self.R = None # rotation
        self.t = None # translation
        self.c = None # camera center
    
    def project(self,X):
        """    Project points in X (4*n array) and normalize coordinates. """
        x = dot(self.P,X)
        for i in range(3):
            x[i] /= x[2]    
        return x
        
    def factor(self):
        """    Factorize the camera matrix into K,R,t as P = K[R|t]. """
        
        # factor first 3*3 part
        K,R = linalg.rq(self.P[:,:3])
        # make diagonal of K positive
        T = diag(sign(diag(K)))
        if linalg.det(T) < 0:
            T[1,1] *= -1
        
        self.K = dot(K,T)
        self.R = dot(T,R) # T is its own inverse
        self.t = dot(linalg.inv(self.K),self.P[:,3])
        
        return self.K, self.R, self.t
    
    def center(self):
        """    Compute and return the camera center. """
    
        if self.c is not None:
            return self.c
        else:
            # compute c by factoring
            self.factor()
            self.c = -dot(self.R.T,self.t)
            return self.c
