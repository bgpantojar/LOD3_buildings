#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 12:43:45 2020

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
from numpy import *

#Function to convert coordinates to homogeneus
def make_homog(points):
    """ Convert a set of points (dim*n array) to
    homogeneous coordinates. """

    return vstack((points,ones((1,points.shape[1]))))