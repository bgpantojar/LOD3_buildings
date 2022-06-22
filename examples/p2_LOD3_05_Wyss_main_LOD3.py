#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 20:06:06 2020
This script contains the codes to generate LOD3 models.
The codes are based on "Generation LOD3 models from structure-from-motion and semantic segmentation" 
by Pantoja-Rosero et., al.
https://doi.org/10.1016/j.autcon.2022.104430
The lines of codes allows to replicate results obtained for the paper.
@author: pantoja
"""
import sys
sys.path.append("../src")
import Part
import FreeCAD
from FreeCAD import Base
import importOBJ
from opening_builder import *

##################################USER INTERACTION#############################
###############USER INPUT
data_folder =  'p2_LOD3_05_Wyss'
data_path = "../results/" + data_folder 
polyfit_path = "../data/" + data_folder + "/polyfit" 
#############USER CALLING FUNCTIONS
print("Building LOD3")
#Builder
#op_builder(data_folder, data_path, polyfit_path) #sparse
op_builder(data_folder, data_path, polyfit_path, dense=True) #dense