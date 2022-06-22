#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:46:06 2020
This script contains the codes to generate LOD3 models.
The codes are based on "Generation LOD3 models from structure-from-motion and semantic segmentation" 
by Pantoja-Rosero et., al.
https://doi.org/10.1016/j.autcon.2022.104430
The lines of codes allows to replicate results obtained for the paper.
@author: pantoja
"""
import sys
sys.path.append("../src")
from openings_projector import op_projector, camera_matrices, load_intrinsics_poses
from opening_detector import main_op_detector

##################################USER INTERACTION#############################
###############USER INPUT
data_folder = 'p2_LOD3_04_Paiano'
bound_box = 'region' 
op_det_nn = 'unet' 
images_path = '../data/' + data_folder + '/images/'
cameras_path = '../data/' + data_folder + '/cameras/'
keypoints_path = '../data/' + data_folder + '/keypoints/'
polyfit_path = '../data/' + data_folder + '/polyfit/'

#For running example where one of the facades fails due to wrong H computation un-comment next 3 lines
how2get_kp = 0 #it can be 0: using detectors, 1: loading npy arrrays, 2: by clicking in the image, 3:im1:numpy arrays, im2:homography, 4: LKT
im1 = ('DJI_0347','DJI_0355_c','DJI_0365', 'DJI_0376') #p2_LOD3_H_fails
im2 = ('DJI_0346','DJI_0354','DJI_0363', 'DJI_0375') # p2_LOD3_H_fails

#For running example where the facade openings are correct using LKT instead of H un-comment next 3 lines
#how2get_kp = 4 #it can be 0: using detectors, 1: loading npy arrrays, 2: by clicking in the image, 3:im1:numpy arrays, im2:homography, 4: LKT
#im1 = ('DJI_0365',) #KLT
#im2 = ('DJI_0363',) #KLT

#############USER CALLING FUNCTIONS
#Detector
if how2get_kp == 0 or how2get_kp == 4:
    opening_4points = main_op_detector(data_folder, images_path, bound_box, op_det_nn, cte_fil4 = .2)    
else:
    opening_4points = None
#Projector
intrinsic, poses = load_intrinsics_poses(cameras_path)
P, K, k_dist = camera_matrices(intrinsic, poses)
op_projector(data_folder, images_path, keypoints_path, polyfit_path, how2get_kp, im1, im2, P, K, k_dist, opening_4points, ctes=[.05, .3, .1, .3])
print("Finished Opening Projection")