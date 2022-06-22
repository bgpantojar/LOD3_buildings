#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:46:06 2020

@author: pantoja
"""
import sys
sys.path.append("../src")
from openings_projector import op_projector, camera_matrices, load_intrinsics_poses
from opening_detector import main_op_detector
import warnings
warnings.filterwarnings("ignore")

##################################USER INTERACTION#############################
###############USER INPUT
data_folder = 'p2_LOD3_05_Wyss'
bound_box = 'region' 
op_det_nn = 'unet' 
images_path = '../data/' + data_folder + '/images/'
cameras_path = '../data/' + data_folder + '/cameras/'
keypoints_path = '../data/' + data_folder + '/keypoints/'
polyfit_path = '../data/' + data_folder + '/polyfit/'
how2get_kp = 0 #it can be 0: using detectors, 1: loading npy arrrays, 2: by clicking in the image, 3:im1:numpy arrays, im2:homography, 4: LKT
im1 = ('DJI_0439','frame0000','DJI_0415', 'DJI_0452')
im2 = ('DJI_0440','frame0060','DJI_0416','DJI_0425') 
#############USER CALLING FUNCTIONS
#Detector
if how2get_kp == 0 or how2get_kp == 4:
    opening_4points = main_op_detector(data_folder, images_path, bound_box, op_det_nn, cte_fil4 = .1)
else:
    opening_4points = None
#Projector
intrinsic, poses = load_intrinsics_poses(cameras_path)
P, K, k_dist = camera_matrices(intrinsic, poses)
op_projector(data_folder, images_path, keypoints_path, polyfit_path, how2get_kp, im1, im2, P, K, k_dist, opening_4points, ctes=[.05, .4, .1, .3]) 
print("Finished Opening Projection")