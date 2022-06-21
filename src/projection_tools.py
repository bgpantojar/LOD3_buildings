#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:22:51 2020

This script contains the codes to generate LOD3 models.
The codes are based on "Generation LOD3 models from structure-from-motion and semantic segmentation" 
by Pantoja-Rosero et., al.
https://doi.org/10.1016/j.autcon.2022.104430

@author: pantoja
"""

import numpy as np
import cv2
import json
from sklearn.linear_model import LinearRegression

#Function to undistort points 
def undistort_points(points2d, K, k_dist):
    """
    Find image coordinates when undistorted

    Args:
        points2d (array): image point coordinates on distorted image
        K (array): intrinsic camera matrix 
        k_dist (array): distortion parameters

    Returns:
        points2d_undist (array): image point coordinates on undistorted image
    """
    
    points2d_ = points2d[:, 0:2].astype('float32')
    points2d_ = np.expand_dims(points2d_, axis=1)  # (n, 1, 2)
    
    distCoef = np.array([0., k_dist[0], 0., 0., 0., k_dist[1], 0., k_dist[2]], dtype=np.float32) 
    
    result = np.squeeze(cv2.undistortPoints(points2d_, K, distCoef))
    if len(result.shape) == 1:  # only a single point
        result = np.expand_dims(result, 0)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    points2d_undist = np.empty_like(points2d)
    for i, (px, py) in enumerate(result):
        points2d_undist[i, 0] = px * fx + cx
        points2d_undist[i, 1] = py * fy + cy
        points2d_undist[i, 2] = points2d[i, 2]

    return points2d_undist 

def load_intrinsics_poses(cameras_path):
    """
    Loads intrinsic and poses informations from sfm output

    Args:
        cameras_path (str): path to folder containing cameras' file from sfm

    Returns:
        intrinsic (dict) : camera intrinsic information
        poses (dict) : camera poses information
    """
    with open(cameras_path +'cameras.sfm', 'r') as fp:
        cameras = json.load(fp)
    
    v = cameras['views']
    i = cameras['intrinsics']
    p = cameras['poses']
    
    #If there are more views than poses, delete extra views(I suppose are those images not taken in SfM by meshroom) 
    iii=0
    while iii<len(p):
        if p[iii]['poseId']==v[iii]['poseId']:
            iii+=1
        else:
            v.remove(v[iii])
    
    
    k_intrinsic = {'pxInitialFocalLength','pxFocalLength', 'principalPoint',
               'distortionParams'}
    intrinsic = {}
    for ii in i:
        key = ii['intrinsicId']
        intrinsic[key] = {}
        for l in k_intrinsic:
            intrinsic[key][l] = ii[l]

    k_poses = {'poseId', 'intrinsicId', 'path', 'rotation', 'center'}
    poses = {}
    for l, view in enumerate(v):
        key = view['path'].split('/')[-1][:-4]
        poses[key] = {}
        for m in k_poses:
           if v[l]['poseId']==p[l]['poseId']:
               if m in v[l]:
                   poses[key][m] = v[l][m]
               else:
                   poses[key][m] = p[l]['pose']['transform'][m]
           else:
                   print("Error: views and poses are not correspondences")
    
   
    return intrinsic, poses

def fast_homography(imm1, imm2, im1_n, im2_n, data_folder, mask_facade = None, save_im_kps = False):
    """
    Find HOMOGRAPHY with opencv 

    Args:
        imm1 (array): image1 
        imm2 (array): image2 
        im1_n (str): image1 name
        im2_n (str): image2 name
        data_folder (str): data folder name
        mask_facade (array, optional): binary mask with segmented facade. Defaults to None.
        save_im_kps (bool, optional): If true it saves the kps. Defaults to False.

    Returns:
        H (array): homography matrix
    """
    
    imm_1 = np.copy(imm1)
    imm_2 = np.copy(imm2)
    
    gray1 = cv2.cvtColor(imm_1,cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(imm_2,cv2.COLOR_RGB2GRAY)
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray1,None)
    kp2, des2 = sift.detectAndCompute(gray2,None)
    
    #saving image with kps detected
    if save_im_kps:
        im_kp1 = cv2.drawKeypoints(gray1,kp1,imm_1,(0,0,255))
        im_kp2 = cv2.drawKeypoints(gray2,kp2,imm_2,(0,0,255))
        cv2.imwrite('../results/' + data_folder + '/' + im1_n+"_init_kps.png", im_kp1)
        cv2.imwrite('../results/' + data_folder + '/' + im2_n+"_init_kps.png", im_kp2)
        
    
    #If is given mask_facade, it will filter the kp1 on im1 to those that are over the segemnted facade
    if mask_facade is not None:
        kp1_filtered = []
        des1_filtered = []
        for kp, des in zip(kp1,des1):
            if mask_facade[int(kp.pt[1]), int(kp.pt[0])] == 1:
                kp1_filtered.append(kp)
                des1_filtered.append(des)
        kp1 = kp1_filtered
        des1 = np.array(des1_filtered)
    
    if mask_facade is not None and save_im_kps:
        im_kp1_f = cv2.drawKeypoints(gray1,kp1,imm_1,(0,0,255))
        cv2.imwrite('../results/' + data_folder + '/' + im1_n+"_init_kps_filtered.png", im_kp1_f)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
     
    if len(good)>10:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4)
        matchesMask = mask.ravel().tolist()
        print("Homography", H)
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    img3 = cv2.drawMatches(imm_1,kp1,imm_2,kp2,good,None,**draw_params)
    
    if save_im_kps:
        cv2.imwrite('../results/' + data_folder + '/' + im1_n+"_matches.png", img3)    
    
    return H


def camera_matrices(intrinsic, poses):
    """
    Calculate Camera Matrix using intrinsic and extrinsic parameters P = K[R|t]

    Args:
        intrinsic (dict) : camera intrinsic information
        poses (dict) : camera poses information

    Returns:
        P (list): camera matrices' list
        K (list): intrinsic matrices' list
        k_dist (list): distorsion parameters' list
    """
    

    K = {}
    k_dist = {}
    for ii in intrinsic:
        #Intrinsic Parameters
        f = float(intrinsic[ii]["pxFocalLength"])
        px = float(intrinsic[ii]["principalPoint"][0])
        py = float(intrinsic[ii]["principalPoint"][1])
        k_dist[ii] = np.ndarray.astype(np.array((intrinsic[ii]["distortionParams"])),float)
        
        K[ii]= np.array([[f,0.,px],
                         [0.,f,py],
                         [0.,0.,1.]])
    
    #Extrinsic Parameters
    R = {}
    t = {}
    C = {}
    
    for key in poses:
        R[key] = (np.float_(poses[key]["rotation"])).reshape((3,3)).T
        C = np.float_(poses[key]["center"]).reshape((3,1))
        t[key] = np.dot(-R[key],C)
    
    #List with camera matrices p
    P = {}
    for key in poses:
        P[key] = {}
        P[key]['P'] = np.dot(K[poses[key]['intrinsicId']],np.concatenate((R[key],t[key]),axis=1))
        P[key]['intrinsicId'] = poses[key]['intrinsicId']
        
    return P, K, k_dist


def line_adjustor(x_lr,y_lr, smooth =1e-13):
    """
    Adjusts line in a set of points and project them later
    to the line
    Args:
        x_lr (array): x coordinates for points
        y_lr (array): x coordinates for points
        smooth (float, optional): smoothing value to avoid zero division. Defaults to 1e-13.

    Returns:
        X_proj (array): x and y projected coordinates to line
    """
    x_lr = x_lr.reshape((-1,1))
    model_lr = LinearRegression()
    model_lr.fit(x_lr,y_lr)
    
    y_adj = model_lr.predict(x_lr)
    
    yu = model_lr.predict(x_lr)
    yv = model_lr.predict(x_lr+.1)
    
    #project the point to the line regretion
    #      /X        
    #     / |
    #    /  |
    # e2/   |
    #  /a   |
    # u--v---P--e1
    X_proj = np.zeros((len(x_lr),2))
    for i, x in enumerate(x_lr):
        u = np.array([x_lr[i], yu[i]])
        v = np.array([x_lr[i]+.1, yv[i]])
        X = np.array([x_lr[i], y_lr[i]])
        
        e1 = (v-u)/(np.linalg.norm(v-u)+smooth)
        e2 = (X-u)/(np.linalg.norm(X-u)+smooth)
        
        Pu = (np.dot(e1,e2)) * np.linalg.norm(X-u)
        P = u + Pu*e1
        X_proj[i,:] = P
    
    
    return X_proj[:,0], X_proj[:,1]