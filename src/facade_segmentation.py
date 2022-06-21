#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:46:06 2020

This script contains the codes to generate LOD3 models.
The codes are based on "Generation LOD3 models from structure-from-motion and semantic segmentation" 
by Pantoja-Rosero et., al.
https://doi.org/10.1016/j.autcon.2022.104430


@author: pantoja
"""


import os
import numpy as np
import cv2 as cv
import torch
from math import sqrt as sqrt
from cnn.foc.src.data_set import open_dataset
from cnn.foc.src.network import U_Net16
from torch.utils.data import DataLoader

def fac_segmentation(data_folder, images_path, out_return=False):
    """
    Uses deep learning model to semmantically segment facade

    Args:
        data_folder (str): input data folder path
        images_path (str): input images folder path
        out_return (bool, optional): if true it returns the binary mask. Defaults to False.

    Returns:
        prediction1_bin_resized (array): binary mask that represent segmented facade
    """

    print("Predicting facades with semantic segmentation...")
       
    # model path
    model = 'model_p2/'
    model_path = '../weights/facade/' + model
    model_path = model_path + os.listdir(model_path)[0]
    image_size = 256

    #data loaders to use deep learning models
    test_ds=open_dataset(images_path, transform='test')
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # load a trained model
    model = U_Net16(num_classes=1)
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model=model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.train(False)
    model.eval()
    
    #Dictionaries to storage prediction and images information    
    prediction1_bin = {}
    prediction1_rgb = {}
    
    #CNN inference
    for ni, image in enumerate(test_dl):
        image = image.to(device)
        SR = model(image)
        SR_probs = torch.sigmoid(SR)
        SR_probs_arr = SR_probs.detach().numpy().reshape(image_size, image_size)
        binary_result = SR_probs_arr > .5
        image_numpy = image.detach().numpy()
        image_numpy = image_numpy[0, 0, :, :]
        image_name = test_dl.dataset.path2imgs[ni].split('/')
        image_name = image_name[-1].split(".")
        image_name = image_name[0]
        facade = np.array(binary_result, dtype='uint8')
        prediction1_bin[image_name] = facade*255
        shp = facade.shape
        facade_rgb = np.zeros((shp[0], shp[1], 3), dtype='uint8')
        facade_rgb[:,:,2] = facade*255
        prediction1_rgb[image_name] = facade_rgb
    
    #Getting original images
    list_images = os.listdir(images_path)
    images = {}
    for img in list_images:
        images[img[:-4]] = cv.imread(images_path + img)
    
    #Resizing predictions to the original size
    prediction1_bin_resized = {}
    prediction1_rgb_resized = {}
    
    for key in images:
        prediction1_bin_resized[key] = cv.resize(prediction1_bin[key], (images[key].shape[1],images[key].shape[0]), interpolation = cv.INTER_CUBIC)
        prediction1_rgb_resized[key] = cv.resize(prediction1_rgb[key], (images[key].shape[1],images[key].shape[0]), interpolation = cv.INTER_CUBIC)
            
    #Overlaying prediction_rgb with original image
    overlayed_prediction1 = {}

    for key in images:
        overlayed_prediction1[key] = cv.addWeighted(images[key], 1.0, prediction1_rgb_resized[key], 0.7, 0)

    #saving binary and overlayed predictions
    #Check if directory exists, if not, create it
    check_dir = os.path.isdir('../results/' + data_folder)
    if not check_dir:
        os.makedirs('../results/' + data_folder)
    
    for key in images:
        cv.imwrite('../results/' + data_folder + '/pred_bin1_' + key + '.png', prediction1_bin_resized[key])
        cv.imwrite('../results/' + data_folder + '/' + key + '.png', images[key])
        cv.imwrite('../results/' + data_folder + '/' + key + '_overlayed1.png', overlayed_prediction1[key])
    
    if out_return:
        return prediction1_bin_resized