"""
This script contains the codes to generate LOD3 models.
The codes are based on "Generation LOD3 models from structure-from-motion and semantic segmentation" 
by Pantoja-Rosero et., al.
https://doi.org/10.1016/j.autcon.2022.104430

"""

# Import Modules
import os
import argparse
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch

random.seed(100)

#Define the open_dataset class:
class open_dataset(Dataset):
    def __init__(self, path2data_i, path2data_m1=None, path2data_m2=None, path2data_m3=None, transform=None):      

        imgsList=[pp for pp in os.listdir(path2data_i)]
        imgsList.sort()
        if transform!="test":
            anntsList1=[pp for pp in os.listdir(path2data_m1)]
            anntsList1.sort()
            anntsList2=[pp for pp in os.listdir(path2data_m2)]
            anntsList2.sort()
            anntsList3=[pp for pp in os.listdir(path2data_m3)]
            anntsList3.sort()

        self.path2imgs = [os.path.join(path2data_i, fn) for fn in imgsList] 
        
        if transform!="test":
            self.path2annts1= [os.path.join(path2data_m1, fn) for fn in anntsList1]
            self.path2annts2= [os.path.join(path2data_m2, fn) for fn in anntsList2]
            self.path2annts3= [os.path.join(path2data_m3, fn) for fn in anntsList3]

        self.transform = transform
    
    def __len__(self):
        return len(self.path2imgs)
      
    def __getitem__(self, idx):
        path2img = self.path2imgs[idx]
        img = Image.open(path2img)
        #print(img)
        if self.transform!="test":
            path2annt1 = self.path2annts1[idx]
            mask1 = Image.open(path2annt1)
            path2annt2 = self.path2annts2[idx]
            mask2 = Image.open(path2annt2)
            path2annt3 = self.path2annts3[idx]
            mask3 = Image.open(path2annt3)
             
        if self.transform=='train':
                if random.random()<.5:
                    img = TF.hflip(img)
                    mask1 = TF.hflip(mask1)
                    mask2 = TF.hflip(mask2)
                    mask3 = TF.hflip(mask3)
                if random.random()<.5:
                    img = TF.vflip(img)
                    mask1 = TF.vflip(mask1)
                    mask2 = TF.vflip(mask2)
                    mask3 = TF.vflip(mask3)
                if random.random()<.5:
                    img = TF.adjust_brightness(img,brightness_factor=.5)
                if random.random()<.5:
                    img = TF.adjust_contrast(img,contrast_factor=.4)
                if random.random()<.5:
                    img = TF.adjust_gamma(img,gamma=1.4)
                if random.random()<.5:
                    trans = T.Grayscale(num_output_channels=3)
                    img = trans(img)
                if random.random()<.0:
                    trans = T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2)
                    img = trans(img)
        
        im_size = 256
        trans = T.Resize((im_size,im_size))
        img = trans(img)
        if self.transform!="test": 
            mask1 = trans(mask1)
            mask2 = trans(mask2)
            mask3 = trans(mask3)
        trans = T.ToTensor()
        img = trans(img)
        if self.transform!="test": 
            mask1 = trans(mask1)
            mask2 = trans(mask2)
            mask3 = trans(mask3)
        
        meanR, meanG, meanB = .5,.5,.5
        stdR, stdG, stdB = .5, .5, .5 
        norm_= T.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB])
        #print(img.shape)
        img = norm_(img)
        if self.transform!='test':
           m = torch.stack((mask1[0],mask2[0],mask3[0]),dim=0)
        
        if self.transform!='test':
            return img, m
        else:
            return img
