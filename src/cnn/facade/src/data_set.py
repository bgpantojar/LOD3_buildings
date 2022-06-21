"""
This script contains the codes to generate LOD3 models.
The codes are based on "Generation LOD3 models from structure-from-motion and semantic segmentation" 
by Pantoja-Rosero et., al.
https://doi.org/10.1016/j.autcon.2022.104430

## NOTICE ##
THIS SCRIPT IS TAKEN FROM TWO SOURCES:"""

# Import Modules
import os
import argparse
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

random.seed(100)

#Define the open_dataset class:
class open_dataset(Dataset):
    def __init__(self, path2data_i, path2data_m=None, transform=None):      

        imgsList=[pp for pp in os.listdir(path2data_i)]
        imgsList.sort()
        if transform!="test":
            anntsList=[pp for pp in os.listdir(path2data_m)]
            anntsList.sort()

        self.path2imgs = [os.path.join(path2data_i, fn) for fn in imgsList] 
        
        if transform!="test":
            self.path2annts= [os.path.join(path2data_m, fn) for fn in anntsList]

        self.transform = transform
    
    def __len__(self):
        return len(self.path2imgs)
      
    def __getitem__(self, idx):
        path2img = self.path2imgs[idx]
        img = Image.open(path2img)
        if self.transform!="test":
            path2annt = self.path2annts[idx]
            mask = Image.open(path2annt)
                
        if self.transform=='train':
                if random.random()<.5:
                    img = TF.hflip(img)
                    mask = TF.hflip(mask)
                if random.random()<.5:
                    img = TF.vflip(img)
                    mask = TF.vflip(mask)
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
        if self.transform!="test": mask = trans(mask)
        trans = T.ToTensor()
        img = trans(img)
        if self.transform!="test": mask = trans(mask)
        
        meanR, meanG, meanB = .5,.5,.5
        stdR, stdG, stdB = .5, .5, .5 
        norm_= T.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB])
        img = norm_(img)
        
        if self.transform!='test':
            return img, mask
        else:
            return img
