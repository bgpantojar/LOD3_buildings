import torch
import random
import torchvision.transforms.functional as FT
from torch.utils.data import Dataset
from PIL import Image
import glob
import math
import numpy as np
from .utils import *
from .global_variables import *

device = DEVICE

class TrainDataset(Dataset):
    """
    Creation of the dataset for training/testing. Can be used both for training and testing/validation.
    """

    def __init__(self, imgs_path, bboxes_path, labels_path, split, ratio):
        """
            Args:
                :imgs_path      - path to original images
                :bboxes_path    - path to bounding boxes used as a ground truth for all images
                :labels_path    - path to labes for each bounding box respectively for all images
                :split          - since we are using slightly different augmentations for training and testing
                                  split must be either "train" or "test"
                :ratio          - ratio for splliting the dataset
        """
        self.split = split.upper()
        self.imgs_path = imgs_path
        self.bboxes_path = bboxes_path
        self.labels_path = labels_path
        self.ratio = ratio
        assert self.split in {"TRAIN", "TEST"}

        #Get all paths with respect to each image
        self.imgs = glob.glob(self.imgs_path + "*")
        self.labels = glob.glob(self.labels_path + "labels*")
        self.bboxes = glob.glob(self.bboxes_path + "bboxes*")

        #Sort
        self.imgs.sort()
        self.labels.sort()
        self.bboxes.sort()

        if(split == "TRAIN"):
            a           = math.floor(len(self.imgs) * ratio)
            self.imgs   = self.imgs[:a]
            self.labels = self.labels[:a]
            self.bboxes = self.bboxes[:a]
        else:
            a = math.ceil(len(self.imgs) * ratio)
            self.imgs   = self.imgs[a:]
            self.labels = self.labels[a:]
            self.bboxes = self.bboxes[a:]


    def __getitem__(self, i):

        #Open an image and convert to RGB
        img = Image.open(self.imgs[i], mode = "r")
        img = img.convert("RGB")

        #Load respective bounding boxes/objects and labels for each img
        objs = np.load(self.bboxes[i])
        bboxes = torch.FloatTensor(objs[:, :4])
        labels = torch.LongTensor(objs[:, 4:])

        #Apply augmentations based on whether we are training or testing
        img, bboxes, labels = apply_augmentations(img, bboxes, labels, self.split)

        return img, bboxes, labels

    def __len__(self):
        return len(self.imgs)

    def collate_fn(self, batch):
        
        """
         For images that have different amount of objects in a batch we use a collate function.

         So we stack the imgs and we return the batch of imgs, with varying size tensors of bounding boxes and respective labels.
        """
            
        imgs, imgs_bboxes, imgs_labels = [], [], []
        
        for b in batch:
            imgs.append(b[0])
            imgs_bboxes.append(b[1])
            imgs_labels.append(b[2])

        imgs = torch.stack(imgs, dim = 0)

        return imgs, imgs_bboxes, imgs_labels

def expand(img, bboxes, mean):
    """
    Performs a zoom out operation with 50% possibility as in the paper.
    Helpful when detecting smaller objects(windows, doors). 
    Ref: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py
    """
    #Convert img to tensor, and redefine mean to fill out surrounding space of the data that our base was trained on. 
    img = FT.to_tensor(img)
    filler = torch.FloatTensor(mean).unsqueeze(1).unsqueeze(1)
    if random.random() > 0.5:
        return img, bboxes
  
    
    height, width = img.size(1), img.size(2)
    #Ratio of expansion
    ratio = random.uniform(1, 4)
    #Expand
    new_img = torch.ones((3, int(ratio * height), int(ratio * width)), dtype=torch.float) * filler

    #place original image 
    left = random.randint(0, int(ratio * width) - width)
    top = random.randint(0, int(ratio * height) - height)
    new_img[:, top:(top+height), left:(left+width)] = img

    #expand bounding boxes respectively
    new_bboxes = bboxes + torch.FloatTensor([left, top, left, top]).unsqueeze(0)  

    return new_img, new_bboxes

def random_crop(image, boxes, labels):
    """
    Performs a random crop operation with multiple possibilities as in the paper.
    Helpful when detecting bigger objects(windows, doors, buildings). 
    Ref: https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
    """
    height, width = image.size(1), image.size(2)

    while True:
        #randomly choose a min overlap
        mode = random.choice([0., .1, .3, .5, .7, .9, None])  # 'None' refers to no cropping

        # Do not crop if NONE
        if mode is None:
            return image, boxes, labels

        # Do 50 trials
        max_trials = 50
        for _ in range(max_trials):
            
            # Crop dimensions must be in [0.3, 1] of original dimensions
            new_height = int(random.uniform(0.3, 1) * height)
            new_width = int(random.uniform(0.3, 1) * width)

            # Aspect ratio must be in [0.5, 2]
            aspect_ratio = new_height / new_width
            if not 0.5 < aspect_ratio < 2:
                continue

            # Get crop coordinates
            left = random.randint(0, width - new_width)
            right = left + new_width
            top = random.randint(0, height - new_height)
            bottom = top + new_height
            crop = torch.FloatTensor([left, top, right, bottom])  # (4)

            # Compute jaccard overlap between crop and bboxes
            overlap = jaccard_overlap(crop.unsqueeze(0),
                                           boxes)
            
            overlap = overlap.squeeze(0)  

            # If all overlaps are smaller try again 
            if overlap.max().item() < mode:
                continue

            #Crop the image
            new_image = image[:, top:bottom, left:right]
            
            #Get centers of bounding boxxes
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.

            # Find bounding boxes whose centers are in the crop
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (bb_centers[:, 1] < bottom)  

            #If no boxes are in the crop try again
            if not centers_in_crop.any():
                continue

            #Remove bounding boxes that do not satisfy cond
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]

            # Compute the positions of bounding boxes in the new img
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:]) 
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels

def random_hflip(img, boxes):
    """
    Horizontal flip of img and bounding boxes with a 50% possibility
    """
    img = FT.to_pil_image(img)
    if random.random() > 0.5:
        return img, boxes
    #Flip Image
    img = FT.hflip(img)
    img_w = img.width

    # Flip bounding boxes
    new_boxes = boxes
    new_boxes[:, 0] = img_w - boxes[:, 0] - 1
    new_boxes[:, 2] = img_w - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return img, new_boxes

def photometric_distort(img):
    #REF: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
    #Apply distortions on brightness, contrast, saturation, hue
    new_img = img

    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is 'adjust_hue':
                #Empirical Values taken out of Original/Caffe Repo
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                #Empirical Values taken out of Original/Caffe Repo
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply each distortion
            new_img = d(new_img, adjust_factor)

    return new_img


def resize(image, boxes, dims=(300, 300)):
    """
    Resize an image to 300, 300 if it is not and its bounding boxes.
    """
    # Resize image
    new_image = FT.resize(image, dims)

    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    #Percent Coordinates
    new_boxes = boxes / old_dims

    return new_image, new_boxes

def apply_augmentations(img, bboxes, labels, split="TRAIN", augment = True):

    #ImageNET
    assert split in {'TRAIN', 'TEST'}

    # Mean and standard deviation of ImageNet data that the base/VGG from torchvision was trained on
    #https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]


    #Apply Augmentations only on Training-Set
    if split == 'TRAIN' and augment == True:

        #Photometric distortions
        img = photometric_distort(img)
        
        #Zoom out operation, expand img
        img, bboxes = expand(img, bboxes, mean)

        #Randomly crop image(zoom in)
        img, bboxes, labels = random_crop(img, bboxes, labels)

        #Horizontal Flip image with a 50% chance
        img, bboxes = random_hflip(img, bboxes)

    # Resize image to (300, 300)
    img, bboxes = resize(img, bboxes, dims=(300, 300))

    # Convert PIL image to Torch tensor
    img = FT.to_tensor(img)

    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    img = FT.normalize(img, mean=mean, std=std)

    return img, bboxes, labels
