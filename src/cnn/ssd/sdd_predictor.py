#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from math import sqrt as sqrt
import numpy as np
import itertools
from torch.autograd import Function
import torch.nn.init as init
from ssd_project import ssd
from ssd_project.detection import *
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv

#Load pretrained model
best_model = torch.load("saved_models/first_best_model_ssd300.pth.tar")
model = ssd.build_ssd(num_classes = 4)
model.load_state_dict(best_model["model_state_dict"])
device = "cuda"
model = model.to(device)

imgs = glob.glob("../data/*")
imgs.sort()
background = glob.glob("../data/background.png")[0]
img_names = list([])
for im in imgs:
    img_names.append(im.split('/')[-1].split('.')[0])

#SDD predictions
predictions = list([])
for i, im in enumerate(imgs):
    img, bboxes, labels, scores = predict_objects(model, im, min_score=0.2, max_overlap = 0.01, top_k=200)
    predictions.append((im,bboxes,labels,scores))
    annotated_img = draw_detected_objects(im, bboxes, labels, scores)
    labels_img = draw_detected_objects(background, bboxes, labels, scores)
    cv.imwrite("../data/ssd_{}.png".format(img_names[i]),cv.cvtColor(annotated_img,cv.COLOR_BGR2RGB))
