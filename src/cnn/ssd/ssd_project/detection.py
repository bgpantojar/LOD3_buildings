import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from math import sqrt as sqrt
import collections
import numpy as np
import itertools
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.init as init
from .utils import *
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as FT
import cv2
from .global_variables import *

def detect_objects(predicted_locs, predicted_scores, priors_cxcy, min_score=0.2, max_overlap=0.5, top_k=100):
    """
    Decode the 8732 loc and conf scores outputed by the SSD300 network.
    Args:
        predicted_locs: predicted boxes w.r.t the 8732 prior boxes. Shape: (batch_size, 8732, 4)
        predicted_scores: class scores for each of the encoded boxe. Shape: (batch_size, 8732, num_classes)
        min_score: minimum threshold for a box to be considered a match for a certain class
        max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non Maximum Suppression-NMS
        top_k: if there are a lot of resulting detections, return only "top_k" predictions

    Return:
        boxes, labels, scores for an img/batch_size
    """
    batch_size = predicted_locs.size(0)
    num_priors = priors_cxcy.size(0)
    predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)
    num_classes = predicted_scores.size(2)

    imgs_bboxes, imgs_labels, imgs_scores = [], [], []
    
    assert num_priors == predicted_locs.size(1) == predicted_scores.size(1)

    for i in range(batch_size):

        # Decode the localization predictions to xy format w.r.t prior_boxes
        decoded_locs = decode_gcxgcy_to_xy(predicted_locs[i], priors_cxcy)  # (8732, 4), these are fractional pt. coordinates


        bboxes, labels, scores = [], [], []

        conf_scores_img = predicted_scores[i].clone()


        #For each class apply non-maximum suppression on predictions that are above min score
        for cls in range(1, num_classes):

            # Keep only predicted boxes and scores where scores for this class are above the minimum score
            class_scores = conf_scores_img[:, cls]  # (8732)
            score_above_min_score = class_scores.gt(min_score)
            num_above_min_score = score_above_min_score.sum().item()

            if num_above_min_score == 0:
                continue

            #Apply Non-Maximum Supression, if multiples priors overlap significantly(max_overlap as threshold)
            #We remove those redudant predictions by supressing all but one with the maximum score.
            bboxes_nms, labels_nms, scores_nms = nms(decoded_locs, class_scores, score_above_min_score,
                                        num_above_min_score, max_overlap, cls)
            bboxes.append(bboxes_nms)
            labels.append(labels_nms)
            scores.append(scores_nms)

        # If no object is found, we add background box, label, score only
        if len(bboxes) == 0:
            bboxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
            labels.append(torch.LongTensor([0]).to(device))
            scores.append(torch.FloatTensor([0.]).to(device))

        # Concatenate into single tensors
        bboxes = torch.cat(bboxes, dim=0)  # (n_objects, 4)
        labels = torch.cat(labels, dim=0)  # (n_objects)
        scores = torch.cat(scores, dim=0)  # (n_objects)
        n_objects = scores.size(0)
    
        # Keep only the top k objects
        if n_objects > top_k:
            scores, sort_ind = scores.sort(dim=0, descending=True)
            scores = scores[:top_k]  # (top_k)
            bboxes = bboxes[sort_ind][:top_k]  # (top_k, 4)
            labels = labels[sort_ind][:top_k]  # (top_k)

        # Append to lists that store predicted boxes and scores for all images
        imgs_bboxes.append(bboxes)
        imgs_labels.append(labels)
        imgs_scores.append(scores)

    return imgs_bboxes, imgs_labels, imgs_scores

def nms(decoded_locs, class_scores, score_above_min_score, num_above_min_score, max_overlap, cls):
    """
    Applies Non-Maximum Supression, which means that if multiples priors overlap significantly(max_overlap as threshold)
    We remove those redudant predictions by supressing all but the one with maximum score.

    Args:

        :decoded_locs           : decoded localizations in format xy coordinates
        :class_scores           : scores given for each class
        :score_above_min_score  : indexes of scores that have a score above minimum score
        :num_above_min_score    : number of scores above minimum score
        :max_overlap            : maximum overlap that two boxes can have so that the one with the lower score is not suppressed via Non Maximum Suppression-NMS
        :cls                    : class/label

    Return:
        boxes, labels, scores respectively for boxes that weren't suppressed
    """

    #Get class_scores that are above min score
    class_scores = class_scores[score_above_min_score]

    #Get class localizations
    class_decoded_locs = decoded_locs[score_above_min_score]

    #Sort them in a descending order
    class_scores, sort_ind = class_scores.sort(dim=0, descending=True)
    class_decoded_locs = class_decoded_locs[sort_ind]

    #Apply jaccard_overlap to all boxes between each other
    overlap = jaccard_overlap(class_decoded_locs, class_decoded_locs)

    suppress = torch.zeros((num_above_min_score), dtype=torch.uint8).to(device)

    #For every box check if it should be suppressed working with max_overlap as threshold
    for box in range(class_decoded_locs.size(0)):

        if suppress[box] == 1:
            continue

        # Suppress boxes whose overlaps are above max_overlap
        suppress = torch.max(suppress, (overlap[box] > max_overlap).byte())

        #Set supress of this box = 0, since it has an overlap of 1 with itself
        suppress[box] = 0

    #Get important boxes with respective scores and labels and return them
    bboxes_nms = class_decoded_locs[1 - suppress]
    labels_nms = torch.LongTensor((1- suppress).sum().item() * [cls]).to(device)
    scores_nms = class_scores[1 - suppress]

    return bboxes_nms, labels_nms, scores_nms

from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as FT
import cv2

def predict_objects(model, path_img, min_score = 0.2, max_overlap = 0.2, top_k = 20):
    """
    Detect objects in an image with a trained Model(SSD300).

    Args:

        :model       : a trained SSD300 network
        :path_img    : path to the image for which predictions are needed
        :min_score   : minimum threshold for a box to be considered a match for a certain class
        :max_overlap : maximum overlap that two boxes can have so that the one with the lower score is not suppressed via Non Maximum Suppression-NMS
        :top_k       : top k objects

    Return:
        :Original Image
        :detected objects with (xmin, ymin, xmax, ymax)
        :detected labels for each object respectively
        :detected scores for each object respectively
    """

    #PIL Image
    original_image = Image.open(path_img).convert('RGB')

    #Resize original image to 300, 300 dimensions
    resize = transforms.Resize((300, 300))

    #Transfor it to tensor
    to_tensor = transforms.ToTensor()

    #Normalize by Mean and standard deviation of ImageNet data that the base/VGG from torchvision was trained on
    #https://pytorch.org/docs/stable/torchvision/models.html
    #This is applied since we did the same when appling augmentations to images used for training
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    img = normalize(to_tensor(resize(original_image))).to(device)


    #Get Predictions by the SSD30
    predicted_locs, predicted_scores = model(img.unsqueeze(0))

    # get best objects and suppress redudant objects
    det_boxes, det_labels, det_scores = detect_objects(predicted_locs, predicted_scores, model.priors_cxcy, min_score,
                                                                   max_overlap, top_k)
    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')
    det_scores = det_scores[0].to("cpu")
    det_labels = det_labels[0].to("cpu")

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)

    # Transoform detected bounding boxes to fit original size image, and not one of 300, 300 dimensions
    det_boxes = det_boxes * original_dims
    rev_label_map = REVERSE_LABEL_MAP
    # Decode class integer labels, put names(window, door, building)
    det_labels = [rev_label_map[l] for l in det_labels.tolist()]

    assert len(det_boxes) == len(det_labels) == len(det_scores)

    return original_image, det_boxes, det_labels, np.around(det_scores.detach().numpy(),decimals=1)

def draw_detected_objects(path_img, det_boxes, det_labels, det_scores, draw_scores = False):

    """
    Draw detected objects in the image.

    Args:
        :path_img    : path to the image
        :det_boxes   : coordinates in (xmin, ymin, xmax, ymax) form of detected objects/boxes in the image
        :det_labels  : respective detected labels for the objects
        :det_scores  : respective detected scores for the objects

    Return:
        :Annotated image with the detected objects
    """

    #Read Img and Convert to RGB
    org_img = cv2.imread(path_img)
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    #For each detected object/box draw a bounding box, with the color corresponding to the label
    #and draw the score
    for i, box in enumerate(det_boxes):

        #For drawing we the bounding boxes we need top_left and bottom_right points since we are drawing rectangles
        top_left     = (box[0], box[1])
        bottom_right = (box[2], box[3])

        #Draw a door
        if(det_labels[i] == "door"):
            color = (230, 25, 75)
            cv2.rectangle(org_img, top_left, bottom_right, color, 10)
            if(draw_scores):
                cv2.putText(org_img, str(det_scores[i]), (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        #Draw a building
        elif(det_labels[i] == "building"):
            color = (255, 255, 0)
            cv2.rectangle(org_img, top_left, bottom_right, color, 10)
            if(draw_scores):
                cv2.putText(org_img, str(det_scores[i]), (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        #Draw a window
        else:
            color = (60, 180, 75)
            cv2.rectangle(org_img, top_left, bottom_right, color, 10)
            if(draw_scores):
                cv2.putText(org_img, str(det_scores[i]), (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    #Return Annotated Image
    return org_img
