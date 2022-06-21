import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import os
from math import sqrt as sqrt
import collections
import numpy as np
import itertools
import torchvision.transforms.functional as FT
from .global_variables import *

device = DEVICE

def decode_gcxgcy_to_xy(bboxes_gcxgcy, priors_cxcy):
    """
    Direct representation of bounding boxes w.r.t to priors in center size format to (xmin, ymin, xmax, ymax) format.
    Which is a combination of the:
        1. decode_gcxgcy_to_center_size
        2. decode_center_size

    Return:
        (xmin, ymin, xmax, ymax) format of bounding boxes
    """

    bboxes_cxcy = decode_gcxgcy_to_center_size(bboxes_gcxgcy, priors_cxcy)
    bboxes_xy = decode_center_size(bboxes_cxcy)

    return bboxes_xy

def encode_xy_to_gcxgcy(bboxes_xy, priors_cxcy):
    """
    Direct representation of bounding boxes of (xmin, ymin, xmax, ymax) format to bounding boxes w.r.t
    priot boxes format.

    Which is a combination of the:
        1. encode_center_size
        2. encode_center_size_bboxes_to_bboxes_gcxgcy

    Return:
        gcxgcy format of bounding boxes
    """
    bboxes_cxcy = encode_center_size(bboxes_xy)
    bboxes_gcxcy = encode_center_size_bboxes_to_bboxes_gcxgcy(bboxes_cxcy, priors_cxcy)

    return bboxes_gcxcy

def encode_center_size(bboxes_xy):
    """
    Direct representation of bounding boxes in (xmin, ymin, xmax, ymax) format
    to center size format (cx, cy, w, h)
    Args:

        :bboxes_xy: (num_boxes, (xmin, ymin, xmax, ymax))

    Return:
        :bboxes_cxcy (num_boxes, (cx, cy, w, h))
    """


    return torch.cat([(bboxes_xy[:, 2:] + bboxes_xy[:, :2]) / 2,  # cx, cy
                       bboxes_xy[:, 2:] - bboxes_xy[:, :2]], 1)  # w, h

def decode_center_size(bboxes_cxcy):
    """
    Inverse from encode_center_size
    """
    return torch.cat([bboxes_cxcy[:, :2] - (bboxes_cxcy[:, 2:]/2),     # xmin, ymin
                      bboxes_cxcy[:, :2] + (bboxes_cxcy[:, 2:]/2)], 1)  # xmax, ymax

def encode_center_size_bboxes_to_bboxes_gcxgcy(bboxes_cxcy, priors_cxcy):
    """We encode the center size bounding boxes w.r.t to prior boxes in center size format.
    Variances of 5 and 10 are used in the original implemetation, totally empirical data.
    Check out caffe repo for more information: https://github.com/weiliu89/caffe/tree/ssd

    Args:
        :bboxes_cxcy:  Ground truth for each prior in (cx, cy, w, h) format Shape: (num_priors, (cx, cy, w, h))
        :priors_cxcy:  Prior boxes in center size format Shape: (num_priors, (cx, cy, w, h)).
    Return:
        :bboxes_gcxgcy, Shape: (num_priors, (gcx, gcy, gw, gh)
    """

    return torch.cat([(bboxes_cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # gcx, gcy
                      torch.log(bboxes_cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # gw, gh

def decode_gcxgcy_to_center_size(bboxes_gcxgcy, priors_cxcy):

    """
    Deconding bounding box w.r.t prior boxes to center size format.
    Inverse of encode_center_size_bboxes_to_bboxes_gcxgcy
    """

    return torch.cat([bboxes_gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(bboxes_gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h


def intersection(box_a, box_b):
    """ Compute intersection between each two boxes in the sets box_a and box_b
    Args:
      :box_a  bounding boxes, Shape: (N,4).
      :box_b  bounding boxes, Shape: (M,4).
    Return:
      Intersection area, Shape: (N,M).
    """

    max_xy = torch.min(box_a[:, 2:].unsqueeze(1), box_b[:, 2:].unsqueeze(0))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1), box_b[:, :2].unsqueeze(0))
    inter = torch.clamp((max_xy - min_xy), min=0)

    return inter[:, :, 0] * inter[:, :, 1]

def jaccard_overlap(set_a, set_b):

    """Compute Jaccard Overlap between 2 boxes, which is: a ∩ B / a ∪ b
        :set_a set of bounding boxes Shape: (N, format of encoding)
        :set_b: set of bounding boxes Shape: (M, format of encoding)
        Format of encoding must be the same for both sets
    Return:
        a ∩ B / a ∪ b

    """
    intersection_seta_setb = intersection(set_a, set_b)

    area_set_a = ((set_a[:, 2] - set_a[:, 0]) * (set_a[:, 3] - set_a[:, 1])).unsqueeze(1)
    area_set_b = ((set_b[:, 2] - set_b[:, 0]) * (set_b[:, 3] - set_b[:, 1])).unsqueeze(0)

    union_seta_setb = area_set_a + area_set_b - intersection_seta_setb

    return intersection_seta_setb/union_seta_setb
