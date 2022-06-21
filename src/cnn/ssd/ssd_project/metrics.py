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
from ssd_project.model import ssd
from ssd_project.functions.detection import *
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from .detection import *
from .multiboxloss import *
from .ssd import *
from .global_variables import *
from .helpers import *
from .transformations import *
from .utils import *
from pprint import PrettyPrinter

pp = PrettyPrinter()


def create_mAP_data(bboxes, labels):
    
    imgs = []
    for i, label in enumerate(labels):
        imgs.extend([i] * label.size(0))
    
    imgs = torch.LongTensor(imgs).to(device)
    labels = torch.cat(labels, dim = 0).squeeze()
    bboxes = torch.cat(bboxes, dim = 0)
    
    return imgs, bboxes, labels 

def extract_class_bboxes(imgs, bboxes, labels, cls):
    cls_imgs = imgs[labels == cls]
    cls_bboxes = bboxes[labels == cls]
    
    return cls_imgs, cls_bboxes

def compute_cumulative_prec_recall(num_objs, true_pos, false_pos):
        sum_true_pos = torch.cumsum(true_pos, dim = 0)
        sum_false_pos = torch.cumsum(false_pos, dim = 0)
        
        return (sum_true_pos / (sum_true_pos + sum_false_pos)), sum_true_pos / num_objs
    
def compute_mAP(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, num_classes):
    assert len(pred_bboxes) == len(pred_labels) == len(pred_scores) == len(gt_bboxes) == len(gt_labels)
    
    gt_imgs, gt_bboxes, gt_labels = create_mAP_data(gt_bboxes, gt_labels)
    pred_imgs, pred_bboxes, pred_labels = create_mAP_data(pred_bboxes, pred_labels)
    pred_scores = torch.cat(pred_scores, dim = 0)
    
    assert pred_imgs.size(0) == pred_bboxes.size(0) == pred_labels.size(0) == pred_scores.size(0)
    assert gt_imgs.size(0) == gt_bboxes.size(0) == gt_labels.size(0)
    avg_per_cls = torch.zeros((num_classes - 1), dtype=torch.float)
    
    #Calculate Average Precesion For Each Class 
    for cls in range(1, num_classes):
        
        
        # Extract only objects with this class
        gt_cls_imgs, gt_cls_bboxes = extract_class_bboxes(gt_imgs, gt_bboxes, gt_labels, cls)

        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        gt_cls_bboxes_det = torch.zeros((gt_cls_bboxes.size(0)), dtype=torch.uint8).to(device)  
        
        pred_cls_imgs, pred_cls_bboxes = extract_class_bboxes(pred_imgs, pred_bboxes, pred_labels, cls)
        pred_cls_scores = pred_scores[pred_labels == cls]
        num_cls_pred = pred_cls_bboxes.size(0)
        
        if(num_cls_pred == 0):
            continue
        
        true_pos = torch.zeros((num_cls_pred), dtype=torch.float).to(device)
        false_pos = torch.zeros((num_cls_pred), dtype=torch.float).to(device)
        pred_cls_scores, sort_ind = torch.sort(pred_cls_scores, dim = 0, descending = True)
        pred_cls_imgs = pred_cls_imgs[sort_ind]
        pred_cls_bboxes = pred_cls_bboxes[sort_ind]
        
        
        for pred in range(num_cls_pred):
            this_pred_bbox = pred_cls_bboxes[pred].unsqueeze(0)
            this_img = pred_cls_imgs[pred]
            
            objs_bboxes = gt_cls_bboxes[gt_cls_imgs == this_img]
            
            if(objs_bboxes.size(0) == 0):
                false_pos[pred] = 1
                continue
            
            
            j_overlaps = jaccard_overlap(this_pred_bbox, objs_bboxes)
            max_overlap, ind = torch.max(j_overlaps.squeeze(0), dim = 0)
            
            org_ind = torch.LongTensor(range(gt_cls_bboxes.size(0)))[gt_cls_imgs == this_img][ind]
            
            if max_overlap.item() > 0.5:
                if gt_cls_bboxes_det[org_ind] == 0:
                    true_pos[pred] = 1
                    gt_cls_bboxes_det[org_ind] = 1
                else:
                    false_pos[pred] = 1
            else:
                false_pos[pred] = 1
        
        precision, recall = compute_cumulative_prec_recall(gt_cls_bboxes.size(0), true_pos, false_pos)
        
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = recall >= t
            if recalls_above_t.any():
                precisions[i] = precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        avg_per_cls[cls - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = avg_per_cls.mean().item()

    return avg_per_cls, mean_average_precision

def evaluate_maP()
    # Parameters
    data_folder = './'
    batch_size = 64
    workers = 4

    path_imgs = "/data/ssd_ilija_data/original_images/"
    path_bboxes = "/data/ssd_ilija_data/ground_truth/bboxes_labels/"
    model = model.to(device)


    val_dataset = TrainDataset(path_imgs, path_bboxes, path_bboxes, "TEST", 0)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                              collate_fn=val_dataset.collate_fn, num_workers=workers, pin_memory=True)




    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels = [], [], [], [], []


    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = detect_objects(predicted_locs, predicted_scores,                                                                                                    model.priors_cxcy,
                                                                                       min_score=0.01, max_overlap=0.6,
                                                                                       top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            pred_bboxes.extend(det_boxes_batch)
            pred_labels.extend(det_labels_batch)
            pred_scores.extend(det_scores_batch)
            gt_bboxes.extend(boxes)
            gt_labels.extend(labels)
         
        # Calculate mAP
        APs, mAP = compute_mAP(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, 4)

        # Print AP for each class
        pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)
       
