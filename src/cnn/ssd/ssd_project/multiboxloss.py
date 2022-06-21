import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt as sqrt
import collections
import numpy as np
import itertools
from .utils import *
from .global_variables import *

device = DEVICE
class MultiBoxLoss(nn.Module):
    """
    For our SSD we use a unique loss function called MultiBoxLoss.

    The loss is branch into:
        1. Localization loss coming from the predicted bounding boxes for objects with respect to ground truth object
        2. Confidence loss coming from the predicted class score for the object with respect to ground truth object class
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()

        self.priors_cxcy = priors_cxcy
        self.priors_xy = decode_center_size(self.priors_cxcy)
        self.threshold = threshold

        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        #L1 loss is used for the predicted localizations w.r.t ground truth.
        self.smooth_l1 = nn.L1Loss()

        #CrossEntropyLoss is used for the predicted confidence scores w.r.t ground truth.
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        """
        Each time the model predicts new localization and confidence scores,
        they are compared to the ground truth objects and classes.

        Args:

            :predicted_locs: predicted localizatios from the model w.r.t to prior-boxes.   Shape: (batch_size, 8732, 4)
            :predicted_scores: confidence scores for each class for each localization box. Shape:  (batch_size, 8732, n_classes)
            :boxes: ground truth objects per image: Shape(batch_size)
            :param labels: ground truth classes per image: Shape(batch_size)


        Return:

            Loss - a scalar
        """
        batch_size = predicted_locs.size(0)
        num_priors = self.priors_cxcy.size(0)
        num_classes = predicted_scores.size(2)

        assert num_priors == predicted_locs.size(1) == predicted_scores.size(1)


        true_locs, true_classes = self.match_priors_objs(boxes, labels, num_priors, num_classes, batch_size)

        # Identify priors that are positive (object/non-background)
        non_bck_priors = true_classes != 0  # (N, 8732)

        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.smooth_l1(predicted_locs[non_bck_priors], true_locs[non_bck_priors])  # (), scalar

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image

        # Number of positive and hard-negative priors per image
        num_positives = non_bck_priors.sum(dim=1)  # (N)
        num_hard_negatives = self.neg_pos_ratio * num_positives  # (N)

        # First, find the loss for all priors
        confidence_loss = self.cross_entropy(predicted_scores.view(-1, num_classes), true_classes.view(-1))  # (N * 8732)
        confidence_loss = confidence_loss.view(batch_size, num_priors)  # (N, 8732)

        # We already know which priors are positive
        confidence_loss_non_bck = confidence_loss[non_bck_priors]

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        confidence_loss_negative = confidence_loss.clone()  # (N, 8732)
        confidence_loss_negative[non_bck_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        confidence_loss_negative, _ = confidence_loss_negative.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness


        hardness_ranks = torch.LongTensor(range(num_priors)).unsqueeze(0).expand_as(confidence_loss_negative).to(device)  # (N, 8732)
        hard_negatives = hardness_ranks < num_hard_negatives.unsqueeze(1)  # (N, 8732)
        confidence_loss_hard_neg = confidence_loss_negative[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (confidence_loss_hard_neg.sum() + confidence_loss_non_bck.sum()) / num_positives.sum().float()

        # TOTAL LOSS
        return conf_loss + self.alpha * loc_loss

    def match_priors_objs(self, boxes, labels, num_priors, num_classes, batch_size):
        """
        Helper function:

            Basically we set a class("background", "window", "door", "building") for each prior.
            This is done by checking what is the overlap between each prior and the ground truth objects.
            If the overlap does not satisfy the threshold(0.5) for overlaping then we consider it background.
        """
        true_locs = torch.zeros((batch_size, num_priors, 4), dtype=torch.float).to(device)  # (batch_size, 8732, 4)
        true_classes = torch.zeros((batch_size, num_priors), dtype=torch.long).to(device)  # (batch_size, 8732)

        for i, bboxes_img in enumerate(boxes):

            #For each img and its objects, compute jaccard overlap between ground truth objects and priors
            num_objects = bboxes_img.size(0)
            obj_prior_overlap = jaccard_overlap(bboxes_img, self.priors_xy) #(num_objects, 8732)

            #Get best object per prior
            overlap_prior, obj_prior = obj_prior_overlap.max(dim = 0) #(8732)

            #Get best prior per object
            overlap_obj, prior_obj   = obj_prior_overlap.max(dim = 1) #(num_objects)

            #Fix that every object has been set to its respective best prior
            obj_prior[prior_obj] = torch.LongTensor(range(num_objects)).to(device)
            overlap_prior[prior_obj] = 1

            #Give a label to the prior
            label_prior = labels[i][obj_prior]
            label_prior[overlap_prior < self.threshold] = 0
            label_prior = label_prior.squeeze()

            true_classes[i] = label_prior

            #Encode it in boxes w.r.t to prior boxes format
            true_locs[i] = encode_xy_to_gcxgcy(bboxes_img[obj_prior], self.priors_cxcy)

        return true_locs, true_classes
