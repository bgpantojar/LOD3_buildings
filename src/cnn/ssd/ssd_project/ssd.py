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

#Implementations followed:
#1. https://arxiv.org/pdf/1512.02325.pdf - original paper
#2. https://github.com/amdegroot/ssd.pytorch/blob/master/ssd.py - SSD architecture
#3. https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py - Generation of Prior Boxes
#4. https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py - Derivation of VGG16

class SSD(nn.Module):
    """ SSD - Single Shot Multibox Architecture
    The network is composed of an already existing image classification architecture,
    a base VGG network which is followed by extra feature layers on top and prediction convolutions.
     Each prediction layer has
        1. conv2d for class conf scores
        2. conv2d for localization predictions
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        base: VGG16 layers for input image of size 300, that provide smaller-level
        feature maps

        extra: Extra layers added on top of the VGG16 that will provide higher-level
        feature maps

        head: consists of loc and conf conv layers that will locate and identify
        objects in the feature map

        num_classes: number of different classes/objects
    """

    def __init__(self, base, extras, head, num_classes):
        super(SSD, self).__init__()

        self.num_classes = num_classes

        #Generation of 8732 Prior Boxes with center size coordinates(cx, cy, w, h)
        self.priors_cxcy = self.generate_ssd_priors()

        # Creation of the SSD structure

        #Base Convolutions
        self.vgg = nn.ModuleList(base)

        # The conv4_3 feature layer has larger scales, therefore we normalize it with L2 Norm
        # and rescale it.
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))
        nn.init.constant_(self.rescale_factors, 20)

        #Auxiliray convolutions
        self.extras = nn.ModuleList(extras)

        #Prediction Convolutions

        #Loc conv layers
        self.loc = nn.ModuleList(head[0])

        #Conf conv layers
        self.conf = nn.ModuleList(head[1])

    def forward(self, img):
        """Apply the layers on input img.

        Args:
            img: batch of images. Shape: (batch_size,3,300,300).

        Return:
            1. localization layers for objects - Shape:(batch_size, 8732, 4)
            2. confidence layers for classes - Shape:(batch_size, 8732, num_classes)

        """

        conv_layers = []

        # apply vgg up to conv4_3 relu
        for k in range(23):
            img = self.vgg[k](img)

        #Normalize conv4_3 layer
        l2norm_img = img.pow(2).sum(dim=1, keepdim=True).sqrt()
        img = img / l2norm_img
        img = img * self.rescale_factors

        #Append conv4_3, as is later used for loc and conf predictions
        conv_layers.append(img)

        # apply vgg up to conv7
        for k in range(23, len(self.vgg)):
            img = self.vgg[k](img)

        #Append conv7, as is later used for loc and conf predictions
        conv_layers.append(img)


        # apply extra layers and cache source layer outputs
        # append only conv8_2, conv9_2, conv10_2, conv11_2 used for loc and conf predictions
        for k, v in enumerate(self.extras):
            img = F.relu(v(img), inplace=True)
            if k % 2 == 1:
                conv_layers.append(img)



        batch_size = conv_layers[0].size(0)

        l = []
        c = []

        #apply loc and conf convolutions on conv4_3, conv_7, conv8_2, conv9_2, conv10_2, conv11_2
        #and get localization and conf predictions
        for i in range(len(conv_layers)):

            loc_conv = self.loc[i](conv_layers[i])
            loc_conv = loc_conv.permute(0,2,3,1).contiguous()
            loc_conv = loc_conv.view(batch_size, -1, 4)

            conf_conv = self.conf[i](conv_layers[i])
            conf_conv = conf_conv.permute(0,2,3,1).contiguous()
            conf_conv = conf_conv.view(batch_size, -1, self.num_classes)

            l.append(loc_conv)
            c.append(conf_conv)


        locs = torch.cat(l, dim=1)  # (N, 8732, 4)
        classes_scores = torch.cat(c, dim=1)  # (N, 8732, n_classes)

        return locs, classes_scores


    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def generate_ssd_priors(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        """
        Create the 8732 pre-computed boxes called priors for the SSD300, as defined in the paper.

        :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
        """

        #Dimension of convolution layers
        fmap_dims = {'conv4_3': 38,
                     'conv7': 19,
                     'conv8_2': 10,
                     'conv9_2': 5,
                     'conv10_2': 3,
                     'conv11_2': 1}

        #Object scales should match the charasteristics of objects found in the img.
        #Since conv4_3 and conv7 are used for detecting smaller objects we use smaller scales.
        #While conv8_2, conv9_2, conv10_2, conv11_2 are used for detecting larger objects,
        #therefore we use larger scales.
        #Scales:  A scale S,  means the priors's area is equal to that of a square with side s
        obj_scales = {'conv4_3': 0.02,
                      'conv7': 0.03,
                      'conv8_2': 0.4,
                      'conv9_2': 0.5,
                      'conv10_2': 0.6,
                      'conv11_2': 0.8}

        #Aspect ratios(w/h) should match the charasteristics of objects(building, windows, door) found in the img.
        aspect_ratios = {'conv4_3': [1., 0.4, 0.333],
                         'conv7': [1., 0.6, 0.5, 0.4, 0.333],
                         'conv8_2': [1., 2., 0.5, 0.4, 0.333],
                         'conv9_2': [1., 2., 0.5, 0.4, 0.333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        #The generation of the priors boxes follows the original paper
        #Note: if we take out the aspect ratio of 1, the code won't generate 8732 boxes.
        #This is also done in the original implemetation.
        for k, fmap in enumerate(fmaps):
            for i, j in itertools.product(range(fmap_dims[fmap]), repeat=2):

                size_fmap = fmap_dims[fmap]
                scale_fmap = obj_scales[fmap]

                cx = (j + 0.5) / size_fmap
                cy = (i + 0.5) / size_fmap

                for ratio in aspect_ratios[fmap]:
                    prior_boxes.append([cx, cy, scale_fmap * sqrt(ratio), scale_fmap / sqrt(ratio)])

                    # In the original implemetation we create always 2 boxes of aspect_ratio 1.
                    #. The difference between the two is that for the second prior box
                    # we use a scale which is the geometric mean of the scale current feature map  and the scale
                    # of the next feature map .
                    if ratio == 1.:
                        try:
                            additional_scale = sqrt(scale_fmap * obj_scales[fmaps[k + 1]])
                        # For the last feature map, there is no "next" feature map
                        except IndexError:
                            additional_scale = 1.
                        prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # Shape: (8732, 4)
        prior_boxes.clamp_(min=0, max=1)  # Shape: (8732, 4)

        #Return the pre-computed prior-boxes which are later used in multibox loss
        return prior_boxes # (8732, 4)


def vgg(cfg, i):
    #Implementation of VGG


    layers = []
    in_channels = i
    for v in cfg:

        if v == 'M':
            layers = layers + [nn.MaxPool2d(kernel_size=2, stride=2)]

        elif v == 'C':
            layers = layers + [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]

        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers = layers + [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

    layers = layers + [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    #All this layers later represent the base convolutions of the SSD architecture
    return layers


def add_extras(in_channels):

    # Extra layers added to VGG for feature scaling
    layers = []

    layers = layers + [nn.Conv2d(in_channels, 256, kernel_size=1, padding = 0)]
    layers = layers + [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding = 1)]

    layers = layers + [nn.Conv2d(512, 128, kernel_size=1, padding = 0)]
    layers = layers + [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding = 1)]

    for i in range(2):
        layers = layers + [nn.Conv2d(256, 128, kernel_size=1, padding=0)]
        layers = layers + [nn.Conv2d(128, 256, kernel_size=3, padding=0)]

    #Returns the extra layers
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):

    loc_layers = []
    conf_layers = []
    #Creatation of:
    #1. localization layers used for objects predictions
    #2. confidence layers used for class predictions


    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):

        loc_layers  = loc_layers + [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers = conf_layers + [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]

    for k, v in enumerate(extra_layers[1::2], 2):

        loc_layers = loc_layers + [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers = conf_layers + [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]

    #Returns the all the layers base, auxiliray, prediction convolutions
    return vgg, extra_layers, (loc_layers, conf_layers)




def build_ssd(num_classes=4):
    """
    Args:
        num_classes: number of classes

    Return:
        The SSD300 network

    """

    #Configuration of the VGG:
        #1. The "Number" in base array mean in_channels or out_channels of conv2d, where "i=3" is the first in_channel
        #2. "M" means to add a layer of MaxPool2d
        #3. "C" means to add a layer of MaxPool2d with ceiling mode set to TRUE
    base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',512, 512, 512]

    mbox = [4, 6, 6, 6, 4, 4]  # number of boxes per feature map location - used in prediction convolutions

    base_, extras_, head_ = multibox(vgg(base, 3),
                                     add_extras(1024),
                                     mbox, num_classes)

    return SSD(base_, extras_, head_, num_classes)
