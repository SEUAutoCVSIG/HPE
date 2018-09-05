# -*- coding: utf-8 -*-
'''
    Created on wed Sept 6 19:34 2018

    Author           : Shaoshu Yang
    Email            : 13558615057@163.com
    Last edit date   : Sept 6 19:34 2018

South East University Automation College, 211189 Nanjing China

The following codes referenced Ayoosh Kathuria's blog:
How to implement a YOLO (v3) object detector from strach in
PyTorch: Part 3
'''

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

def pred_transform(prediction, in_dim, anchors, class_num, CUDA=True):
    '''
        Args:
             prediction   : (tensor) output of the network
             in_dim       : (int) input dimension
             anchors      : (tensor) describe anchors
             class_num    : (int) class numbers
             CUDA         : (bool) defines the accessibility to CUDA
                             and GPU computing
        Returns:

    '''
    batch_size = prediction.size(0)
    stride = in_dim//prediction.size(2)
    grid_size = in_dim//stride
    bbox_attr = 5 + class_num
    anchor_num = len(anchors)

    # Transfroms to the prediction
    prediction = prediction.view(batch_size, bbox_attr*anchor_num,
                                 grid_size*grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, anchor_num*grid_size**2,
                                 bbox_attr)
    anchors = [(a[0]//stride, a[1]//stride) for a in anchors]

    # Adding sigmoid to the x_coord, y__coord and objscore
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    # Add offset to the central coordinates
    offset_x = torch.FloatTensor(a).view(-1, 1)
    offset_y = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        offset_x = offset_x.cuda()
        offset_y = offset_y.cuda()

    offset_x_y = torch.cat((offset_x, offset_y), 1).view(-1, 2).squeeze(0)
    prediction[:, :, :2] += offset_x_y

    # Add log-space transforms
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).squeeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4])*anchors

    # Add sigmoid to classes possibility
    prediction[:, :, 5:5 + class_num] = torch.sigmoid(prediction[:, :, 5 +
                                                                class_num])

    # Resize the detection map to the original image size
    prediction[:, :, :4] *= stride

    return prediction


