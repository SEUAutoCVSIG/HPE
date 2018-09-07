# -*- coding: utf-8 -*-
'''
    Created on wed Sept 6 19:34 2018

    Author           : Shaoshu Yang
    Email            : 13558615057@163.com
    Last edit date   : Sept 6 23:47 2018

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
    grid_size = prediction.size(2)
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


    offset_x_y = torch.cat((offset_x, offset_y), 1).repeat(anchor_num,
                                                 1).view(-1, 2).squeeze(0)
    prediction[:, :, :2] += offset_x_y

    # Add log-space transforms
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).squeeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4])*anchors

    # Add sigmoid to classes possibility
    prediction[:, :, 5:5 + class_num] = torch.sigmoid(prediction[:, :, 5:5 +
                                                                class_num])

    # Resize the detection map to the original image size
    prediction[:, :, :4] *= stride

    return prediction

# Adding objectness score thresholding and Non-maximal suppression
def write_results(prediction, confidence, class_num, nms_conf=0.4):
    '''
        Args:
             prediction  : (tensor) output tensor form darknet
             confidence  : (float) object score threshold
             class_num   : (num) number of classes
             num_conf    : (float) confidence of Non-maximum suppresion
        Returns:
             Results after confidence threshold and Non-maximum suppresion
             process
    '''
    # Set the attributes of a bounding-box to zero when its score
    # is below the threshold
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask

    # Transform the bx, by, bw, bh to the coordinates of the top-left x,
    # top_left y, right_bottom x, right_bottom y
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2]/2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3]/2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2]/2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3]/2)
    prediction[:, :, :4] = box_corner[:, :, 4]

    batch_size = prediction.size(0)
    write = 0

    for i in range(batch_size):
        # Get images form batch i
        img_pred = prediction[i]

        max_conf, max_conf_score = torch.max(img_pred[:, 5:5+class_num], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        img_pred = torch.cat((img_pred[:, :5], max_conf, max_conf_score), 1)

        non_zero_id = torch.nonzero(img_pred[:, 4])
        try:
            img_pred_ = img_pred[non_zero_id, :].view(-1, 7)
        except:
            continue

        if img_pred_.shape(0) == 0:
            continue

        # Class number of the images in the batch
        img_classes = unique(img_pred_[:, -1])

        for cls in img_classes:
            # Get a particular class
            cls_mask = img_pred_*(img_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_idx = torch.nonzero(cls_mask[:, -2]).squeeze()
            img_pred_class = img_pred_[class_mask_idx].view(-1, 7)

            # Sort detections so the maximum is at the top
            conf_sort_idx = torch.sort(img_pred_class[:, 4], descending=True)[1]
            img_pred_class = img_pred_class[conf_sort_idx]
            idx = img_pred_class.size(0)

            # Perform NMS
            for ind in range(idx):
                # Get all IOUS for boxes
                try:
                    IOUs = bbox_IOU(img_pred_class[ind].unsqueeze(0),
                                                        img_pred_class[ind+1:])
                except ValueError:
                    break
                except IndexError:
                    break

                # Remove b.boxes when iou < nms_conf
                IOU_mask = (IOUs < nms_conf).float().unsequeeze(1)
                img_pred_class[ind+1:] *= IOU_mask
                non_zero_idx = torch.nonzero(img_pred_class[:, 4]).squeeze()
                img_pred_class = img_pred_class[non_zero_idx].view(-1, 7)

            # Repeat the batch_id for as many detections of the class cls in the
            # image
            batch_ind = img_pred_class.new(img_pred_class.size(0), 1).fill_(
                i)
            seq = batch_ind, img_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = True

            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

        try:
            # in case that output is empty
            return output

        except:
            return 0

def unique(tensor):
    """
        Args:
             tensor    : (tensor) input tensor
        Returns:
             Tensor used the method numpy.unique()
    """
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def bbox_IOU(box1, box2):
    '''
        Args:
             box1     : (tensor) coordinates of box1
             box2     : (tensor) coordinates of box2
        Returns:
             The IOU between box1 and box2
    '''
    # b.box coordinates
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], \
                                                                 box1[:, 3]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], \
                                                                 box2[:, 3]

    # Coordinates of internal rectangle
    inter_rect_x1 = torch.max(box1_x1, box2_x1)
    inter_rect_y1 = torch.max(box1_y1, box2_y1)
    inter_rect_x2 = torch.max(box1_x2, box2_x2)
    inter_rect_y2 = torch.max(box1_y2, box2_y2)

    # Intersection area
    inter_s = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0)*torch.clamp(
                          inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # Union area
    box1_s = (box1_x2 - box1_x1 + 1)*(box1_y2 - box1_y1 + 1)
    box2_s = (box2_x2 - box2_x1 + 1)*(box2_y2 - box2_y1 + 1)

    return inter_s/(box1_s + box2_s - inter_s)

