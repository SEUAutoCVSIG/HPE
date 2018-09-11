# -*- coding: utf-8 -*-
'''
    Created on wed Sept 8 14：59 2018

    Author           : Yue Han, Shaoshu Yang
    Email            : 1015985094@qq.com
                       13558615057@163.com
    Last edit date   : Sept 8 14:59 2018

South East University Automation College, 211189 Nanjing China

The following codes referenced Ayoosh Kathuria's blog:
How to implement a YOLO (v3) object detector from strach in
PyTorch: Part 5
'''

from __future__ import division
import time
import numpy as np
import cv2
from src.utils import *
import argparse
import os
import os.path
from src.model.darknet import darknet
import pickle as pkl
import pandas as pd
import random

class detector():
    def __init__(self, model):
        '''
            Args:
                 model      : (nn.Module) darknet that loaded weights
        '''
        self.model = model
        self.img_size = model.img_size

    def detect(self, img):
        '''
            Args:
                 img        : (ndarray) img matrix from cv2.imread(),
                              If you want to use plt.imread() or other
                              RGB format method, ensure to transform
                              from RGB to BGR, HWC to CHW
            Returns:
                 Prediction bounding-boxes
        '''
        # Get input dimensions
        img_h, img_w = img.shape[1], img.shape[2]

        new_h = int(img_h*min(self.img_size/img_h, self.img_size/img_w))
        new_w = int(img_w*min(self.img_size/img_h, self.img_size/img_w))

        pad_h = (self.img_size - new_h)//2
        pad_w = (self.img_size - new_w)//2

        # Pre-processing
        img_ = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        canvas = np.full((self.img_size, self.img_size, 3), 128)
        canvas[(self.img_size - new_h)//2:(self.img_size - new_h)//2 + new_h,
               (self.img_size - new_w)//2:(self.img_size - new_w)//2 + new_w,
                :] = img
        canvas = canvas[:, :, ::-1].transpose(2, 0, 1)

        # Normalization
        canvas = torch.FloatTensor(canvas).div(255.0).unsqueeze(0)

        # Make prediction and transform the prediction to the original scale
        prediction = self.model(canvas)
        prediction[:, 0] -= prediction[:, 2]/2
        prediction[:, 1] -= prediction[:, 3]/2
        prediction[:, 2] += prediction[:, 0]
        prediction[:, 3] += prediction[:, 1]

        prediction[:, [0, 2]] -= pad_w
        prediction[:, [1, 3]] -= pad_h
        prediction[:, [0, 2]] *= img_w/new_w
        prediction[:, [1, 3]] *= img_h/new_h

        return prediction[:, :4]

    def detect_test(self, img, waitkey):
        '''
                    Args:
                         img        : (ndarray) img matrix from cv2.imread(),
                                      If you want to use plt.imread() or other
                                      RGB format method, ensure to transform
                                      from RGB to BGR, HWC to CHW
                         waitkey    : (int) input for cv2.waitKey()
                    Returns:
                         Prediction bounding-boxes
                '''
        # Get input dimensions
        prediction = self.detect(img)

        for prediction_ in prediction:
            coord1 = tuple(map(int, prediction_[:2]))
            coord2 = tuple(map(int, prediction_[2:4]))
            cv2.rectangle(img, coord1, coord2, (0, 255, 0), 2)

        cv2.imshow('prediction.jpg', img)
        cv2.waitKey(waitkey)
