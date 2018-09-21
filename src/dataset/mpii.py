# -*- coding: utf-8 -*-
'''
    Created on Mon Sep 03 22:14 2018

    Author          ：Yu Du
    Email           : 1239988498@qq.com
    Last edit date  : Tue Sep 08 20:45 2018

South East University Automation College, 211189 Nanjing China
'''

import torch
from PIL import Image
import numpy as np
import scipy.io as sio


class Mpii:
    '''
    MPII Dataset
    Iterator has already been a member variable
    '''

    def __init__(self, imageFolderPath, annoPath):
        self.imageFolderPath = imageFolderPath
        # Load in annotation
        self.anno = sio.loadmat(annoPath)['RELEASE']
        self.num_img = self.anno['img_train'][0][0][0].shape[0]
        # Index of Part
        self.parts = ['rank', 'rkne', 'rhip',
                      'lhip', 'lkne', 'lank',
                      'pelv', 'thrx', 'neck', 'head',
                      'rwri', 'relb', 'rsho',
                      'lsho', 'lelb', 'lwri']
        self.num_part = len(self.parts)

    def getfullpath(self, idx):
        return self.imageFolderPath + '/' + str(self.anno['annolist'][0][0][0]['image'][idx][0]['name'][0][0])

    def getimgname(self, idx):
        return str(self.anno['annolist'][0][0][0]['image'][idx][0]['name'][0][0])


    def isTrain(self, idx):
        '''
        Should be used before other functions
        Return: bool type
        '''
        return (self.anno['img_train'][0][0][0][idx] and
                self.anno['annolist'][0][0][0]['annorect'][idx].size > 0 and
                'annopoints' in self.anno['annolist'][0][0][0]['annorect'][idx].dtype.fields)

    def num_pp(self, idx):
        '''
        Return: Number of people showed in current image
        '''
        example = self.anno['annolist'][0][0][0]['annorect'][idx]
        if len(example) > 0:
            return len(example[0])
        else:
            return 0

    def location(self, idx, idx_pp):
        '''
        param idx_pp: index for people in this image
        return: (object position(np.array), scale(float))
        '''
        example = self.anno['annolist'][0][0][0]['annorect'][idx]
        if ((not (example.dtype.fields is None)) and
                'scale' in example.dtype.fields and
                example['scale'][0][idx_pp].size > 0 and
                example['objpos'][0][idx_pp].size > 0):
            scale = example['scale'][0][idx_pp][0][0]
            x = example['objpos'][0][idx_pp][0][0]['x'][0][0]
            y = example['objpos'][0][idx_pp][0][0]['y'][0][0]
            return np.array([x, y]), scale
        else:
            # If there is no "scale" and "object position" data
            return [-1, -1], -1

    def partinfo(self, idx, idx_pp, part):
        '''
        param idx_pp: index for people in this image
        param part: both index of part and part name is OK
        return: (joint position(np.array), isVisible(bool))
        '''
        # unify the parameter's type
        if type(part) == type(''):
            part = self.parts.index(part)
        example = self.anno['annolist'][0][0][0]['annorect'][idx]
        if example['annopoints'][0][idx_pp].size > 0:
            parts_info = example['annopoints'][0][idx_pp][0][0][0][0]
            for i in range(len(parts_info)):
                if parts_info[i]['id'][0][0] == part:
                    # Since the original annotation is confused
                    if 'is_visible' in parts_info.dtype.fields:
                        v = parts_info[i]['is_visible']
                        v = v[0][0] if len(v) > 0 else 1
                        if type(v) is str:
                            v = int(v)
                    else:
                        v = 1
                    return np.array([parts_info[i]['x'][0][0], parts_info[i]['y'][0][0]], int), v
            return np.zeros(2, int), 0
        # return -np.ones(2, int), -1
        return np.zeros(2, int), -1

    def normalization(self, idx, idx_pp):
        # Get head height for distance normalization
        if self.isTrain(idx):
            example = self.anno['annolist'][0][0][0]['annorect'][idx]
            x1, y1 = int(example['x1'][0][idx_pp][0][0]), int(example['y1'][0][idx_pp][0][0])
            x2, y2 = int(example['x2'][0][idx_pp][0][0]), int(example['y2'][0][idx_pp][0][0])
            diff = np.array([y2 - y1, x2 - x1], np.float)
            # Calculate norm 2(Why?)
            return np.linalg.norm(diff) * .6
        return -1

    def torsoangle(self, idx, idx_pp):
        # Get angle from pelvis to thorax(angle of the body), 0 means the torso is up vertically
        pt1 = self.partinfo(idx, idx_pp, 'pelv')[0]
        pt2 = self.partinfo(idx, idx_pp, 'thrx')[0]
        if not (pt1[0] - pt2[0] == 0):
            return 90 + np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]) * 180. / np.pi
        else:
            return 0
