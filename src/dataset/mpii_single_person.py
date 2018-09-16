# -*- coding: utf-8 -*-
'''
    Created on Sun Sep 15 21:14 2018

    Author          ：Yu Du
    Email           : 1239988498@qq.com
    Last edit date  :Mon Sep 16 23:55 2018

South East University Automation College, 211189 Nanjing China
'''
from src.dataset.mpii import Mpii
from src.dataset.dataloader import calcul_heatmap
import cv2
import random
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms as T
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.misc as misc
import os
from math import *


# class DataContainer_sig():
#     '''
#     As a container to store information of the image with one person, one joint by cropping
#     '''
#
#     def __init__(self, mpii, imgidx, coor1, coor2, idx_pp, part):
#         '''
#         Args:
#             mpii    : class Mpii
#             imgidx  : integer (index of image annotation got from .mat file)
#             coor1   : coordinate of the top left corner of the bounding box
#             coor2   : coordinate of the lower right corner of the bounding box
#             idx_pp  : index for people in this image
#             part    : both index of part and part name is OK
#         '''
#         self.imgidx = imgidx
#         self.imgname = mpii.getfullpath(imgidx)  # Full Name
#         self.istrain = mpii.isTrain(imgidx)
#         ori_partcoor, self.isvisible = mpii.partinfo(imgidx, idx_pp, part)
#         self.partcoor = (ori_partcoor[0] - coor1[0], ori_partcoor[1] - coor1[1])
#         self.coor1 = coor1
#         self.coor2 = coor2
#         self.width = coor2[0] - coor1[0]
#         self.height = coor2[1] - coor1[1]
#
#     def gen_heatmap(self):
#         '''
#         return  : (numpy.ndarray) heatmap
#         '''
#         self.partcoor[0] *= 512 / self.width
#         self.partcoor[1] *= 512 / self.height
#         if self.partcoor[0] > 0 and self.partcoor[1] > 0:
#             return calcul_heatmap(512, 512, self.partcoor[0], self.partcoor[1], 1)
#
#     def sqrpadding(self):
#         '''
#         return  : (numpy.ndarray) image open with cv2 then padded to square in gray(128)
#         '''
#         img = cv2.imread(self.imgname)
#         (x1, y1) = self.coor1
#         (x2, y2) = self.coor2
#         img = img[y1:y2, x1:x2]
#         height, width = img.shape[:2]
#         max = height if height > width else width
#         if height > width:
#             bar = (height - width) // 2
#             img = cv2.copyMakeBorder(img, 0, 0, bar, bar, cv2.BORDER_CONSTANT, value=(128, 128, 128))
#             self.partcoor[0] += bar
#         elif height < width:
#             bar = (width - height) // 2
#             img = cv2.copyMakeBorder(img, bar, bar, 0, 0, cv2.BORDER_CONSTANT, value=(128, 128, 128))
#             self.partcoor[1] += bar
#         self.height = self.width = max
#         return img


def calcul_heatmap(img_width, img_height, c_x, c_y, sigma):
    X1 = np.linspace(1, img_width, img_width)
    Y1 = np.linspace(1, img_height, img_height)
    [X, Y] = np.meshgrid(X1, Y1)
    X = X - c_x
    Y = Y - c_y
    D2 = X * X + Y * Y
    E2 = 2.0 * sigma * sigma
    Exponent = D2 / E2
    heatmap = np.exp(-Exponent)
    return heatmap


class Person:
    '''
    Obj: people in the image
    '''

    def __init__(self, mpii, idx, idx_pp):
        self.imgname = mpii.getfullpath(idx)  # Full Name
        self.parts = np.zeros((mpii.num_part, 2))
        self.partname = mpii.parts
        self.num_part = mpii.num_part
        self.visible = -np.ones(mpii.num_part)
        self.istrain = mpii.isTrain(idx)
        if self.istrain:
            self.objpos, self.scale = mpii.location(idx, idx_pp)
            self.normalize = mpii.normalization(idx, idx_pp)
            self.torsoangle = mpii.torsoangle(idx, idx_pp)
            for part in range(mpii.num_part):
                self.parts[part], self.visible[part] = mpii.partinfo(idx, idx_pp, part)
            halfside = int(self.scale * 105)
            self.size = self.side = 2 * halfside
            self.coor1 = (self.objpos[0] - halfside, self.objpos[1] - halfside)
            self.coor2 = (self.objpos[0] + halfside, self.objpos[1] + halfside)

    def getjoint(self, part):
        if type(part) == type(''):
            part = self.partname.index(part)
        return self.parts[part], self.visible[part]

    def gen_heatmap(self):
        '''
        return  : (numpy.ndarray) heatmap(16, 1024, 1024)
        '''
        heatmap = np.zeros((self.num_part, self.size, self.size))
        for part in range(self.num_part)
            if self.visible[part]:
                heatmap[part] = calcul_heatmap(self.size, self.size, self.parts[part][0], self.parts[part][1], 1)
        return heatmap

    def sqrpadding(self):
        '''
        return  : (numpy.ndarray) image open with cv2 then padded to square in gray(128)
        '''
        img = cv2.imread(self.imgname)
        (x1, y1) = self.coor1
        (x2, y2) = self.coor2
        img = img[y1:y2, x1:x2]
        self.size = 1024
        bar = (self.size - self.side) / 2
        img = cv2.copyMakeBorder(img, bar, bar, bar, bar, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        self.parts += bar
        return img


class MpiiDataSet_sig(data.Dataset):
    def __init__(self, imageFolderPath, annoPath, PIL=False):
        super(MpiiDataset, self).__init__()
        self.mpii = Mpii(imageFolderPath, annoPath)
        self.num_person = 0
        self.containers = []  # dtype: Person
        self.imageFolderPath = imageFolderPath
        self.PIL = PIL
        for imgidx in range(self.mpii.num_img):
            for idx_pp in range(self.mpii.num_pp(idx)):
                self.add_person(imgidx, idx_pp)

    def __getitem__(self, idx):
        '''
        return  : depends:
                if PIL=True : return PIL Image, heatmap
                else    : return numpy.ndarray, heatmap
                heatmap : numpy.ndarray(16, 1024, 1024)
                obj[idx] == obj.__getitem__(idx)
        (e.g. obj   : MpiiDataset[idx] return PIL Image, heatmap or numpy.ndarray, heatmap)
        '''
        try:
            img = self.containers[idx].sqrpadding()
            heatmap = self.containers[idx].gen_heatmap()
        except:
            # If failed to load the pointed image, using a random image
            new_idx = random.randint(0, self.num_person - 1)
            img = self.containers[new_idx].sqrpadding()
            heatmap = self.containers[new_idx].gen_heatmap()
        if self.PIL:
            PILimg = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            return PILimg, heatmap
        else:
            return img, heatmap

    def __len__(self):
        return self.num_person

    def add_person(self, imgidx, idx_pp):
        self.containers += [Person(self.mpii, imgidx, idx_pp)]
        self.num_person += 1
