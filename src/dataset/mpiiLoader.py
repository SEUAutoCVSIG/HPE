# -*- coding: utf-8 -*-
'''
    Created on Sun Sep 15 21:14 2018

    Author          ：Yu Du
    Email           : 1239988498@qq.com
    Last edit date  : Tue Sep 20 11:10 2018

South East University Automation College, 211189 Nanjing China
'''
from src.dataset.mpii import Mpii
import cv2
import random
from torch.utils import data
from PIL import Image
import numpy as np
import torch



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
        self.scale = 1
        self.isFirstLoad = True
        if self.istrain:
            self.objpos, self.scale = mpii.location(idx, idx_pp)
            self.normalize = mpii.normalization(idx, idx_pp)
            self.torsoangle = mpii.torsoangle(idx, idx_pp)
            for part in range(mpii.num_part):
                self.parts[part], self.visible[part] = mpii.partinfo(idx, idx_pp, part)
            halfside = int(self.scale * 105)
            self.coor1 = (self.objpos[0] - halfside, self.objpos[1] - halfside)
            self.coor2 = (self.objpos[0] + halfside, self.objpos[1] + halfside)

    def getjoint(self, part):
        if type(part) == type(''):
            part = self.partname.index(part)
        return self.parts[part], self.visible[part]

    def gen_heatmap(self):
        '''
        return  : (numpy.ndarray) heatmap(16, 128, 128)
        '''
        heatmap = np.zeros((self.num_part, 128, 128))
        for part in range(self.num_part):
            # if not(self.visible[part] == 0 or self.visible[part] == -1):
            if self.visible[part] != -1:
                heatmap[part] = calcul_heatmap(128, 128, self.parts[part][0]*self.scale/2, self.parts[part][1]*self.scale/2, 1)
        return heatmap

    def sqrpadding(self):
        '''
        return  : (numpy.ndarray) image open with cv2 then padded to square in gray(128)
        '''
        img = cv2.imread(self.imgname)
        height, width = img.shape[:2]
        (x1, y1) = self.coor1
        (x2, y2) = self.coor2
        img = img[max(y1, 0):min(y2, height), max(x1, 0):min(x2, width)]
        if x1 > 0 and self.isFirstLoad:
            self.parts[:, 0] -= x1
        if y1 > 0 and self.isFirstLoad:
            self.parts[:, 1] -= y1
        new_height, new_width = img.shape[:2]
        max_ = max(new_width, new_height)
        self.scale = 256/max_
        if new_height > new_width:
            left = (new_height - new_width) // 2
            right = new_height - new_height - left
            img = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
            for part in range(self.num_part):
                if self.parts[part, 0] != 0 and self.isFirstLoad:
                    self.parts[part, 0] += left
        elif new_height < new_width:
            top = (new_width - new_height) // 2
            bottom = new_width - new_height - top
            img = cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(128, 128, 128))
            for part in range(self.num_part):
                if self.parts[part, 1] != 0 and self.isFirstLoad:
                    self.parts[part, 1] += top
        if max_ > 256:
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        # self.size = 1280
        # top = (self.size - new_height) // 2
        # left = (self.size - new_width) // 2
        # bottom = self.size - new_height - top
        # right = self.size - new_width - left
        # img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        # self.parts = self.parts*256/self.max_
        # self.normalize = self.normalize*256/self.max_
        self.isFirstLoad = False
        return img

    def get_norm(self):
        return self.normalize*self.scale

    def get_parts(self):
        return self.parts*self.scale

class MpiiDataSet_sig(data.Dataset):
    def __init__(self, imageFolderPath, annoPath, if_train=True):
        super(MpiiDataSet_sig, self).__init__()
        self.mpii = Mpii(imageFolderPath, annoPath)
        self.num_person = 0
        self.containers = []  # dtype: Person
        self.imageFolderPath = imageFolderPath
        self.if_train = if_train
        count = 0
        for imgidx in range(self.mpii.num_img):
        # for imgidx in range(35):
            for idx_pp in range(self.mpii.num_pp(imgidx)):
                if self.mpii.isTrain(imgidx):
                    self.add_person(imgidx, idx_pp)
            #         count += 1
            # if count >= 1:
            #     break


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
            idx = random.randint(0, self.num_person - 1)
            img = self.containers[idx].sqrpadding()
            heatmap = self.containers[idx].gen_heatmap()
        if self.if_train:
            img = img.swapaxes(1, 2).swapaxes(0, 1)
            img = torch.from_numpy(img).float()/255
            heatmap = torch.from_numpy(heatmap.swapaxes(1, 2)).repeat(2, 1, 1)
            return idx, img, heatmap
        else:
            img_ = img.swapaxes(1, 2).swapaxes(0, 1)
            img_ = torch.from_numpy(img_).float() / 255
            heatmap = torch.from_numpy(heatmap.swapaxes(1, 2)).repeat(2, 1, 1)
            return idx, img_, img, heatmap

    def __len__(self):
        return self.num_person

    def add_person(self, imgidx, idx_pp):
        self.containers += [Person(self.mpii, imgidx, idx_pp)]
        self.num_person += 1

    def get_cvimg(self, idx):
        return self.containers[idx].sqrpadding()

    def get_parts(self, idx):
        return self.containers[idx].get_parts()

    def get_num_part(self, idx):
        return self.containers[idx].num_part

    def get_visibility(self, idx, part):
        return self.containers[idx].visible[part]

    def get_norm(self, idx):
        return self.containers[idx].get_norm()