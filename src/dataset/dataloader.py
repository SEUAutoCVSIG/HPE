# -*- coding: utf-8 -*-
'''
    Created on Mon Sep 08 20:45 2018

    Author          ：Yu Du
    Email           : 1239988498@qq.com
    Last edit date  : Tue Sep 06 01:09 2018

South East University Automation College, 211189 Nanjing China
'''
import random
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms as T
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.misc as misc
import h5py
from src.dataset.mpii import Mpii
import cv2

keys = ['index', 'person', 'imgname', 'center', 'scale', 'part', 'visible', 'normalize', 'torsoangle', 'multi',
        'istrain']


class People:
    '''
    Obj: people in the image
    '''

    def __init__(self, mpii, idx, idx_pp):
        self.parts = np.zeros((mpii.num_part, 2))
        self.partname = mpii.parts
        self.visible = -np.ones(mpii.num_part)
        self.istrain = mpii.isTrain(idx)
        if self.istrain:
            self.objpos, self.scale = mpii.location(idx, idx_pp)
            self.normalize = mpii.normalization(idx, idx_pp)
            self.torsoangle = mpii.torsoangle(idx, idx_pp)
            for part in range(mpii.num_part):
                self.parts[part], self.visible[part] = mpii.partinfo(idx, idx_pp, part)

    def getjoint(self, part):
        if type(part) == type(''):
            part = self.partname.index(part)
        return self.parts[part], self.visible[part]


class DataContainer:
    '''
    As a container to store information of one image
    '''

    def __init__(self, mpii, idx):
        self.idx = idx
        self.imgname = mpii.getname(idx)
        self.num_pp = mpii.num_pp(idx)
        self.istrain = mpii.isTrain(idx)
        self.peoples = []
        for idx_pp in range(self.num_pp):
            self.peoples += [People(mpii, idx, idx_pp)]

    def getpeople(self, idx_pp):
        return self.peoples[idx_pp]

    def getjoint(self, idx_pp, part):
        '''
        param part: both index of part and part name is OK
        return: (numpy.array(1,2), int)
        '''
        return self.peoples[idx_pp].getjoint(part)

class MpiiDataset(data.Dataset):
    '''
    Providing a set of containers and methods to handle image
    '''

    def __init__(self, imageFolderPath, annoPath, transforms=None):
        super(MpiiDataset, self).__init__()
        mpii = Mpii(imageFolderPath, annoPath)
        self.num_img = mpii.num_img
        self.containers = []
        self.transforms = transforms
        for imidx in range(self.num_img):
            self.containers += [DataContainer(mpii, imidx)]

    def __getitem__(self, idx):
        '''
        return: PIL Image
        obj[idx] == obj.__getitem__(idx) (e.g. obj: MpiiDataset[idx] return PIL Image type)
        '''
        try:
            fname = self.containers[idx].imgname
            data = Image.open(fname, "r")
            if self.transforms:
                data = self.transforms(data)
            return data
        except:
            # If failed to load the pointed image, using a random image
            new_idx = random.randint(0, self.num_img - 1)
            return self[new_idx]

    def __len__(self):
        return self.num_img

    def loadimg(self, idx):
        '''
        Return: PIL Image
        '''
        if (idx >= self.num_img):
            return 0
        fname = self.containers[idx].imgname
        return Image.open(fname, "r")

    def Im2Tensor(self, idx):
        '''
        return: Tensor in range(0, 1) of image(for *.jpg the number of channel is 3)
        "return 0" means one epoch has been completed.
        '''
        if (idx >= self.num_img):
            print('Index out od range!')
            return 0
        img = self.loadimg(idx)
        imgnp = np.array(img)
        imgTen = T.ToTensor(imgnp)
        return imgTen

    def getpeople(self, idx, idx_pp):
        return self.containers[idx].getpeople(idx_pp)

    def calcul_heatmap(self, img_width, img_height, c_x, c_y, sigma):
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

    def heatmap(self, idx, idx_pp, part):
        img = cv2.imread(self.containers[idx].imgname)
        img = img[:, :, ::-1]  # I cannot see the difference, when I change the last argument
        height, width, _ = np.shape(img)
        [joint_x, joint_y], _ = self.containers[idx].getjoint(idx_pp, part)
        if joint_x > 0 and joint_y > 0:
            heatmap = self.calcul_heatmap(width, height, joint_x, joint_y, 25)
            plt.imshow(heatmap)
            plt.show()
