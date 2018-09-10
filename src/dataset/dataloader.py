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
import scipy.io as sio
import scipy.misc as misc
import h5py
from src.dataset.mpii import Mpii

keys = ['index', 'person', 'imgname', 'center', 'scale', 'part', 'visible', 'normalize', 'torsoangle', 'multi',
        'istrain']


class DataContainer:
    '''
    As a container to store information of one image
    '''

    def __init__(self, mpii, idx):
        self.idx = idx
        self.imgname = mpii.getname(idx)
        self.num_pp = mpii.num_pp(idx)
        self.istrain = mpii.isTrain(idx)
        self.objpos = []
        self.scale = []
        self.parts = np.zeros((self.num_pp, mpii.num_part, 2))
        self.visible = -np.ones((self.num_pp, mpii.num_part))
        if self.istrain:
            for idx_pp in range(self.num_pp):
                objpos, scale = mpii.location(idx, idx_pp)
                self.objpos += [objpos]
                self.scale += [scale]
                self.normalize = mpii.normalization(idx, idx_pp)
                self.torsoangle = mpii.torsoangle(idx, idx_pp)
                for part in range(16):
                    self.parts[idx_pp][part], self.visible[idx_pp][part] = mpii.partinfo(idx, idx_pp, part)


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
            new_idx = random.randint(0, self.num_img-1)
            return self[new_idx]

    def __len__(self):
        return self.num_img

    def loadimg(self, idx):
        '''
        Return: Current Image
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
