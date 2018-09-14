# -*- coding: utf-8 -*-
'''
    Created on Mon Sep 08 20:45 2018

    Author          ：Yu Du
    Email           : 1239988498@qq.com
    Last edit date  : Tue Sep 10 21:43 2018

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
import os
from math import *

keys = ['index', 'person', 'imgname', 'center', 'scale', 'part', 'visible', 'normalize', 'torsoangle', 'multi',
        'istrain']
imgidlen = 9


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
        self.imgname = mpii.getimgname(idx)  # Just file name
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
        self.imageFolderPath = imageFolderPath
        for imidx in range(self.num_img):
            self.containers += [DataContainer(mpii, imidx)]

    def __getitem__(self, idx):
        '''
        return: PIL Image
        obj[idx] == obj.__getitem__(idx) (e.g. obj: MpiiDataset[idx] return PIL Image type)
        '''
        try:
            PILimage = self.sqrpadding(idx)
        except:
            # If failed to load the pointed image, using a random image
            new_idx = random.randint(0, self.num_img - 1)
            PILimage = self.sqrpadding(new_idx)
        return PILimage.resize((512, 512), Image.ANTIALIAS)

    def __len__(self):
        return self.num_img

    def getfullpath(self, idx):
        return self.imageFolderPath + '/' + self.containers[idx].imgname

    def loadimg(self, idx):
        '''
        Return: PIL Image
        '''
        if (idx >= self.num_img):
            return 0
        fname = self.getfullpath(idx)
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
        img = cv2.imread(self.getfullpath(idx))
        img = img[:, :, ::-1]  # I cannot see the difference, when I change the last argument
        height, width, _ = np.shape(img)
        [joint_x, joint_y], _ = self.containers[idx].getjoint(idx_pp, part)
        if joint_x > 0 and joint_y > 0:
            heatmap = self.calcul_heatmap(width, height, joint_x, joint_y, 1)
            plt.imshow(heatmap)
            plt.show()

    def __makedir(self, path):
        # remove blank
        path = path.strip()
        # remove '\\' at the end (for Windows)
        path = path.rstrip('\\')
        # remove '/' at the end (for Linux)
        path = path.rstrip('/')
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)

    def __saveAugImage(self, idx, img):
        path = self.imageFolderPath[:-len('images')] + 'images_augmentation/'
        self.__makedir(path)
        i = 0
        while True:
            imgpath = path + self.containers[idx].imgname[:-len('.jpg')] + '_' + str(i) + '.jpg'
            if not os.path.isfile(imgpath):
                break
            i += 1
        cv2.imwrite(imgpath, img)
        print('Save successfully: ', imgpath)

    def rotate(self, idx, degree, save=False):
        img = cv2.imread(self.getfullpath(idx))
        height, width = img.shape[:2]
        heightNew = int(width * abs(sin(radians(degree))) + height * abs(cos(radians(degree))))
        widthNew = int(height * abs(sin(radians(degree))) + width * abs(cos(radians(degree))))
        matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
        # I'm working on this step
        matRotation[0, 2] += (widthNew - width) / 2
        matRotation[1, 2] += (heightNew - height) / 2
        img = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(128, 128, 128))
        if save:
            self.__saveAugImage(idx, img)
        else:
            cv2.imshow("imgScaling", img)
            cv2.waitKey(0)

    def scale(self, idx, scaling, save=False):
        img = cv2.imread(self.getfullpath(idx))
        cv2.imshow("img", img)
        if (scaling >= 1):
            img = cv2.resize(img, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_CUBIC)
        elif (scaling <= 0):
            print('Bad Argument')
            return
        else:
            img = cv2.resize(img, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA)
        if save:
            self.__saveAugImage(idx, img)
        else:
            cv2.imshow("imgScaling", img)
            cv2.waitKey(0)

    def sqrpadding(self, idx):
        img = cv2.imread(self.getfullpath(idx))
        height, width = img.shape[:2]
        if height == width:
            return
        elif height > width:
            bar = (height -width)//2
            img = cv2.copyMakeBorder(img, 0, 0, bar, bar, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        else:
            bar = (width - height)//2
            img = cv2.copyMakeBorder(img, bar, bar, 0, 0, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        PILimg = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return PILimg
