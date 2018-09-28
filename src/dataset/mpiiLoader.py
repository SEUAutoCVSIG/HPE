# -*- coding: utf-8 -*-
'''
    Created on Sun Sep 15 21:14 2018

    Author          ：Yu Du
    Email           : 1239988498@qq.com
    Last edit date  :  2018

South East University Automation College, 211189 Nanjing China
'''
from src.dataset.mpii import Mpii
import cv2
import random
from torch.utils import data
import numpy as np
import torch
import os
from math import sin, cos, radians
from src.utils import calcul_heatmap


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
        return  : (numpy.ndarray) heatmap(16, 64, 64)
        '''
        if self.isFirstLoad:
            # The origin data must be pre-process first.
            self.sqrpadding()
            self.gen_heatmap()
        else:
            heatmap = np.zeros((self.num_part, 64, 64))
            for part in range(self.num_part):
                if self.visible[part] != -1:
                    heatmap[part] = calcul_heatmap(64, 64, self.parts[part][0] * self.scale / 4,
                                                   self.parts[part][1] * self.scale / 4, 1)
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
        self.scale = 256 / max_
        if new_height > new_width:
            left = (new_height - new_width) // 2
            right = abs(new_height - new_height - left)
            # print('height = %f width = %f left = %f right = %f' % (new_height, new_width,left, right))
            img = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
            for part in range(self.num_part):
                if self.parts[part, 0] != 0 and self.isFirstLoad:
                    self.parts[part, 0] += left
        elif new_height < new_width:
            top = (new_width - new_height) // 2
            bottom = abs(new_width - new_height - top)
            # print('height = %f width = %f  top = %f bottom = %f' % (new_height, new_width, top, bottom))
            img = cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(128, 128, 128))
            for part in range(self.num_part):
                if self.parts[part, 1] != 0 and self.isFirstLoad:
                    self.parts[part, 1] += top
        if max_ > 256:
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        if self.isFirstLoad:
            self.isFirstLoad = False
        return img

    def get_norm(self):
        return self.normalize * self.scale

    def get_parts(self):
        return self.parts * self.scale

    def get_cropimg(self):
        deta_x = 0
        deta_y = 0
        img = cv2.imread(self.imgname)
        height, width = img.shape[:2]
        (x1, y1) = self.coor1
        (x2, y2) = self.coor2
        img = img[max(y1, 0):min(y2, height), max(x1, 0):min(x2, width)]
        if x1 > 0:
            deta_x = -x1
        if y1 > 0:
            deta_y = -y1
        return img, deta_x, deta_y


class MpiiDataSet_sig(data.Dataset):
    '''
    Containing infomation of images cropped into one person
    '''

    def __init__(self, imageFolderPath, annoPath, if_train=True, is_eval=False, is_augment=False):
        super(MpiiDataSet_sig, self).__init__()
        self.mpii = Mpii(imageFolderPath, annoPath)
        self.num_person = 0
        self.containers = []  # dtype: Person
        self.imageFolderPath = imageFolderPath
        self.if_train = if_train
        self.is_eval = is_eval
        self.is_augment = is_augment
        self.train_len =26112
        count = 0
        for imgidx in range(self.mpii.num_img):
            for idx_pp in range(self.mpii.num_pp(imgidx)):
                if self.mpii.isTrain(imgidx):
                    self.add_person(imgidx, idx_pp)
                    count += 1
            if count >= self.train_len and if_train:
                break

    def __getitem__(self, idx):
        '''
        return  : depends:
                if PIL=True : return PIL Image, heatmap
                else    : return numpy.ndarray, heatmap
                heatmap : numpy.ndarray(16, 1024, 1024)
                obj[idx] == obj.__getitem__(idx)
        (e.g. obj   : MpiiDataset[idx] return PIL Image, heatmap or numpy.ndarray, heatmap)
        '''
        if self.is_eval:
            idx += self.train_len
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
            img = torch.from_numpy(img).float() / 255
            heatmap = torch.from_numpy(heatmap.swapaxes(1, 2)).repeat(2, 1, 1)
            return idx, img, heatmap
        else:
            img_ = img.swapaxes(1, 2).swapaxes(0, 1)
            img_ = torch.from_numpy(img_).float() / 255
            # heatmap = self.containers[idx].gen_heatmap()
            heatmap = torch.from_numpy(heatmap.swapaxes(1, 2)).repeat(2, 1, 1)
            if self.is_eval:
                return img_, heatmap
            else:
                return idx, img_, img, heatmap

    def __len__(self):
        if self.is_eval:
            return self.num_person-self.train_len
        else:
            return self.num_person

    def add_person(self, imgidx, idx_pp):
        self.containers += [Person(self.mpii, imgidx, idx_pp)]
        self.num_person += 1


    def add_person_aug(self, idx):
        pass

    def get_target(self, idx):
        target = torch.FloatTensor(len(idx), 32, 128, 128)
        for i in range(len(idx)):
            heatmap = self.containers[int(idx[i])].gen_heatmap()
            heatmap = torch.from_numpy(heatmap.swapaxes(1, 2)).repeat(2, 1, 1)
            target[i] = heatmap
        return target

    def get_padimg(self, idx):
        return self.containers[idx].sqrpadding()

    def get_parts(self, idx):
        return self.containers[idx].get_parts()

    def get_num_part(self, idx):
        return self.containers[idx].num_part

    def get_visibility(self, idx, part):
        return self.containers[idx].visible[part]

    def get_norm(self, idx):
        return self.containers[idx].get_norm()

    def get_cropimg(self, idx):
        return self.containers[idx].get_cropimg()

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
            imgpath = path + str(idx) + '_' + str(i) + '.jpg'
            if not os.path.isfile(imgpath):
                break
            i += 1
        cv2.imwrite(imgpath, img)
        print('Save successfully: ', imgpath)

    def rotate(self, idx, degree, save=False):
        img, dx, dy = self.get_cropimg()
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
        return dx, dy

    def scale(self, idx, scaling, save=False):
        img, dx, dy = self.get_cropimg()
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
        return dx, dy
