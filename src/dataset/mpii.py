# -*- coding: utf-8 -*-
'''
    Created on Mon Sep 03 22:14 2018

    Author          ：Yu Du
    Email           : 1239988498@qq.com
    Last edit date  :

South East University Automation College, 211189 Nanjing China
'''

import torch
from PIL import Image
import numpy as np


# imagepath = '/Users/midora/Desktop/Python/HPElocal/res/images/' # My local path

class Mpii:
    def __init__(self, imageFolderPath, ofile='res/images.txt'):
        '''
        param imageFolderPath: The path of MPII Folder
        param ofile: It may not work well in Windows, so you need to give it yourself
        '''
        self.folder = imageFolderPath
        self.ofile = ofile
        self.idx = 0
        self.image = ['0']  # This looks so stupid
        with open(self.ofile, 'r') as ofileobj:
            isFirst = True
            for fname in ofileobj:
                fname = fname.rstrip()
                if (isFirst):  # As well as this
                    self.image[0] = self.folder + '/' + fname
                    isFirst = False
                self.image.append(self.folder + '/' + fname)
                '''
                self.image is a list whose elements are full path of one image
                '''
        self.amount = len(self.image)

    def loadimg(self):
        '''
        return: Tensor of image(for *.jpg the number of channel is 3)
        "return 0" means one epoch has been completed.
        '''
        if (self.idx == self.amount):
            return 0
        img = Image.open(self.image[self.idx], "r")
        self.idx += 1
        imgnp = np.array(img)
        imgTen = torch.from_numpy(imgnp)
        return imgTen
