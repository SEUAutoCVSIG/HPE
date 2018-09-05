# -*- coding: utf-8 -*-
'''
    Created on Mon Sep 04 19:32 2018

    Author          ：Yu Du
    Email           : 1239988498@qq.com
    Last edit date  :

South East University Automation College, 211189 Nanjing China
'''

from src.dataset.mpii import Mpii
from src.model.hourglass import StackedHourglass

FolderPath = '/Users/midora/Desktop/Python/HPElocal/res/images'
ImageName = '/Users/midora/Desktop/Python/HPE/res/images.txt'
chan_out = 1

mpii= Mpii(FolderPath, ImageName)
shg = StackedHourglass(chan_out)

