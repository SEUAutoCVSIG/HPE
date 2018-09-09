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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os

'''
You need to change the following parameters according to your own condition.
'''
FolderPath = '/Users/midora/Desktop/Python/HPElocal/res/images'  # My local path
Label = '/Users/midora/Desktop/Python/HPEOnline/res/mpii_human_pose_v1_u12_1.mat'
chan_out = 1

# Dataset
mpii = Mpii(FolderPath, Label)

# Model
model = StackedHourglass(chan_out)

# Loss Function


# SGD Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


def train(Dataset, Model, Loss, Optimizer):
    # for i in range(0, mpii.amount):
    output = model.forward(mpii.loadimg())
    print(output)
    # Save the model
    torch.save(model.state_dict(), 'src.modeldata')
