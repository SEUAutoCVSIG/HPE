# -*- coding: utf-8 -*-
'''
    Created on Mon Sep 04 19:32 2018

    Author          ：Yu Du
    Email           : 1239988498@qq.com
    Last edit date  :

South East University Automation College, 211189 Nanjing China
'''

from src.model.hourglass import StackedHourglass
from src.dataset.dataloader import MpiiDataset
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models
from torchvision import transforms as T
from torch.utils.data import DataLoader
import time
import os

# You need to change the following parameters according to your own condition.

FolderPath = '/Users/midora/Desktop/Python/HPElocal/res/images'  # My local path
Label = '/Users/midora/Desktop/Python/HPEOnline/res/mpii_human_pose_v1_u12_1.mat'

'''********************************************************************'''
'''
You need to change the following parameters according to your own condition.
'''
# My local path
FolderPath = '/Users/midora/Desktop/Python/HPElocal/res/images'
Annotation = '/Users/midora/Desktop/Python/HPEOnline/res/mpii_human_pose_v1_u12_1.mat'

chan_out = 1
epochs = 1
batch_size = 1
shuffle = True
sampler = None
num_workers = 0
transforms = T.Compose([
    T.Resize(256),
    T.ToTensor()
])
'''********************************************************************'''

# Dataset
mpiidataset = MpiiDataset(FolderPath, Annotation, transforms)
dataloader = DataLoader(
    mpiidataset,
    batch_size,
    shuffle,
    # sampler,
    # num_workers,
)
mpiidataset.scale(20, 0.75, save=True)
mpiidataset.scale(21, 0.75, save=True)
mpiidataset.scale(22, 0.75, save=True)

# Model
model = StackedHourglass(chan_out)

# Loss Function
# loss = nn.MSELoss(size_average=False, reduce=False)

# SGD Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


def train(Model, DataLoader, epochs, batch_size, learn_rate, momentum):
    pass

def train_(Dataloader, Model, Loss, Optimizer):
    for i in range(0, mpiidataset.num_img):
        output = model.forward(dataloader[i])
    print(output)
    # Save the model
    torch.save(model.state_dict(), 'data.SHdata')
