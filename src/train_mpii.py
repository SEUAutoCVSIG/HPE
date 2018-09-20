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
from src.dataset.mpiiLoader import MpiiDataSet_sig
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models
from torchvision import transforms as T
from torch.utils.data import DataLoader
from PIL import Image
import time
import os

'''********************************************************************'''
'''
You need to change the following parameters according to your own condition.
'''
# My local path
FolderPath = '/Users/midora/Desktop/Python/HPElocal/res/images'
Annotation = '/Users/midora/Desktop/Python/HPEOnline/res/mpii_human_pose_v1_u12_1.mat'
WeightPath = '/Users/midora/Desktop/Python/HPEOnline/data/'

'''********************************************************************'''


def train(model, FolderPath, Annotation, epochs, batch_size, learn_rate, momentum,
          decay, check_point, weight_file_name):
    '''
        Args:
             model        : (nn.Module) untrained darknet
             root         : (string) directory of root
             list_dir     : (string) directory to list file
             epochs      : (int) max epoches
             batch_size   : (int) batch size
             learn_rate   : (float) learn rate
             momentum     : (float) momentum
             decay        : (float) weight decay
             check_point  : (int) interval between weights saving
             weight_file_name
                          : (string) name of the weight file
        Returns:
             Output training status and save weight
    '''
    # Define data loader
    data_loader = DataLoader(MpiiDataSet_sig(FolderPath, Annotation),
                             batch_size=batch_size, shuffle=True)

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=momentum,
                          weight_decay=decay)

    # Loss Function
    loss_func = nn.MSELoss()

    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    parts = ['rank', 'rkne', 'rhip',
             'lhip', 'lkne', 'lank',
             'pelv', 'thrx', 'neck', 'head',
             'rwri', 'relb', 'rsho',
             'lsho', 'lelb', 'lwri']

    # Train process
    for epoch in range(epochs):
        for i, (data, target) in enumerate(data_loader):
            # print(data)
            # print(target)
            # print('data = ', data.shape)
            # print('target = ', target.shape)
            data = Variable(data.type(Tensor))
            target = Variable(target.type(Tensor), requires_grad=False)
            optimizer.zero_grad()
            output = model(data)
            # print(output)
            # print('output = ', output.shape)
            # print('target = ', target.shape)
            loss = loss_func(output, target)
            # print(loss)
            # print(type(loss))
            loss.backward()
            optimizer.step()

            # Output train info
            print('[Epoch %d/%d, Batch %d/%d] [Loss: ' % (epoch+1, epochs, i+1, len(data_loader)), end='')
            for part in range(len(parts)):
                part_target = torch.cat((target[:, part, :, :], target[:, part + 16, :, :]), 1)
                part_output = torch.cat((output[:, part, :, :], output[:, part + 16, :, :]), 1)
                loss_ = loss_func(part_output, part_target)
                print('%s %f ' % (parts[part], loss_), end='')
            print('total %f]' % loss)

        if epoch % check_point == 0:
            torch.save(model.state_dict(), weight_file_name)




if __name__ == '__main__':
    weight_file_name = WeightPath+"stacked_hourglass.pkl"
    # Model
    model = StackedHourglass(16)
    if os.path.isfile(weight_file_name):
        model.load_state_dict(torch.load(weight_file_name))
    if torch.cuda.is_available():
        model.cuda()

    model.train()

    train(model, FolderPath, Annotation, epochs=100, batch_size=1, learn_rate=2.5e-4, momentum=0.9, decay=0.0005,
          check_point=1, weight_file_name=weight_file_name)
