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
from math import sqrt

'''********************************************************************'''
'''
You need to change the following parameters according to your own condition.
'''
# My local path
FolderPath = '/Users/midora/Desktop/Python/HPElocal/res/images'
Annotation = '/Users/midora/Desktop/Python/HPE/res/mpii_human_pose_v1_u12_1.mat'
WeightPath = '/Users/midora/Desktop/Python/HPE/data/'

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
    dataset = MpiiDataSet_sig(FolderPath, Annotation)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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
    correct = np.zeros((16, 2))
    for epoch in range(epochs):
        for i, (idx, data, target) in enumerate(data_loader):
            # print(idx.shape)
            # print(idx[0])
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
            zero = np.zeros(16)
            correct += if_correct(idx, dataset, output, target, batch_size)
            accuracy = correct[:, 0]/correct[:, 1]
            tt_acc = np.sum(accuracy)/len(accuracy)

            # Output train info
            print('[Epoch %d/%d, Batch %d/%d] [Loss: ' % (epoch+1, epochs, i+1, len(data_loader)), end='')
            # for part in range(len(parts)):
            #     part_target = torch.cat((target[:, part, :, :], target[:, part + 16, :, :]), 1)
            #     part_output = torch.cat((output[:, part, :, :], output[:, part + 16, :, :]), 1)
            #     loss_ = loss_func(part_output, part_target)
            #     print('%s %f ' % (parts[part], loss_), end='')
            print('total %f     Accuracy %f]' % (loss, tt_acc))

        if epoch % check_point == 0:
            torch.save(model.state_dict(), weight_file_name)


def if_correct(idx, dataset, output, target, batch_size):
    correct = np.zeros((16, 2))  # [correct, total]
    op_coor = np.zeros((16, 2))
    tg_coor = np.zeros((16, 2))
    for img in range(batch_size):
        num_part = dataset.get_num_part(int(idx[img]))
        norm = dataset.get_norm(int(idx[img]))
        print('norm = ', norm)
        for part in range(num_part):
            part_output = output[img, part + num_part, :, :]
            part_target = target[img, part + num_part, :, :]
            op_coor[part][0], op_coor[part][1] = np.where(part_output == part_output.max())
            if part_target.max() != 0:
                tg_coor[part][0], tg_coor[part][1] = np.where(part_target == part_target.max())
        dis = op_coor - tg_coor
        dis = dis**2
        dis = dis[:, 0] + dis[:, 1]
        for part in range(num_part):
            if dataset.get_visibility(int(idx[img]), part) != -1:
                correct[part, 1] += 1
                if sqrt(dis[part]) <= norm:
                    correct[part, 0] += 1
    return correct




if __name__ == '__main__':
    weight_file_name = WeightPath+"stacked_hourglass.pkl"
    # Model
    model = StackedHourglass(16)
    if os.path.isfile(weight_file_name):
        model.load_state_dict(torch.load(weight_file_name))
    if torch.cuda.is_available():
        model.cuda()

    model.train()

    train(model, FolderPath, Annotation, epochs=100, batch_size=2, learn_rate=2.5e-4, momentum=0.9, decay=0.0005,
          check_point=1, weight_file_name=weight_file_name)
