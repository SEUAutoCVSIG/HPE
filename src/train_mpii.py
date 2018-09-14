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
from PIL import Image
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
# mpiidataset.scale(20, 0.75, save=True)
# mpiidataset.scale(21, 0.75, save=True)
# mpiidataset.scale(22, 0.75, save=True)
# mpiidataset.heatmap(4, 1, 10)
# mpiidataset.rotate(20, 30, True)
data = mpiidataset[20]
data.show()
# Model
model = StackedHourglass(chan_out)

# Loss Function
loss = nn.MSELoss(size_average=False, reduce=False)

# SGD Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# for data in dataloader:
#     output = model.forward(data)
#     print(output)
#     break


def train(Model, DataLoader, epochs, batch_size, learn_rate, momentum):
    pass


# def train_(Dataloader, Model, Loss, Optimizer):
#     for i in range(0, mpiidataset.num_img):
#         output = model.forward(data)
#     print(output)
#     # Save the model
#     torch.save(model.state_dict(), 'data.SHdata')


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
    data_loader = DataLoader(MpiiDataset(FolderPath, Annotation, transforms), batch_size=
                                              batch_size, shuffle=True)

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=momentum,
                          weight_decay=decay)

    # Loss Function
    loss = nn.MSELoss(size_average=False, reduce=False)

    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Train process
    for epoch in range(epochs):
        for i, (canvas, target) in enumerate(data_loader):
            canvas = Variable(canvas.type(Tensor))
            target = Variable(target.type(Tensor), requires_grad=False)

            optimizer.zero_grad()

            loss = model(canvas, target)
            loss.backward()
            optimizer.step()

        # Output train info
        print('[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, \
                                                    cls %f, total %f, recall: %.5f]' %
                                        (epoch, epochs, batch_size, len(data_loader),
               model.losses['loss_x'], model.losses['loss_y'], model.losses['loss_w'],
               model.losses['loss_h'], model.losses['loss_conf'], model.losses['loss_cls'],
               loss.item(), model.losses['recall']))

        if epoch % check_point == 0:
            torch.save(model, 'sh.pkl')
if __name__ == '__main__':
    # Model
    model = StackedHourglass(chan_out)

    if torch.cuda.is_available():
        model.cuda()

    model.train()

    train(model, FolderPath, Annotation, 100, 64, 0.001, 0.9, 0.0005, 10,
          "yolov3-1.weights")