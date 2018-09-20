# -*- coding: utf-8 -*-
'''
    Created on wed Sept 9 22:32 2018

    Author           : Shaoshu Yang
    Email            : 13558615057@163.com
    Last edit date   : Sept 9 22:32 2018

South East University Automation College, 211189 Nanjing China
'''

from src.dataset.coco import *
from src.model.darknet import darknet

import time
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from detect import *

def train(model, root, list_dir, epochs, batch_size, learn_rate, momentum,
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
    data_loader = torch.utils.data.DataLoader(COCO(root, list_dir), batch_size=
                                              batch_size, shuffle=False)

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=momentum,
                          weight_decay=decay)

    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    losses = []
    file = open("loss_recorder.txt", 'a+')

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
                                        (epoch, epochs, i, len(data_loader),
               model.losses['x'], model.losses['y'], model.losses['w'],
               model.losses['h'], model.losses['conf'], model.losses['cls'],
               loss.item(), model.losses['recall']))
            file.write("loss: %f, recall: %f\n"%(loss.item(), model.losses['recall']))

        if epoch % check_point == 0:
            model.save_weight(weight_file_name)

    model.save_weight(weight_file_name)
    file.close()

if __name__ == '__main__':
    model = darknet("D:/ShaoshuYang/HPE/cfg/yolov3-1.cfg", 1)
    model.load_weight("D:/ShaoshuYang/HPE/src/yolov3-1-1.weights")

    if torch.cuda.is_available():
        model.cuda()

    model.train()

    train(model, "D:/ShaoshuYang/COCO/", "coco_anno.txt", 30, 6, 0.001, 0.9, 0.0005, 1,
          "yolov3-1-1.weights")
