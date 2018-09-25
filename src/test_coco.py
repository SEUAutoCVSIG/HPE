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
from src.utils import  *
from detect import *

def test(model, root, list_dir, batch_size):
    '''
        Args:
             model        : (nn.Module) untrained darknet
             root         : (string) directory of root
             list_dir     : (string) directory to list file
             batch_size   : (int) batch size
        Returns:
             Output test info
    '''
    # Define data loader
    data_loader = torch.utils.data.DataLoader(COCO(root, list_dir), batch_size=
                                              batch_size, shuffle=True)

    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    losses = []
    recall = []

    # Test process
    for i, (canvas, target) in enumerate(data_loader):
        canvas = Variable(canvas.type(Tensor))
        target = Variable(target.type(Tensor), requires_grad=False)

        loss = model(canvas, target)
        prediction = model(canvas)
        prediction = non_max_suppression(prediction, 1)
        prediction = prediction[..., :4]

        target[:, 1:] *= 416
        target[:, 1] -= target[:, 3]/2
        target[:, 2] -= target[:, 4]/2
        target[:, 3] += target[:, 1]
        target[:, 4] += target[:, 2]


        # Output train info
        print('[Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, \
                                                cls %f, total %f, recall: %.5f]' %
                                    (i, len(data_loader),
           model.losses['x'], model.losses['y'], model.losses['w'],
           model.losses['h'], model.losses['conf'], model.losses['cls'],
           loss.item(), model.losses['recall']))
        losses.append(loss.item())
        recall.append(model.losses['recall'])

    print("loss: %f, recall %f"%(sum(losses)/len(losses), sum(recall)/len(recall)))

def precision()
if __name__ == '__main__':
    model = darknet("D:/ShaoshuYang/HPE/cfg/yolov3-1.cfg", 1)
    model.load_weight("D:/ShaoshuYang/HPE/src/yolov3-1.weights")

    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    test(model, "D:/ShaoshuYang/COCO/", "coco_anno.txt", 1)
