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
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

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
                                              batch_size, shuffle=True)

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=momentum,
                          weight_decay=decay)

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
            model.save_weight(weight_file_name)

if __name__ == '__main__':
    model = darknet("HPE/cfg/yolov3-1.cfg")

    if torch.cuda.is_available():
        model.cuda()

    model.train()

    train(model, "HPE/", "HPE/data/coco_anno.txt", 100, 64, 0.001, 0.9, 0.0005, 10,
          "yolov3-1.weights")