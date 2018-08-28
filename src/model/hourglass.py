# -*- coding: utf-8 -*-
'''
    Created on Tues Aug 28 8:52 2018

    Author           : Shaoshu Yang
    Email            : 13558615057@163.com
    Last edit date   : Aug 28 20:00 2018

South East University Automation College, 211189 Nanjing China
'''

from residual import Residual
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

# Definition of hourglass module
class Hourglass(nn.Module):
    def __init__(self, chan_in, chan_out, n):
        '''
            args:
                 chan_in   : (int) number of input channels
                 chan_out  : (int) number of output channels
                 n         : (int) order of the hourglass model
        '''
        super(Hourglass, self).__init__()

        # Upper branch definition
        self.uppper = nn.Sequential(
            Residual(chan_in, 256),
            Residual(256, 256),
            Residual(256, chan_out),
            )

        # Lower branch definition
        if n > 1:
            self.lower = nn.Sequential(
                nn.MaxPool2d(2, stride=2),
                Residual(chan_in, 256),
                Residual(256, 256),
                Residual(256, 256),
                Hourglass(256, chan_out, n - 1),
                Residual(chan_out, chan_out),
                nn.UpsamplingNearest2d(scale_factor=2)
                )
        else:
            self.lower = nn.Sequential(
                nn.MaxPool2d(2, stride=2),
                Residual(chan_in, 256),
                Residual(256, 256),
                Residual(256, 256),
                Residual(256, chan_out),
                Residual(chan_out, chan_out),
                nn.UpsamplingNearest2d(scale_factor=2)
                )

    # Override of forward method
    def forward(slef, x):
        return self.upper(x) + self.lower(x)

class StackedHourglass(nn.Module):
    def __init__()
