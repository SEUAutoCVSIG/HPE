# -*- coding: utf-8 -*-
'''
    Created on Mon Aug 27 10:19 2018

    Author          ：Shaoshu Yang
    Email           : 13558615057@163.com
    Last edit date  : Aug 28 0:30 2018

South East University Automation College, 211189 Nanjing China
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# The definition of residual module class in stacked hourglass
class Residual(nn.Module):
    def __init__(self, chan_in=64, chan_out=64):
        '''
           args:
                chan_in   : (int) number of input channels
                chan_out  : (int) number of output channels
        '''
        super(Residual, self).__init__()
        
        # Main convolutional block definition
        self.mainconv = nn.Sequential(
            nn.Conv2d(chan_in, chan_out//2, 1, stride=1),
            nn.BatchNorm2d(chan_out//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(chan_out//2, chan_out//2, 3, stride=1,
                                            padding=(1, 1)),
            nn.BatchNorm2d(chan_out//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(chan_out//2, chan_out, 1, stride=1),
            nn.BatchNorm2d(chan_out)
        )

        # Skip layer definition
        self.skip = nn.Sequential(
            nn.Conv2d(chan_in, chan_out, 1, stride=1, bias=False),
            nn.BatchNorm2d(chan_out)
        )

    # Override the forward method s
    def forward(self, x):
        residual = self.mainconv(x)
        out = self.skip(x) + residual
        return out
