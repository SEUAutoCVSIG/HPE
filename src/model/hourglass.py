# -*- coding: utf-8 -*-
'''
    Created on Tues Aug 28 8:52 2018

    Author           : Shaoshu Yang
    Email            : 13558615057@163.com
    Last edit date   : Sept 2 23:46 2018

South East University Automation College, 211189 Nanjing China
'''

from residual import Residual
import torch.nn as nn

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
    def forward(self, x):
        return self.upper(x) + self.lower(x)

class StackedHourglass(nn.Module):
    def __init__(self, chan_out):
        '''
            Args:
                 chan_out  : (int) number of output channels
        '''
        super(nn.Module, StackedHourglass).__init__()
        
        # Initial processing of th image
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.ReLU1 = nn.ReLU(inpace=True)
        self.r1 = Residual(64, 128)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.r4 = Residual(128, 128)
        self.r5 = Residual(128, 128)
        self.r6 = Residual(128, 256)
        
        # First hourglass
        self.hg1 = Hourglass(256, 512, 4)
        
        # Linear layers to produce first set of predictions
        self.l1 = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inpace=True)
            )
        self.l2 = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inpace=True)
            )
        
        # First predicted heatmaps
        self.out1 = nn.Conv2d(256, chan_out, 1)
        self.out1_ = nn.Conv2d(chan_out, 256+128, 1)
        
        # Concatenate with previous linear features
        self.cat1 = nn.Conv2d(256+128, 256+128, 1)
        
        # Second hourglass
        self.hg2 = Hourglass(4, 256+128, 512)
        
        # Linear layers to produce predictions again
        self.l3 = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inpace=True)
            )
        self.l4 = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inpace=True)
            )
        
        # Output heatmaps
        self.out2 = nn.Conv2d(512, chan_out, 1)
    
    # Override the forward method
    def forward(self, x):
        out = []
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.ReLU1(x)
        x = self.r1(x)
        x = self.pool1(x)
        
        # Forward pass on level1
        lv1 = self.r4(x)
        lv1 = self.r5(lv1)
        lv1 = self.r6(lv1)
        lv1 = self.hg1(lv1)
        lv1 = self.l1(lv1)
        lv1 = self.l2(lv1)
        out1 = self.out1(lv1)
        
        # Append output level1
        out.append(out1)
        out1 = self.out1_(out1)
        
        # Joint of pool1 & l2
        lv2 = []
        lv2.append(x)
        lv2.append(lv1)
        
        # Forward pass on level2
        lv2 = self.cat1(lv2)
        lv2 = lv2 + out1
        lv2 = self.hg2(lv2)
        lv2 = self.l3(lv2)
        lv2 = self.l4(lv2)
        
        # Append output level2
        out.append(self.out2(lv2))
        
        return out
        