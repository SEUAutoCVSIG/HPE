# -*- coding: utf-8 -*-
'''
    Created on wed Sept 5 21:12 2018

    Author           : Shaoshu Yang
    Email            : 13558615057@163.com
    Last edit date   : Sept 5 23:37 2018

South East University Automation College, 211189 Nanjing China

The following codes referenced Ayoosh Kathuria's blog:
How to implement a YOLO (v3) object detector from strach in
PyTorch: Part 2
'''

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def parse_cfg(cfgfile):
    '''
        Args :
             cfgfile   : (string) directory to *.cfg

        Returns :
             A list of blocks, describing how the neural network
             would be built. Block is represented as a dictionary
    '''

    # Pre-processing of the .cfg file
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    # Parse and store the network structure
    for line in lines:
        # '[' is the mark of a new layer
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}

            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

def create_modules(blocks):
    # Stores the info about the network
    net_info = blocks[0]

    chan_in = 3
    chan_out = []
    module_list = nn.ModuleList()

    for i, layer in enumerate(blocks[1:]):
        # Create module for every layers
        module = nn.Sequential()

        # Check the type of block
        # Append convolutional layer
        if layer["type"] == "convolutional":
            activation = layer["activation"]
            try:
                batch_norm = int(layer["batch_normalize"])
                bias = False
            except:
                batch_norm = 0
                bias = True

            filters = int(layer["filters"])
            padding = int(layer["pad"])
            kernel_size = int(layer["size"])
            stride = int(layer["stride"])

            if padding:
                pad = (kernel_size - 1)//2
            else:
                pad = 0

            conv = nn.Conv2d(chan_in, filters, kernel_size, stride,
                                                    pad, bias = bias)
            module.add_module("conv{0}".format(i), conv)

            # Append Batch Normalization layer
            if batch_norm:
                BN = nn.BatchNorm2d(filters)
                module.add_module("batch_norm{0}".format(i), BN)

            # Append activation
            if activation == "leaky":
                activation_ = nn.LeakyReLU(0.1, inpace=True)
                module.add_module("leaky{0}".format(i), activation_)

        # Append upsample layer
        elif layer["type"] == "upsample":
            stride = int(layer["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="bilinear")
            module.add_module("upsample{}".format(i), upsample)

        # Append route layer
        elif layer["type"] == "route":
            layer["layer"] = layer["layer"].split(',')
            # Param one of route
            start = int(layer["layer"][0])
            # Detect param two
            try:
                end = int(layer["layer"][1])
            except:
                end = 0

            # Positive annotation
            if start > 0:
                start = start - i
            if end > 0:
                end = end - i
            # pass, unfinished yet

        # Append shortcut layer
        #elif layer["type"] == "shortcut":
        # Append yolo layer
        #elif layer["type"] == "yolo":