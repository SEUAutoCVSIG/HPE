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

# EmptyLayer is registered as the route layer in darknet
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

# DetectLayer is registered for the yolo layer in darknet
class DetectLayer(nn.Module):
    def __init__(self, anchors):
        '''
            Args:
                 anchors   : (list) the list describing anchors of
                             yolo v3
        '''
        super(DetectLayer, self).__init__()
        self.anchors = anchors

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
            route = EmptyLayer()
            module.add_module("route{0}".format(i), route)

            if end < 0:
                filters = chan_out[i+start] + chan_out[i+end]
            else:
                filters = chan_out[i+start]
        # Append shortcut layer
        elif layer["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut{0}".format(i), shortcut)

        # Append yolo layer
        elif layer["type"] == "yolo":
            mask = layer["mask"].split(",")
            mask = [int(a) for a in mask]

            anchors = layer["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0,
                                                   len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectLayer(anchors)
            module.add_module("detection{0}".format(i), detection)

        module_list.append(module)
        chan_in = filters
        chan_out.append(filters)

    return net_info, module_list

# darknet structure definition
class darknet(nn.Module):
    def __init__(self, cfgfile):
        '''
            Args:
                 cfgfile : (string) directory of *.cfg file
        '''
        super(darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        '''
            Args:
                 x       : (tensor) input tensor
                 CUDA    : (bool) determines whether this device has
                          access to CUDA and GPU computing
            return:
                 The prediction of the network
        '''
        # Cache the output of route layers
        route_output = {}

        for i, module in enumerate(self.blocks[1:]):
            module_type = module["type"]

            # Convolution or upsample computing
            if module_type == "convolutional" or module_type == \
                                                        "upsample":
                x = self.module_list[i](x)

            # Concatenate from previous outputs
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if layers[0] > 0:
                    layers[0] -= i

                if len(layers) == 1:
                    x = route_output[i + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] -= i

                    x = torch.cat((route_output[i + layers[0]],
                                   route_output[i + layers[1]]), 1)

            # Add outputs of the previous layer
            # to the previous layer(-from_)
            elif module_type == "shortcut":
                from_ = int(module["form"])
                x = route_output[i - 1] + route_output[i + from_]

            elif module_type == "yolo":
                anchors = self.module_list[i](0).anchors

                # Get input dimensions
                in_dim = int(self.net_info["height"])
                # Get classes number
                class_num = int(module["classes"])



