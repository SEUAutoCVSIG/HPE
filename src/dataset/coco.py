# -*- coding: utf-8 -*-
'''
    Created on wed Sept 9 15:34 2018

    Author           : Shaoshu Yang
    Email            : 13558615057@163.com
    Last edit date   : Sept 9 22:32 2018

South East University Automation College, 211189 Nanjing China
'''

import glob
import random
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.transform import resize
import sys

class COCO(Dataset):
    def __init__(self, root, list_dir, img_size=416):
        '''
            Args:
                 root          : (string) root directory
                 list_dir      : (string) directory to list file
                 img_size      : (int) input image size
        '''
        with open(root+list_dir, 'r') as imgfile:
            lines = imgfile.read().split('\n')
        lines = lines[2::2]
        self.ann_list = []
        self.root = root
        self.img_size = img_size
        last_img = None
        for i, anno in enumerate(lines):
            anno = anno.split()
            name = anno[0].rstrip("',").lstrip("b'")

            if name == last_img:
                # The image has aready be recorded
                bbox = list(map(float, anno[2].lstrip(",").split('_')))
                self.ann_list[-1]["bbox"].append(bbox)

            else:
                # Create a new dictionary for a new image
                ann_list = {}

                person_num = int(anno[1].strip(","))
                bbox = list(map(float, anno[2].lstrip(",").split('_')))

                ann_list["name"] = name
                ann_list["person_num"] = person_num
                ann_list["bbox"] = [[]]
                ann_list["bbox"][0] = bbox
                self.ann_list.append(ann_list)
                last_img = name

    def __getitem__(self, idx):
        '''
            Args:
                 idx       : (int) required index for data and target
            Returns:
                 Required label and data
        '''
        # Read image
        img = cv2.imread(self.root + "train/" + self.ann_list[idx]["name"])
        img /= 255.0
        img_h, img_w = img.shape(0), img.shape(1)

        new_w = int(img_w*min(self.img_size/img_h, self.img_size/img_w))
        new_h = int(img_h*min(self.img_size/img_h, self.img_size/img_w))

        pad_h = (self.img_size - new_h)//2
        pad_w = (self.img_size - new_w)//2

        # Image pre-processing
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        canvas = np.full((self.img_size, self.img_size, 3), 128)
        canvas[(self.img_size - new_h) // 2:(self.img_size - new_h) // 2 + new_h,
        (self.img_size - new_w) // 2:(self.img_size - new_w) // 2 + new_w, :] = img

        # Transform BGR to RGB, HWC to CHW
        canvas = canvas[:, :, ::-1].transpose((2, 0, 1))
        canvas = torch.from_numpy(canvas).float().unsequeeze(0)

        # Label pre-processing
        label = torch.tensor(self.ann_list[idx]["bbox"]).float()
        label[:, 0] *= new_w/img_w
        label[:, 1] *= new_h/img_h
        label[:, 2] *= new_w/img_w
        label[:, 3] *= new_h/img_h

        return canvas, label

    def __len__(self):
        return len(self.ann_list)

    def show_dataset(self):
        for i, anno in enumerate(self.ann_list):
            img = cv2.imread(self.root + "train/" + anno["name"])

            '''
            label = np.array(anno["bbox"])
            for bbox in anno["bbox"]:
                draw_rect(bbox, img)

            cv2.imshow('labeled.jpg', img)
            cv2.waitKey(500)
            '''


            img_h, img_w = img.shape(0), img.shape(1)

            new_w = int(img_w * min(self.img_size / img_h, self.img_size / img_w))
            new_h = int(img_h * min(self.img_size / img_h, self.img_size / img_w))

            pad_h = (self.img_size - new_h) // 2
            pad_w = (self.img_size - new_w) // 2

            # Image pre-processing
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            canvas = np.full((self.img_size, self.img_size, 3), 128)
            canvas[(self.img_size - new_h) // 2:(self.img_size - new_h) // 2 + new_h,
            (self.img_size - new_w) // 2:(self.img_size - new_w) // 2 + new_w, :] = img

            # Label pre-processing
            label = np.array(anno["bbox"])
            label[:, 0] *= new_w / img_w
            label[:, 1] *= new_h / img_h
            label[:, 2] *= new_w / img_w
            label[:, 3] *= new_h / img_h
            label[:, 0] += pad_w
            label[:, 1] += pad_h

            for bbox in anno["bbox"]:
                draw_rect(bbox, img)

            cv2.imshow(img)


def draw_rect(bbox, img):
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(x1 + bbox[2])
    y2 = int(y1 + bbox[3])
    coord1 = (x1, y1)
    coord2 = (x2, y2)
    cv2.rectangle(img, coord1, coord2, (0, 255, 0), 2)
    return img

if __name__ == '__main__':
    coco = COCO("D:/ShaoshuYang/COCO/", "coco_anno.txt")
    coco.show_dataset()




