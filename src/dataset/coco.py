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

class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.img_shape = (img_size, img_size)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = np.array(Image.open(img_path))
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        return img_path, input_img

    def __len__(self):
        return len(self.files)

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
        self.max_objects = 30
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

        self.ann_list = self.ann_list[:64000]

    def __getitem__(self, idx):
        '''
            Args:
                 idx       : (int) required index for data and target
            Returns:
                 Required label and data
        '''
        # Read image
        img = np.array(cv2.imread(self.root + "train/" + self.ann_list[idx]["name"]),
                                                                        dtype=float)
        img /= 255.0
        img_h, img_w = img.shape[0], img.shape[1]

        new_w = img_w * min(self.img_size / img_h, self.img_size / img_w)
        new_h = img_h * min(self.img_size / img_h, self.img_size / img_w)
        new_w_int = int(new_w)
        new_h_int = int(new_h)

        pad_h = (self.img_size - new_h)//2
        pad_w = (self.img_size - new_w)//2

        # Image pre-processing
        # Image pre-processing
        img = cv2.resize(img, (new_w_int, new_h_int), interpolation=cv2.INTER_CUBIC)
        canvas = np.full((self.img_size, self.img_size, 3), 128)
        canvas[(self.img_size - new_h_int) // 2:(self.img_size - new_h_int) // 2 + new_h_int,
        (self.img_size - new_w_int) // 2:(self.img_size - new_w_int) // 2 + new_w_int, :] = img

        # Transform BGR to RGB, HWC to CHW
        canvas = canvas[:, :, ::-1].transpose((2, 0, 1))
        canvas = torch.FloatTensor(canvas.copy())


        # Label pre-processing
        label = self.ann_list[idx]["bbox"]
        for label_ in label:
            label_.append(0)
            label_[1:] = label_[0:4]

            # 0 represent the person class
            label_[0] = 0

        label = torch.FloatTensor(label)
        label[:, 1] *= new_w / float(img_w)
        label[:, 2] *= new_h / float(img_h)
        label[:, 3] *= new_w / float(img_w)
        label[:, 4] *= new_h / float(img_h)
        label[:, 1] += pad_w
        label[:, 2] += pad_h
        label[:, 1] += label[:, 3] / 2.0
        label[:, 2] += label[:, 4] / 2.0
        label[:, 1:] /= self.img_size

        for label_ in label:
            label_[1] = label_[1] if label_[1] < 1 else 0.999
            label_[2] = label_[2] if label_[2] < 1 else 0.999

        filled_label = np.zeros((self.max_objects, 5))
        filled_label[range(len(label))[:self.max_objects]] = label[:self.max_objects]
        filled_label = torch.FloatTensor(filled_label)

        return canvas, filled_label

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

            img_h, img_w = img.shape[0], img.shape[1]

            new_w = img_w * min(self.img_size / img_h, self.img_size / img_w)
            new_h = img_h * min(self.img_size / img_h, self.img_size / img_w)
            new_w_int = int(new_w)
            new_h_int = int(new_h)

            pad_h = (self.img_size - new_h) / 2.0
            pad_w = (self.img_size - new_w) / 2.0

            # Image pre-processing
            img = cv2.resize(img, (new_w_int, new_h_int), interpolation=cv2.INTER_CUBIC)
            canvas = np.full((self.img_size, self.img_size, 3), 128)
            canvas[(self.img_size - new_h_int) // 2:(self.img_size - new_h_int) // 2 + new_h_int,
            (self.img_size - new_w_int) // 2:(self.img_size - new_w_int) // 2 + new_w_int, :] = img

            # Label pre-processing
            label = np.array(anno["bbox"])
            label[:, 0] *= new_w / img_w
            label[:, 1] *= new_h / img_h
            label[:, 2] *= new_w / img_w
            label[:, 3] *= new_h / img_h
            label[:, 0] += pad_w
            label[:, 1] += pad_h

            for bbox in label:
                draw_rect(bbox, canvas)

            plt.imshow(canvas[:, :, ::-1])
            plt.pause(0.1)

            if i%10 == 1:
                plt.show()


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




