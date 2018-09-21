# -*- coding: utf-8 -*-
'''
    Created on Thu Sep 20 21:25 2018

    Author          ：Yu Du
    Email           : 1239988498@qq.com
    Last edit date  :

South East University Automation College, 211189 Nanjing China
'''

from src.model.hourglass import StackedHourglass
from src.dataset.dataloader import MpiiDataset
from src.dataset.mpiiLoader import MpiiDataSet_sig
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models
from torchvision import transforms as T
from torch.utils.data import DataLoader
from PIL import Image
import time
import os
import cv2
from src.train_mpii import *



class Estimator:
    def __init__(self, model, camera=False):
        self.model = model
        self.camera = camera
        self.parts = ['rank', 'rkne', 'rhip',
                 'lhip', 'lkne', 'lank',
                 'pelv', 'thrx', 'neck', 'head',
                 'rwri', 'relb', 'rsho',
                 'lsho', 'lelb', 'lwri']

    def test(self, dataset):
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        for i, (data, img, target) in enumerate(data_loader):
            gt_np = dataset.get_parts(i)
            img = np.array(img[0])
            # Using ground truth
            img_gt = img.copy
            # Using heatmap of ground truth
            img_tg = img.copy
            output = self.model(data)
            coor_np = np.zeros((16, 2), dtype=int)
            tg_np = np.zeros((16, 2), dtype=int)
            for part in range(len(self.parts)):
                part_heatmap = output[0, part+len(self.parts), :, :]
                part_target = target[0, part, :, :]
                coor_np[part][0], coor_np[part][1] = np.where(part_heatmap == part_heatmap.max())
                tg_np[part][0], tg_np[part][1] = np.where(part_target == part_target.max())
            coor = [[0, 0]]*len(self.parts)
            gt = [[0, 0]]*len(self.parts)
            tg = [[0, 0]]*len(self.parts)
            for part in range(len(self.parts)):
                coor[part] = coor_np[part][0], coor_np[part][1]
                gt[part] = gt_np[part][0], gt_np[part][1]
                tg[part] = tg_np[part][0], tg_np[part][1]

            img = cv2.line(img, coor[0], coor[1], (0, 255, 0), 3)
            img = cv2.line(img, coor[1], coor[2], (0, 255, 0), 3)
            img = cv2.line(img, coor[2], coor[6], (0, 255, 0), 3)
            img = cv2.line(img, coor[3], coor[6], (0, 255, 0), 3)
            img = cv2.line(img, coor[3], coor[4], (0, 255, 0), 3)
            img = cv2.line(img, coor[4], coor[5], (0, 255, 0), 3)
            img = cv2.line(img, coor[6], coor[7], (0, 255, 0), 3)
            img = cv2.line(img, coor[7], coor[8], (0, 255, 0), 3)
            img = cv2.line(img, coor[8], coor[9], (0, 255, 0), 3)
            img = cv2.line(img, coor[7], coor[12], (0, 255, 0), 3)
            img = cv2.line(img, coor[11], coor[12], (0, 255, 0), 3)
            img = cv2.line(img, coor[10], coor[11], (0, 255, 0), 3)
            img = cv2.line(img, coor[7], coor[13], (0, 255, 0), 3)
            img = cv2.line(img, coor[13], coor[14], (0, 255, 0), 3)
            img = cv2.line(img, coor[14], coor[15], (0, 255, 0), 3)

            img_gt = cv2.line(img_gt, gt[0], gt[1], (0, 255, 0), 3)
            img_gt = cv2.line(img_gt, gt[1], gt[2], (0, 255, 0), 3)
            img_gt = cv2.line(img_gt, gt[2], gt[6], (0, 255, 0), 3)
            img_gt = cv2.line(img_gt, gt[3], gt[6], (0, 255, 0), 3)
            img_gt = cv2.line(img_gt, gt[3], gt[4], (0, 255, 0), 3)
            img_gt = cv2.line(img_gt, gt[4], gt[5], (0, 255, 0), 3)
            img_gt = cv2.line(img_gt, gt[6], gt[7], (0, 255, 0), 3)
            img_gt = cv2.line(img_gt, gt[7], gt[8], (0, 255, 0), 3)
            img_gt = cv2.line(img_gt, gt[8], gt[9], (0, 255, 0), 3)
            img_gt = cv2.line(img_gt, gt[7], gt[12], (0, 255, 0), 3)
            img_gt = cv2.line(img_gt, gt[11], gt[12], (0, 255, 0), 3)
            img_gt = cv2.line(img_gt, gt[10], gt[11], (0, 255, 0), 3)
            img_gt = cv2.line(img_gt, gt[7], gt[13], (0, 255, 0), 3)
            img_gt = cv2.line(img_gt, gt[13], gt[14], (0, 255, 0), 3)
            img_gt = cv2.line(img_gt, gt[14], gt[15], (0, 255, 0), 3)

            img_tg = cv2.line(img_tg, tg[0], tg[1], (0, 255, 0), 3)
            img_tg = cv2.line(img_tg, tg[1], tg[2], (0, 255, 0), 3)
            img_tg = cv2.line(img_tg, tg[2], tg[6], (0, 255, 0), 3)
            img_tg = cv2.line(img_tg, tg[3], tg[6], (0, 255, 0), 3)
            img_tg = cv2.line(img_tg, tg[3], tg[4], (0, 255, 0), 3)
            img_tg = cv2.line(img_tg, tg[4], tg[5], (0, 255, 0), 3)
            img_tg = cv2.line(img_tg, tg[6], tg[7], (0, 255, 0), 3)
            img_tg = cv2.line(img_tg, tg[7], tg[8], (0, 255, 0), 3)
            img_tg = cv2.line(img_tg, tg[8], tg[9], (0, 255, 0), 3)
            img_tg = cv2.line(img_tg, tg[7], tg[12], (0, 255, 0), 3)
            img_tg = cv2.line(img_tg, tg[11], tg[12], (0, 255, 0), 3)
            img_tg = cv2.line(img_tg, tg[10], tg[11], (0, 255, 0), 3)
            img_tg = cv2.line(img_tg, tg[7], tg[13], (0, 255, 0), 3)
            img_tg = cv2.line(img_tg, tg[13], tg[14], (0, 255, 0), 3)
            img_tg = cv2.line(img_tg, tg[14], tg[15], (0, 255, 0), 3)

            cv2.imshow("Estimator", img)
            cv2.imshow("Ground Truth", img_gt)
            cv2.imshow("Target", img_tg)
            cv2.waitKey(0)



if __name__ == "__main__":
    weight_file_name = WeightPath + "stacked_hourglass.pkl"
    # Dataset
    dataset = MpiiDataSet_sig(FolderPath, Annotation, if_train=False)
    # Model
    model = StackedHourglass(16)
    if os.path.isfile(weight_file_name):
        model.load_state_dict(torch.load(weight_file_name))
    if torch.cuda.is_available():
        model.cuda()

    # Estimator
    estimator = Estimator(model)
    estimator.test(dataset)



