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
from copy import deepcopy



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
            # print(target.shape)
            gt_np = dataset.get_parts(i)
            img = np.array(img[0])
            # Using ground truth
            img_gt = deepcopy(img)
            # Using heatmap of ground truth
            img_tg = deepcopy(img)
            output = self.model(data)
            coor_np = np.zeros((16, 2), dtype=int)
            tg_np = np.zeros((16, 2), dtype=int)
            for part in range(len(self.parts)):
                part_heatmap = output[0, part+len(self.parts), :, :]
                part_target = target[0, part+len(self.parts), :, :]
                # print(part_heatmap.shape)
                # print(part_target.shape)

                coor_np[part][0], coor_np[part][1] = np.where(part_heatmap == part_heatmap.max())
                if part_target.max() >= 0.4:
                    tg_np[part][0], tg_np[part][1] = np.where(part_target == part_target.max())
            coor = [[0, 0]]*len(self.parts)
            gt = [[0, 0]]*len(self.parts)
            tg = [[0, 0]]*len(self.parts)
            for part in range(len(self.parts)):
                coor[part] = coor_np[part][0]*2, coor_np[part][1]*2
                gt[part] = int(gt_np[part][0]), int(gt_np[part][1])
                tg[part] = int(tg_np[part][0]*2), int(tg_np[part][1]*2)

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

            if not((gt[0][0] == 0 and gt[0][1] == 0) or (gt[1][0] == 0 and gt[1][1] == 0)):
                img_gt = cv2.line(img_gt, gt[0], gt[1], (0, 255, 0), 3)
            if not((gt[1][0] == 0 and gt[1][1] == 0) or (gt[2][0] == 0 and gt[2][1] == 0)):
                img_gt = cv2.line(img_gt, gt[1], gt[2], (0, 255, 0), 3)
            if not((gt[2][0] == 0 and gt[2][1] == 0) or (gt[6][0] == 0 and gt[6][1] == 0)):
                img_gt = cv2.line(img_gt, gt[2], gt[6], (0, 255, 0), 3)
            if not((gt[3][0] == 0 and gt[3][1] == 0) or (gt[6][0] == 0 and gt[6][1] == 0)):
                img_gt = cv2.line(img_gt, gt[3], gt[6], (0, 255, 0), 3)
            if not((gt[3][0] == 0 and gt[3][1] == 0) or (gt[4][0] == 0 and gt[4][1] == 0)):
                img_gt = cv2.line(img_gt, gt[3], gt[4], (0, 255, 0), 3)
            if not((gt[4][0] == 0 and gt[4][1] == 0) or (gt[5][0] == 0 and gt[5][1] == 0)):
                img_gt = cv2.line(img_gt, gt[4], gt[5], (0, 255, 0), 3)
            if not((gt[6][0] == 0 and gt[6][1] == 0) or (gt[7][0] == 0 and gt[7][1] == 0)):
                img_gt = cv2.line(img_gt, gt[6], gt[7], (0, 255, 0), 3)
            if not((gt[7][0] == 0 and gt[7][1] == 0) or (gt[8][0] == 0 and gt[8][1] == 0)):
                img_gt = cv2.line(img_gt, gt[7], gt[8], (0, 255, 0), 3)
            if not((gt[8][0] == 0 and gt[8][1] == 0) or (gt[9][0] == 0 and gt[9][1] == 0)):
                img_gt = cv2.line(img_gt, gt[8], gt[9], (0, 255, 0), 3)
            if not((gt[7][0] == 0 and gt[7][1] == 0) or (gt[12][0] == 0 and gt[12][1] == 0)):
                img_gt = cv2.line(img_gt, gt[7], gt[12], (0, 255, 0), 3)
            if not((gt[11][0] == 0 and gt[11][1] == 0) or (gt[12][0] == 0 and gt[12][1] == 0)):
                img_gt = cv2.line(img_gt, gt[11], gt[12], (0, 255, 0), 3)
            if not((gt[10][0] == 0 and gt[10][1] == 0) or (gt[11][0] == 0 and gt[11][1] == 0)):
                img_gt = cv2.line(img_gt, gt[10], gt[11], (0, 255, 0), 3)
            if not((gt[7][0] == 0 and gt[7][1] == 0) or (gt[13][0] == 0 and gt[13][1] == 0)):
                img_gt = cv2.line(img_gt, gt[7], gt[13], (0, 255, 0), 3)
            if not((gt[13][0] == 0 and gt[13][1] == 0) or (gt[14][0] == 0 and gt[14][1] == 0)):
                img_gt = cv2.line(img_gt, gt[13], gt[14], (0, 255, 0), 3)
            if not((gt[14][0] == 0 and gt[14][1] == 0) or (gt[15][0] == 0 and gt[15][1] == 0)):
                img_gt = cv2.line(img_gt, gt[14], gt[15], (0, 255, 0), 3)



            if not((tg[0][0] == 0 and tg[0][1] == 0) or (tg[1][0] == 0 and tg[1][1] == 0)):
                img_tg = cv2.line(img_tg, tg[0], tg[1], (0, 255, 0), 3)
            if not((tg[1][0] == 0 and tg[1][1] == 0) or (tg[2][0] == 0 and tg[2][1] == 0)):
                img_tg = cv2.line(img_tg, tg[1], tg[2], (0, 255, 0), 3)
            if not((tg[2][0] == 0 and tg[2][1] == 0) or (tg[6][0] == 0 and tg[6][1] == 0)):
                img_tg = cv2.line(img_tg, tg[2], tg[6], (0, 255, 0), 3)
            if not((tg[3][0] == 0 and tg[3][1] == 0) or (tg[6][0] == 0 and tg[6][1] == 0)):
                img_tg = cv2.line(img_tg, tg[3], tg[6], (0, 255, 0), 3)
            if not((tg[3][0] == 0 and tg[3][1] == 0) or (tg[4][0] == 0 and tg[4][1] == 0)):
                img_tg = cv2.line(img_tg, tg[3], tg[4], (0, 255, 0), 3)
            if not((tg[4][0] == 0 and tg[4][1] == 0) or (tg[5][0] == 0 and tg[5][1] == 0)):
                img_tg = cv2.line(img_tg, tg[4], tg[5], (0, 255, 0), 3)
            if not((tg[6][0] == 0 and tg[6][1] == 0) or (tg[7][0] == 0 and tg[7][1] == 0)):
                img_tg = cv2.line(img_tg, tg[6], tg[7], (0, 255, 0), 3)
            if not((tg[7][0] == 0 and tg[7][1] == 0) or (tg[8][0] == 0 and tg[8][1] == 0)):
                img_tg = cv2.line(img_tg, tg[7], tg[8], (0, 255, 0), 3)
            if not((tg[8][0] == 0 and tg[8][1] == 0) or (tg[9][0] == 0 and tg[9][1] == 0)):
                img_tg = cv2.line(img_tg, tg[8], tg[9], (0, 255, 0), 3)
            if not((tg[7][0] == 0 and tg[7][1] == 0) or (tg[12][0] == 0 and tg[12][1] == 0)):
                img_tg = cv2.line(img_tg, tg[7], tg[12], (0, 255, 0), 3)
            if not((tg[11][0] == 0 and tg[11][1] == 0) or (tg[12][0] == 0 and tg[12][1] == 0)):
                img_tg = cv2.line(img_tg, tg[11], tg[12], (0, 255, 0), 3)
            if not((tg[10][0] == 0 and tg[10][1] == 0) or (tg[11][0] == 0 and tg[11][1] == 0)):
                img_tg = cv2.line(img_tg, tg[10], tg[11], (0, 255, 0), 3)
            if not((tg[7][0] == 0 and tg[7][1] == 0) or (tg[13][0] == 0 and tg[13][1] == 0)):
                img_tg = cv2.line(img_tg, tg[7], tg[13], (0, 255, 0), 3)
            if not((tg[13][0] == 0 and tg[13][1] == 0) or (tg[14][0] == 0 and tg[14][1] == 0)):
                img_tg = cv2.line(img_tg, tg[13], tg[14], (0, 255, 0), 3)
            if not((tg[14][0] == 0 and tg[14][1] == 0) or (tg[15][0] == 0 and tg[15][1] == 0)):
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



