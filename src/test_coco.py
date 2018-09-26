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
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from src.utils import  *
from detect import *

def test(model, root, list_dir, batch_size):
    '''
        Args:
             model        : (nn.Module) untrained darknet
             root         : (string) directory of root
             list_dir     : (string) directory to list file
             batch_size   : (int) batch size
        Returns:
             Output test info
    '''
    # Define data loader
    data_loader = torch.utils.data.DataLoader(COCO(root, list_dir), batch_size=
                                              batch_size, shuffle=False)

    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    losses = []
    recall = []
    precision = []

    # Test process
    for i, (canvas, target) in enumerate(data_loader):
        canvas = Variable(canvas.type(Tensor))
        target = Variable(target.type(Tensor), requires_grad=False)

        prediction = model(canvas)
        loss = model(canvas, target)
        prediction = non_max_suppression(prediction, 80, conf_thres=0.8, nms_thres=0.4)

        try:
            pred_box = []
            for prediction_ in prediction[0]:
                if prediction_[6] == 0:
                    pred_box.append(np.array(prediction_[:4]))

            prediction = np.array(pred_box)
        except:
            precision.append(0.)
            continue

        # Transform target to top-left and bottom-right coordinates
        target[..., 1:] *= 416
        target[..., 1] -= target[..., 3]/2
        target[..., 2] -= target[..., 4]/2
        target[..., 3] += target[..., 1]
        target[..., 4] += target[..., 2]

        nCorrect = 0.
        nTotal = float(prediction.shape[0])

        for target_ in target[0]:
            if sum(target_) == 0:
                break

            # Get b-box IOU
            gt_box = torch.FloatTensor(np.array([target_[1], target_[2], target_[3], target_[4]])).unsqueeze(0)
            try:
                IOU = bbox_IOU(gt_box, torch.FloatTensor(prediction), x1y1x2y2=True)
            except:
                break

            max, best_n = torch.max(IOU, 0)
            if max > 0.5:
                nCorrect += 1.
                prediction = np.delete(prediction, best_n, 0)

        precision.append(nCorrect/(nTotal + 1e-16))
        # Output train info
        print('[Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, \
                                                cls %f, total %f, recall: %.5f, precision: %.5f]' %
                                    (i, len(data_loader),
           model.losses['x'], model.losses['y'], model.losses['w'],
           model.losses['h'], model.losses['conf'], model.losses['cls'],
           loss.item(), model.losses['recall'], precision[-1]))
        losses.append(loss.item())
        recall.append(model.losses['recall'])

    print("loss: %f, recall: %f, mAP: %f"%(sum(losses)/len(losses), sum(recall)/len(recall),
                                                            sum(precision)/len(precision)))

if __name__ == '__main__':
    model = darknet("D:/ShaoshuYang/HPE/cfg/yolov3.cfg", 1)
    model.load_weight("D:/ShaoshuYang/HPE/yolov3.weights")

    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    test(model, "D:/ShaoshuYang/COCO/", "coco_anno.txt", 1)
