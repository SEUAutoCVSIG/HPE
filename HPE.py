# -*- coding: utf-8 -*-
'''
    Created on wed Sept 20 11:32 2018

    Author           : Shaoshu Yang
    Email            : 13558615057@163.com
    Last edit date   : Sept 25 23:26 2018

South East University Automation College, 211189 Nanjing China
'''

from src.model.darknet import darknet
from src.model.hourglass import StackedHourglass
from src.utils import *
from src.dataset.coco  import COCO
from src.dataset.mpiiLoader import MpiiDataSet_sig
from detect import detector
from estimate import Estimator

import torch
import cv2

class HPE():
    def __init__(self):
        # Deploy darknet53 model on cooresponding device
        yolov3 = darknet("cfg/yolov3-1.cfg", 1)
        yolov3.load_weight("yolov3-1.weights")

        # Deploy stacked hourglass model
        stackedhourglass = StackedHourglass(16)
        stackedhourglass.load_state_dict(torch.load("stacked_hourlgass.pkl"))

        cuda = torch.cuda.is_available()
        if cuda:
            yolov3.cuda()
            stackedhourglass.cuda()

        yolov3.eval()

        self.detector = detector(yolov3)
        self.estimator = estimator(stackedhourglass)

    # Capture the frount camera
    def video_cap(self):
        cap = cv2.VideoCapture(0)

        while 1:
            ret, frame = cap.read()
            cv2.imshow("capture", frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

    # Execute human detection on the image sequences from camera
    def human_det(self):
        # Capture the camera
        cap = cv2.VideoCapture(0)

        while 1:
            ret, frame = cap.read()
            try:
                # Making prediction
                prediction = self.detector.detect(frame)[..., :4]

                # Drawing bounding-box
                self.draw_bbox(prediction, frame)

                # Press 'q' to exit
                cv2.imshow("target", frame)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break

            except:
                cv2.imshow("target", frame)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break

        cap.release()

    def draw_bbox(self, prediction, img):
        '''
            Args:
                 prediction       : (list) list that record the prediction bounding-box
                 img              : (ndarray) original image
            Returns:
                 Image with bounding-box on it
        '''
        for prediction_ in prediction:
            coord1 = tuple(map(int, prediction_[:2]))
            coord2 = tuple(map(int, prediction_[2:4]))
            cv2.rectangle(img, coord1, coord2, (0, 255, 0), 2)

        return img

    def pose_estimate(self):
        cap = cv2.VideoCapture(0)

        while 1:
            ret, frame = cap.read()
            try:
                # Geting dimensions, normalization and transforming
                img_h, img_w = frame.shape[0], frame.shape[1]
                img = torch.FloatTensor(frame[:, :, ::-1].transpose(2, 0, 1).copy()).div(255.).unsqueeze(0)

                # Making prediction
                prediction = self.detector.detect(frame)[..., :4]

                # Prepare container for key point coordinates
                estimation = []

                # Get estimation
                for prediction_ in prediction:
                    prediction_ = list(map(int, prediction_))

                    # Coordinates shall not exceed the boundary of origin image
                    for xcoord in prediction_[::2]:
                        xcoord = xcoord if xcoord >= 0 else 0
                        xcoord = xcoord if xcoord <= img_w else img_w

                    for ycoord in prediction_[1::2]:
                        ycoord = ycoord if ycoord >=0 else 0
                        ycoord = ycoord if ycoord <= img_h else img_h

                    estimation.append(estimator.estimate(img, prediction_))

                # Draw key points
                for estimation_ in estimation:
                    draw(frame, estimation_, 2)

                # Press 'q' to exit
                cv2.imshow("target", frame)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break

            except:
                cv2.imshow("target", frame)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break

        cap.release()

    def show_coco(self):
        # Display coco dataset and its annotations
        coco = COCO('D:/ShaoshuYang/COCO/', 'anno_list.txt')
        coco.show_dataset()

    def show_mpii(self):
        # Display mpii dataset and its annotaions
        mpii = MpiiDataSet_sig('D:/ShaoshuYang/MPII/', 'D:/ShaoshuYang/HPE/res/mpii_human_pose_v1_u12_1.mat')
        self.estimator.tg_check(mpii)

if __name__ == '__main__':
    test = HPE()
    test.human_det()
