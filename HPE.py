# -*- coding: utf-8 -*-
'''
    Created on wed Sept 20 11:32 2018

    Author           : Shaoshu Yang
    Email            : 13558615057@163.com
    Last edit date   : Sept 20 11:32 2018

South East University Automation College, 211189 Nanjing China
'''

from src.model.darknet import darknet
from detect import detector
import torch
import cv2

class HPE():
    def __init__(self):
        # Deploy darknet53 model on cooresponding device
        model = darknet("cfg/yolov3-1.cfg", 1)
        model.load_weight("yolov3-1-1.weights")

        cuda = torch.cuda.is_available()
        if cuda:
            model.cuda()

        model.eval()

        self.detector = detector(model)

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
                prediction = self.detector.detect(frame)

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

    def draw_keypoints(self, prediction, img):
        '''
            Args:
                 prediction       : (list) list that record the coordinates of key points
                 img              : (ndarray) original image
            Returns:
                 Image with key points
        '''





if __name__ == '__main__':
    test = HPE()
    test.human_det()