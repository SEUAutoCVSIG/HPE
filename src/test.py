# import time
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import math
# from PIL import Image
#
#
# def CenterLabelHeatMap(img_width, img_height, c_x, c_y, sigma):
#     X1 = np.linspace(1, img_width, img_width)
#     Y1 = np.linspace(1, img_height, img_height)
#     [X, Y] = np.meshgrid(X1, Y1)
#     X = X - c_x
#     Y = Y - c_y
#     D2 = X * X + Y * Y
#     E2 = 2.0 * sigma * sigma
#     Exponent = D2 / E2
#     heatmap = np.exp(-Exponent)
#     return heatmap
#
#
# # Compute gaussian kernel
# def CenterGaussianHeatMap(img_height, img_width, c_x, c_y, variance):
#     gaussian_map = np.zeros((img_height, img_width))
#     for x_p in range(img_width):
#         for y_p in range(img_height):
#             dist_sq = (x_p - c_x) * (x_p - c_x) + \
#                       (y_p - c_y) * (y_p - c_y)
#             exponent = dist_sq / 2.0 / variance / variance
#             gaussian_map[y_p, x_p] = np.exp(-exponent)/math.sqrt(2*3.1415926)/variance
#     return gaussian_map
#
#
# image_file = '/Users/midora/Desktop/Python/HPElocal/res/images/008115925.jpg'
# img = cv2.imread(image_file)
# img = img[:, :, ::-1]
#
#
# height, width, _ = np.shape(img)
# cy, cx = height/4.0, width/4.0
#
# start = time.time()
# heatmap1 = CenterLabelHeatMap(width, height, cx, cy, 21)
# t1 = time.time() - start
#
# start = time.time()
# heatmap2 = CenterGaussianHeatMap(height, width, cx, cy, 21)
# t2 = time.time() - start
#
# print(t1, t2)
#
# plt.subplot(1,2,1)
# plt.imshow(heatmap1)
# plt.subplot(1,2,2)
# plt.imshow(heatmap2)
# plt.show()
#
# print('End.')
#

import cv2
from math import *
import numpy as np

# img = cv2.imread("/Users/midora/Desktop/Python/HPElocal/res/images/008115925.jpg")
#
# height,width=img.shape[:2]
#
# degree=30
# #旋转后的尺寸
# heightNew=int(width*fabs(sin(radians(degree)))+height*fabs(cos(radians(degree))))
# widthNew=int(height*fabs(sin(radians(degree)))+width*fabs(cos(radians(degree))))
#
# matRotation=cv2.getRotationMatrix2D((width/2,height/2),degree,1)
#
# # matRotation[0,2] +=(widthNew-width)/2  #重点在这步，目前不懂为什么加这步
# # matRotation[1,2] +=(heightNew-height)/2  #重点在这步
#
# imgRotation=cv2.warpAffine(img,matRotation,(widthNew,heightNew),borderValue=(255,255,255))
#
# cv2.imshow("img",img)
# cv2.imshow("imgRotation",imgRotation)
# cv2.waitKey(0)

# height = 10
# width = 59
# max = height if height > width else width
# newheight = newwidth = max
# print(newheight, newwidth)

class DataTest:
    def __init__(self, id, address):
        self.id = id
        self.address = address
        self.d = {self.id: 1,
                  self.address: "192.168.1.1"
                  }

    def __getitem__(self, key):
        a = key[0]
        b = key[1]
        c = key[2]
        print(a, b, c)



data = DataTest(1, "192.168.2.11")
data[(1, 2, 3)]