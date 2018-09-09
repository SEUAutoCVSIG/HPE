# -*- coding: utf-8 -*-
'''
    Created on wed Sept 6 19:34 2018

    Author           : Shaoshu Yang, Heng Tan
    Email            : 13558615057@163.com
                       1608857488@qq.com

    Last edit date   : Sept 9 14:55 2018

South East University Automation College, 211189 Nanjing China

The following codes referenced Ayoosh Kathuria's blog:
How to implement a YOLO (v3) object detector from strach in
PyTorch: Part 3/4/5
'''

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
from PIL import Image
from PIL import ImageDraw
import csv
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# initialize COCO api for person keypoints annotations
dataDir='F:'
dataType='train2017'
annFile = '{}/Python 3.6/PYTHON 项目/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
coco_kps=COCO(annFile)

# display COCO categories and supercategories
cats = coco_kps.loadCats(coco_kps.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# get all images containing given categories, select one at random
catIds = coco_kps.getCatIds(catNms=['person']);
imgIds = coco_kps.getImgIds(catIds=catIds );
print ('there are %d images containing human'%len(imgIds))

def getBndboxKeypointsGT():
    csvFile = open('F:/Python 3.6/PYTHON 项目/KeypointBndboxGT.txt','w')
    keypointsWriter = csv.writer(csvFile)
    firstRow = ['imageName             ','personNumber    ','bndbox']
    keypointsWriter.writerow(firstRow)
    for i in range(len(imgIds)):
        imageNameTemp = coco_kps.loadImgs(imgIds[i])[0]
        imageName = imageNameTemp['file_name'].encode('raw_unicode_escape')
        img = coco_kps.loadImgs(imgIds[i])[0]
        annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco_kps.loadAnns(annIds)
        personNumber = len(anns)
        for j in range(personNumber):
            bndbox = anns[j]['bbox']
            keyPoints = anns[j]['keypoints']
            keypointsRow = [imageName,'        ',str(personNumber),'        ',
                            str(bndbox[0])+'_'+str(bndbox[1])+'_'+str(bndbox[2])+'_'+str(bndbox[3])]

            keypointsWriter.writerow(keypointsRow)

    csvFile.close()

if __name__ == "__main__":
    print ('Writing bndbox and keypoints to csv files..."')
    getBndboxKeypointsGT()