# -*- coding: utf-8 -*-
'''
    Created on Thu Sep 20 21:25 2018

    Author          ：Yu Du
    Email           : 1239988498@qq.com
    Last edit date  : Sat Sep 25 15:00 2018

South East University Automation College, 211189 Nanjing China
'''

import cv2
from src.train_mpii import *
from copy import deepcopy
from src.utils import draw


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
        cuda = torch.cuda.is_available()
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        for i, (idx, data, img, target) in enumerate(data_loader):
            data = Variable(data.type(Tensor))
            target = Variable(target.type(Tensor), requires_grad=False)
            # print(target.shape)
            gt_np = dataset.get_parts(int(idx))
            img = np.array(img[0])
            # Using ground truth
            img_gt = deepcopy(img)
            # Using heatmap of ground truth
            img_tg = deepcopy(img)
            output = self.model(data)
            op_np = np.zeros((16, 2), dtype=int)
            tg_np = np.zeros((16, 2), dtype=int)
            for part in range(len(self.parts)):
                part_output = output[0, part + len(self.parts), :, :]
                part_target = target[0, part + len(self.parts), :, :]
                if part_output.max() != 0:
                    op_np[part][0] = np.where(part_output == part_output.max())[0][0]
                    op_np[part][1] = np.where(part_output == part_output.max())[1][0]
                if part_target.max() != 0:
                    tg_np[part][0] = np.where(part_target == part_target.max())[0][0]
                    tg_np[part][1] = np.where(part_target == part_target.max())[1][0]
            op = [[0, 0]] * len(self.parts)
            gt = [[0, 0]] * len(self.parts)
            tg = [[0, 0]] * len(self.parts)
            for part in range(len(self.parts)):
                op[part] = op_np[part][0] * 4, op_np[part][1] * 4
                gt[part] = int(gt_np[part][0]), int(gt_np[part][1])
                tg[part] = int(tg_np[part][0] * 4), int(tg_np[part][1] * 4)

            draw('Estimator', img, op, 3)
            draw('Ground Truth', img_gt, gt, 3)
            draw('Target', img_tg, tg, 3)
            cv2.waitKey(0)

    def tg_check(self, dataset):
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        cuda = torch.cuda.is_available()
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        for i, (idx, _, img, target) in enumerate(data_loader):
            target = Variable(target.type(Tensor), requires_grad=False)
            img = np.array(img[0])
            # Using ground truth
            # Using heatmap of ground truth
            tg_np = np.zeros((16, 2), dtype=int)
            for part in range(len(self.parts)):
                part_target = target[0, part + len(self.parts), :, :]
                if part_target.max() != 0:
                    tg_np[part][0] = np.where(part_target == part_target.max())[0][0]
                    tg_np[part][1] = np.where(part_target == part_target.max())[1][0]
            # print('target = ', tg_np)
            tg = [[0, 0]] * len(self.parts)
            for part in range(len(self.parts)):
                tg[part] = int(tg_np[part][0] * 4), int(tg_np[part][1] * 4)
            draw('Target', img, tg, 3, 0)



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
    # estimator.test(dataset)
    estimator.tg_check(dataset)
