# -*- coding: utf-8 -*-
'''
    Created on Thu Sep 20 21:25 2018

    Author          ：Yu Du
    Email           : 1239988498@qq.com
    Last edit date  : Sat Sep 25 11:17 2018

South East University Automation College, 211189 Nanjing China
'''

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
                # print(part_heatmap.shape)
                # print(part_target.shape)
                if part_output.max() != 0:
                    op_np[part][0], op_np[part][1] = np.where(part_output == part_output.max())
                if part_target.max() != 0:
                    tg_np[part][0], tg_np[part][1] = np.where(part_target == part_target.max())
            # print('target = ', tg_np)
            # print('output = ', op_np)
            op = [[0, 0]] * len(self.parts)
            gt = [[0, 0]] * len(self.parts)
            tg = [[0, 0]] * len(self.parts)
            for part in range(len(self.parts)):
                op[part] = op_np[part][0] * 2, op_np[part][1] * 2
                gt[part] = int(gt_np[part][0]), int(gt_np[part][1])
                tg[part] = int(tg_np[part][0] * 2), int(tg_np[part][1] * 2)

            if not ((op[0][0] == 0 and op[0][1] == 0) or (op[1][0] == 0 and op[1][1] == 0)):
                img = cv2.line(img, op[0], op[1], (0, 255, 0), 3)
            if not ((op[1][0] == 0 and op[1][1] == 0) or (op[2][0] == 0 and op[2][1] == 0)):
                img = cv2.line(img, op[1], op[2], (0, 255, 0), 3)
            if not ((op[2][0] == 0 and op[2][1] == 0) or (op[6][0] == 0 and op[6][1] == 0)):
                img = cv2.line(img, op[2], op[6], (0, 255, 0), 3)
            if not ((op[3][0] == 0 and op[3][1] == 0) or (op[6][0] == 0 and op[6][1] == 0)):
                img = cv2.line(img, op[3], op[6], (0, 255, 0), 3)
            if not ((op[3][0] == 0 and op[3][1] == 0) or (op[4][0] == 0 and op[4][1] == 0)):
                img = cv2.line(img, op[3], op[4], (0, 255, 0), 3)
            if not ((op[4][0] == 0 and op[4][1] == 0) or (op[5][0] == 0 and op[5][1] == 0)):
                img = cv2.line(img, op[4], op[5], (0, 255, 0), 3)
            if not ((op[6][0] == 0 and op[6][1] == 0) or (op[7][0] == 0 and op[7][1] == 0)):
                img = cv2.line(img, op[6], op[7], (0, 255, 0), 3)
            if not ((op[7][0] == 0 and op[7][1] == 0) or (op[8][0] == 0 and op[8][1] == 0)):
                img = cv2.line(img, op[7], op[8], (0, 255, 0), 3)
            if not ((op[8][0] == 0 and op[8][1] == 0) or (op[9][0] == 0 and op[9][1] == 0)):
                img = cv2.line(img, op[8], op[9], (0, 255, 0), 3)
            if not ((op[7][0] == 0 and op[7][1] == 0) or (op[12][0] == 0 and op[12][1] == 0)):
                img = cv2.line(img, op[7], op[12], (0, 255, 0), 3)
            if not ((op[11][0] == 0 and op[11][1] == 0) or (op[12][0] == 0 and op[12][1] == 0)):
                img = cv2.line(img, op[11], op[12], (0, 255, 0), 3)
            if not ((op[10][0] == 0 and op[10][1] == 0) or (op[11][0] == 0 and op[11][1] == 0)):
                img = cv2.line(img, op[10], op[11], (0, 255, 0), 3)
            if not ((op[7][0] == 0 and op[7][1] == 0) or (op[13][0] == 0 and op[13][1] == 0)):
                img = cv2.line(img, op[7], op[13], (0, 255, 0), 3)
            if not ((op[13][0] == 0 and op[13][1] == 0) or (op[14][0] == 0 and op[14][1] == 0)):
                img = cv2.line(img, op[13], op[14], (0, 255, 0), 3)
            if not ((op[14][0] == 0 and op[14][1] == 0) or (op[15][0] == 0 and op[15][1] == 0)):
                img = cv2.line(img, op[14], op[15], (0, 255, 0), 3)

            if not ((gt[0][0] == 0 and gt[0][1] == 0) or (gt[1][0] == 0 and gt[1][1] == 0)):
                img_gt = cv2.line(img_gt, gt[0], gt[1], (0, 255, 0), 3)
            if not ((gt[1][0] == 0 and gt[1][1] == 0) or (gt[2][0] == 0 and gt[2][1] == 0)):
                img_gt = cv2.line(img_gt, gt[1], gt[2], (0, 255, 0), 3)
            if not ((gt[2][0] == 0 and gt[2][1] == 0) or (gt[6][0] == 0 and gt[6][1] == 0)):
                img_gt = cv2.line(img_gt, gt[2], gt[6], (0, 255, 0), 3)
            if not ((gt[3][0] == 0 and gt[3][1] == 0) or (gt[6][0] == 0 and gt[6][1] == 0)):
                img_gt = cv2.line(img_gt, gt[3], gt[6], (0, 255, 0), 3)
            if not ((gt[3][0] == 0 and gt[3][1] == 0) or (gt[4][0] == 0 and gt[4][1] == 0)):
                img_gt = cv2.line(img_gt, gt[3], gt[4], (0, 255, 0), 3)
            if not ((gt[4][0] == 0 and gt[4][1] == 0) or (gt[5][0] == 0 and gt[5][1] == 0)):
                img_gt = cv2.line(img_gt, gt[4], gt[5], (0, 255, 0), 3)
            if not ((gt[6][0] == 0 and gt[6][1] == 0) or (gt[7][0] == 0 and gt[7][1] == 0)):
                img_gt = cv2.line(img_gt, gt[6], gt[7], (0, 255, 0), 3)
            if not ((gt[7][0] == 0 and gt[7][1] == 0) or (gt[8][0] == 0 and gt[8][1] == 0)):
                img_gt = cv2.line(img_gt, gt[7], gt[8], (0, 255, 0), 3)
            if not ((gt[8][0] == 0 and gt[8][1] == 0) or (gt[9][0] == 0 and gt[9][1] == 0)):
                img_gt = cv2.line(img_gt, gt[8], gt[9], (0, 255, 0), 3)
            if not ((gt[7][0] == 0 and gt[7][1] == 0) or (gt[12][0] == 0 and gt[12][1] == 0)):
                img_gt = cv2.line(img_gt, gt[7], gt[12], (0, 255, 0), 3)
            if not ((gt[11][0] == 0 and gt[11][1] == 0) or (gt[12][0] == 0 and gt[12][1] == 0)):
                img_gt = cv2.line(img_gt, gt[11], gt[12], (0, 255, 0), 3)
            if not ((gt[10][0] == 0 and gt[10][1] == 0) or (gt[11][0] == 0 and gt[11][1] == 0)):
                img_gt = cv2.line(img_gt, gt[10], gt[11], (0, 255, 0), 3)
            if not ((gt[7][0] == 0 and gt[7][1] == 0) or (gt[13][0] == 0 and gt[13][1] == 0)):
                img_gt = cv2.line(img_gt, gt[7], gt[13], (0, 255, 0), 3)
            if not ((gt[13][0] == 0 and gt[13][1] == 0) or (gt[14][0] == 0 and gt[14][1] == 0)):
                img_gt = cv2.line(img_gt, gt[13], gt[14], (0, 255, 0), 3)
            if not ((gt[14][0] == 0 and gt[14][1] == 0) or (gt[15][0] == 0 and gt[15][1] == 0)):
                img_gt = cv2.line(img_gt, gt[14], gt[15], (0, 255, 0), 3)

            if not ((tg[0][0] == 0 and tg[0][1] == 0) or (tg[1][0] == 0 and tg[1][1] == 0)):
                img_tg = cv2.line(img_tg, tg[0], tg[1], (0, 255, 0), 3)
            if not ((tg[1][0] == 0 and tg[1][1] == 0) or (tg[2][0] == 0 and tg[2][1] == 0)):
                img_tg = cv2.line(img_tg, tg[1], tg[2], (0, 255, 0), 3)
            if not ((tg[2][0] == 0 and tg[2][1] == 0) or (tg[6][0] == 0 and tg[6][1] == 0)):
                img_tg = cv2.line(img_tg, tg[2], tg[6], (0, 255, 0), 3)
            if not ((tg[3][0] == 0 and tg[3][1] == 0) or (tg[6][0] == 0 and tg[6][1] == 0)):
                img_tg = cv2.line(img_tg, tg[3], tg[6], (0, 255, 0), 3)
            if not ((tg[3][0] == 0 and tg[3][1] == 0) or (tg[4][0] == 0 and tg[4][1] == 0)):
                img_tg = cv2.line(img_tg, tg[3], tg[4], (0, 255, 0), 3)
            if not ((tg[4][0] == 0 and tg[4][1] == 0) or (tg[5][0] == 0 and tg[5][1] == 0)):
                img_tg = cv2.line(img_tg, tg[4], tg[5], (0, 255, 0), 3)
            if not ((tg[6][0] == 0 and tg[6][1] == 0) or (tg[7][0] == 0 and tg[7][1] == 0)):
                img_tg = cv2.line(img_tg, tg[6], tg[7], (0, 255, 0), 3)
            if not ((tg[7][0] == 0 and tg[7][1] == 0) or (tg[8][0] == 0 and tg[8][1] == 0)):
                img_tg = cv2.line(img_tg, tg[7], tg[8], (0, 255, 0), 3)
            if not ((tg[8][0] == 0 and tg[8][1] == 0) or (tg[9][0] == 0 and tg[9][1] == 0)):
                img_tg = cv2.line(img_tg, tg[8], tg[9], (0, 255, 0), 3)
            if not ((tg[7][0] == 0 and tg[7][1] == 0) or (tg[12][0] == 0 and tg[12][1] == 0)):
                img_tg = cv2.line(img_tg, tg[7], tg[12], (0, 255, 0), 3)
            if not ((tg[11][0] == 0 and tg[11][1] == 0) or (tg[12][0] == 0 and tg[12][1] == 0)):
                img_tg = cv2.line(img_tg, tg[11], tg[12], (0, 255, 0), 3)
            if not ((tg[10][0] == 0 and tg[10][1] == 0) or (tg[11][0] == 0 and tg[11][1] == 0)):
                img_tg = cv2.line(img_tg, tg[10], tg[11], (0, 255, 0), 3)
            if not ((tg[7][0] == 0 and tg[7][1] == 0) or (tg[13][0] == 0 and tg[13][1] == 0)):
                img_tg = cv2.line(img_tg, tg[7], tg[13], (0, 255, 0), 3)
            if not ((tg[13][0] == 0 and tg[13][1] == 0) or (tg[14][0] == 0 and tg[14][1] == 0)):
                img_tg = cv2.line(img_tg, tg[13], tg[14], (0, 255, 0), 3)
            if not ((tg[14][0] == 0 and tg[14][1] == 0) or (tg[15][0] == 0 and tg[15][1] == 0)):
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
