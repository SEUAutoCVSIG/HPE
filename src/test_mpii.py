# -*- coding: utf-8 -*-
'''
    Created on Wed Sep 26 20:46 2018

    Author          ：Yu Du
    Email           : 1239988498@qq.com
    Last edit date  : Tue Oct 2 14:59 2018

South East University Automation College, 211189 Nanjing China
'''
from src.train_mpii import *
from src.utils import get_points_multi
from poseevaluation.pcp_pck import *

def eval_SH(weight_file_name):
    '''
        Args:
             model        : (nn.Module) untrained darknet
             root         : (string) directory of root
             list_dir     : (string) directory to list file
             epochs      : (int) max epoches
             batch_size   : (int) batch size
             learn_rate   : (float) learn rate
             momentum     : (float) momentum
             decay        : (float) weight decay
             check_point  : (int) interval between weights saving
             weight_file_name
                          : (string) name of the weight file
        Returns:
             Output training status and save weight
    '''

    # Define data loader
    dataset = MpiiDataSet_sig(FolderPath, Annotation, if_train=False, is_eval=True)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    stack_hourglass = StackedHourglass(16)
    if os.path.isfile(weight_file_name):
       stack_hourglass.load_state_dict(torch.load(weight_file_name))
    if torch.cuda.is_available():
        stack_hourglass.cuda()

    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    for i in range(1, 21):
        mAPs = np.zeros((16, 2))
        for data, target in data_loader:
            data = Variable(data.type(Tensor))
            target = Variable(target.type(Tensor), requires_grad=False)
            output = stack_hourglass(data)
            op = get_points_multi(output)
            tg = get_points_multi(target)
            mAPs += eval_pckh('mpii', tg, op)
            print('thresh:   %f  mAP     %f' % (0.05 * i, mAPs))
        mAPs /= len(data_loader)
        with open('SH_mAP.txt', 'a+') as fp:
            fp.write('thresh:   %f  mAP     %f\n' %(0.05*i, mAPs))
        print('thresh:   %f  mAP     %f' %(0.05*i, mAPs))


if __name__ == '__main__':
    weight_file_name = WeightPath+"stacked_hourglass.pkl"
    eval_SH(weight_file_name)