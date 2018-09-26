# -*- coding: utf-8 -*-
'''
    Created on Wed Sep 26 20:46 2018

    Author          ：Yu Du
    Email           : 1239988498@qq.com
    Last edit date  :

South East University Automation College, 211189 Nanjing China
'''
from src.train_mpii import *
from src.utils import get_points
from poseevaluation.pcp_pck import *

def eval_SH():
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
    dataset = MpiiDataSet_sig(FolderPath, Annotation)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    stack_hourglass = StackedHourglass(16)
    if os.path.isfile(weight_file_name):
       stack_hourglass.load_state_dict(torch.load(weight_file_name))
    if torch.cuda.is_available():
        stack_hourglass.cuda()

    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Train process
    for i, (idx, data, target) in enumerate(data_loader):
        data = Variable(data.type(Tensor))
        target = Variable(target.type(Tensor), requires_grad=False)
        output = model(data)
        op = get_points(output)
        tg = get_points(target)
        eval_pckh(mpii)

