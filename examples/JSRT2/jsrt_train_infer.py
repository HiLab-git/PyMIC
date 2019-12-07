# -*- coding: utf-8 -*-
from __future__ import print_function, division

import sys
from pymic.train_infer.train_infer import  TrainInferAgent
from pymic.util.parse_config import parse_config
from my_net2d import MyUNet2D 

if __name__ == "__main__":
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print('    python train_infer.py train config.cfg')
        exit()
    stage    = str(sys.argv[1])
    cfg_file = str(sys.argv[2])
    config   = parse_config(cfg_file)

    # use custormized CNN
    net_param = {'in_chns': 1,
                 'feature_chns':[16, 32, 64, 128],
                 'class_num': 2,
                 'acti_func': 'relu',
                 'dropout': True}
    config['network'] = net_param
    net    = MyUNet2D(net_param)

    agent  = TrainInferAgent(config, stage)
    agent.set_network(net)
    agent.run()
