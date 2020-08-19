# -*- coding: utf-8 -*-
from __future__ import print_function, division

import sys
from pymic.util.parse_config import parse_config
from pymic.net_run.net_run_agent import  NetRunAgent
from pymic.net.net_dict import NetDict
from pymic.loss.loss_dict import LossDict
from my_net2d import MyUNet2D 
from my_loss  import MyFocalDiceLoss

my_net_dict = NetDict
my_net_dict["MyUNet2D"] = MyUNet2D
my_loss_dict = LossDict
my_loss_dict["MyFocalDiceLoss"] = MyFocalDiceLoss

def main():
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print('    python train_infer.py train config.cfg')
        exit()
    stage    = str(sys.argv[1])
    cfg_file = str(sys.argv[2])
    config   = parse_config(cfg_file)

    # use custormized CNN and loss function
    agent  = NetRunAgent(config, stage)
    agent.set_network_dict(my_net_dict)
    agent.set_loss_dict(my_loss_dict)
    agent.run()

if __name__ == "__main__":
    main()
