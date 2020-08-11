# -*- coding: utf-8 -*-
from __future__ import print_function, division

import sys
from pymic.net_run.net_run import  TrainInferAgent
from pymic.net_run.net_factory import net_dict
from pymic.util.parse_config import parse_config
from my_net2d import MyUNet2D 
from my_loss  import MySegmentationLossCalculator

my_net_dict = {
    "MyUNet2D": MyUNet2D
}

def get_network(params):
    net_type = params["net_type"]
    if(net_type in my_net_dict):
        net = my_net_dict[net_type](params)
    elif(net_type in net_dict):
        net = net_dict[net_type](params)
    else:
        raise ValueError("Undefined network: {0:}".format(net_type))
    return net 

def main():
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print('    python train_infer.py train config.cfg')
        exit()
    stage    = str(sys.argv[1])
    cfg_file = str(sys.argv[2])
    config   = parse_config(cfg_file)

    # use custormized CNN and loss function
    net      = get_network(config['network'])
    loss_cal = MySegmentationLossCalculator(config['training'])
    agent  = TrainInferAgent(config, stage)
    agent.set_network(net)
    agent.set_loss_calculater(loss_cal)
    agent.run()

if __name__ == "__main__":
    main()
