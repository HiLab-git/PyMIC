# -*- coding: utf-8 -*-
from __future__ import print_function, division

import sys
from pymic.util.parse_config import parse_config
from pymic.net_run.net_run_agent import  NetRunAgent
from pymic.net.net_factory import net_dict
from pymic.loss.loss_factory import loss_dict
from my_net2d import MyUNet2D 
from my_loss  import MyFocalDiceLoss

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

my_loss_dict = {
    "MyFocalDiceLoss": MyFocalDiceLoss
}

def get_loss(params):
    loss_type = params["loss_type"]
    if(loss_type in my_loss_dict):
        loss_obj = my_loss_dict[loss_type](params)
    elif(loss_type in net_dict):
        loss_obj = loss_dict[loss_type](params)
    else:
        raise ValueError("Undefined loss: {0:}".format(loss_type))
    return loss_obj

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
    loss_obj = get_loss(config['training'])
    agent  = NetRunAgent(config, stage)
    agent.set_network(net)
    agent.set_loss_calculater(loss_obj)
    agent.run()

if __name__ == "__main__":
    main()
