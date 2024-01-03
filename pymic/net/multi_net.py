# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn

class MultiNet(nn.Module):
    '''
    A combination of multiple networks. 
    Parameters should be saved in the `params` dictionary. 

    :param `net_names`: (list) A list of network class name.
    :param `infer_mode`: (int) Mode for inference. 0: only use the first network. 
        1: taking an average of all the networks. 
    '''
    def __init__(self, net_dict, params):
        super(MultiNet, self).__init__() 
        net_names        = params['net_type'] # should be a list of network class name
        self.output_mode = params.get('infer_mode', 0)
        self.networks    = nn.ModuleList([net_dict[item](params) for item in net_names]) 

    def forward(self, x):
        if(self.training):
            output = [net(x) for  net in self.networks]
        else:
            output = self.networks[0](x)
            if(self.output_mode == 1):
                for i in range(1, len(self.networks)):
                    output += self.networks[i](x)
                output = output / len(self.networks)
        return output
            