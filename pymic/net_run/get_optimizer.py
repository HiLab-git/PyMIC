# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.optim as optim

def get_optimiser(name, net_params, optim_params):
    lr = optim_params['learning_rate']
    momentum = optim_params['momentum']
    weight_decay = optim_params['weight_decay']
    if(name == "SGD"):
        return optim.SGD(net_params, lr, 
            momentum = momentum, weight_decay = weight_decay)
    elif(name == "Adam"):
        return optim.Adam(net_params, lr, weight_decay = weight_decay)
    elif(name == "SparseAdam"):
        return optim.SparseAdam(net_params, lr)
    elif(name == "Adadelta"):
        return optim.Adadelta(net_params, lr, weight_decay = weight_decay)
    elif(name == "Adagrad"):
        return optim.Adagrad(net_params, lr, weight_decay = weight_decay)
    elif(name == "Adamax"):
        return optim.Adamax(net_params, lr, weight_decay = weight_decay)
    elif(name == "ASGD"):
        return optim.ASGD(net_params, lr, weight_decay = weight_decay)
    elif(name == "LBFGS"):
        return optim.LBFGS(net_params, lr)
    elif(name == "RMSprop"):
        return optim.RMSprop(net_params, lr, momentum = momentum,
            weight_decay = weight_decay)
    elif(name == "Rprop"):
        return optim.Rprop(net_params, lr)
    else:
        raise ValueError("unsupported optimizer {0:}".format(name))
