# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
from torch import optim
from torch.optim import lr_scheduler
from pymic.util.general import keyword_match

def get_optimizer(name, net_params, optim_params):
    lr = optim_params['learning_rate']
    momentum = optim_params['momentum']
    weight_decay = optim_params['weight_decay']
    if(keyword_match(name, "SGD")):
        return optim.SGD(net_params, lr, 
            momentum = momentum, weight_decay = weight_decay)
    elif(keyword_match(name, "Adam")):
        return optim.Adam(net_params, lr, weight_decay = weight_decay)
    elif(keyword_match(name, "SparseAdam")):
        return optim.SparseAdam(net_params, lr)
    elif(keyword_match(name, "Adadelta")):
        return optim.Adadelta(net_params, lr, weight_decay = weight_decay)
    elif(keyword_match(name, "Adagrad")): 
        return optim.Adagrad(net_params, lr, weight_decay = weight_decay)
    elif(keyword_match(name, "Adamax")): 
        return optim.Adamax(net_params, lr, weight_decay = weight_decay)
    elif(keyword_match(name, "ASGD")): 
        return optim.ASGD(net_params, lr, weight_decay = weight_decay)
    elif(keyword_match(name, "LBFGS")): 
        return optim.LBFGS(net_params, lr)
    elif(keyword_match(name, "RMSprop")): 
        return optim.RMSprop(net_params, lr, momentum = momentum,
            weight_decay = weight_decay)
    elif(keyword_match(name, "Rprop")): 
        return optim.Rprop(net_params, lr)
    else:
        raise ValueError("unsupported optimizer {0:}".format(name))


def get_lr_scheduler(optimizer, sched_params):
    name = sched_params["lr_scheduler"]
    if(name is None):
        return None
    lr_gamma = sched_params["lr_gamma"]
    if(keyword_match(name, "ReduceLROnPlateau")):
        patience_it = sched_params["ReduceLROnPlateau_patience".lower()]
        val_it = sched_params["iter_valid"]
        patience = patience_it / val_it
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
            mode = "max", factor=lr_gamma, patience = patience)
    elif(keyword_match(name, "MultiStepLR")):
        lr_milestones = sched_params["lr_milestones"]
        last_iter     = sched_params["last_iter"]
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                    lr_milestones, lr_gamma, last_iter)
    else:
        raise ValueError("unsupported lr scheduler {0:}".format(name))
    return scheduler