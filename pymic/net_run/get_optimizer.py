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
    # see https://www.codeleading.com/article/44815584159/
    param_group = [{'params': net_params, 'initial_lr': lr}]
    if(keyword_match(name, "SGD")):
        nesterov = optim_params.get('nesterov', True)
        return optim.SGD(param_group, lr, 
            momentum = momentum, weight_decay = weight_decay, nesterov = nesterov)
    elif(keyword_match(name, "Adam")):
        return optim.Adam(param_group, lr, weight_decay = weight_decay)
    elif(keyword_match(name, "SparseAdam")):
        return optim.SparseAdam(param_group, lr)
    elif(keyword_match(name, "Adadelta")):
        return optim.Adadelta(param_group, lr, weight_decay = weight_decay)
    elif(keyword_match(name, "Adagrad")): 
        return optim.Adagrad(param_group, lr, weight_decay = weight_decay)
    elif(keyword_match(name, "Adamax")): 
        return optim.Adamax(param_group, lr, weight_decay = weight_decay)
    elif(keyword_match(name, "ASGD")): 
        return optim.ASGD(param_group, lr, weight_decay = weight_decay)
    elif(keyword_match(name, "LBFGS")): 
        return optim.LBFGS(param_group, lr)
    elif(keyword_match(name, "RMSprop")): 
        return optim.RMSprop(param_group, lr, momentum = momentum,
            weight_decay = weight_decay)
    elif(keyword_match(name, "Rprop")): 
        return optim.Rprop(param_group, lr)
    else:
        raise ValueError("unsupported optimizer {0:}".format(name))


def get_lr_scheduler(optimizer, sched_params):
    name       = sched_params["lr_scheduler"]
    val_it     = sched_params["iter_valid"]
    epoch_last = sched_params["last_iter"] 
    if(epoch_last > 0):
        epoch_last = int(epoch_last / val_it)
    if(name is None):
        return None
    if(keyword_match(name, "ReduceLROnPlateau")):
        patience_it = sched_params["ReduceLROnPlateau_patience".lower()]
        patience = patience_it / val_it   
        lr_gamma = sched_params["lr_gamma"]
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
            mode = "max", factor=lr_gamma, patience = patience)
    elif(keyword_match(name, "MultiStepLR")):
        lr_milestones = sched_params["lr_milestones"]
        lr_milestones = [int(item / val_it) for item in lr_milestones]
        lr_gamma  = sched_params["lr_gamma"]
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                    lr_milestones, lr_gamma, epoch_last)
    elif(keyword_match(name, "StepLR")):
        lr_step   = sched_params["lr_step"] / val_it
        lr_gamma  = sched_params["lr_gamma"]
        scheduler = lr_scheduler.StepLR(optimizer,
                    lr_step, lr_gamma, epoch_last)
    elif(keyword_match(name, "CosineAnnealingLR")):
        epoch_max  = sched_params["iter_max"] / val_it
        lr_min     = sched_params.get("lr_min", 0)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                    epoch_max, lr_min, epoch_last)
    elif(keyword_match(name, "PolynomialLR")):
        epoch_max  = sched_params["iter_max"] / val_it
        power      = sched_params["lr_power"]
        scheduler = lr_scheduler.PolynomialLR(optimizer,
                    epoch_max, power, epoch_last)
    else:
        raise ValueError("unsupported lr scheduler {0:}".format(name))
    return scheduler