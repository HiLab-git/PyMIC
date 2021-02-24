# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import time
import random
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from abc import ABCMeta, abstractmethod
from datetime import datetime
from pymic.util.parse_config import parse_config
from pymic.net_run.get_optimizer import get_optimiser

def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class NetRunAgent(object):
    __metaclass__ = ABCMeta
    def __init__(self, config, stage = 'train'):
        assert(stage in ['train', 'inference', 'test'])
        self.config = config
        self.stage  = stage
        if(stage == 'inference'):
            self.stage = 'test'
        self.train_set = None 
        self.valid_set = None 
        self.test_set  = None
        self.net       = None
        self.optimizer = None
        self.scheduler = None 
        self.loss_calculater = None 
        self.transform_dict  = None
        self.tensor_type   = config['dataset']['tensor_type']
        self.deterministic = config['training'].get('deterministic', True)
        self.random_seed   = config['training'].get('random_seed', 1)
        if(self.deterministic):
            seed_torch(self.random_seed)
        
    def set_datasets(self, train_set, valid_set, test_set):
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set  = test_set

    def set_transform_dict(self, custom_transform_dict):
        self.transform_dict = custom_transform_dict

    def set_network(self, net):
        self.net = net 

    def set_loss(self, loss):
        self.loss_calculater = loss 

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    @abstractmethod    
    def get_stage_dataset_from_config(self, stage):
        raise(ValueError("not implemented"))

    @abstractmethod
    def get_parameters_to_update(self):
        raise(ValueError("not implemented"))

    @abstractmethod
    def create_network(self):
        raise(ValueError("not implemented"))

    @abstractmethod
    def training(self):
        raise(ValueError("not implemented"))
        
    @abstractmethod
    def validation(self):
        raise(ValueError("not implemented"))

    @abstractmethod
    def train_valid(self):
        raise(ValueError("not implemented"))
    
    @abstractmethod
    def infer(self):
        raise(ValueError("not implemented"))

    def create_dataset(self):
        if(self.stage == 'train'):
            if(self.train_set is None):
                self.train_set = self.get_stage_dataset_from_config('train')
            if(self.valid_set is None):
                self.valid_set = self.get_stage_dataset_from_config('valid')
            if(self.deterministic):
                def worker_init_fn(worker_id):
                    random.seed(self.random_seed+worker_id)
                worker_init = worker_init_fn
            else:
                worker_init = None

            batch_size = self.config['training']['batch_size']
            self.train_loader = torch.utils.data.DataLoader(self.train_set, 
                batch_size = batch_size, shuffle=True, num_workers=batch_size * 4,
                worker_init_fn=worker_init)
            self.valid_loader = torch.utils.data.DataLoader(self.valid_set, 
                batch_size = 1, shuffle=False, num_workers= 4,
                worker_init_fn=worker_init)
        else:
            if(self.test_set  is None):
                self.test_set  = self.get_stage_dataset_from_config('test')
            self.test_loder = torch.utils.data.DataLoader(self.test_set, 
                batch_size = 1, shuffle=False, num_workers= 4)
       
    def create_optimizer(self, params):
        if(self.optimizer is None):
            self.optimizer = get_optimiser(self.config['training']['optimizer'],
                    params, 
                    self.config['training'])
        last_iter = -1
        if(self.checkpoint is not None):
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            last_iter = self.checkpoint['iteration'] - 1
        if(self.scheduler is None):
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                    self.config['training']['lr_milestones'],
                    self.config['training']['lr_gamma'],
                    last_epoch = last_iter)

    def convert_tensor_type(self, input_tensor):
        if(self.tensor_type == 'float'):
            return input_tensor.float()
        else:
            return input_tensor.double()

    def run(self):
        self.create_dataset()
        self.create_network()
        if(self.stage == 'train'):
            self.train_valid()
        else:
            self.infer()

