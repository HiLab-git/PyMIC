# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
import numpy as np
import time
import torch
from pymic.loss.seg.ssl import EntropyLoss
from pymic.net_run.weak_sup import WSLSegAgent
from pymic.util.ramps import get_rampup_ratio

class WSLEntropyMinimization(WSLSegAgent):
    """
    Weakly supervised segmentation based on Entropy Minimization.

    * Reference: Yves Grandvalet and Yoshua Bengio:
      Semi-supervised Learningby Entropy Minimization.
      `NeurIPS, 2005. <https://papers.nips.cc/paper/2004/file/96f2b50b5d3613adf9c27049b2a888c7-Paper.pdf>`_ 
    
    :param config: (dict) A dictionary containing the configuration.
    :param stage: (str) One of the stage in `train` (default), `inference` or `test`. 

    .. note::

        In the configuration dictionary, in addition to the four sections (`dataset`,
        `network`, `training` and `inference`) used in fully supervised learning, an 
        extra section `weakly_supervised_learning` is needed. See :doc:`usage.wsl` for details.
    """
    def __init__(self, config, stage = 'train'):
        super(WSLEntropyMinimization, self).__init__(config, stage)

    def training(self):
        class_num   = self.config['network']['class_num']
        iter_valid  = self.config['training']['iter_valid']
        wsl_cfg     = self.config['weakly_supervised_learning']
        iter_max     = self.config['training']['iter_max']
        rampup_start = wsl_cfg.get('rampup_start', 0)
        rampup_end   = wsl_cfg.get('rampup_end', iter_max)
        train_loss, train_loss_sup, train_loss_reg = 0, 0, 0
        data_time, gpu_time, loss_time, back_time = 0, 0, 0, 0
        self.net.train()
        for it in range(iter_valid):
            t0 = time.time()
            try:
                data = next(self.trainIter)
            except StopIteration:
                self.trainIter = iter(self.train_loader)
                data = next(self.trainIter)
            t1 = time.time()
            # get the inputs
            inputs = self.convert_tensor_type(data['image'])
            y      = self.convert_tensor_type(data['label_prob'])  
                         
            inputs, y = inputs.to(self.device), y.to(self.device)
            
            # zero the parameter gradients
            self.optimizer.zero_grad()
                
            # forward + backward + optimize
            outputs = self.net(inputs)
            t2 = time.time()

            loss_sup = self.get_loss_value(data, outputs, y)
            loss_dict= {"prediction":outputs, 'softmax':True}
            loss_reg = EntropyLoss()(loss_dict)
            
            rampup_ratio = get_rampup_ratio(self.glob_it, rampup_start, rampup_end, "sigmoid")
            regular_w = wsl_cfg.get('regularize_w', 0.1) * rampup_ratio
            loss = loss_sup + regular_w*loss_reg
            t3 = time.time()
            loss.backward()
            t4 = time.time()
            self.optimizer.step()

            train_loss = train_loss + loss.item()
            train_loss_sup = train_loss_sup + loss_sup.item()
            train_loss_reg = train_loss_reg + loss_reg.item() 

            data_time = data_time + t1 - t0 
            gpu_time  = gpu_time  + t2 - t1
            loss_time = loss_time + t3 - t2
            back_time = back_time + t4 - t3
        train_avg_loss = train_loss / iter_valid
        train_avg_loss_sup = train_loss_sup / iter_valid
        train_avg_loss_reg = train_loss_reg / iter_valid

        train_scalers = {'loss': train_avg_loss, 'loss_sup':train_avg_loss_sup,
            'loss_reg':train_avg_loss_reg, 'regular_w':regular_w,
            'data_time': data_time, 'forward_time':gpu_time, 
            'loss_time':loss_time, 'backward_time':back_time }
        return train_scalers
    
    def write_scalars(self, train_scalars, valid_scalars, lr_value, glob_it):
        loss_reg_scalar  = {'train':train_scalars['loss_reg']}
        weight_scalar    = {'regular_w':train_scalars['regular_w']}
        self.summ_writer.add_scalars('loss_reg', loss_reg_scalar, glob_it)
        self.summ_writer.add_scalars('weight', weight_scalar, glob_it)
        super(WSLEntropyMinimization, self).write_scalars(train_scalars, valid_scalars, lr_value, glob_it) 