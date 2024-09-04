# -*- coding: utf-8 -*-

from __future__ import print_function, division
import logging
import os
import sys
import numpy as np
import time
import torch
import torch.nn as nn
from pymic.loss.seg.util import get_soft_label
from pymic.loss.seg.util import reshape_prediction_and_ground_truth
from pymic.loss.seg.util import get_classwise_dice
from pymic.loss.seg.util import reshape_tensor_to_2D
from pymic.net_run.agent_seg import SegmentationAgent
from pymic.util.parse_config import *
from pymic.util.ramps import get_rampup_ratio

class NLLCoTeaching(SegmentationAgent):
    """
    Co-teaching for noisy-label learning. 

    * Reference: Bo Han, Quanming Yao, Xingrui Yu, Gang Niu, Miao Xu, Weihua Hu, 
      Ivor Tsang, Masashi Sugiyama.  Co-teaching: Robust Training of Deep Neural Networks with Extremely 
      Noisy Labels. `NeurIPS 201. <https://arxiv.org/abs/1804.06872>`_
    
    :param config: (dict) A dictionary containing the configuration.
    :param stage: (str) One of the stage in `train` (default), `inference` or `test`. 

    .. note::

        In the configuration dictionary, in addition to the four sections (`dataset`,
        `network`, `training` and `inference`) used in fully supervised learning, an 
        extra section `noisy_label_learning` is needed. See :doc:`usage.nll` for details.
    """
    def __init__(self, config, stage = 'train'):
        super(NLLCoTeaching, self).__init__(config, stage)
        loss_type = config['training']["loss_type"]
        if(loss_type != "CrossEntropyLoss"):
            logging.warn("only CrossEntropyLoss supported for" +  
            " coteaching, the specified loss {0:} is ingored".format(loss_type))

    def training(self):
        class_num   = self.config['network']['class_num']
        iter_valid  = self.config['training']['iter_valid']
        nll_cfg     = self.config['noisy_label_learning']
        select_ratio = nll_cfg['co_teaching_select_ratio']
        iter_max     = self.config['training']['iter_max']
        rampup_start = nll_cfg.get('rampup_start', 0)
        rampup_end   = nll_cfg.get('rampup_end', iter_max)

        train_loss_no_select1, train_loss_no_select2  = 0, 0
        train_loss1, train_avg_loss2 = 0, 0
        train_dice_list = []
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
            inputs      = self.convert_tensor_type(data['image'])
            labels_prob = self.convert_tensor_type(data['label_prob'])
            inputs, labels_prob = inputs.to(self.device), labels_prob.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()
                
            # forward + backward + optimize
            outputs1, outputs2 = self.net(inputs)
            t2 = time.time()
            prob1 = nn.Softmax(dim = 1)(outputs1)
            prob2 = nn.Softmax(dim = 1)(outputs2)
            prob1_2d = reshape_tensor_to_2D(prob1) * 0.999 + 5e-4
            prob2_2d = reshape_tensor_to_2D(prob2) * 0.999 + 5e-4
            y_2d  = reshape_tensor_to_2D(labels_prob)

            loss1 = - y_2d* torch.log(prob1_2d)
            loss1 = torch.sum(loss1, dim = 1) # shape is [N]
            ind_1_sorted = torch.argsort(loss1)

            loss2 = - y_2d* torch.log(prob2_2d)
            loss2 = torch.sum(loss2, dim = 1) # shape is [N]
            ind_2_sorted = torch.argsort(loss2)

            rampup_ratio = get_rampup_ratio(self.glob_it, rampup_start, rampup_end, "sigmoid")
            forget_ratio = (1 - select_ratio) * rampup_ratio
            remb_ratio   = 1 - forget_ratio
            num_remb = int(remb_ratio * len(loss1))

            ind_1_update = ind_1_sorted[:num_remb]
            ind_2_update = ind_2_sorted[:num_remb]

            loss1_select = loss1[ind_2_update]
            loss2_select = loss2[ind_1_update]
            
            loss = loss1_select.mean() + loss2_select.mean()
            t3 = time.time()
            loss.backward()
            t4 = time.time()
            self.optimizer.step()

            train_loss_no_select1 = train_loss_no_select1 + loss1.mean().item()
            train_loss_no_select2 = train_loss_no_select2 + loss2.mean().item()
            train_loss1 = train_loss1 + loss1_select.mean().item()
            train_loss2 = train_loss2 + loss2_select.mean().item()

            outputs1_argmax = torch.argmax(outputs1, dim = 1, keepdim = True)
            soft_out1       = get_soft_label(outputs1_argmax, class_num, self.tensor_type)
            soft_out1, labels_prob = reshape_prediction_and_ground_truth(soft_out1, labels_prob)  
            dice_list   = get_classwise_dice(soft_out1, labels_prob).detach().cpu().numpy()
            train_dice_list.append(dice_list)

            data_time = data_time + t1 - t0 
            gpu_time  = gpu_time  + t2 - t1
            loss_time = loss_time + t3 - t2
            back_time = back_time + t4 - t3
        train_avg_loss_no_select1 = train_loss_no_select1 / iter_valid
        train_avg_loss_no_select2 = train_loss_no_select2 / iter_valid
        train_avg_loss1 = train_loss1 / iter_valid
        train_avg_loss2 = train_loss2 / iter_valid
        train_cls_dice = np.asarray(train_dice_list).mean(axis = 0)
        train_avg_dice = train_cls_dice[1:].mean()

        train_scalers = {'loss': (train_avg_loss1 + train_avg_loss2) / 2, 
            'loss1':train_avg_loss1, 'loss2': train_avg_loss2,
            'loss_no_select1':train_avg_loss_no_select1, 
            'loss_no_select2':train_avg_loss_no_select2,
            'select_ratio':remb_ratio, 'avg_fg_dice':train_avg_dice, 'class_dice': train_cls_dice,
            'data_time': data_time, 'forward_time':gpu_time, 
            'loss_time':loss_time, 'backward_time':back_time }
        return train_scalers
    
    def write_scalars(self, train_scalars, valid_scalars, lr_value, glob_it):
        loss_scalar ={'train':train_scalars['loss'], 
                      'valid':valid_scalars['loss']}
        loss_no_select_scalar  = {'net1':train_scalars['loss_no_select1'],
                                  'net2':train_scalars['loss_no_select2']}

        dice_scalar ={'train':train_scalars['avg_fg_dice'], 'valid':valid_scalars['avg_fg_dice']}
        self.summ_writer.add_scalars('loss', loss_scalar, glob_it)
        self.summ_writer.add_scalars('loss_no_select', loss_no_select_scalar, glob_it)
        self.summ_writer.add_scalars('select_ratio', {'select_ratio':train_scalars['select_ratio']}, glob_it)
        self.summ_writer.add_scalars('lr', {"lr": lr_value}, glob_it)
        self.summ_writer.add_scalars('dice', dice_scalar, glob_it)
        class_num = self.config['network']['class_num']
        for c in range(class_num):
            cls_dice_scalar = {'train':train_scalars['class_dice'][c], \
                'valid':valid_scalars['class_dice'][c]}
            self.summ_writer.add_scalars('class_{0:}_dice'.format(c), cls_dice_scalar, glob_it)

        logging.info('train loss {0:.4f}, avg foreground dice {1:.4f} '.format(
            train_scalars['loss'], train_scalars['avg_fg_dice']) + "[" + \
            ' '.join("{0:.4f}".format(x) for x in train_scalars['class_dice']) + "]")        
        logging.info('valid loss {0:.4f}, avg foreground dice {1:.4f} '.format(
            valid_scalars['loss'], valid_scalars['avg_fg_dice']) + "[" + \
            ' '.join("{0:.4f}".format(x) for x in valid_scalars['class_dice']) + "]") 
        logging.info("data: {0:.2f}s, forward: {1:.2f}s, loss: {2:.2f}s, backward: {3:.2f}s".format(
                train_scalars['data_time'], train_scalars['forward_time'], 
                train_scalars['loss_time'], train_scalars['backward_time']))  
