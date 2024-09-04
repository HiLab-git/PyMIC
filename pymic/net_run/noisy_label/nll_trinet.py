# -*- coding: utf-8 -*-

from __future__ import print_function, division
import logging
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pymic.loss.seg.util import get_soft_label
from pymic.loss.seg.util import reshape_prediction_and_ground_truth
from pymic.loss.seg.util import get_classwise_dice
from pymic.loss.seg.util import reshape_tensor_to_2D
from pymic.net_run.agent_seg import SegmentationAgent
from pymic.net.net_dict_seg import SegNetDict
from pymic.util.parse_config import *
from pymic.util.ramps import get_rampup_ratio

class NLLTriNet(SegmentationAgent):
    """
    Implementation of trinet for learning from noisy samples for 
    segmentation tasks.

    * Reference: Tianwei Zhang, Lequan Yu, Na Hu, Su Lv, Shi Gu:
      Robust Medical Image Segmentation from Non-expert Annotations with Tri-network.
      `MICCAI 2020. <https://link.springer.com/chapter/10.1007/978-3-030-59719-1_25>`_
         
    :param config: (dict) A dictionary containing the configuration.
    :param stage: (str) One of the stage in `train` (default), `inference` or `test`. 

    .. note::

        In the configuration dictionary, in addition to the four sections (`dataset`,
        `network`, `training` and `inference`) used in fully supervised learning, an 
        extra section `noisy_label_learning` is needed. See :doc:`usage.nll` for details.
    """
    def __init__(self, config, stage = 'train'):
        super(NLLTriNet, self).__init__(config, stage)
       
    def get_loss_and_confident_mask(self, pred, labels_prob, conf_ratio):
        prob = nn.Softmax(dim = 1)(pred)
        prob_2d = reshape_tensor_to_2D(prob) * 0.999 + 5e-4
        y_2d  = reshape_tensor_to_2D(labels_prob)

        loss = - y_2d* torch.log(prob_2d)
        loss = torch.sum(loss, dim = 1) # shape is [N]
        threshold   = torch.quantile(loss, conf_ratio)
        mask = loss < threshold
        return loss, mask

    def training(self):
        class_num   = self.config['network']['class_num']
        iter_valid  = self.config['training']['iter_valid']
        nll_cfg     = self.config['noisy_label_learning']
        iter_max     = self.config['training']['iter_max']
        select_ratio = nll_cfg['trinet_select_ratio']
        rampup_start = nll_cfg.get('rampup_start', 0)
        rampup_end   = nll_cfg.get('rampup_end', iter_max)

        train_loss_no_select1 = 0
        train_loss_no_select2 = 0
        train_loss1, train_loss2, train_loss3 = 0, 0, 0
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
            outputs1, outputs2, outputs3 = self.net(inputs)
            t2 = time.time()
            rampup_ratio = get_rampup_ratio(self.glob_it, rampup_start, rampup_end)
            forget_ratio = (1 - select_ratio) * rampup_ratio
            remb_ratio   = 1 - forget_ratio

            loss1, mask1 = self.get_loss_and_confident_mask(outputs1, labels_prob, remb_ratio)
            loss2, mask2 = self.get_loss_and_confident_mask(outputs2, labels_prob, remb_ratio)
            loss3, mask3 = self.get_loss_and_confident_mask(outputs3, labels_prob, remb_ratio)
            mask12, mask13, mask23 = mask1 * mask2, mask1 * mask3, mask2 * mask3 
            mask12, mask13, mask23 = mask12.detach(), mask13.detach(), mask23.detach()

            loss1_avg = torch.sum(loss1 * mask23) / mask23.sum()
            loss2_avg = torch.sum(loss2 * mask13) / mask13.sum()
            loss3_avg = torch.sum(loss3 * mask12) / mask12.sum()
            loss = (loss1_avg + loss2_avg + loss3_avg) / 3
            t3 = time.time()
            loss.backward()
            t4 = time.time()
            self.optimizer.step()

            train_loss_no_select1 = train_loss_no_select1 + loss1.mean().item()
            train_loss_no_select2 = train_loss_no_select2 + loss2.mean().item()
            train_loss1 = train_loss1 + loss1_avg.item()
            train_loss2 = train_loss2 + loss2_avg.item()

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

        logging.info('train loss {0:.4f}, avg foregournd dice {1:.4f} '.format(
            train_scalars['loss'], train_scalars['avg_fg_dice']) + "[" + \
            ' '.join("{0:.4f}".format(x) for x in train_scalars['class_dice']) + "]")        
        logging.info('valid loss {0:.4f}, avg foreground dice {1:.4f} '.format(
            valid_scalars['loss'], valid_scalars['avg_fg_dice']) + "[" + \
            ' '.join("{0:.4f}".format(x) for x in valid_scalars['class_dice']) + "]")
        logging.info("data: {0:.2f}s, forward: {1:.2f}s, loss: {2:.2f}s, backward: {3:.2f}s".format(
                train_scalars['data_time'], train_scalars['forward_time'], 
                train_scalars['loss_time'], train_scalars['backward_time'])) 
