# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
import numpy as np
import random
import torch
import torchvision.transforms as transforms
from pymic.io.nifty_dataset import NiftyDataset
from pymic.loss.seg.util import get_soft_label
from pymic.loss.seg.util import reshape_prediction_and_ground_truth
from pymic.loss.seg.util import get_classwise_dice
from pymic.loss.seg.ssl import EntropyLoss
from pymic.net_run.agent_seg import SegmentationAgent
from pymic.transform.trans_dict import TransformDict
from pymic.util.ramps import sigmoid_rampup

class WSLEntropyMinimization(SegmentationAgent):
    """
    Training and testing agent for semi-supervised segmentation
    """
    def __init__(self, config, stage = 'train'):
        super(WSLSegAgent, self).__init__(config, stage)
        self.transform_dict  = TransformDict

    def training(self):
        class_num   = self.config['network']['class_num']
        iter_valid  = self.config['training']['iter_valid']
        wsl_cfg     = self.config['weakly_supervised_learning']
        train_loss  = 0
        train_loss_sup = 0
        train_loss_unsup = 0
        train_dice_list = []
        self.net.train()
        for it in range(iter_valid):
            try:
                data = next(self.trainIter)
            except StopIteration:
                self.trainIter = iter(self.train_loader)
                data = next(self.trainIter)
            
            # get the inputs
            inputs = self.convert_tensor_type(data['image'])
            y      = self.convert_tensor_type(data['label_prob'])  
                         
            inputs, y = inputs.to(self.device), y.to(self.device)
            
            # zero the parameter gradients
            self.optimizer.zero_grad()
                
            # forward + backward + optimize
            outputs = self.net(inputs)
            loss_sup = self.get_loss_value(data, outputs, y)
            loss_dict = {"prediction":outputs, 'softmax':True}
            loss_unsup = EntropyLoss()(loss_dict)
            
            iter_max = self.config['training']['iter_max']
            ramp_up_length = wsl_cfg.get('ramp_up_length', iter_max)
            consis_w = 0.0
            if(self.glob_it > wsl_cfg.get('iter_sup', 0)):
                consis_w = wsl_cfg.get('consis_w', 0.1)
                if(ramp_up_length is not None and self.glob_it < ramp_up_length):
                    consis_w = consis_w * sigmoid_rampup(self.glob_it, ramp_up_length)
            loss = loss_sup + consis_w*loss_unsup
            # if (self.config['training']['use'])
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            train_loss = train_loss + loss.item()
            train_loss_sup   = train_loss_sup + loss_sup.item()
            train_loss_unsup = train_loss_unsup + loss_unsup.item() 
            # get dice evaluation for each class in annotated images
            if(isinstance(p0, tuple) or isinstance(p0, list)):
                p0 = p0[0] 
            p0_argmax = torch.argmax(p0, dim = 1, keepdim = True)
            p0_soft   = get_soft_label(p0_argmax, class_num, self.tensor_type)
            p0_soft, y0 = reshape_prediction_and_ground_truth(p0_soft, y0) 
            dice_list   = get_classwise_dice(p0_soft, y0)
            train_dice_list.append(dice_list.cpu().numpy())
        train_avg_loss = train_loss / iter_valid
        train_avg_loss_sup = train_loss_sup / iter_valid
        train_avg_loss_unsup = train_loss_unsup / iter_valid
        train_cls_dice = np.asarray(train_dice_list).mean(axis = 0)
        train_avg_dice = train_cls_dice.mean()

        train_scalers = {'loss': train_avg_loss, 'loss_sup':train_avg_loss_sup,
            'loss_unsup':train_avg_loss_unsup, 'consis_w':consis_w,
            'avg_dice':train_avg_dice,     'class_dice': train_cls_dice}
        return train_scalers
        
    def write_scalars(self, train_scalars, valid_scalars, glob_it):
        loss_scalar ={'train':train_scalars['loss'], 
                      'valid':valid_scalars['loss']}
        loss_sup_scalar  = {'train':train_scalars['loss_sup']}
        loss_upsup_scalar  = {'train':train_scalars['loss_unsup']}
        dice_scalar ={'train':train_scalars['avg_dice'], 'valid':valid_scalars['avg_dice']}
        self.summ_writer.add_scalars('loss', loss_scalar, glob_it)
        self.summ_writer.add_scalars('loss_sup', loss_sup_scalar, glob_it)
        self.summ_writer.add_scalars('loss_unsup', loss_upsup_scalar, glob_it)
        self.summ_writer.add_scalars('consis_w', {'consis_w':train_scalars['consis_w']}, glob_it)
        self.summ_writer.add_scalars('dice', dice_scalar, glob_it)
        class_num = self.config['network']['class_num']
        for c in range(class_num):
            cls_dice_scalar = {'train':train_scalars['class_dice'][c], \
                'valid':valid_scalars['class_dice'][c]}
            self.summ_writer.add_scalars('class_{0:}_dice'.format(c), cls_dice_scalar, glob_it)
        logging.info('train loss {0:.4f}, avg dice {1:.4f} '.format(
            train_scalars['loss'], train_scalars['avg_dice']) + "[" + \
            ' '.join("{0:.4f}".format(x) for x in train_scalars['class_dice']) + "]")        
        logging.info('valid loss {0:.4f}, avg dice {1:.4f} '.format(
            valid_scalars['loss'], valid_scalars['avg_dice']) + "[" + \
            ' '.join("{0:.4f}".format(x) for x in valid_scalars['class_dice']) + "]")  
