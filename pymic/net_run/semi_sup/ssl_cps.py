# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
import numpy as np
import torch
from random import random
from pymic.loss.seg.util import get_soft_label
from pymic.loss.seg.util import reshape_prediction_and_ground_truth
from pymic.loss.seg.util import get_classwise_dice
from pymic.io.image_read_write import save_nd_array_as_image
from pymic.net_run.semi_sup import SSLSegAgent
from pymic.util.ramps import get_rampup_ratio
from pymic.util.general import mixup, tensor_shape_match

class SSLCPS(SSLSegAgent):
    """
    Using cross pseudo supervision for semi-supervised segmentation.

    * Reference: Xiaokang Chen, Yuhui Yuan, Gang Zeng, Jingdong Wang, 
      Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision,
      `CVPR 2021 <https://arxiv.org/abs/2106.01226>`_, pp. 2613-2022.
    
    :param config: (dict) A dictionary containing the configuration.
    :param stage: (str) One of the stage in `train` (default), `inference` or `test`. 

    .. note::

        In the configuration dictionary, in addition to the four sections (`dataset`,
        `network`, `training` and `inference`) used in fully supervised learning, an 
        extra section `semi_supervised_learning` is needed. See :doc:`usage.ssl` for details.
    """
    def __init__(self, config, stage = 'train'):
        super(SSLCPS, self).__init__(config, stage)

    def training(self):
        class_num   = self.config['network']['class_num']
        iter_valid  = self.config['training']['iter_valid']
        ssl_cfg     = self.config['semi_supervised_learning']
        iter_max    = self.config['training']['iter_max']
        mixup_prob  = self.config['training'].get('mixup_probability', 0.0)
        rampup_start = ssl_cfg.get('rampup_start', 0)
        rampup_end   = ssl_cfg.get('rampup_end', iter_max)
        train_loss  = 0
        train_loss_sup1,  train_loss_pseudo_sup1 = 0, 0
        train_loss_sup2,  train_loss_pseudo_sup2 = 0, 0
        train_dice_list = []
        self.net.train()
        for it in range(iter_valid):
            try:
                data_lab = next(self.trainIter)
            except StopIteration:
                self.trainIter = iter(self.train_loader)
                data_lab = next(self.trainIter)
            try:
                data_unlab = next(self.trainIter_unlab)
            except StopIteration:
                self.trainIter_unlab = iter(self.train_loader_unlab)
                data_unlab = next(self.trainIter_unlab)

            # get the inputs
            x0   = self.convert_tensor_type(data_lab['image'])
            y0   = self.convert_tensor_type(data_lab['label_prob'])  
            x1   = self.convert_tensor_type(data_unlab['image'])

            # for debug
            # for i in range(x0.shape[0]):
            #     image_i = x0[i][0]
            #     label_i = np.argmax(y0[i], axis = 0)
            #     # pixw_i  = pix_w[i][0]
            #     print(image_i.shape, label_i.shape)
            #     image_name = "temp/image_{0:}_{1:}.nii.gz".format(it, i)
            #     label_name = "temp/label_{0:}_{1:}.nii.gz".format(it, i)
            #     save_nd_array_as_image(image_i, image_name, reference_name = None)
            #     save_nd_array_as_image(label_i, label_name, reference_name = None)
            # continue
            if(mixup_prob > 0 and random() < mixup_prob):
                x0, y0 = mixup(x0, y0) 
            inputs = torch.cat([x0, x1], dim = 0)               
            inputs, y0 = inputs.to(self.device), y0.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()
                
            outputs1, outputs2 = self.net(inputs) 
            outputs_soft1 = torch.softmax(outputs1, dim=1)
            outputs_soft2 = torch.softmax(outputs2, dim=1)

            n0 = list(x0.shape)[0] 
            p0 = outputs_soft1[:n0]
            loss_sup1 = self.get_loss_value(data_lab, outputs1[:n0], y0)
            loss_sup2 = self.get_loss_value(data_lab, outputs2[:n0], y0)
            
            # Get pseudo labels of unannotated data and convert to one-hot
            pse_outputs1 = torch.argmax(outputs_soft1[n0:].detach(), dim=1, keepdim=True)
            pse_outputs2 = torch.argmax(outputs_soft2[n0:].detach(), dim=1, keepdim=True)
            pse_prob1 = get_soft_label(pse_outputs1, class_num, self.tensor_type)
            pse_prob2 = get_soft_label(pse_outputs2, class_num, self.tensor_type)

            pse_sup1 = self.get_loss_value(data_unlab, outputs1[n0:], pse_prob2)
            pse_sup2 = self.get_loss_value(data_unlab, outputs2[n0:], pse_prob1)

            rampup_ratio = get_rampup_ratio(self.glob_it, rampup_start, rampup_end, "sigmoid")
            regular_w = ssl_cfg.get('regularize_w', 0.1) * rampup_ratio

            model1_loss = loss_sup1 + regular_w * pse_sup1
            model2_loss = loss_sup2 + regular_w * pse_sup2
            loss = model1_loss + model2_loss

            loss.backward()
            self.optimizer.step()

            train_loss = train_loss + loss.item()
            train_loss_sup1  = train_loss_sup1 + loss_sup1.item()
            train_loss_sup2  = train_loss_sup2 + loss_sup2.item() 
            train_loss_pseudo_sup1 = train_loss_pseudo_sup1 + pse_sup1.item()
            train_loss_pseudo_sup2 = train_loss_pseudo_sup2 + pse_sup2.item()

            # get dice evaluation for each class in annotated images
            if(isinstance(p0, tuple) or isinstance(p0, list)):
                p0 = p0[0] 
            p0_argmax = torch.argmax(p0, dim = 1, keepdim = True)
            p0_soft   = get_soft_label(p0_argmax, class_num, self.tensor_type)
            p0_soft, y0 = reshape_prediction_and_ground_truth(p0_soft, y0) 
            dice_list   = get_classwise_dice(p0_soft, y0)
            train_dice_list.append(dice_list.cpu().numpy())
        train_avg_loss = train_loss / iter_valid
        train_avg_loss_sup1 = train_loss_sup1 / iter_valid
        train_avg_loss_sup2 = train_loss_sup2 / iter_valid
        train_avg_loss_pse_sup1 = train_loss_pseudo_sup1 / iter_valid 
        train_avg_loss_pse_sup2 = train_loss_pseudo_sup2 / iter_valid 
        train_cls_dice = np.asarray(train_dice_list).mean(axis = 0)
        train_avg_dice = train_cls_dice[1:].mean()

        train_scalers = {'loss': train_avg_loss, 
            'loss_sup1':train_avg_loss_sup1, 'loss_sup2': train_avg_loss_sup2,
            'loss_pse_sup1':train_avg_loss_pse_sup1, 'loss_pse_sup2': train_avg_loss_pse_sup2,
            'regular_w':regular_w, 'avg_fg_dice':train_avg_dice, 'class_dice': train_cls_dice}
        return train_scalers
  
    def write_scalars(self, train_scalars, valid_scalars, lr_value, glob_it):
        loss_scalar ={'train':train_scalars['loss'], 
                      'valid':valid_scalars['loss']}
        loss_sup_scalar  = {'net1':train_scalars['loss_sup1'],
                            'net2':train_scalars['loss_sup2']}
        loss_pse_sup_scalar = {'net1':train_scalars['loss_pse_sup1'],
                               'net2':train_scalars['loss_pse_sup2']}
        dice_scalar ={'train':train_scalars['avg_fg_dice'], 'valid':valid_scalars['avg_fg_dice']}
        self.summ_writer.add_scalars('loss', loss_scalar, glob_it)
        self.summ_writer.add_scalars('loss_sup', loss_sup_scalar, glob_it)
        self.summ_writer.add_scalars('loss_pseudo_sup', loss_pse_sup_scalar, glob_it)
        self.summ_writer.add_scalars('regular_w', {'regular_w':train_scalars['regular_w']}, glob_it)
        self.summ_writer.add_scalars('lr', {"lr": lr_value}, glob_it)
        self.summ_writer.add_scalars('dice', dice_scalar, glob_it)
        class_num = self.config['network']['class_num']
        for c in range(class_num):
            cls_dice_scalar = {'train':train_scalars['class_dice'][c], \
                'valid':valid_scalars['class_dice'][c]}
            self.summ_writer.add_scalars('class_{0:}_dice'.format(c), cls_dice_scalar, glob_it)

        logging.info('train loss {0:.4f}, avg dice {1:.4f} '.format(
            train_scalars['loss'], train_scalars['avg_fg_dice']) + "[" + \
            ' '.join("{0:.4f}".format(x) for x in train_scalars['class_dice']) + "]")        
        logging.info('valid loss {0:.4f}, avg dice {1:.4f} '.format(
            valid_scalars['loss'], valid_scalars['avg_fg_dice']) + "[" + \
            ' '.join("{0:.4f}".format(x) for x in valid_scalars['class_dice']) + "]") 