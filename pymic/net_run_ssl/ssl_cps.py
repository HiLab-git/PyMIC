# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
import numpy as np
import torch
import torch.optim as optim
from pymic.loss.seg.util import get_soft_label
from pymic.loss.seg.util import reshape_prediction_and_ground_truth
from pymic.loss.seg.util import get_classwise_dice
from pymic.net_run.get_optimizer import get_optimizer, get_lr_scheduler
from pymic.net_run_ssl.ssl_abstract import SSLSegAgent
from pymic.net.net_dict_seg import SegNetDict
from pymic.util.ramps import sigmoid_rampup
from pymic.util.general import keyword_match

class SSLCPS(SSLSegAgent):
    """
    Using cross pseudo supervision according to the following paper:
    Xiaokang Chen, Yuhui Yuan, Gang Zeng, Jingdong Wang, 
    Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision,
    CVPR 2021, pp. 2613-2022.
    https://arxiv.org/abs/2106.01226 
    """
    def __init__(self, config, stage = 'train'):
        super(SSLCPS, self).__init__(config, stage)
        self.net2 = None 
        self.optimizer2 = None 
        self.scheduler2 = None

    def create_network(self):
        super(SSLCPS, self).create_network()
        if(self.net2 is None):
            net_name = self.config['network']['net_type']
            if(net_name not in SegNetDict):
                raise ValueError("Undefined network {0:}".format(net_name))
            self.net2 = SegNetDict[net_name](self.config['network'])
        if(self.tensor_type == 'float'):
            self.net2.float()
        else:
            self.net2.double()

    def train_valid(self):
        # create optimizor for the second network
        opt_params = self.config['training']
        if(self.optimizer2 is None):
            self.optimizer2 = get_optimizer(opt_params['optimizer'],
                    self.net2.parameters(), opt_params)
        last_iter = -1
        # if(self.checkpoint is not None):
        #     self.optimizer2.load_state_dict(self.checkpoint['optimizer_state_dict'])
        #     last_iter = self.checkpoint['iteration'] - 1
        if(self.scheduler2 is None):
            opt_params["laster_iter"] = last_iter
            self.scheduler2 = get_lr_scheduler(self.optimizer, opt_params)
        super(SSLCPS, self).train_valid()

    def training(self):
        class_num   = self.config['network']['class_num']
        iter_valid  = self.config['training']['iter_valid']
        ssl_cfg     = self.config['semi_supervised_learning']
        train_loss  = 0
        train_loss_sup1,  train_loss_pseudo_sup1 = 0, 0
        train_loss_sup2,  train_loss_pseudo_sup2 = 0, 0
        train_dice_list = []
        self.net.train()
        self.net2.train()
        self.net2.to(self.device)
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
            inputs = torch.cat([x0, x1], dim = 0)               
            inputs, y0 = inputs.to(self.device), y0.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()
            self.optimizer2.zero_grad()
                
            outputs1, outputs2 = self.net(inputs), self.net2(inputs)
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

            iter_max = self.config['training']['iter_max']
            ramp_up_len = ssl_cfg.get('ramp_up_length', iter_max)
            regular_w = 0.0
            if(self.glob_it > ssl_cfg.get('iter_sup', 0)):
                regular_w = ssl_cfg.get('regularize_w', 0.1)
                if(ramp_up_len is not None and self.glob_it < ramp_up_len):
                    regular_w = regular_w * sigmoid_rampup(self.glob_it, ramp_up_len)

            model1_loss = loss_sup1 + regular_w * pse_sup1
            model2_loss = loss_sup2 + regular_w * pse_sup2
            loss = model1_loss + model2_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer2.step()
            if(not keyword_match(self.config['training']['lr_scheduler'], "ReduceLROnPlateau")):
                self.scheduler.step()
                self.scheduler2.step()   

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
        train_avg_dice = train_cls_dice.mean()

        train_scalers = {'loss': train_avg_loss, 
            'loss_sup1':train_avg_loss_sup1, 'loss_sup2': train_avg_loss_sup2,
            'loss_pse_sup1':train_avg_loss_pse_sup1, 'loss_pse_sup2': train_avg_loss_pse_sup2,
            'regular_w':regular_w, 'avg_dice':train_avg_dice, 'class_dice': train_cls_dice}
        return train_scalers

    def validation(self):
        return_value =  super(SSLCPS, self).validation()
        if(keyword_match(self.config['training']['lr_scheduler'], "ReduceLROnPlateau")):
            self.scheduler2.step(return_value['avg_dice'])
        return return_value
    
    def write_scalars(self, train_scalars, valid_scalars, glob_it):
        loss_scalar ={'train':train_scalars['loss'], 
                      'valid':valid_scalars['loss']}
        loss_sup_scalar  = {'net1':train_scalars['loss_sup1'],
                            'net2':train_scalars['loss_sup2']}
        loss_pse_sup_scalar = {'net1':train_scalars['loss_pse_sup1'],
                               'net2':train_scalars['loss_pse_sup2']}
        dice_scalar ={'train':train_scalars['avg_dice'], 'valid':valid_scalars['avg_dice']}
        self.summ_writer.add_scalars('loss', loss_scalar, glob_it)
        self.summ_writer.add_scalars('loss_sup', loss_sup_scalar, glob_it)
        self.summ_writer.add_scalars('loss_pseudo_sup', loss_pse_sup_scalar, glob_it)
        self.summ_writer.add_scalars('regular_w', {'regular_w':train_scalars['regular_w']}, glob_it)
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