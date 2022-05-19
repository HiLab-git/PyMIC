# -*- coding: utf-8 -*-
from __future__ import print_function, division

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

class SSLSegAgent(SegmentationAgent):
    """
    Training and testing agent for semi-supervised segmentation
    """
    def __init__(self, config, stage = 'train'):
        super(SSLSegAgent, self).__init__(config, stage)
        self.transform_dict  = TransformDict
        self.train_set_unlab = None 

    def get_unlabeled_dataset_from_config(self):
        root_dir  = self.config['dataset']['root_dir']
        modal_num = self.config['dataset']['modal_num']
        transform_names = self.config['dataset']['train_transform_unlab']
        
        self.transform_list  = []
        if(transform_names is None or len(transform_names) == 0):
            data_transform = None 
        else:
            transform_param = self.config['dataset']
            transform_param['task'] = 'segmentation' 
            for name in transform_names:
                if(name not in self.transform_dict):
                    raise(ValueError("Undefined transform {0:}".format(name))) 
                one_transform = self.transform_dict[name](transform_param)
                self.transform_list.append(one_transform)
            data_transform = transforms.Compose(self.transform_list)

        csv_file = self.config['dataset'].get('train_csv_unlab', None)
        dataset  = NiftyDataset(root_dir=root_dir,
                                csv_file  = csv_file,
                                modal_num = modal_num,
                                with_label= False,
                                transform = data_transform )
        return dataset

    def create_dataset(self):
        super(SSLSegAgent, self).create_dataset()
        if(self.stage == 'train'):
            if(self.train_set_unlab is None):
                self.train_set_unlab = self.get_unlabeled_dataset_from_config()
            if(self.deterministic):
                def worker_init_fn(worker_id):
                    random.seed(self.random_seed+worker_id)
                worker_init = worker_init_fn
            else:
                worker_init = None

            bn_train_unlab = self.config['dataset']['train_batch_size_unlab']
            num_worker = self.config['dataset'].get('num_workder', 16)
            self.train_loader_unlab = torch.utils.data.DataLoader(self.train_set_unlab, 
                batch_size = bn_train_unlab, shuffle=True, num_workers= num_worker,
                worker_init_fn=worker_init)

    def training(self):
        class_num   = self.config['network']['class_num']
        iter_valid  = self.config['training']['iter_valid']
        ssl_cfg     = self.config['semi_supervised_learning']
        train_loss  = 0
        train_loss_sup = 0
        train_loss_unsup = 0
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
            inputs = torch.cat([x0, x1], dim = 0)               
            inputs, y0 = inputs.to(self.device), y0.to(self.device)
            
            # zero the parameter gradients
            self.optimizer.zero_grad()
                
            # forward + backward + optimize
            outputs = self.net(inputs)
            n0 = list(x0.shape)[0] 
            p0 = outputs[:n0]
            loss_sup = self.get_loss_value(data_lab, x0, p0, y0)
            loss_dict = {"prediction":outputs, 'softmax':True}
            loss_unsup = EntropyLoss()(loss_dict)
            
            iter_max = self.config['training']['iter_max']
            ramp_up_length = ssl_cfg.get('ramp_up_length', iter_max)
            consis_w = 0.0
            if(self.glob_it > ssl_cfg.get('iter_sup', 0)):
                consis_w = ssl_cfg.get('consis_w', 0.1)
                if(ramp_up_length is not None and ramp_up_length > 0):
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
       
        print('train loss {0:.4f}, avg dice {1:.4f}'.format(
            train_scalars['loss'], train_scalars['avg_dice']), train_scalars['class_dice'])        
        print('valid loss {0:.4f}, avg dice {1:.4f}'.format(
            valid_scalars['loss'], valid_scalars['avg_dice']), valid_scalars['class_dice'])  

    def train_valid(self):
        self.trainIter_unlab = iter(self.train_loader_unlab)   
        super(SSLSegAgent, self).train_valid()    
