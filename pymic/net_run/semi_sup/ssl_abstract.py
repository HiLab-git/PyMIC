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

class SSLSegAgent(SegmentationAgent):
    """
    Abstract class for semi-supervised segmentation.

    :param config: (dict) A dictionary containing the configuration.
    :param stage: (str) One of the stage in `train` (default), `inference` or `test`. 

    .. note::

        In the configuration dictionary, in addition to the four sections (`dataset`,
        `network`, `training` and `inference`) used in fully supervised learning, an 
        extra section `semi_supervised_learning` is needed. See :doc:`usage.ssl` for details.
    """
    def __init__(self, config, stage = 'train'):
        super(SSLSegAgent, self).__init__(config, stage)
        self.transform_dict  = TransformDict
        self.train_set_unlab = None 

    def get_unlabeled_dataset_from_config(self):
        """
        Create a dataset for the unlabeled images based on configuration.
        """
        train_dir  = self.config['dataset']['train_dir']
        modal_num = self.config['dataset'].get('modal_num', 1)
        transform_names = self.config['dataset']['train_transform_unlab']
        
        self.transform_list  = []
        if(transform_names is None or len(transform_names) == 0):
            data_transform = None 
        else:
            transform_param = self.config['dataset']
            transform_param['task'] = self.task_type
            for name in transform_names:
                if(name not in self.transform_dict):
                    raise(ValueError("Undefined transform {0:}".format(name))) 
                one_transform = self.transform_dict[name](transform_param)
                self.transform_list.append(one_transform)
            data_transform = transforms.Compose(self.transform_list)

        csv_file = self.config['dataset'].get('train_csv_unlab', None)
        dataset  = NiftyDataset(root_dir  = train_dir,
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
            num_worker = self.config['dataset'].get('num_worker', 16)
            self.train_loader_unlab = torch.utils.data.DataLoader(self.train_set_unlab, 
                batch_size = bn_train_unlab, shuffle=True, num_workers= num_worker,
                worker_init_fn=worker_init, drop_last = True)

    def write_scalars(self, train_scalars, valid_scalars, lr_value, glob_it):
        loss_scalar ={'train':train_scalars['loss'], 
                      'valid':valid_scalars['loss']}
        loss_sup_scalar  = {'train':train_scalars['loss_sup']}
        loss_upsup_scalar  = {'train':train_scalars['loss_reg']}
        dice_scalar ={'train':train_scalars['avg_fg_dice'], 'valid':valid_scalars['avg_fg_dice']}
        self.summ_writer.add_scalars('loss', loss_scalar, glob_it)
        self.summ_writer.add_scalars('loss_sup', loss_sup_scalar, glob_it)
        self.summ_writer.add_scalars('loss_reg', loss_upsup_scalar, glob_it)
        self.summ_writer.add_scalars('regular_w', {'regular_w':train_scalars['regular_w']}, glob_it)
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

    def train_valid(self):
        self.trainIter_unlab = iter(self.train_loader_unlab)   
        super(SSLSegAgent, self).train_valid()    
