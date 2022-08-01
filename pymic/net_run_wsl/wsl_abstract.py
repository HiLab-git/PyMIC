# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
from pymic.net_run.agent_seg import SegmentationAgent

class WSLSegAgent(SegmentationAgent):
    """
    Training and testing agent for semi-supervised segmentation
    """
    def __init__(self, config, stage = 'train'):
        super(WSLSegAgent, self).__init__(config, stage)

    def training(self):
        pass
        
    def write_scalars(self, train_scalars, valid_scalars, glob_it):
        loss_scalar ={'train':train_scalars['loss'], 
                      'valid':valid_scalars['loss']}
        loss_sup_scalar  = {'train':train_scalars['loss_sup']}
        loss_upsup_scalar  = {'train':train_scalars['loss_reg']}
        dice_scalar ={'train':train_scalars['avg_dice'], 'valid':valid_scalars['avg_dice']}
        self.summ_writer.add_scalars('loss', loss_scalar, glob_it)
        self.summ_writer.add_scalars('loss_sup', loss_sup_scalar, glob_it)
        self.summ_writer.add_scalars('loss_reg', loss_upsup_scalar, glob_it)
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
