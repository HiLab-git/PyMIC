# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
from pymic.net_run.agent_seg import SegmentationAgent

class WSLSegAgent(SegmentationAgent):
    """
    Abstract agent for weakly supervised segmentation.

    :param config: (dict) A dictionary containing the configuration.
    :param stage: (str) One of the stage in `train` (default), `inference` or `test`. 

    .. note::

        In the configuration dictionary, in addition to the four sections (`dataset`,
        `network`, `training` and `inference`) used in fully supervised learning, an 
        extra section `weakly_supervised_learning` is needed. See :doc:`usage.wsl` for details.
    """
    def __init__(self, config, stage = 'train'):
        super(WSLSegAgent, self).__init__(config, stage)
        
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
