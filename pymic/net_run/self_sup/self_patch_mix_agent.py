# -*- coding: utf-8 -*-
from __future__ import print_function, division
import copy
import os
import sys
import shutil
import time
import logging
import scipy
import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from random import random
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from pymic.io.image_read_write import save_nd_array_as_image
from pymic.io.nifty_dataset import NiftyDataset
from pymic.net.net_dict_seg import SegNetDict
from pymic.net_run.agent_abstract import NetRunAgent
from pymic.net_run.infer_func import Inferer
from pymic.loss.loss_dict_seg import SegLossDict
from pymic.loss.seg.combined import CombinedLoss
from pymic.loss.seg.deep_sup import DeepSuperviseLoss
from pymic.loss.seg.util import get_soft_label
from pymic.loss.seg.util import reshape_prediction_and_ground_truth
from pymic.loss.seg.util import get_classwise_dice
from pymic.transform.trans_dict import TransformDict
from pymic.util.post_process import PostProcessDict
from pymic.util.image_process import convert_label
from pymic.util.parse_config import *
from pymic.io.image_read_write import save_nd_array_as_image
from pymic.net_run.self_sup.util import patch_mix
from pymic.net_run.agent_seg import SegmentationAgent

class SelfSLPatchMixAgent(SegmentationAgent):
    """
    Abstract class for self-supervised segmentation.

    :param config: (dict) A dictionary containing the configuration.
    :param stage: (str) One of the stage in `train` (default), `inference` or `test`. 

    .. note::

        In the configuration dictionary, in addition to the four sections (`dataset`,
        `network`, `training` and `inference`) used in fully supervised learning, an 
        extra section `semi_supervised_learning` is needed. See :doc:`usage.ssl` for details.
    """
    def __init__(self, config, stage = 'train'):
        super(SelfSLPatchMixAgent, self).__init__(config, stage)
 
    def training(self):
        class_num   = self.config['network']['class_num']
        iter_valid  = self.config['training']['iter_valid']
        fg_num    = self.config['network']['class_num'] - 1
        patch_num = self.config['patch_mix']['patch_num_range']
        size_d    = self.config['patch_mix']['patch_depth_range']
        size_h    = self.config['patch_mix']['patch_height_range']
        size_w    = self.config['patch_mix']['patch_width_range']

        train_loss  = 0
        train_dice_list = []
        self.net.train()
        for it in range(iter_valid):
            try:
                data = next(self.trainIter)
            except StopIteration:
                self.trainIter = iter(self.train_loader)
                data = next(self.trainIter)
            # get the inputs
            inputs  = self.convert_tensor_type(data['image'])  
            inputs, labels_prob = patch_mix(inputs, fg_num, patch_num, size_d, size_h, size_w)
                   
            # # for debug
            # if(it==10):
            #     break
            # for i in range(inputs.shape[0]):
            #     image_i = inputs[i][0]
            #     label_i = np.argmax(labels_prob[i], axis = 0)
            #     # pixw_i  = pix_w[i][0]
            #     print(image_i.shape, label_i.shape)
            #     image_name = "temp/image_{0:}_{1:}.nii.gz".format(it, i)
            #     label_name = "temp/label_{0:}_{1:}.nii.gz".format(it, i)
            #     # weight_name= "temp/weight_{0:}_{1:}.nii.gz".format(it, i)
            #     save_nd_array_as_image(image_i, image_name, reference_name = None)
            #     save_nd_array_as_image(label_i, label_name, reference_name = None)
            #     # save_nd_array_as_image(pixw_i, weight_name, reference_name = None)
            # continue

            inputs, labels_prob = inputs.to(self.device), labels_prob.to(self.device)
            
            # zero the parameter gradients
            self.optimizer.zero_grad()
                
            # forward + backward + optimize
            outputs = self.net(inputs)
            loss = self.get_loss_value(data, outputs, labels_prob)
            loss.backward()
            self.optimizer.step()
            train_loss = train_loss + loss.item()
            # get dice evaluation for each class
            if(isinstance(outputs, tuple) or isinstance(outputs, list)):
                outputs = outputs[0] 
            outputs_argmax = torch.argmax(outputs, dim = 1, keepdim = True)
            soft_out       = get_soft_label(outputs_argmax, class_num, self.tensor_type)
            soft_out, labels_prob = reshape_prediction_and_ground_truth(soft_out, labels_prob) 
            dice_list = get_classwise_dice(soft_out, labels_prob)
            train_dice_list.append(dice_list.cpu().numpy())
        train_avg_loss = train_loss / iter_valid
        train_cls_dice = np.asarray(train_dice_list).mean(axis = 0)
        train_avg_dice = train_cls_dice[1:].mean()

        train_scalers = {'loss': train_avg_loss, 'avg_fg_dice':train_avg_dice,\
            'class_dice': train_cls_dice}
        return train_scalers

def main():
    cfg_file = str(sys.argv[1])
    if(not os.path.isfile(cfg_file)):
        raise ValueError("The config file does not exist: " + cfg_file)
    config   = parse_config(cfg_file)
    config   = synchronize_config(config)
    log_dir  = config['training']['ckpt_save_dir']
    if(not os.path.exists(log_dir)):
        os.makedirs(log_dir, exist_ok=True)
    dst_cfg = cfg_file if "/" not in cfg_file else cfg_file.split("/")[-1]
    shutil.copy(cfg_file, log_dir + "/" + dst_cfg)
    if sys.version.startswith("3.9"):
        logging.basicConfig(filename=log_dir+"/log_train_{0:}.txt".format(str(datetime.now())[:-7]), 
                            level=logging.INFO, format='%(message)s', force=True) # for python 3.9
    else:
        logging.basicConfig(filename=log_dir+"/log_train_{0:}.txt".format(str(datetime.now())[:-7]), 
                            level=logging.INFO, format='%(message)s') # for python 3.6
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging_config(config)
    agent = SelfSLPatchMixAgent(config)
    agent.run()


if __name__ == "__main__":
    main()       