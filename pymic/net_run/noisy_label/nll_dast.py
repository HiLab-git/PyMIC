# -*- coding: utf-8 -*-
from __future__ import print_function, division
import random
import torch
import numpy as np 
import torch.nn as nn
import torchvision.transforms as transforms
from pymic.io.nifty_dataset import NiftyDataset
from pymic.loss.seg.util import get_soft_label
from pymic.loss.seg.util import reshape_prediction_and_ground_truth
from pymic.loss.seg.util import get_classwise_dice
from pymic.net_run.agent_seg import SegmentationAgent
from pymic.util.parse_config import *
from pymic.util.ramps import get_rampup_ratio

class Rank(object):
    """
    Dynamically rank the current training sample with specific metrics.

    :param  quene_length: (int) The lenght for a quene.
    """
    def __init__(self, quene_length = 100):
        self.vals = []
        self.quene_length = quene_length

    def add_val(self, val):
        """
        Update the quene and calculate the order of the input value.

        :param val: (float) a value adding to the quene.
        :return: rank of the input value with a range of  (0, self.quene_length)
        """
        if len(self.vals) < self.quene_length:
            self.vals.append(val)
            rank = -1
        else:
            self.vals.pop(0)
            self.vals.append(val)
            assert len(self.vals) == self.quene_length
            idxes = np.argsort(self.vals)
            rank = np.where(idxes == self.quene_length-1)[0][0]
        return rank

class ConsistLoss(nn.Module):
    def __init__(self):
        super(ConsistLoss, self).__init__()

    def kl_div_map(self, input, label):
        kl_map = torch.sum(label * (torch.log(label + 1e-16) - torch.log(input + 1e-16)), dim = 1)
        return kl_map

    def kl_loss(self,input, target, size_average=True):
        kl_div = self.kl_div_map(input, target)
        if size_average:
            return torch.mean(kl_div)
        else:
            return kl_div

    def forward(self, input1, input2, size_average = True):
        kl1 = self.kl_loss(input1, input2.detach(), size_average=size_average)
        kl2 = self.kl_loss(input2, input1.detach(), size_average=size_average)
        return (kl1 + kl2) / 2

def get_ce(prob, soft_y, size_avg = True):
    prob = prob * 0.999 + 5e-4
    ce = - soft_y* torch.log(prob)
    ce = torch.sum(ce, dim = 1) # shape is [N]
    if(size_avg):
        ce = torch.mean(ce)
    return ce

@torch.no_grad()
def select_criterion(no_noisy_sample, cl_noisy_sample, label):
    """
    Obtain the sample selection criterion score.

    :param no_noisy_sample: noisy branch's output probability for noisy sample.
    :param cl_noisy_sample: clean branch's output probability for noisy sample.
    :param label: noisy label.
    """
    l_n = get_ce(no_noisy_sample, label, size_avg = False)
    l_c = get_ce(cl_noisy_sample, label, size_avg = False)
    js_distance = ConsistLoss()
    variance = js_distance(no_noisy_sample, cl_noisy_sample, size_average=False)
    exp_variance = torch.exp(-16 * variance)
    loss_n = torch.mean(l_c * exp_variance).item()
    loss_c = torch.mean(l_n * exp_variance).item()
    return loss_n, loss_c

class NLLDAST(SegmentationAgent):
    """
    Divergence-Aware Selective Training for noisy label learning.

    * Reference: Shuojue Yang, Guotai Wang, Hui Sun, Xiangde Luo, Peng Sun, 
      Kang Li, Qijun Wang, Shaoting Zhang: Learning COVID-19 Pneumonia Lesion 
      Segmentation from Imperfect  Annotations via Divergence-Aware Selective Training.
      `JBHI 2022. <https://ieeexplore.ieee.org/document/9770406>`_    
    
    :param config: (dict) A dictionary containing the configuration.
    :param stage: (str) One of the stage in `train` (default), `inference` or `test`. 

    .. note::

        In the configuration dictionary, in addition to the four sections (`dataset`,
        `network`, `training` and `inference`) used in fully supervised learning, an 
        extra section `noisy_label_learning` is needed. See :doc:`usage.nll` for details.
    """
    def __init__(self, config, stage = 'train'):
        super(NLLDAST, self).__init__(config, stage)
        self.train_set_noise = None 
        self.train_loader_noise = None 
        self.trainIter_noise    = None
        self.noisy_rank = None 
        self.clean_rank = None

    def get_noisy_dataset_from_config(self):
        """
        Create a dataset for images with noisy labels based on configuraiton.
        """
        trans_names, trans_params = self.get_transform_names_and_parameters('train')
        transform_list  = []
        if(trans_names is not None and len(trans_names) > 0):
            for name in trans_names:
                if(name not in self.transform_dict):
                    raise(ValueError("Undefined transform {0:}".format(name))) 
                one_transform = self.transform_dict[name](trans_params)
                transform_list.append(one_transform)
        data_transform = transforms.Compose(transform_list)

        modal_num = self.config['dataset'].get('modal_num', 1)
        csv_file = self.config['dataset'].get('train_csv_noise', None)
        dataset  = NiftyDataset(root_dir  = self.config['dataset']['train_dir'],
                                csv_file  = csv_file,
                                modal_num = modal_num,
                                with_label= True,
                                transform = data_transform , 
                                task = self.task_type)
        return dataset


    def create_dataset(self):
        super(NLLDAST, self).create_dataset()
        if(self.stage == 'train'):
            if(self.train_set_noise is None):
                self.train_set_noise = self.get_noisy_dataset_from_config()
            if(self.deterministic):
                def worker_init_fn(worker_id):
                    random.seed(self.random_seed + worker_id)
                worker_init = worker_init_fn
            else:
                worker_init = None

            bn_train_noise = self.config['dataset']['train_batch_size_noise']
            num_worker = self.config['dataset'].get('num_worker', 16)
            self.train_loader_noise = torch.utils.data.DataLoader(self.train_set_noise, 
                batch_size = bn_train_noise, shuffle=True, num_workers= num_worker,
                worker_init_fn=worker_init)

    def training(self):
        class_num   = self.config['network']['class_num']
        iter_valid  = self.config['training']['iter_valid']
        nll_cfg     = self.config['noisy_label_learning']
        iter_max     = self.config['training']['iter_max']
        rampup_start = nll_cfg.get('rampup_start', 0)
        rampup_end   = nll_cfg.get('rampup_end', iter_max)
        train_loss   = 0
        train_loss_sup = 0
        train_loss_reg = 0
        train_dice_list = []
        self.net.train()

        rank_length = nll_cfg.get("dast_rank_length", 20)
        consist_loss = ConsistLoss()
        for it in range(iter_valid):
            try:
                data_cl = next(self.trainIter)
            except StopIteration:
                self.trainIter = iter(self.train_loader)
                data_cl = next(self.trainIter)
            try:
                data_no = next(self.trainIter_noise)
            except StopIteration:
                self.trainIter_noise = iter(self.train_loader_noise)
                data_no = next(self.trainIter_noise)

            # get the inputs
            x0 = self.convert_tensor_type(data_cl['image'])  # clean sample
            y0 = self.convert_tensor_type(data_cl['label_prob'])  
            x1 = self.convert_tensor_type(data_no['image'])  # noisy sample
            y1 = self.convert_tensor_type(data_no['label_prob']) 
            inputs = torch.cat([x0, x1], dim = 0).to(self.device)               
            y0, y1 = y0.to(self.device), y1.to(self.device)
            
            # zero the parameter gradients
            self.optimizer.zero_grad()
                
            # forward + backward + optimize
            b0_pred, b1_pred = self.net(inputs) 
            n0 = list(x0.shape)[0]  # number of clean samples
            b0_x0_pred = b0_pred[:n0] # predication of clean samples from clean branch
            b0_x1_pred = b0_pred[n0:] # predication of noisy samples from clean branch
            b1_x1_pred = b1_pred[n0:] # predication of noisy samples from noisy branch

            # supervised loss for the clean and noisy branches, respectively
            loss_sup_cl = self.get_loss_value(data_cl, b0_x0_pred, y0)
            loss_sup_no = self.get_loss_value(data_no, b1_x1_pred, y1)
            loss_sup = (loss_sup_cl + loss_sup_no) / 2
            loss = loss_sup

            # Severe Noise supression & Supplementary Training
            rampup_ratio = get_rampup_ratio(self.glob_it, rampup_start, rampup_end, "sigmoid")
            w_dbc = nll_cfg.get('dast_dbc_w', 0.1) * rampup_ratio
            w_st  = nll_cfg.get('dast_st_w',  0.1) * rampup_ratio
            b1_x1_prob = nn.Softmax(dim = 1)(b1_x1_pred)
            b0_x1_prob = nn.Softmax(dim = 1)(b0_x1_pred)
            loss_n, loss_c = select_criterion(b1_x1_prob, b0_x1_prob, y1)
            rank_n = self.noisy_rank.add_val(loss_n)
            rank_c = self.clean_rank.add_val(loss_c)
            if loss_n < loss_c:
                select_ratio = nll_cfg.get('dast_select_ratio', 0.2)
                if rank_c >= rank_length * (1 - select_ratio):
                    loss_dbc = consist_loss(b1_x1_prob, b0_x1_prob)
                    loss = loss + loss_dbc * w_dbc
                if rank_n <= rank_length * select_ratio:
                    b0_x1_argmax = torch.argmax(b0_x1_pred, dim = 1, keepdim = True)
                    b0_x1_lab    = get_soft_label(b0_x1_argmax, class_num, self.tensor_type)
                    b1_x1_argmax = torch.argmax(b1_x1_pred, dim = 1, keepdim = True)
                    b1_x1_lab    = get_soft_label(b1_x1_argmax, class_num, self.tensor_type)
                    pseudo_label = (b0_x1_lab + b1_x1_lab + y1) / 3
                    sharpen = lambda p,T: p**(1.0/T)/(p**(1.0/T) + (1-p)**(1.0/T))
                    b0_x1_prob   = nn.Softmax(dim = 1)(b0_x1_pred)
                    loss_st  = torch.mean(torch.abs(b0_x1_prob - sharpen(pseudo_label, 0.5)))
                    loss = loss + loss_st * w_st

            loss.backward()
            self.optimizer.step()

            train_loss = train_loss + loss.item()
            train_loss_sup = train_loss_sup + loss_sup.item()
            # train_loss_reg = train_loss_reg + loss_reg.item() 
            # get dice evaluation for each class in annotated images
            if(isinstance(b0_x0_pred, tuple) or isinstance(b0_x0_pred, list)):
                p0 = b0_x0_pred[0] 
            else:
                p0 = b0_x0_pred
            p0_argmax = torch.argmax(p0, dim = 1, keepdim = True)
            p0_soft   = get_soft_label(p0_argmax, class_num, self.tensor_type)
            p0_soft, y0 = reshape_prediction_and_ground_truth(p0_soft, y0) 
            dice_list   = get_classwise_dice(p0_soft, y0)
            train_dice_list.append(dice_list.cpu().numpy())
        train_avg_loss = train_loss / iter_valid
        train_avg_loss_sup = train_loss_sup / iter_valid
        train_avg_loss_reg = train_loss_reg / iter_valid
        train_cls_dice = np.asarray(train_dice_list).mean(axis = 0)
        train_avg_dice = train_cls_dice[1:].mean()

        train_scalers = {'loss': train_avg_loss, 'loss_sup':train_avg_loss_sup,
            'loss_reg':train_avg_loss_reg, 'regular_w':w_dbc,
            'avg_fg_dice':train_avg_dice,     'class_dice': train_cls_dice}
        return train_scalers

    def train_valid(self):
        self.trainIter_noise = iter(self.train_loader_noise)   
        nll_cfg     = self.config['noisy_label_learning']
        rank_length = nll_cfg.get("dast_rank_length", 20)
        self.noisy_rank = Rank(rank_length)
        self.clean_rank = Rank(rank_length)
        super(NLLDAST, self).train_valid()    