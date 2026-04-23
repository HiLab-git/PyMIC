# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from pymic.loss.seg.dice import DiceLoss
from pymic.loss.seg.ce   import CrossEntropyLoss
from pymic.net_run.weak_sup import WSLSegAgent
from pymic.util.ramps import get_rampup_ratio


class WSLPSSEG(WSLSegAgent):
    """
    PS-Seg: Learning from Partial Scribbles for 3D Multiple Abdominal Organ Segmentation

    * Reference: Meng Han, Xiaochuan Ma, Xiangde Luo, Wenjun Liao, Shichuan Zhang, Shaoting Zhang, Guotai Wang.
      PS-seg: Learning from partial scribbles for 3D multiple abdominal organ segmentation.
      `Neurocomputing 2026. <https://doi.org/10.1016/j.neucom.2026.132837>`_ 

    :param config: (dict) A dictionary containing the configuration.
    :param stage: (str) One of the stage in `train` (default), `inference` or `test`.

    .. note::

        In the configuration dictionary, in addition to the four sections (`dataset`,
        `network`, `training` and `inference`) used in fully supervised learning, an
        extra section `weakly_supervised_learning` is needed. See :doc:`usage.wsl` for details.
    """

    def __init__(self, config, stage="train"):
        super(WSLPSSEG, self).__init__(config, stage)

    def create_network(self):
        if(self.net is None):
            net_name = self.config['network']['net_type']
            if net_name not in ["TDNet3D", "TDNet2D"]:
                logging.warn("By defualt, the TDNet3D or  TDNet2D are used by PSSEG." + 
                    "using your customized network {0:} should be careful.".format(net_name))
            if(net_name not in self.net_dict):
                    raise ValueError("Undefined network {0:}".format(net_name))
            self.net = self.net_dict[net_name](self.config['network'])
        super(WSLPSSEG, self).create_network() 

    def weight_with_uncertainty(self, p):
        class_num = int(p.shape[1])
        unc = -torch.sum(p * torch.log(p), dim=1, keepdim=True)
        unc =  unc / torch.tensor(np.log(class_num))
        y1 = torch.exp(-unc)
        return y1


    def calculate_uspc_loss_by_KL(self, outputs_list):
        probs = [F.softmax(logits, dim=1) for i, logits in enumerate(outputs_list)]
        probs = [5e-4 + p * (1 - 1e-3) for p in probs]
        mixture_label = (torch.stack(probs)).mean(axis=0)
        weight = self.weight_with_uncertainty(mixture_label)
        kl_values = []
        for prob in probs:
            kl_i = prob * (torch.log(prob) - torch.log(mixture_label))
            kl_i = torch.sum(kl_i, dim = 1, keepdim = True)
            kl_i = torch.sum(weight * kl_i) / torch.sum(weight)
            kl_values.append(kl_i)
        consistency = sum(kl_values) / len(probs)
        loss_KL = torch.mean(consistency)

        return loss_KL

    def calculate_ccac_loss_by_MSE(self, outputsSoft_list):
        batch_size  = outputsSoft_list[0].shape[0]
        num_classes = outputsSoft_list[0].shape[1]
        outputsSoft_list_r = [
            item.reshape(batch_size, num_classes, -1) for item in outputsSoft_list
        ]
        outputsSoft_list_rNorm = [
            F.normalize(item, p=2, dim=2) for item in outputsSoft_list_r
        ]
        outputsSoft_list_rNormT = [
            item.permute(0, 2, 1) for item in outputsSoft_list_rNorm
        ]

        affinity_list = []
        for i in range(len(outputsSoft_list)):
            outputsSoft_list_rNormT_other = (
                outputsSoft_list_rNormT[:i] + outputsSoft_list_rNormT[i + 1 :]
            )
            for j in range(len(outputsSoft_list_rNormT_other)):
                affinity_ori = torch.bmm(
                    outputsSoft_list_rNorm[i], outputsSoft_list_rNormT_other[j]
                )
                affinity_list.append(affinity_ori)

        maskI = (
            torch.eye(affinity_list[0].shape[1])
            .repeat(affinity_list[1].shape[0], 1, 1)
            .cuda()
        )

        consistency_loss = 0
        for i in range(len(affinity_list)):
            consistency_loss = consistency_loss + torch.mean(
                (affinity_list[i] - maskI) ** 2
            )
        consistency_loss = consistency_loss / len(affinity_list)
        return consistency_loss

    def training(self):
        class_num = self.config["network"]["class_num"]
        iter_valid = self.config["training"]["iter_valid"]
        wsl_cfg     = self.config['weakly_supervised_learning']
        rampup_start = wsl_cfg.get('rampup_start', 0)
        rampup_end   = wsl_cfg.get('rampup_end', 2000)

        train_loss, train_loss_sup, train_loss_uspc, train_loss_ccac = 0, 0, 0, 0
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
            inputs = self.convert_tensor_type(data["image"])
            y      = self.convert_tensor_type(data["label_prob"])

            inputs, y = inputs.to(self.device), y.to(self.device)
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs_list, embeddings1_list, embeddings2_list, embeddings3_list = self.net(inputs)
            t2 = time.time()

            # supervised
            loss_sup_list = [
                self.get_loss_value(data, output, y) for output in outputs_list
            ]
            loss_sup = sum(loss_sup_list) / len(loss_sup_list)

            # unsupervised loss
            rampup_ratio = get_rampup_ratio(self.glob_it, rampup_start, rampup_end, "sigmoid")
            w_uspc = wsl_cfg.get('uspc_weight', 8.0) * rampup_ratio
            w_ccac = wsl_cfg.get('ccac_weight', 0.1) * rampup_ratio

            loss_uspc = self.calculate_uspc_loss_by_KL(outputs_list)
            loss_aff0 = self.calculate_ccac_loss_by_MSE(outputs_list)
            loss_aff1 = self.calculate_ccac_loss_by_MSE(embeddings1_list)
            loss_aff2 = self.calculate_ccac_loss_by_MSE(embeddings2_list)
            loss_aff3 = self.calculate_ccac_loss_by_MSE(embeddings3_list)
            loss_ccac = (loss_aff0 + loss_aff1 + loss_aff2 + loss_aff3) / 4.0

            loss = loss_sup + w_uspc * loss_uspc  + w_ccac * loss_ccac
            
            t3 = time.time()
            loss.backward()
            t4 = time.time()
            self.optimizer.step()

            train_loss = train_loss + loss.item()
            train_loss_sup  = train_loss_sup + loss_sup.item()
            train_loss_uspc = train_loss_uspc + loss_uspc.item()
            train_loss_ccac = train_loss_ccac + loss_ccac.item() 

            data_time = data_time + t1 - t0
            gpu_time = gpu_time + t2 - t1
            loss_time = loss_time + t3 - t2
            back_time = back_time + t4 - t3
        train_avg_loss = train_loss / iter_valid
        train_avg_loss_sup  = train_loss_sup / iter_valid
        train_avg_loss_uspc = train_loss_uspc / iter_valid
        train_avg_loss_ccac = train_loss_ccac / iter_valid

        train_scalers = {
            "loss": train_avg_loss, 'loss_sup':train_avg_loss_sup,
            "loss_uspc" :train_avg_loss_uspc, "loss_ccac" :train_avg_loss_ccac, 
            "weight_uspc": w_uspc, "weight_ccac": w_ccac,
            "data_time": data_time, "forward_time": gpu_time,
            "loss_time": loss_time, "backward_time": back_time,
        }
        return train_scalers

    def write_scalars(self, train_scalars, valid_scalars, lr_value, glob_it):
        loss_uspc_scalar  = {'train':train_scalars['loss_uspc']}
        loss_ccac_scalar  = {'train':train_scalars['loss_ccac']}
        weight_scalar = {'uspc':train_scalars['weight_uspc'],
                         'ccac':train_scalars['weight_ccac']}
        self.summ_writer.add_scalars('loss_uspc', loss_uspc_scalar, glob_it)
        self.summ_writer.add_scalars('loss_ccac', loss_ccac_scalar, glob_it)
        self.summ_writer.add_scalars('weight', weight_scalar, glob_it)
        super(WSLPSSEG, self).write_scalars(train_scalars, valid_scalars, lr_value, glob_it)
