# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
from pymic.loss.seg.abstract import AbstractSegLoss

class MumfordShahLoss(AbstractSegLoss):
    """
    Implementation of Mumford Shah Loss for weakly supervised learning.

    * Reference: Boah Kim and Jong Chul Ye: Mumford–Shah Loss Functional 
      for Image Segmentation With Deep Learning. IEEE TIP, 2019.
    
    The oringial implementation is availabel at `Github. 
    <https://github.com/jongcye/CNN_MumfordShah_Loss>`_ 
    Currently only 2D version is supported.

    The parameters should be written in the `params` dictionary, and it has the
    following fields:

    :param `loss_softmax`: (bool) Apply softmax to the prediction of network or not. 
    :param `MumfordShahLoss_penalty`: (optional, str) `l1` or `l2`. Default is `l1`.
    :param `MumfordShahLoss_lambda`: (optional, float) Hyper-parameter lambda, default is 1.0.
    """
    def __init__(self, params = None):
        super(MumfordShahLoss, self).__init__()
        if(params is None):
            params = {}
        self.penalty = params.get('MumfordShahLoss_penalty', "l1")
        self.grad_w  = params.get('MumfordShahLoss_lambda', 1.0)

    def get_levelset_loss(self, output, target):
        """
        Get the level set loss value. 

        :param `output`: (tensor) softmax output of a network.
        :param `target`: (tensor) the input image. 
        :return: the level set loss.
        """
        outshape = output.shape
        tarshape = target.shape
        loss = 0.0
        for ich in range(tarshape[1]):
            target_ = torch.unsqueeze(target[:,ich], 1)
            target_ = target_.expand(tarshape[0], outshape[1], tarshape[2], tarshape[3])
            pcentroid = torch.sum(target_ * output, (2,3))/torch.sum(output, (2,3))
            pcentroid = pcentroid.view(tarshape[0], outshape[1], 1, 1)
            plevel = target_ - pcentroid.expand(tarshape[0], outshape[1], tarshape[2], tarshape[3])
            pLoss = plevel * plevel * output
            loss += torch.sum(pLoss)
        return loss

    def get_gradient_loss(self, pred, penalty = "l2"):
        dH = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        dW = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        if penalty == "l2":
            dH = dH * dH
            dW = dW * dW
        loss = torch.sum(dH) + torch.sum(dW)
        return loss

    def forward(self, loss_input_dict):
        """
        Forward pass for calculating the loss.
        The arguments should be written in the `loss_input_dict` dictionary, 
        and it has the following fields:

        :param `prediction`: (tensor) Prediction of a network, with the 
            shape of [N, C, D, H, W] or [N, C, H, W].
        :param `image`: (tensor) Image, with the 
            shape of [N, C, D, H, W] or [N, C, H, W]. 
        
        :return: Loss function value.
        """
        predict = loss_input_dict['prediction']
        image   = loss_input_dict['image']
        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(self.acti_func is not None):
            predict = self.get_activated_prediction(predict, self.acti_func)

        pred_shape  = list(predict.shape)
        if(len(pred_shape) == 5):
                [N, C, D, H, W] = pred_shape
                new_shape  = [N*D, C, H, W]
                predict = torch.transpose(predict, 1, 2)
                predict = torch.reshape(predict, new_shape)
                [N, C, D, H, W] = list(image.shape)
                new_shape    = [N*D, C, H, W]
                image = torch.transpose(image, 1, 2)
                image = torch.reshape(image, new_shape)
        loss0 = self.get_levelset_loss(predict, image)
        loss1 = self.get_gradient_loss(predict, self.penalty)
        loss = loss0 + self.grad_w * loss1
        return loss/torch.numel(predict)
