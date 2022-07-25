# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import sys
import math
import torch
import numpy as np
from torch.nn.functional import interpolate

class Inferer(object):
    def __init__(self, model, config):
        self.model   = model 
        self.config  = config
        
    def __infer(self, image):
        use_sw  = self.config.get('sliding_window_enable', False)
        if(not use_sw):
            outputs = self.model(image)
        else:
            outputs = self.__infer_with_sliding_window(image)
        return outputs

    def __get_prediction_number_and_scales(self, tempx):
        """
        If the network outputs multiple tensors with different sizes, return the
        number of tensors and the scale of each tensor compared with the first one
        """
        img_dim = len(tempx.shape) - 2
        output = self.model(tempx)
        if(isinstance(output, (tuple, list))):
            output_num = len(output)
            scales = [[1.0] * img_dim]
            shape0 = list(output[0].shape[2:])
            for  i in range(1, output_num):
                shapei= list(output[i].shape[2:])
                scale = [(shapei[d] + 0.0) / shape0[d] for d in range(img_dim)]
                scales.append(scale)
        else:
            output_num, scales = 1, None
        return output_num, scales

    def __infer_with_sliding_window(self, image):
        """
        Use sliding window to predict segmentation for large images.
        Note that the network may output a list of tensors with difference sizes.
        """
        window_size   = [x for x in self.config['sliding_window_size']]
        window_stride = [x for x in self.config['sliding_window_stride']]
        class_num     = self.config['class_num']
        img_full_shape = list(image.shape)
        batch_size = img_full_shape[0]
        img_shape  = img_full_shape[2:]
        img_dim    = len(img_shape)
        if(img_dim != 2 and img_dim !=3):
            raise ValueError("Inference using sliding window only supports 2D and 3D images")

        for d in range(img_dim):
            if (window_size[d] is None) or window_size[d] > img_shape[d]:
                window_size[d]  = img_shape[d]
            if (window_stride[d] is None) or window_stride[d] > window_size[d]:
                window_stride[d] = window_size[d]
                
        if all([window_size[d] >= img_shape[d] for d in range(img_dim)]):
            output = self.model(image)
            return output

        crop_start_list  = []
        for w in range(0, img_shape[-1], window_stride[-1]):
            w_min = min(w, img_shape[-1] - window_size[-1])
            for h in range(0, img_shape[-2], window_stride[-2]):
                h_min = min(h, img_shape[-2] - window_size[-2])
                if(img_dim == 2):
                    crop_start_list.append([h_min, w_min])
                else:
                    for d in range(0, img_shape[0], window_stride[0]):
                        d_min = min(d, img_shape[0] - window_size[0])
                        crop_start_list.append([d_min, h_min, w_min])

        output_shape = [batch_size, class_num] + img_shape
        mask_shape   = [batch_size, class_num] + window_size
        counter      = torch.zeros(output_shape).to(image.device)
        temp_mask    = torch.ones(mask_shape).to(image.device)
        temp_in_shape = img_full_shape[:2] + window_size
        tempx = torch.ones(temp_in_shape).to(image.device)
        out_num, scale_list = self.__get_prediction_number_and_scales(tempx)
        if(out_num == 1): # for a single prediction
            output = torch.zeros(output_shape).to(image.device)
            for c0 in crop_start_list:
                c1 = [c0[d] + window_size[d] for d in range(img_dim)]
                if(img_dim == 2):
                    patch_in = image[:, :, c0[0]:c1[0], c0[1]:c1[1]]
                else:
                    patch_in = image[:, :, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]]
                patch_out = self.model(patch_in) 
                if(isinstance(patch_out, (tuple, list))):
                    patch_out = patch_out[0]
                if(img_dim == 2):
                    output[:, :, c0[0]:c1[0], c0[1]:c1[1]] += patch_out
                    counter[:, :, c0[0]:c1[0], c0[1]:c1[1]] += temp_mask
                else:
                    output[:, :, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]] += patch_out
                    counter[:, :, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]] += temp_mask
            return output/counter
        else: # for multiple prediction
            output_list= []
            for i in range(out_num):
                output_shape_i = [batch_size, class_num] + \
                    [int(img_shape[d] * scale_list[i][d]) for d in range(img_dim)]
                output_list.append(torch.zeros(output_shape_i).to(image.device))

            for c0 in crop_start_list:
                c1 = [c0[d] + window_size[d] for d in range(img_dim)]
                if(img_dim == 2):
                    patch_in = image[:, :, c0[0]:c1[0], c0[1]:c1[1]]
                else:
                    patch_in = image[:, :, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]]
                patch_out = self.model(patch_in) 

                for i in range(out_num):
                    c0_i = [int(c0[d] * scale_list[i][d]) for d in range(img_dim)]
                    c1_i = [int(c1[d] * scale_list[i][d]) for d in range(img_dim)]
                    if(img_dim == 2):
                        output_list[i][:, :, c0_i[0]:c1_i[0], c0_i[1]:c1_i[1]] += patch_out[i]
                        counter[:, :, c0[0]:c1[0], c0[1]:c1[1]] += temp_mask
                    else:
                        output_list[i][:, :, c0_i[0]:c1_i[0], c0_i[1]:c1_i[1], c0_i[2]:c1_i[2]] += patch_out[i]
                        counter[:, :, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]] += temp_mask
            for i in range(out_num):  
                counter_i = interpolate(counter, scale_factor = scale_list[i])
                output_list[i] = output_list[i] / counter_i
            return output_list

    def run(self, image):
        tta_mode  = self.config.get('tta_mode', 0)
        if(tta_mode == 0):
            outputs = self.__infer(image)
        elif(tta_mode == 1): # test time augmentation with flip in 2D
            outputs1 = self.__infer(image)
            outputs2 = self.__infer(torch.flip(image, [-2]))
            outputs3 = self.__infer(torch.flip(image, [-3]))
            outputs4 = self.__infer(torch.flip(image, [-2, -3]))
            if(isinstance(outputs1, (tuple, list))):
                outputs = []
                for i in range(len(outputs)):
                    temp_out1 = outputs1[i]
                    temp_out2 = torch.flip(outputs2[i], [-2])
                    temp_out3 = torch.flip(outputs3[i], [-3])
                    temp_out4 = torch.flip(outputs4[i], [-2, -3])
                    temp_mean = (temp_out1 + temp_out2 + temp_out3 + temp_out4) / 4
                    outputs.append(temp_mean)
            else:
                outputs2 = torch.flip(outputs2, [-2])
                outputs3 = torch.flip(outputs3, [-3])
                outputs4 = torch.flip(outputs4, [-2, -3])
                outputs = (outputs1 + outputs2 + outputs3 + outputs4) / 4
        else:
            raise ValueError("Undefined tta_mode {0:}".format(tta_mode))
        return outputs

