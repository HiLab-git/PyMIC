# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import numpy as np 
from scipy.ndimage.filters import gaussian_filter
from torch.nn.functional import interpolate

class Inferer(object):
    """
    The class for inference.
    The arguments should be written in the `config` dictionary, 
    and it has the following fields:

    :param `sliding_window_enable`: (optional, bool) Default is `False`.
    :param `sliding_window_size`: (optional, list) The sliding window size. 
    :param `sliding_window_stride`: (optional, list) The sliding window stride. 
    :param `tta_mode`: (optional, int) The test time augmentation mode. Default
        is 0 (no test time augmentation). The other option is 1 (augmentation 
        with horinzontal and vertical flipping) and 2 (ensemble of inference
        in axial, sagittal and coronal views for 2D networks applied to 3D volumes)
    """
    def __init__(self, config):
        self.config = config
        
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

    def __get_gaussian_weight_map(self, window_size, sigma_scale = 1.0/8):
        w = np.zeros(window_size)
        center = [i//2 for i in window_size]
        sigmas = [i*sigma_scale for i in window_size]
        w[tuple(center)] = 1.0
        w = gaussian_filter(w, sigmas, 0, mode='constant', cval=0)
        return w 

    def __infer_with_sliding_window(self, image):
        """
        Use sliding window to predict segmentation for large images. The outupt of each
        sliding window is weighted by a Gaussian map that hihglights contributions of windows
        with a centroid closer to a given pixel. 
        Note that the network may output a list of tensors with difference sizes for multi-scale prediction.
        """
        window_size   = [x for x in self.config['sliding_window_size']]
        window_stride = [x for x in self.config['sliding_window_stride']]
        window_batch  = self.config.get('sliding_window_batch', 1)
        class_num     = self.config['class_num']
        img_full_shape = list(image.shape)
        batch_size = img_full_shape[0]
        assert(batch_size == 1 or window_batch == 1)
        img_chns   = img_full_shape[1]
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
        weight       = torch.zeros(output_shape).to(image.device)
        temp_w       = self.__get_gaussian_weight_map(window_size)
        temp_w       = np.broadcast_to(temp_w, [batch_size, class_num] + window_size)
        temp_w       = torch.from_numpy(np.array(temp_w)).to(image.device)
        temp_in_shape = img_full_shape[:2] + window_size
        tempx = torch.ones(temp_in_shape).to(image.device)
        out_num, scale_list = self.__get_prediction_number_and_scales(tempx)

        window_num = len(crop_start_list)
        assert(window_num >= window_batch)
        patches_shape = [window_batch, img_chns] + window_size
        patches_in    =  torch.ones(patches_shape).to(image.device)
        if(out_num == 1): # for a single prediction
            output = torch.zeros(output_shape).to(image.device)
            for w_i in range(0, window_num, window_batch):
                for k in range(window_batch):
                    if(w_i + k >= window_num):
                        break
                    c0 = crop_start_list[w_i + k]
                    c1 = [c0[d] + window_size[d] for d in range(img_dim)]
                    if(img_dim == 2):
                        patches_in[k] = image[:, :, c0[0]:c1[0], c0[1]:c1[1]]
                    else:
                        patches_in[k] = image[:, :, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]]
                patches_out = self.model(patches_in) 
                if(isinstance(patches_out, (tuple, list))):
                    patches_out = patches_out[0]
                for k in range(window_batch):
                    if(w_i + k >= window_num):
                        break
                    c0 = crop_start_list[w_i + k]
                    c1 = [c0[d] + window_size[d] for d in range(img_dim)]
                    if(img_dim == 2):
                        output[:, :, c0[0]:c1[0], c0[1]:c1[1]] += patches_out[k] * temp_w
                        weight[:, :, c0[0]:c1[0], c0[1]:c1[1]] += temp_w
                    else:
                        output[:, :, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]] += patches_out[k] * temp_w
                        weight[:, :, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]] += temp_w
            return output/weight
        else: # for multiple prediction
            output_list= []
            for i in range(out_num):
                output_shape_i = [batch_size, class_num] + \
                    [int(img_shape[d] * scale_list[i][d]) for d in range(img_dim)]
                output_list.append(torch.zeros(output_shape_i).to(image.device))
            temp_ws = [interpolate(temp_w, scale_factor = scale_list[i]) for i in range(out_num)]
            weights = [interpolate(weight, scale_factor = scale_list[i]) for i in range(out_num)]
            for w_i in range(0, window_num, window_batch):
                for k in range(window_batch):
                    if(w_i + k >= window_num):
                        break
                    c0 = crop_start_list[w_i + k]
                    c1 = [c0[d] + window_size[d] for d in range(img_dim)]
                    if(img_dim == 2):
                        patches_in[k] = image[:, :, c0[0]:c1[0], c0[1]:c1[1]]
                    else:
                        patches_in[k] = image[:, :, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]]
                patches_out = self.model(patches_in) 

                for i in range(out_num):
                    for k in range(window_batch):
                        if(w_i + k >= window_num):
                            break
                        c0 = crop_start_list[w_i + k]
                        c0_i = [int(c0[d] * scale_list[i][d]) for d in range(img_dim)]
                        c1_i = [int(c1[d] * scale_list[i][d]) for d in range(img_dim)]
                        if(img_dim == 2):
                            output_list[i][:, :, c0_i[0]:c1_i[0], c0_i[1]:c1_i[1]] += patches_out[i][k] * temp_ws[i]
                            weights[i][:, :, c0_i[0]:c1_i[0], c0_i[1]:c1_i[1]] += temp_ws[i]
                        else:
                            output_list[i][:, :, c0_i[0]:c1_i[0], c0_i[1]:c1_i[1], c0_i[2]:c1_i[2]] += patches_out[i][k] * temp_ws[i]
                            weights[i][:, :, c0_i[0]:c1_i[0], c0_i[1]:c1_i[1], c0_i[2]:c1_i[2]] += temp_ws[i]
            for i in range(out_num):  
                output_list[i] = output_list[i] / weights[i]
            return output_list

    def run(self, model, image):
        """
        Using `model` for inference on `image`.

        :param model: (nn.Module) a network.
        :param image: (tensor) An image.
        """
        self.model = model
        tta_mode   = self.config.get('tta_mode', 0)
        if(tta_mode == 0):
            outputs = self.__infer(image)
        elif(tta_mode == 1): 
            # test time augmentation with flip in 2D
            # you may define your own method for test time augmentation
            outputs1 = self.__infer(image)
            outputs2 = self.__infer(torch.flip(image, [-2]))
            outputs3 = self.__infer(torch.flip(image, [-1]))
            outputs4 = self.__infer(torch.flip(image, [-2, -1]))
            if(isinstance(outputs1, (tuple, list))):
                outputs = []
                for i in range(len(outputs1)):
                    temp_out1 = outputs1[i]
                    temp_out2 = torch.flip(outputs2[i], [-2])
                    temp_out3 = torch.flip(outputs3[i], [-1])
                    temp_out4 = torch.flip(outputs4[i], [-2, -1])
                    temp_mean = (temp_out1 + temp_out2 + temp_out3 + temp_out4) / 4
                    outputs.append(temp_mean)
            else:
                outputs2 = torch.flip(outputs2, [-2])
                outputs3 = torch.flip(outputs3, [-1])
                outputs4 = torch.flip(outputs4, [-2, -1])
                outputs = (outputs1 + outputs2 + outputs3 + outputs4) / 4
        elif(tta_mode == 2):
            outputs1 = self.__infer(image)
            outputs2 = self.__infer(torch.transpose(image, -1, -3))
            outputs3 = self.__infer(torch.transpose(image, -2, -3))
            if(isinstance(outputs1, (tuple, list))):
                outputs = []
                for i in range(len(outputs1)):
                    temp_out1 = outputs1[i]
                    temp_out2 = torch.transpose(outputs2[i], -1, -3)
                    temp_out3 = torch.transpose(outputs3[i], -2, -3)
                    temp_mean = (temp_out1 + temp_out2 + temp_out3) / 3
                    outputs.append(temp_mean)
            else:
                outputs2 = torch.transpose(outputs2, -1, -3)
                outputs3 = torch.transpose(outputs3, -2, -3)
                outputs = (outputs1 + outputs2 + outputs3) / 3
        else:
            raise ValueError("Undefined tta_mode {0:}".format(tta_mode))
        return outputs

