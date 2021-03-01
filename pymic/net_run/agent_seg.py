# -*- coding: utf-8 -*-
from __future__ import print_function, division

import copy
import os
import sys
import time
import random
import scipy
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from scipy import special
from datetime import datetime
from tensorboardX import SummaryWriter
from pymic.io.image_read_write import save_nd_array_as_image
from pymic.io.nifty_dataset import NiftyDataset
from pymic.transform.trans_dict import TransformDict
from pymic.net.net_dict_seg import SegNetDict
from pymic.net_run.agent_abstract import NetRunAgent
from pymic.net_run.infer_func import volume_infer
from pymic.loss.loss_dict_seg import SegLossDict
from pymic.loss.seg.util import get_soft_label
from pymic.loss.seg.util import reshape_prediction_and_ground_truth
from pymic.loss.seg.util import get_classwise_dice
from pymic.util.image_process import convert_label
from pymic.util.parse_config import parse_config

class SegmentationAgent(NetRunAgent):
    def __init__(self, config, stage = 'train'):
        super(SegmentationAgent, self).__init__(config, stage)
        self.transform_dict  = TransformDict
        
    def get_stage_dataset_from_config(self, stage):
        assert(stage in ['train', 'valid', 'test'])
        root_dir  = self.config['dataset']['root_dir']
        modal_num = self.config['dataset']['modal_num']

        transform_key = stage +  '_transform'
        if(stage == "valid" and transform_key not in self.config['dataset']):
            transform_key = "train_transform"
        transform_names = self.config['dataset'][transform_key]
        
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

        csv_file = self.config['dataset'].get(stage + '_csv', None)
        dataset  = NiftyDataset(root_dir=root_dir,
                                csv_file  = csv_file,
                                modal_num = modal_num,
                                with_label= not (stage == 'test'),
                                transform = data_transform )
        return dataset

    def create_network(self):
        if(self.net is None):
            net_name = self.config['network']['net_type']
            if(net_name not in SegNetDict):
                raise ValueError("Undefined network {0:}".format(net_name))
            self.net = SegNetDict[net_name](self.config['network'])
        if(self.tensor_type == 'float'):
            self.net.float()
        else:
            self.net.double()
        param_number = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print('parameter number:', param_number)

    def get_parameters_to_update(self):
        return self.net.parameters()

    def get_class_level_weight(self):
        class_num   = self.config['network']['class_num']
        class_weight= self.config['training'].get('loss_class_weight', None)
        if(class_weight is None):
            class_weight = torch.ones(class_num)
        else:
            assert(len(class_weight) == class_num)
            class_weight = torch.from_numpy(np.asarray(class_weight))
        class_weight = self.convert_tensor_type(class_weight)
        return class_weight

    def get_image_level_weight(self, data):
        imageweight_enb = self.config['training'].get('loss_with_image_weight', False)
        img_w = None 
        if(imageweight_enb):
            if('image_weight' not in data):
                raise ValueError("image weight is enabled not not provided")
            img_w = self.convert_tensor_type(data['image_weight'])
        else:
            batch_size = data['image'].shape[0]
            img_w = self.convert_tensor_type(torch.ones(batch_size))
        return img_w 

    def get_pixel_level_weight(self, data):
        pixelweight_enb = self.config['training'].get('loss_with_pixel_weight', False)
        pix_w = None
        if(pixelweight_enb):
            if(self.net.training):
                if('pixel_weight' not in data):
                    raise ValueError("pixel weight is enabled but not provided")
                pix_w = data['pixel_weight']
            else:
                pix_w = data.get('pixel_weight', None)
        if(pix_w is None):
            pix_w_shape = list(data['label_prob'].shape)
            pix_w_shape[1] = 1
            pix_w = torch.ones(pix_w_shape)
        pix_w = self.convert_tensor_type(pix_w)
        return pix_w
        
    def get_loss_value(self, data, inputs, outputs, labels_prob):
        cls_w = self.get_class_level_weight()
        img_w = self.get_image_level_weight(data)
        pix_w = self.get_pixel_level_weight(data)
        if(self.net.training):
            img_w, pix_w = img_w.to(self.device), pix_w.to(self.device)
            cls_w = cls_w.to(self.device)
        loss_input_dict = {'image':inputs, 'prediction':outputs, 'ground_truth':labels_prob,
                'image_weight': img_w, 'pixel_weight': pix_w, 'class_weight': cls_w, 
                'softmax': True}
        loss_value = self.loss_calculater(loss_input_dict)
        return loss_value
    
    def training(self):
        class_num   = self.config['network']['class_num']
        iter_valid  = self.config['training']['iter_valid']
        train_loss = 0
        train_dice_list = []
        self.net.train()
        for it in range(iter_valid):
            try:
                data = next(self.trainIter)
            except StopIteration:
                self.trainIter = iter(self.train_loader)
                data = next(self.trainIter)
            # get the inputs
            inputs      = self.convert_tensor_type(data['image'])
            labels_prob = self.convert_tensor_type(data['label_prob'])                 
            
            # # for debug
            # for i in range(inputs.shape[0]):
            #     image_i = inputs[i][0]
            #     label_i = labels_prob[i][1]
            #     pixw_i  = pix_w[i][0]
            #     print(image_i.shape, label_i.shape, pixw_i.shape)
            #     image_name = "temp/image_{0:}_{1:}.nii.gz".format(it, i)
            #     label_name = "temp/label_{0:}_{1:}.nii.gz".format(it, i)
            #     weight_name= "temp/weight_{0:}_{1:}.nii.gz".format(it, i)
            #     save_nd_array_as_image(image_i, image_name, reference_name = None)
            #     save_nd_array_as_image(label_i, label_name, reference_name = None)
            #     save_nd_array_as_image(pixw_i, weight_name, reference_name = None)
            # continue

            inputs, labels_prob = inputs.to(self.device), labels_prob.to(self.device)
            
            # zero the parameter gradients
            self.optimizer.zero_grad()
                
            # forward + backward + optimize
            outputs = self.net(inputs)
            loss = self.get_loss_value(data, inputs, outputs, labels_prob)
            # if (self.config['training']['use'])
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

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
        train_avg_dice = train_cls_dice.mean()

        train_scalers = {'loss': train_avg_loss, 'avg_dice':train_avg_dice,\
            'class_dice': train_cls_dice}
        return train_scalers
        
    def validation(self):
        class_num   = self.config['network']['class_num']
        mini_batch_size    = self.config['testing']['mini_batch_size']
        mini_patch_inshape = self.config['testing']['mini_patch_input_shape']
        mini_patch_outshape= self.config['testing']['mini_patch_output_shape']
        mini_patch_stride  = self.config['testing']['mini_patch_stride']
        output_num         = self.config['testing'].get('output_num', 1)
        valid_loss = 0.0
        valid_dice_list = []
        validIter  = iter(self.valid_loader)
        with torch.no_grad():
            self.net.eval()
            for data in validIter:
                inputs      = self.convert_tensor_type(data['image'])
                labels_prob = self.convert_tensor_type(data['label_prob'])

                outputs = volume_infer(inputs, self.net, self.device, class_num, 
                    mini_batch_size, mini_patch_inshape, mini_patch_outshape, mini_patch_stride, output_num)
                outputs = self.convert_tensor_type(torch.from_numpy(outputs))
                # The tensors are on CPU when calculating loss for validation data
                loss = self.get_loss_value(data, inputs, outputs, labels_prob)
                valid_loss = valid_loss + loss.item()

                if(isinstance(outputs, tuple) or isinstance(outputs, list)):
                    outputs = outputs[0] 
                outputs_argmax = torch.argmax(outputs, dim = 1, keepdim = True)
                soft_out  = get_soft_label(outputs_argmax, class_num, self.tensor_type)
                soft_out, labels_prob = reshape_prediction_and_ground_truth(soft_out, labels_prob) 
                dice_list = get_classwise_dice(soft_out, labels_prob)
                valid_dice_list.append(dice_list.cpu().numpy())

        valid_avg_loss = valid_loss / len(validIter)
        valid_cls_dice = np.asarray(valid_dice_list).mean(axis = 0)
        valid_avg_dice = valid_cls_dice.mean()
        
        valid_scalers = {'loss': valid_avg_loss, 'avg_dice': valid_avg_dice,\
            'class_dice': valid_cls_dice}
        return valid_scalers

    def write_scalars(self, train_scalars, valid_scalars, glob_it):
        loss_scalar ={'train':train_scalars['loss'], 'valid':valid_scalars['loss']}
        dice_scalar ={'train':train_scalars['avg_dice'], 'valid':valid_scalars['avg_dice']}
        self.summ_writer.add_scalars('loss', loss_scalar, glob_it)
        self.summ_writer.add_scalars('dice', dice_scalar, glob_it)
        class_num = self.config['network']['class_num']
        for c in range(class_num):
            cls_dice_scalar = {'train':train_scalars['class_dice'][c], \
                'valid':valid_scalars['class_dice'][c]}
            self.summ_writer.add_scalars('class_{0:}_dice'.format(c), cls_dice_scalar, glob_it)
       
        print("{0:} it {1:}".format(str(datetime.now())[:-7], glob_it))
        print('train loss {0:.4f}, avg dice {1:.4f}'.format(
            train_scalars['loss'], train_scalars['avg_dice']), train_scalars['class_dice'])        
        print('valid loss {0:.4f}, avg dice {1:.4f}'.format(
            valid_scalars['loss'], valid_scalars['avg_dice']), valid_scalars['class_dice'])  

    def train_valid(self):
        self.device = torch.device(self.config['training']['device_name'])
        self.net.to(self.device)
        chpt_prefx  = self.config['training']['checkpoint_prefix']
        iter_start  = self.config['training']['iter_start']
        iter_max    = self.config['training']['iter_max']
        iter_valid  = self.config['training']['iter_valid']
        iter_save   = self.config['training']['iter_save']

        self.max_val_dice = 0.0
        self.max_val_it   = 0
        self.best_model_wts = None 
        self.checkpoint = None
        if(iter_start > 0):
            checkpoint_file = "{0:}_{1:}.pt".format(chpt_prefx, iter_start)
            self.checkpoint = torch.load(checkpoint_file, map_location = self.device)
            assert(self.checkpoint['iteration'] == iter_start)
            self.net.load_state_dict(self.checkpoint['model_state_dict'])
            self.max_val_dice = self.checkpoint.get('valid_pred', 0)
            self.max_val_it   = self.checkpoint['iteration']
            self.best_model_wts = self.checkpoint['model_state_dict']
        
        params = self.get_parameters_to_update()
        self.create_optimizer(params)
        if(self.loss_calculater is None):
            loss_name = self.config['training']['loss_type']
            if(loss_name in SegLossDict):
                self.loss_calculater = SegLossDict[loss_name](self.config['training'])
            else:
                raise ValueError("Undefined loss function {0:}".format(loss_name))

        self.trainIter  = iter(self.train_loader)
        
        print("{0:} training start".format(str(datetime.now())[:-7]))
        self.summ_writer = SummaryWriter(self.config['training']['summary_dir'])
        for it in range(iter_start, iter_max, iter_valid):
            train_scalars = self.training()
            valid_scalars = self.validation()
            glob_it = it + iter_valid
            self.write_scalars(train_scalars, valid_scalars, glob_it)

            if(valid_scalars['avg_dice'] > self.max_val_dice):
                self.max_val_dice = valid_scalars['avg_dice']
                self.max_val_it   = glob_it
                self.best_model_wts = copy.deepcopy(self.net.state_dict())

            if (glob_it % iter_save ==  0):
                save_dict = {'iteration': glob_it,
                             'valid_pred': valid_scalars['avg_dice'],
                             'model_state_dict': self.net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict()}
                save_name = "{0:}_{1:}.pt".format(chpt_prefx, glob_it)
                torch.save(save_dict, save_name) 
        # save the best performing checkpoint
        save_dict = {'iteration': self.max_val_it,
                    'valid_pred': self.max_val_dice,
                    'model_state_dict': self.best_model_wts,
                    'optimizer_state_dict': self.optimizer.state_dict()}
        save_name = "{0:}_{1:}.pt".format(chpt_prefx, self.max_val_it)
        torch.save(save_dict, save_name) 
        print('The best perfroming iter is {0:}, valid dice {1:}'.format(\
            self.max_val_it, self.max_val_dice))
        self.summ_writer.close()
    
    
    def infer(self):
        device = torch.device(self.config['testing']['device_name'])
        self.net.to(device)
        # laod network parameters and set the network as evaluation mode
        self.checkpoint = torch.load(self.config['testing']['checkpoint_name'], map_location = device)
        self.net.load_state_dict(self.checkpoint['model_state_dict'])
        
        if(self.config['testing']['evaluation_mode'] == True):
            self.net.eval()
            if(self.config['testing']['test_time_dropout'] == True):
                def test_time_dropout(m):
                    if(type(m) == nn.Dropout):
                        print('dropout layer')
                        m.train()
                self.net.apply(test_time_dropout)
        output_dir   = self.config['testing']['output_dir']
        class_num    = self.config['network']['class_num']
        mini_batch_size      = self.config['testing']['mini_batch_size']
        mini_patch_inshape   = self.config['testing']['mini_patch_input_shape']
        mini_patch_outshape  = self.config['testing']['mini_patch_output_shape']
        mini_patch_stride    = self.config['testing']['mini_patch_stride']
        output_num       = self.config['testing'].get('output_num', 1)
        multi_pred_avg   = self.config['testing'].get('multi_pred_avg', False)
        save_probability = self.config['testing'].get('save_probability', False)
        save_var         = self.config['testing'].get('save_multi_pred_var', False)
        label_source = self.config['testing'].get('label_source', None)
        label_target = self.config['testing'].get('label_target', None)
        filename_replace_source = self.config['testing'].get('filename_replace_source', None)
        filename_replace_target = self.config['testing'].get('filename_replace_target', None)
        if(not os.path.exists(output_dir)):
            os.mkdir(output_dir)

        # automatically infer outupt shape
        # if(mini_patch_inshape is not None):
        #     patch_inshape = [1, self.config['dataset']['modal_num']] + mini_patch_inshape
        #     testx = np.random.random(patch_inshape)
        #     testx = torch.from_numpy(testx)
        #     testx = torch.tensor(testx)
        #     testx = self.convert_tensor_type(testx)
        #     testx = testx.to(device)
        #     testy = self.net(testx)
        #     if(isinstance(testy, tuple) or isinstance(testy, list)):
        #         testy = testy[0] 
        #     testy = testy.detach().cpu().numpy()
        #     mini_patch_outshape = testy.shape[2:]
        #     print('mini patch in shape', mini_patch_inshape)
        #     print('mini patch out shape', mini_patch_outshape)
        
        infer_time_list = []
        with torch.no_grad():
            for data in self.test_loder:
                images = self.convert_tensor_type(data['image'])
                names  = data['names']
                print(names[0])
                # for debug
                # for i in range(images.shape[0]):
                #     image_i = images[i][0]
                #     label_i = images[i][0]
                #     image_name = "temp/{0:}_image.nii.gz".format(names[0])
                #     label_name = "temp/{0:}_label.nii.gz".format(names[0])
                #     save_nd_array_as_image(image_i, image_name, reference_name = None)
                #     save_nd_array_as_image(label_i, label_name, reference_name = None)
                # continue
                start_time = time.time()
                
                data['predict']  = volume_infer(images, self.net, device, class_num, 
                    mini_batch_size, mini_patch_inshape, mini_patch_outshape, mini_patch_stride, output_num)
                
                for i in reversed(range(len(self.transform_list))):
                    if (self.transform_list[i].inverse):
                        data = self.transform_list[i].inverse_transform_for_prediction(data) 
                predict_list = [data['predict']]
                if(isinstance(data['predict'], tuple) or isinstance(data['predict'], list)):
                    predict_list = data['predict']

                infer_time = time.time() - start_time
                infer_time_list.append(infer_time)

                prob_list = [scipy.special.softmax(predict[0], axis = 0) for predict in predict_list]
                if(multi_pred_avg):
                    if(output_num == 1):
                        raise ValueError("multiple predictions expected, but output_num was set to 1")
                    if(output_num != len(prob_list)):
                        raise ValueError("expected output_num was set to {0:}, but {1:} outputs obtained".format(
                              output_dir, len(prob_list)))
                    prob_stack   = np.asarray(prob_list, np.float32)
                    prob   = np.mean(prob_stack, axis = 0)
                    var    = np.var(prob_stack, axis = 0)
                else:
                    prob = prob_list[0]
                # output = predict_list[2][0]
                output = np.asarray(np.argmax(prob,  axis = 0), np.uint8)

                if((label_source is not None) and (label_target is not None)):
                    output = convert_label(output, label_source, label_target)
                # save the output and (optionally) probability predictions
                root_dir  = self.config['dataset']['root_dir']
                save_name = names[0].split('/')[-1]
                if((filename_replace_source is  not None) and (filename_replace_target is not None)):
                    save_name = save_name.replace(filename_replace_source, filename_replace_target)
                save_name = "{0:}/{1:}".format(output_dir, save_name)
                save_nd_array_as_image(output, save_name, root_dir + '/' + names[0])
                save_name_split = save_name.split('.')
                if('.nii.gz' in save_name):
                    save_prefix = '.'.join(save_name_split[:-2])
                    save_format = 'nii.gz'
                else:
                    save_prefix = '.'.join(save_name_split[:-1])
                    save_format = save_name_split[-1]
                if(save_probability):
                    class_num = prob.shape[0]
                    for c in range(0, class_num):
                        temp_prob = prob[c]
                        prob_save_name = "{0:}_prob_{1:}.{2:}".format(save_prefix, c, save_format)
                        if(len(temp_prob.shape) == 2):
                            temp_prob = np.asarray(temp_prob * 255, np.uint8)
                        save_nd_array_as_image(temp_prob, prob_save_name, root_dir + '/' + names[0])
                if(save_var):
                    var = var[1]
                    var_save_name = "{0:}_var.{1:}".format(save_prefix, save_format)
                    save_nd_array_as_image(var, var_save_name, root_dir + '/' + names[0])

        infer_time_list = np.asarray(infer_time_list)
        time_avg = infer_time_list.mean()
        time_std = infer_time_list.std()
        print("testing time {0:} +/- {1:}".format(time_avg, time_std))