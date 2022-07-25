# -*- coding: utf-8 -*-
from __future__ import print_function, division
from cmath import log

import copy
import os
import sys
import time
import random
import logging
import scipy
import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
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
from pymic.util.image_process import convert_label
from pymic.util.parse_config import parse_config

class SegmentationAgent(NetRunAgent):
    def __init__(self, config, stage = 'train'):
        super(SegmentationAgent, self).__init__(config, stage)
        self.transform_dict  = TransformDict
        
    def get_stage_dataset_from_config(self, stage):
        assert(stage in ['train', 'valid', 'test'])
        root_dir  = self.config['dataset']['root_dir']
        modal_num = self.config['dataset'].get('modal_num', 1)

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
        dataset  = NiftyDataset(root_dir  = root_dir,
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
        logging.info('parameter number {0:}'.format(param_number))

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
            if(self.net.training):
                if('image_weight' not in data):
                    raise ValueError("image weight is enabled not not provided")
                img_w = data['image_weight']
            else:
                img_w = data.get('image_weight', None)
        if(img_w is None):        
            batch_size = data['image'].shape[0]
            img_w = torch.ones(batch_size)
        img_w = self.convert_tensor_type(img_w)
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

    def create_loss_calculator(self):
        if(self.loss_dict is None):
            self.loss_dict = SegLossDict
        loss_name = self.config['training']['loss_type']
        if isinstance(loss_name, (list, tuple)):
            base_loss = CombinedLoss(self.config['training'], self.loss_dict)
        elif (loss_name not in self.loss_dict):
            raise ValueError("Undefined loss function {0:}".format(loss_name))
        else:
            base_loss = self.loss_dict[loss_name](self.config['training'])
        if(self.config['network'].get('deep_supervise', False)):
            weight = self.config['network'].get('deep_supervise_weight', None)
            params = {'deep_supervise_weight': weight, 'base_loss':base_loss}
            self.loss_calculator = DeepSuperviseLoss(params)
        else:
            self.loss_calculator = base_loss
                
    def get_loss_value(self, data, pred, gt, param = None):
        """
        Assume pred and gt has been sent to self.device
        data is obtained by dataloaderis  and is dictionary containing extra 
        information, such as pixel-level weight, class-level weight.
        By default, such information is not used by standard loss functions 
        such as Dice loss and cross entropy loss.  
        """
        loss_input_dict = {'prediction':pred, 'ground_truth': gt}
        if data.get('pixel_weight', None) is not None:
            loss_input_dict['pixel_weight'] = data['pixel_weight'].to(pred.device)
        loss_value = self.loss_calculator(loss_input_dict)
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
            loss = self.get_loss_value(data, outputs, labels_prob)
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
        class_num = self.config['network']['class_num']
        infer_cfg = self.config['testing']
        infer_cfg['class_num'] = class_num
        
        valid_loss_list = []
        valid_dice_list = []
        validIter  = iter(self.valid_loader)
        with torch.no_grad():
            self.net.eval()
            infer_obj = Inferer(self.net, infer_cfg)
            for data in validIter:
                inputs      = self.convert_tensor_type(data['image'])
                labels_prob = self.convert_tensor_type(data['label_prob'])
                inputs, labels_prob  = inputs.to(self.device), labels_prob.to(self.device)
                batch_n = inputs.shape[0]
                outputs = infer_obj.run(inputs)

                # The tensors are on CPU when calculating loss for validation data
                loss = self.get_loss_value(data, outputs, labels_prob)
                valid_loss_list.append(loss.item())

                if(isinstance(outputs, tuple) or isinstance(outputs, list)):
                    outputs = outputs[0] 
                outputs_argmax = torch.argmax(outputs, dim = 1, keepdim = True)
                soft_out  = get_soft_label(outputs_argmax, class_num, self.tensor_type)
                for i in range(batch_n):
                    soft_out_i, labels_prob_i = reshape_prediction_and_ground_truth(\
                        soft_out[i:i+1], labels_prob[i:i+1])
                    temp_dice = get_classwise_dice(soft_out_i, labels_prob_i)
                    valid_dice_list.append(temp_dice.cpu().numpy())

        valid_avg_loss = np.asarray(valid_loss_list).mean()
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
       
        logging.info('train loss {0:.4f}, avg dice {1:.4f} '.format(
            train_scalars['loss'], train_scalars['avg_dice']) + "[" + \
            ' '.join("{0:.4f}".format(x) for x in train_scalars['class_dice']) + "]")        
        logging.info('valid loss {0:.4f}, avg dice {1:.4f} '.format(
            valid_scalars['loss'], valid_scalars['avg_dice']) + "[" + \
            ' '.join("{0:.4f}".format(x) for x in valid_scalars['class_dice']) + "]")        

    def train_valid(self):
        device_ids = self.config['training']['gpus']
        if(len(device_ids) > 1):
            self.device = torch.device("cuda:0")
            self.net = nn.DataParallel(self.net, device_ids = device_ids)
        else:
            self.device = torch.device("cuda:{0:}".format(device_ids[0]))
        self.net.to(self.device)
        ckpt_dir    = self.config['training']['ckpt_save_dir']
        ckpt_prefx  = ckpt_dir.split('/')[-1]
        iter_start  = self.config['training']['iter_start']
        iter_max    = self.config['training']['iter_max']
        iter_valid  = self.config['training']['iter_valid']
        iter_save   = self.config['training']['iter_save']
        if(isinstance(iter_save, (tuple, list))):
            iter_save_list = iter_save
        else:
            iter_save_list = range(iter_start, iter_max +1, iter_save)

        self.max_val_dice = 0.0
        self.max_val_it   = 0
        self.best_model_wts = None 
        self.checkpoint = None
        if(iter_start > 0):
            checkpoint_file = "{0:}/{1:}_{2:}.pt".format(ckpt_dir, ckpt_prefx, iter_start)
            self.checkpoint = torch.load(checkpoint_file, map_location = self.device)
            # assert(self.checkpoint['iteration'] == iter_start)
            if(len(device_ids) > 1):
                self.net.module.load_state_dict(self.checkpoint['model_state_dict'])
            else:
                self.net.load_state_dict(self.checkpoint['model_state_dict'])
            self.max_val_dice = self.checkpoint.get('valid_pred', 0)
            # self.max_val_it   = self.checkpoint['iteration']
            self.max_val_it   = iter_start
            self.best_model_wts = self.checkpoint['model_state_dict']
            
        self.create_optimizer(self.get_parameters_to_update())
        self.create_loss_calculator()
    
        self.trainIter  = iter(self.train_loader)
        
        logging.info("{0:} training start".format(str(datetime.now())[:-7]))
        self.summ_writer = SummaryWriter(self.config['training']['ckpt_save_dir'])
        self.glob_it = iter_start
        for it in range(iter_start, iter_max, iter_valid):
            t0 = time.time()
            train_scalars = self.training()
            t1 = time.time()
            valid_scalars = self.validation()
            t2 = time.time()
            self.glob_it = it + iter_valid
            logging.info("{0:} it {1:}".format(str(datetime.now())[:-7], self.glob_it))
            logging.info("training/validation time: {0:.2f}s/{1:.2f}s".format(t1-t0, t2-t1))
            self.write_scalars(train_scalars, valid_scalars, self.glob_it)
            if(valid_scalars['avg_dice'] > self.max_val_dice):
                self.max_val_dice = valid_scalars['avg_dice']
                self.max_val_it   = self.glob_it
                if(len(device_ids) > 1):
                    self.best_model_wts = copy.deepcopy(self.net.module.state_dict())
                else:
                    self.best_model_wts = copy.deepcopy(self.net.state_dict())

            if (self.glob_it in iter_save_list):
                save_dict = {'iteration': self.glob_it,
                             'valid_pred': valid_scalars['avg_dice'],
                             'model_state_dict': self.net.module.state_dict() \
                                 if len(device_ids) > 1 else self.net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict()}
                save_name = "{0:}/{1:}_{2:}.pt".format(ckpt_dir, ckpt_prefx, self.glob_it)
                torch.save(save_dict, save_name) 
                txt_file = open("{0:}/{1:}_latest.txt".format(ckpt_dir, ckpt_prefx), 'wt')
                txt_file.write(str(self.glob_it))
                txt_file.close()
        # save the best performing checkpoint
        save_dict = {'iteration': self.max_val_it,
                    'valid_pred': self.max_val_dice,
                    'model_state_dict': self.best_model_wts,
                    'optimizer_state_dict': self.optimizer.state_dict()}
        save_name = "{0:}/{1:}_{2:}.pt".format(ckpt_dir, ckpt_prefx, self.max_val_it)
        torch.save(save_dict, save_name) 
        txt_file = open("{0:}/{1:}_best.txt".format(ckpt_dir, ckpt_prefx), 'wt')
        txt_file.write(str(self.max_val_it))
        txt_file.close()
        logging.info('The best performing iter is {0:}, valid dice {1:}'.format(\
            self.max_val_it, self.max_val_dice))
        self.summ_writer.close()
    
    def infer(self):
        device_ids = self.config['testing']['gpus']
        device = torch.device("cuda:{0:}".format(device_ids[0]))
        self.net.to(device)

        if(self.config['testing'].get('evaluation_mode', True)):
            self.net.eval()
            if(self.config['testing'].get('test_time_dropout', False)):
                def test_time_dropout(m):
                    if(type(m) == nn.Dropout):
                        logging.info('dropout layer')
                        m.train()
                self.net.apply(test_time_dropout)

        ckpt_mode = self.config['testing']['ckpt_mode']
        ckpt_name = self.get_checkpoint_name()
        if(ckpt_mode == 3):
            assert(isinstance(ckpt_name, (tuple, list)))
            self.infer_with_multiple_checkpoints()
            return 
        else:
            if(isinstance(ckpt_name, (tuple, list))):
                raise ValueError("ckpt_mode should be 3 if ckpt_name is a list")

        # load network parameters and set the network as evaluation mode
        checkpoint = torch.load(ckpt_name, map_location = device)
        self.net.load_state_dict(checkpoint['model_state_dict'])

        infer_cfg = self.config['testing']
        infer_cfg['class_num'] = self.config['network']['class_num']
        infer_obj = Inferer(self.net, infer_cfg)
        infer_time_list = []
        with torch.no_grad():
            for data in self.test_loader:
                images = self.convert_tensor_type(data['image'])
                images = images.to(device)
    
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
                
                pred = infer_obj.run(images)
                # convert tensor to numpy
                if(isinstance(pred, (tuple, list))):
                    pred = [item.cpu().numpy() for item in pred]
                else:
                    pred = pred.cpu().numpy()
                data['predict'] = pred
                # inverse transform
                for transform in self.transform_list[::-1]:
                    if (transform.inverse):
                        data = transform.inverse_transform_for_prediction(data) 

                infer_time = time.time() - start_time
                infer_time_list.append(infer_time)
                self.save_ouputs(data)
        infer_time_list = np.asarray(infer_time_list)
        time_avg, time_std = infer_time_list.mean(), infer_time_list.std()
        print("testing time {0:} +/- {1:}".format(time_avg, time_std))

    def infer_with_multiple_checkpoints(self):
        """
        inference with ensemble of multilple check points
        """
        device_ids = self.config['testing']['gpus']
        device = torch.device("cuda:{0:}".format(device_ids[0]))

        ckpt_names = self.config['testing']['ckpt_name']
        infer_cfg  = self.config['testing']
        infer_cfg['class_num'] = self.config['network']['class_num']
        infer_obj = Inferer(self.net, infer_cfg)
        infer_time_list = []
        with torch.no_grad():
            for data in self.test_loader:
                images = self.convert_tensor_type(data['image'])
                images = images.to(device)
    
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
                predict_list = []
                for ckpt_name in ckpt_names:
                    checkpoint = torch.load(ckpt_name, map_location = device)
                    self.net.load_state_dict(checkpoint['model_state_dict'])
                    
                    pred = infer_obj.run(images)
                    # convert tensor to numpy
                    if(isinstance(pred, (tuple, list))):
                        pred = [item.cpu().numpy() for item in pred]
                    else:
                        pred = pred.cpu().numpy()
                    predict_list.append(pred)
                pred = np.mean(predict_list, axis=0)
                data['predict'] = pred
                # inverse transform
                for transform in self.transform_list[::-1]:
                    if (transform.inverse):
                        data = transform.inverse_transform_for_prediction(data) 
                
                infer_time = time.time() - start_time
                infer_time_list.append(infer_time)
                self.save_ouputs(data)
        infer_time_list = np.asarray(infer_time_list)
        time_avg, time_std = infer_time_list.mean(), infer_time_list.std()
        print("testing time {0:} +/- {1:}".format(time_avg, time_std))

    def save_ouputs(self, data):
        output_dir = self.config['testing']['output_dir']
        ignore_dir = self.config['testing'].get('filename_ignore_dir', True)
        save_prob  = self.config['testing'].get('save_probability', False)
        label_source = self.config['testing'].get('label_source', None)
        label_target = self.config['testing'].get('label_target', None)
        filename_replace_source = self.config['testing'].get('filename_replace_source', None)
        filename_replace_target = self.config['testing'].get('filename_replace_target', None)
        if(not os.path.exists(output_dir)):
            os.mkdir(output_dir)

        names, pred = data['names'], data['predict']
        if(isinstance(pred, (list, tuple))):
            pred =  pred[0]
        prob   = scipy.special.softmax(pred, axis = 1) 
        output = np.asarray(np.argmax(prob,  axis = 1), np.uint8)
        if((label_source is not None) and (label_target is not None)):
            output = convert_label(output, label_source, label_target)
        # save the output and (optionally) probability predictions
        root_dir  = self.config['dataset']['root_dir']
        for i in range(len(names)):
            save_name = names[i].split('/')[-1] if ignore_dir else \
                names[i].replace('/', '_')
            if((filename_replace_source is  not None) and (filename_replace_target is not None)):
                save_name = save_name.replace(filename_replace_source, filename_replace_target)
            print(save_name)
            save_name = "{0:}/{1:}".format(output_dir, save_name)
            save_nd_array_as_image(output[i], save_name, root_dir + '/' + names[i])
            save_name_split = save_name.split('.')

            if(not save_prob):
                continue
            if('.nii.gz' in save_name):
                save_prefix = '.'.join(save_name_split[:-2])
                save_format = 'nii.gz'
            else:
                save_prefix = '.'.join(save_name_split[:-1])
                save_format = save_name_split[-1]
            
            class_num = prob.shape[1]
            for c in range(0, class_num):
                temp_prob = prob[i][c]
                prob_save_name = "{0:}_prob_{1:}.{2:}".format(save_prefix, c, save_format)
                if(len(temp_prob.shape) == 2):
                    temp_prob = np.asarray(temp_prob * 255, np.uint8)
                save_nd_array_as_image(temp_prob, prob_save_name, root_dir + '/' + names[i])
