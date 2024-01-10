# -*- coding: utf-8 -*-
from __future__ import print_function, division
import copy
import os
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
from pymic.net.multi_net import MultiNet
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
from pymic.util.general import mixup, tensor_shape_match

class SegmentationAgent(NetRunAgent):
    def __init__(self, config, stage = 'train'):
        super(SegmentationAgent, self).__init__(config, stage)
        self.transform_dict   = TransformDict
        self.net_dict         = SegNetDict
        self.postprocess_dict = PostProcessDict
        self.postprocessor    = None

    def get_transform_names_and_parameters(self, stage):
        """
        Get a list of transform objects for creating a dataset
        """
        assert(stage in ['train', 'valid', 'test'])
        transform_key = stage +  '_transform'
        trans_names  = self.config['dataset'][transform_key]
        trans_params = self.config['dataset']
        trans_params['task'] = self.task_type
        return trans_names, trans_params

    def get_stage_dataset_from_config(self, stage):
        trans_names, trans_params = self.get_transform_names_and_parameters(stage)
        transform_list  = []
        if(trans_names is not None and len(trans_names) > 0):
            for name in trans_names:
                if(name not in self.transform_dict):
                    raise(ValueError("Undefined transform {0:}".format(name))) 
                one_transform = self.transform_dict[name](trans_params)
                transform_list.append(one_transform)
        data_transform = transforms.Compose(transform_list)

        csv_file  = self.config['dataset'].get(stage + '_csv', None)
        if(stage == 'test'):
            with_label = False 
            self.test_transforms = transform_list
        else:
            with_label = self.config['dataset'].get(stage + '_label', True)
        modal_num = self.config['dataset'].get('modal_num', 1)
        stage_dir = self.config['dataset'].get('train_dir', None)
        if(stage == 'valid' and "valid_dir" in self.config['dataset']):
            stage_dir = self.config['dataset']['valid_dir']
        if(stage == 'test' and "test_dir" in self.config['dataset']):
            stage_dir = self.config['dataset']['test_dir']
        dataset  = NiftyDataset(root_dir  = stage_dir,
                                csv_file  = csv_file,
                                modal_num = modal_num,
                                with_label= with_label,
                                transform = data_transform, 
                                task = self.task_type)
        return dataset

    def create_network(self):
        if(self.net is None):
            net_name = self.config['network']['net_type']
            if(isinstance(net_name, (tuple, list))):
                self.net = MultiNet(self.net_dict, self.config['network'])
            else:
                if(net_name not in self.net_dict):
                    raise ValueError("Undefined network {0:}".format(net_name))
                self.net = self.net_dict[net_name](self.config['network'])
        if(self.tensor_type == 'float'):
            self.net.float()
        else:
            self.net.double()
        if(hasattr(self.net, "set_stage")):
            self.net.set_stage(self.stage)
        param_number = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        logging.info('parameter number {0:}'.format(param_number))

    def get_parameters_to_update(self):
        if hasattr(self.net, "get_parameters_to_update"):
            params = self.net.get_parameters_to_update()
        else:
            params = self.net.parameters()
        return params


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
        if(self.config['training'].get('deep_supervise', False)):
            weight = self.config['training'].get('deep_supervise_weight', None)
            mode   = self.config['training'].get('deep_supervise_mode', 2)
            params = {'deep_supervise_weight': weight, 
                      'deep_supervise_mode': mode, 
                      'base_loss':base_loss}
            self.loss_calculator = DeepSuperviseLoss(params)
        else:
            self.loss_calculator = base_loss
                
    def get_loss_value(self, data, pred, gt, param = None):
        loss_input_dict = {'prediction':pred, 'ground_truth': gt}
        if(isinstance(pred, tuple) or isinstance(pred, list)):
            device = pred[0].device
        else:
            device = pred.device
        pixel_weight = data.get('pixel_weight', None) 
        if(pixel_weight is not None):
            loss_input_dict['pixel_weight'] = pixel_weight.to(device)

        class_weight = self.config['training'].get('class_weight', None)
        if(class_weight is not None):
            class_num = self.config['network']['class_num']
            assert(len(class_weight) == class_num)
            class_weight = torch.from_numpy(np.asarray(class_weight))
            class_weight = self.convert_tensor_type(class_weight)
            loss_input_dict['class_weight'] = class_weight.to(device)
        loss_value = self.loss_calculator(loss_input_dict)
        return loss_value
    
    def set_postprocessor(self, postprocessor):
        """
        Set post processor after prediction. 

        :param postprocessor: post processor, such as an instance of 
            `pymic.util.post_process.PostProcess`.
        """
        self.postprocessor = postprocessor

    def training(self):
        class_num   = self.config['network']['class_num']
        iter_valid  = self.config['training']['iter_valid']
        mixup_prob  = self.config['training'].get('mixup_probability', 0.0)
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
            inputs      = self.convert_tensor_type(data['image'])
            labels_prob = self.convert_tensor_type(data['label_prob'])                 
            if(mixup_prob > 0 and random() < mixup_prob):
                inputs, labels_prob = mixup(inputs, labels_prob) 
                   
            # for debug
            # print("current iteration", it)
            # if(it > 10):
            #     break
            # for i in range(inputs.shape[0]):
            #     image_i = inputs[i][0]
            #     # label_i = labels_prob[i][1]
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
        
    def validation(self):
        class_num = self.config['network']['class_num']
        if(self.inferer is None):
            infer_cfg = {}
            infer_cfg['class_num'] = class_num
            infer_cfg['sliding_window_enable'] = self.config['testing'].get('sliding_window_enable', False)
            if(infer_cfg['sliding_window_enable']):
                patch_size = self.config['dataset'].get('patch_size', None)
                if(patch_size is None):
                    patch_size = self.config['testing']['sliding_window_size']
                infer_cfg['sliding_window_size']   = patch_size
                infer_cfg['sliding_window_stride'] = [i//2 for i in patch_size]
            self.inferer = Inferer(infer_cfg)
        
        valid_loss_list = []
        valid_dice_list = []
        validIter  = iter(self.valid_loader)
        with torch.no_grad():
            self.net.eval()
            for data in validIter:
                inputs      = self.convert_tensor_type(data['image'])
                if('label_prob' not in data):
                    raise ValueError("label_prob is not found in validation data, make sure" + 
                        "that LabelToProbability is used in valid_transform.")
                labels_prob = self.convert_tensor_type(data['label_prob'])
                inputs, labels_prob  = inputs.to(self.device), labels_prob.to(self.device)
                batch_n = inputs.shape[0]
                outputs = self.inferer.run(self.net, inputs)

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
        valid_avg_dice = valid_cls_dice[1:].mean()
        valid_scalers = {'loss': valid_avg_loss, 'avg_fg_dice': valid_avg_dice,\
            'class_dice': valid_cls_dice}
        return valid_scalers

    def write_scalars(self, train_scalars, valid_scalars, lr_value, glob_it):
        loss_scalar ={'train':train_scalars['loss'], 'valid':valid_scalars['loss']}
        dice_scalar ={'train':train_scalars['avg_fg_dice'], 'valid':valid_scalars['avg_fg_dice']}
        self.summ_writer.add_scalars('loss', loss_scalar, glob_it)
        self.summ_writer.add_scalars('dice', dice_scalar, glob_it)
        self.summ_writer.add_scalars('lr', {"lr": lr_value}, glob_it)
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

    def load_pretrained_weights(self, network, pretrained_dict, device_ids):
        if(len(device_ids) > 1):
            if(hasattr(network.module, "get_parameters_to_load")):
                model_dict = network.module.get_parameters_to_load()
            else:
                model_dict = network.module.state_dict()
        else:
            if(hasattr(network, "get_parameters_to_load")):
                model_dict = network.get_parameters_to_load()
            else:
                model_dict = network.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if \
            k in model_dict and tensor_shape_match(pretrained_dict[k], model_dict[k])}
        logging.info("Initializing the following parameters with pre-trained model")
        for k in pretrained_dict:
            logging.info(k)
        if (len(device_ids) > 1):
            network.module.load_state_dict(pretrained_dict, strict = False)
        else:
            network.load_state_dict(pretrained_dict, strict = False) 

    def train_valid(self):
        device_ids = self.config['training']['gpus']
        if(len(device_ids) > 1):
            self.device = torch.device("cuda:0")
            self.net = nn.DataParallel(self.net, device_ids = device_ids)
        else:
            self.device = torch.device("cuda:{0:}".format(device_ids[0]))
        self.net.to(self.device)
        
        ckpt_dir    = self.config['training']['ckpt_save_dir']
        if(ckpt_dir[-1] == "/"):
            ckpt_dir = ckpt_dir[:-1]
        ckpt_prefix = self.config['training'].get('ckpt_prefix', None)
        if(ckpt_prefix is None):
            ckpt_prefix = ckpt_dir.split('/')[-1]
        # iter_start  = self.config['training']['iter_start']       
        iter_start  = 0     
        iter_max    = self.config['training']['iter_max']
        iter_valid  = self.config['training']['iter_valid']
        iter_save   = self.config['training'].get('iter_save', None)
        early_stop_it = self.config['training'].get('early_stop_patience', None)
        if(iter_save is None):
            iter_save_list = [iter_max]
        elif(isinstance(iter_save, (tuple, list))):
            iter_save_list = iter_save
        else:
            iter_save_list = range(0, iter_max + 1, iter_save)

        self.max_val_dice = 0.0
        self.max_val_it   = 0
        self.best_model_wts = None 
        checkpoint = None
        # initialize the network with pre-trained weights
        ckpt_init_name = self.config['training'].get('ckpt_init_name', None)
        ckpt_init_mode = self.config['training'].get('ckpt_init_mode', 0)
        ckpt_for_optm  = None 
        if(ckpt_init_name is not None):
            checkpoint = torch.load(ckpt_dir + "/" + ckpt_init_name, map_location = self.device)
            pretrained_dict = checkpoint['model_state_dict']
            self.load_pretrained_weights(self.net, pretrained_dict, device_ids)

            if(ckpt_init_mode > 0): # Load  other information
                self.max_val_dice = checkpoint.get('valid_pred', 0)
                iter_start = checkpoint['iteration']
                self.max_val_it = iter_start
                self.best_model_wts = checkpoint['model_state_dict']
                ckpt_for_optm = checkpoint
        self.create_optimizer(self.get_parameters_to_update(), ckpt_for_optm)
        self.create_loss_calculator()
    
        self.trainIter  = iter(self.train_loader)
        
        logging.info("{0:} training start".format(str(datetime.now())[:-7]))
        self.summ_writer = SummaryWriter(self.config['training']['ckpt_save_dir'])
        self.glob_it = iter_start
        for it in range(iter_start, iter_max, iter_valid):
            lr_value = self.optimizer.param_groups[0]['lr']
            t0 = time.time()
            train_scalars = self.training()
            t1 = time.time()
            valid_scalars = self.validation()
            t2 = time.time()
            if(isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau)):
                self.scheduler.step(valid_scalars['avg_fg_dice'])
            else:
                self.scheduler.step()

            self.glob_it = it + iter_valid
            logging.info("\n{0:} it {1:}".format(str(datetime.now())[:-7], self.glob_it))
            logging.info('learning rate {0:}'.format(lr_value))
            logging.info("training/validation time: {0:.2f}s/{1:.2f}s".format(t1-t0, t2-t1))
            self.write_scalars(train_scalars, valid_scalars, lr_value, self.glob_it)

            if(valid_scalars['avg_fg_dice'] > self.max_val_dice):
                self.max_val_dice = valid_scalars['avg_fg_dice']
                self.max_val_it   = self.glob_it
                if(len(device_ids) > 1):
                    self.best_model_wts = copy.deepcopy(self.net.module.state_dict())
                else:
                    self.best_model_wts = copy.deepcopy(self.net.state_dict())
                save_dict = {'iteration': self.max_val_it,
                    'valid_pred': self.max_val_dice,
                    'model_state_dict': self.best_model_wts,
                    'optimizer_state_dict': self.optimizer.state_dict()}
                save_name = "{0:}/{1:}_best.pt".format(ckpt_dir, ckpt_prefix)
                torch.save(save_dict, save_name) 
                txt_file = open("{0:}/{1:}_best.txt".format(ckpt_dir, ckpt_prefix), 'wt')
                txt_file.write(str(self.max_val_it))
                txt_file.close()

            stop_now = True if (early_stop_it is not None and \
                self.glob_it - self.max_val_it > early_stop_it) else False
            if ((self.glob_it in iter_save_list) or stop_now):
                save_dict = {'iteration': self.glob_it,
                             'valid_pred': valid_scalars['avg_fg_dice'],
                             'model_state_dict': self.net.module.state_dict() \
                                 if len(device_ids) > 1 else self.net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict()}
                save_name = "{0:}/{1:}_{2:}.pt".format(ckpt_dir, ckpt_prefix, self.glob_it)
                torch.save(save_dict, save_name) 
                txt_file = open("{0:}/{1:}_latest.txt".format(ckpt_dir, ckpt_prefix), 'wt')
                txt_file.write(str(self.glob_it))
                txt_file.close()
            if(stop_now):
                logging.info("The training is early stopped")
                break
        # save the best performing checkpoint
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
        print("ckpt name", ckpt_name)
        checkpoint = torch.load(ckpt_name, map_location = device)
        self.net.load_state_dict(checkpoint['model_state_dict'])

        if(self.inferer is None):
            infer_cfg = self.config['testing']
            infer_cfg['class_num'] = self.config['network']['class_num']
            self.inferer = Inferer(infer_cfg)
        postpro_name = self.config['testing'].get('post_process', None)
        if(self.postprocessor is None and postpro_name is not None):
            self.postprocessor = PostProcessDict[postpro_name](self.config['testing'])
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
                
                pred = self.inferer.run(self.net, images)
                # convert tensor to numpy
                if(isinstance(pred, (tuple, list))):
                    pred = [item.cpu().numpy() for item in pred]
                else:
                    pred = pred.cpu().numpy()
                data['predict'] = pred
                # inverse transform
                for transform in self.test_transforms[::-1]:
                    if (transform.inverse):
                        data = transform.inverse_transform_for_prediction(data) 

                infer_time = time.time() - start_time
                infer_time_list.append(infer_time)
                self.save_outputs(data)
        infer_time_list = np.asarray(infer_time_list)
        time_avg, time_std = infer_time_list.mean(), infer_time_list.std()
        logging.info("testing time {0:} +/- {1:}".format(time_avg, time_std))

    def infer_with_multiple_checkpoints(self):
        """
        Inference with ensemble of multilple check points.
        """
        device_ids = self.config['testing']['gpus']
        device = torch.device("cuda:{0:}".format(device_ids[0]))

        if(self.inferer is None):
            infer_cfg  = self.config['testing']
            infer_cfg['class_num'] = self.config['network']['class_num']
            self.inferer = Inferer(infer_cfg)
        ckpt_names = self.config['testing']['ckpt_name']
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
                    
                    pred = self.inferer.run(self.net, images)
                    # convert tensor to numpy
                    if(isinstance(pred, (tuple, list))):
                        pred = [item.cpu().numpy() for item in pred]
                    else:
                        pred = pred.cpu().numpy()
                    predict_list.append(pred)
                pred = np.mean(predict_list, axis=0)
                data['predict'] = pred
                # inverse transform
                for transform in self.test_transforms[::-1]:
                    if (transform.inverse):
                        data = transform.inverse_transform_for_prediction(data) 
                
                infer_time = time.time() - start_time
                infer_time_list.append(infer_time)
                self.save_outputs(data)
        infer_time_list = np.asarray(infer_time_list)
        time_avg, time_std = infer_time_list.mean(), infer_time_list.std()
        logging.info("testing time {0:} +/- {1:}".format(time_avg, time_std))

    def save_outputs(self, data):
        """
        Save prediction output. 

        :param data: (dictionary) A data dictionary with prediciton result and other 
            information such as input image name. 
        """
        output_dir = self.config['testing']['output_dir']
        ignore_dir = self.config['testing'].get('filename_ignore_dir', True)
        save_prob  = self.config['testing'].get('save_probability', False)
        label_source = self.config['testing'].get('label_source', None)
        label_target = self.config['testing'].get('label_target', None)
        filename_replace_source = self.config['testing'].get('filename_replace_source', None)
        filename_replace_target = self.config['testing'].get('filename_replace_target', None)
        if(not os.path.exists(output_dir)):
            os.makedirs(output_dir, exist_ok=True)

        names, pred = data['names'], data['predict']
        if(isinstance(pred, (list, tuple))):
            pred =  pred[0]
        prob   = scipy.special.softmax(pred, axis = 1) 
        output = np.asarray(np.argmax(prob,  axis = 1), np.uint8)
        if((label_source is not None) and (label_target is not None)):
            output = convert_label(output, label_source, label_target)
        if(self.postprocessor is not None):
            for i in range(len(names)):
                output[i] = self.postprocessor(output[i])
        # save the output and (optionally) probability predictions
        test_dir = self.config['dataset'].get('test_dir', None)
        if(test_dir is None):
            test_dir = self.config['dataset']['train_dir']

        for i in range(len(names)):
            save_name = names[i][0].split('/')[-1] if ignore_dir else \
                names[i][0].replace('/', '_')
            if((filename_replace_source is  not None) and (filename_replace_target is not None)):
                save_name = save_name.replace(filename_replace_source, filename_replace_target)
            print(save_name)
            save_name = "{0:}/{1:}".format(output_dir, save_name)
            save_nd_array_as_image(output[i], save_name, test_dir + '/' + names[i][0])
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
                save_nd_array_as_image(temp_prob, prob_save_name, test_dir + '/' + names[i][0])
