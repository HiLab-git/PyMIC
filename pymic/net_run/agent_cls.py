# -*- coding: utf-8 -*-
from __future__ import print_function, division

import copy
import csv
import os
import sys
import time
import random
import scipy
import torch
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from PIL import Image 
from scipy import special
from datetime import datetime
from tensorboardX import SummaryWriter
from pymic.io.image_read_write import save_nd_array_as_image
from pymic.io.nifty_dataset import ClassificationDataset
from pymic.loss.loss_dict_cls import PyMICClsLossDict
from pymic.net.net_dict_cls import TorchClsNetDict
from pymic.net_run.get_optimizer import get_optimiser
from pymic.transform.trans_dict import TransformDict
from pymic.util.image_process import convert_label
from pymic.util.parse_config import parse_config
from pymic.net_run.agent_abstract import NetRunAgent
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

class ClassificationAgent(NetRunAgent):
    def __init__(self, config, stage = 'train'):
        super(ClassificationAgent, self).__init__(config, stage)
        self.transform_dict  = TransformDict

    def get_stage_dataset_from_config(self, stage):
        assert(stage in ['train', 'valid', 'test'])
        root_dir  = self.config['dataset']['root_dir']
        modal_num = self.config['dataset']['modal_num']

        if(stage == "train"):
            transform_names = self.config['dataset']['train_transform']
        elif(stage == "valid"):
            transform_names = self.config['dataset']['valid_transform']
        elif(stage == "test"):
            transform_names = self.config['dataset']['test_transform']
        else:
            raise ValueError("Incorrect value for stage: {0:}".format(stage))
        self.transform_list  = []
        if(transform_names is None or len(transform_names) == 0):
            data_transform = None 
        else:
            transform_param = self.config['dataset']
            transform_param['task'] = 'classification' 
            for name in transform_names:
                if(name not in self.transform_dict):
                    raise(ValueError("Undefined transform {0:}".format(name))) 
                one_transform = self.transform_dict[name](transform_param)
                self.transform_list.append(one_transform)
            data_transform = transforms.Compose(self.transform_list)

        csv_file = self.config['dataset'].get(stage + '_csv', None)
        class_num = self.config['network']['class_num']
        dataset  = ClassificationDataset(root_dir=root_dir,
                                csv_file  = csv_file,
                                modal_num = modal_num,
                                class_num = class_num,
                                with_label= not (stage == 'test'),
                                transform = data_transform )
        return dataset

    def create_network(self):
        if(self.net is None):
            net_name = self.config['network']['net_type']
            if(net_name not in TorchClsNetDict):
                raise ValueError("Undefined network {0:}".format(net_name))
            self.net = TorchClsNetDict[net_name](self.config['network'])
        if(self.tensor_type == 'float'):
            self.net.float()
        else:
            self.net.double()
        param_number = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print('parameter number:', param_number)

    def get_parameters_to_update(self):
        params = self.net.get_parameters_to_update()
        return params

    def get_loss_value(self, data, inputs, outputs, labels):
        loss_input_dict = {}
        loss_input_dict['prediction'] = outputs
        loss_input_dict['ground_truth'] = labels
        loss_value = self.loss_calculater(loss_input_dict)
        return loss_value
        
    def training(self):
        iter_valid  = self.config['training']['iter_valid']
        sample_num   = 0
        running_loss = 0
        running_corrects = 0
        self.net.train()
        for it in range(iter_valid):
            try:
                data = next(self.trainIter)
            except StopIteration:
                self.trainIter = iter(self.train_loader)
                data = next(self.trainIter)
            inputs = self.convert_tensor_type(data['image'])
            labels = data['label'].long()          
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward + backward + optimize
            outputs = self.net(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.get_loss_value(data, inputs, outputs, labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # statistics
            sample_num   += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_avg_loss = running_loss / sample_num
        train_avg_acc  = running_corrects.double() / sample_num
        train_scalers = {'loss': train_avg_loss, 'acc': train_avg_acc}
        return train_scalers

    def validation(self):
        validIter  = iter(self.valid_loader)
        sample_num   = 0
        running_loss = 0
        running_corrects = 0
        with torch.no_grad():
            self.net.eval()
            for data in validIter:
                inputs = self.convert_tensor_type(data['image'])
                labels = data['label'].long()          
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.net(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.get_loss_value(data, inputs, outputs, labels)
                                
                # statistics
                sample_num   += inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        avg_loss = running_loss / sample_num
        avg_acc  = running_corrects.double() / sample_num
        scalers = {'loss': avg_loss, 'acc': avg_acc}
        return scalers

    def write_scalars(self, train_scalars, valid_scalars, glob_it):
        loss_scalar ={'train':train_scalars['loss'], 'valid':valid_scalars['loss']}
        acc_scalar  ={'train':train_scalars['acc'], 'valid':valid_scalars['acc']}
        self.summ_writer.add_scalars('loss', loss_scalar, glob_it)
        self.summ_writer.add_scalars('acc', acc_scalar, glob_it)
        
        print("{0:} it {1:}".format(str(datetime.now())[:-7], glob_it))
        print('train loss {0:.4f}, avg acc {1:.4f}'.format(
            train_scalars['loss'], train_scalars['acc']))
        print('valid loss {0:.4f}, avg acc {1:.4f}'.format(
            valid_scalars['loss'], valid_scalars['acc'])) 

    def train_valid(self):
        self.device = torch.device(self.config['training']['device_name'])
        self.net.to(self.device)
        chpt_prefx  = self.config['training']['checkpoint_prefix']
        iter_start  = self.config['training']['iter_start']
        iter_max    = self.config['training']['iter_max']
        iter_valid  = self.config['training']['iter_valid']
        iter_save   = self.config['training']['iter_save']

        self.max_val_acc  = 0.0
        self.max_val_it   = 0
        self.best_model_wts = None 
        self.checkpoint = None
        if(iter_start > 0):
            checkpoint_file = "{0:}_{1:}.pt".format(chpt_prefx, iter_start)
            self.checkpoint = torch.load(checkpoint_file, map_location = self.device)
            assert(self.checkpoint['iteration'] == iter_start)
            self.net.load_state_dict(self.checkpoint['model_state_dict'])
            self.max_val_acc  = self.checkpoint.get('valid_pred', 0)
            self.max_val_it   = self.checkpoint['iteration']
            self.best_model_wts = self.checkpoint['model_state_dict']
        
        params = self.get_parameters_to_update()
        self.create_optimizer(params)
        if(self.loss_calculater is None):
            loss_name = self.config['training']['loss_type']
            if(loss_name in PyMICClsLossDict):
                self.loss_calculater = PyMICClsLossDict[loss_name](self.config['training'])
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

            if(valid_scalars['acc'] > self.max_val_acc):
                self.max_val_acc = valid_scalars['acc']
                self.max_val_it  = glob_it
                self.best_model_wts = copy.deepcopy(self.net.state_dict())
            
            if (glob_it % iter_save ==  0):
                save_dict = {'iteration': glob_it,
                             'valid_pred': valid_scalars['acc'],
                             'model_state_dict': self.net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict()}
                save_name = "{0:}_{1:}.pt".format(chpt_prefx, glob_it)
                torch.save(save_dict, save_name)

        # save the best performing checkpoint
        save_dict = {'iteration': self.max_val_it,
                    'valid_pred': self.max_val_acc,
                    'model_state_dict': self.best_model_wts,
                    'optimizer_state_dict': self.optimizer.state_dict()}
        save_name = "{0:}_{1:}.pt".format(chpt_prefx, self.max_val_it)
        torch.save(save_dict, save_name) 
        print('The best perfroming iter is {0:}, valid acc {1:}'.format(\
            self.max_val_it, self.max_val_acc))
        self.summ_writer.close()


    def infer(self):
        device = torch.device(self.config['testing']['device_name'])
        self.net.to(device)
        # laod network parameters and set the network as evaluation mode
        self.checkpoint = torch.load(self.config['testing']['checkpoint_name'], map_location = device)
        self.net.load_state_dict(self.checkpoint['model_state_dict'])
        
        if(self.config['testing']['evaluation_mode'] == True):
            self.net.eval()
            
        output_csv   = self.config['testing']['output_csv']
        class_num    = self.config['network']['class_num']
        save_probability = self.config['testing'].get('save_probability', False)
        
        infer_time_list = []
        out_prob_list   = []
        out_lab_list    = []
        with torch.no_grad():
            for data in self.test_loder:
                names  = data['names']
                inputs = self.convert_tensor_type(data['image'])
                inputs = inputs.to(device) 
                print(names[0])
                start_time = time.time()
                out_digit = self.net(inputs)
                out_prob  = nn.Softmax(dim = 1)(out_digit)
                out_prob  = out_prob.detach().cpu().numpy()[0]
                out_lab   = np.argmax(out_prob)
                infer_time = time.time() - start_time
                infer_time_list.append(infer_time)
                out_lab_list.append([names[0], out_lab])
                out_prob_list.append([names[0]] + out_prob.tolist())
        
        with open(output_csv, mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', 
                                quotechar='"',quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['image', 'label'])
            for item in out_lab_list:
                csv_writer.writerow(item)
        if(save_probability):
            prob_csv = output_csv.replace(".csv", "_prob.csv")
            with open(prob_csv, mode='w') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',', 
                                    quotechar='"',quoting=csv.QUOTE_MINIMAL)
                head = ['image']+['prob{}'.format(i) for i in range(class_num)]
                csv_writer.writerow(head)
                for item in out_prob_list:
                    csv_writer.writerow(item)

        infer_time_list = np.asarray(infer_time_list)
        time_avg = infer_time_list.mean()
        time_std = infer_time_list.std()
        print("testing time {0:} +/- {1:}".format(time_avg, time_std))

