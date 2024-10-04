# -*- coding: utf-8 -*-
from __future__ import print_function, division

import copy
import csv
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from random import random
from torch.optim import lr_scheduler
from torchvision import transforms
from tensorboardX import SummaryWriter
from pymic import TaskType
from pymic.io.nifty_dataset import ClassificationDataset
from pymic.loss.loss_dict_cls import PyMICClsLossDict
from pymic.net.net_dict_cls import TorchClsNetDict
from pymic.transform.trans_dict import TransformDict
from pymic.net_run.agent_abstract import NetRunAgent
from pymic.util.general import mixup, tensor_shape_match
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

class ClassificationAgent(NetRunAgent):
    """
    The agent for image classificaiton tasks.

    :param config: (dict) A dictionary containing the configuration.
    :param stage: (str) One of the stage in `train` (default), `inference` or `test`. 

    .. note::

        The config dictionary should have at least four sections: `dataset`,
        `network`, `training` and `inference`. See :doc:`usage.quickstart` and
        :doc:`usage.fsl` for example.
    """
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
            transform_param['task'] = self.task_type
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
                                transform = data_transform,
                                task = self.task_type)
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
        logging.info('parameter number {0:}'.format(param_number))

    def get_parameters_to_update(self):
        params = self.net.get_parameters_to_update()
        return params

    def create_loss_calculator(self):
        if(self.loss_dict is None):
            self.loss_dict = PyMICClsLossDict
        loss_name = self.config['training']['loss_type']
        if(loss_name != "SigmoidCELoss" and self.task_type == TaskType.CLASSIFICATION_COEXIST):
            raise ValueError("SigmoidCELoss should be used when task_type is cls_coexist")
        if(loss_name in self.loss_dict):
            self.loss_calculater = self.loss_dict[loss_name](self.config['training'])
        else:
            raise ValueError("Undefined loss function {0:}".format(loss_name))

    def get_loss_value(self, data, pred, gt, param = None):
        loss_input_dict = {}
        loss_input_dict['prediction'] = pred
        loss_input_dict['ground_truth'] = gt
        loss_value = self.loss_calculater(loss_input_dict)
        return loss_value
        
    def get_evaluation_score(self, outputs, labels):
        """
        Get evaluation score for a prediction.

        :param outputs: (tensor) Prediction obtained by a network with size N X C. 
        :param labels: (tensor) The ground truth with size N X C.
        """
        metrics = self.config['training'].get("evaluation_metric", "accuracy")
        if(metrics != "accuracy"): # default classification accuracy
            raise ValueError("Not implemeted for metric {0:}".format(metrics))
        if(self.task_type == TaskType.CLASSIFICATION_ONE_HOT):
            out_argmax = torch.argmax(outputs, 1)
            lab_argmax = torch.argmax(labels, 1)
            consis = self.convert_tensor_type(out_argmax ==  lab_argmax)
            score  = torch.mean(consis) 
        elif(self.task_type == TaskType.CLASSIFICATION_COEXIST):
            preds = self.convert_tensor_type(outputs > 0.5)
            consis= self.convert_tensor_type(preds ==  labels.data)
            score = torch.mean(consis) 
        return score

    def training(self):
        iter_valid   = self.config['training']['iter_valid']
        mixup_prob   = self.config['training'].get('mixup_probability', 0.5)
        sample_num   = 0
        running_loss = 0
        running_score= 0
        self.net.train()
        for it in range(iter_valid):
            try:
                data = next(self.trainIter)
            except StopIteration:
                self.trainIter = iter(self.train_loader)
                data = next(self.trainIter)
            inputs = self.convert_tensor_type(data['image'])
            labels = self.convert_tensor_type(data['label_prob'])  
            if(random() < mixup_prob):
                inputs, labels = mixup(inputs, labels)    
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward + backward + optimize
            outputs = self.net(inputs)
            
            loss = self.get_loss_value(data, outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            # statistics
            sample_num   += labels.size(0)
            running_loss += loss.item() * labels.size(0)
            running_score+= self.get_evaluation_score(outputs, labels) * labels.size(0)

        avg_loss = running_loss / sample_num
        avg_score= running_score.double() / sample_num
        metrics =self.config['training'].get("evaluation_metric", "accuracy")
        train_scalers = {'loss': avg_loss, metrics: avg_score}
        return train_scalers

    def validation(self):
        validIter  = iter(self.valid_loader)
        sample_num   = 0
        running_loss = 0
        running_score= 0
        with torch.no_grad():
            self.net.eval()
            for data in validIter:
                inputs = self.convert_tensor_type(data['image'])
                labels = self.convert_tensor_type(data['label_prob'])            
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.get_loss_value(data, outputs, labels)
                                
                # statistics
                sample_num   += labels.size(0)
                running_loss += loss.item() * labels.size(0)
                running_score+= self.get_evaluation_score(outputs, labels) * labels.size(0)

        avg_loss = running_loss / sample_num
        avg_score= running_score.double() / sample_num
        metrics  = self.config['training'].get("evaluation_metric", "accuracy")
        valid_scalers = {'loss': avg_loss, metrics: avg_score}
        return valid_scalers

    def write_scalars(self, train_scalars, valid_scalars, lr_value, glob_it):
        metrics = self.config['training'].get("evaluation_metric", "accuracy")
        loss_scalar ={'train':train_scalars['loss'], 'valid':valid_scalars['loss']}
        acc_scalar  ={'train':train_scalars[metrics],'valid':valid_scalars[metrics]}
        self.summ_writer.add_scalars('loss', loss_scalar, glob_it)
        self.summ_writer.add_scalars(metrics, acc_scalar, glob_it)
        self.summ_writer.add_scalars('lr', {"lr": lr_value}, glob_it)
        
        logging.info('train loss {0:.4f}, avg {1:} {2:.4f}'.format(
            train_scalars['loss'], metrics, train_scalars[metrics]))
        logging.info('valid loss {0:.4f}, avg {1:} {2:.4f}'.format(
            valid_scalars['loss'], metrics, valid_scalars[metrics])) 

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

        ckpt_dir    = self.config['training']['ckpt_dir']
        if(ckpt_dir[-1] == "/"):
            ckpt_dir = ckpt_dir[:-1]
        ckpt_prefix = self.config['training'].get('ckpt_prefix', None)
        if(ckpt_prefix is None):
            ckpt_prefix = ckpt_dir.split('/')[-1]
        iter_start  = 0
        iter_max    = self.config['training']['iter_max']
        iter_valid  = self.config['training']['iter_valid']
        iter_save   = self.config['training']['iter_save']
        early_stop_it = self.config['training'].get('early_stop_patience', None)
        metrics     = self.config['training'].get("evaluation_metric", "accuracy")
        if(iter_save is None):
            iter_save_list = [iter_max]
        elif(isinstance(iter_save, (tuple, list))):
            iter_save_list = iter_save
        else:
            iter_save_list = range(0, iter_max + 1, iter_save)
        
        self.max_val_score  = 0.0
        self.max_val_it     = 0
        self.best_model_wts = None 
        ckpt_init_name = self.config['training'].get('ckpt_init_name', None)
        ckpt_init_mode = self.config['training'].get('ckpt_init_mode', 0)

        if(ckpt_init_name is not None):
            checkpoint = torch.load(ckpt_dir + "/" + ckpt_init_name, map_location = self.device)
            pretrained_dict = checkpoint['model_state_dict']
            self.load_pretrained_weights(self.net, pretrained_dict, device_ids)
            if(ckpt_init_mode > 0): # Load  other information
                iter_start = checkpoint['iteration']
                self.max_val_score  = checkpoint.get('valid_pred', 0)
                self.max_val_it     = checkpoint['iteration']
                self.best_model_wts = checkpoint['model_state_dict']
        
        self.create_optimizer(self.get_parameters_to_update())
        self.create_loss_calculator()

        self.trainIter  = iter(self.train_loader)

        logging.info("{0:} training start".format(str(datetime.now())[:-7]))
        self.summ_writer = SummaryWriter(self.config['training']['ckpt_dir'])
        self.glob_it = iter_start
        for it in range(iter_start, iter_max, iter_valid):
            lr_value = self.optimizer.param_groups[0]['lr']
            t0 = time.time()
            train_scalars = self.training()
            t1 = time.time()
            valid_scalars = self.validation()
            
            t2 = time.time()
            if(isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau)):
                self.scheduler.step(valid_scalars[metrics])
            else:
                self.scheduler.step()

            self.glob_it = it + iter_valid
            logging.info("\n{0:} it {1:}".format(str(datetime.now())[:-7], self.glob_it))
            logging.info('learning rate {0:}'.format(lr_value))
            logging.info("training/validation time: {0:.2f}s/{1:.2f}s".format(t1-t0, t2-t1))
            self.write_scalars(train_scalars, valid_scalars, lr_value, self.glob_it)
            if(valid_scalars[metrics] > self.max_val_score):
                self.max_val_score = valid_scalars[metrics]
                self.max_val_it    = self.glob_it
                if(len(device_ids) > 1):
                    self.best_model_wts = copy.deepcopy(self.net.module.state_dict())
                else:
                    self.best_model_wts = copy.deepcopy(self.net.state_dict())
                save_dict = {'iteration': self.max_val_it,
                    'valid_pred': self.max_val_score,
                    'model_state_dict': self.best_model_wts,
                    'optimizer_state_dict': self.optimizer.state_dict()}
                save_name = "{0:}/{1:}_best.pt".format(ckpt_dir, ckpt_prefix)
                torch.save(save_dict, save_name) 
                txt_file = open("{0:}/{1:}_best.txt".format(ckpt_dir, ckpt_prefix), 'wt')
                txt_file.write(str(self.max_val_it))
                txt_file.close()
            
            stop_now = True if(early_stop_it is not None and \
                self.glob_it - self.max_val_it > early_stop_it) else False

            if ((self.glob_it in iter_save_list) or stop_now):
                save_dict = {'iteration': self.glob_it,
                             'valid_pred': valid_scalars[metrics],
                             'model_state_dict': self.net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict()}
                save_name = "{0:}/{1:}_{2:}.pt".format(ckpt_dir, ckpt_prefix, self.glob_it)
                torch.save(save_dict, save_name) 
                txt_file = open("{0:}/{1:}_latest.txt".format(ckpt_dir, ckpt_prefix), 'wt')
                txt_file.write(str(self.glob_it))
                txt_file.close()
            if(stop_now):
                logging.info("The training is early stopped")
                break
        logging.info('The best perfroming iter is {0:}, valid {1:} {2:}'.format(\
            self.max_val_it, metrics, self.max_val_score))
        self.summ_writer.close()

    def infer(self):
        device_ids = self.config['testing']['gpus']
        device = torch.device("cuda:{0:}".format(device_ids[0]))
        self.net.to(device)
        # load network parameters and set the network as evaluation mode
        checkpoint_name = self.get_checkpoint_name()
        checkpoint = torch.load(checkpoint_name, map_location = device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        
        if(self.config['testing'].get('evaluation_mode', True)):
            self.net.eval()
        
        output_csv   = self.config['testing']['output_dir'] + '/' + self.config['testing']['output_csv']
        class_num    = self.config['network']['class_num']
        save_probability = self.config['testing'].get('save_probability', False)
        
        infer_time_list = []
        out_prob_list   = []
        out_lab_list    = []
        with torch.no_grad():
            for data in self.test_loader:
                names  = data['names']
                inputs = self.convert_tensor_type(data['image'])
                inputs = inputs.to(device) 
                
                start_time = time.time()
                out_digit = self.net(inputs)
                infer_time = time.time() - start_time
                infer_time_list.append(infer_time)

                if (self.task_type == TaskType.CLASSIFICATION_ONE_HOT):
                    out_prob  = nn.Softmax(dim = 1)(out_digit).detach().cpu().numpy()
                    out_lab   = np.argmax(out_prob, axis=1)
                else: #self.task_type == TaskType.CLASSIFICATION_COEXIST
                    out_prob  = nn.Sigmoid()(out_digit).detach().cpu().numpy() 
                    out_lab   = np.asarray(out_prob > 0.5, np.uint8)              
                for i in range(len(names)):
                    print(names[i], out_lab[i])
                    if(self.task_type == TaskType.CLASSIFICATION_ONE_HOT):
                        out_lab_list.append([names[i]] + [out_lab[i]])
                    else:
                        out_lab_list.append([names[i]] + out_lab[i].tolist())
                    out_prob_list.append([names[i]] + out_prob[i].tolist())
        
        with open(output_csv, mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', 
                                quotechar='"',quoting=csv.QUOTE_MINIMAL)
            head = ['image', 'label']
            if(len(out_lab_list[0]) > 2):
                head = ['image'] + ['label{0:}'.format(i) for i in range(class_num)]
            csv_writer.writerow(head)
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

