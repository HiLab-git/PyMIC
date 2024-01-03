# -*- coding: utf-8 -*-
from __future__ import print_function, division
import copy
import logging
import time
import logging
import numpy as np
import os
import scipy 
import torch
import torch.nn as nn
from datetime import datetime
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from pymic.io.image_read_write import save_nd_array_as_image
from pymic.net_run.infer_func import Inferer
from pymic.net_run.agent_seg import SegmentationAgent
from pymic.loss.seg.mse import MAELoss, MSELoss
from pymic.util.general import mixup, tensor_shape_match

ReconstructionLossDict = {
    'MAELoss': MAELoss,
    'MSELoss': MSELoss
    }

class ReconstructionAgent(SegmentationAgent):
    """
    An agent for image reconstruction (pixel-level intensity prediction).
    """
    def __init__(self, config, stage = 'train'):
        super(ReconstructionAgent, self).__init__(config, stage)

    def create_loss_calculator(self):
        if(self.loss_dict is None):
            self.loss_dict = ReconstructionLossDict
        loss_name = self.config['training']['loss_type']
        if isinstance(loss_name, (list, tuple)):
            raise ValueError("Undefined loss function {0:}".format(loss_name))
        elif (loss_name not in self.loss_dict):
            raise ValueError("Undefined loss function {0:}".format(loss_name))
        else:
            loss_param = self.config['training']
            base_loss = self.loss_dict[loss_name](self.config['training'])
        if(self.config['training'].get('deep_supervise', False)):
            raise ValueError("Deep supervised loss not implemented for reconstruction tasks")
            # weight = self.config['training'].get('deep_supervise_weight', None)
            # mode   = self.config['training'].get('deep_supervise_mode', 2)
            # params = {'deep_supervise_weight': weight, 
            #           'deep_supervise_mode': mode, 
            #           'base_loss':base_loss}
            # self.loss_calculator = DeepSuperviseLoss(params)
        else:
            self.loss_calculator = base_loss

    def training(self):
        iter_valid  = self.config['training']['iter_valid']
        train_loss  = 0
        self.net.train()
        for it in range(iter_valid):
            try:
                data = next(self.trainIter)
            except StopIteration:
                self.trainIter = iter(self.train_loader)
                data = next(self.trainIter)
            # get the inputs
            inputs  = self.convert_tensor_type(data['image'])
            label   = self.convert_tensor_type(data['label'])                 
                   
            # for debug
            # from pymic.io.image_read_write import save_nd_array_as_image
            # print(inputs.shape)
            # for i in range(inputs.shape[0]):
            #     image_i = inputs[i][0]
            #     label_i = label[i][0]
            #     image_name = "temp/image_{0:}_{1:}.nii.gz".format(it, i)
            #     label_name = "temp/label_{0:}_{1:}.nii.gz".format(it, i)
            #     save_nd_array_as_image(image_i, image_name, reference_name = None)
            #     save_nd_array_as_image(label_i, label_name, reference_name = None)
            # if(it > 10):
            #     break
            # return

            inputs, label = inputs.to(self.device), label.to(self.device)
            
            # zero the parameter gradients
            self.optimizer.zero_grad()
                
            # forward + backward + optimize
            outputs = self.net(inputs)
            
            # for debug
            # if it < 5:
            #     outputs = nn.Tanh()(outputs)
            #     for i in range(inputs.shape[0]):
            #         out_name = "temp/output_{0:}_{1:}.nii.gz".format(it, i)
            #         output = outputs[i][0]
            #         output = output.cpu().detach().numpy()
            #         save_nd_array_as_image(output, out_name, reference_name = None)
            # else:
            #     break

            loss = self.get_loss_value(data, outputs, label)
            loss.backward()
            self.optimizer.step()
            train_loss = train_loss + loss.item()
            # get dice evaluation for each class
            if(isinstance(outputs, tuple) or isinstance(outputs, list)):
                outputs = outputs[0] 

        train_avg_loss = train_loss / iter_valid
        train_scalers = {'loss': train_avg_loss}
        return train_scalers
        
    def validation(self):
        class_num = self.config['network']['class_num']
        if(self.inferer is None):
            infer_cfg = self.config['testing']
            infer_cfg['class_num'] = class_num
            self.inferer = Inferer(infer_cfg)
        
        valid_loss_list = []
        validIter  = iter(self.valid_loader)
        with torch.no_grad():
            self.net.eval()

            # for debug
            # save_num = 0
            for data in validIter:
                inputs = self.convert_tensor_type(data['image'])
                label  = self.convert_tensor_type(data['label'])
                inputs, label  = inputs.to(self.device), label.to(self.device)
                outputs = self.inferer.run(self.net, inputs)
                # The tensors are on CPU when calculating loss for validation data
                loss = self.get_loss_value(data, outputs, label)
                valid_loss_list.append(loss.item())

                # for debug
                # print(inputs.shape, label.shape, outputs.shape)
                # inputs = inputs.cpu().numpy()
                # label  = label.cpu().numpy()
                # outputs = outputs.cpu().numpy()
                # for i in range(inputs.shape[0]):
                #     image_i = inputs[i][0]
                #     label_i = label[i][0]
                #     output_i  = outputs[i][0]
                #     image_name = "temp/case{0:}_image.nii.gz".format(save_num + i)
                #     label_name = "temp/case{0:}_label.nii.gz".format(save_num + i)
                #     output_name= "temp/case{0:}_output.nii.gz".format(save_num + i)
                #     save_nd_array_as_image(image_i, image_name, reference_name = None)
                #     save_nd_array_as_image(label_i, label_name, reference_name = None)
                #     save_nd_array_as_image(output_i, output_name, reference_name = None)
                # save_num += inputs.shape[0]
                # if(save_num > 20):
                #     break
        valid_avg_loss = np.asarray(valid_loss_list).mean()
        valid_scalers = {'loss': valid_avg_loss}
        return valid_scalers

    def write_scalars(self, train_scalars, valid_scalars, lr_value, glob_it):
        loss_scalar ={'train':train_scalars['loss'], 
                      'valid':valid_scalars['loss']}
        self.summ_writer.add_scalars('loss', loss_scalar, glob_it)
        self.summ_writer.add_scalars('lr', {"lr": lr_value}, glob_it)
        logging.info('train loss {0:.4f}'.format(train_scalars['loss']))        
        logging.info('valid loss {0:.4f}'.format(valid_scalars['loss']))  

    def train_valid(self):
        device_ids = self.config['training']['gpus']
        if(len(device_ids) > 1):
            self.device = torch.device("cuda:0")
            self.net = nn.DataParallel(self.net, device_ids = device_ids)
        else:
            self.device = torch.device("cuda:{0:}".format(device_ids[0]))
        self.net.to(self.device)

        ckpt_dir    = self.config['training']['ckpt_save_dir']
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

        self.min_val_loss = 10000.0
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
            model_dict = self.net.module.state_dict() if (len(device_ids) > 1) else self.net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if \
                k in model_dict and tensor_shape_match(pretrained_dict[k], model_dict[k])}
            logging.info("Initializing the following parameters with pre-trained model")
            for k in pretrained_dict:
                logging.info(k)
            if (len(device_ids) > 1):
                self.net.module.load_state_dict(pretrained_dict, strict = False)
            else:
                self.net.load_state_dict(pretrained_dict, strict = False)
            if(ckpt_init_mode > 0): # Load  other information
                self.min_val_loss = checkpoint.get('valid_loss', 10000)
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
                self.scheduler.step(-valid_scalars['loss'])
            else:
                self.scheduler.step()

            self.glob_it = it + iter_valid
            logging.info("\n{0:} it {1:}".format(str(datetime.now())[:-7], self.glob_it))
            logging.info('learning rate {0:}'.format(lr_value))
            logging.info("training/validation time: {0:.2f}s/{1:.2f}s".format(t1-t0, t2-t1))
            self.write_scalars(train_scalars, valid_scalars, lr_value, self.glob_it)
            if(valid_scalars['loss'] < self.min_val_loss):
                self.min_val_loss = valid_scalars['loss']
                self.max_val_it   = self.glob_it
                if(len(device_ids) > 1):
                    self.best_model_wts = copy.deepcopy(self.net.module.state_dict())
                else:
                    self.best_model_wts = copy.deepcopy(self.net.state_dict())
                
                save_dict = {'iteration': self.max_val_it,
                    'valid_loss': self.min_val_loss,
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
                             'valid_loss': valid_scalars['loss'],
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
        logging.info('The best performing iter is {0:}, valid loss {1:}'.format(\
            self.max_val_it, self.min_val_loss))
        self.summ_writer.close()
    
    def save_outputs(self, data):
        """
        Save prediction output. 

        :param data: (dictionary) A data dictionary with prediciton result and other 
            information such as input image name. 
        """
        output_dir = self.config['testing']['output_dir']
        ignore_dir = self.config['testing'].get('filename_ignore_dir', True)
        filename_replace_source = self.config['testing'].get('filename_replace_source', None)
        filename_replace_target = self.config['testing'].get('filename_replace_target', None)
        if(not os.path.exists(output_dir)):
            os.makedirs(output_dir, exist_ok=True)

        names, pred = data['names'], data['predict']
        if(isinstance(pred, (list, tuple))):
            pred =  pred[0]
        pred = np.tanh(pred)
        # pred = scipy.special.expit(pred)
        # save the output predictions
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
            save_nd_array_as_image(pred[i][i], save_name, test_dir + '/' + names[i][0])

            