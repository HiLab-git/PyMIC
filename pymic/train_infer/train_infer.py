# -*- coding: utf-8 -*-
from __future__ import print_function, division

import sys
import time
import scipy
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from scipy import special
from datetime import datetime
from tensorboardX import SummaryWriter
from pymic.io.image_read_write import save_nd_array_as_image
from pymic.io.nifty_dataset import NiftyDataset
from pymic.io.transform3d import get_transform
from pymic.train_infer.net_factory import get_network
from pymic.train_infer.infer_func import volume_infer
from pymic.train_infer.loss import *
from pymic.train_infer.get_optimizer import get_optimiser
from pymic.util.image_process import convert_label
from pymic.util.parse_config import parse_config


class TrainInferAgent(object):
    def __init__(self, config, stage = 'train'):
        assert(stage in ['train', 'inference', 'test'])
        self.config = config
        self.stage  = stage
        if(stage == 'inference'):
            self.stage = 'test'
        self.net    = None
        self.train_set = None 
        self.valid_set = None 
        self.test_set  = None
        self.loss_calculater = None 
        self.tensor_type = config['dataset']['tensor_type']
        
    def set_datasets(self, train_set, valid_set, test_set):
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set  = test_set

    def set_network(self, net):
        self.net = net 

    def set_loss_calculater(self, loss_calculater):
        self.loss_calculater = loss_calculater

    def get_stage_dataset_from_config(self, stage):
        assert(stage in ['train', 'valid', 'test'])
        root_dir  = self.config['dataset']['root_dir']
        modal_num = self.config['dataset']['modal_num']
        if(stage == "train" or stage == "valid"):
            transform_names = self.config['dataset']['train_transform']
        elif(stage == "test"):
            transform_names = self.config['dataset']['test_transform']
        else:
            raise ValueError("Incorrect value for stage: {0:}".format(stage))

        self.transform_list = [get_transform(name, self.config['dataset']) \
                    for name in transform_names ]    
        csv_file = self.config['dataset'].get(stage + '_csv', None)
        dataset  = NiftyDataset(root_dir=root_dir,
                                csv_file  = csv_file,
                                modal_num = modal_num,
                                with_label= not (stage == 'test'),
                                transform = transforms.Compose(self.transform_list))
        return dataset

    def create_dataset(self):
        if(self.stage == 'train'):
            if(self.train_set is None):
                self.train_set = self.get_stage_dataset_from_config('train')
            if(self.valid_set is None):
                self.valid_set = self.get_stage_dataset_from_config('valid')

            batch_size = self.config['training']['batch_size']
            self.train_loader = torch.utils.data.DataLoader(self.train_set, 
                batch_size = batch_size, shuffle=True, num_workers=batch_size * 4)
            self.valid_loader = torch.utils.data.DataLoader(self.valid_set, 
                batch_size = batch_size, shuffle=False, num_workers=batch_size * 4)
        else:
            if(self.test_set  is None):
                self.test_set  = self.get_stage_dataset_from_config('test')
            batch_size = 1
            self.test_loder = torch.utils.data.DataLoader(self.test_set, 
                batch_size=batch_size, shuffle=False, num_workers=batch_size)

    def create_network(self):
        if(self.net is None):
            self.net = get_network(self.config['network'])
        if(self.tensor_type == 'float'):
            self.net.float()
        else:
            self.net.double()
        
    def create_optimizer(self):
        self.optimizer = get_optimiser(self.config['training']['optimizer'],
                self.net.parameters(), 
                self.config['training'])
        last_iter = -1
        if(self.checkpoint is not None):
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            last_iter = self.checkpoint['iteration'] - 1
        self.schedule = optim.lr_scheduler.MultiStepLR(self.optimizer,
                self.config['training']['lr_milestones'],
                self.config['training']['lr_gamma'],
                last_epoch = last_iter)

    def convert_tensor_type(self, input_tensor):
        if(self.tensor_type == 'float'):
            return input_tensor.float()
        else:
            return input_tensor.double()

    def train(self):
        device = torch.device(self.config['training']['device_name'])
        self.net.to(device)

        summ_writer = SummaryWriter(self.config['training']['summary_dir'])
        chpt_prefx  = self.config['training']['checkpoint_prefix']
        iter_start  = self.config['training']['iter_start']
        iter_max    = self.config['training']['iter_max']
        iter_valid  = self.config['training']['iter_valid']
        iter_save   = self.config['training']['iter_save']
        class_num   = self.config['network']['class_num']

        if(iter_start > 0):
            checkpoint_file = "{0:}_{1:}.pt".format(chpt_prefx, iter_start)
            self.checkpoint = torch.load(checkpoint_file)
            assert(self.checkpoint['iteration'] == iter_start)
            self.net.load_state_dict(self.checkpoint['model_state_dict'])
        else:
            self.checkpoint = None
        self.create_optimizer()

        train_loss      = 0
        train_dice_list = []
        if(self.loss_calculater is None):
            loss_func   = self.config['training']['loss_function']
            self.loss_calculater = SegmentationLossCalculator(loss_func, True)
        trainIter = iter(self.train_loader)
        print("{0:} training start".format(str(datetime.now())[:-7]))
        for it in range(iter_start, iter_max):
            try:
                data = next(trainIter)
            except StopIteration:
                trainIter = iter(self.train_loader)
                data = next(trainIter)
            
            # get the inputs
            inputs      = self.convert_tensor_type(data['image'])
            labels_prob = self.convert_tensor_type(data['label_prob']) 

            # # for debug
            # for i in range(inputs.shape[0]):
            #     image_i = inputs[i][0]
            #     label_i = labels[i][0]
            #     image_name = "temp/image_{0:}_{1:}.nii.gz".format(it, i)
            #     label_name = "temp/label_{0:}_{1:}.nii.gz".format(it, i)
            #     save_nd_array_as_image(image_i, image_name, reference_name = None)
            #     save_nd_array_as_image(label_i, label_name, reference_name = None)
            # continue
            inputs, labels_prob = inputs.to(device), labels_prob.to(device)
            
            # zero the parameter gradients
            self.optimizer.zero_grad()
            self.schedule.step()
                
            # forward + backward + optimize
            outputs = self.net(inputs)
            loss_input_dict = {'prediction':outputs, 'ground_truth':labels_prob}
            if ('label_distance' in data):
                label_distance = self.convert_tensor_type(data['label_distance'])
                loss_input_dict['label_distance'] = label_distance.to(device)
            loss   = self.loss_calculater.get_loss(loss_input_dict)
            # if (self.config['training']['use'])
            loss.backward()
            self.optimizer.step()

            # get dice evaluation for each class
            if(isinstance(outputs, tuple) or isinstance(outputs, list)):
                outputs = outputs[0] 
            outputs_argmax = torch.argmax(outputs, dim = 1, keepdim = True)
            soft_out       = get_soft_label(outputs_argmax, class_num, self.tensor_type)
            soft_out, labels_prob = reshape_prediction_and_ground_truth(soft_out, labels_prob) 
            dice_list = get_classwise_dice(soft_out, labels_prob)
            train_dice_list.append(dice_list.cpu().numpy())

            # evaluate performance on validation set
            train_loss = train_loss + loss.item()
            if (it % iter_valid == iter_valid - 1):
                train_avg_loss = train_loss / iter_valid
                train_cls_dice = np.asarray(train_dice_list).mean(axis = 0)
                train_avg_dice = train_cls_dice.mean()
                train_loss = 0.0
                train_dice_list = []

                valid_loss = 0.0
                valid_dice_list = []
                with torch.no_grad():
                    for data in self.valid_loader:
                        inputs      = self.convert_tensor_type(data['image'])
                        labels_prob = self.convert_tensor_type(data['label_prob'])
                        inputs, labels_prob = inputs.to(device), labels_prob.to(device)
                        outputs = self.net(inputs)
                        loss_input_dict = {'prediction':outputs, 'ground_truth':labels_prob}
                        if ('label_distance' in data):
                            label_distance = self.convert_tensor_type(data['label_distance'])
                            loss_input_dict['label_distance'] = label_distance.to(device)
                        loss   = self.loss_calculater.get_loss(loss_input_dict)
                        valid_loss = valid_loss + loss.item()

                        if(isinstance(outputs, tuple) or isinstance(outputs, list)):
                            outputs = outputs[0] 
                        outputs_argmax = torch.argmax(outputs, dim = 1, keepdim = True)
                        soft_out  = get_soft_label(outputs_argmax, class_num, self.tensor_type)
                        soft_out, labels_prob = reshape_prediction_and_ground_truth(soft_out, labels_prob) 
                        dice_list = get_classwise_dice(soft_out, labels_prob)
                        valid_dice_list.append(dice_list.cpu().numpy())

                valid_avg_loss = valid_loss / len(self.valid_loader)
                valid_cls_dice = np.asarray(valid_dice_list).mean(axis = 0)
                valid_avg_dice = valid_cls_dice.mean()
                loss_scalers = {'train': train_avg_loss, 'valid': valid_avg_loss}
                summ_writer.add_scalars('loss', loss_scalers, it + 1)
                dice_scalers = {'train': train_avg_dice, 'valid': valid_avg_dice}
                summ_writer.add_scalars('class_avg_dice', dice_scalers, it + 1)
                print('train cls dice', train_cls_dice.shape, train_cls_dice)
                print('valid cls dice', valid_cls_dice.shape, valid_cls_dice)
                for c in range(class_num):
                    dice_scalars = {'train':train_cls_dice[c], 'valid':valid_cls_dice[c]}
                    summ_writer.add_scalars('class_{0:}_dice'.format(c), dice_scalars, it + 1)
                
                print("{0:} it {1:}, loss {2:.4f}, {3:.4f}".format(
                    str(datetime.now())[:-7], it + 1, train_avg_loss, valid_avg_loss))
            if (it % iter_save ==  iter_save - 1):
                save_dict = {'iteration': it + 1,
                             'model_state_dict': self.net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict()}
                save_name = "{0:}_{1:}.pt".format(chpt_prefx, it + 1)
                torch.save(save_dict, save_name)    
        summ_writer.close()
    
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
        output_dir       = self.config['testing']['output_dir']
        save_probability = self.config['testing']['save_probability']
        label_source = self.config['testing']['label_source']
        label_target = self.config['testing']['label_target']
        class_num    = self.config['network']['class_num']
        mini_batch_size     = self.config['testing']['mini_batch_size']
        mini_patch_inshape  = self.config['testing']['mini_patch_shape']
        mini_patch_stride   = self.config['testing']['mini_patch_stride']
        filename_replace_source = self.config['testing']['filename_replace_source']
        filename_replace_target = self.config['testing']['filename_replace_target']
        mini_patch_outshape = None
        # automatically infer outupt shape
        if(mini_patch_inshape is not None):
            patch_inshape = [1, self.config['dataset']['modal_num']] + mini_patch_inshape
            testx = np.random.random(patch_inshape)
            testx = torch.from_numpy(testx)
            testx = torch.tensor(testx)
            testx = testx.to(device)
            testy = self.net(testx)
            if(isinstance(testy, tuple) or isinstance(testy, list)):
                testy = testy[0] 
            testy = testy.detach().cpu().numpy()
            mini_patch_outshape = testy.shape[2:]
            print('mini patch in shape', mini_patch_inshape)
            print('mini patch out shape', mini_patch_outshape)
        start_time = time.time()
        with torch.no_grad():
            for data in self.test_loder:
                images = self.convert_tensor_type(data['image'])
                images = data['image'].float()
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
                data['predict'] = volume_infer(images, self.net, device, class_num, 
                    mini_batch_size, mini_patch_inshape, mini_patch_outshape, mini_patch_stride)

                for i in reversed(range(len(self.transform_list))):
                    if (self.transform_list[i].inverse):
                        data = self.transform_list[i].inverse_transform_for_prediction(data) 
                output = np.argmax(data['predict'][0], axis = 0)
                output = np.asarray(output, np.uint8)

                if((label_source is not None) and (label_target is not None)):
                    output = convert_label(output, label_source, label_target)
                # save the output and (optionally) probability predictions
                root_dir  = self.config['dataset']['root_dir']
                save_name = names[0].split('/')[-1]
                if((filename_replace_source is  not None) and (filename_replace_target is not None)):
                    save_name = save_name.replace(filename_replace_source, filename_replace_target)
                save_name = "{0:}/{1:}".format(output_dir, save_name)
                save_nd_array_as_image(output, save_name, root_dir + '/' + names[0])
                if(save_probability):
                    save_name_split = save_name.split('.')
                    if('.nii.gz' in save_name):
                        save_prefix = '.'.join(save_name_split[:-2])
                        save_format = 'nii.gz'
                    else:
                        save_prefix = '.'.join(save_name_split[:-1])
                        save_format = save_name_split[-1]
                    prob = scipy.special.softmax(data['predict'][0],axis = 0)
                    class_num = prob.shape[0]
                    for c in range(0, class_num):
                        temp_prob = prob[c]
                        prob_save_name = "{0:}_prob_{1:}.{2:}".format(save_prefix, c, save_format)
                        if(len(temp_prob.shape) == 2):
                            temp_prob = np.asarray(temp_prob * 255, np.uint8)
                        save_nd_array_as_image(temp_prob, prob_save_name, root_dir + '/' + names[0])

        avg_time = (time.time() - start_time) / len(self.test_loder)
        print("average testing time {0:}".format(avg_time))

    def run(self):
        self.create_dataset()
        self.create_network()
        if(self.stage == 'train'):
            self.train()
        else:
            self.infer()

if __name__ == "__main__":
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print('    python train_infer.py train config.cfg')
        exit()
    stage    = str(sys.argv[1])
    cfg_file = str(sys.argv[2])
    config   = parse_config(cfg_file)
    agent    = TrainInferAgent(config, stage)
    agent.run()

