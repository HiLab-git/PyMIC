# -*- coding: utf-8 -*-
from __future__ import print_function, division

import copy
import csv
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
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from PIL import Image 
from scipy import special
from datetime import datetime
from tensorboardX import SummaryWriter
from pymic.io.image_read_write import save_nd_array_as_image
from pymic.io.nifty_dataset import ClassificationDataset
from pymic.transform.trans_dict import TransformDict
from pymic.net.cls.cls_net_dict import ClsNetDict
from pymic.net_run.get_optimizer import get_optimiser
from pymic.loss.cls_loss.cls_loss_dict import ClsLossDict
from pymic.loss.util import get_soft_label
from pymic.loss.util import reshape_prediction_and_ground_truth
from pymic.loss.util import get_classwise_dice
from pymic.util.image_process import convert_label
from pymic.util.parse_config import parse_config
from pymic.net_run.net_run_agent import NetRunAgent
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

class ClassificationAgent(NetRunAgent):
    def __init__(self, config, stage = 'train'):
        super(ClassificationAgent, self).__init__(config, stage)
        self.net_dict  = ClsNetDict
        self.loss_dict = ClsLossDict

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

    def train(self):
        class_num   = self.config['network']['class_num']
        summ_writer = SummaryWriter(self.config['training']['summary_dir'])
        chpt_prefx  = self.config['training']['checkpoint_prefix']
        epoch_start = self.config['training']['epoch_start']
        epoch_end   = self.config['training']['epoch_end']
        epoch_save  = self.config['training']['epoch_save']
        device = torch.device(self.config['training']['device_name']) 

        sampleweight_key = self.config['training']['loss_type'] + "_enable_sample_weight"
        sampleweight_enabled = self.config['training'][sampleweight_key.lower()]
        class_weight = self.config['training'].get('class_weight', None)
        if(class_weight is not None):
            assert(len(class_weight) == class_num)
            class_weight = torch.from_numpy(np.asarray(class_weight))
            class_weight = self.convert_tensor_type(class_weight)
            class_weight = class_weight.to(device)

        # Initialize state dict, optimizer, and loss 
        if(epoch_start > 0):
            checkpoint_file = "{0:}_{1:}.pt".format(chpt_prefx, epoch_start)
            self.checkpoint = torch.load(checkpoint_file)
            assert(self.checkpoint['iteration'] == epoch_start)
            self.net.load_state_dict(self.checkpoint['model_state_dict'])
        else:
            self.checkpoint = None
        self.create_optimizer() 
        loss_name = self.config['training']['loss_type']
        if(loss_name not in self.loss_dict):
            raise ValueError("Undefined loss function {0:}".format(loss_name))
        self.loss_calculater = self.loss_dict[loss_name](self.config['training'])
    
        self.net.to(device)
        max_val_acc = 0.0
        max_epoch = 0
        best_model_wts = copy.deepcopy(self.net.state_dict())
        for epoch in range(epoch_start, epoch_end):
            print('-' * 10)
            print('{}, Epoch {}/{}'.format(str(datetime.now())[:-7],epoch, epoch_end - 1))
            
            for phase in ['train', 'valid']:
                if (phase == 'train'):
                    self.net.train()
                    data_loader = self.train_loader
                else:
                    self.net.eval()
                    data_loader = self.valid_loader

                loss_sum, correct_sum = 0, 0
                step_num, sample_num  = 0, 0

                for data in data_loader:
                    step_num = step_num + 1
                    inputs = self.convert_tensor_type(data['image'])
                    labels = self.convert_tensor_type(data['label_prob'])          
                    inputs, labels = inputs.to(device), labels.to(device)
                    if(sampleweight_enabled):
                        sample_w = self.convert_tensor_type(data['weight'])
                        sample_w = sample_w.to(device)
                    else:
                        sample_w = None

                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        # forward + backward + optimize
                        outputs = self.net(inputs)
                        loss_input_dict = {'prediction':outputs, 'ground_truth':labels,
                            'sample_weight': sample_w, 'class_weight': class_weight, 'softmax': True}
                        loss   = self.loss_calculater(loss_input_dict)
                        if(phase == 'train'):
                            loss.backward()
                            self.optimizer.step()
                    
                    # get accuracy 
                    output_idx = torch.argmax(outputs, dim = 1, keepdim = False)
                    labels_idx = torch.argmax(labels, dim = 1, keepdim = False)
                    correct_sum += torch.sum(output_idx == labels_idx)
                    loss_sum    += loss.item() * inputs.size(0)
                    sample_num  += inputs.size(0)

                avg_loss = loss_sum / sample_num
                acc = (correct_sum + 0.0) / sample_num

                if(phase == 'train'):
                    train_loss, train_acc = avg_loss, acc
                else:
                    valid_loss, valid_acc = avg_loss, acc
                    if(valid_acc > max_val_acc):
                        max_val_acc = valid_acc
                        max_epoch = epoch
                        best_model_wts = copy.deepcopy(self.net.state_dict())
            self.schedule.step()
            print('Train loss: {:.4f}, Acc: {:.4f}'.format(train_loss, train_acc))
            print('Valid loss: {:.4f}, Acc: {:.4f}'.format(valid_loss, valid_acc))
            
            loss_scalers = {'train': train_loss, 'valid': valid_loss}
            summ_writer.add_scalars('loss', loss_scalers, epoch + 1)
            acc_scalers = {'train': train_acc, 'valid': valid_acc}
            summ_writer.add_scalars('accuracy', acc_scalers, epoch + 1)
        
            if (epoch % epoch_save ==  epoch_save - 1):
                save_dict = {'epoch': epoch + 1,
                            'model_state_dict': self.net.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()}
                save_name = "{0:}_{1:}.pt".format(chpt_prefx, epoch + 1)
                torch.save(save_dict, save_name)  
        # save the best model
        save_dict =  {'epoch': max_epoch + 1,
                      'model_state_dict': best_model_wts} 
        save_name = "{0:}_{1:}.pt".format(chpt_prefx, max_epoch + 1)
        torch.save(save_dict, save_name)  
        summ_writer.close()
        print("Best performing epoch: {}, valid acc {}".format(max_epoch + 1, max_val_acc))

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

def main():
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print('    python train_infer.py train config.cfg')
        exit()
    stage    = str(sys.argv[1])
    cfg_file = str(sys.argv[2])
    config   = parse_config(cfg_file)
    agent    = ClassificationAgent(config, stage)
    agent.run()

if __name__ == "__main__":
    main()