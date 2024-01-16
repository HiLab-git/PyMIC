# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from pymic import TaskDict
import configparser
import logging

def is_int(val_str):
    start_digit = 0
    if(val_str[0] =='-'):
        start_digit = 1
    flag = True
    for i in range(start_digit, len(val_str)):
        if(str(val_str[i]) < '0' or str(val_str[i]) > '9'):
            flag = False
            break
    return flag

def is_float(val_str):
    flag = False
    if('.' in val_str and len(val_str.split('.'))==2 and not('./' in val_str)):
        if(is_int(val_str.split('.')[0]) and is_int(val_str.split('.')[1])):
            flag = True
        else:
            flag = False
    elif('e' in val_str and val_str[0] != 'e' and len(val_str.split('e'))==2):
        if(is_int(val_str.split('e')[0]) and is_int(val_str.split('e')[1])):
            flag = True
        else:
            flag = False       
    else:
        flag = False
    return flag 

def is_bool(var_str):
    if( var_str.lower() =='true' or var_str.lower() == 'false'):
        return True
    else:
        return False
    
def parse_bool(var_str):
    if(var_str.lower() =='true'):
        return True
    else:
        return False
     
def is_list(val_str):
    if(val_str[0] == '[' and val_str[-1] == ']'):
        return True
    else:
        return False
    
def parse_list(val_str):
    sub_str = val_str[1:-1]
    splits = sub_str.split(',')
    output = []
    for item in splits:
        item = item.strip()
        if(is_int(item)):
            output.append(int(item))
        elif(is_float(item)):
            output.append(float(item))
        elif(is_bool(item)):
            output.append(parse_bool(item))
        elif(item.lower() == 'none'):
            output.append(None)
        else:
            output.append(item)
    return output

def parse_value_from_string(val_str):
#     val_str = val_str.encode('ascii','ignore')
    if(is_int(val_str)):
        val = int(val_str)
    elif(is_float(val_str)):
        val = float(val_str)
    elif(is_list(val_str)):
        val = parse_list(val_str)
    elif(is_bool(val_str)):
        val = parse_bool(val_str)
    elif(val_str.lower() == 'none'):
        val = None
    else:
        val = val_str
    return val

def parse_config(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    output = {}
    for section in config.sections():
        output[section] = {}
        for key in config[section]:
            val_str = str(config[section][key])
            if(len(val_str)>0):
                val = parse_value_from_string(val_str)
                output[section][key] = val
            else:
                val = None
            print(section, key, val)
    return output
            
def synchronize_config(config):
    data_cfg = config['dataset'] 
    data_cfg["task_type"] = TaskDict[data_cfg["task_type"]]
    if('network' in config):
        net_cfg  = config['network']
    # data_cfg["modal_num"] = net_cfg["in_chns"]
        data_cfg["LabelToProbability_class_num".lower()] = net_cfg["class_num"] 
    transform = []
    if('transform' in data_cfg and data_cfg['transform'] is not None):
        transform.extend(data_cfg['transform'])
    if('train_transform' in data_cfg and data_cfg['train_transform'] is not None):
        transform.extend(data_cfg['train_transform'])
    if('valid_transform' in data_cfg and data_cfg['valid_transform'] is not None):
        transform.extend(data_cfg['valid_transform'])
    if('test_transform' in data_cfg and data_cfg['test_transform'] is not None):
        transform.extend(data_cfg['test_transform'])
    if ( "PartialLabelToProbability" in transform and 'network' in config):
        data_cfg["PartialLabelToProbability_class_num".lower()] = net_cfg["class_num"]
    patch_size = data_cfg.get('patch_size', None)
    if(patch_size is not None):
        if('Pad' in transform and 'Pad_output_size'.lower() not in data_cfg):
            data_cfg['Pad_output_size'.lower()] = patch_size
        if('CenterCrop' in transform and 'CenterCrop_output_size'.lower() not in data_cfg):
            data_cfg['CenterCrop_output_size'.lower()] = patch_size
        if('RandomCrop' in transform and 'RandomCrop_output_size'.lower() not in data_cfg):
            data_cfg['RandomCrop_output_size'.lower()] = patch_size
        if('RandomResizedCrop' in transform and \
            'RandomResizedCrop_output_size'.lower() not in data_cfg):
            data_cfg['RandomResizedCrop_output_size'.lower()] = patch_size
    config['dataset'] = data_cfg
    # config['network'] = net_cfg
    return config 

def logging_config(config):
    for section in config:
        if(isinstance(config[section], dict)):
            for key in config[section]:
                value = config[section][key]
                logging.info("{0:} {1:} = {2:}".format(section, key, value))
        else:
            logging.info("{0:} = {1:}".format(section, config[section]))

if __name__ == "__main__":
    print(is_int('555'))
    print(is_float('555.10'))
    a='[1 ,2 ,3 ]'
    print(a)
    print(parse_list(a))
    
    
