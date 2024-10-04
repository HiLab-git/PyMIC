# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import logging
import os
import sys
import shutil
from datetime import datetime
from pymic import TaskType
from pymic.util.parse_config import *
from pymic.net_run.agent_cls import ClassificationAgent
from pymic.net_run.agent_seg import SegmentationAgent
from pymic.net_run.agent_rec import ReconstructionAgent
from pymic.net_run.semi_sup import SSLMethodDict
from pymic.net_run.weak_sup import WSLMethodDict
from pymic.net_run.self_sup import SelfSupMethodDict
from pymic.net_run.noisy_label import NLLMethodDict

def get_seg_rec_agent(config, sup_type):
    assert(sup_type in ['fully_sup', 'semi_sup', 'self_sup', 'weak_sup', 'noisy_label'])
    if(sup_type == 'fully_sup'):
        logging.info("\n********** Fully Supervised Learning **********\n")
        if config['dataset']['task_type'] == TaskType.SEGMENTATION:
            agent = SegmentationAgent(config, 'train')
        else:
            agent = ReconstructionAgent(config, 'train')
    elif(sup_type == 'semi_sup'):
        logging.info("\n********** Semi Supervised Learning **********\n")
        method = config['semi_supervised_learning']['method_name']
        agent = SSLMethodDict[method](config, 'train')
    elif(sup_type == 'weak_sup'):
        logging.info("\n********** Weakly Supervised Learning **********\n")
        method = config['weakly_supervised_learning']['method_name']
        agent = WSLMethodDict[method](config, 'train')
    elif(sup_type == 'noisy_label'):
        logging.info("\n********** Noisy Label Learning **********\n")
        method = config['noisy_label_learning']['method_name']
        agent = NLLMethodDict[method](config, 'train')
    elif(sup_type == 'self_sup'):
        logging.info("\n********** Self Supervised Learning **********\n")
        method = config['self_supervised_learning']['method_name']
        agent = SelfSupMethodDict[method](config, 'train')
    else:
        raise ValueError("undefined supervision type: {0:}".format(sup_type))
    return agent

def main():
    """
    The main function for running a network for training.
    """
    if(len(sys.argv) < 2):
        print('Number of arguments should be at least 2. e.g.')
        print('   pymic_train config.cfg -train_csv train.csv')
        exit()
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", help="configuration file for training")
    parser.add_argument("-train_csv", help="the csv file for training images", 
                        required=False, default=None)
    parser.add_argument("-valid_csv", help="the csv file for validation images", 
                    required=False, default=None)
    parser.add_argument("-ckpt_dir", help="the output dir for trained model", 
                    required=False, default=None)
    parser.add_argument("-iter_max", help="the maximal iteration number for training", 
                    required=False, default=None)
    parser.add_argument("-gpus", help="the gpus for runing, e.g., [0]", 
                    required=False, default=None)
    args = parser.parse_args()
    if(not os.path.isfile(args.cfg)):
        raise ValueError("The config file does not exist: " + args.cfg)
    config   = parse_config(args)
    config   = synchronize_config(config)

    log_dir  = config['training']['ckpt_dir']
    if(not os.path.exists(log_dir)):
        os.makedirs(log_dir, exist_ok=True)
    datetime_str = str(datetime.now())[:-7].replace(":", "_")
    if sys.version.startswith("3.9"):
        logging.basicConfig(filename=log_dir+"/log_train_{0:}.txt".format(datetime_str), 
                            level=logging.INFO, format='%(message)s', force=True) # for python 3.9
    else:
        logging.basicConfig(filename=log_dir+"/log_train_{0:}.txt".format(datetime_str), 
                            level=logging.INFO, format='%(message)s') # for python 3.6
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    dst_cfg = args.cfg if "/" not in args.cfg else args.cfg.split("/")[-1]
    wrtie_config(config, log_dir + "/" + dst_cfg)

    task     = config['dataset']['task_type']
    if(task == TaskType.CLASSIFICATION_ONE_HOT or task == TaskType.CLASSIFICATION_COEXIST):
        agent = ClassificationAgent(config, 'train')
    else:
        sup_type = config['dataset'].get('supervise_type', 'fully_sup')
        agent = get_seg_rec_agent(config, sup_type)

    agent.run()

if __name__ == "__main__":
    main()
    

