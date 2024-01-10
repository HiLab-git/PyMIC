# -*- coding: utf-8 -*-
from __future__ import print_function, division
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
        print('Number of arguments should be 2. e.g.')
        print('   pymic_train config.cfg')
        exit()
    cfg_file = str(sys.argv[1])
    if(not os.path.isfile(cfg_file)):
        raise ValueError("The config file does not exist: " + cfg_file)
    config   = parse_config(cfg_file)
    config   = synchronize_config(config)
    log_dir  = config['training']['ckpt_save_dir']
    if(not os.path.exists(log_dir)):
        os.makedirs(log_dir, exist_ok=True)
    dst_cfg = cfg_file if "/" not in cfg_file else cfg_file.split("/")[-1]
    shutil.copy(cfg_file, log_dir + "/" + dst_cfg)
    datetime_str = str(datetime.now())[:-7].replace(":", "_")
    if sys.version.startswith("3.9"):
        logging.basicConfig(filename=log_dir+"/log_train_{0:}.txt".format(datetime_str), 
                            level=logging.INFO, format='%(message)s', force=True) # for python 3.9
    else:
        logging.basicConfig(filename=log_dir+"/log_train_{0:}.txt".format(datetime_str), 
                            level=logging.INFO, format='%(message)s') # for python 3.6
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging_config(config)
    task     = config['dataset']['task_type']
    if(task == TaskType.CLASSIFICATION_ONE_HOT or task == TaskType.CLASSIFICATION_COEXIST):
        agent = ClassificationAgent(config, 'train')
    else:
        sup_type = config['dataset'].get('supervise_type', 'fully_sup')
        agent = get_seg_rec_agent(config, sup_type)

    agent.run()

if __name__ == "__main__":
    main()
    

