# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
import os
import sys
from datetime import datetime
from pymic import TaskType
from pymic.util.parse_config import *
from pymic.net_run.agent_cls import ClassificationAgent
from pymic.net_run.agent_seg import SegmentationAgent
from pymic.net_run.agent_rec import ReconstructionAgent

def main():
    """
    The main function for running a network for training or inference.
    """
    if(len(sys.argv) < 2):
        print('Number of arguments should be 2. e.g.')
        print('   pymic_test config.cfg')
        exit()
    cfg_file = str(sys.argv[1])
    if(not os.path.isfile(cfg_file)):
        raise ValueError("The config file does not exist: " + cfg_file)
    config   = parse_config(cfg_file)
    config   = synchronize_config(config)
    log_dir  = config['testing']['output_dir']
    if(not os.path.exists(log_dir)):
        os.makedirs(log_dir, exist_ok=True)

    if sys.version.startswith("3.9"):
        logging.basicConfig(filename=log_dir+"/log_test.txt", 
                            level=logging.INFO, format='%(message)s', force=True) # for python 3.9
    else:
        logging.basicConfig(filename=log_dir+"/log_test.txt", 
                            level=logging.INFO, format='%(message)s') # for python 3.6
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging_config(config)
    task    = config['dataset']['task_type']
    if(task == TaskType.CLASSIFICATION_ONE_HOT or task == TaskType.CLASSIFICATION_COEXIST):
        agent = ClassificationAgent(config, 'test')
    elif(task == TaskType.SEGMENTATION):
        agent = SegmentationAgent(config, 'test')
    elif(task == TaskType.RECONSTRUCTION):
        agent = ReconstructionAgent(config, 'test')
    else:
        raise ValueError("Undefined task for inference: {0:}".format(task))
    agent.run()

if __name__ == "__main__":
    main()
    

