# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import sys
from datetime import datetime
from pymic.util.parse_config import *
from pymic.net_run.agent_preprocess import PreprocessAgent


def main():
    """
    The main function for data preprocessing.
    """
    if(len(sys.argv) < 2):
        print('Number of arguments should be 2. e.g.')
        print('   pymic_preprocess config.cfg')
        exit()
    cfg_file = str(sys.argv[1])
    if(not os.path.isfile(cfg_file)):
        raise ValueError("The config file does not exist: " + cfg_file)
    config = parse_config(cfg_file)
    config = synchronize_config(config)
    agent  = PreprocessAgent(config)
    agent.run()

if __name__ == "__main__":
    main()
    
    

