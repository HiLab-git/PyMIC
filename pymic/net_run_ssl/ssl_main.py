
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging 
import os
import sys
from pymic.util.parse_config import *
from pymic.net_run_ssl.ssl_em import SSLEntropyMinimization
from pymic.net_run_ssl.ssl_mt import SSLMeanTeacher
from pymic.net_run_ssl.ssl_uamt import SSLUncertaintyAwareMeanTeacher
from pymic.net_run_ssl.ssl_cct import SSLCCT
from pymic.net_run_ssl.ssl_cps import SSLCPS
from pymic.net_run_ssl.ssl_urpc import SSLURPC


SSLMethodDict = {'EntropyMinimization': SSLEntropyMinimization,
    'MeanTeacher': SSLMeanTeacher,
    'UAMT': SSLUncertaintyAwareMeanTeacher,
    'CCT': SSLCCT,
    'CPS': SSLCPS,
    'URPC': SSLURPC}

def main():
    """
    Main function for running a semi-supervised method.
    """
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print('   pymic_ssl train config.cfg')
        exit()
    stage    = str(sys.argv[1])
    cfg_file = str(sys.argv[2])
    config   = parse_config(cfg_file)
    config   = synchronize_config(config)
    log_dir  = config['training']['ckpt_save_dir']
    if(not os.path.exists(log_dir)):
        os.mkdir(log_dir)
    logging.basicConfig(filename=log_dir+"/log_{0:}.txt".format(stage), level=logging.INFO,
                        format='%(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging_config(config)
    ssl_method = config['semi_supervised_learning']['ssl_method']
    agent = SSLMethodDict[ssl_method](config, stage)
    agent.run()

if __name__ == "__main__":
    main()

    