
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging 
import os
import sys
from pymic.util.parse_config import *
from pymic.net_run_nll.nll_co_teaching import NLLCoTeaching
from pymic.net_run_nll.nll_trinet import NLLTriNet
from pymic.net_run_nll.nll_dast import NLLDAST

NLLMethodDict = {'CoTeaching': NLLCoTeaching,
    "TriNet": NLLTriNet,
    "DAST": NLLDAST}

def main():
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print('   pymic_nll train config.cfg')
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
    nll_method = config['noisy_label_learning']['nll_method']
    agent = NLLMethodDict[nll_method](config, stage)
    agent.run()

if __name__ == "__main__":
    main()

    