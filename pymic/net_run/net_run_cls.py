# -*- coding: utf-8 -*-
from __future__ import print_function, division
import sys
from pymic.util.parse_config import parse_config
from pymic.net_run.agent_cls import ClassificationAgent

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
    

