# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import csv 
import sys
import math
import pandas as pd
import random
import configparser
import numpy as np
from scipy import ndimage
from pymic.io.image_read_write import *
from pymic.util.image_process import *
from pymic.util.parse_config import parse_config


def evaluation(config_file):
    config = parse_config(config_file)['evaluation']
    metric = config['metric']
    gt_csv  = config['ground_truth_csv']
    lab_csv = config['prediction_csv']
    gt_items  = pd.read_csv(gt_csv)
    lab_items = pd.read_csv(lab_csv)
    assert(len(gt_items) == len(lab_items))
    for i in range(len(gt_items)):
        assert(gt_items.iloc[i, 0] == lab_items.iloc[i, 0])
    gt_data  = np.asarray(gt_items.iloc[:, -1])
    lab_data = np.asarray(lab_items.iloc[:, -1])
    correct_pred = gt_data == lab_data
    acc = (correct_pred.sum() + 0.0 ) / len(gt_items)
    print("accuracy {}".format(acc))


def main():
    if(len(sys.argv) < 2):
        print('Number of arguments should be 2. e.g.')
        print('    python pyMIC.util/evaluation.py config.cfg')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    evaluation(config_file)
    
if __name__ == '__main__':
    main()
