"""Script for writing cvs files
"""

import os
import csv
import pandas as pd
import random
from random import shuffle

def create_csv_file(data_root, output_file, fields):
    """
    create a csv file to store the paths of files for each patient
    """
    filenames = []
    patient_names = os.listdir(data_root + '/' + fields[0])
    print(len(patient_names))
    for patient_name in patient_names:
        patient_image_names = []
        for field in fields:
            image_name = data_root + '/' + field + '/' + patient_name
            image_name = image_name[len(data_root) + 1 :]
            patient_image_names.append(image_name)
        filenames.append(patient_image_names)

    with open(output_file, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', 
                            quotechar='"',quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(fields)
        for item in filenames:
            csv_writer.writerow(item)

def random_split_dataset():
    random.seed(2019)
    input_file = 'config/jsrt_all.csv'
    train_names_file = 'config/jsrt_train.csv'
    valid_names_file = 'config/jsrt_valid.csv'
    test_names_file  = 'config/jsrt_test.csv'
    with open(input_file, 'r') as f:
        lines = f.readlines()
    data_lines = lines[1:]
    shuffle(data_lines)
    train_lines  = data_lines[:180]
    valid_lines  = data_lines[180:200]
    test_lines   = data_lines[200:247]
    with open(train_names_file, 'w') as f:
        f.writelines(lines[:1] + train_lines)
    with open(valid_names_file, 'w') as f:
        f.writelines(lines[:1] + valid_lines)
    with open(test_names_file, 'w') as f:
        f.writelines(lines[:1] + test_lines)

def get_evaluation_image_pairs(test_csv, gt_seg_csv):
    with open(test_csv, 'r') as f:
        input_lines = f.readlines()[1:]
        output_lines = []
        for item in input_lines:
            gt_name = item.split(',')[1]
            gt_name = gt_name.rstrip()
            seg_name = gt_name.split('/')[-1]
            output_lines.append([gt_name, seg_name])
    with open(gt_seg_csv, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', 
                            quotechar='"',quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["ground_truth", "segmentation"])
        for item in output_lines:
            csv_writer.writerow(item)

if __name__ == "__main__":
    # create cvs file for JSRT dataset
    JSRT_root   = '/home/disk2t/data/JSRT'
    output_file = 'config/jsrt_all.csv'
    fields      = ['image', 'label']
    create_csv_file(JSRT_root, output_file, fields)

    # split JSRT dataset in to training, validation and testing
    random_split_dataset()

    # obtain ground truth and segmentation pairs for evaluation
    test_csv    = "./config/jsrt_test.csv"
    gt_seg_csv  = "./config/jsrt_test_gt_seg.csv"
    get_evaluation_image_pairs(test_csv, gt_seg_csv)
