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
    patient_names = os.listdir(data_root + '/' + fields[1])
    patient_names.sort()
    print('total number of images {0:}'.format(len(patient_names)))
    for patient_name in patient_names:
        patient_image_names = []
        for field in fields:
            image_name = field + '/' + patient_name
            # if(field == 'image'):
            #     image_name = image_name.replace('_seg.', '.')
            #     #image_name = image_name[:-4]
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
    input_file = 'config/data/image_all.csv'
    train_names_file = 'config/data/image_train.csv'
    valid_names_file = 'config/data/image_valid.csv'
    test_names_file  = 'config/data/image_test.csv'
    with open(input_file, 'r') as f:
        lines = f.readlines()
    data_lines = lines[1:]
    shuffle(data_lines)
    N = len(data_lines)
    n1 = int(N * 0.7)
    n2 = int(N * 0.8)
    print('image number', N)
    print('training number', n1)
    print('validation number', n2 - n1)
    print('testing number', N - n2)
    train_lines  = data_lines[:n1]
    valid_lines  = data_lines[n1:n2]
    test_lines   = data_lines[n2:]
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
    # create cvs file for promise 2012
    fields      = ['image', 'label']
    data_dir    = '/home/disk2t/data/prostate/promise12/preprocess/train'
    output_file = 'config/data/image_all.csv'
    create_csv_file(data_dir, output_file, fields)

    # split the data into training, validation and testing
    random_split_dataset()

    # obtain ground truth and segmentation pairs for evaluation
    test_csv    = "./config/data/image_test.csv"
    gt_seg_csv  = "./config/data/image_test_gt_seg.csv"
    get_evaluation_image_pairs(test_csv, gt_seg_csv)

