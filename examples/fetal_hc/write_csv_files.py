"""Script for writing cvs files
"""

import os
import csv
import pandas as pd
import random
from random import shuffle

def create_csv_file(data_root, output_file):
    """
    create a csv file to store the paths of files for each patient
    """
    image_folder = data_root + "/" + "training_set"
    label_folder = data_root + "/" + "training_set_label"
    filenames = os.listdir(label_folder)
    filenames = [item for item in filenames if item[0] != '.']
    file_list = []
    for filename in filenames:
        image_name = "training_set" + "/" + filename.replace("_seg.", ".")
        label_name = "training_set_label" + "/" + filename
        file_list.append([image_name, label_name])
    
    with open(output_file, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', 
                            quotechar='"',quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["image", "label"])
        for item in file_list:
            csv_writer.writerow(item)

def random_split_dataset():
    random.seed(2019)
    input_file = 'config/fetal_hc_all.csv'
    train_names_file = 'config/fetal_hc_train.csv'
    valid_names_file = 'config/fetal_hc_valid.csv'
    test_names_file  = 'config/fetal_hc_test.csv'
    with open(input_file, 'r') as f:
        lines = f.readlines()
    data_lines = lines[1:]
    shuffle(data_lines)
    train_lines  = data_lines[:780]
    valid_lines  = data_lines[780:850]
    test_lines   = data_lines[850:]
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
            seg_name = seg_name.replace('_seg.', '.')
            output_lines.append([gt_name, seg_name])
    with open(gt_seg_csv, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', 
                            quotechar='"',quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["ground_truth", "segmentation"])
        for item in output_lines:
            csv_writer.writerow(item)

if __name__ == "__main__":
    # create cvs file for training set
    HC_root     = '/home/disk2t/data/Fetal_HC'
    output_file = 'config/fetal_hc_all.csv'
    create_csv_file(HC_root, output_file)

    # split fetal_hc training_set in to training, validation and testing
    random_split_dataset()

    # obtain ground truth and segmentation pairs for evaluation
    test_csv    = "./config/fetal_hc_test.csv"
    gt_seg_csv  = "./config/fetal_hc_test_gt_seg.csv"
    get_evaluation_image_pairs(test_csv, gt_seg_csv)
