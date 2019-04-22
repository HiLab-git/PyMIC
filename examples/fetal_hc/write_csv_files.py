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
    input_file = 'config/fetal_hc_train_all.csv'
    train_names_file = 'config/fetal_hc_train_train.csv'
    valid_names_file = 'config/fetal_hc_train_valid.csv'
    test_names_file  = 'config/fetal_hc_train_test.csv'
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

def obtain_patient_names():
    """
    extract the patient names from csv files
    """
    split_names = ['train', 'valid', 'test']
    for split_name in split_names:
        csv_file = 'config/fetal_hc_train_{0:}.csv'.format(split_name)
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        data_lines = lines[1:]
        patient_names = []
        for data_line in data_lines:
            patient_name = data_line.split(',')[0]
            patient_name = patient_name.split('/')[-1][:-4]
            print(patient_name)
            patient_names.append(patient_name)
        output_filename = 'config/fetal_hc_train_{0:}_patient.txt'.format(split_name)
        with open(output_filename, 'w') as f:
            for patient_name in patient_names:
                f.write('{0:}\n'.format(patient_name))
        
if __name__ == "__main__":
    # create cvs file for training set
    data_root   = '/home/guotai/data/Fetal_HC'
    output_file = 'config/fetal_hc_train_all.csv'
    create_csv_file(data_root, output_file)

    # # split fetal_hc training_set in to training, validation and testing
    random_split_dataset()

    # obtain patient names the splitted dataset
    obtain_patient_names()
