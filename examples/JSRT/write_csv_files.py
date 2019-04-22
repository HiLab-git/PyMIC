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

def obtain_patient_names():
    """
    extract the patient names from csv files
    """
    split_names = ['train', 'valid', 'test']
    for split_name in split_names:
        csv_file = 'config/jsrt_{0:}.csv'.format(split_name)
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        data_lines = lines[1:]
        patient_names = []
        for data_line in data_lines:
            patient_name = data_line.split(',')[0]
            patient_name = patient_name[6:-4]
            print(patient_name)
            patient_names.append(patient_name)
        output_filename = 'config/jsrt_{0:}_names.txt'.format(split_name)
        with open(output_filename, 'w') as f:
            for patient_name in patient_names:
                f.write('{0:}\n'.format(patient_name))
        
if __name__ == "__main__":
    # create cvs file for JSRT dataset
    data_root   = '/home/guotai/data/JSRT'
    output_file = 'config/jsrt_all.csv'
    fields      = ['image', 'label']
    create_csv_file(data_root, output_file, fields)

    # split JSRT dataset in to training, validation and testing
    random_split_dataset()

    # obtain patient names the splitted dataset
    obtain_patient_names()
