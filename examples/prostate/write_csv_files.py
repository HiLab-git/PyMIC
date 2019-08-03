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

def obtain_patient_names():
    """
    extract the patient names from csv files
    """
    split_names = ['train', 'valid', 'test']
    for split_name in split_names:
        csv_file = 'config/data/image_{0:}.csv'.format(split_name)
        if(not os.path.isfile(csv_file)):
            continue
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        data_lines = lines[1:]
        patient_names = []
        for data_line in data_lines:
            patient_name = data_line.split('/')[-1]
            patient_name = patient_name.split('.')[0]
            print(patient_name)
            patient_names.append(patient_name)
        output_filename = 'config/data/image_{0:}_names.txt'.format(split_name)
        with open(output_filename, 'w') as f:
            for patient_name in patient_names:
                f.write('{0:}\n'.format(patient_name))
        
if __name__ == "__main__":
    # create cvs file for promise 2012
    fields      = ['image', 'label']
    data_dir    = 'data/promise12/preprocess'
    output_file = 'config/data/image_all.csv'
    create_csv_file(data_dir, output_file, fields)

    # split the data into training, validation and testing
    random_split_dataset()

    #obtain image names the splitted dataset
    obtain_patient_names()
