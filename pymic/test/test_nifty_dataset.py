# -*- coding: utf-8 -*-
from __future__ import print_function, division
import sys
import numpy as np 
from pymic.util.parse_config import *
from pymic.net_run.agent_cls import ClassificationAgent
from pymic.net_run.agent_seg import SegmentationAgent
import SimpleITK as sitk

def save_array_as_nifty_volume(data, image_name, reference_name = None):
    """
    Save a numpy array as nifty image

    :param data:  (numpy.ndarray) A numpy array with shape [Depth, Height, Width].
    :param image_name:  (str) The ouput file name.
    :param reference_name:  (str) File name of the reference image of which 
        meta information is used.
    """
    img = sitk.GetImageFromArray(data)
    if(reference_name is not None):
        img_ref = sitk.ReadImage(reference_name)
        #img.CopyInformation(img_ref)
        img.SetSpacing(img_ref.GetSpacing())
        img.SetOrigin(img_ref.GetOrigin())
        img.SetDirection(img_ref.GetDirection())
    sitk.WriteImage(img, image_name)

def main():
    """
    The main function for running a network for training or inference.
    """
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print('python test_nifty_dataset.py train config.cfg')
        exit()
    stage    = str(sys.argv[1])
    cfg_file = str(sys.argv[2])
    config   = parse_config(cfg_file)
    config   = synchronize_config(config)
    # task     = config['dataset']['task_type']
    # assert task in ['cls', 'cls_nexcl', 'seg']
    # if(task == 'cls' or task == 'cls_nexcl'):
    #     agent = ClassificationAgent(config, stage)
    # else:
    #     agent = SegmentationAgent(config, stage)
    agent = SegmentationAgent(config, stage)
    agent.create_dataset()
    data_loader = agent.train_loader if stage == "train" else agent.test_loader
    it = 0
    for data in data_loader:
        inputs      = agent.convert_tensor_type(data['image'])
        labels_prob = agent.convert_tensor_type(data['label_prob']) 
        for i in range(inputs.shape[0]):
            image_i = inputs[i][0]
            label_i = np.argmax(labels_prob[i], axis = 0)
            print(image_i.shape, label_i.shape)
            image_name = "temp/image_{0:}_{1:}.nii.gz".format(it, i)
            label_name = "temp/label_{0:}_{1:}.nii.gz".format(it, i)
            save_array_as_nifty_volume(image_i, image_name, reference_name = None)
            save_array_as_nifty_volume(label_i, label_name, reference_name = None)
        it = it + 1
        if(it == 10):
            break

if __name__ == "__main__":
    main()
    

