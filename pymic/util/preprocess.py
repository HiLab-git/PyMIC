import os
import numpy as np 
import SimpleITK as sitk 
from pymic.io.image_read_write import load_image_as_nd_array
from pymic.transform.trans_dict import TransformDict
from pymic.util.parse_config import parse_config

def get_transform_list(trans_config_file):
    config   = parse_config(trans_config_file)
    transform_list = []

    transform_param = config['dataset']
    transform_param['task'] = 'segmentation' 
    transform_names = config['dataset']['transform']
    for name in transform_names:
        print(name)
        if(name not in TransformDict):
            raise(ValueError("Undefined transform {0:}".format(name))) 
        one_transform = TransformDict[name](transform_param)
        transform_list.append(one_transform)
    return transform_list

def preprocess_with_transform(transforms, img_in_name, img_out_name, 
    lab_in_name = None, lab_out_name = None):
    """
    using data transforms for preprocessing, such as image normalization, 
    cropping, etc. 
    TODO: support multip-modality preprocessing.
    """
    image_dict = load_image_as_nd_array(img_in_name)
    sample = {'image': np.asarray(image_dict['data_array'], np.float32), 
            'origin':image_dict['origin'],
            'spacing': image_dict['spacing'],
            'direction':image_dict['direction']}
    if(lab_in_name is not None):
        label_dict = load_image_as_nd_array(lab_in_name)
        sample['label'] = label_dict['data_array']
    for transform in transforms:
        sample = transform(sample)

    out_img = sitk.GetImageFromArray(sample['image'][0])
    out_img.SetSpacing(sample['spacing'])
    out_img.SetOrigin(sample['origin'])
    out_img.SetDirection(sample['direction'])
    sitk.WriteImage(out_img, img_out_name)
    if(lab_in_name is not None and lab_out_name is not None):
        out_lab = sitk.GetImageFromArray(sample['label'][0])
        out_lab.CopyInformation(out_img)
        sitk.WriteImage(out_lab, lab_out_name)



