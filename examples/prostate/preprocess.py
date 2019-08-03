"""Script for preprocess
"""
import os
import sys
import numpy as np
import SimpleITK as sitk 

from scipy import ndimage
from pymic.util.image_process import *

def resize_ND_volume_to_given_shape(volume, out_shape, order = 3):
    shape0 = volume.shape
    assert(len(shape0) == len(out_shape))
    scale = [(out_shape[i] + 0.0)/shape0[i] for i in range(len(shape0))]
    return ndimage.interpolation.zoom(volume, scale, order = order)

def label_smooth(volume):
    [D, H, W] = volume.shape
    print(volume.shape)
    s = ndimage.generate_binary_structure(2,1)
    for d in range(D):
        if(volume[d].sum() > 0):
            volume_d = get_largest_component(volume[d])
            print(d, volume_d.sum())
            if(volume_d.sum() < 10):
                volume[d] = np.zeros_like(volume[d])
                continue
            volume_d = ndimage.morphology.binary_closing(volume_d, s)
            volume_d = ndimage.morphology.binary_opening(volume_d, s)
            volume[d] = volume_d
    return volume

def image_resample_crop(input_img_name, input_lab_name, output_img_name, output_lab_name):
    img_obj = sitk.ReadImage(input_img_name)
    origin  = img_obj.GetOrigin()
    spacing = img_obj.GetSpacing()
    direction = img_obj.GetDirection()
    img_data = sitk.GetArrayFromImage(img_obj)
    img_shape = img_data.shape

    if(len(img_shape) == 4):
        img_data = img_data[0]
        img_shape = img_data.shape 
        direction = np.asarray(direction)
        direction = np.reshape(direction, (4, 4))
        direction = direction[0:3, 0:3]
        direction = np.reshape(direction, 9)

    lab_obj  = sitk.ReadImage(input_lab_name)
    lab_data = sitk.GetArrayFromImage(lab_obj)
    lab_data = np.asarray(lab_data, np.float32)

    new_shape = [int(img_shape[i] * spacing[2-i]) for i in range(3)]
    out_data = resize_ND_volume_to_given_shape(img_data, new_shape, order = 3)
    
    out_lab = resize_ND_volume_to_given_shape(lab_data, new_shape, order = 3)
    out_lab = out_lab > 0.5
    out_lab = label_smooth(out_lab)
    out_lab = np.asarray(out_lab, np.uint8)

    # crop the image to [100, 120, 120]
    offset = [50, 60, 60]
    bb_min, bb_max = get_ND_bounding_box(out_lab)
    dim = len(bb_min)
    center  = [int((bb_min[i] + bb_max[i])/2) for i in range(dim)]
    bb_min1 = [max(0, center[i] - offset[i]) for i in range(dim)]
    bb_max1 = [min(out_data.shape[i], center[i] + offset[i]) for i in range(dim)]
    out_data = crop_ND_volume_with_bounding_box(out_data, bb_min1, bb_max1)
    out_lab  = crop_ND_volume_with_bounding_box(out_lab, bb_min1, bb_max1)

    out_obj  = sitk.GetImageFromArray(out_data)
    out_obj.SetOrigin(origin)
    out_obj.SetSpacing([1.0, 1.0, 1.0])
    out_obj.SetDirection(direction)

    
    out_lab_obj = sitk.GetImageFromArray(out_lab)
    out_lab_obj.SetOrigin(origin)
    out_lab_obj.SetSpacing([1.0, 1.0, 1.0])
    out_lab_obj.SetDirection(direction)

    sitk.WriteImage(out_obj, output_img_name) 
    sitk.WriteImage(out_lab_obj, output_lab_name) 

def preprocess_promis12(data_dir):
    for part in [1, 2, 3]:
        input_folder = data_dir + "TrainingData_Part{0:}".format(part)
        output_folder = data_dir +"preprocess/"
        file_names = os.listdir(input_folder)
        file_names = [item for item in file_names \
                if (("mhd" in item) and ("segmentation" not in item)) ]
        print(len(file_names))
        for file_name in file_names:
                print(file_name)
                img_name_full = "{0:}/{1:}".format(input_folder, file_name)
                lab_name = file_name.replace(".mhd", "_segmentation.mhd")
                lab_name_full = "{0:}/{1:}".format(input_folder, lab_name)
                
                out_name = file_name
                out_name = out_name.replace("Case", "promise_")
                out_name = out_name.replace("mhd", "nii.gz")
                out_name_full = output_folder + "image/" + out_name
                out_lab_full  = output_folder + "label/" + out_name
                image_resample_crop(img_name_full, lab_name_full, out_name_full, out_lab_full)

if __name__ == "__main__":
    data_dir = "data/promise12/"
    preprocess_promis12(data_dir) 