"""Script for convert elipse to binary mask
"""

import os
import sys
import numpy as np
from scipy import ndimage
from PIL import Image

def get_backgrouond(img): # 2D or 3D
    if(img.sum()==0):
        print('the largest component is null')
        return img
    if(len(img.shape) == 3):
        s = ndimage.generate_binary_structure(3,1) # iterate structure
    elif(len(img.shape) == 2):
        s = ndimage.generate_binary_structure(2,1) # iterate structure
    else:
        raise ValueError("the dimension number shoud be 2 or 3")
    labeled_array, numpatches = ndimage.label(img,s) # labeling
    assert(labeled_array.max() == 2)
    return np.asarray(labeled_array == 1, np.uint8)

def get_segmentation_ground_truth():
    data_root   = '/home/guotai/data/Fetal_HC/training_set'
    target_dir  = '/home/guotai/data/Fetal_HC/training_set_label'
    filenames = os.listdir(data_root)
    filenames = [item for item in filenames if "Annotation.png" in item \
        and item[0] != '.']
    num = len(filenames)
    for filename in filenames:
        print(filename)
        full_filename = "{0:}/{1:}".format(data_root, filename)
        image = Image.open(full_filename)
        data  = np.asarray(image)
        data  = np.asarray(data == 0)
        data_bg = get_backgrouond(data)
        data_fg = np.asarray(1.0 - data_bg, np.uint8)*255
        out_img = Image.fromarray(data_fg)
        out_name = filename.replace('Annotation', 'seg')
        out_name = target_dir + '/' + out_name
        out_img.save(out_name)

if __name__ == "__main__":
    get_segmentation_ground_truth()
