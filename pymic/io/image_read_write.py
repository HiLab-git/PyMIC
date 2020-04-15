# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import numpy as np
import SimpleITK as sitk

from PIL import Image

def load_nifty_volume_as_4d_array(filename):
    """Read a nifty image and return a dictionay storing data array, spacing and direction
    output['data_array'] 4d array with shape [C, D, H, W]
    output['spacing']    a list of spacing in z, y, x axis 
    output['direction']  a 3x3 matrix for direction
    """
    img_obj    = sitk.ReadImage(filename)
    data_array = sitk.GetArrayFromImage(img_obj)
    origin     = img_obj.GetOrigin()
    spacing    = img_obj.GetSpacing()
    direction  = img_obj.GetDirection()
    shape = data_array.shape
    if(len(shape) == 4):
        assert(shape[3] == 1) 
    elif(len(shape) == 3):
        data_array = np.expand_dims(data_array, axis = 0)
    else:
        raise ValueError("unsupported image dim: {0:}".format(len(shape)))
    output = {}
    output['data_array'] = data_array
    output['origin']     = origin
    output['spacing']    = (spacing[2], spacing[1], spacing[0])
    output['direction']  = direction
    return output

def load_rgb_image_as_3d_array(filename):
    image = np.asarray(Image.open(filename))
    image_shape = image.shape
    image_dim   = len(image_shape)
    assert(image_dim == 2 or image_dim == 3)
    if(image_dim == 2):
        image = np.expand_dims(image, axis = 0)
    else:
        # transpose rgb image from [H, W, C] to [C, H, W]
        assert(image_shape[2] == 3 or image_shape[2] == 4)
        if(image_shape[2] == 4):
            image = image[:, :, range(3)]
        image = np.transpose(image, axes = [2, 0, 1])
    output = {}
    output['data_array'] = image
    output['origin']     = (0, 0)
    output['spacing']    = (1.0, 1.0)
    output['direction']  = 0
    return output

def load_image_as_nd_array(image_name):
    """
    return a 4D array with shape [C, D, H, W], or 3D array with shape [C, H, W]
    """
    if (image_name.endswith(".nii.gz") or image_name.endswith(".nii") or
        image_name.endswith(".mha")):
        image_dict = load_nifty_volume_as_4d_array(image_name)
    elif(image_name.endswith(".jpg") or image_name.endswith(".jpeg") or
         image_name.endswith(".tif") or image_name.endswith(".png")):
        image_dict = load_rgb_image_as_3d_array(image_name)
    else:
        raise ValueError("unsupported image format")
    return image_dict

def save_array_as_nifty_volume(data, image_name, reference_name = None):
    """
    save a numpy array as nifty image
    inputs:
        data: a numpy array with shape [Depth, Height, Width]
        image_name: the ouput file name
        reference_name: file name of the reference image of which affine and header are used
    outputs: None
    """
    img = sitk.GetImageFromArray(data)
    if(reference_name is not None):
        img_ref = sitk.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    sitk.WriteImage(img, image_name)

def save_array_as_rgb_image(data, image_name):
    """
    save a numpy array as rgb image
    inputs:
        data: a numpy array with shape [3, H, W] or [H, W, 3] or [H, W]
        image_name: the output file name
    outputs: None
    """
    data_dim = len(data.shape)
    if(data_dim == 3):
        assert(data.shape[0] == 3 or data.shape[2] == 3)
        if(data.shape[0] == 3):
            data = np.transpose(data, [1, 2, 0])
    img = Image.fromarray(data)
    img.save(image_name)

def save_nd_array_as_image(data, image_name, reference_name = None):
    """
    save a 3D or 2D numpy array as medical image or RGB image
    inputs:
        data: a numpy array with shape [D, H, W] or [C, H, W]
        image_name: the output file name 
    outputs: None
    """
    data_dim = len(data.shape)
    assert(data_dim == 2 or data_dim == 3)
    if (image_name.endswith(".nii.gz") or image_name.endswith(".nii") or
        image_name.endswith(".mha")):
        assert(data_dim == 3)
        save_array_as_nifty_volume(data, image_name, reference_name)

    elif(image_name.endswith(".jpg") or image_name.endswith(".jpeg") or
         image_name.endswith(".tif") or image_name.endswith(".png")):
         assert(data_dim == 2)
         save_array_as_rgb_image(data, image_name)
    else:
        raise ValueError("unsupported image format {0:}".format(
            image_name.split('.')[-1]))