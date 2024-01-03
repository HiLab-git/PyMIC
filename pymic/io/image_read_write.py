# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import numpy as np
import SimpleITK as sitk
from PIL import Image

def load_nifty_volume_as_4d_array(filename):
    """
    Read a nifty image and return a dictionay storing data array, origin, 
    spacing and direction.\n
    output['data_array'] 4D array with shape [C, D, H, W];\n
    output['spacing']    A list of spacing in z, y, x axis;\n
    output['direction']  A 3x3 matrix for direction.

    :param filename: (str) The input file name
    :return: A dictionay storing data array, origin, spacing and direction.
    """
    img_obj    = sitk.ReadImage(filename)
    data_array = sitk.GetArrayFromImage(img_obj)
    origin     = img_obj.GetOrigin()
    spacing    = img_obj.GetSpacing()
    direction  = img_obj.GetDirection()
    shape = data_array.shape
    if(len(shape) == 3):
        data_array = np.expand_dims(data_array, axis = 0)
    elif(len(shape) > 4 or len(shape) < 3):
        raise ValueError("unsupported image dim: {0:}".format(len(shape)))
    output = {}
    output['data_array'] = data_array
    output['origin']     = origin
    output['spacing']    = (spacing[2], spacing[1], spacing[0])
    output['direction']  = direction
    return output

def load_rgb_image_as_3d_array(filename):
    """
    Read an RGB image and return a dictionay storing data array, origin, 
    spacing and direction. \n
    output['data_array'] 3D array with shape [D, H, W]; \n
    output['spacing']    a list of spacing in z, y, x axis;  \n
    output['direction']  a 3x3 matrix for direction.

    :param filename: (str) The input file name
    :return: A dictionay storing data array, origin, spacing and direction.
    """
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
    Load an image and return a 4D array with shape [C, D, H, W], 
    or 3D array with shape [C, H, W].

    :param filename: (str) The input file name
    :return: A dictionay storing data array, origin, spacing and direction.
    """
    if (image_name.endswith(".nii.gz") or image_name.endswith(".nii") or
        image_name.endswith(".mha")):
        image_dict = load_nifty_volume_as_4d_array(image_name)
    elif(image_name.endswith(".jpg") or image_name.endswith(".jpeg") or
         image_name.endswith(".tif") or image_name.endswith(".png")):
        image_dict = load_rgb_image_as_3d_array(image_name)
    else:
        raise ValueError("unsupported image format: {0:}".format(image_name))
    return image_dict

def save_array_as_nifty_volume(data, image_name, reference_name = None, spacing = [1.0,1.0,1.0]):
    """
    Save a numpy array as nifty image

    :param data:  (numpy.ndarray) A numpy array with shape [Depth, Height, Width].
    :param image_name:  (str) The ouput file name.
    :param reference_name:  (str) File name of the reference image of which 
        meta information is used.
    :param spacing: (list or tuple) the spacing of a volume data when `reference_name` is not provided.  
    """
    img = sitk.GetImageFromArray(data)
    if(reference_name is not None):
        img_ref = sitk.ReadImage(reference_name)
        #img.CopyInformation(img_ref)
        img.SetSpacing(img_ref.GetSpacing())
        img.SetOrigin(img_ref.GetOrigin())
        direction0 = img_ref.GetDirection()
        direction1 = img.GetDirection()
        if(len(direction0) == len(direction1)):
            img.SetDirection(direction0)
    else:
        nifty_spacing = spacing[1:] + spacing[:1]
        img.SetSpacing(nifty_spacing)
    sitk.WriteImage(img, image_name)

def save_array_as_rgb_image(data, image_name):
    """
    Save a numpy array as rgb image.

    :param data:  (numpy.ndarray) A numpy array with shape [3, H, W] or
        [H, W, 3] or [H, W]. 
    :param image_name:  (str) The output file name.
    """
    data_dim = len(data.shape)
    if(data_dim == 3):
        assert(data.shape[0] == 3 or data.shape[2] == 3)
        if(data.shape[0] == 3):
            data = np.transpose(data, [1, 2, 0])
    img = Image.fromarray(data)
    img.save(image_name)

def save_nd_array_as_image(data, image_name, reference_name = None, spacing = [1.0,1.0,1.0]):
    """
    Save a 3D or 2D numpy array as medical image or RGB image
    
    :param data:  (numpy.ndarray) A numpy array with shape [3, H, W] or
        [H, W, 3] or [H, W]. 
    :param reference_name: (str) File name of the reference image of which 
        meta information is used.
    :param spacing: (list or tuple) the spacing of a volume data when `reference_name` is not provided.  
    """
    data_dim = len(data.shape)
    assert(data_dim == 2 or data_dim == 3)
    if (image_name.endswith(".nii.gz") or image_name.endswith(".nii") or
        image_name.endswith(".mha")):
        assert(data_dim == 3)
        save_array_as_nifty_volume(data, image_name, reference_name, spacing)

    elif(image_name.endswith(".jpg") or image_name.endswith(".jpeg") or
         image_name.endswith(".tif") or image_name.endswith(".png")):
         assert(data_dim == 2)
         save_array_as_rgb_image(data, image_name)
    else:
        raise ValueError("unsupported image format {0:}".format(
            image_name.split('.')[-1]))

def rotate_nifty_volume_to_LPS(filename_or_image_dict, origin = None, direction = None):
    '''
    Rotate the axis of a 3D volume to LPS

    :param filename_or_image_dict: (str) Filename of the nifty file (str) or image dictionary 
        returned by load_nifty_volume_as_4d_array. If supplied with the former, 
        the flipped image data will be saved to override the original file. 
        If supplied with the later, only flipped image data will be returned.\n
    :param origin: (list/tuple) The origin of the image.
    :param direction: (list or tuple) The direction of the image.

    :return: A dictionary for image data and meta info, with ``data_array``,
        ``origin``, ``direction`` and ``spacing``.
    '''

    if type(filename_or_image_dict) == str:
        image_data = load_nifty_volume_as_4d_array(filename_or_image_dict)
        save_nifty = True
    elif type(filename_or_image_dict) == dict:
        image_data = filename_or_image_dict
        save_nifty = False

    data_array = image_data['data_array']
    if not origin:
        origin = image_data['origin']
    if not direction:
        direction = image_data['direction']
    spacing = image_data['spacing']

    fliped = False
    if direction[0] == -1.:
        data_array = np.flip(data_array, axis = 3)
        fliped = True
    if direction[4] == -1.:
        data_array = np.flip(data_array, axis = 2)
        fliped = True
    if direction[8] == -1.:
        data_array = np.flip(data_array, axis = 1)
        fliped = True

    if save_nifty:
        if not fliped:
            return
        else:
            print(f'rotate {filename_or_image_dict} to LPS')
            img = sitk.GetImageFromArray(data_array[0])
            img.SetSpacing(spacing)
            img.SetOrigin(origin)
            img.SetDirection([1., 0., 0., 0., 1., 0., 0., 0., 1.])
            sitk.WriteImage(img, filename_or_image_dict)
    else:
        image_data['data_array'] = data_array
        image_data['direction'] = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
        return image_data
