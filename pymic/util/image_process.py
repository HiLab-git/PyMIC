# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

def get_ND_bounding_box(volume, margin = None):
    """
    Get the bounding box of nonzero region in an ND volume.

    :param volume: An ND numpy array. 
    :param margin: (list)
        The margin of bounding box along each axis. 

    :return bb_min: (list) A list for the minimal value of each axis 
            of the bounding box. 
    :return bb_max: (list) A list for the maximal value of each axis 
            of the bounding box. 
    """
    input_shape = volume.shape
    if(margin is None):
        margin = [0] * len(input_shape)
    assert(len(input_shape) == len(margin))
    indxes = np.nonzero(volume)
    bb_min = []
    bb_max = []
    for i in range(len(input_shape)):
        bb_min.append(int(indxes[i].min()))
        bb_max.append(int(indxes[i].max()) + 1)

    for i in range(len(input_shape)):
        bb_min[i] = max(bb_min[i] - margin[i], 0)
        bb_max[i] = min(bb_max[i] + margin[i], input_shape[i])
    return bb_min, bb_max

def crop_ND_volume_with_bounding_box(volume, bb_min, bb_max):
    """
    Extract a subregion form an ND image.

    :param volume: The input ND array. 
    :param bb_min: (list) The lower bound of the bounding box for each axis.
    :param bb_max: (list) The upper bound of the bounding box for each axis.

    :return: A croped ND image.
    """
    dim = len(volume.shape)
    assert(dim >= 2 and dim <= 5)
    assert(bb_max[0] - bb_min[0] <= volume.shape[0])
    if(dim == 2):
        output = volume[bb_min[0]:bb_max[0], bb_min[1]:bb_max[1]]
    elif(dim == 3):
        output = volume[bb_min[0]:bb_max[0], bb_min[1]:bb_max[1], bb_min[2]:bb_max[2]]
    elif(dim == 4):
        output = volume[bb_min[0]:bb_max[0], bb_min[1]:bb_max[1], bb_min[2]:bb_max[2], bb_min[3]:bb_max[3]]
    elif(dim == 5):
        output = volume[bb_min[0]:bb_max[0], bb_min[1]:bb_max[1], bb_min[2]:bb_max[2], bb_min[3]:bb_max[3], bb_min[4]:bb_max[4]]
    else:
        raise ValueError("the dimension number shoud be 2 to 5")
    return output

def set_ND_volume_roi_with_bounding_box_range(volume, bb_min, bb_max, sub_volume, addition = True):
    """
    Set the subregion of an ND image. If `addition` is `True`, the original volume is added by the given sub volume.

    :param volume: The input ND volume.
    :param bb_min: (list) The lower bound of the bounding box for each axis.
    :param bb_max: (list) The upper bound of the bounding box for each axis.
    :param sub_volume: The sub volume to replace the target region of the orginal volume. 
    :param addition: (optional, bool) If True, the sub volume will be added
        to the target region of the input volume.
    """
    dim = len(bb_min)
    out = volume
    if(dim == 2):
        if(addition):
            out[bb_min[0]:bb_max[0], bb_min[1]:bb_max[1]] += sub_volume
        else:
            out[bb_min[0]:bb_max[0], bb_min[1]:bb_max[1]] = sub_volume
    elif(dim == 3):
        if(addition):
            out[bb_min[0]:bb_max[0], bb_min[1]:bb_max[1], bb_min[2]:bb_max[2]] += sub_volume
        else:
            out[bb_min[0]:bb_max[0], bb_min[1]:bb_max[1], bb_min[2]:bb_max[2]] = sub_volume
    elif(dim == 4):
        if(addition):
            out[bb_min[0]:bb_max[0], bb_min[1]:bb_max[1], bb_min[2]:bb_max[2], bb_min[3]:bb_max[3]]  += sub_volume
        else:
            out[bb_min[0]:bb_max[0], bb_min[1]:bb_max[1], bb_min[2]:bb_max[2], bb_min[3]:bb_max[3]] = sub_volume
    elif(dim == 5):
        if(addition):
            out[bb_min[0]:bb_max[0], bb_min[1]:bb_max[1], bb_min[2]:bb_max[2], bb_min[3]:bb_max[3], bb_min[4]:bb_max[4]] += sub_volume
        else:
            out[bb_min[0]:bb_max[0], bb_min[1]:bb_max[1], bb_min[2]:bb_max[2], bb_min[3]:bb_max[3], bb_min[4]:bb_max[4]] = sub_volume
    else:
        raise ValueError("array dimension should be 2 to 5")
    return out

def crop_and_pad_ND_array_to_desired_shape(image, out_shape, pad_mod):
    """
    Crop and pad an image to a given shape. 

    :param image: The input ND array.
    :param out_shape: (list) The desired output shape. 
    :param pad_mod: (str) See `numpy.pad <https://numpy.org/doc/stable/reference/generated/numpy.pad.html>`_
    """
    in_shape   = image.shape 
    dim        = len(in_shape)
    crop_shape = [min(out_shape[i], in_shape[i])  for i in range(dim)]
    mgnc = [max(0, in_shape[i] - crop_shape[i]) for i in range(dim)]
    if(max(mgnc) == 0):
        image_crp = image
    else:
        ml   = [int(mgnc[i]/2)  for i in range(dim)]
        mr   = [mgnc[i] - ml[i] for i in range(dim)] 
        if(dim == 2):
            image_crp = image[ml[0]:(in_shape[0] - mr[0]), ml[1]:(in_shape[1] - mr[1])]
        elif(dim == 3):
            image_crp = image[ml[0]:(in_shape[0] - mr[0]), ml[1]:(in_shape[1] - mr[1]), ml[2]:(in_shape[2] - mr[2])]
        elif(dim == 4):
            image_crp = image[ml[0]:(in_shape[0] - mr[0]), ml[1]:(in_shape[1] - mr[1]), ml[2]:(in_shape[2] - mr[2]), ml[3]:(in_shape[3] - mr[3])]
        elif(dim == 5):
            image_crp = image[ml[0]:(in_shape[0] - mr[0]), ml[1]:(in_shape[1] - mr[1]), ml[2]:(in_shape[2] - mr[2]), ml[3]:(in_shape[3] - mr[3]), ml[4]:(in_shape[4] - mr[4])]
        else:
            raise ValueError("array dimension should be 2 to 5")

    mgnp = [out_shape[i] - crop_shape[i] for i in range(dim)]
    if(max(mgnp) == 0):
        image_pad = image_crp
    else:
        ml   = [int(mgnp[i]/2)  for i in range(dim)]
        mr   = [mgnp[i] - ml[i] for i in range(dim)] 
        pad  = [(ml[i], mr[i])  for i in range(dim)]
        pad  = tuple(pad)
        image_pad = np.pad(image_crp, pad, pad_mod) 
        
    return image_pad

def get_largest_k_components(image, k = 1):
    """
    Get the largest K components from 2D or 3D binary image.

    :param image: The input ND array for binary segmentation.
    :param k: (int) The value of k.

    :return: An output array with only the largest K components of the input. 
    """
    dim = len(image.shape)
    if(image.sum() == 0 ):
        print('the largest component is null')
        return image
    if(dim < 2 or dim > 3):
        raise ValueError("the dimension number should be 2 or 3")
    s = ndimage.generate_binary_structure(dim,1)      
    labeled_array, numpatches = ndimage.label(image, s)
    sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
    sizes_sort = sorted(sizes, reverse = True)
    kmin = min(k, numpatches)
    output = np.zeros_like(image)
    for i in range(kmin):
        labeli = np.where(sizes == sizes_sort[i])[0] + 1
        output = output + np.asarray(labeled_array == labeli, np.uint8)
    return  output

def get_euclidean_distance(image, dim = 3, spacing = [1.0, 1.0, 1.0]):
    """
    Get euclidean distance transform of 3D binary images.
    The output distance map is unsigned.

    :param image: The input 3D array.
    :param dim: (int) Using 2D (dim = 2) or 3D (dim = 3) distance transforms.
    :param spacing: (list) The spacing along each axis.

    """
    img_shape = image.shape
    input_dim = len(img_shape)
    if(input_dim != 3):
        raise ValueError("Not implemented for {0:}D image".format(input_dim))
    if(dim == 2):
        raise ValueError("Not implemented for {0:}D image".format(input_dim))
        # dis_map = np.ones_like(image, np.float32)
        # for d in range(img_shape[0]):
        #     if(image[d].sum() > 0):
        #         dis_d = ndimage.morphology.distance_transform_edt(image[d])
        #         dis_map[d] = dis_d/dis_d.max()
    elif(dim == 3):
        fg_dis_map = ndimage.morphology.distance_transform_edt(image > 0.5)
        bg_dis_map = ndimage.morphology.distance_transform_edt(image <= 0.5)
        dis_map = bg_dis_map - fg_dis_map
    else:
        raise ValueError("Not implemented for {0:}D distance".format(dim))
    return dis_map

def convert_label(label, source_list, target_list):
    """
    Convert a label map based a source list and a target list of labels
    
    :param label: (numpy.array) The input label map. 
    :param source_list: A list of labels that will be converted, e.g. [0, 1, 2, 4]
    :param target_list: A list of target labels, e.g. [0, 1, 2, 3]
    """
    assert(len(source_list) == len(target_list))
    label_converted = np.zeros_like(label)
    for i in range(len(source_list)):
        label_temp = np.asarray(label == source_list[i], label.dtype)
        label_temp = label_temp * target_list[i]
        label_converted = label_converted + label_temp
    return label_converted

def resample_sitk_image_to_given_spacing(image, spacing, order):
    """
    Resample an sitk image objct to a given spacing. 

    :param image: The input sitk image object.
    :param spacing: (list/tuple) Target spacing along x, y, z direction.
    :param order: (int) Order for interpolation.

    :return: A resampled sitk image object.
    """
    spacing0 = image.GetSpacing()
    data = sitk.GetArrayFromImage(image)
    zoom = [spacing0[i] / spacing[i] for i in range(3)]
    zoom = [zoom[2], zoom[0], zoom[1]]
    data = ndimage.interpolation.zoom(data, zoom, order = order)
    out_img = sitk.GetImageFromArray(data)
    out_img.SetSpacing(spacing)
    out_img.SetDirection(image.GetDirection())
    return out_img
