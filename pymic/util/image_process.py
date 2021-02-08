# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

def get_ND_bounding_box(volume, margin = None):
    """
    get the bounding box of nonzero region in an ND volume
    """
    input_shape = volume.shape
    if(margin is None):
        margin = [0] * len(input_shape)
    assert(len(input_shape) == len(margin))
    indxes = np.nonzero(volume)
    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):
        idx_min.append(int(indxes[i].min()))
        idx_max.append(int(indxes[i].max()) + 1)

    for i in range(len(input_shape)):
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i])
    return idx_min, idx_max

def crop_ND_volume_with_bounding_box(volume, min_idx, max_idx):
    """
    crop/extract a subregion form an nd image.
    """
    dim = len(volume.shape)
    assert(dim >= 2 and dim <= 5)
    assert(max_idx[0] - min_idx[0] <= volume.shape[0])
    if(dim == 2):
        output = volume[np.ix_(range(min_idx[0], max_idx[0]),
                               range(min_idx[1], max_idx[1]))]
    elif(dim == 3):
        output = volume[np.ix_(range(min_idx[0], max_idx[0]),
                               range(min_idx[1], max_idx[1]),
                               range(min_idx[2], max_idx[2]))]
    elif(dim == 4):
        output = volume[np.ix_(range(min_idx[0], max_idx[0]),
                               range(min_idx[1], max_idx[1]),
                               range(min_idx[2], max_idx[2]),
                               range(min_idx[3], max_idx[3]))]
    elif(dim == 5):
        output = volume[np.ix_(range(min_idx[0], max_idx[0]),
                               range(min_idx[1], max_idx[1]),
                               range(min_idx[2], max_idx[2]),
                               range(min_idx[3], max_idx[3]),
                               range(min_idx[4], max_idx[4]))]
    else:
        raise ValueError("the dimension number shoud be 2 to 5")
    return output

def set_ND_volume_roi_with_bounding_box_range(volume, bb_min, bb_max, sub_volume, addition = True):
    """
    set a subregion to an nd image. if addition is True, the original volume is added by the subregion.
    """
    dim = len(bb_min)
    out = volume
    if(dim == 2):
        if(addition):
            out[np.ix_(range(bb_min[0], bb_max[0]),
                    range(bb_min[1], bb_max[1]))]  += sub_volume
        else:
            out[np.ix_(range(bb_min[0], bb_max[0]),
                    range(bb_min[1], bb_max[1]))]   = sub_volume
    elif(dim == 3):
        if(addition):
            out[np.ix_(range(bb_min[0], bb_max[0]),
                    range(bb_min[1], bb_max[1]),
                    range(bb_min[2], bb_max[2]))]  += sub_volume
        else:
            out[np.ix_(range(bb_min[0], bb_max[0]),
                    range(bb_min[1], bb_max[1]),
                    range(bb_min[2], bb_max[2]))]   = sub_volume
    elif(dim == 4):
        if(addition):
            out[np.ix_(range(bb_min[0], bb_max[0]),
                    range(bb_min[1], bb_max[1]),
                    range(bb_min[2], bb_max[2]),
                    range(bb_min[3], bb_max[3]))]  += sub_volume
        else:
            out[np.ix_(range(bb_min[0], bb_max[0]),
                    range(bb_min[1], bb_max[1]),
                    range(bb_min[2], bb_max[2]),
                    range(bb_min[3], bb_max[3]))]   = sub_volume
    elif(dim == 5):
        if(addition):
            out[np.ix_(range(bb_min[0], bb_max[0]),
                    range(bb_min[1], bb_max[1]),
                    range(bb_min[2], bb_max[2]),
                    range(bb_min[3], bb_max[3]),
                    range(bb_min[4], bb_max[4]))]  += sub_volume
        else:
            out[np.ix_(range(bb_min[0], bb_max[0]),
                    range(bb_min[1], bb_max[1]),
                    range(bb_min[2], bb_max[2]),
                    range(bb_min[3], bb_max[3]),
                    range(bb_min[4], bb_max[4]))]   = sub_volume
    else:
        raise ValueError("array dimension should be 2 to 5")
    return out

def crop_and_pad_ND_array_to_desired_shape(image, out_shape, pad_mod):
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
            image_crp = image[np.ix_(range(ml[0], in_shape[0] - mr[0]),
                                    range(ml[1], in_shape[1] - mr[1]))]
        elif(dim == 3):
            image_crp = image[np.ix_(range(ml[0], in_shape[0] - mr[0]),
                                    range(ml[1], in_shape[1] - mr[1]),
                                    range(ml[2], in_shape[2] - mr[2]))]
        elif(dim == 4):
            image_crp = image[np.ix_(range(ml[0], in_shape[0] - mr[0]),
                                    range(ml[1], in_shape[1] - mr[1]),
                                    range(ml[2], in_shape[2] - mr[2]),
                                    range(ml[3], in_shape[3] - mr[3]))]
        elif(dim == 5):
            image_crp = image[np.ix_(range(ml[0], in_shape[0] - mr[0]),
                                    range(ml[1], in_shape[1] - mr[1]),
                                    range(ml[2], in_shape[2] - mr[2]),
                                    range(ml[3], in_shape[3] - mr[3]),
                                    range(ml[4], in_shape[4] - mr[4]))]
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

def get_largest_component(image):
    """
    get the largest component from 2D or 3D binary image
    image: nd array
    """
    dim = len(image.shape)
    if(image.sum() == 0 ):
        print('the largest component is null')
        return image
    if(dim == 2):
        s = ndimage.generate_binary_structure(2,1)
    elif(dim == 3):
        s = ndimage.generate_binary_structure(3,1)
    else:
        raise ValueError("the dimension number should be 2 or 3")
    labeled_array, numpatches = ndimage.label(image, s)
    sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
    max_label = np.where(sizes == sizes.max())[0] + 1
    output = np.asarray(labeled_array == max_label, np.uint8)
    return  output

def get_euclidean_distance(image, dim = 3, spacing = [1.0, 1.0, 1.0]):
    """
    get euclidean distance transform of 2D or 3D binary images
    the output distance map is unsigned
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
    convert a label map based a source list and a target list of labels
    label: nd array 
    source_list: a list of labels that will be converted, e.g. [0, 1, 2, 4]
    target_list: a list of target labels, e.g. [0, 1, 2, 3]
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
    image: an sitk image object
    spacing: 3D tuple / list for spacing along x, y, z direction
    order: order for interpolation
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
