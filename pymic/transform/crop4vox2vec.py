# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch

import json
import math
import random
import numpy as np
from imops import crop_to_box
from typing import *
from scipy import ndimage
from pymic import TaskType
from pymic.transform.abstract_transform import AbstractTransform
from pymic.transform.crop import CenterCrop
from pymic.util.image_process import *
from pymic.transform.intensity import *

def normalize_axis_list(axis, ndim):
    return list(np.core.numeric.normalize_axis_tuple(axis, ndim))

def scale_hu(image_hu: np.ndarray, window_hu: Tuple[float, float]) -> np.ndarray:
    min_hu, max_hu = window_hu
    assert min_hu < max_hu
    return np.clip((image_hu - min_hu) / (max_hu - min_hu), 0, 1)

# def gaussian_filter(
#         x: np.ndarray,
#         sigma: Union[float, Sequence[float]],
#         axis: Union[int, Sequence[int]]
# ) -> np.ndarray:
#     axis = normalize_axis_list(axis, x.ndim)
#     sigma = np.broadcast_to(sigma, len(axis))
#     for sgm, ax in zip(sigma, axis):
#         x = ndimage.gaussian_filter1d(x, sgm, ax)
#     return x

# def gaussian_sharpen(
#         x: np.ndarray,
#         sigma_1: Union[float, Sequence[float]],
#         sigma_2: Union[float, Sequence[float]],
#         alpha: float,
#         axis: Union[int, Sequence[int]]
# ) -> np.ndarray:
#     """ See https://docs.monai.io/en/stable/transforms.html#gaussiansharpen """
#     blurred = gaussian_filter(x, sigma_1, axis)
#     return blurred + alpha * (blurred - gaussian_filter(blurred, sigma_2, axis))

def sample_box(image_size, patch_size, anchor_voxel=None):
    image_size = np.array(image_size, ndmin=1)
    patch_size = np.array(patch_size, ndmin=1)

    if not np.all(image_size >= patch_size):
        raise ValueError(f'Can\'t sample patch of size {patch_size} from image of size {image_size}')

    min_start = 0
    max_start = image_size - patch_size
    if anchor_voxel is not None:
        anchor_voxel = np.array(anchor_voxel, ndmin=1)
        min_start = np.maximum(min_start, anchor_voxel - patch_size + 1)
        max_start = np.minimum(max_start, anchor_voxel)
    start = np.random.randint(min_start, max_start + 1)
    return np.array([start, start + patch_size])

def sample_views(
    image: np.ndarray,
    min_overlap: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    max_num_voxels: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ For 3D volumes, the image shape should be [C, D, H, W].
    """
    img_size = image.shape[1:]
    overlap  = [random.randint(min_overlap[i], patch_size[i]) for i in range(3)]
    union_size = [2*patch_size[i] - overlap[i] for i in range(3)]
    anchor_max = [img_size[i] - union_size[i] for i in range(3)]
    crop_min_1 = [random.randint(0, anchor_max[i]) for i in range(3)]
    crop_min_2 = [crop_min_1[i] + patch_size[i] - overlap[i] for i in range(3)]
    patch_1 = sample_view(image, crop_min_1, patch_size)
    patch_2 = sample_view(image, crop_min_2, patch_size)

    coords = [range(crop_min_2[i], crop_min_2[i] + overlap[i]) for i in range(3)]
    coords = np.asarray(np.meshgrid(coords[0], coords[1], coords[2]))
    coords = coords.reshape(3, -1).transpose()
    roi_voxels_1 = coords - crop_min_1
    roi_voxels_2 = coords - crop_min_2

    indices = range(coords.shape[0])
    if len(indices) > max_num_voxels:
        indices = np.random.choice(indices, max_num_voxels, replace=False)

    return patch_1, patch_2, roi_voxels_1[indices], roi_voxels_2[indices]


def sample_view(image, crop_min, patch_size):
    """ For 3D volumes, the image shape should be [C, D, H, W].
    """
    assert image.ndim == 4
    C = image.shape[0]
    crop_max = [crop_min[i] + patch_size[i] for i in range(3)]
    out = crop_ND_volume_with_bounding_box(image, [0] + crop_min, [C] + crop_max) 

    # intensity augmentations
    for c in range(C):
        if(random.random() < 0.8):
            out[c] = gaussian_noise(out[c], 0.05, 0.1)
        if(random.random() < 0.5):
            out[c] = gaussian_blur(out[c], 0.5, 1.5)
        else:
            alpha = random.uniform(0.0, 2.0)
            out[c] = gaussian_sharpen(out[c], 0.5, 2.0, alpha)
        if(random.random() < 0.8):
            out[c] = gamma_correction(out[c], 0.5, 2.0)
        if(random.random() < 0.8):
            out[c] = window_level_augment(out[c])
    return out

class Crop4Vox2Vec(CenterCrop):
    """
    Randomly crop an volume into two views with augmentation. This is used for
    self-supervised pretraining in Vox2vec.

    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `DualViewCrop_output_size`: (list/tuple) Desired output size [D, H, W].
        The output channel is the same as the input channel. 
    :param `DualViewCrop_scale_lower_bound`: (list/tuple) Lower bound of the range of scale
        for each dimension. e.g. (1.0, 0.5, 0.5).
    param `DualViewCrop_scale_upper_bound`: (list/tuple) Upper bound of the range of scale
        for each dimension. e.g. (1.0, 2.0, 2.0).
    :param `DualViewCrop_inverse`: (optional, bool) Is inverse transform needed for inference.
        Default is `False`. Currently, the inverse transform is not supported, and 
        this transform is assumed to be used only during training stage. 
    """
    def __init__(self, params):
        self.output_size = params['Crop4Vox2Vec_output_size'.lower()]
        self.min_overlap = params.get('Crop4Vox2Vec_min_overlap'.lower(), [8, 12, 12])
        self.max_voxel   = params.get('Crop4Vox2Vec_max_voxel'.lower(), 1024)
        self.inverse     = params.get('Crop4Vox2Vec_inverse'.lower(), False)
        self.task        = params['Task'.lower()]
        assert isinstance(self.output_size, (list, tuple))
        
    def __call__(self, sample):
        image = sample['image']
        channel, input_size = image.shape[0], image.shape[1:]
        input_dim = len(input_size)
        assert channel == 1
        assert(input_dim == len(self.output_size))
        invalid_size = [input_size[i] < self.output_size[i]*2 - self.min_overlap[i] for i in range(3)]
        if True in invalid_size:
            raise ValueError("The overlap requirement {0:} is too weak for the given patch size \
                {1:} and input size {2:}".format( self.min_overlap, self.output_size,input_size))    
 
        patches_1, patches_2, voxels_1, voxels_2 = sample_views(image, 
            self.min_overlap, self.output_size, self.max_voxel)
        sample['image'] = patches_1, patches_2, voxels_1, voxels_2
        return sample

   
