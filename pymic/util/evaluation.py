# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import sys
import math
import random
import configparser
import numpy as np
from scipy import ndimage
from pymic.io.image_read_write import *
from pymic.util.image_process import *
from pymic.util.parse_config import parse_config

# Dice evaluation
def binary_dice(s, g, resize = False):
    """
    calculate the Dice score of two N-d volumes.
    s: the segmentation volume of numpy array
    g: the ground truth volume of numpy array
    resize: if s and g have different shapes, resize s to match g.
    """
    assert(len(s.shape)== len(g.shape))
    if(resize):
        size_match = True
        for i in range(len(s.shape)):
            if(s.shape[i] != g.shape[i]):
                size_match = False
                break
        if(size_match is False):
            s = resize_ND_volume_to_given_shape(s, g.shape, order = 0)
    prod = np.multiply(s, g)
    s0 = prod.sum()
    s1 = s.sum()
    s2 = g.sum()
    dice = (2.0*s0 + 1e-5)/(s1 + s2 + 1e-5)
    return dice

def dice_of_images(s_name, g_name):
    s = load_image_as_nd_array(s_name)['data_array']
    g = load_image_as_nd_array(g_name)['data_array']
    dice = binary_dice(s, g)
    return dice

# IOU evaluation
def binary_iou(s,g):
    assert(len(s.shape)== len(g.shape))
    intersecion = np.multiply(s, g)
    union = np.asarray(s + g >0, np.float32)
    iou = intersecion.sum()/(union.sum() + 1e-10)
    return iou

def iou_of_images(s_name, g_name):
    s = load_image_as_nd_array(s_name)['data_array']
    g = load_image_as_nd_array(g_name)['data_array']
    margin = (3, 8, 8)
    g = get_detection_binary_bounding_box(g, margin)
    return binary_iou(s, g)

# Hausdorff and ASD evaluation
def get_points_on_contour(img):
    assert(len(img.shape) == 2)
    point_list = []
    [H, W] = img.shape
    offset_h  = [ -1, 1,  0, 0]
    offset_w  = [ 0, 0, -1, 1]
    for h in range(1, H-1):
        for w in range(1, W-1):
            if(img[h, w] > 0):
                edge_flag = False
                for idx in range(4):
                    if(img[h + offset_h[idx], w + offset_w[idx]] == 0):
                        edge_flag = True
                        break
                if(edge_flag):
                    point_list.append([h, w])
    return point_list

def get_points_on_surface(img):
    strt = ndimage.generate_binary_structure(3,2)
    img  = ndimage.morphology.binary_closing(img, strt, 5)
    point_list = []
    [D, H, W] = img.shape
    offset_d  = [-1, 1,  0, 0,  0, 0]
    offset_h  = [ 0, 0, -1, 1,  0, 0]
    offset_w  = [ 0, 0,  0, 0, -1, 1]
    for d in range(1, D-1):
        for h in range(1, H-1):
            for w in range(1, W-1):
                if(img[d, h, w] > 0):
                    edge_flag = False
                    for idx in range(6):
                        if(img[d + offset_d[idx], h + offset_h[idx], w + offset_w[idx]] == 0):
                            edge_flag = True
                            break
                    if(edge_flag):
                        point_list.append([d, h, w])
    return point_list

def get_sampled_surface_points(img):
    """
    get list of surface points of an images. If the image is in 3D, the 
    surface point list is resampled for efficiency
    inputs:
        img: a 2D or 3D numpy array with binary values 
    outputs:
        a list of surface points
    """
    image_dim = len(img.shape)
    if(image_dim == 2):
        point_list = get_points_on_contour(img)
    else:
        point_list = get_points_on_surface(img)
    return point_list

def hausdorff95_from_one_surface_to_another(point_list_s, point_list_g, spacing):
    point_dim = len(spacing)
    n_max = 300
    if(len(point_list_s) > n_max):
        point_list_s = random.sample(point_list_s, n_max)
    if(len(point_list_g) > n_max * 10):
        point_list_g = random.sample(point_list_g, n_max * 10)
    dist_list = []
    for ps in point_list_s:
        ps_nearest = 1e8
        for pg in point_list_g:
            dd = spacing[0]*(ps[0] - pg[0])
            dh = spacing[1]*(ps[1] - pg[1])
            temp_dis_square = dd*dd + dh*dh
            if(point_dim == 3):
                dw = spacing[2]*(ps[2] - pg[2])
                temp_dis_square += dw*dw
            if(temp_dis_square < ps_nearest):
                ps_nearest = temp_dis_square
        dis = math.sqrt(ps_nearest)
        dist_list.append(dis)
    dist_list = sorted(dist_list)
    hd95 = dist_list[int(len(dist_list)*0.95)]
    return hd95

def binary_hausdorff95(s, g, spacing = None):
    """
    get the hausdorff distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    image_dim = len(s.shape)
    assert(image_dim == len(g.shape))
    if(spacing == None):
        spacing = [1.0] * image_dim
    else:
        assert(image_dim == len(spacing))
    point_list_s = get_sampled_surface_points(s)
    point_list_g = get_sampled_surface_points(g)

    dis1 = hausdorff95_from_one_surface_to_another(
            point_list_s, point_list_g, spacing)
    dis2 = hausdorff95_from_one_surface_to_another(
            point_list_g, point_list_s, spacing)
    return max(dis1, dis2)


def average_surface_distance(point_list_s, point_list_g, spacing):
    """
    ASD for 2D contours or 3D surfaces 
    """
    distance_sum = 0.0
    assert(len(spacing) == 2 or len(spacing) == 3)
    assert(len(point_list_s[0]) == len(point_list_g[0]) and \
        len(point_list_s[0]) == len(spacing))
    for ps in point_list_s:
        ps_nearest = 1e10
        for pg in point_list_g:
            dd = spacing[0]*(ps[0] - pg[0])
            dh = spacing[1]*(ps[1] - pg[1])
            temp_dis_square = dd*dd + dh*dh
            if(len(spacing) == 3):
                dw = spacing[2]*(ps[2] - pg[2])
                temp_dis_square = temp_dis_square + dw*dw
            if(temp_dis_square < ps_nearest):
                ps_nearest = temp_dis_square
        distance_sum = distance_sum + math.sqrt(ps_nearest)
    asd = distance_sum/len(point_list_s)
    return asd

def binary_assd(s, g, spacing = None):
    """
    ASD for 2D contours and 3D surfaces 
    """
    image_dim = len(s.shape)
    assert(image_dim == len(g.shape))
    if(spacing == None):
        spacing = [1.0] * image_dim
    else:
        assert(image_dim == len(spacing))
    point_list_s = get_sampled_surface_points(s)
    point_list_g = get_sampled_surface_points(g)

    n1 = len(point_list_s)
    n2 = len(point_list_g)
    asd1 = average_surface_distance(point_list_s, point_list_g, spacing)
    asd2 = average_surface_distance(point_list_g, point_list_s, spacing)
    assd = (asd1 * n1 + asd2 * n2)/(n1 + n2)
    return assd

# relative volume error evaluation
def binary_relative_volume_error(s_volume, g_volume):
    s_v = float(s_volume.sum())
    g_v = float(g_volume.sum())
    assert(g_v > 0)
    rve = abs(s_v - g_v)/g_v
    return rve

def get_evaluation_score(s_volume, g_volume, spacing, metric):
    if(len(s_volume.shape) == 4):
        assert(s_volume.shape[0] == 1 and g_volume.shape[0] == 1)
        s_volume = np.reshape(s_volume, s_volume.shape[1:])
        g_volume = np.reshape(g_volume, g_volume.shape[1:])
    if(s_volume.shape[0] == 1):
        s_volume = np.reshape(s_volume, s_volume.shape[1:])
        g_volume = np.reshape(g_volume, g_volume.shape[1:])
    metric_lower = metric.lower()

    if(metric_lower == "dice"):
        score = binary_dice(s_volume, g_volume)

    elif(metric_lower == "iou"):
        score = binary_iou(s_volume,g_volume)

    elif(metric_lower == 'assd'):
        score = binary_assd(s_volume, g_volume, spacing)

    elif(metric_lower == "hausdorff95"):
        score = binary_hausdorff95(s_volume, g_volume, spacing)

    elif(metric_lower == "rve"):
        score = binary_relative_volume_error(s_volume, g_volume)

    elif(metric_lower == "volume"):
        voxel_size = 1.0
        for dim in range(len(spacing)):
            voxel_size = voxel_size * spacing[dim]
        score = g_volume.sum()*voxel_size
    else:
        raise ValueError("unsupported evaluation metric: {0:}".format(metric))

    return score

def evaluation(config_file):
    config = parse_config(config_file)['evaluation']
    metric = config['metric']
    labels = config['label_list']
    organ_name = config['organ_name']
    ground_truth_label_convert_source = config.get('ground_truth_label_convert_source', None)
    ground_truth_label_convert_target = config.get('ground_truth_label_convert_target', None)
    segmentation_label_convert_source = config.get('segmentation_label_convert_source', None)
    segmentation_label_convert_target = config.get('segmentation_label_convert_target', None)
    s_folder_list = config['segmentation_folder_list']
    g_folder_list = config['ground_truth_folder_list']
    s_format  = config['segmentation_format']
    g_format  = config['ground_truth_format']
    s_postfix = config.get('segmentation_postfix',None)
    g_postfix = config.get('ground_truth_postfix',None)

    s_postfix_long = '.' + s_format
    if(s_postfix is not None):
        s_postfix_long = '_' + s_postfix + s_postfix_long
    g_postfix_long = '.' + g_format
    if(g_postfix is not None):
        g_postfix_long = '_' + g_postfix + g_postfix_long

    patient_names_file = config['patient_file_names']
    with open(patient_names_file) as f:
            content = f.readlines()
            patient_names = [x.strip() for x in content] 

    for s_folder in s_folder_list:
        score_all_data = []
        for i in range(len(patient_names)):
            # load segmentation and ground truth
            s_name = os.path.join(s_folder, patient_names[i] + s_postfix_long)
            if(not os.path.isfile(s_name)):
                break

            for g_folder in g_folder_list:
                g_name = os.path.join(g_folder, patient_names[i] + g_postfix_long)
                if(os.path.isfile(g_name)):
                    break
            s_dict = load_image_as_nd_array(s_name)
            g_dict = load_image_as_nd_array(g_name)
            s_volume = s_dict["data_array"]; s_spacing = s_dict["spacing"]
            g_volume = g_dict["data_array"]; g_spacing = g_dict["spacing"]
            # for dim in range(len(s_spacing)):
            #     assert(s_spacing[dim] == g_spacing[dim])
            if((ground_truth_label_convert_source is not None) and \
                ground_truth_label_convert_target is not None):
                g_volume = convert_label(g_volume, ground_truth_label_convert_source, \
                    ground_truth_label_convert_target)

            if((segmentation_label_convert_source is not None) and \
                segmentation_label_convert_target is not None):
                s_volume = convert_label(s_volume, segmentation_label_convert_source, \
                    segmentation_label_convert_target)

            # fuse multiple labels
            s_volume_sub = np.zeros_like(s_volume)
            g_volume_sub = np.zeros_like(g_volume)
            for lab in labels:
                s_volume_sub = s_volume_sub + np.asarray(s_volume == lab, np.uint8)
                g_volume_sub = g_volume_sub + np.asarray(g_volume == lab, np.uint8)
            
            # get evaluation score
            temp_score = get_evaluation_score(s_volume_sub > 0, g_volume_sub > 0,
                        g_spacing, metric)
            score_all_data.append(temp_score)
            print(patient_names[i], temp_score)
        score_all_data = np.asarray(score_all_data)
        score_mean = [score_all_data.mean(axis = 0)]
        score_std  = [score_all_data.std(axis = 0)]
        np.savetxt("{0:}/{1:}_{2:}_all.txt".format(s_folder, organ_name, metric), score_all_data)
        np.savetxt("{0:}/{1:}_{2:}_mean.txt".format(s_folder, organ_name, metric), score_mean)
        np.savetxt("{0:}/{1:}_{2:}_std.txt".format(s_folder, organ_name, metric), score_std)
        print("{0:} mean ".format(metric), score_mean)
        print("{0:} std  ".format(metric), score_std) 


def main():
    if(len(sys.argv) < 2):
        print('Number of arguments should be 2. e.g.')
        print('    python pyMIC.util/evaluation.py config.cfg')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    evaluation(config_file)
    
if __name__ == '__main__':
    main()
