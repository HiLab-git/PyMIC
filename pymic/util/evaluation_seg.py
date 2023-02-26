# -*- coding: utf-8 -*-
"""
Evaluation module for segmenation tasks.
"""
from __future__ import absolute_import, print_function
import csv
import os
import sys
import pandas as pd
import numpy as np
from scipy import ndimage
from pymic.io.image_read_write import *
from pymic.util.image_process import *
from pymic.util.parse_config import parse_config


def binary_dice(s, g, resize = False):
    """
    Calculate the Dice score of two N-d volumes for binary segmentation.

    :param s: The segmentation volume of numpy array.
    :param g: the ground truth volume of numpy array.
    :param resize: (optional, bool) 
        If s and g have different shapes, resize s to match g.
        Default is `True`.
    
    :return: The Dice value.
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
    """
    Calculate the Dice score given the image names of binary segmentation 
    and ground truth, respectively.

    :param s_name: (str) The filename of segmentation result. 
    :param g_name: (str) The filename of ground truth.

    :return: The Dice value. 
    """
    s = load_image_as_nd_array(s_name)['data_array']
    g = load_image_as_nd_array(g_name)['data_array']
    dice = binary_dice(s, g)
    return dice


def binary_iou(s,g):
    """
    Calculate the IoU score of two N-d volumes for binary segmentation.

    :param s: The segmentation volume of numpy array.
    :param g: the ground truth volume of numpy array.
    
    :return: The IoU value.
    """
    assert(len(s.shape)== len(g.shape))
    intersecion = np.multiply(s, g)
    union = np.asarray(s + g >0, np.float32)
    iou = (intersecion.sum() + 1e-5)/(union.sum() + 1e-5)
    return iou

# Hausdorff and ASSD evaluation
def get_edge_points(img):
    """
    Get edge points of a binary segmentation result.

    :param img: (numpy.array) a 2D or 3D array of binary segmentation.
    :return: an edge map. 
    """
    dim = len(img.shape)
    if(dim == 2):
        strt = ndimage.generate_binary_structure(2,1)
    else:
        strt = ndimage.generate_binary_structure(3,1)
    ero  = ndimage.binary_erosion(img, strt)
    edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8) 
    return edge 


def binary_hd95(s, g, spacing = None):
    """
    Get the 95 percentile of hausdorff distance between a binary segmentation 
    and the ground truth.

    :param s: (numpy.array) a 2D or 3D binary image for segmentation.
    :param g: (numpy.array) a 2D or 2D binary image for ground truth.
    :param spacing: (list) A list for image spacing, length should be 2 or 3.
    
    :return: The HD95 value.
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert(image_dim == len(g.shape))
    if(spacing == None):
        spacing = [1.0] * image_dim
    else:
        assert(image_dim == len(spacing))
    s_dis = ndimage.distance_transform_edt(1-s_edge, sampling = spacing)
    g_dis = ndimage.distance_transform_edt(1-g_edge, sampling = spacing)
  
    dist_list1 = s_dis[g_edge > 0]
    dist_list1 = sorted(dist_list1)
    dist1 = dist_list1[int(len(dist_list1)*0.95)]
    dist_list2 = g_dis[s_edge > 0]
    dist_list2 = sorted(dist_list2)
    dist2 = dist_list2[int(len(dist_list2)*0.95)]
    return max(dist1, dist2)


def binary_assd(s, g, spacing = None):
    """
    Get the Average Symetric Surface Distance (ASSD) between a binary segmentation 
    and the ground truth.

    :param s: (numpy.array) a 2D or 3D binary image for segmentation.
    :param g: (numpy.array) a 2D or 2D binary image for ground truth.
    :param spacing: (list) A list for image spacing, length should be 2 or 3.
    
    :return: The ASSD value.
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert(image_dim == len(g.shape))
    if(spacing == None):
        spacing = [1.0] * image_dim
    else:
        assert(image_dim == len(spacing))
    s_dis = ndimage.distance_transform_edt(1-s_edge, sampling = spacing)
    g_dis = ndimage.distance_transform_edt(1-g_edge, sampling = spacing)

    ns = s_edge.sum()
    ng = g_edge.sum()
    s_dis_g_edge = s_dis * g_edge
    g_dis_s_edge = g_dis * s_edge
    assd = (s_dis_g_edge.sum() + g_dis_s_edge.sum()) / (ns + ng) 
    return assd

# relative volume error evaluation
def binary_relative_volume_error(s, g):
    """
    Get the Relative Volume Error (RVE) between a binary segmentation 
    and the ground truth.

    :param s: (numpy.array) a 2D or 3D binary image for segmentation.
    :param g: (numpy.array) a 2D or 2D binary image for ground truth.

    :return: The RVE value.
    """
    s_v = float(s.sum())
    g_v = float(g.sum())
    assert(g_v > 0)
    rve = abs(s_v - g_v)/g_v
    return rve

def get_binary_evaluation_score(s_volume, g_volume, spacing, metric):
    """
    Evaluate the performance of binary segmentation using a specified metric. 
    The metric options are {`dice`, `iou`, `assd`, `hd95`, `rve`, `volume`}. 

    :param s_volume: (numpy.array) a 2D or 3D binary image for segmentation.
    :param g_volume: (numpy.array) a 2D or 2D binary image for ground truth.
    :param spacing: (list) A list for image spacing, length should be 2 or 3.
    :param metric: (str) The metric name. 

    :return: The metric value.
    """
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
    elif(metric_lower == "hd95"):
        score = binary_hd95(s_volume, g_volume, spacing)
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

def get_multi_class_evaluation_score(s_volume, g_volume, label_list, fuse_label, spacing, metric):
    """
    Evaluate the segmentation performance  using a specified metric for a list of labels. 
    The metric options are {`dice`, `iou`, `assd`, `hd95`, `rve`, `volume`}. 
    If `fuse_label` is `True`, the labels in `label_list` will be merged as foreground
    and other labels will be merged as background as a binary segmentation result. 

    :param s_volume: (numpy.array) A 2D or 3D image for segmentation.
    :param g_volume: (numpy.array) A 2D or 2D image for ground truth.
    :param label_list: (list) A list of target labels. 
    :param fuse_label: (bool) Fuse the labels in `label_list` or not.
    :param spacing: (list) A list for image spacing, length should be 2 or 3.
    :param metric: (str) The metric name. 

    :return: The metric value list.
    """
    if(fuse_label):
        s_volume_sub = np.zeros_like(s_volume)
        g_volume_sub = np.zeros_like(g_volume)
        for lab in label_list:
            s_volume_sub = s_volume_sub + np.asarray(s_volume == lab, np.uint8)
            g_volume_sub = g_volume_sub + np.asarray(g_volume == lab, np.uint8)
        label_list = [1]
        s_volume = np.asarray(s_volume_sub > 0, np.uint8)
        g_volume = np.asarray(g_volume_sub > 0, np.uint8)
    score_list = []
    for label in label_list:
        temp_score = get_binary_evaluation_score(s_volume == label, g_volume == label,
                    spacing, metric)
        score_list.append(temp_score)
    return score_list

def evaluation(config):
    """
    Run evaluation of segmentation results based on a configuration dictionary `config`.
    The following fields should be provided in `config`:

    :param metric_list: (list) The list of metrics for evaluation. 
        The metric options are {`dice`, `iou`, `assd`, `hd95`, `rve`, `volume`}. 
    :param label_list: (list) The list of labels for evaluation. 
    :param label_fuse: (option, bool) If true, fuse the labels in the `label_list`
        as the foreground, and other labels as the background. Default is False. 
    :param organ_name: (str) The name of the organ for segmentation.
    :param ground_truth_folder_root: (str) The root dir of ground truth images. 
    :param segmentation_folder_root: (str or list) The root dir of segmentation images. 
        When a list is given, each list element should be the root dir of the results of one method. 
    :param evaluation_image_pair: (str) The csv file that provide the segmentation 
        images and the corresponding ground truth images. 
    :param ground_truth_label_convert_source: (optional, list) The list of source
        labels for label conversion in the ground truth. 
    :param ground_truth_label_convert_target: (optional, list) The list of target
        labels for label conversion in the ground truth. 
    :param segmentation_label_convert_source: (optional, list) The list of source
        labels for label conversion in the segmentation. 
    :param segmentation_label_convert_target: (optional, list) The list of target
        labels for label conversion in the segmentation.     
    """
    
    metric_list = config['metric_list']
    label_list = config['label_list']
    label_fuse = config.get('label_fuse', False)
    organ_name = config['organ_name']
    gt_root    = config['ground_truth_folder_root']
    seg_root   = config['segmentation_folder_root']
    if(not(isinstance(seg_root, tuple) or isinstance(seg_root, list))):
        seg_root = [seg_root]
    image_pair_csv = config['evaluation_image_pair']
    ground_truth_label_convert_source = config.get('ground_truth_label_convert_source', None)
    ground_truth_label_convert_target = config.get('ground_truth_label_convert_target', None)
    segmentation_label_convert_source = config.get('segmentation_label_convert_source', None)
    segmentation_label_convert_target = config.get('segmentation_label_convert_target', None)

    image_items = pd.read_csv(image_pair_csv)
    item_num    = len(image_items)
    
    for seg_root_n in seg_root: # for each segmentation method
        for metric in metric_list:
            score_all_data = []
            name_score_list= []
            for i in range(item_num):
                gt_name  = image_items.iloc[i, 0]
                seg_name = image_items.iloc[i, 1]
                # seg_name = seg_name.replace(".nii.gz", "_pred.nii.gz")
                gt_full_name  = gt_root  + '/' + gt_name
                seg_full_name = seg_root_n + '/' + seg_name
                
                s_dict = load_image_as_nd_array(seg_full_name)
                g_dict = load_image_as_nd_array(gt_full_name)
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

                score_vector = get_multi_class_evaluation_score(s_volume, g_volume, label_list, 
                    label_fuse, s_spacing, metric )
                if(len(label_list) > 1):
                    score_vector.append(np.asarray(score_vector).mean())
                score_all_data.append(score_vector)
                name_score_list.append([seg_name] + score_vector)
                print(seg_name, score_vector)
            score_all_data = np.asarray(score_all_data)
            score_mean = score_all_data.mean(axis = 0)
            score_std  = score_all_data.std(axis = 0)
            name_score_list.append(['mean'] + list(score_mean))
            name_score_list.append(['std'] + list(score_std))
        
            # save the result as csv 
            score_csv = "{0:}/{1:}_{2:}_all.csv".format(seg_root_n, organ_name, metric)
            with open(score_csv, mode='w') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',', 
                                quotechar='"',quoting=csv.QUOTE_MINIMAL)
                head = ['image'] + ["class_{0:}".format(i) for i in label_list]
                if(len(label_list) > 1):
                    head = head + ["average"]
                csv_writer.writerow(head)
                for item in name_score_list:
                    csv_writer.writerow(item)

            print("{0:} mean ".format(metric), score_mean)
            print("{0:} std  ".format(metric), score_std) 

def main():
    """
    Main function for evaluation of segmentation results. 
    A configuration file is needed for runing. e.g., 
    
    .. code-block:: none

        pymic_evaluate_cls config.cfg

    The configuration file should have an `evaluation` section.
    See :mod:`pymic.util.evaluation_seg.evaluation` for details of the configuration required.
    """
    if(len(sys.argv) < 2):
        print('Number of arguments should be 2. e.g.')
        print('    pymic_evaluate_seg config.cfg')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    config = parse_config(config_file)['evaluation']
    evaluation(config)
    
if __name__ == '__main__':
    main()
