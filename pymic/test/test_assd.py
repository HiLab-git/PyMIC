from scipy import ndimage
from PIL import Image
import numpy as np 
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pymic.util.evaluation_seg import get_edge_points

def test_assd_2d():
    img_name = "/home/x/projects/PyMIC_project/PyMIC_examples/PyMIC_data/JSRT/label/JPCLN001.png"
    img = Image.open(img_name)
    img_array = np.asarray(img)
    img_edge = get_edge_points(img_array > 0)
    s_dis = ndimage.distance_transform_edt(1-img_edge)
    plt.subplot(1,2,1)
    plt.imshow(img_edge)
    plt.subplot(1,2,2)
    plt.imshow(s_dis)
    plt.show()

def test_assd_3d():
    # img_name = "/home/x/projects/PyMIC_project/PyMIC_examples/seg_ssl/ACDC/result/unet2d_baseline/patient001_frame01.nii.gz"
    img_name = "/home/disk4t/data/heart/ACDC/preprocess/patient001_frame12_gt.nii.gz"
    img_obj = sitk.ReadImage(img_name) 
    spacing = img_obj.GetSpacing()
    spacing = spacing[::-1]
    img_data = sitk.GetArrayFromImage(img_obj)
    print(img_data.shape)
    print(spacing)
    img_edge = get_edge_points(img_data > 0)
    s_dis = ndimage.distance_transform_edt(1-img_edge, sampling=spacing)
    dis_obj = sitk.GetImageFromArray(s_dis)
    dis_obj.CopyInformation(img_obj)
    sitk.WriteImage(dis_obj, "test_dis.nii.gz")



if __name__ == "__main__":
    test_assd_3d()