# PyMIC: A Pytorch-Based Toolkit for Medical Image Computing

PyMIC is a pytorch-based toolkit for medical image computing with deep learning. Despite that pytorch is a fantastic platform for deep learning, using it for medical image computing is not straightforward as medical images are often with higher dimension, multiple modalities and low contrast. The toolkit is developed to facilitate medical image computing researchers so that training and testing deep learning models become easier. It is very friendly to researchers who are new to this area. Even without writing any code, you can use PyMIC commands to train and test a model by simply editing configure files.  

Currently PyMIC supports 2D/3D medical image classification and segmentation, and it is still under development. It was originally developed for COVID-19 pneumonia lesion segmentation from CT images. If you use this toolkit, please cite the following paper:


*  G. Wang, X. Liu, C. Li, Z. Xu, J. Ruan, H. Zhu, T. Meng, K. Li, N. Huang, S. Zhang. 
[A Noise-robust Framework for Automatic Segmentation of COVID-19 Pneumonia Lesions from CT Images.][tmi2020] IEEE Transactions on Medical Imaging. 39(8):2653-2663, 2020. DOI: [10.1109/TMI.2020.3000314][tmi2020]

[tmi2020]:https://ieeexplore.ieee.org/document/9109297


# Advantages
PyMIC provides some basic modules for medical image computing that can be share by different applications. We currently provide the following functions:
* Easy-to-use I/O interface to read and write different 2D and 3D images.
* Re-useable training and testing pipeline that can be transferred to different tasks.
* Various data pre-processing methods before sending a tensor into a network.
* Implementation of loss functions, especially for image segmentation.
* Implementation of evaluation metrics to get quantitative evaluation of your methods (for segmentation). 

# Usage
## Requirement
* [Pytorch][torch_link] version >=1.0.1
* [TensorboardX][tbx_link] to visualize training performance
* Some common python packages such as Numpy, Pandas, SimpleITK

[torch_link]:https://pytorch.org/
[tbx_link]:https://github.com/lanpa/tensorboardX 

## Installation
Run the following command to install the current released version of PyMIC:

```bash
pip install PYMIC
```
To install a specific version of PYMIC such as 0.2.4, run:

```bash
pip install PYMIC==0.2.4
```
Alternatively, you can download the source code for the latest version. Run the following command to compile and install:

```bash
python setup.py install
``` 

## Examples
[PyMIC_examples][examples] provides some examples of starting to use PyMIC. For beginners, you only need to simply change the configuration files to select different datasets, networks and training methods for running the code. For advanced users, you can develop your own modules based on this package. You can find both types of examples 

[examples]: https://github.com/HiLab-git/PyMIC_examples 

# Projects based on PyMIC
Using PyMIC, it becomes easy to develop deep learning models for different projects, such as the following:

1, [MyoPS][myops] Winner of the MICCAI 2020 myocardial pathology segmentation (MyoPS) Challenge.

2, [COPLE-Net][coplenet] (TMI 2020), COVID-19 Pneumonia Segmentation from CT images. 

3, [Head-Neck-GTV][hn_gtv] (NeuroComputing 2020) Nasopharyngeal Carcinoma (NPC) GTV segmentation from Head and Neck CT images. 

4, [UGIR][ugir] (MICCAI 2020) Uncertainty-guided interactive refinement for medical image segmentation. 

[myops]: https://github.com/HiLab-git/MyoPS2020
[coplenet]:https://github.com/HiLab-git/COPLE-Net
[hn_gtv]: https://github.com/HiLab-git/Head-Neck-GTV
[ugir]: https://github.com/HiLab-git/UGIR

