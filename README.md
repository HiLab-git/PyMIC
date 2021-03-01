# PyMIC: A Pytorch-Based Toolkit for Medical Image Computing

This repository proivdes a library and some examples of using pytorch for medical image computing. The toolkit is under development. Currently it supports 2D and 3D image segmentation. It was originally developped for COVID-19 pneumonia lesion segmentation from CT images. If you use this toolkit, please cite the following paper:

*  G. Wang, X. Liu, C. Li, Z. Xu, J. Ruan, H. Zhu, T. Meng, K. Li, N. Huang, S. Zhang. 
[A Noise-robust Framework for Automatic Segmentation of COVID-19 Pneumonia Lesions from CT Images.][tmi2020] IEEE Transactions on Medical Imaging. 39(8):2653-2663, 2020. DOI: [10.1109/TMI.2020.3000314][tmi2020]

[tmi2020]:https://ieeexplore.ieee.org/document/9109297

# Requirement
* [Pytorch][torch_link] version >=1.0.1
* [TensorboardX][tbx_link] to visualize training performance
* Some common python packages such as Numpy, Pandas, SimpleITK

[torch_link]:https://pytorch.org/
[tbx_link]:https://github.com/lanpa/tensorboardX 

# Advantages
This package provides some basic modules for medical image computing that can be share by different applications. We currently provide the following functions:
* Easy-to-use I/O interface to read and write different 2D and 3D images.
* Re-userable training and testing pipeline that can be transfered to different tasks.
* Various data pre-processing methods before sending a tensor into a network.
* Implementation of loss functions (for image segmentation).
* Implementation of evaluation metrics to get quantitative evaluation of your methods (for segmentation). 

# Usage
Run the following command to install PyMIC:

```bash
pip install PYMIC
```

[PyMIC_examples][examples] provides some examples for using PyMIC. For beginners, you only need to simply change the configuration files to select different datasets, networks and training methods for running the code. For advanced users, you can develop your own modules based on this package. You can find both types of examples 

[examples]: https://github.com/HiLab-git/PyMIC_examples 

# Projects based on PyMIC
Using PyMIC, it becomes easy to develop deep learning models for different projects, such as the following:

1, [COPLE-Net][coplenet] (TMI 2020), COVID-19 Pneumonia Segmentation from CT images. 

2, [Head-Neck-GTV][hn_gtv] (NeuroComputing 2020) Nasopharyngeal Carcinoma (NPC) GTV segmentation from Head and Neck CT images. 

3, [UGIR][ugir] (MICCAI 2020) Uncertainty-guided interactive refinement for medical image segmentation. 

[coplenet]:https://github.com/HiLab-git/COPLE-Net
[hn_gtv]: https://github.com/HiLab-git/Head-Neck-GTV
[ugir]: https://github.com/HiLab-git/UGIR

