# PyMIC: A Pytorch-Based Toolkit for Medical Image Computing

This repository proivdes a library and some examples of using pytorch for medical image computing. The package is under development. Currently it supports 2D and 3D image segmentation.

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

# Examples
Go to 'examples' to see some examples for using PyMIC. For beginners, you only need to simply change the configuration files to select different datasets, networks and training methods for running the code. For advanced users, you can develop your own modules based on this packange.
