## Welcome to PyMIC 

PyMIC is a pytorch-based toolkit for medical image computing with annotation-efficient deep learning. Despite that pytorch is a fantastic platform for deep learning, using it for medical image computing is not straightforward as medical images are often with high dimension and large volume, multiple modalities and difficulies in annotating. This toolkit is developed to facilitate medical image computing researchers so that training and testing deep learning models become easier. It is very friendly to researchers who are new to this area. Even without writing any code, you can use PyMIC commands to train and test a model by simply editing configuration files.  PyMIC is developed to support learning with imperfect labels, including semi-supervised and weakly supervised learning, and learning with noisy annotations.

Currently PyMIC supports 2D/3D medical image classification and segmentation, and it is still under development. It was originally developed for COVID-19 pneumonia lesion segmentation from CT images.

### Features
PyMIC provides flixible modules for medical image computing tasks including classification and segmentation. It currently provides the following functions:
* Support for annotation-efficient image segmentation, especially for semi-supervised, weakly-supervised and noisy-label learning.
* User friendly: For beginners, you only need to edit the configuration files for model training and inference, without writing code. For advanced users, you can customize different modules (networks, loss functions, training pipeline, etc) and easily integrate them into PyMIC.
* Easy-to-use I/O interface to read and write different 2D and 3D images.
* Various data pre-processing/transformation methods before sending a tensor into a network.
* Implementation of typical neural networks for medical image segmentation.
* Re-useable training and testing pipeline that can be transferred to different tasks.
* Evaluation metrics for quantitative evaluation of your methods.

### Installation
Run the following command to install the current released version of PyMIC:

```bash
pip install PYMIC
```

Alternatively, you can download the source code for the latest version. Run the following command to compile and install:

```bash
python setup.py install
``` 

### Quick Start
[PyMIC_examples][examples] provides some examples of starting to use PyMIC. At the beginning, you only need to  edit the configuration files to select different datasets, networks and training methods for running the code. When you are more familiar with PyMIC, you can customize different modules in the PyMIC package. You can find both types of examples: 

[examples]: https://github.com/HiLab-git/PyMIC_examples

