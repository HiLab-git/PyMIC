# PyMIC: A Pytorch-Based Toolkit for Medical Image Computing

PyMIC is a pytorch-based toolkit for medical image computing with annotation-efficient deep learning. Despite that pytorch is a fantastic platform for deep learning, using it for medical image computing is not straightforward as medical images are often with high dimension and large volume, multiple modalities and difficulies in annotating. This toolkit is developed to facilitate medical image computing researchers so that training and testing deep learning models become easier. It is very friendly to researchers who are new to this area. Even without writing any code, you can use PyMIC commands to train and test a model by simply editing configuration files.  PyMIC is developed to support learning with imperfect labels, including semi-supervised, self-supervised, and weakly supervised learning, and learning with noisy annotations.

Currently PyMIC supports 2D/3D medical image classification and segmentation, and it is still under development. If you use this toolkit, please cite the following paper:

* G. Wang, X. Luo, R. Gu, S. Yang, Y. Qu, S. Zhai, Q. Zhao, K. Li, S. Zhang. (2023). 
[PyMIC: A deep learning toolkit for annotation-efficient medical image segmentation.][arxiv2022] Computer Methods and Programs in Biomedicine (CMPB). February 2023, 107398.

[arxiv2022]:http://arxiv.org/abs/2208.09350

BibTeX entry:

    @article{Wang2022pymic,
    author = {Guotai Wang and Xiangde Luo and Ran Gu and Shuojue Yang and Yijie Qu and Shuwei Zhai and Qianfei Zhao and Kang Li and Shaoting Zhang},
    title = {{PyMIC: A deep learning toolkit for annotation-efficient medical image segmentation}},
    year = {2023},
    url = {https://doi.org/10.1016/j.cmpb.2023.107398},
    journal = {Computer Methods and Programs in Biomedicine},
    volume = {231},
    pages = {107398},
    }

# Features
PyMIC provides flixible modules for medical image computing tasks including classification and segmentation. It currently provides the following functions:
* Support for annotation-efficient image segmentation, especially for semi-supervised, self-supervised, self-supervised, weakly-supervised and noisy-label learning.
* User friendly: For beginners, you only need to edit the configuration files for model training and inference, without writing code. For advanced users, you can customize different modules (networks, loss functions, training pipeline, etc) and easily integrate them into PyMIC.
* Easy-to-use I/O interface to read and write different 2D and 3D images.
* Various data pre-processing/transformation methods before sending a tensor into a network.
* Implementation of typical neural networks for medical image segmentation.
* Re-useable training and testing pipeline that can be transferred to different tasks.
* Evaluation metrics for quantitative evaluation of your methods. 

# Usage
## Requirement
* [Pytorch][torch_link] version >=1.0.1
* [TensorboardX][tbx_link] to visualize training performance
* Some common python packages such as Numpy, Pandas, SimpleITK
* See `requirements.txt` for details.

[torch_link]:https://pytorch.org/
[tbx_link]:https://github.com/lanpa/tensorboardX 

## Installation
Run the following command to install the latest released version of PyMIC:

```bash
pip install PYMIC
```
To install a specific version of PYMIC such as 0.4.1, run:

```bash
pip install PYMIC==0.4.1
```
Alternatively, you can download the source code for the latest version. Run the following command to compile and install:

```bash
python setup.py install
``` 

## How to start
* [PyMIC_examples][exp_link] shows some examples of starting to use PyMIC. 
* [PyMIC_doc][docs_link] provides documentation of this project. 

[docs_link]:https://pymic.readthedocs.io/en/latest/
[exp_link]:https://github.com/HiLab-git/PyMIC_examples 

## Projects based on PyMIC
Using PyMIC, it becomes easy to develop deep learning models for different projects, such as the following:

1, [MyoPS][myops] Winner of the MICCAI 2020 myocardial pathology segmentation (MyoPS) Challenge.

2, [COPLE-Net][coplenet] (TMI 2020), COVID-19 Pneumonia Segmentation from CT images. 

3, [Head-Neck-GTV][hn_gtv] (NeuroComputing 2020) Nasopharyngeal Carcinoma (NPC) GTV segmentation from Head and Neck CT images. 

4, [UGIR][ugir] (MICCAI 2020) Uncertainty-guided interactive refinement for medical image segmentation. 

[myops]: https://github.com/HiLab-git/MyoPS2020
[coplenet]:https://github.com/HiLab-git/COPLE-Net
[hn_gtv]: https://github.com/HiLab-git/Head-Neck-GTV
[ugir]: https://github.com/HiLab-git/UGIR

