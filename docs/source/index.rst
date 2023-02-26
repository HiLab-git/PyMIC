Welcome to PyMIC's documentation!
===================================

PyMIC is a pytorch-based toolkit for medical image computing with annotation-efficient 
deep learning. PyMIC is developed to support learning with imperfect labels, including 
semi-supervised and weakly supervised learning, and learning with noisy annotations.

Check out the :doc:`installation` section for install PyMIC, and go to the :doc:`usage`
section for understanding modules for the segmentation pipeline designed in PyMIC. 
Please follow `PyMIC_examples <https://github.com/HiLab-git/PyMIC_examples/>`_
to quickly start with using PyMIC. 

.. note::

   This project is under active development. It will be updated later.


.. toctree::
    :maxdepth: 2
    :caption: Getting Started

    installation
    usage
    api

Citation
--------

If you use PyMIC for your research, please acknowledge it accordingly by citing our paper:

`G. Wang, X. Luo, R. Gu, S. Yang, Y. Qu, S. Zhai, Q. Zhao, K. Li, S. Zhang. 
PyMIC: A deep learning toolkit for annotation-efficient medical image segmentation. 
Computer Methods and Programs in Biomedicine (CMPB). 231 (2023): 107398. <http://arxiv.org/abs/2208.09350>`_


BibTeX entry:

.. code-block:: none

    @article{Wang2022pymic,
    author = {Guotai Wang and Xiangde Luo and Ran Gu and Shuojue Yang and Yijie Qu and Shuwei Zhai and Qianfei Zhao and Kang Li and Shaoting Zhang},
    title = {{PyMIC: A deep learning toolkit for annotation-efficient medical image segmentation}},
    year = {2022},
    url = {https://doi.org/10.1016/j.cmpb.2023.107398},
    journal = {Computer Methods and Programs in Biomedicine},
    volume = {231},
    pages = {107398},
    }
