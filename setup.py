# -*- coding: utf-8 -*-
import setuptools 

# Get the summary
description = 'An open-source deep learning platform' + \
              ' for annotation-efficient medical image computing'

# Get the long description
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name    = 'PYMIC',
    version = "0.4.1",
    author  ='PyMIC Consortium',
    author_email = 'wguotai@gmail.com',
    description  = description,
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url      = 'https://github.com/HiLab-git/PyMIC',
    license  = 'Apache 2.0',
    packages = setuptools.find_packages(),
    install_requires=[
        "h5py",
        "matplotlib>=3.1.2",
        "numpy>=1.17.4",
        "pandas>=0.25.3",
        "scikit-image>=0.16.2",
        "scikit-learn>=0.22",
        "scipy>=1.3.3",
        "SimpleITK>=2.0.0",
        "tensorboard",
        "tensorboardX",
    ],
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],
    python_requires = '>=3.6',
    entry_points = {
        'console_scripts': [
            'pymic_preprocess = pymic.net_run.preprocess:main',
            'pymic_train = pymic.net_run.train:main',
            'pymic_test  = pymic.net_run.predict:main',
            'pymic_eval_cls = pymic.util.evaluation_cls:main',
            'pymic_eval_seg = pymic.util.evaluation_seg:main'
        ],
    },
)
