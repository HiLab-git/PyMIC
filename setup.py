# -*- coding: utf-8 -*-
import setuptools 

# Get the summary
description = 'An open-source deep learning platform' + \
              ' for medical image computing'

# Get the long description
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name    = 'PYMIC',
    version = "0.2.5",
    author  ='PyMIC Consortium',
    author_email = 'wguotai@gmail.com',
    description  = description,
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url      = 'https://github.com/HiLab-git/PyMIC',
    license  = 'Apache 2.0',
    packages = setuptools.find_packages(),
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],
    python_requires = '>=3.6',
    entry_points = {
        'console_scripts': [
            'pymic_run  = pymic.net_run.net_run:main',
            'pymic_ssl  = pymic.net_run_ssl.ssl_main:main',
            'pymic_eval_cls = pymic.util.evaluation_cls:main',
            'pymic_eval_seg = pymic.util.evaluation_seg:main'
        ],
    },
)
