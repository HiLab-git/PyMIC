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
    version = "0.2.1",
    author  ='PyMIC Consortium',
    author_email = 'wguotai@gmail.com',
    description  = description,
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url      = 'https://github.com/ihil/PyMIC',
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
            'pymic_net_run_cls  = pymic.net_run.net_run_cls:main',
            'pymic_net_run_seg  = pymic.net_run.net_run_seg:main',
            'pymic_evaluate_cls = pymic.util.evaluation_cls:main',
            'pymic_evaluate_seg = pymic.util.evaluation_seg:main'
        ],
    },
)
