# Author: Kenta Nakamura <c60evaporator@gmail.com>
# Copyright (c) 2025-2025 Kenta Nakamura
# License: MIT

from setuptools import setup, find_packages
import torch_extend

DESCRIPTION = "torch-extend: TorchVision extension for data loading, metrics calculation, and visialization"
NAME = 'torch-extend'
AUTHOR = 'Kenta Nakamura'
AUTHOR_EMAIL = 'c60evaporator@gmail.com'
URL = 'https://github.com/c60evaporator/torch-extend'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/c60evaporator/torch-extend'
VERSION = torch_extend.__version__
PYTHON_REQUIRES = ">=3.6"

INSTALL_REQUIRES = [
    'torch>=2.6.0',
    'torchvision>=0.21.0',
    'albumentations>=2.0.5',
    'transformers>=4.49.0',
    'torchmetrics>=1.6.3'
    'pycocotools>=2.0.8',
    'opencv-python>=4.11.0',
    'seaborn>=0.13.2',
    'scikit-learn>=1.6.1',
]

EXTRAS_REQUIRE = {
    'tutorial': [
        'tqdm>=4.66.1',
        'mlflow>=2.21.0'
    ]
}

CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Programming Language :: Python :: 3 :: Only',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Visualization',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Image Processing',
    'Topic :: Scientific/Engineering :: Image Recognition',
    'Framework :: Jupyter',
]

with open('README.rst', 'r') as fp:
    readme = fp.read()
with open('CONTACT.txt', 'r') as fp:
    contacts = fp.read()
long_description = readme + '\n\n' + contacts

setup(name=NAME,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=AUTHOR,
      maintainer_email=AUTHOR_EMAIL,
      description=DESCRIPTION,
      long_description=long_description,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      python_requires=PYTHON_REQUIRES,
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      packages=find_packages(),
      classifiers=CLASSIFIERS
    )
