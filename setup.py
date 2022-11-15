"""
Conda-based installation instructions:

```
conda create -p ./env python=3.8
conda activate ./env
conda install pytorch torchvision cudatoolkit=11.6 -c pytorch -c conda-forge
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
pip install -r requirements.txt
```
"""

import os
from setuptools import find_packages, setup

os.environ['PYTORCH3D_NO_EXTENSION'] = '1'

setup(
    name='diff_stral',
    python_requires='>=3.6<3.10',
    version='0.0.1',
    install_requires=[
        'matplotlib>=3.4,<4.0',
        'numpy>=1.17,<2.0',
        (
            'pytorch3d '
            '@ git+https://github.com/facebookresearch/pytorch3d.git@stable'
        ),
        'tensorboard>=2.0,<3.0',
        'torch>=1.8,<2.0',
        'torchvision==0.9.1',
        'yacs==0.1.8',
    ],
    packages=find_packages(
        where='.',
        exclude=['TestConfigs', 'WorkingConfigs'],
    ),
)
