"""
Pure Python-based installation instructions (Linux):

```
python -m venv ./env
source ./env/bin/activate
python -m pip install .
```

Conda-based installation instructions (Linux):

```
conda create -p ./env python=3.8
conda activate ./env
pip install .
```
"""

import os
from setuptools import find_packages, setup
import subprocess
import sys

TORCH_REQUIREMENT = 'torch>=1.8,<2.0'

# Install PyTorch 'manually' because it has to be installed for
# PyTorch3D. `pip` dependency resolution cannot fix this.
# This is the officially recommended way to use `pip` in scripts.
subprocess.check_call(
    [sys.executable, '-m', 'pip', 'install', TORCH_REQUIREMENT])

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
        (
            'pytorch_minimize '
            '@ git+https://github.com/janEbert/pytorch-minimize.git'
        ),
        'tensorboard>=2.0,<3.0',
        TORCH_REQUIREMENT,
        'torchvision==0.9.1',
        'yacs==0.1.8',
    ],
    packages=find_packages(
        where='.',
        exclude=['TestConfigs', 'WorkingConfigs'],
    ),
)
