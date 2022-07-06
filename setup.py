from setuptools import find_packages, setup

setup(
    name='diff_stral',
    python_requires='>=3.6<3.10',
    version='0.0.1',
    install_requires=[
        'matplotlib>=3.4<4.0',
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
