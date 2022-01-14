from setuptools import setup

setup(
    name='diff_stral',
    python_requires='>=3.6<3.10',
    version='0.0.1',
    install_requires=[
        'matplotlib>=3.4<4.0,',
        'numpy>=1.17,<2.0',
        'tensorboard>=2.0,<3.0',
        'torch>=1.8,<2.0',
        'yacs>=0.1,<0.2',
    ],
)
