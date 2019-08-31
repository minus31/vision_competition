#nsml: nsml/ml:cuda9.0-cudnn7-tf-1.11torch1.0keras2.2
# PyTorch version - 1.1.0
from distutils.core import setup
setup(
    name='airush1',
    version='1.0',
    install_requires=[
            'tqdm',
            'torch>=1.0',
            'pickle-mixin',
            'torchvision',
            'pandas',
    ]
)
