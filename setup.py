# setup.py
from setuptools import setup, find_packages

setup(
    name='neuralnetwork',
    version='0.1',
    packages=find_packages(include=['neuralnetwork']),
    install_requires=['numpy'],
    author='Mahdi Alhakim',
    description='A neural network library for feed forward neural networks.',
)
