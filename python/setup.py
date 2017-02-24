#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='dataprovider',
    version='0.0.1',
    description='Deep learning platform-independent volumetric data provider for 3D neural nets',
    author='Kisuk Lee',
    author_email='kisuklee@mit.edu',
    url='https://github.com/torms3/DataProvider',
    # packages=find_packages()
    packages=['dataprovider','dataprovider.augmentation'],
)
