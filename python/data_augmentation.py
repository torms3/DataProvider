#!/usr/bin/env python
__doc__ = """

DataAugmentor

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import numpy as np
from dataset import *
from transform import *

class DataAugmentor(object):
    """
    Data augmentation.
    """

    def __init__(self):
        pass

    def random_sample(self, dataset, spec):
        pass


class DataAugment(object):
    """
    DataAugment interface.
    """

    def __init__(self):
        pass

    def prepare(self, spec):
        pass

    def augment(self, sample):
        pass


class FlipAugment(DataAugment):
    """
    Random flip.
    """

    def prepare(self, spec):
        return dict(spec)

    def augment(self, sample):
        rule = np.random.rand(4) > 0.5
        return transform_sample(sample, 'flip', rule=rule)


class WarpAugment(DataAugment):
    """
    Warping.
    """

    def prepare(self, spec):
        pass

    def augment(self, sample):
        pass