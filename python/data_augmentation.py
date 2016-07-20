#!/usr/bin/env python
__doc__ = """

DataAugmentor

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import numpy as np
from transform import *

class DataAugmentor(object):
    """
    Data augmentation.

    Attributes:
        _aug_list: List of data augmentation. Will be executed sequentially.
    """

    def __init__(self, spec):
        """
        TODO(kisuk): Documentation.
        """
        aug_list = []
        for s in spec:
            t = s['type']
            del s['type']
            t = t[0].capitalize() + t[1:].lower() + 'Augment'
            if t not in globals():
                raise RuntimeError('unknown data augmentation type [%s]' % t)
            aug = eval(t + '(**s)')
            aug_list.append(aug)
        self._aug_list = aug_list

    def next_sample(self, dataset):
        raise NotImplementedError

    def random_sample(self, dataset):
        spec = self._prepare(dataset.get_spec())
        sample, transform = dataset.random_sample(spec=spec)
        for aug in self._aug_list:
            sample = aug.augment(sample)
        return sample, transform

    def _prepare(self, spec):
        ret = dict(spec)
        for aug in reversed(self._aug_list):
            ret = aug.prepare(ret)
        return ret


class DataAugment(object):
    """
    DataAugment interface.
    """

    def prepare(self, spec):
        raise NotImplementedError

    def augment(self, sample):
        raise NotImplementedError


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