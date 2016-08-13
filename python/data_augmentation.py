#!/usr/bin/env python
__doc__ = """

DataAugmentor

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

from collections import OrderedDict
import numpy as np
from transform import *

"""
Data augmentaion pool.

Whenever adding new data augmentation outside this module, its type name should
be appended to this list.
"""
aug_pool = ['flip','warp','misalign']


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
            t = t.lower()
            if t not in aug_pool:
                raise RuntimeError('unknown data augmentation type [%s]' % t)
            t = t[0].capitalize() + t[1:] + 'Augment'
            aug = eval(t + '(**s)')
            aug_list.append(aug)
        self._aug_list = aug_list

    def next_sample(self, dataset):
        raise NotImplementedError

    def random_sample(self, dataset):
        """
        TODO(kisuk): Documentation.
        """
        spec = self._prepare(dataset)
        sample, transform = dataset.random_sample(spec=spec)
        for aug in self._aug_list:
            sample = aug.augment(sample)
        # Ensure that sample is ordered by key.
        sample = OrderedDict(sorted(sample.items(), key=lambda x: x[0]))
        return sample, transform

    def _prepare(self, dataset):
        ret = dict(dataset.get_spec())
        for aug in reversed(self._aug_list):
            ret = aug.prepare(ret, **dataset.params)
        return ret


class DataAugment(object):
    """
    DataAugment interface.
    """

    def prepare(self, spec, **kwargs):
        raise NotImplementedError

    def augment(self, sample):
        raise NotImplementedError

"""
Whenever adding new data augmentation outside this module, it should be properly
imported below.
"""

class FlipAugment(DataAugment):
    """
    Random flip.
    """

    def prepare(self, spec, **kwargs):
        return dict(spec)

    def augment(self, sample):
        rule = np.random.rand(4) > 0.5
        return sample_func.flip(sample, rule=rule)


class WarpAugment(DataAugment):
    """
    Warping.
    """

    def prepare(self, spec, **kwargs):
        pass

    def augment(self, sample):
        pass


from misalign import MisalignAugment
