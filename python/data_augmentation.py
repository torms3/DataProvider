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
        _aug_list:
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

    def next_sample(self, dataset, spec):
        spec = self._prepare(spec)
        sample, transform = dataset.next_sample(spec=spec)
        for aug in self._aug_list:
            sample = aug.augment(sample)
        return sample, transform

    def random_sample(self, dataset, spec):
        spec = self._prepare(spec)
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

    def __init__(self):
        pass

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


if __name__ == "__main__":

    params = {}
    params['augment'] = [{'type':'flip'},
                         {'type':'warp'}]

    d = DataAugmentor(params['augment'])