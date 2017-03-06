#!/usr/bin/env python
__doc__ = """

DaraProvider classes.

Kisuk Lee <kisuklee@mit.edu>, 2017
"""

from collections import OrderedDict
import numpy as np

from .augmentation.augmentor import DataAugment, Augmentor
from dataset import Dataset, VolumeDataset
from transformer import Transform, Transformer
from sequence import *

class DataProvider(object):
    """
    DataProvider interface.
    """

    def next_sample(self):
        raise NotImplementedError

    def random_sample(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class VolumeDataProvider(DataProvider):
    """
    DataProvider for volumetric data.

    Attributes:
        datasets: List of datasets.
        augmentor: Sampler augmentor.
        transformer: Sample transformer.
    """

    def __init__(self, datasets=None, aug=None, tf=None):
        # Datasets.
        self.datasets = list()
        if datasets is not None:
            for d in datasets:
                self.add_dataset(d)
        # Data augmentation.
        self.set_augmentor(aug)
        # Sample transformation.
        self.set_transformer(tf)

    def add_dataset(self, dataset):
        assert isinstance(dataset, Dataset)
        self.datasets.append(dataset)

    def add_augment(self, aug):
        assert isinstance(aug, DataAugment)
        self.augment.append(aug)

    def add_transform(self, tf):
        assert isinstance(tf, Transform)
        self.transform.append(tf)

    def set_augmentor(self, aug):
        if isinstance(aug, Augmentor):
            self.augment = aug
        else:
            self.augment = Augmentor()

    def set_transformer(self, tf):
        if isinstance(tf, Transformer):
            self.transform = tf
        else:
            self.transform = Transformer()

    def next_sample(self, **kwargs):
        """Fetch the next sample in a predefined sequence, if any."""
        return self._sample('next', **kwargs)

    def random_sample(self, **kwargs):
        """Fetch sample randomly."""
        return self._sample('random', **kwargs)

    def _sample(self, mode, **kwargs):
        assert len(self.datasets) > 0
        assert mode is 'random' or mode is 'next'
        # Pick one dataset randomly.
        drange = range(len(self.datasets))
        if 'drange' in kwargs:
            drange = kwargs['drange']
        idx = np.random.choice(len(drange), 1)
        dataset = self.datasets[idx]
        params  = dataset.get_params()
        params.update(kwargs)
        # Pick sample randomly.
        while True:
            try:
                spec = dataset.get_spec()
                spec = self.augment.prepare(spec, **params)
                sample = getattr(dataset, mode+'_sample')(spec=spec)
                break
            except:
                pass
        # Apply data augmentation.
        sample = self.augment(sample, **params)
        # Ensure that sample is ordered by key.
        sample = OrderedDict(sorted(sample.items(), key=lambda x: x[0]))
        return sample

    def __call__(self, mode, **kwargs):
        return self.transform(self._sample(mode, **kwargs), **kwargs)
