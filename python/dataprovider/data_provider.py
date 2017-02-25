#!/usr/bin/env python
__doc__ = """

DaraProvider classes.

Kisuk Lee <kisuklee@mit.edu>, 2017
"""

from collections import OrderedDict
import numpy as np

from .augmentation.augmentor import Augmentor
from dataset import VolumeDataset
from transformer import Transformer

class DataProvider(object):
    """
    DataProvider interface.
    """

    def next_sample(self):
        raise NotImplementedError

    def random_sample(self):
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
        self.set_augment(aug)
        # Sample transformation.
        self.set_transform(tf)

    def add_dataset(self, dataset):
        assert isinstance(dataset, Dataset)
        self.datasets.append(dataset)

    def set_augment(self, aug):
        if isinstance(aug, Augmentor):
            self.augment = aug
        else:
            self.augment = Augmentor()

    def set_transform(self, tf):
        if isinstance(tf, Transformer):
            self.transform = tf
        else:
            self.transform = Transformer()

    def next_sample(self):
        """Fetch the next sample in a predefined sequence, if any."""
        raise NotImplementedError

    def random_sample(self, **kwargs):
        """Fetch sample randomly."""
        # Pick one dataset randomly.
        drange = range(len(self.datasets))
        if 'drange' in kwargs:
            drange = kwargs['drange']
        idx = np.random.choice(len(drange), 1)
        dataset = self.datasets[idx]
        spec    = dataset.get_spec()
        params  = dataset.get_params()
        # Pick sample randomly.
        while True:
            try:
                spec = self.augment.prepare(spec, **params)
                sample = dataset.random_sample(spec=spec)
                break
            except:
                pass
        # Apply data augmentation.
        sample = self.augment(sample, **params)
        # Ensure that sample is ordered by key.
        sample = OrderedDict(sorted(sample.items(), key=lambda x: x[0]))
        return sample

    def __call__(self, **kwargs):
        return self.transform(self.random_sample(**kwargs), **kwargs)
