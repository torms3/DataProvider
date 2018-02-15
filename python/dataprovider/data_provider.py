from __future__ import print_function
from collections import OrderedDict
import numpy as np

from .augmentation.augmentor import DataAugment, Augmentor
from .dataset import Dataset, VolumeDataset
from .transformer import Transform, Transformer
from .sequence import *

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
        preprocessor: Sample transformer, before augmentation.
        augmentor: Sample augmentor.
        postprocessor: Sample transformer, after augmentation.
    """

    def __init__(self, datasets=None, pre=None, aug=None, post=None):
        # Datasets.
        self.datasets = list()
        if datasets is not None:
            for d in datasets:
                self.add_dataset(d)
        # Preprocessing.
        self.set_preprocessor(pre)
        # Data augmentation.
        self.set_augmentor(aug)
        # Postprocessing.
        self.set_postprocessor(post)
        # Sampling weights.
        self.set_sampling_weights()

    def set_sampling_weights(self, p=None):
        """Set probability of each dataset being chosen at each sampling."""
        if p is None:
            assert np.all([d.has_spec() for d in self.datasets])
            p = [d.num_sample() for d in self.datasets]
        # Normalize.
        p = np.asarray(p, dtype='float32')
        p = p/np.sum(p)
        # Set sampling weights.
        # print('Sampling weights: {}'.format(['%0.3f' % x for x in p]))
        assert len(p)==len(self.datasets)
        self.p = p

    def random_dataset(self, **kwargs):
        """Pick one dataset randomly."""
        assert len(self.datasets) > 0
        drange = range(len(self.datasets))
        if 'drange' in kwargs:
            drange = kwargs['drange']
        p = [self.p[d] for d in drange]
        # Normalize again.
        p = np.array(p)
        p /= p.sum()
        idx = np.random.choice(len(drange), size=1, p=p)
        return self.datasets[idx[0]]

    def next_sample(self, **kwargs):
        """Fetch the next sample in a predefined sequence, if any."""
        return self._sample('next', **kwargs)

    def random_sample(self, **kwargs):
        """Fetch a sample randomly."""
        return self._sample('random', **kwargs)

    def _sample(self, mode, **kwargs):
        assert mode in ['random', 'next']
        dataset = self.random_dataset(**kwargs)
        params = dataset.get_params()  ## Dataset-specific parameters.
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
        # Preprocessing.
        sample = self.preprocess(sample, **params)
        # Apply data augmentation.
        sample = self.augment(sample, **params)
        # Postprocessing.
        sample = self.postprocess(sample, **params)
        # Ensure that sample is ordered by key.
        return OrderedDict(sorted(sample.items(), key=lambda x: x[0]))

    def __call__(self, mode, **kwargs):
        return self._sample(mode, **kwargs)

    ####################################################################
    ## Setters.
    ####################################################################

    def add_dataset(self, dataset):
        assert isinstance(dataset, Dataset)
        self.datasets.append(dataset)

    def add_preprocess(self, tf):
        assert isinstance(tf, Transform)
        self.preprocess.append(tf)

    def add_augment(self, aug):
        assert isinstance(aug, DataAugment)
        self.augment.append(aug)

    def add_postprocess(self, tf):
        assert isinstance(tf, Transform)
        self.postprocess.append(tf)

    def set_preprocessor(self, tf):
        if isinstance(tf, Transformer):
            self.preprocess = tf
        else:
            self.preprocess = Transformer()

    def set_augmentor(self, aug):
        if isinstance(aug, Augmentor):
            self.augment = aug
        else:
            self.augment = Augmentor()

    def set_postprocessor(self, tf):
        if isinstance(tf, Transformer):
            self.postprocess = tf
        else:
            self.postprocess = Transformer()
