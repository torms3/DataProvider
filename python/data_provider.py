#!/usr/bin/env python
__doc__ = """

DaraProvider classes.

Kisuk Lee <kisuklee@mit.edu>, 2015-2016
"""

import numpy as np
import parser
from dataset import *
from data_augmentation import DataAugmentor
from transform import *

class DataProvider(object):
    """
    DataProvider interface.
    """

    def __init__(self):
        pass

    def next_sample(self):
        pass

    def random_sample(self):
        pass


class VolumeDataProvider(DataProvider):
    """
    DataProvider for volumetric data.

    Attributes:
        _datasets:
        _sampling_weights:
        _net_spec:
    """

    def __init__(self, dspec_path, net_spec, params, drange, dprior=None):
        """
        Initialize DataProvider.

        Args:
            dspec_path: Path to the dataset specification file.
            net_spec: Net specification.
            params:
            drange:
            dprior:
        """
        # Build Datasets.
        p = parser.Parser(dspec_path, net_spec, params)
        self._datasets = []
        for dataset_id in drange:
            print 'constructing dataset %d...' % dataset_id
            config = p.parse_dataset(dataset_id)
            dataset = VolumeDataset(config)
            self._datasets.append(dataset)

        # TODO(kisuk): Process sampling weight.
        self._sampling_weights = [1.0/len(drange)] * len(drange)  # Temp

        # Setup data augmentation.
        self._data_aug = DataAugmentor(params['augment'])

    def next_sample(self):
        """Fetch next sample in a sample sequence."""
        return self.random_sample()

    def random_sample(self):
        # Pick one dataset randomly.
        dataset = self._get_random_dataset()

        # Draw a random sample and apply data augmenation.
        # sample, transform = self._data_aug.random_sample(dataset, self.net_spec)
        sample, transform = dataset.random_sample()

        # Apply transformation.
        return self._transform(sample, transform)

    ####################################################################
    ## Private Helper Methods
    ####################################################################

    def _get_random_dataset(self):
        """
        Pick one dataset randomly, according to the given sampling weights:

        Returns:
            Randomly chosen dataset.
        """
        # Take a single experiment with a multinomial distribution, whose
        # probabilities indicate how likely each sample be selected.
        # Output is an one-hot vector.
        sq = np.random.multinomial(1, self._sampling_weights, size=1)
        sq = np.squeeze(sq)

        # Get the index of non-zero element.
        idx = np.nonzero(sq)[0]

        return self._datasets[idx]

    def _transform(self, sample, transform):
        """
        TODO(kisuk): Documentation.
        """
        ret = {}
        for name, data in sample.iteritems():
            for tf in transform[name]:
                func = tf['type']
                del tf['type']
                data = transform_tensor(data, func, **tf)
            ret[name] = data
        return ret
