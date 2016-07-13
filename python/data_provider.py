#!/usr/bin/env python
__doc__ = """

DaraProvider classes.

Kisuk Lee <kisuklee@mit.edu>, 2015-2016
"""

import numpy as np

class DataProvider(object):
    """
    DataProvider interface.
    """

    def __init__(self, spec):
        """
        Initialize DataProvider.

        Args:
            spec: contains every information needed for initialization.
        """
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

    def __init__(self, config, net_spec, drange=None, dprior=None):
        """
        Initialize DataProvider.

        Args:
        """

        _datasets = []
        _sampling_weights = []
        _net_spec = {}

        # TODO(kisuk): Build datasets based on config, range
        for i in drange:
            # TODO(kisuk): Build dataset.
            _datasets.append(dataset)

        # Setup data augmentation.

        # Setup label/mask processing.

        pass

    def next_sample(self):
        """Fetch next sample in a sample sequence."""
        return self.random_sample()

    def random_sample(self):

        # Sampling procedure:
        #   (0) Pick one dataset randomly.
        #   (1) Draw random parameters for data augmentation.
        #   (2) Compute new patch size required for data augmentation.
        #   (3) Set new patch size and draw a random sample.
        #   (4) Apply data augmentaion.
        #   (5) Transform label & mask (boundary, affinity, semantic, etc.).
        #   (6) Crop the final sample.

        # (0)
        dataset = self._get_random_dataset()

        # (1)
        # sample = DataAugmentor.random_sample(dataset, self._net_spec)

        # (5)
        # sample = transform(sample)

        # (6)
        # TODO(kisuk): Final crop.


        pass

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

        # Get the index of non-zero element
        idx = np.nonzero(sq)[0]

        return self._datasets[idx]



        pass

    def _transform_label(self, sample):
        """
        TODO(kisuk): Documentation.
        """
