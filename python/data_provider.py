#!/usr/bin/env python
__doc__ = """

DaraProvider classes.

Kisuk Lee <kisuklee@mit.edu>, 2015-2016
"""

import numpy as np
import parser

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
        self._datasets = []
        self._net_spec = net_spec

        # TODO(kisuk): Process sampling weight.

        # Build Datasets.
        p = parser.Parser(dspec_path, net_spec, params)
        for dataset_id in drange:
            config = p.parse_dataset(dataset_id)
            dataset = VolumeDataset(config)
            self._datasets.append(dataset)

        # TODO(kisuk): Setup data augmentation.

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

        # (1) Increment spec size by 1 if affinity.
        # spec = self._net_spec + 1

        # (2)
        # sample = DataAugmentor.random_sample(dataset, spec)

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
