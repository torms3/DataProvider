#!/usr/bin/env python
__doc__ = """

DaraProvider classes.

Kisuk Lee <kisuklee@mit.edu>, 2015-2016
"""

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
    """

    def __init__(self, config_path):
        """
        Initialize DataProvider.

        Args:
        """
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
        # dataset

        # (1)
        # sample = DataAugmentor.random_sample(dataset, spec)

        # (5)
        # TODO(kisuk): Label & mask transformation.
        # sample = transform(sample)

        # (6)
        # TODO(kisuk): Final crop.


        pass

    def _transform_label(self, sample):
        """
        TODO(kisuk): Documentation.
        """
