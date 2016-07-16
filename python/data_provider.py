#!/usr/bin/env python
__doc__ = """

DaraProvider classes.

Kisuk Lee <kisuklee@mit.edu>, 2015-2016
"""

import numpy as np
import ConfigParser

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

        # TODO(kisuk): Preprocessing params.

        # TODO(kisuk): Process sampling weight.

        # Construct a ConfigParser
        config = ConfigParser.ConfigParser()
        config.read(data_spec)

        # Build Datasets.
        for dataset_id in drange:
            # Build section name.
            section = 'dataset%d' % dataset_id
            # Add FoV.
            self._add_fov(config, section, net_spec)
            # Add border mirroring.
            if params['border_mode'] is 'mirror':
                self._add_mirror_border(config, section, net_spec)
                # TODO(kisuk): Check config if 'border_mirror' was added.
            # Construct a Dataset
            dataset = VolumeDataset(config, section, net_spec)
            self._datasets.append(dataset)

        # TODO(kisuk): Setup data augmentation.

    def _add_mirror_border(self, config, section, net_spec):
        """Add preprocessing 'mirror_border' to each images."""
        # For each layer's name and dimension:
        for name, dim in net_spec.iteritems():
            key = config.get(section, name)
            if 'image' in key:  # Apply border mirroring only to images.
                # Border mirroring is appended to the preprocessing list.
                pp = config.get(key, 'preprocess')
                pp += '\n'
                pp += "{'type':'mirror_border','fov':%s}" % str(dim[-3:])
                config.set(key, 'preprocess', pp)

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
