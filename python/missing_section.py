#!/usr/bin/env python
__doc__ = """

Missing section data augmentation.

Kisuk Lee <kisuklee@mit.edu>, 2017
"""

import data_augmentation
import numpy as np

class MissingAugment(data_augmentation.DataAugment):
    """
    Introduce missing section(s) to a training example. The number of missing
    sections to introduce is randomly drawn from the uniform distribution
    between [0, MAX_SEC]. Default MAX_SEC is 1, which can be overwritten by
    user-specified value.

    TODO(kisuk):
        1. Specific control of the location to introduce missing sections
        2. Weight mask modification (put more weight on the missing section(s))
    """

    def __init__(self, max_sec=1, skip_ratio=0.3):
        """Initialize MissingSectionAugment."""
        self.set_max_sections(max_sec)
        self.set_skip_ratio(skip_ratio)

    def set_max_sections(self, max_sec):
        """Set the maximum number of missing sections to introduce."""
        self.MAX_SEC = max_sec

    def set_skip_ratio(self, ratio):
        """Set the probability of skipping augmentation."""
        self.ratio = ratio

    def prepare(self, spec, **kwargs):
        """No change in spec."""
        self.spec = spec
        return dict(spec)

    def augment(self, sample, **kwargs):
        """Apply missing section data augmentation."""
        ret = sample

        if np.random.rand() > self.ratio:
            # Randomly draw the number of sections to introduce.
            num_sec = np.random.randint(0, self.MAX_SEC + 1)
            print "num_sec = %d" % num_sec

            # Assume that the sample contains only one input volume,
            # or multiple input volumes of same size.
            imgs  = kwargs['imgs']
            zdims = set([])
            for key in imgs:
                zdim = self.spec[key][-3]
                assert num_sec < zdim
                zdims.add(zdim)
            assert len(zdims) == 1
            zdim = zdims.pop()

            # Randomly draw z-slices to black out.
            locs = np.random.choice(zdim, num_sec, replace=False)

            # Discard the selected sections.
            for key in imgs:
                ret[key][...,locs,:,:] *= 0

        return ret
