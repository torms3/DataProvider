#!/usr/bin/env python
__doc__ = """

Out-of-focus (Gaussian blur) section augmentation.

Kisuk Lee <kisuklee@mit.edu>, 2017
"""

import data_augmentation
import numpy as np

class BlurAugment(data_augmentation.DataAugment):
    """
    Introduce out-of-focus section(s) to a training example. The number of
    out-of-focus sections to introduce is randomly drawn from the uniform
    distribution between [0, MAX_SEC]. Default MAX_SEC is 1, which can be
    overwritten by user-specified value.

    Out-of-focus process is implemented with Gaussian blurring and noise.
    """

    def __init__(self, max_sec=1, skip_ratio=0.3):
        """Initialize BlurAugment."""
        pass

    def prepare(self, spec, **kwargs):
        """No change in spec."""
        self.spec = spec
        return dict(spec)

    def augment(self, sample, **kwargs):
        """Apply out-of-focus section data augmentation."""
        pass
