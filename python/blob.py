#!/usr/bin/env python
__doc__ = """

Blob augmentation.

Kisuk Lee <kisuklee@mit.edu>, 2017
"""

import data_augmentation
import numpy as np

class BlobAugment(data_augmentation.DataAugment):
    """
    TODO(kisuk): Documentation
    """

    def __init__(self, skip_ratio=0.3):
        """Initialize BlobAugment."""
        pass

    def prepare(self, spec, **kwargs):
        """No change in spec."""
        self.spec = spec
        return dict(spec)

    def augment(self, sample, **kwargs):
        """Apply blob data augmentation."""
        pass
