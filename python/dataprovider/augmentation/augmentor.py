#!/usr/bin/env python
__doc__ = """

Augmentor classes.

Kisuk Lee <kisuklee@mit.edu>, 2016-2017
"""

from collections import OrderedDict
import numpy as np

class DataAugment(object):
    """
    DataAugment interface.
    """

    def prepare(self, spec, **kwargs):
        """Prepare data augmentation.

        Some data augmentation require larger sample size than the original,
        depending on the augmentation parameters that are randomly chosen.
        For such cases, here we determine random augmentation parameters and
        return an updated sample spec accordingly.
        """
        raise NotImplementedError

    def __call__(self, sample, **kwargs):
        """Apply data augmentation."""
        raise NotImplementedError

    def factory(aug_type, **kwargs):
        """Factory method for data augmentation classes."""
        if aug_type is 'box':       return BoxOcclusion(**kwargs)
        if aug_type is 'blur':      return Blur(**kwargs)
        if aug_type is 'flip':      return Flip(**kwargs)
        if aug_type is 'warp':      return Warp(**kwargs)
        if aug_type is 'misalign':  return Misalign(**kwargs)
        if aug_type is 'missing':   return MissingSection(**kwargs)
        if aug_type is 'greyscale': return Greyscale(**kwargs)
        assert False, "Unknown data augmentation type: [%s]" % aug_type

    factory = staticmethod(factory)


class Augmentor(object):
    """
    Data augmentor.
    """

    def __init__(self):
        self._augments = list()

    def append(self, aug, **kwargs):
        """Append data augmentation.

        Augmentation type and parameters should be specified.

        Args:
            aug: (1) DataAugment object.
                 (2) Dictionary of augmentation type and parameters.
                     e.g. dict(type='flip')
                 (3) str indicating augmentation type
                     e.g. 'flip'
            kwargs: Only used in case (3).
        """
        if isinstance(aug, DataAugment):
            pass
        elif type(aug) is dict:
            assert 'type' in aug
            aug_type = aug['type']
            del aug['type']
            aug = DataAugment.factory(aug_type, **aug)
        elif type(aug) is str:
            aug = DataAugment.factory(aug, **kwargs)
        else:
            assert False, "Bad data augmentation " + aug
        self._augments.append(aug)

    def prepare(self, spec, **kwargs):
        """Prepare random parameters and modify sample spec accordingly."""
        for aug in reversed(self._augments):
            spec = aug.prepare(spec, **kwargs)
        return dict(spec)

    def __call__(self, sample, **kwargs):
        # Apply a list of data augmentation.
        for aug in self._augments:
            sample = aug(sample, **kwargs)
        # Ensure that sample is ordered by key.
        sample = OrderedDict(sorted(sample.items(), key=lambda x: x[0]))
        return sample

from box import BoxOcclusion
from blur import Blur
from flip import Flip
from warp import Warp
from misalign import Misalign
from missing_section import MissingSection
from greyscale import Greyscale
