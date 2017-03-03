#!/usr/bin/env python
__doc__ = """

Transform classes.

Kisuk Lee <kisuklee@mit.edu>, 2017
"""

import numpy as np
import time

import transform as tf

class Transformer(object):
    """
    A sequence of transfors.
    """

    def __init__(self):
        self._transforms = list()

    def __call__(self, sample, **kwargs):
        for tf in self._transforms:
            sample = tf(sample, **kwargs)
        return sample

    def append(self, tf):
        assert isinstance(tf, Transform)
        self._transforms.append(tf)


class Transform(object):
    """
    Transform interface.
    """

    def __call__(self, sample, **kwargs):
        return sample


class Affinity(Transform):
    """
    Expand segmentation into affinity represntation.
    """

    def __init__(self, dst, source, target, crop=None, rebalance=True):
        """Initialize parameters.

        Args:
            dst: List of 3-tuples, each indicating affinity distance in (z,y,x).
            source: Key to source data from which to construct affinity.
            target: Key to target data.
            crop: 3-tuple indicating crop offset.
            rebalance: Class-rebalanced gradient weight mask.
        """
        self.dst = dst
        self.source = source
        self.target = target
        self.crop = crop
        self.rebalance = rebalance

    def __call__(self, sample):
        """Affinity label processing."""
        seg  = sample[self.source]
        msk  = np.ones_like(seg)
        affs = list()
        msks = list()
        # Affinitize.
        for dst in self.dst:
            affs.append(tf.affinitize(seg, dst=dst))
            msks.append(tf.affinitize_mask(msk, dst=dst))
        aff = np.concatenate(affs, axis=0)
        msk = np.concatenate(msks, axis=0)
        # Rebalancing.
        if self.rebalance:
            for c in xrange(aff.shape[0]):
                msk[c,...] *= tf.rebalance_binary_class(aff[c,...], msk=msk[c,...])
        # Update sample.
        sample[self.target] = aff
        sample[self.target+'_mask'] = msk
        # Crop.
        if self.crop is not None:
            for k, v in sample.iteritems():
                sample[k] = tf.crop(v, offset=self.crop)
        return sample


class Semantic(Transform):
    """
    Expand semantic segmentation into multiclass represntation.
    """

    def __init__(self, ids, source, target, rebalance=True):
        """Initialize parameters.

        Args:
            ids: List of ids to expand.
            source: Key to source data from which to construct target.
            target: Key to target data.
            rebalance: Class-rebalanced gradient weight mask.
        """
        self.ids = ids
        self.source = source
        self.target = target
        self.rebalance = rebalance

    def __call__(self, sample):
        """Semantic label processing."""
        sem = sample[self.source]
        # Semantic class expansion.
        lbl, msk = tf.multiclass_expansion(sem, ids=self.ids)
        # Rebalancing.
        if self.rebalance:
            for i, _ in enumerate(self.ids):
                msk[i,...] = tf.rebalance_binary_class(lbl[i,...], msk[i,...])
        # Replace sample.
        sample[self.target] = lbl
        sample[self.target+'_mask'] = msk
        return sample


class ObjectInstance(Transform):
    """
    Object instance segmentation.
    """

    def __init__(self, source, target, rebalance=True):
        self.source = source
        self.target = target
        self.rebalance = rebalance

    def __call__(self, sample):
        seg = sample[self.source]
        # Binarize.
        lbl = tf.binarize_center_object(seg)
        # Rebalancing.
        if self.rebalance:
            msk = tf.rebalance_binary_class(lbl, msk=np.ones_like(lbl))
        # Replace sample.
        sample[self.target] = lbl
        sample[self.target+'_mask'] = msk
        return sample
