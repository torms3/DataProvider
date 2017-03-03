#!/usr/bin/env python
__doc__ = """

Transformer classes.

Kisuk Lee <kisuklee@mit.edu>, 2017
"""

import numpy as np

import transform as tf

class Transformer(object):
    """
    Transformer interface.
    """
    def __call__(self, sample):
        raise NotImplementedError


class Affinity(Transformer):
    """
    Expand segmentation into affinity represntation.
    """

    def __init__(self, dst, source, target, crop=None, rebalance=False):
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


class Semantic(Transformer):
    """
    Expand semantic segmentation into multiclass represntation.
    """

    def __init__(self, ids, source, target, rebalance=False):
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
                msk[i,...] = tf.rebalance_binary_class(lbl[i,...],msk[i,...])
        # Replace sample.
        sample[self.target] = lbl
        sample[self.target+'_mask'] = msk
        return sample
