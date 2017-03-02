#!/usr/bin/env python
__doc__ = """

Transformer classes.

Kisuk Lee <kisuklee@mit.edu>, 2017
"""

import numpy as np
import time

import transform as tf

class Transformer(object):
    """
    Transformer interface.
    """

    def __call__(self, sample, **kwargs):
        return sample


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
        # DEBUG(kisuk)
        t0 = time.time()
        seg  = sample[self.source]
        msk  = np.ones_like(seg)
        affs = list()
        msks = list()
        # Affinitize.
        t1 = time.time()
        for dst in self.dst:
            affs.append(tf.affinitize(seg, dst=dst))
            msks.append(tf.affinitize_mask(msk, dst=dst))
        t2 = time.time()
        aff = np.concatenate(affs, axis=0)
        msk = np.concatenate(msks, axis=0)
        t3 = time.time()
        # Rebalancing.
        if self.rebalance:
            for c in xrange(aff.shape[0]):
                msk[c,...] *= tf.rebalance_binary_class(aff[c,...], msk=msk[c,...])
        t4 = time.time()
        # Update sample.
        sample[self.target] = aff
        sample[self.target+'_mask'] = msk
        # Crop.
        t5 = time.time()
        if self.crop is not None:
            for k, v in sample.iteritems():
                sample[k] = tf.crop(v, offset=self.crop)
        t6 = time.time()
        # DEBUG(kisuk)
        print "Affinitize: %.3f (%.3f, %.3f, %.3f, %.3f)" % (time.time()-t0,
                t2-t1,t3-t2,t4-t3,t6-t5)
        return sample


class Semantic(Transformer):
    """
    Expand semantic segmentation into multiclass represntation.
    """

    def __init__(self, ids, source, target, rebalance=False):
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


class ObjectInstance(Transformer):
    """
    Object instance segmentation.
    """

    def __init__(self, source, target, rebalance=False):
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
