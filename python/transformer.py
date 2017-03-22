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
        msk  = get_mask(sample, self.source)
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


class Affinity1(Transformer):
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
        msk  = get_mask(sample, self.source)
        affs = list()
        msks = list()
        # Affinitize.
        for dst in self.dst:
            affs.append(tf.affinitize1(seg, dst=dst))
            msks.append(tf.affinitize1_mask(msk, dst=dst))
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

    def __init__(self, ids, source, target, rebalance=True):
        self.ids = ids
        self.source = source
        self.target = target
        self.rebalance = rebalance

    def __call__(self, sample):
        """Semantic label processing."""
        sem = sample[self.source]
        # Semantic class expansion.
        lbl, msk = tf.multiclass_expansion(sem, ids=self.ids)
        # Combine with a given mask.
        msk *= get_mask(sample, self.source)
        # Rebalancing.
        if self.rebalance:
            for i, _ in enumerate(self.ids):
                msk[i,...] = tf.rebalance_binary_class(lbl[i,...],msk[i,...])
        # Update sample.
        sample[self.target] = lbl
        sample[self.target+'_mask'] = msk
        return sample


class Synapse(Transformer):
    """
    Transform synapse segmentation into binary representation.
    """

    def __init__(self, source, target, rebalance=True, base_w=0.0):
        self.source = source
        self.target = target
        self.rebalance = rebalance
        self.base_w = base_w

    def __call__(self, sample):
        """Synapse label processing."""
        syn = sample[self.source]
        # Binarize.
        lbl = tf.binarize(syn)
        msk = get_mask(sample, self.source)
        # Rebalancing.
        if self.rebalance:
            msk = tf.rebalance_binary_class(lbl, msk, base_w=self.base_w)
        # Update sample.
        sample[self.target] = lbl
        sample[self.target+'_mask'] = msk
        return sample

####################################################################
## Helper.
####################################################################

def get_mask(sample, key):
    """Return mask if any, or else default one (all ones)."""
    msk = None
    if key in sample:
        if key+'_mask' in sample:
            msk = sample[key+'_mask'].astype('float32')
        else:
            msk = np.ones(sample[key].shape, 'float32')
    return msk
