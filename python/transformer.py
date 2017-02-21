#!/usr/bin/env python
__doc__ = """

Transformer classes.

Kisuk Lee <kisuklee@mit.edu>, 2017
"""

import transform as tf

class Transformer(object):
    """
    Transformer interface.
    """
    def __call__(self, sample):
        raise NotImplementedError


class Semantic(Transformer):
    """
    Expand semantic segmentation into multiclass represntation.
    """

    def __init__(self, ids, key, rebalance=False):
        self.ids = ids
        self.key = key
        self.rebalance = rebalance

    def __call__(self, sample):
        """Semantic label processing."""
        sem = sample[self.key]
        # Semantic class expansion.
        lbl, msk = tf.multiclass_expansion(sem, ids=self.ids)
        # Rebalancing.
        if self.rebalance:
            for i, _ in enumerate(self.ids):
                msk[i,...] = tf.rebalance_binary_class(lbl[i,...],msk[i,...])
        # Replace sample.
        sample[self.key] = lbl
        sample[self.key+'_mask'] = msk
        return sample
