#!/usr/bin/env python
__doc__ = """

Label transform functions.

Kisuk Lee <kisuklee@mit.edu>, 2015-2016
"""

import numpy as np
import transform

class LabelFunction(object):
    """
    TODO(kisuk): Documentation.
    """

    def evaluate(self, sample, key, spec):
        if spec is not None:
            d = dict(spec)
            func = d['type']
            del d['type']
            f = globals()[func]
            f(sample, key, **d)


label_func = LabelFunction()


def binarize(sample, key):
    """Binarize label."""
    # Update sample.
    sample[key] = transform.binarize(sample[key])


def binary_class(sample, key):
    binarize(sample, key)
    multiclass_expansion(sample, key, 2)


def affinitize(sample, key):
    """Transfrom segmentation to 3D affinity graph."""
    affs = transform.affinitize(sample[key])
    msks = transform.affinitize_mask(sample[key+'_mask'])
    # Rebalancing
    wmsk =transform.tensor_func.rebalance_class(affs)
    # Update sample.
    sample[key] = affs
    sample[key+'_mask'] = msks*wmsk
    # Crop by 1.
    for key, data in sample.iteritems():
        sample[key] = transform.tensor_func.crop(data, (1,1,1))


def multiclass_expansion(sample, key, N):
    """For semantic segmentation."""
    lbls = transform.multiclass_expansion(sample[key], N)
    msks = np.tile(sample[key+'_mask'], (N,1,1,1))
    # Rebalancing
    wmsk = transform.rebalance_class(sample[key])
    wmsk = np.tile(wmsk, (N,1,1,1))
    # Update sample.
    sample[key] = lbls
    sample[key+'_mask'] = msks*wmsk