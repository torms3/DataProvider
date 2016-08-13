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


def binarize(sample, key, rebalancing=True):
    """Binarize label."""
    # Update sample.
    sample[key] = transform.binarize(sample[key])
    # Rebalancing.
    if rebalancing:
        wmsk = transform.rebalance_class(sample[key])
        sample[key+'_mask'] *= wmsk


def binary_class(sample, key, rebalancing=True):
    binarize(sample, key, False)
    multiclass_expansion(sample, key, 2, rebalancing)


def affinitize(sample, key, rebalancing=True):
    """Transfrom segmentation to 3D affinity graph."""
    affs = transform.affinitize(sample[key])
    msks = transform.affinitize_mask(sample[key+'_mask'])
    # Update sample.
    sample[key] = affs
    sample[key+'_mask'] = msks
    # Rebalancing.
    if rebalancing:
        wmsk = transform.tensor_func.rebalance_class(affs)
        sample[key+'_mask'] *= wmsk


def multiclass_expansion(sample, key, N, rebalancing=True):
    """For semantic segmentation."""
    lbls = transform.multiclass_expansion(sample[key], N)
    msks = np.tile(sample[key+'_mask'], (N,1,1,1))
    # Update sample.
    sample[key] = lbls
    sample[key+'_mask'] = msks
    # Rebalancing.
    if rebalancing:
        wmsk = transform.rebalance_class(sample[key])
        wmsk = np.tile(wmsk, (N,1,1,1))
        sample[key+'_mask'] *= wmsk
