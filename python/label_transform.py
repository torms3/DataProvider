#!/usr/bin/env python
__doc__ = """

Label transform functions.

Kisuk Lee <kisuklee@mit.edu>, 2015-2016
"""

import numpy as np
import transform
import utils

class LabelFunction(object):
    """
    Transform label.
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
    multiclass_expansion(sample, key, [0,1], rebalancing)


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


def multiclass_expansion(sample, key, ids, rebalancing=True):
    """For semantic segmentation."""
    lbl  = sample[key]
    msk  = utils.check_volume(sample[key+'_mask'])
    lbls, msk2 = transform.multiclass_expansion(lbl, ids)
    msk *= msk2
    msks = np.tile(msk, (len(ids),1,1,1))
    # Update sample.
    sample[key] = lbls
    sample[key+'_mask'] = msks
    # Rebalancing.
    if rebalancing:
        wmsk = transform.rebalance_class(lbl)
        wmsk = np.tile(wmsk, (len(ids),1,1,1))
        sample[key+'_mask'] *= wmsk
