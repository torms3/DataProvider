#!/usr/bin/env python
__doc__ = """

Utility functions.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import numpy as np

def check_volume(data):
    """Ensure that data is numpy 3D array."""
    assert isinstance(data, np.ndarray)

    if data.ndim == 2:
        data = data[np.newaxis,...]
    elif data.ndim == 3:
        pass
    elif data.ndim == 4:
        assert data.shape[0]==1
        data = np.reshape(data, data.shape[-3:])
    else:
        raise RuntimeError('data must be a numpy 3D array')

    assert data.ndim==3
    return data


def check_tensor(data):
    """Ensure that data is numpy 4D array."""
    assert isinstance(data, np.ndarray)

    if data.ndim == 2:
        data = data[np.newaxis,np.newaxis,...]
    elif data.ndim == 3:
        data = data[np.newaxis,...]
    elif data.ndim == 4:
        pass
    else:
        raise RuntimeError('data must be a numpy 4D array')

    assert data.ndim==4
    return data


def fill_data(shape, filler={'type':'zero'}, dtype='float32'):
    """
    Return numpy array of shape, filled with specified values.

    Args:
        shape: Array shape.
        filler: {'type':'zero'} (default)
                {'type':'one'}
                {'type':'constant', 'value':%f}
                {'type':'gaussian', 'loc':%f, 'scale':%f}
                {'type':'uniform', 'low':%f, 'high':%f}

    Returs:
        data: Numpy array of shape, filled with specified values.
    """
    data = np.zeros(shape, dtype=dtype)

    assert 'type' in filler
    if filler['type'] is 'zero':
        # Fill zeros.
        pass
    elif filler['type'] is 'one':
        # Fill ones.
        data = np.ones(shape, dtype=dtype)
    elif filler['type'] is 'constant':
        # Fill constant value.
        assert 'value' in filler
        data[:] = filler['value']
    elif filler['type'] is 'gaussian':
        # Fill random numbers from Gaussian(loc, scale).
        loc = filler['mean'] if 'mean' in filler else 0.0
        scale = filler['std'] if 'std' in filler else 1.0
        data[:] = np.random.normal(loc=loc, scale=scale, size=shape)
    elif filler['type'] is 'uniform':
        # Fill random numbers from Uniform(low, high).
        low = filler['low'] if 'low' in filler else 0.0
        high = filler['high'] if 'high' in filler else 1.0
        data[:] = np.random.uniform(low=low, high=high, size=shape)
    else:
        raise RuntimeError('invalid filler type [%s]' % filler['type'])

    return data
