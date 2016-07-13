#!/usr/bin/env python
__doc__ = """

Functions for data transformation.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import numpy as np
from vector import Vec3d, minimum, maximum

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

def transform_sample(sample, func, *args, **kwargs):
    """Apply func to a sample."""
    ret = {}

    for key, val in sample.iteritems():
        ret[key] = transform_tensor(val, func, *args, **kwargs)

    return ret

def transform_tensor(data, func, *args, **kwargs):
    """Apply func to each channel of data (4D tensor)."""
    data = check_tensor(data)
    arrs = list()

    f = globals()[func]

    for c in xrange(data.shape[0]):
        vol = f(data[c,...], *args, **kwargs)
        arrs.append(vol[np.newaxis,...])

    return np.concatenate(arrs, axis=0)


def crop(img, size, offset=(0,0,0)):
    """
    TODO(kisuk):
        Do we need crop? Yes we do.
        E.g. transform_sample(sample, 'crop', size, offset)
    """
    img = check_volume(img)
    v1  = Vec3d(offset)
    v2  = v1 + Vec3d(size)
    ret = np.zeros(size, dtype=img.dtype)
    ret[:] = img[v1[0]:v2[0],v1[1]:v2[1],v1[2]:v2[2]]
    return ret

####################################################################
## Preprocessing
####################################################################

def mirror_border(img, fov):
    """
    TODO(kisuk) Documentation.
    """
    img = check_volume(img)

    # Validate FoV
    fov = np.asarray(fov)
    fov = Vec3d(fov.astype('uint32'))

    # Pad size
    top = fov / 2
    btm = fov - top - (1,1,1)
    pad_with = [(top[0],btm[0]),(top[1],btm[1]),(top[2],btm[2])]
    # TODO(kisuk): Should we force an odd-sized fov?

    # TODO(kisuk): 'symmetric' or 'reflect'?
    return np.pad(img, pad_with, mode='reflect')


def standardize(img, mode='2D', dtype='float32'):
    """Standard normalization (zero mean, unit standard deviation)."""
    img = check_volume(img)
    ret = np.zeros(img.shape, dtype=dtype)

    # Standardize function (zero mean & unit standard deviation).
    f = lambda x: (x - np.mean(x)) / np.std(x)

    if mode == '2D':
        for z in xrange(img.shape[0]):
            ret[z,:,:] = f(img[z,:,:])
    elif mode == '3D':
        ret[:] = f(img)
    else:
        raise RuntimeError("mode must be either '2D' or '3D'")

    return ret

####################################################################
## Data Augmentations
####################################################################

def flip(data, rule):
    """Flip data according to a specified rule.

    Args:
        data: 3D numpy array to be transformed.
        rule: Transform rule, specified as a Boolean array.
             [z reflection,
              y reflection,
              x reflection,
              xy transpose]

    Returns:
        data: Transformed data.
    """
    data = check_tensor(data)

    assert np.size(rule)==4

    # z reflection
    if rule[0]:
        data = data[:,::-1,:,:]
    # y reflection
    if rule[1]:
        data = data[:,:,::-1,:]
    # x reflection
    if rule[2]:
        data = data[:,:,:,::-1]
    # Transpose in xy
    if rule[3]:
        data = data.transpose(0,1,3,2)

    return data

####################################################################
## Label Transformations
####################################################################

def binarize(img, dtype='float32'):
    """Binarize image.

    Normally used to turn a ground truth segmentation into a ground truth
    boundary map, binary representation for each voxel being neuronal boundary
    or not.

    Args:
        img: 3D indexed image, with each index corresponding to each segment.

    Returns:
        ret: Binarized image.
    """
    img = check_volume(img)
    ret = np.zeros(img.shape, dtype=dtype)
    ret[:] = (img>0).astype(dtype)
    return ret


def multiclass_expansion(lbl, N=None, dtype='float32'):
    """
    TODO(kisuk): Semantic segmentation.
    """
    lbl = check_volume(lbl)
    unique_lbls = np.unique(lbl)

    # If N is not given, infer from the maximum ID in lbl.
    if N == None:
        N = np.max(unique_lbls) + 1

    ret = np.zeros((N,) + lbl.shape, dtype=dtype)

    for l in unique_lbls:
        ret[l,...] = (img == l)

    return ret


def binary_class(img, dtype='float32'):
    """
    TODO(kisuk): Documentation.
    """
    img = check_volume(img)
    img = binarize(img, dtype=dtype)
    return multiclass_expansion(img, N=2, dtype=dtype)


def affinitize(img, dtype='float32'):
    """Transform segmentation to 3D affinity graph.

    Args:
        img: 3D indexed image, with each index corresponding to each segment.

    Returns:
        ret: 3D affinity graph (4D tensor), 3 channels for z, y, x direction.
    """
    img = check_volume(img)
    ret = np.zeros((3,) + img.shape, dtype=dtype)

    ret[2,1:,:,:] = (img[1:,:,:]==img[:-1,:,:]) & (img[1:,:,:]>0)  # z affinity
    ret[1::,1:,:] = (img[:,1:,:]==img[:,:-1,:]) & (img[:,1:,:]>0)  # y affinity
    ret[0::,:,1:] = (img[:,:,1:]==img[:,:,:-1]) & (img[:,:,1:]>0)  # x affinity

    return ret

####################################################################
## Mask Transformations
####################################################################

def affinity_mask(msk, dtype='float32'):
    """Transform binary mask to affinity mask.

    Args:
        msk: 3D binary mask.

    Returns:
        ret: 3D affinity mask (4D tensor), 3 channels for z, y, x direction.
    """
    img = check_volume(img)
    ret = np.zeros((3,) + msk.shape, dtype=dtype)

    ret[2,1:,:,:] = (msk[1:,:,:]>0) | (msk[:-1,:,:]>0)
    ret[1,:,1:,:] = (msk[:,1:,:]>0) | (msk[:,:-1,:]>0)
    ret[0,:,:,1:] = (msk[:,:,1:]>0) | (msk[:,:,:-1]>0)

    return ret

####################################################################
## Rebalancing
####################################################################

# TODO(kisuk): Implement gradient rebalancing.

########################################################################
## Unit Testing
########################################################################
if __name__ == "__main__":

    import unittest

    ####################################################################
    class UnitTestTransform(unittest.TestCase):

        def setup(self):
            pass

        def testCrop(self):
            img = np.random.rand(4,4,4)
            a = crop(img, (3,3,3))
            b = img[:-1,:-1,:-1]
            self.assertTrue(np.array_equal(a,b))
            a = crop(img, (3,3,3), (1,1,1))
            b = img[1:,1:,1:]
            self.assertTrue(np.array_equal(a,b))

    ####################################################################
    unittest.main()

    ####################################################################