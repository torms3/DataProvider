#!/usr/bin/env python
__doc__ = """

Create border on segmentation.

Modified the code from
https://github.com/cremi/cremi_python

Kisuk Lee <kisuklee@mit.edu>, 2017
"""

import numpy as np
import scipy
from utils import *

def create_border(seg, max_dist, axis=0):
    """
    Overlay a border mask with background_label onto input data.
    A pixel is part of a border if one of its 4-neighbors has different label.

    Parameters
    ----------
    seg : numpy.ndarray - Segmentation.
    max_dist : int or float - Maximum distance from border for pixels to be included into the mask.
    axis : int - Axis of iteration (perpendicular to 2d images for which mask will be generated)
    """
    seg = check_tensor(seg)
    ret = np.copy(seg)
    sl  = [slice(None) for d in range(seg.ndim)]

    for z in range(seg.shape[axis]):
        sl[axis] = z
        border = create_border_mask_2d(seg[tuple(sl)], max_dist)
        target = ret[tuple(sl)]
        target[np.logical_not(border)] = 0
        ret[tuple(sl)] = target


def create_border_mask_2d(image, max_dist):
    """
    Create binary border mask for image.
    A pixel is part of a border if one of its 4-neighbors has different label.

    Parameters
    ----------
    image : numpy.ndarray - Image containing integer labels.
    max_dist : int or float - Maximum distance from border for pixels to be included into the mask.

    Returns
    -------
    mask : numpy.ndarray - Binary mask of border pixels. Same shape as image.
    """
    max_dist = max(max_dist, 0)

    padded = np.pad(image, 1, mode='edge')

    border_pixels = np.logical_and(
        np.logical_and( image == padded[:-2, 1:-1], image == padded[2:, 1:-1] ),
        np.logical_and( image == padded[1:-1, :-2], image == padded[1:-1, 2:] )
        )

    distances = scipy.ndimage.distance_transform_edt(
        border_pixels,
        return_distances=True,
        return_indices=False
        )

    return distances <= max_dist
