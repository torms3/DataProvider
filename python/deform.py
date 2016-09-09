#!/usr/bin/env python
__doc__ = """

Linear deformation data augmentation.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

from box import Box
import data_augmentation
from linear_deformation import warping
import numpy as np
from utils import check_tensor

class DeformAugment(data_augmentation.DataAugment):
    """
    Linear deformation.
        1. Continuous rotation
        2. Shear
        3. Twist
        4. Scale
        5. Perspective stretch
    """

    def __init__(self):
        """Initialize DeformAugment."""
        pass

    def prepare(self, spec, **kwargs):
        """
        Randomly draw deformation parameters and compute required (mostly
        larger than original) image sizes.
        """
        # Compute the largest image size.
        b = Box((0,0,0), (0,0,0))  # Empty box.
        for k, v in spec.iteritems():
            b = b.merge(Box((0,0,0), v[-3:]))

        # Randomly draw deformation parameters.
        # TODO(kisuk): Optional parameter 'amount'?
        params = warping.getWarpParams(tuple(b.size()))
        req_size, rot, shear, scale, stretch, twist = params

        # Replace every shape to the largest required one.
        # TODO(kisuk): Is this correct?
        ret = dict()
        for k, v in spec.iteritems():
            ret[k] = v[:-3] + req_size
        return ret

    def augment(self, sample, **kwargs):
        """Apply linear deformation data augmentation."""
        pass


if __name__ == "__main__":

    from vector import Vec3d

    # Fov.
    spec = dict()
    spec['input/p3'] = (5,109,109)
    spec['input/p2'] = (7,73,73)
    spec['input/p1'] = (9,45,45)
    spec['label']    = (1,1,1)

    # In/out size.
    outsz = Vec3d(5,100,100)
    for k, v in spec.iteritems():
        spec[k] = tuple(Vec3d(v) + outsz - Vec3d(1,1,1))

    # Augmentation.
    aug = DeformAugment()

    # Test.
    ret = aug.prepare(spec)
    print ret
