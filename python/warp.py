#!/usr/bin/env python
__doc__ = """

Warp data augmentation.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

from box import Box
import data_augmentation
from warping import warping
import numpy as np
from utils import check_tensor, check_volume
from vector import Vec3d

class WarpAugment(data_augmentation.DataAugment):
    """
    Warp.
        1. Continuous rotation
        2. Shear
        3. Twist
        4. Scale
        5. Perspective stretch
    """

    def __init__(self, skip_ratio=0.3):
        """Initialize WarpAugment."""
        self.ratio = skip_ratio

        # DEBUG
        # self.count = dict(skip=0, warp=0)
        # self.counter = 0

    def prepare(self, spec, **kwargs):
        """
        Randomly draw warp parameters and compute required (mostly
        larger than original) image sizes.
        """
        # Skip.
        self.skip = False
        if self.ratio > np.random.rand():
            self.skip = True
            return dict(spec)

        imgs = kwargs['imgs']

        # Compute the largest image size.
        b = Box((0,0,0), (0,0,0))  # Empty box.
        for k, v in spec.iteritems():
            b = b.merge(Box((0,0,0), v[-3:]))
        maxsz = tuple(b.size())

        # Randomly draw warp parameters.
        # TODO(kisuk): Optional parameter 'amount'?
        params = warping.getWarpParams(maxsz, **kwargs)
        self.size = tuple(x for x in params[0])  # Convert to tuple.
        size_diff = tuple(x - y for x, y in zip(self.size,maxsz))
        self.rot     = params[1]
        self.shear   = params[2]
        self.scale   = params[3]
        self.stretch = params[4]
        self.twist   = params[5]

        # Save original spec.
        self.spec = dict(spec)

        # Replace every shape to the largest required one.
        # TODO(kisuk): Is this correct?
        ret = dict()
        for k, v in spec.iteritems():
            if k in imgs:  # Images.
                ret[k] = v[:-3] + self.size
            else:  # Labels and masks.
                ret[k] = v[:-3] + tuple(x + y for x, y in zip(v[-3:],size_diff))
        return ret

    def augment(self, sample, **kwargs):
        """Apply warp data augmentation."""
        # DEBUG
        # print '\n[WarpAugment]'
        # self.counter += 1
        # if self.skip:
        #     self.count['skip'] += 1
        # else:
        #     self.count['warp'] += 1
        # for k,v in self.count.iteritems():
        #     print '{}={}'.format(k,'%0.3f'%(v/float(self.counter)))

        if self.skip:
            return sample

        # DEBUG
        #print 'rot      = {}'.format(self.rot)
        #print 'shear    = {}'.format(self.shear)
        #print 'scale    = {}'.format(self.scale)
        #print 'stretch  = {}'.format(self.stretch)
        #print 'twist    = {}'.format(self.twist)
        #print 'req_size = {}'.format(self.size)

        imgs = kwargs['imgs']

        # Apply warp to each tensor.
        for k, v in sample.iteritems():
            v = check_tensor(v)
            v = np.transpose(v, (1,0,2,3))
            if k in imgs:  # Images.
                v = warping.warp3d(v, self.spec[k][-3:],
                    self.rot, self.shear, self.scale, self.stretch, self.twist)
            else:  # Labels and masks.
                v = warping.warp3dLab(v, self.spec[k][-3:], self.size,
                    self.rot, self.shear, self.scale, self.stretch, self.twist)
            sample[k] = np.transpose(v, (1,0,2,3))
        return sample


if __name__ == "__main__":

    # Fov.
    spec = dict()
    spec['input/p3'] = (5,109,109)
    spec['input/p2'] = (7,73,73)
    spec['input/p1'] = (9,45,45)
    spec['label']    = (3,1,1,1)

    # In/out size.
    outsz = Vec3d(5,100,100)
    for k, v in spec.iteritems():
        newv = tuple(Vec3d(v[-3:]) + outsz - Vec3d(1,1,1))
        spec[k] = v[:-3] + newv

    # Augmentation.
    aug = WarpAugment()

    # Test.
    ret = aug.prepare(spec, imgs=['input/p3','input/p2','input/p1'])
    print ret
    print aug.spec
    print aug.rot
    print aug.shear
    print aug.scale
    print aug.stretch
    print aug.twist
