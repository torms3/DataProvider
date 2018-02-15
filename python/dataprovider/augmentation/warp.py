from __future__ import print_function
import numpy as np
import time

import augmentor
from ..box import Box
from ..utils import check_tensor, check_volume
from ..vector import Vec3d
from .warping import warping

class Warp(augmentor.DataAugment):
    """
    Warping data augmentation.

    1. Continuous rotation.
    2. Shear.
    3. Twist.
    4. Scale.
    5. Perspective stretch.
    """

    def __init__(self, skip_ratio=0.3):
        self.set_skip_ratio(skip_ratio)

        # DEBUG
        # self.count = dict(skip=0, warp=0)
        # self.counter = 0

    def prepare(self, spec, **kwargs):
        """Randomly draw warp parameters and compute required (mostly larger
        than original) image sizes."""
        # Skip.
        self.skip = False
        if self.skip_ratio > np.random.rand():
            self.skip = True
            return spec

        imgs = kwargs['imgs']

        # Compute the largest image size.
        b = Box((0,0,0), (0,0,0))  # Empty box.
        for k, v in spec.items():
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
        for k, v in spec.items():
            if k in imgs:  # Images.
                ret[k] = v[:-3] + self.size
            else:  # Labels and masks.
                ret[k] = v[:-3] + tuple(x + y for x, y in zip(v[-3:],size_diff))
        return ret

    def __call__(self, sample, **kwargs):
        """Apply warp data augmentation."""
        # DEBUG(kisuk)
        # t0 = time.time()
        # print('\n[Warp]')
        # self.counter += 1
        # if self.skip:
        #     self.count['skip'] += 1
        # else:
        #     self.count['warp'] += 1
        # for k,v in self.count.items():
        #     print('{}={}'.format(k,'%0.3f'%(v/float(self.counter))))

        if self.skip:
            return sample

        # DEBUG(kisuk)
        # print('rot      = {}'.format(self.rot))
        # print('shear    = {}'.format(self.shear))
        # print('scale    = {}'.format(self.scale))
        # print('stretch  = {}'.format(self.stretch))
        # print('twist    = {}'.format(self.twist))
        # print('req_size = {}'.format(self.size))

        imgs = kwargs['imgs']

        # Apply warp to each tensor.
        for k, v in sample.items():
            v = check_tensor(v)
            v = np.transpose(v, (1,0,2,3))
            if k in imgs:  # Images.
                v = warping.warp3d(v, self.spec[k][-3:],
                    self.rot, self.shear, self.scale, self.stretch, self.twist)
            else:  # Labels and masks.
                v = warping.warp3dLab(v, self.spec[k][-3:], self.size,
                    self.rot, self.shear, self.scale, self.stretch, self.twist)
            # Prevent potential negative stride issues by copying.
            sample[k] = np.copy(np.transpose(v, (1,0,2,3)))
        # DEBUG(kisuk)
        # print("Elapsed: %.3f" % (time.time()-t0))
        return sample

    ####################################################################
    ## Setters.
    ####################################################################

    def set_skip_ratio(self, ratio):
        """Set the probability of skipping augmentation."""
        assert ratio >= 0.0 and ratio <= 1.0
        self.skip_ratio = ratio
