#!/usr/bin/env python
__doc__ = """

Misalignment data augmentation.

Karan Kathpalia <karank@cs.princeton.edu>
Kisuk Lee <kisuklee@mit.edu>, 2016
"""

from utils import *

class MisalignAugment(DataAugment):
    """
    Misalignment.
    """

    def prepare(self, spec):
        """
        TODO(kisuk): Documentation.
        """
        self.spec = spec

        # TODO(kisuk): Temporary magic number. Does this need be a parameter?
        self.MAX_TRANS = 20.0

        # Random translation (positive integer in [0,MAX_TRANS])
        self.x_t = int(round(self.MAX_TRANS*np.random.rand(1)))
        self.y_t = int(round(self.MAX_TRANS*np.random.rand(1)))

        ret = dict()

        # Randomly draw x/y translation independently.
        for k, v in self.spec.iteritems():
            z, y, x = v[-3:]
            assert z>0
            # Trivial 2D case
            if z == 1:
                ret[k] = v[:-3] + (z, y, x)
            else:
                x_dim  = self.x_t + v[-1]
                y_dim  = self.y_t + v[-2]
                ret[k] = v[:-2] + (y_dim, x_dim)

        return ret

    def augment(self, sample):
        """Apply misalignment data augmentation and then crop to self.spec."""

        ret = dict()

        for k, v in sample.iteritems():

            # Ensure data is 4D tensor.
            data = check_tensor(v)
            new_data = np.zeros(self.spec[k], dtype=data.dtype)
            new_data = check_tensor(new_data)

            # Dimension
            z, y, x = v.shape[-3:]
            assert z >= 1

            # Trivial case
            if z == 1:
                assert new_data.shape==data.shape
                new_data[:] = data
            else:
                # Introduce misalignment at pivot.
                pivot = np.random.randint(z - 1)

                # Random direction of translation.
                # Always bottom box is translated.
                x_t = int(np.random.rand(1) > 0.5) * self.x_t
                y_t = int(np.random.rand(1) > 0.5) * self.y_t

                # Copy upper box.
                xmin = max(x_t, 0)
                ymin = max(y_t, 0)
                xmax = min(x_t, 0) + x
                ymax = min(x_t, 0) + y
                new_data[:,0:pivot,...] = data[:,0:pivot,ymin:ymax,xmin:xmax]

                # Copy lower box.
                xmin = max(-x_t, 0)
                ymin = max(-y_t, 0)
                xmax = min(-x_t, 0) + x
                ymax = min(-y_t, 0) + y
                new_data[:,pivot:,...] = data[:,0:pivot,ymin:ymax,xmin:xmax]

            ret[k] = new_data

        return ret