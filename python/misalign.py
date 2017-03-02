#!/usr/bin/env python
__doc__ = """
Misalignment data augmentation.
Karan Kathpalia <karank@cs.princeton.edu>
Kisuk Lee <kisuklee@mit.edu>, 2016-2017
"""

import data_augmentation
import numpy as np
from utils import check_tensor

class MisalignAugment(data_augmentation.DataAugment):
    """
    Misalignment data augmentation.
    """

    def __init__(self, max_trans=15.0, slip_ratio=0.3, skip_ratio=0.0):
        self.set_max_translation(max_trans)
        self.set_slip_ratio(slip_ratio)
        self.set_skip_ratio(skip_ratio)

    def prepare(self, spec, **kwargs):
        self.spec = dict(spec)
        self.skip = np.random.rand() < self.skip_ratio

        if self.skip:
            ret = spec
        else:
            # Max translation.
            if 'max_trans' in kwargs:
                max_trans = kwargs['max_trans']
            else:
                max_trans = self.max_trans

            # Random translation.
            # Always lower box is translated.
            self.x_t = int(round(max_trans * np.random.rand(1)))
            self.y_t = int(round(max_trans * np.random.rand(1)))

            # Randomly draw x/y translation independently.
            ret, pvt, zs = dict(), dict(), list()
            for k, v in self.spec.iteritems():
                z, y, x = v[-3:]
                assert z > 0
                x_dim  = self.x_t + v[-1]
                y_dim  = self.y_t + v[-2]
                ret[k] = v[:-2] + (y_dim, x_dim)
                pvt[k] = z
                zs.append(z)

            # Random direction of translation.
            x_sign = np.random.choice(['+','-'])
            y_sign = np.random.choice(['+','-'])
            self.x_t = int(eval(x_sign + str(self.x_t)))
            self.y_t = int(eval(y_sign + str(self.y_t)))

            zmin = min(zs)

            # Trivial 2D case
            if zmin == 1:
                self.do_augment = False
                ret = dict(spec)
            else:
                self.do_augment = True
                # Introduce misalignment at pivot.
                pivot = np.random.randint(1, zmin - 1)
                for k, v in pvt.iteritems():
                    offset = int(v - zmin)/2  # Compute offset.
                    pvt[k] = offset + pivot
                self.pivot = pvt

            # Slip?
            self.slip = np.random.rand() < self.slip_ratio

        return ret

    def __call__(self, sample, **kwargs):
        if not self.skip:
            sample = self.augment(sample, **kwargs)
        return sample

    def augment(self, sample, **kwargs):
        # DEBUG(kisuk)
        # print "\n[Misalign]"
        # for k, v in sample.iteritems():
        #     print "Slip = {}".format(self.slip)
        #     print "{} at {}".format(k, self.pivot[k])

        ret = dict()

        if self.do_augment:
            for k, v in sample.iteritems():
                # Ensure data is a 4D tensor.
                data = check_tensor(v)
                new_data = np.zeros(self.spec[k], dtype=data.dtype)
                new_data = check_tensor(new_data)
                # Dimension.
                z, y, x = v.shape[-3:]
                assert z > 1
                if self.slip:
                    # Copy whole box.
                    xmin = max(self.x_t, 0)
                    ymin = max(self.y_t, 0)
                    xmax = min(self.x_t, 0) + x
                    ymax = min(self.y_t, 0) + y
                    new_data[:,:,...] = data[:,:,ymin:ymax,xmin:xmax]
                    # Slip.
                    xmin = max(-self.x_t, 0)
                    ymin = max(-self.y_t, 0)
                    xmax = min(-self.x_t, 0) + x
                    ymax = min(-self.y_t, 0) + y
                    pvot = self.pivot[k]
                    new_data[:,pvot,...] = data[:,pvot,ymin:ymax,xmin:xmax]
                else:
                    # Copy upper box.
                    xmin = max(self.x_t, 0)
                    ymin = max(self.y_t, 0)
                    xmax = min(self.x_t, 0) + x
                    ymax = min(self.y_t, 0) + y
                    pvot = self.pivot[k]
                    new_data[:,0:pvot,...] = data[:,0:pvot,ymin:ymax,xmin:xmax]
                    # Copy lower box.
                    xmin = max(-self.x_t, 0)
                    ymin = max(-self.y_t, 0)
                    xmax = min(-self.x_t, 0) + x
                    ymax = min(-self.y_t, 0) + y
                    pvot = self.pivot[k]
                    new_data[:,pvot:,...] = data[:,pvot:,ymin:ymax,xmin:xmax]
                # Augmented sample.
                ret[k] = new_data
        else:
            ret = sample

        return ret

    ####################################################################
    ## Setters.
    ####################################################################

    def set_max_translation(self, max_trans):
        """Set the maximum amount of translation in x and y."""
        assert max_trans > 0
        self.max_trans = max_trans

    def set_slip_ratio(self, ratio):
        """How often is slip miaslignment introduced?"""
        assert ratio >= 0.0 and ratio <= 1.0
        self.slip_ratio = ratio

    def set_skip_ratio(self, ratio):
        """Set the probability of skipping augmentation."""
        assert ratio >= 0.0 and ratio <= 1.0
        self.skip_ratio = ratio
