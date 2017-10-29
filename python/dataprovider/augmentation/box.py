#!/usr/bin/env python
__doc__ = """

Box occlusion augmentation.

Kisuk Lee <kisuklee@mit.edu>, 2017
"""

import math
import numpy as np
from scipy.ndimage.filters import gaussian_filter

import augmentor
from ..box import *

class BoxOcclusion(augmentor.DataAugment):
    """
    Add random box occlusion masks.
    """

    def __init__(self, min_dim, max_dim, aspect_ratio, max_density, mode=[2,2,0.3,0.3,1], skip_ratio=0.0, sigma_max=5.0):
        """Initialize BoxAugment."""
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.aspect_ratio = aspect_ratio
        self.max_density = max_density
        self.mode = np.asarray(mode, dtype='float32')
        self.set_skip_ratio(skip_ratio)
        self.sigma_max = sigma_max

        # TODO(kisuk): Allow nonzero alpha?
        self.alpha = 0.0

    def prepare(self, spec, **kwargs):
        # No change in spec.
        self.spec = spec
        return dict(spec)

    def __call__(self, sample, **kwargs):
        """Apply box data augmentation."""
        if np.random.rand() > self.skip_ratio:
            sample = self.augment(sample, **kwargs)
        return sample

    def augment(self, sample, **kwargs):
        imgs = kwargs['imgs']

        # Find union of bounding boxes.
        self.bbox = dict()
        bbox = None
        for key in imgs:
            dim = self.spec[key][-3:]
            b = centered_box((0,0,0), dim)
            bbox = b if bbox is None else bbox.merge(b)
            self.bbox[key] = b

        # Create a mask.
        self.offset = bbox.min()
        self.dim    = bbox.size()

        for key in imgs:
            # TODO(kisuk): Augmentation mask.
            # # Extract augmentation mask.
            # if key+'_augmask' in sample:
            #     msk = sample[key+'_mask'].astype('float32')
            # else:
            #     msk = np.zeros(sample[key].shape, 'float32')

            # Random box augmentation.
            count   = 0
            density = self.max_density*np.random.rand()
            goal    = bbox.volume()*density
            # DEBUG(kisuk):
            # print 'density: %.2f' % density
            while True:
                # Random location.
                m = self.min_dim  # Margin.
                z = np.random.randint(0, self.dim[0])
                y = np.random.randint(0, self.dim[1])
                x = np.random.randint(0, self.dim[2])
                loc = Vec3d(z,y,x) + self.offset
                # Random box size.
                dim = np.random.randint(self.min_dim, self.max_dim + 1, 3)
                # Anisotropy.
                dim[0] /= int(self.aspect_ratio)
                # Box.
                box = bbox.intersect(centered_box(loc, dim))
                # Local coordiate.
                box.translate(-self.offset)
                vmin = box.min()
                vmax = box.max()
                sz   = box.size()

                # Random choice.
                enabled = self.mode > 0
                rule = np.random.rand(5)
                rule[np.logical_not(enabled)] = 0
                rule = rule >= rule.max()

                # Slices.
                # s0 = vmin[0]:vmax[0]
                # s1 = vmin[1]:vmax[1]
                # s2 = vmin[2]:vmax[2]
                s0 = slice(vmin[0],vmax[0])
                s1 = slice(vmin[1],vmax[1])
                s2 = slice(vmin[2],vmax[2])

                # (1) Random fill-out.
                if rule[0]:
                    assert enabled[0]
                    val = self.mode[0]  # Fill-out value.
                    if val > 1:
                        val = np.random.rand()
                    sample[key][...,s0,s1,s2] = val

                # (2) Alpha.
                if rule[1]:
                    assert enabled[1]
                    alpha = np.random.rand() * self.mode[1]
                    sample[key][...,s0,s1,s2] *= alpha

                # (3) Gaussian white noise (additive or multiplicative).
                if rule[2]:
                    assert enabled[2]
                    scale = self.mode[2]
                    if np.random.rand() < 0.5:
                        val = np.random.normal(loc=0.0, scale=scale, size=sz)
                        sample[key][...,s0,s1,s2] += val[...]
                    else:
                        val = np.random.normal(loc=1.0, scale=scale, size=sz)
                        sample[key][...,s0,s1,s2] *= val[...]

                # (4) Uniform white noise.
                if rule[3]:
                    assert enabled[3]
                    val = np.random.rand(sz[0],sz[1],sz[2])
                    # Random Gaussian blur.
                    sigma = [0,0,0]
                    sigma[0] = np.random.rand() * self.sigma_max
                    sigma[1] = np.random.rand() * self.sigma_max
                    sigma[2] = np.random.rand() * self.sigma_max
                    # Anisotropy.
                    sigma[0] /= self.aspect_ratio
                    val = gaussian_filter(val, sigma=sigma)
                    sample[key][...,s0,s1,s2] = val[...]

                # (5) 3D blur.
                if rule[4]:
                    assert enabled[4]
                    img = sample[key][...,s0,s1,s2]
                    # Random Gaussian blur.
                    sigma = [0] * img.ndim
                    sigma[-3] = np.random.rand() * self.sigma_max
                    sigma[-2] = np.random.rand() * self.sigma_max
                    sigma[-1] = np.random.rand() * self.sigma_max
                    # Anisotropy.
                    sigma[-3] /= self.aspect_ratio
                    img = gaussian_filter(img, sigma=sigma)
                    sample[key][...,s0,s1,s2] = img

                # # Update augmentation mask.
                # msk[...,s0,s1,s2] = 1

                # Stop condition.
                count += box.volume()
                if count > goal:
                    break;

            # Clip.
            sample[key] = np.clip(sample[key], 0, 1)

            # # Augmentation mask.
            # sample[key+'_augmask'] = msk

        return sample

    ####################################################################
    ## Setters.
    ####################################################################

    def set_skip_ratio(self, ratio):
        """Set the probability of skipping augmentation."""
        assert ratio >= 0.0 and ratio <= 1.0
        self.skip_ratio = ratio
