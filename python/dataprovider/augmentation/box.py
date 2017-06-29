#!/usr/bin/env python
__doc__ = """

Box occlusion augmentation.

Kisuk Lee <kisuklee@mit.edu>, 2017
"""

import math
import numpy as np

import augmentor
from ..box import *

class BoxOcclusion(augmentor.DataAugment):
    """
    Add random box occlusion masks.
    """

    def __init__(self, min_dim, max_dim, aspect_ratio, max_density, skip_ratio=0.0, random_color=True):
        """Initialize BoxAugment."""
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.aspect_ratio = aspect_ratio
        self.max_density = max_density
        self.set_skip_ratio(skip_ratio)
        self.random_color = random_color

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
            # Random box augmentation.
            count   = 0
            density = self.max_density*np.random.rand()
            goal    = bbox.volume()*density
            print 'density: %.2f' % density
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
                rule = np.random.rand(4)
                rule = rule >= rule.max()

                # (1) Alpha.
                if rule[0]:
                    alpha = np.random.rand() * 2.0
                    sample[key][...,vmin[0]:vmax[0],vmin[1]:vmax[1],vmin[2]:vmax[2]] *= alpha

                # (2) Random fill-out.
                if rule[1]:
                    val = np.random.rand() if self.random_color else 0  # Fill-out value.
                    sample[key][...,vmin[0]:vmax[0],vmin[1]:vmax[1],vmin[2]:vmax[2]] = val

                # (3) Gaussian white noise (additive).
                if rule[2]:
                    # TODO(kisuk): Parametrize scale.
                    val = np.random.normal(loc=0.0, scale=0.3, size=sz)
                    sample[key][...,vmin[0]:vmax[0],vmin[1]:vmax[1],vmin[2]:vmax[2]] += val[...]

                # (4) Uniform white noise.
                if rule[3]:
                    val = np.random.rand(sz[0],sz[1],sz[2])
                    sample[key][...,vmin[0]:vmax[0],vmin[1]:vmax[1],vmin[2]:vmax[2]] = val[...]

                # Stop condition.
                count += box.volume()
                if count > goal:
                    break;

            # Clip.
            sample[key] = np.clip(sample[key], 0, 1)

        return sample

    ####################################################################
    ## Setters.
    ####################################################################

    def set_skip_ratio(self, ratio):
        """Set the probability of skipping augmentation."""
        assert ratio >= 0.0 and ratio <= 1.0
        self.skip_ratio = ratio
