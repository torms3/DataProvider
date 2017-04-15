#!/usr/bin/env python
__doc__ = """

Box augmentation.

Kisuk Lee <kisuklee@mit.edu>, 2017
"""

from box import *
import data_augmentation
import math
import numpy as np

class BoxAugment(data_augmentation.DataAugment):
    """
    Add random box masks.
    """

    def __init__(self, min_dim, max_dim, aspect_ratio, density, skip_ratio=0.3, random_color=False):
        """Initialize BoxAugment."""
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.aspect_ratio = aspect_ratio
        self.density = density
        self.set_skip_ratio(skip_ratio)
        self.random_color = random_color
        # self.min_dim = 20
        # self.max_dim = 60
        # self.aspect_ratio = 6
        # self.density = 0.2

    def prepare(self, spec, **kwargs):
        """Prepare mask."""
        # No change in spec.
        self.spec = spec
        return dict(spec)

    def augment(self, sample, **kwargs):
        """Apply box data augmentation."""
        if np.random.rand() > self.skip_ratio:
            sample = self._do_augment(sample, **kwargs)
        return sample

    def _do_augment(self, sample, **kwargs):
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
            count = 0
            goal  = bbox.volume()*self.density*np.random.rand()
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
                val = np.random.rand() if self.random_color else 0  # Fill-out value.
                sample[key][...,vmin[0]:vmax[0],vmin[1]:vmax[1],vmin[2]:vmax[2]] = val
                # Stop condition.
                count += box.volume()
                if count > goal:
                    break;
        return sample

    def set_skip_ratio(self, ratio):
        """Set the probability of skipping augmentation."""
        assert ratio >= 0.0 and ratio <= 1.0
        self.skip_ratio = ratio
