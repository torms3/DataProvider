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

    def __init__(self, min_dim, max_dim, aspect_ratio, density):
        """Initialize BoxAugment."""
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.aspect_ratio = aspect_ratio
        self.density = density
        # self.min_dim = 20
        # self.max_dim = 60
        # self.aspect_ratio = 6
        # self.density = 0.2
        # self.alpha = 0.5

    def prepare(self, spec, **kwargs):
        """Prepare mask."""
        # No change in spec.
        self.spec = spec

        # Find union of bounding boxes.
        self.bbox = dict()
        bbox = None
        imgs = kwargs['imgs']
        for key in imgs:
            dim = spec[key][-3:]
            b = centered_box((0,0,0), dim)
            bbox = b if bbox is None else bbox.merge(b)
            self.bbox[key] = b

        # Create a mask.
        self.offset = bbox.min()
        self.dim    = bbox.size()
        self.mask   = np.ones(self.dim, dtype='float32')

        # Random box augmentation.
        count = 0
        goal  = bbox.volume()*self.density
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
            dim[0] /= self.aspect_ratio
            # Alpha.
            # alpha = np.random.rand() * self.alpha
            # Box.
            box = bbox.intersect(centered_box(loc, dim))
            # Local coordiate.
            box.translate(-self.offset)
            vmin = box.min()
            vmax = box.max()
            # Apply box.
            # self.mask[vmin[0]:vmax[0],vmin[1]:vmax[1],vmin[2]:vmax[2]] *= alpha
            self.mask[vmin[0]:vmax[0],vmin[1]:vmax[1],vmin[2]:vmax[2]] = 0
            # Stop condition.
            count += box.volume()
            if count > goal:
                break;

        return dict(spec)

    def augment(self, sample, **kwargs):
        """Apply box data augmentation."""
        imgs = kwargs['imgs']
        for key in imgs:
            b = self.bbox[key]
            b.translate(-self.offset)
            vmin = b.min()
            vmax = b.max()
            sample[key][...,:,:,:] *= self.mask[
                vmin[0]:vmax[0],vmin[1]:vmax[1],vmin[2]:vmax[2]]

        return sample
