#!/usr/bin/env python
__doc__ = """

Greyscale value augmentation.

Kisuk Lee <kisuklee@mit.edu>, 2017
"""

import numpy as np

import augmentor

class GreyAugment(Augmentor.DataAugment):
    """
    Greyscale value augmentation.

    Randomly adjust contrast/brightness, and apply random gamma correction.
    """

    def __init__(self, mode='mix', skip_ratio=0.3):
        """Initialize parameters.

        Args:
            mode: '2D', '3D', 'mix'
            skip_ratio: Probability of skipping augmentation.
        """
        assert mode=='3D' or mode=='2D' or mode=='mix'
        self.mode  = mode
        self.ratio = skip_ratio
        self.CONTRAST_FACTOR   = 0.3
        self.BRIGHTNESS_FACTOR = 0.3

    def prepare(self, spec, **kwargs):
        # No change in sample spec.
        return spec

    def __call__(self, sample, **kwargs):
        #print '\n[GreyAugment]'  # DEBUG
        ret = sample
        if np.random.rand() > self.ratio:
            if self.mode == 'mix':
                mode = '3D' if np.random.rand() > 0.5 else '2D'
            else:
                mode = self.mode
            ret = eval('self.augment{}(sample, **kwargs)'.format(mode))
        return ret

    def augment2D(self, sample, **kwargs):
        """
        Adapted from ELEKTRONN (http://elektronn.org/).
        """
        #print '2D greyscale augmentation'  # DEBUG

        # Greyscale augmentation.
        imgs = kwargs['imgs']
        for key in imgs:
            for z in xrange(sample[key].shape[-3]):
                img = sample[key][...,z,:,:]
                img *= 1 + (np.random.rand() - 0.5)*self.CONTRAST_FACTOR
                img += (np.random.rand() - 0.5)*self.BRIGHTNESS_FACTOR
                img = np.clip(img, 0, 1)
                img **= 2.0**(np.random.rand()*2 - 1)
                sample[key][...,z,:,:] = img

        return sample

    def augment3D(self, sample, **kwargs):
        """
        Adapted from ELEKTRONN (http://elektronn.org/).
        """
        #print '3D greyscale augmentation'  # DEBUG

        # Greyscale augmentation.
        imgs = kwargs['imgs']
        for key in imgs:
            sample[key] *= 1 + (np.random.rand() - 0.5)*self.CONTRAST_FACTOR
            sample[key] += (np.random.rand() - 0.5)*self.BRIGHTNESS_FACTOR
            sample[key] = np.clip(sample[key], 0, 1)
            sample[key] **= 2.0**(np.random.rand()*2 - 1)

        return sample
