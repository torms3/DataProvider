#!/usr/bin/env python
__doc__ = """

Missing section data augmentation.

Kisuk Lee <kisuklee@mit.edu>, 2017
"""

import data_augmentation
import numpy as np

class MissingAugment(data_augmentation.DataAugment):
    """
    Introduce missing section(s) to a training example. The number of missing
    sections to introduce is randomly drawn from the uniform distribution
    between [0, MAX_SEC]. Default MAX_SEC is 1, which can be overwritten by
    user-specified value.
    """

    def __init__(self, max_sec=1, skip_ratio=0.3, mode='mix'):
        """Initialize MissingSectionAugment."""
        self.set_max_sections(max_sec)
        self.set_skip_ratio(skip_ratio)
        self.set_mode(mode)

        # DEBUG(kisuk)
        # self.hist = [0] * (max_sec + 1)

    def set_max_sections(self, max_sec):
        """Set the maximum number of missing sections to introduce."""
        assert max_sec >= 0
        self.MAX_SEC = max_sec

    def set_skip_ratio(self, ratio):
        """Set the probability of skipping augmentation."""
        assert skip_ratio >= 0.0 and skip_ratio <= 1.0
        self.ratio = ratio

    def set_mode(self, mode):
        """Set full/partial/mix missing section mode."""
        assert mode=='full' or mode=='partial' or mode=='mix'
        self.mode = mode

    def prepare(self, spec, **kwargs):
        """No change in spec."""
        self.spec = spec
        return dict(spec)

    def augment(self, sample, **kwargs):
        """Apply missing section data augmentation."""
        if np.random.rand() > self.ratio:
            self._do_augment(sample, kwargs)

        # DEBUG(kisuk): Record keeping.
        #     self.hist[num_sec] += 1
        # else:
        #     self.hist[0] += 1

        # DEBUG(kisuk)
        # count = sum(self.hist)
        # hist = [x/float(count) for x in self.hist]
        # stat = "[ "
        # for v in hist:
        #     stat += "{} ".format("%0.3f" % v)
        # stat += "]"
        # print stat

        return sample

    def _do_augment(self, sample, **kwargs):
        """Apply missing section data augmentation."""
        # Randomly draw the number of sections to introduce.
        num_sec = np.random.randint(1, self.MAX_SEC + 1)

        # DEBUG(kisuk)
        # print "num_sec = %d" % num_sec

        # Assume that the sample contains only one input volume,
        # or multiple input volumes of same size.
        imgs  = kwargs['imgs']
        zdims = set([])
        for key in imgs:
            zdim = self.spec[key][-3]
            assert num_sec < zdim
            zdims.add(zdim)
        assert len(zdims) == 1
        zdim = zdims.pop()

        # Randomly draw z-slices to black out.
        zlocs = np.random.choice(zdim, num_sec, replace=False)

        # Apply full or partial missing sections according to the mode.
        if self.mode == 'full':
            for key in imgs:
                sample[key][...,zlocs,:,:] *= 0
        else:
            for z in zlocs:
                if self.mode == 'mix' and np.random.rand() > 0.5:
                    for key in imgs:
                        sample[key][...,z,:,:] *= 0
                else:
                    xdim = self.spec[key][-1]
                    ydim = self.spec[key][-2]
                    # Draw a random xy-coordinate.
                    x = np.random.randint(0, xdim)
                    y = np.random.randint(0, ydim)
                    # 1st quadrant.
                    if np.random.rand() > 0.5:
                        for key in imgs:
                            sample[key][...,z,:y,:x] *= 0
                    # 2nd quadrant.
                    if np.random.rand() > 0.5:
                        for key in imgs:
                            sample[key][...,z,y:,:x] *= 0
                    # 3nd quadrant.
                    if np.random.rand() > 0.5:
                        for key in imgs:
                            sample[key][...,z,:y,x:] *= 0
                    # 4nd quadrant.
                    if np.random.rand() > 0.5:
                        for key in imgs:
                            sample[key][...,z,y:,x:] *= 0

        return sample
