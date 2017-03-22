#!/usr/bin/env python
__doc__ = """

Missing section data augmentation.

Kisuk Lee <kisuklee@mit.edu>, 2017
"""

import data_augmentation
import numpy as np

class MissingAugment(data_augmentation.DataAugment):
    """
    Missing section data augmentation.

    Introduce missing section(s) to a training example. The number of missing
    sections to introduce is randomly drawn from the uniform distribution
    between [0, MAX_SEC]. Default MAX_SEC is 1, which can be overwritten by
    a user-specified value.
    """

    def __init__(self, max_sec=1, skip_ratio=0.3, mode='full',
                 consecutive=False, random_color=False):
        """Initialize MissingSectionAugment."""
        self.set_max_sections(max_sec)
        self.set_skip_ratio(skip_ratio)
        self.set_mode(mode)
        self.consecutive = consecutive
        self.random_color = random_color

        # DEBUG(kisuk)
        # self.hist = [0] * (max_sec + 1)

    def set_max_sections(self, max_sec):
        """Set the maximum number of missing sections to introduce."""
        assert max_sec >= 0
        self.MAX_SEC = max_sec

    def set_skip_ratio(self, ratio):
        """Set the probability of skipping augmentation."""
        assert ratio >= 0.0 and ratio <= 1.0
        self.skip_ratio = ratio

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
        if np.random.rand() > self.skip_ratio:
            sample = self._do_augment(sample, **kwargs)

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

        # Assume that the sample contains only one input volume, or multiple
        # input volumes of same size.
        imgs = kwargs['imgs']
        dims = set([])
        for key in imgs:
            dim = self.spec[key][-3:]
            assert num_sec < dim[-3]
            dims.add(dim)
        assert len(dims) == 1
        dim  = dims.pop()
        xdim = dim[-1]
        ydim = dim[-2]
        zdim = dim[-3]

        # Randomly draw z-slices to black out.
        if self.consecutive:
            zloc  = np.random.randint(0, zdim - num_sec + 1)
            zlocs = range(zloc, zloc + num_sec)
        else:
            zlocs = np.random.choice(zdim, num_sec, replace=False)

        # Fill-out value.
        val = np.random.rand() if self.random_color else 0

        # Apply full or partial missing sections according to the mode.
        if self.mode == 'full':
            for key in imgs:
                sample[key][...,zlocs,:,:] = val
        else:
            # Draw a random xy-coordinate.
            x = np.random.randint(0, xdim)
            y = np.random.randint(0, ydim)
            rule = np.random.rand(4) > 0.5

            for z in zlocs:
                val = np.random.rand() if self.random_color else 0
                if self.mode == 'mix' and np.random.rand() > 0.5:
                    for key in imgs:
                        sample[key][...,z,:,:] = val
                else:
                    # Independent coordinates across sections.
                    if not self.consecutive:
                        x = np.random.randint(0, xdim)
                        y = np.random.randint(0, ydim)
                        rule = np.random.rand(4) > 0.5
                    # 1st quadrant.
                    if rule[0]:
                        for key in imgs:
                            sample[key][...,z,:y,:x] = val
                    # 2nd quadrant.
                    if rule[1]:
                        for key in imgs:
                            sample[key][...,z,y:,:x] = val
                    # 3nd quadrant.
                    if rule[2]:
                        for key in imgs:
                            sample[key][...,z,:y,x:] = val
                    # 4nd quadrant.
                    if rule[3]:
                        for key in imgs:
                            sample[key][...,z,y:,x:] = val

        return sample
