#!/usr/bin/env python
__doc__ = """

Probabilistic sample sequence generator.

Kisuk Lee <kisuklee@mit.edu>, 2017
"""

import numpy as np

import emio

class SampleSequence(object):
    """
    Sample sequence generator.

    Attributes:
        probmap:    Probability map.
        indexmap:   Map of sampled coordinates.
        length:     Sequence length.
        sequence:   Sampled coordinate sequence.
        pointer:    Sample pointer.
    """

    def __init__(self, probmap, length):
        self.set_probability(probmap)
        self.length = length
        self.sample()

    def set_probability(self, probmap):
        if type(probmap) is str:
            probmap = emio.imread(probmap)
        assert isinstance(probmap, np.ndarray)
        self.probmap = probmap

    def get_sample_coord(self):
        # Resample if sequence is empty.
        if len(self.pointer) == 0:
            self.sample()
        p = self.pointer.pop()
        z = self.sequence[0][p]
        y = self.sequence[1][p]
        x = self.sequence[2][p]
        return (z, y, x)

    def sample(self):
        """Randomly generate a sample sequence of given length."""
        rnd = np.random.rand(*self.probmap.shape)
        idx = self.probmap > rnd
        seq = np.nonzero(idx)
        assert len(seq) == 3
        self.sequence = seq
        self.indexmap = idx
        # Randomly permute the sampled coords, and cut off the sequence
        # according to the given length.
        pointer = np.random.permutation(len(seq[0]))
        self.pointer = list(pointer[:self.length])
