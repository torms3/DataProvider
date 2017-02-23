#!/usr/bin/env python
__doc__ = """

Dataset classes.

Kisuk Lee <kisuklee@mit.edu>, 2016-2017
"""

from collections import OrderedDict
import copy
import numpy as np

from box import Box
from tensor import TensorData
from vector import Vec3d

class Dataset(object):
    """
    Dataset interface.
    """

    def next_sample(self):
        raise NotImplementedError

    def random_sample(self):
        raise NotImplementedError


class VolumeDataset(Dataset):
    """
    Dataset for volumetric data.

    Attributes:
        params: Dataset-specific parameters.

        _data:  Dictionary mapping key to TensorData, each of which contains
                4D volumetric data. (e.g. EM image stacks, segmentation, etc.)
        _spec:  Sample specification. Dictionary mapping key to dimension,
                which can be either a list or tuple with at least 3 elements.
        _range: Range of valid coordinates for accessing data given the sample
                spec. It depends both on the data and sample spec.
    """

    def __init__(self, **kwargs):
        # Initialize attributes.
        self._reset()
        # Set dataset-specific params.
        for k, v in kwargs.iteritems():
            self.params[k] = v

    def add_raw_data(self, key, data, fov=(0,0,0), offset=(0,0,0)):
        """Add a raw volume to the dataset."""
        self.add_data(key, TensorData(data,fov,offset))

    def add_data(self, key, data):
        """Add data to the dataset."""
        # TODO(kisuk): Check if data is TensorData.
        self._data[key] = data

    def get_spec(self):
        """Return sample spec."""
        return copy.deepcopy(self._spec)

    def set_spec(self, spec):
        """Set smaple spec and update the valid range of data samples."""
        # Order by key.
        self._spec = OrderedDict(sorted(spec.items(), key=lambda x: x[0]))
        self._update_range()

    def num_sample(self):
        """Return the number of samples."""
        s = self._range.size()
        return s[0]*s[1]*s[2]

    def get_range(self):
        """Return the valid range box."""
        return Box(self._range)

    def get_sample(self, pos):
        """Extract a sample centered on pos.

        Every data in the sample is guaranteed to be center-aligned.

        Args:
            pos: Center coordinates of the sample.

        Returns:
            Sample, a dictionary mapping key to data.
        """
        sample = OrderedDict()
        for key in self._spec.keys():
            sample[key] = self._data[key].get_patch(pos)
        return sample

    def next_sample(self, spec=None):
        """Fetch the next sample in a predefined sequence, if any."""
        # Currently just pick randomly.
        return self.random_sample(spec)

    def random_sample(self, spec=None):
        """Fetch sample randomly"""
        # Dynamically change spec.
        if spec is not None:
            original_spec = self._spec
            try:
                self.set_spec(spec)
            except:
                # It's very important to revert to the original sample spec
                # when failed to set the dynamic spec.
                self.set_spec(original_spec)
                raise

        # Pick a random sample.
        pos = self._random_location()
        ret = self.get_sample(pos)

        # Revert to the original sample spec.
        if spec is not None:
            self.set_spec(original_spec)

        return ret

    ####################################################################
    ## Private Helper Methods.
    ####################################################################

    def _reset(self):
        """Reset all attributes."""
        self.params = dict()
        self._data  = dict()
        self._spec  = None
        self._range = None

    def _random_location(self):
        """Return one of the valid locations randomly."""
        s = self._range.size()
        z = np.random.randint(0, s[0])
        y = np.random.randint(0, s[1])
        x = np.random.randint(0, s[2])
        # Global coordinate system.
        return Vec3d(z,y,x) + self._range.min()

    def _update_range(self):
        """Update the valid range.

        Compute the intersection of the valid range of each TensorData.
        """
        assert self._spec is not None
        # Valid range.
        vr = None
        for key, dim in self._spec.iteritems():
            # Update patch size.
            self._data[key].set_fov(dim[-3:])
            # Update valid range.
            r = self._data[key].range()
            vr = r if vr is None else vr.intersect(r)
        self._range = vr
