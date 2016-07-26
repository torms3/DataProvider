#!/usr/bin/env python
__doc__ = """

Dataset classes.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import copy
from collections import OrderedDict
import numpy as np
from box import Box
from config_data import *
import emio
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
        _data:
        _label:
        _spec:
        _range:
    """

    def __init__(self, config):
        """Initialize VolumeDataset."""
        self.build_from_config(config)

    def reset(self):
        """Reset all attributes."""
        self._data  = dict()
        self._label = list()
        self._spec  = None
        self._range = None

    def build_from_config(self, config):
        """Build dataset from config."""
        self.reset()

        # First pass for images and labels.
        for name, data in config.items('dataset'):
            assert config.has_section(data)
            if '_mask' in data:
                continue
            if 'label' in data:
                self._data[name] = ConfigLabel(config, data)
                self._label.append(name)
            else:
                self._data[name] = ConfigData(config, data)

        # Second pass for masks.
        for name, data in config.items('dataset'):
            if '_mask' in data:
                if config.has_option(data, 'shape'):
                    label = data.strip('_mask')
                    shape = self._data[label].shape()
                    config.set(data, 'shape', shape)
                self._data[name] = ConfigData(config, data)

        # Set spec.
        spec = {}
        for name, data in self._data.iteritems():
            spec[name] = tuple(data.fov())
        self.set_spec(spec)

    def get_spec(self):
        return copy.deepcopy(self._spec)

    def set_spec(self, spec):
        """Set spec and update valid range."""
        self._spec = spec
        self._update_range()

    def num_sample(self):
        s = self._range.size()
        return s[0]*s[1]*s[2]

    def get_sample(self, pos, spec=None):
        """Draw a sample centered on pos.

        Args:
            pos:
            spec:

        Returns:
            data:
            transform:
        """
        # Dynamically change spec.
        if spec is not None:
            original_spec = self._spec
            self.set_spec(spec)

        data = dict()
        for name in self._spec.keys():
            data[name] = self._data[name].get_patch(pos)

        transform = dict()
        for name in self._label:
            transform[name] = self._data[name].get_transform()

        # Return to original spec.
        if spec is not None:
            self.set_spec(original_spec)

        # Order by key
        sample = OrderedDict(sorted(data.items(), key=lambda x: x[0]))

        return sample, transform

    def next_sample(self, spec=None):
        """Fetch next sample in a sample sequence."""
        return self.random_sample(spec)  # Currently just pick randomly.

    def random_sample(self, spec=None):
        """Fetch sample randomly"""
        pos = self._random_location()
        return self.get_sample(pos, spec)

    ####################################################################
    ## Private Helper Methods
    ####################################################################

    def _random_location(self):
        """Return one of the valid locations randomly."""
        s = self._range.size()
        z = np.random.randint(0, s[0])
        y = np.random.randint(0, s[1])
        x = np.random.randint(0, s[2])
        return Vec3d(z,y,x) + self._range.min()

    def _update_range(self):
        """Update valid range."""
        self._range = None
        for name, dim in self._spec.iteritems():
            # Update patch size.
            self._data[name].set_fov(dim[-3:])
            # Update valid range.
            r = self._data[name].range()
            if self._range is None:
                self._range = r
            else:
                self._range = self._range.intersect(r)