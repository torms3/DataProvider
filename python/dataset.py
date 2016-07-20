#!/usr/bin/env python
__doc__ = """

Dataset classes.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

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

    def __init__(self):
        pass

    def next_sample(self):
        pass

    def random_sample(self):
        pass


class VolumeDataset(Dataset):
    """
    Dataset for volumetric data.

    Attributes:
        _data:
        _range:
    """

    def __init__(self, config):
        """Build dataset from config."""
        self.build_from_config(config)

    def reset(self):
        """Reset all attributes."""
        self._data  = {}
        self._range = Box()

    def build_from_config(self, config):
        """
        TODO(kisuk): Documentation.
        """
        self.reset()

        # First pass for images and labels.
        for name, data in config.items('dataset'):
            assert config.has_section(data)
            if '_mask' in data:
                continue
            self._data[name] = ConfigData(config, data)

        # Second pass for masks.
        for name, data in config.items('dataset'):
            if '_mask' in data:
                if config.has_option(data, 'shape'):
                    label = data.strip('_mask')
                    shape = self._data[label].shape()
                    config.set(data, 'shape', shape)
                self._data[name] = ConfigData(config, data)

    def set_spec(self, spec):
        """
        TODO(kisuk): Documentation.
        """
        self._spec = spec
        self._update_range()

    def get_sample(self, pos, spec=None):
        """Draw a sample centered on pos.

        Args:
            pos:
            spec:

        Returns:

        """
        data, imgs, lbls, msks = {}, [], [], []

        if spec is None:
            spec = self._spec    # Use the current spec.
        else:
            self.set_spec(spec)  # Dynamically change spec.

        for name in self._spec.keys():
            data[name] = self._data[name].get_patch(pos)
            if name in self._imgs: imgs.append(name)
            if name in self._lbls: lbls.append(name)
            if name in self._msks: msks.append(name)

        assert (len(imgs)+len(lbls)+len(msks))==len(data)
        # TODO(kisuk): Which one is better? Multiple returns or a dictionary?
        # return data, imgs, lbls, msks
        return {'data':data, 'imgs':imgs, 'lbls':lbls, 'msks':msks}

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

    def _mask_name(self, name):
        if name in self._lbls:
            ret = name + '_name'
        else:
            ret = None
        return ret

    def _update_range(self):
        """Update valid range."""
        self._range = None

        for name, dim in self._spec.iteritems():
            # Update patch size.
            self._data[name].set_fov(dim[-3:])
            # Update mask, if any.
            mask_name = self._mask_name(name)
            if mask_name is not None:
                self._data[mask_name].set_fov(dim[-3:])

            # Update valid range.
            r = self._data[name].range()
            if self._range is None:
                self._range = r
            else:
                self._range = self._range.intersect(r)