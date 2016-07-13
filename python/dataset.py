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
        _imgs: Image dict. {key: [preprocessing1,...]}
        _lbls: Label dict. {key: [preprocessing1,...]}
        _msks:
        _spec:
        _range:
    """

    def __init__(self, config=None, dataset_id=None):
        """Build dataset from config, if any."""
        if config_path is not None:
            self.build_from_config(config, dataset_id)
        else:
            self.reset()

    def reset(self):
        """Reset all attributes."""
        self._data  = {}
        self._spec  = {}
        self._range = Box()

    def build_from_config(self, config, dataset_id):
        """
        TODO(kisuk): Documentation.
        """
        self.reset()

        # Construct a ConfigParser object.
        config = ConfigParser.ConfigParser()
        config.read(config_path)
        section = 'dataset%d' & dataset_id

        # Build dataset.
        for key, val in config.items(section):
            assert config.has_section(val)
            if 'image' in val:
                self._data[key] = ConfigImage(config, val)
            elif 'label' in val:
                self._data[key] = ConfigLabel(config, val)
                # Add mask.
                config.has_option(val,)
            else:
                raise RuntimeError('unknown section type [%s]' % data)

    def add_image(self, name, data, offset=(0,0,0)):
        """
        TODO(kisuk): Documentation.
        """
        self._data[name] = TensorData(data, offset=offset)

    def add_label(self, name, data, offset=(0,0,0), mask=None):
        """
        TODO(kisuk): Documentation.
        """
        lbl = TensorData(data, offset=offset)
        self._data[name] = lbl
        self._lbls.append(name)

        # Add a corresponding mask.
        if mask is None:
            mask = np.ones(lbl.shape(), dtype='float32')
        else:
            assert lbl.shape()==mask.shape

        mask_name = self._mask_name(name)
        assert mask_name is not None
        self._data[mask_name] = TensorData(mask, lbl.offset())
        self._msks.append(mask_name)

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