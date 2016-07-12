#!/usr/bin/env python
__doc__ = """

Dataset classes.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import numpy as np
from box import *
from tensor import *
from vector import *

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
        _data: Data dictionary {key: tensor_data}.
        _imgs: Image key list.
        _lbls: Label key list.
        _msks: Mask  key list.
        _spec:
        _range:
    """

    def __init__(self, config=None):
        """
        TODO(kisuk): Documentation.
        """
        if config is not None:
            self.build_from_config()
        else:
            self.reset()

    def reset(self):
        """
        TODO(kisuk): Documentation.
        """
        self._data  = {}
        self._imgs  = []
        self._lbls  = []
        self._msks  = []
        self._spec  = {}
        self._range = Box()

    def build_from_config(self, config):
        """
        TODO(kisuk): Documentation.
        """
        self.reset()

    def add_image(self, name, data, offset=(0,0,0)):
        """
        TODO(kisuk): Documentation.
        """
        self._data[name] = TensorData(data, offset=offset)
        self._imgs.append(name)

    def add_label(self, name, data, offset=(0,0,0), mask=None):
        """
        TODO(kisuk): Documentation.
        """
        lbl = TensorData(data, offset=offset)
        self._data[name] = lbl
        self._lbls.append(name)

        # Add a corresponding mask.
        if mask is None:
            # lbl is a TensorData object, which has the shape() method.
            # Don't be confused with numpy array's shape attribute.
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

        # Update valid range as it could be changed.
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