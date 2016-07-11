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
        self._data  = {}
        self._imgs  = []
        self._lbls  = []
        self._msks  = []
        self._spec  = {}
        self._range = Box()

        if config is not None:
            # TODO(kisuk): Parse config to consturct dataset.
            pass

    def add_image(self, name, data, offset=None):
        """
        TODO(kisuk): Documentation.
        """
        self._data[name] = self._check_data(data, offset)
        self._imgs.append(name)

    def add_label(self, name, data, offset=None, mask=None):
        """
        TODO(kisuk): Documentation.
        """
        lbl = self._check_data(data, offset)
        self._data[name] = lbl
        self._lbls.append(name)

        # Add mask.
        if mask is None:
            # lbl is a TensorData object, which has the shape() method.
            # Don't be confused with numpy array's shape attribute.
            mask = np.ones(lbl.shape(), dtype='float32')

        mask_name = name + '_mask'
        offset = lbl.offset()
        self._data[mask_name] = self._check_data(mask, offset)
        self._msks.append(mask_name)

    def set_spec(self, spec):
        """
        TODO(kisuk): Documentation.
        """
        for key, val in spec.iteritems():
            # Replace spec only when key already exists.
            # Adding a new key-value pair is only possible through add_image
            # and add_label.
            if key in self._spec:
                self._spec[key] = val

        # Update valid range as it could be changed.
        self._update_range()

    def get_sample(self, pos, spec=None):
        """
        TODO(kisuk): Documentation.
        """
        data, imgs, lbls, msks = {}, [], [], []

        if spec is None:
            spec = self._spec

        for name in self.spec.keys():
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

    def _check_data(self, data, offset):
        """
        TODO(kisuk): Documentation.
        """
        if offset is None:
            offset = (0,0,0)

        if isinstance(data, TensorData):
            ret = data
        else:
            ret = TensorData(data, offset=offset)

        return ret

    def _update_range(self):
        """
        TODO(kisuk): Documentation.
        """
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

