from __future__ import print_function
from collections import OrderedDict
import copy
import numpy as np

from box import Box
from sequence import SampleSequence
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
        _params: Dataset-specific parameters.
        _data: Dictionary mapping key to TensorData, each of which contains
                4D volumetric data. (e.g. EM image stacks, segmentation, etc.)
        _spec: Sample specification. Dictionary mapping key to dimension,
                which can be either a list or tuple with at least 3 elements.
        _range: Range of valid coordinates for accessing data given the sample
                spec. It depends both on the data and sample spec.
        _sequence:
        _locs: Valid locations.
    """

    def __init__(self, **kwargs):
        # Initialize attributes.
        self._reset()
        # Set dataset-specific params.
        for k, v in kwargs.items():
            self._params[k] = v

    def add_raw_data(self, key, data, fov=(0,0,0), offset=(0,0,0)):
        """Add a raw volume to the dataset."""
        self.add_data(key, TensorData(data,fov,offset))

    def add_data(self, key, data):
        """Add data to the dataset."""
        assert isinstance(data, TensorData)
        self._data[key] = data

    def add_raw_mask(self, key, data, loc=False, **kwargs):
        self.add_raw_data(key, data, **kwargs)
        if loc:
            self._add_location(self._data[key])

    def add_mask(self, key, data, loc=False):
        self.add_data(key, data)
        if loc:
            self._add_location(self._data[key])

    def set_sequence(self, seq):
        """Add sample sequence generator."""
        assert isinstance(seq, SampleSequence)
        self._sequence = seq

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
            if key in self._data:
                patch = self._data[key].get_patch(pos)
                if patch is None:
                    raise
                else:
                    sample[key] = patch
        return sample

    def next_sample(self, spec=None):
        """Fetch the next sample in a predefined sequence, if any."""
        if self._sequence is None:
            ret = self.random_sample(spec=spec)
        else:
            assert self.has_spec()
            original_spec = self.get_spec()
            try:
                # Dynamically change spec.
                if spec is not None: self.set_spec(spec)
                # Pick a random sample.
                pos = self._sequence()
                ret = self.get_sample(pos)
                # Revert to the original sample spec.
                if spec is not None: self.set_spec(original_spec)
            except:
                self.set_spec(original_spec)
                raise
        return ret

    def random_sample(self, spec=None):
        """Fetch a sample randomly."""
        assert self.has_spec()
        original_spec = self.get_spec()
        try:
            # Dynamically change spec.
            if spec is not None: self.set_spec(spec)
            # Pick a random sample.
            pos = self._random_location()
            ret = self.get_sample(pos)
            # Revert to the original sample spec.
            if spec is not None: self.set_spec(original_spec)
        except:
            self.set_spec(original_spec)
            raise
        return ret

    ####################################################################
    ## Getters and setters.
    ####################################################################

    def get_spec(self):
        """Return sample spec."""
        return copy.deepcopy(self._spec)

    def set_spec(self, spec):
        """Set smaple spec and update the valid range of data samples."""
        # Order by key.
        self._spec = OrderedDict(sorted(spec.items(), key=lambda x: x[0]))
        self._update_range()

    def has_spec(self):
        return self._spec is not None

    def get_param(self, key):
        assert key in self._params
        return self._params[key]

    def set_param(self, key, value):
        self._params[key] = value

    def get_params(self):
        return copy.deepcopy(self._params)

    def num_sample(self):
        """Return the number of samples."""
        n = 0
        if self._sequence is None:
            s = self._range.size()
            n = s[0]*s[1]*s[2]
        else:
            if self._locs is None:
                n = self._sequence.get_length()
            else:
                n = len(self._locs[0])
        return n

    def get_range(self):
        """Return the valid range box."""
        return Box(self._range)

    ####################################################################
    ## Private Helper Methods.
    ####################################################################

    def _reset(self):
        """Reset all attributes."""
        self._params   = dict()
        self._data     = dict()
        self._spec     = None
        self._range    = None
        self._sequence = None
        # Valid locations (optional).
        self._locs     = None
        self._offset   = None

    def _add_location(self, data):
        assert isinstance(data, TensorData)
        self._locs   = data.get_data().nonzero()[-3:]
        self._offset = data.offset()

    def _random_location(self):
        """Return one of the valid locations randomly."""
        if self._locs is None:
            s = self._range.size()
            z = np.random.randint(0, s[0])
            y = np.random.randint(0, s[1])
            x = np.random.randint(0, s[2])
            # Global coordinate system.
            loc =  Vec3d(z,y,x) + self._range.min()
            # DEBUG(kisuk)
            # print('loc = {}'.format(loc))
        else:
            while True:
                idx = np.random.randint(0, self._locs[0].size)
                loc = tuple([x[idx] for x in self._locs])
                # Global coordinate system.
                loc = Vec3d(loc) + self._offset
                if self._range.contains(loc):
                    break
        return loc

    def _update_range(self):
        """Update the valid range.

        Compute the intersection of the valid range of each TensorData.
        """
        assert self.has_spec()
        # Valid range.
        vr = None
        for key, dim in self._spec.items():
            if key in self._data:
                # Update patch size.
                self._data[key].set_fov(dim[-3:])
                # Update valid range.
                r = self._data[key].range()
                vr = r if vr is None else vr.intersect(r)
        self._range = vr


########################################################################
## VolumeDataset demo.
########################################################################

if __name__ == "__main__":

    import argparse
    import emio
    import h5py
    import os
    import time
    import transform

    dsc = 'VolumeDataset demo.'
    parser = argparse.ArgumentParser(description=dsc)

    parser.add_argument('z', type=int, help='sample z dim.')
    parser.add_argument('y', type=int, help='sample y dim.')
    parser.add_argument('x', type=int, help='sample x dim.')
    parser.add_argument('img', help='image file (h5 or tif) path.')
    parser.add_argument('lbl', help='label file (h5 or tif) path.')

    args = parser.parse_args()

    # Load data.
    img = emio.imread(args.img)
    lbl = emio.imread(args.lbl)

    # Preprocess.
    img = transform.divideby(img, val=255.0)

    # Create dataset and add data.
    vdset = VolumeDataset()
    vdset.add_raw_data(key='input', data=img)
    vdset.add_raw_data(key='label', data=lbl)

    # Random sample.
    size = (args.z, args.y, args.x)
    spec = dict(input=size, label=size)
    vdset.set_spec(spec)
    sample = vdset.random_sample()

    # Dump a single random sample.
    print('Save as file...')
    fname = 'sample.h5'
    if os.path.exists(fname):
        os.remove(fname)
    f = h5py.File(fname)
    for key, data in sample.items():
        f.create_dataset('/' + key, data=data)
    f.close()
