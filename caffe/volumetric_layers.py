#!/usr/bin/env python
__doc__ = """

Volumetric data layer for Caffe.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import caffe
import numpy as np
from python.data_provider import VolumeDataProvider

class VolumeDataLayer(caffe.Layer):
    """
    TODO(kisuk): Documentation.
    """

    def setup(self, bottom, top):
        """
        TODO(kisuk): Documentation.
        """
        # Config
        params = eval(self.param_str)
        # self.voc_dir = params['voc_dir']
        # self.split = params['split']
        # self.mean = np.array(params['mean'])
        # self.random = params.get('randomize', True)
        # self.seed = params.get('seed', None)

        # TODO(kisuk): Set data provider.
        self.data_provider = VolumeDataProvider(dspec_path, net_spec, params)

        # TODO(kisuk): Infer ntop
        if len(top) != ntop:
            raise Exception('Need to define %d tops.' % ntop)
        # Data layers have no bottoms.
        if len(bottom) != 0:
            raise Exception('Do not define a bottom.')

    def reshape(self, bottom, top):
        # load image + label image pair
        self.sample = self.data_provider.random_sample()
        # Reshape tops to fit (leading 1 is for batch dimension).
        idx = 0
        for _, data in iter(sorted(self.sample.iteritems())):
            top[idx].reshape(1, *data.shape)
            idx += 1

    def forward(self, bottom, top):
        # Assign output.
        idx = 0
        for _, data in iter(sorted(self.sample.iteritems())):
            top[idx].data[...] = data
            idx += 1

    def backward(self, top, propagate_down, bottom):
        pass