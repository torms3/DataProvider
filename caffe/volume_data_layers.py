#!/usr/bin/env python
__doc__ = """

Caffe Layer for Volumetric Data.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import caffe
from collections import OrderedDict
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
        layer_params = eval(self.param_str)

        # Data & net specs.
        dspec_path = layer_params['dspec_path']
        net_spec   = layer_params['net_spec']

        # Parameters for constructing data provider.
        params = dict()
        params['border']  = layer_params.get('border', None)
        params['augment'] = layer_params.get('augment', [])
        params['drange']  = layer_params['drange']
        params['dprior']  = layer_params.get('dprior', None)

        # Construct a VolumeDataProvider object.
        auto_mask = layer_params.get('auto_mask', True)
        self.data_provider = VolumeDataProvider(dspec_path, net_spec, params,
            auto_mask=auto_mask)

        # Infer ntop from net spec.
        ntop = len(net_spec)
        if len(top) != ntop:
            raise Exception('Need to define %d tops.' % ntop)
        # Data layers have no bottoms.
        if len(bottom) != 0:
            raise Exception('Do not define a bottom.')

    def reshape(self, bottom, top):
        # Fetch random sample.
        self.sample = self.data_provider.random_sample()
        # Reshape tops to fit (leading 1 is for batch dimension).
        for idx, key in enumerate(self.sample):
            top[idx].reshape(1, *self.sample[key].shape)

    def forward(self, bottom, top):
        # Assign output.
        for idx, key in enumerate(self.sample):
            top[idx].data[...] = self.sample[key]

    def backward(self, top, propagate_down, bottom):
        pass
