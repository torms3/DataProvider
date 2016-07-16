#!/usr/bin/env python
__doc__ = """

ConfigData classes.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import numpy as np

import emio
from tensor import TensorData

class ConfigData(TensorData):
    """
    ConfigData.
    """

    def __init__(self, config, section):
        """Build data from config."""
        assert config.has_section(section)

        # Either read data from specified files, or generate data with
        # specified shape and filler.
        if config.has_option(section, 'files'):
            fnames = config.get(section, 'files').split('\n')
            arr_list = emio.read_from_files(fnames)
            data = np.concatenate(arr_list, axis=0)
        elif config.has_option(section, 'shape'):
            shape = tuple(eval(config.get(section, 'shape')))
            if config.has_option(section, 'filler'):
                filler = eval(config.get(section, 'filler'))
            else:
                filler = {'type':'zero'}
            data = fill_data(shape, filler=filler)
        else:
            raise RuntimeError('invalid data section [%s]' % section)

        # FoV (optional)
        if config.has_option(section, 'fov'):
            fov = config.get(section, 'fov')
            fov = tuple(eval(fov))
        else:
            fov = (0,0,0)

        # Offset (optional)
        if config.has_option(section, 'offset'):
            offset = config.get(section, 'offset')
            offset = tuple(eval(offset))
        else:
            offset = (0,0,0)

        # Initialize TensorData
        super(ConfigData, self).__init__(data, fov=fov, offset=offset)

        # Preprocessing
        self._preprocessing(config, section)

        # Transformation
        self._transformation(config, section)

    def _preprocessing(self, config, section):
        """
        TODO(kisuk): Documentation.
        """
        # A list of global preprocessing (data)
        if config.has_option(section, 'preprocess'):
            preprocess = config.get(section, 'preprocess').split('\n')
            preprocess = [eval(x) for x in preprocess]
        else:
            preprocess = list()

        # Check the validity of each preprocessing
        for pp in preprocess:
            assert isinstance(pp, dict)
            assert 'type' in pp

        # TODO(kisuk): Perform preprocessing.

    def _transformation(self):
        """
        TODO(kisuk): Documentation.
        """
        # A list of local transformation (sample)
        if config.has_option(section, 'transform'):
            transform = config.get(section, 'transform').split('\n')
            transform = [eval(x) for x in transform]
        else:
            transform = list()

        # Check the validity of each transformation
        for t in transform:
            assert isinstance(t, dict)
            assert 'type' in t

        self.transform = transform