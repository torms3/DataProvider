#!/usr/bin/env python
__doc__ = """

ConfigData classes.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import numpy as np
import emio
from tensor import TensorData
from utils import *

class ConfigData(TensorData):
    """
    ConfigData class.
    """

    def __init__(self, config, section):
        """Build data from config."""
        # Preprocessing
        data, fov, offset = self._prepare_data(config, section)

        # Initialize TensorData
        super(ConfigData, self).__init__(data, fov=fov, offset=offset)

        # Transformation
        self._transformation(config, section)

    ####################################################################
    ## Private Helper Methods
    ####################################################################

    def _prepare_data(self, config, section):
        """
        TODO(kisuk): Documentation.
        """
        assert config.has_section(section)

        # Either read data from specified file, or generate data with
        # specified shape and filler.
        if config.has_option(section, 'file'):
            data = emio.imread(config.get(section, 'file'))
        elif config.has_option(section, 'shape'):
            shape = config.get(section, 'shape')
            # Ensure that shape is tuple.
            shape = tuple(eval(str(shape)))
            if config.has_option(section, 'filler'):
                filler = eval(config.get(section, 'filler'))
            else:
                filler = {'type':'zero'}
            data = fill_data(shape, filler=filler)
        else:
            raise RuntimeError('invalid data section [%s]' % section)

        # FoV
        if config.has_option(section, 'fov'):
            fov = config.get(section, 'fov')
            # Ensure that fov is tuple.
            fov = tuple(eval(str(fov)))
        else:
            fov = (0,0,0)

        # Offset
        if config.has_option(section, 'offset'):
            offset = config.get(section, 'offset')
            # Ensure that offset is tuple.
            offset = tuple(eval(str(offset)))
        else:
            offset = (0,0,0)

        # List of global preprocessing.
        if config.has_option(section, 'preprocess'):
            preprocess = config.get(section, 'preprocess').split('\n')
            preprocess = [eval(x) for x in preprocess]
        else:
            preprocess = list()

        # Check the validity of each preprocessing.
        for pp in preprocess:
            assert isinstance(pp, dict)
            assert 'type' in pp

        # Perform preprocessing.
        data = check_tensor(data)
        for pp in preprocess:
            func = pp['type']
            del pp['type']
            data = transform_tensor(data, func, **pp):

        return data, fov, offset

    def _transformation(self):
        """
        TODO(kisuk): Documentation.
        """
        # List of local transformation.
        if config.has_option(section, 'transform'):
            transform = config.get(section, 'transform').split('\n')
            transform = [eval(x) for x in transform]
        else:
            transform = list()

        # Check the validity of each transformation.
        for t in transform:
            assert isinstance(t, dict)
            assert 'type' in t

        self.transform = transform