#!/usr/bin/env python
__doc__ = """

ConfigData classes.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import emio
from tensor import TensorData

class ConfigData(TensorData):
    """
    ConfigData.
    """

    def __init__(self, config, section, option='fnames'):
        """Build data from config."""

        # Read data from files
        fnames = config.get(section, option).split(',')
        arr_list = emio.read_from_files(fnames)
        data = np.concatenate(arr_list, axis=0)

        # Offset (optional)
        if config.has_option(section, 'offset'):
            offset = config.get(section, 'offset').split(',')
            offset = tuple(int(x) for x in offset)
        else:
            offset = (0,0,0)

        # Initialize TensorData
        super(ConfigData, self).__init__(data, offset=offset)

        # Global preprocessing

        # Local transformation


class ConfigImage(ConfigData):
    """
    ConfigImage.
    """

    def __init__(self, config, section):
        """Build image from config."""

        # Initialize ConfigData
        super(ConfigImage, self).__init__(config, section)

        # Global preprocessing

        # Local transformation


class ConfigLabel(ConfigData):
    """
    ConfigLabel.
    """

    def __init__(self, config, section):
        """Build label from config."""

        # Initialize ConfigData
        super(ConfigImage, self).__init__(config, section)

        # Global preprocessing

        # Local transformation