#!/usr/bin/env python
__doc__ = """

Volumetric label classes.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import numpy as np
from tensor import *

class VolumeLabel(TensorData):
    """
    Volumetric label class.
    """

    def __init__(self):
        pass

    def get_patch(self, pos):
        pass


class BoundaryLabel(VolumeLabel):
    """
    Volumetric label class.
    """

    def __init__(self):
        pass

    def get_patch(self, pos):
        pass


class AffinityLabel(VolumeLabel):
    """
    Volumetric label class.
    """

    def __init__(self):
        pass

    def get_patch(self, pos):
        pass


class SemanticLabel(VolumeLabel):
    """
    Volumetric label class.
    """

    def __init__(self):
        pass

    def get_patch(self, pos):
        pass