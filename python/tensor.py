#!/usr/bin/env python
__doc__ = """

Read-only/writable TensorData classes.

Kisuk Lee <kisuklee@mit.edu>, 2015-2016
"""

import math
import numpy as np

from box import *
from vector import *

class TensorData(object):
    """Read-only tensor data.

    The 1st dimension is regarded as parallel channels, and arbitrary access
    along this dimension is not allowed. Threfore, every data access should be
    made through 3D vector, not 4D.

    Attributes:
        _data:   numpy 4D array (channel,z,y,x)
        _dim:    Dimension of each channel
        _offset: Coordinate offset from the origin
        _bb:     Bounding box
        _fov:    Patch size
        _rg:     Range (update dep.: dim, offset, fov)
    """

    def __init__(self, data, fov=(0,0,0), offset=(0,0,0)):
        """
        Initialize a TensorData object.
        """
        # Set immutable attributes.
        self._data   = self._check_data(data)
        self._dim    = Vec3d(self._data.shape[1:])
        self._offset = Vec3d(offset)

        # Set bounding box.
        self._bb = Box((0,0,0), self._dim)
        self._bb.translate(self._offset)

        # Set fov (patch size).
        self.set_fov(fov)

    def set_fov(self, fov):
        """
        Set a nonnegative field of view (FoV), i.e., patch size.
        """
        # Zero FoV indicates covering the whole volume.
        fov = Vec3d(fov)
        if fov == (0,0,0):
            fov = Vec3d(self._dim)

        # FoV should be nonnegative, and smaller than data dimension.
        assert fov==minimum(maximum(fov,(0,0,0)), self._dim)

        # Set FoV.
        self._fov = fov

        # Update range.
        self._set_range()

    def get_patch(self, pos):
        """
        Extract a patch of size _fov centered on pos.
        """
        # Check validity.
        assert self._rg.contains(pos)
        # Local coordinate system
        loc  = pos - self._offset
        box  = centered_box(loc, self._fov)
        vmin = box.min()
        vmax = box.max()
        return np.copy(self._data[:,vmin[0]:vmax[0],
                                    vmin[1]:vmax[1],
                                    vmin[2]:vmax[2]])

    ####################################################################
    ## Public methods for accessing attributes
    ####################################################################

    def get_data(self):
        return self._data

    def shape(self):
        """Return data shape (c,z,y,x)."""
        return self._data.shape

    def dim(self):
        """Return channel shape (z,y,x)."""
        return Vec3d(self._dim)

    def fov(self):
        return Vec3d(self._fov)

    def offset(self):
        return Vec3d(self._offset)

    def bounding_box(self):
        return Box(self._bb)

    def range(self):
        return Box(self._rg)

    ####################################################################
    ## Private helper methods.
    ####################################################################

    def _check_data(self, data):
        # Data should be either numpy 3D or 4D array.
        assert isinstance(data, np.ndarray)
        assert data.ndim==3 or data.ndim==4

        # Add channel dimension if data is 3D array.
        if data.ndim == 3:
            data = data[np.newaxis,...]

        return data

    def _set_range(self):
        """Set a valid range for extracting patches."""
        top  = self._fov/2                # Top margin
        btm  = self._fov - top - (1,1,1)  # Bottom margin
        vmin = self._offset + top
        vmax = self._offset + self._dim - btm

        self._rg = Box(vmin, vmax)

    # String representaion (for printing and debugging).
    def __str__( self ):
        return "<TensorData>\nshape: %s\ndim: %s\nFoV: %s\noffset: %s\n" % \
               (self.shape(), self._dim, self._fov, self._offset)


class WritableTensorData(TensorData):
    """
    Writable tensor data.
    """

    def __init__(self, data_or_shape, fov=(0,0,0), offset=(0,0,0)):
        """
        Initialize a writable tensor data, or create a new tensor of zeros.
        """
        if isinstance(data_or_shape, np.ndarray):
            TensorData.__init__(self, data_or_shape, fov, offset)
        else:
            data = np.zeros(data_or_shape, dtype='float32')
            TensorData.__init__(self, data, fov, offset)

    def set_patch(self, pos, patch, op=None):
        """
        Write a patch of size _fov centered on pos.
        """
        assert self._rg.contains(pos)
        patch = self._check_data(patch)
        dim = patch.shape[1:]
        assert dim==self._fov
        box = centered_box(pos, dim)

        # Local coordinate
        box.translate(-self._offset)
        vmin = box.min()
        vmax = box.max()
        lval = 'self._data[:,vmin[0]:vmax[0],vmin[1]:vmax[1],vmin[2]:vmax[2]]'
        rval = 'patch'
        if op is None:
            exec '{}={}'.format(lval, rval)
        else:
            exec '{}={}({},{})'.format(lval, op, lval, rval)


class WritableTensorDataWithMask(WritableTensorData):
    """
    Writable tensor data with blending mask.
    """

    def __init__(self, data_or_shape, fov=(0,0,0), offset=(0,0,0)):
        """
        Initialize a writable tensor data, or create a new tensor of zeros.
        """
        WritableTensorData.__init__(self, data_or_shape, fov, offset)

        # Set norm.
        self._norm = WritableTensorData(self.shape(), fov, offset)

    def set_patch(self, pos, patch, op='np.add', mask=None):
        """
        Write a patch of size _fov centered on pos.
        """
        # Default mask.
        if mask is None:
            mask = np.ones(patch.shape, dtype='float32')
        # Set patch.
        WritableTensorData.set_patch(self, pos, mask*patch, op)
        # Set normalization.
        self._norm.set_patch(pos, mask, op='np.add')

    def get_norm(self):
        return self._norm._data

    def get_data(self):
        # return self._data/self._norm._data
        # Temporary in-place normalization.
        self._data = np.divide(self._data, self._norm._data, self._data)
        return self._data

    def get_unnormalized_data(self):
        return WritableTensorData.get_data(self)


########################################################################
## Unit Testing
########################################################################
if __name__ == "__main__":

    import unittest

    ####################################################################
    class UnitTestTensorData(unittest.TestCase):

        def setup(self):
            pass

        def testCreation(self):
            data = np.zeros((4,4,4,4))
            T = TensorData(data, (3,3,3), (1,1,1))
            self.assertTrue(T.shape()==(4,4,4,4))
            self.assertTrue(T.offset()==(1,1,1))
            self.assertTrue(T.fov()==(3,3,3))
            bb = T.bounding_box()
            rg = T.range()
            self.assertTrue(bb==Box((1,1,1),(5,5,5)))
            self.assertTrue(rg==Box((2,2,2),(4,4,4)))

        def testGetPatch(self):
            # (4,4,4) random 3D araray
            data = np.random.rand(4,4,4)
            T = TensorData(data, (3,3,3))
            p = T.get_patch((2,2,2))
            self.assertTrue(np.array_equal(data[1:,1:,1:], p[0, ...]))
            T.set_fov((2,2,2))
            p = T.get_patch((2,2,2))
            self.assertTrue(np.array_equal(data[1:3,1:3,1:3], p[0, ...]))


    ####################################################################
    class UnitTestWritableTensorData(unittest.TestCase):

        def setup(self):
            pass

        def testCreation(self):
            data = np.zeros((1,4,4,4))
            T = WritableTensorData(data, (3,3,3), (1,1,1))
            self.assertTrue(T.shape()==(1,4,4,4))
            self.assertTrue(T.offset()==(1,1,1))
            self.assertTrue(T.fov()==(3,3,3))
            bb = T.bounding_box()
            rg = T.range()
            self.assertTrue(bb==Box((1,1,1),(5,5,5)))
            self.assertTrue(rg==Box((2,2,2),(4,4,4)))

        def testSetPatch(self):
            T = WritableTensorData(np.zeros((1,5,5,5)), (3,3,3), (1,1,1))
            p = np.random.rand(1,3,3,3)
            self.assertFalse(np.array_equal(p, T.get_patch((4,4,4))))
            T.set_patch((4,4,4), p)
            self.assertTrue(np.array_equal(p, T.get_patch((4,4,4))))

    ####################################################################
    unittest.main()

    ####################################################################
