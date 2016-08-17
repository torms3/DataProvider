#!/usr/bin/env python
__doc__ = """

ForwardScanner.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import numpy as np
import math

import blend
from box import Box, centered_box
from tensor import WritableTensorData as WTD, WritableTensorDataWithMask as WTDM
from vector import *

class ForwardScanner(object):
    """
    ForwardScanner.

    Attributes:
        dataset:
        scan_spec:
        params:
    """

    def __init__(self, dataset, scan_spec, params=None):
        """
        Initialize ForwardScanner.
        """
        self._init()

        self.dataset   = dataset
        self.scan_spec = scan_spec
        self.params    = params if params is not None else dict()

        self._setup()

    def pull(self):
        """
        TODO(kisuk): Documentation.
        """
        ret = None
        if self.counter < len(self.locs):
            assert self.current is None
            idx = self.counter
            loc = self.locs[idx]
            print '({}/{}) loc: {}'.format(idx+1, len(self.locs), tuple(loc))
            ret, _ = self.dataset.get_sample(loc)
            self.current = loc
            self.counter += 1
        return ret

    def push(self, sample):
        """
        TODO(kisuk): Documentation

        Args:
            sample:
            kwargs:
        """
        assert self.current is not None
        self.outputs.push(self.current, sample)
        self.current = None

    ####################################################################
    ## Private Methods.
    ####################################################################

    def _init(self):
        """Initialize all attributes."""
        self.dataset        = None
        self.scan_spec      = dict()
        self.params         = dict()
        self.offset         = (0,0,0)
        self.stride         = (0,0,0)
        self.grid           = (0,0,0)
        self.vmin           = None
        self.vmax           = None
        self.default_stride = None
        self.coords         = [None]*3
        self.locs           = None
        self.counter        = 0
        self.current        = None
        self.outputs        = None

    def _setup(self):
        """
        TODO(kisuk): Documentation.
        """
        self.offset = Vec3d(self.params.get('offset', (0,0,0)))
        self.stride = Vec3d(self.params.get('stride', (0,0,0)))
        self.grid   = Vec3d(self.params.get('grid',   (0,0,0)))

        # TODO(kisuk): Validity check?

        self.vmin = self.dataset.get_range().min() + self.offset
        self.vmax = self.dataset.get_range().max()

        # TODO(kisuk): Validity check?

        # Order is important!
        self._setup_stride()
        self._setup_coords()
        self._prepare_outputs()

    def _setup_stride(self):
        """
        TODO(kisuk): Documentation.
        """
        stride = None
        for k, v in self.scan_spec.iteritems():
            box = centered_box(Vec3d(0,0,0), v[-3:])
            if stride is None:
                stride = box
            else:
                stride = stride.intersect(box)
        self.default_stride = stride.size()

    def _setup_coords(self):
        """
        TODO(kisuk): Documentation.
        """
        self._setup_coord(0)  # z-dimension
        self._setup_coord(1)  # y-dimension
        self._setup_coord(2)  # x-dimension

        self.locs = list()
        for z in self.coords[0]:
            for y in self.coords[1]:
                for x in self.coords[2]:
                    self.locs.append(Vec3d(z,y,x))

    def _setup_coord(self, dim):
        """
        TODO(kisuk): Documenatation.

        Args:
            dim: 0: z-dimension.
                 1: y-dimension.
                 2: x-dimension.
        """
        assert dim < 3

        # min & max coordinates.
        cmin = int(self.vmin[dim])
        cmax = int(self.vmax[dim])
        assert cmin < cmax

        # Dimension-specific params.
        stride = self.stride[dim]
        grid   = int(self.grid[dim])
        coord  = set()

        # Non-overlapping stride.
        if stride == 0:
            stride = self.default_stride[dim]
        # Overlapping stride given by an overlapping ratio.
        elif stride > 0 and stride < 1:
            stride = math.ceil(stride * self.default_stride[dim])
        self.stride[dim] = int(stride)
        stride = self.stride[dim]

        # Automatic full spanning.
        if grid == 0:
            grid = (cmax - cmin - 1)/stride + 1
            coord.add(cmax-1)  # Offcut

        # Scan coordinates.
        for i in range(grid):
            c = cmin + i*stride
            if c >= cmax:
                break
            coord.add(c)

        # Sanity check.
        assert cmin+(grid-1)*stride < cmax

        # Sort coords.
        self.coords[dim] = sorted(coord)

    def _prepare_outputs(self):
        """Prepare outputs according to the blending mode."""
        # Inference with overlapping window.
        diff = self.stride - self.default_stride
        overlap = True if diff[0]<0 or diff[1]<0 or diff[2]<0 else False
        # Prepare outputs.
        blend_mode = self.params.get('blend', '')
        self.outputs = blend.prepare_outputs(self.scan_spec, self.locs,
                                    blend=overlap, blend_mode=blend_mode)

if __name__ == "__main__":

    import data_provider

    # Data spec path
    dspec_path = 'test_spec/piriform.spec'

    # Net specification
    net_spec = {}
    net_spec['input'] = (18,208,208)

    # Parameters
    params = {}
    params['border']  = 'mirror'
    params['drange']  = [1]

    # VolumeDataProvider
    dp = VolumeDataProvider(dspec_path, net_spec, params)
