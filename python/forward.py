#!/usr/bin/env python
__doc__ = """

ForwardScanner.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import numpy as np

from box import Box, centered_box
from tensor import WritableTensorData
from vector import *

class ForwardScanner(object):
    """
    ForwardScanner.
    """

    def __init__(self, dataset, scan_spec, params=None):
        """
        Initialize ForwardScanner.
        """
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
            print '({}/{}) loc: {}'.format(idx, len(self.locs), loc)
            ret, _ = self.dataset.get_sample(loc)
            self.current = loc
            self.counter += 1
        return ret

    def push(self, sample):
        """
        TODO(kisuk): Documentation
        """
        assert self.current is not None
        # TODO(kisuk): Write to outputs.
        self.current = None

    ####################################################################
    ## Private Methods
    ####################################################################

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

        locs = list()
        for z in self.coords[0]:
            for y in self.coords[1]:
                for x in self.coords[2]:
                    locs.append(Vec3d(z,y,x))
        self.locs = locs
        self.counter = 0

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
        stride = int(self.stride[dim])
        grid   = int(self.grid[dim])
        coord  = set()

        # Non-overlapping stride.
        if stride == 0:
            stride = int(self.default_stride[dim])

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
        """
        TODO(kisuk): Documentation.
        """
        self.outputs = dict()

        rmin = self.locs[0]
        rmax = self.locs[-1]

        for k, v in self.scan_spec.iteritems():
            a = centered_box(rmin, v[-3:])
            b = centered_box(rmax, v[-3:])
            self.outputs[k] = WritableTensorData(v, fov=a+b)


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
