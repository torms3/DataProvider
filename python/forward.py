#!/usr/bin/env python
__doc__ = """

ForwardScanner.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import numpy as np

from tensor import WritableTensor
from vector import *

class ForwardScanner(object):
    """
    ForwardScanner.
    """

    def __init__(self, dataset, params, scan_spec):
        """
        Initialize ForwardScanner.
        """
        self.dataset   = dataset
        self.params    = params
        self.scan_spec = scan_spec

        self._setup()

    def pull(self):
        """
        TODO(kisuk): Documentation.
        """
        ret = None
        if self.counter < len(self.locs):
            loc = self.locs[self.counter]
            ret, _ = self.dataset.get_sample(loc)
            self.counter += 1
        return ret

    def push(self, sample):
        """
        TODO(kisuk): Documentation
        """
        # TODO(kisuk): Write to outputs.
        pass

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

        self._setup_coords()

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
            # TODO(kisuk)
            pass

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
        pass

    def _add_output(self):
        pass

    def _min_coord(self):
        pass

    def _max_coord(self):
        pass