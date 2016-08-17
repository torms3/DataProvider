#!/usr/bin/env python
__doc__ = """

Inference outputs.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import numpy as np

import box
from tensor import WritableTensorData as WTD, WritableTensorDataWithMask as WTDM

def prepare_outputs(spec, locs, blend=False, blend_mode=''):
    blend_pool = ['','bump']
    b = blend_mode.lower()
    if b not in blend_pool:
        raise RuntimeError('unknown output blend type [%s]' % b)
    if b == '':
        b = 'Blend'
    else:
        b = b[0].capitalize() + b[1:] + 'Blend'
    outputs = eval(b + '(spec, locs, blend)')
    return outputs


class Blend(object):
    """
    Blend interface.
    """

    def __init__(self, spec, locs, blend=False):
        """Initialize Blend."""
        self.spec  = spec
        self.locs  = locs
        self.blend = blend
        self._prepare_data()

    def push(self, loc, sample):
        """Write to data."""
        for k, v in sample.iteritems():
            assert k in self.data
            self.data[k].set_patch(loc, v, op=self.op)

    def get_data(self, key):
        """Get inference output data."""
        assert key in self.data
        return self.data[key].get_data()

    ####################################################################
    ## Private Methods.
    ####################################################################

    def _prepare_data(self):
        """
        TODO(kisuk): Documentation.
        """
        assert len(self.locs) > 0
        rmin = self.locs[0]
        rmax = self.locs[-1]

        self.data = dict()
        self.op   = None
        for k, v in self.spec.iteritems():
            fov = v[-3:]
            a = box.centered_box(rmin, fov)
            b = box.centered_box(rmax, fov)
            c = a.merge(b)
            shape = v[:-3] + tuple(c.size())
            # Inference with overlapping window.
            if self.blend:
                self.data[k] = WTDM(shape, fov, c.min())
                self.op = 'np.add'
            else:
                self.data[k] = WTD(shape, fov, c.min())


class BumpBlend(Blend):
    """
    Blending with bump function.
    """

    def __init__(self, spec, locs, blend=False):
        """Initialize BumpBlend."""
        Blend.__init__(self, spec, locs, blend)

        # Inference with overlapping window.
        self.max_logits = None
        if blend:
            max_logits = dict()
            # Compute max_logit for numerical stability.
            for k, v in self.data.iteritems():
                fov = tuple(v.fov())
                data = np.zeros(v.dim())
                data.fill(-np.inf)
                max_logit = WTD(data, fov, v.offset())
                max_logit_window = self._bump_logit_map(fov)
                for loc in self.locs:
                    max_logit.set_patch(loc, max_logit_window, op='np.maximum')
                max_logits[k] = max_logit
            self.max_logits = max_logits

    def push(self, loc, sample):
        """Blend with data."""
        for k, v in sample.iteritems():
            assert k in self.data
            mask = self._get_mask(k, loc)
            self.data[k].set_patch(loc, v, op=self.op, mask=mask)

    ####################################################################
    ## Private methods.
    ####################################################################

    def _get_mask(self, key, loc):
        mask = None
        if self.blend:
            assert key in self.max_logits
            max_logit = self.max_logits[key].get_patch(loc)
            mask = self._bump_map(max_logit.shape[-3:], max_logit[0,...])
        return mask

    def _bump_logit(self, z, y, x, t=1.5):
        return -(x*(1-x))**(-t)-(y*(1-y))**(-t)-(z*(1-z))**(-t)

    def _bump_logit_map(self, dim):
        x = range(dim[-1])
        y = range(dim[-2])
        z = range(dim[-3])
        zv, yv, xv = np.meshgrid(z, y, x, indexing='ij')
        xv = (xv+1.0)/(dim[-1]+1.0)
        yv = (yv+1.0)/(dim[-2]+1.0)
        zv = (zv+1.0)/(dim[-3]+1.0)
        return self._bump_logit(zv, yv, xv)

    def _bump_map(self, dim, max_logit):
        return np.exp(self._bump_logit_map(dim) - max_logit)
