#!/usr/bin/env python
__doc__ = """

Warping (ELEKTRONN)

"""

import numpy as np
from warping import getWarpParams, warp2dJoint, warp3dJoint

def _warpAugment(img, lbl, ps, fov):
    ext_size, rot, shear, scale, stretch, twist = getWarpParams(ps)
    try:
        D, L = _cutPatch(img, lbl, ps=ext_size, fov=fov)
        d, l = warp3dJoint(D, L, ps, rot, shear, scale, stretch, twist)  # ignores label if non-image
    except ValueError:  # the ext_size is to big for this data cube
        assert False

    return D, L, d, l

def _cutPatch(img, lab, ps, fov):
    try:
        shift = [int(np.random.randint(0, s - p, 1)) for p, s in zip(ps, np.array(img.shape)[[0, 2, 3]])]
    except ValueError:
        if np.all(np.equal(ps, np.array(img.shape)[[0, 2, 3]])):
            shift = [0, 0, 0]
        else:
            raise ValueError("Image smaller than patch size: Image shape=%s, patch size=%s"
                             % (img.shape[1:], ps))

    cut_img = img[shift[0]:shift[0] + ps[0], :, shift[1]:shift[1] + ps[1], shift[2]:shift[2] + ps[2]]

    off = np.array(fov)/2
    cut_lab = lab[off[0] + shift[0]:shift[0] + ps[0] - off[0],
                  off[1] + shift[1]:shift[1] + ps[1] - off[1],
                  off[2] + shift[2]:shift[2] + ps[2] - off[2]]

    return cut_img, cut_lab
