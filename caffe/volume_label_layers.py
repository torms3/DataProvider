import caffe
import numpy as np
import python.transform as transform

class AffinityLayer(caffe.Layer):
    """
    Transform segmentation to 3D affinity graph.
    """

    def setup(self, bottom, top):
        # Check inputs.
        if not (len(bottom)==1 or len(bottom)==2):
            raise Exception("Need one (label) or two inputs (label, mask) .")
        # Mask?
        if len(bottom) == 2:
            self.has_mask = True

    def reshape(self, bottom, top):
        # Affinity
        shape = bottom[0].shape
        shape = tuple(x-1 for x in shape)  # Crop shape by 1
        top[0].reshape(1, 3, shape)
        # Mask, if any
        if self.has_mask:
            top[1].reshape(top[0].data.shape)

    def forward(self, bottom, top):
        # Affinitize label.
        aff = transform.affinitize(bottom[0].data[-3:])
        aff = transform.transform_tensor(aff, 'crop', offset=(1,1,1))
        top[0].data[...] = aff
        # Affinitize mask, if any.
        if self.has_mask = True:
            msk = transform.affinitize_mask(bottom[1].data[-3:])
            msk = transform.transform_tensor(msk, 'crop', offset=(1,1,1))
            top[1].data[...] = msk
        # TODO(kisuk): Rebalancing.

    def backward(self, top, propagate_down, bottom):
        pass

# class MulticlassLayer(caffe.Layer):
#     """
#     """

#     def setup(self, bottom, top):
#         # Check inputs.
#         if not (len(bottom)==1 or len(bottom)==2):
#             raise Exception("Need one (label) or two inputs (label, mask) .")
#         # Number of classes
#         layer_params = eval(self.param_str)
#         self.N = layer_params['num_class']

#     def reshape(self, bottom, top):
#         top[0].reshape(1, N, bottom[0].shape)
#         # Mask, if any
#         if len(bottom) == 2:
#             top[1].reshape(top[0].shape)

#     def forward(self, bottom, top):

#     def backward(self, top, propagate_down, bottom):
#         pass