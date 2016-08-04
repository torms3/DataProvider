import caffe
import numpy as np

class SigmoidCrossEntropyLossLayer(caffe.Layer):
    """
    Binomial cross-entropy loss layer with mask.
    """

    def setup(self, bottom, top):
        # Check inputs.
        if len(bottom) != 3:
            raise Exception("Need three inputs (propagated, label, mask).")
        # Threshold for computing classification error
        self.thresh = 0.5

    def reshape(self, bottom, top):
        # Check input dimensions match.
        if (bottom[0].count != bottom[1].count or
            bottom[0].count != bottom[2].count):
            raise Exception("Inputs must have the same dimension.")
        # Difference is shape of inputs.
        self.diff = np.zeros_like(bottom[0].data)
        self.cost = np.zeros_like(bottom[0].data)
        self.cerr = np.zeros_like(bottom[0].data)
        # Loss outputs are scalar.
        top[0].reshape(1)  # Rebalanced loss
        top[1].reshape(1)  # Unbalanced loss
        top[2].reshape(1)  # Classification error

    def forward(self, bottom, top):
        prob  = self.sigmoid(bottom[0].data)
        label = bottom[1].data
        mask  = bottom[2].data
        # Gradient
        self.diff[...] = mask*(prob - label)
        # Cross entropy
        self.cost[...] = self.cross_entropy(prob, label)
        # Classification error
        self.cerr[...] = (mask>0)*((prob>self.thresh) != (label>self.thresh))
        # Rebalanced cost
        top[0].data[...] = np.sum(mask*self.cost)
        # Unbalanced cost
        top[1].data[...] = np.sum(self.cost)
        # Classification error
        top[2].data[...] = np.sum(self.cerr)

    def backward(self, top, propagate_down, bottom):
        if propagate_down[1] or propagate_down[2]:
            raise Exception("Cannot backpropagate to label or mask inputs.")
        if propagate_down[0]:
            bottom[0].diff[...] = self.diff

    def sigmoid(self, x):
        """Numerically-stable sigmoid function."""
        ret = np.zeros_like(x)
        idx = x >= 0
        z = np.exp(-x[idx])
        ret[idx] = 1 / (1 + z)
        idx = x < 0
        z = np.exp(x[idx])
        ret[idx] = z / (1 + z)
        return ret

    def cross_entropy(self, y, t):
        """Numerically-stable binomial cross-entropy.

        Args:
            y: Prediction
            t: Target (ground truth)
        """
        return -x*(y - (x>=0)) + np.log(1 + np.exp(x - 2*x*(x>=0)))

