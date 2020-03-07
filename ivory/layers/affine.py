from ivory.common.context import np
from ivory.core.layer import Layer


class Affine(Layer):
    def init(self):
        self.W = self.add_weight(self.shape).randn()
        self.b = self.add_weight(self.shape[-1:]).zeros()
        self.weight_decay = self.add_state(0)

    def forward(self):
        self.W_d = self.W.d.T if self.W.transpose else self.W.d
        self.y.d = self.x.d @ self.W_d + self.b.d

    def backward(self):
        self.x.g = self.y.g @ self.W_d.T
        axis = (0,) if self.x.g.ndim == 2 else (0, 1)
        dW = np.tensordot(self.x.d, self.y.g, axes=[axis, axis])
        if self.W.transpose:
            dW = dW.T
        self.W.g = dW + self.weight_decay.d * self.W.d
        self.b.g = self.y.g.sum(axis=axis)
