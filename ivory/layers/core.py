import operator
from functools import reduce

from ivory.common.context import np
from ivory.core.layer import Layer


class MatMul(Layer):
    def init(self):
        self.W = self.add_weight(self.shape).randn()

    def forward(self):
        self.y.d = self.x.d @ self.W.d

    def backward(self):
        self.x.g = self.y.g @ self.W.d.T
        axis = (0,) if self.x.g.ndim == 2 else (0, 1)
        self.W.g = np.tensordot(self.x.d, self.y.g, axes=[axis, axis])


class MatMulMean(Layer):
    def init(self):
        self.W = self.add_weight(self.shape[1:]).randn()

    def forward(self):
        self.y.d = np.sum(self.x.d @ self.W.d, axis=1) / self.shape[0]

    def backward(self):
        dx = (self.y.g @ self.W.d.T) / self.shape[0]
        self.x.g = np.repeat(dx, self.shape[0], axis=0).reshape(*self.x.d.shape)
        self.W.g = np.sum(self.x.d.T @ self.y.g, axis=1) / self.shape[0]


class Flatten(Layer):
    def init(self):
        self.x.shape = self.shape
        self.shape = self.shape + (reduce(operator.mul, self.shape, 1),)
        self.y.shape = self.shape[-1:]

    def forward(self):
        self.y.d = self.x.d.reshape(self.x.d.shape[0], -1)

    def backward(self):
        self.x.g = self.y.g.reshape(*self.x.d.shape)
