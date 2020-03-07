"""http://arxiv.org/abs/1502.03167"""
import operator
from functools import reduce

from ivory.common.context import np
from ivory.core.layer import Layer


class BatchNormalization(Layer):
    input_ndim = 0

    def init(self):
        shape = (reduce(operator.mul, self.shape, 1),)
        self.gamma = self.add_weight(shape).ones()
        self.beta = self.add_weight(shape).zeros()
        self.running_mean = self.add_state(shape).zeros()
        self.running_var = self.add_state(shape).zeros()
        self.train = self.add_state(True)

    def forward(self):
        shape = self.x.d.shape
        x_2d = self.x.d.reshape(self.x.d.shape[0], -1)
        y_2d = self.forward_2d(x_2d)
        self.y.d = y_2d.reshape(shape)

    def forward_2d(self, x_2d):
        if self.train.d:
            mu = x_2d.mean(axis=0)
            self.xc = x_2d - mu
            var = np.mean(self.xc ** 2, axis=0)
            self.std = np.sqrt(var + 1e-7)
            self.xn = self.xc / self.std

            momentum = 0.9
            self.running_mean.d = momentum * self.running_mean.d + (1 - momentum) * mu
            self.running_var.d = momentum * self.running_var.d + (1 - momentum) * var
        else:
            self.xc = x_2d - self.running_mean.d
            self.xn = self.xc / np.sqrt(self.running_var.d + 1e-7)

        return self.gamma.d * self.xn + self.beta.d

    def backward(self):
        shape = self.y.g.shape
        dy_2d = self.y.g.reshape(self.y.g.shape[0], -1)
        dx_2d = self.backward_2d(dy_2d)
        self.x.g = dx_2d.reshape(shape)

    def backward_2d(self, dy_2d):
        self.beta.g = dy_2d.sum(axis=0)
        self.gamma.g = np.sum(self.xn * dy_2d, axis=0)
        dxn = self.gamma.d * dy_2d
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std

        batch_size = dy_2d.shape[0]
        dxc += (2.0 / batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        return dxc - dmu / batch_size
