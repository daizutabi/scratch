from ivory.common.context import np
from ivory.core.layer import Layer


class Activation(Layer):
    input_ndim = 0


class Sigmoid(Activation):
    def forward(self):
        self.y.d = 1 / (1 + np.exp(-self.x.d))

    def backward(self):
        self.x.g = (1 - self.y.d) * self.y.d * self.y.g


class Relu(Activation):
    def forward(self):
        self.mask = self.x.d <= 0
        y = self.x.d.copy()
        y[self.mask] = 0
        self.y.d = y

    def backward(self):
        dx = self.y.g.copy()
        dx[self.mask] = 0
        self.x.g = dx
