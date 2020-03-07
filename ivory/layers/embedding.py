from ivory.common.context import np
from ivory.core.layer import Layer


class Embedding(Layer):
    input_ndim = -2

    def init(self):
        self.y.shape = self.shape[-1:]
        self.W = self.add_weight(self.shape).randn()

    def forward(self):
        self.y.d = self.W.d[self.x.d]

    def backward(self):
        grad = np.zeros_like(self.W.d)
        np.scatter_add(grad, self.x.d, self.y.g)  # np.add.at
        self.W.g = grad


class EmbeddingMean(Layer):
    input_ndim = -2

    def init(self):
        self.y.shape = self.shape[-1:]
        self.W = self.add_weight(self.shape[1:]).randn()

    def forward(self):
        self.y.d = self.W.d[self.x.d].sum(axis=1) / self.shape[0]

    def backward(self):
        grad = np.zeros_like(self.W.d)
        np.scatter_add(grad, self.x.d.T, self.y.g)  # np.add.at
        grad /= self.shape[0]
        self.W.g = grad


class EmbeddingDot(Layer):
    def init(self):
        self.W = self.add_weight(self.shape[::-1]).randn()
        self.t = self.add_input()
        self.y.shape = ()

    def forward(self):
        self.t_W = self.W.d[self.t.d]
        self.y.d = np.sum(self.x.d * self.t_W, axis=1)

    def backward(self):
        dy = self.y.g.reshape(self.y.g.shape[0], 1)
        grad = np.zeros_like(self.W.d)
        np.scatter_add(grad, self.t.d, dy * self.x.d)  # np.add.at
        self.W.g = grad
        self.x.g = dy * self.t_W
