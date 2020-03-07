from ivory.common.context import np
from ivory.core.layer import LossLayer


class SigmoidCrossEntropy(LossLayer):
    def forward(self):
        self.y.d = 1 / (1 + np.exp(-self.x.d))
        y = self.y.d.reshape(-1)
        self.size = y.shape[0]
        loss = np.c_[1 - y, y][np.arange(self.size), self.t.d.reshape(-1)]
        self.loss.d = -np.sum(np.log(loss + 1e-7)) / self.size

    def backward(self):
        self.x.g = (self.y.d - self.t.d) / self.size

    def predict(self) -> float:
        return (self.x.d > 0).astype(int)

    @property
    def accuracy(self) -> float:
        return float(np.average(self.predict() == self.t.d))


class SoftmaxCrossEntropy(LossLayer):
    def forward(self):
        y = np.exp(self.x.d - self.x.d.max(axis=-1, keepdims=True))
        y /= y.sum(axis=-1, keepdims=True)
        self.y.d = y
        self.y_2d = self.y.d.reshape(-1, self.y.d.shape[-1])
        self.t_1d = self.t.d.reshape(-1)
        self.size = self.y_2d.shape[0]
        loss = self.y_2d[np.arange(self.size), self.t_1d]
        self.loss.d = -np.sum(np.log(loss + 1e-7)) / self.size

    def backward(self):
        self.y_2d[np.arange(self.size), self.t_1d] -= 1
        self.x.g = self.y_2d.reshape(*self.x.d.shape) / self.size

    def predict(self) -> float:
        return np.argmax(self.x.d, axis=-1)

    @property
    def accuracy(self) -> float:
        return float(np.average(self.predict() == self.t.d))
