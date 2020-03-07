"""http://arxiv.org/abs/1207.0580"""

from ivory.common.context import np
from ivory.core.layer import Layer

# class Dropout(Layer):
#     input_ndim = 0
#
#     def init(self):
#         self.dropout_ratio = self.add_state(0.5)
#         self.train = self.add_state(True)
#
#     def forward(self):
#         if self.train.d:
#             self.mask = np.random.rand(*self.x.d.shape) > self.dropout_ratio.d
#             self.y.d = self.x.d * self.mask
#         else:
#             self.y.d = self.x.d * (1 - self.dropout_ratio.d)
#
#     def backward(self):
#         self.x.g = self.y.g * self.mask


class Dropout(Layer):
    input_ndim = 0

    def init(self):
        self.dropout_ratio = self.add_state(0.5)
        self.train = self.add_state(True)

    def forward(self):
        if self.train.d:
            flag = np.random.rand(*self.x.d.shape) > self.dropout_ratio.d
            scale = 1 / (1.0 - self.dropout_ratio.d)
            self.mask = flag.astype(self.dtype) * scale
            self.y.d = self.x.d * self.mask
        else:
            self.y.d = self.x.d

    def backward(self):
        self.x.g = self.y.g * self.mask
