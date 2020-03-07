from ivory.common.context import np
from ivory.common.util import col2im, im2col
from ivory.core.layer import Layer


class Convolution(Layer):
    """Convolution((C, H, W, FN, FH, FW)).

    * Input: (N, C, H, W)
    * Filter : (FN, C, FH, FW)
    * Output:  (N, FN, OH, OW)
    """

    input_ndim = 3

    def init(self, stride=1, padding=0):
        C, W, H, FN, FH, FW = self.shape
        self.W = self.add_weight((FN, C, FH, FW)).randn()
        self.b = self.add_weight((FN,)).zeros()
        self.stride = self.add_state(stride)
        self.padding = self.add_state(padding)
        OH = 1 + int((H - FH + 2 * padding) / stride)
        OW = 1 + int((W - FW + 2 * padding) / stride)
        self.y.shape = FN, OH, OW

    def forward(self):
        FN, C, FH, FW = self.W.shape
        FN, OH, OW = self.y.shape
        self.x_2d = im2col(self.x.d, FH, FW, self.stride.d, self.padding.d)
        self.W_2d = self.W.d.reshape(FN, -1).T
        y_2d = self.x_2d @ self.W_2d + self.b.d
        N = self.x.d.shape[0]
        self.y.d = y_2d.reshape(N, OH, OW, -1).transpose(0, 3, 1, 2)

    def backward(self):
        FN, C, FH, FW = self.W.shape
        dy_2d = self.y.g.transpose(0, 2, 3, 1).reshape(-1, FN)
        self.b.g = np.sum(dy_2d, axis=0)
        dW_2d = self.x_2d.T @ dy_2d
        self.W.g = dW_2d.transpose(1, 0).reshape(FN, C, FH, FW)
        dx_2d = dy_2d @ self.W_2d.T
        self.x.g = col2im(dx_2d, self.x.d.shape, FH, FW, self.stride.d, self.padding.d)


class Pooling(Layer):
    """Pooling((C, H, W, PH, PW)).

    * Input: (N, C, H, W)
    * Output:  (N, C, OH, OW)
    """

    input_ndim = 3

    def init(self, stride=0, padding=0):
        C, H, W, PH, PW = self.shape
        stride = stride or PH
        self.stride = self.add_state(stride)
        self.padding = self.add_state(padding)
        OH = 1 + int((H - PH + 2 * padding) / stride)
        OW = 1 + int((W - PW + 2 * padding) / stride)
        self.y.shape = C, OH, OW

    def forward(self):
        PH, PW = self.shape[3:5]
        C, OH, OW = self.y.shape
        x_2d = im2col(self.x.d, PH, PW, self.stride.d, self.padding.d)
        pool_size = self.shape[3] * self.shape[4]
        x_2d = x_2d.reshape(-1, pool_size)
        self.arg_max = np.argmax(x_2d, axis=1)
        N = self.x.d.shape[0]
        self.y.d = np.max(x_2d, axis=1).reshape(N, OH, OW, C).transpose(0, 3, 1, 2)

    def backward(self):
        PH, PW = self.shape[3:5]
        dy = self.y.g.transpose(0, 2, 3, 1)
        pool_size = self.shape[3] * self.shape[4]
        dmax = np.zeros((dy.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dy.flatten()
        dmax = dmax.reshape(dy.shape + (pool_size,))
        dx_2d = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        self.x.g = col2im(dx_2d, self.x.d.shape, PH, PW, self.stride.d, self.padding.d)
