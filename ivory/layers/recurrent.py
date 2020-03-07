from ivory.common.context import np
from ivory.core.layer import Layer


class RNN(Layer):
    def init(self):
        L, M = self.shape
        self.W = self.add_weight((L, M)).randn()
        self.U = self.add_weight((M, M)).randn()
        self.b = self.add_weight((M,)).zeros()
        self.h = self.add_state((M,))
        self.h_prev = None
        self.stateful = self.add_state(True)

    def forward(self):
        L, M = self.shape
        N, T, L = self.x.d.shape
        if self.h.d is None or not self.stateful.d:
            self.h_prev = np.zeros((N, M), dtype=self.dtype)
        else:
            self.h_prev = self.h.d
        x = self.x.d @ self.W.d
        y = np.empty((N, T, M), dtype=self.dtype)
        for t in range(T):
            h = self.h_prev if t == 0 else y[:, t - 1]
            y[:, t] = np.tanh(x[:, t] + h @ self.U.d + self.b.d)
        self.y.d = y
        self.h.d = y[:, -1]

    def backward(self):
        L, M = self.shape
        N, T, L = self.x.d.shape
        dx = np.empty((N, T, L), dtype=self.dtype)
        self.h.g = 0
        for t in reversed(range(T)):
            dy = self.y.g[:, t] + self.h.g
            dt = dy * (1 - self.y.d[:, t] ** 2)
            self.b.g = np.sum(dt, axis=0)
            self.W.g = self.x.d[:, t].T @ dt
            h = self.h_prev if t == 0 else self.y.d[:, t - 1]
            self.U.g = h.T @ dt
            self.h.g = dt @ self.U.d.T
            dx[:, t] = dt @ self.W.d.T
        self.x.g = dx

    def reset_state(self):
        self.h.d = None


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LSTM(Layer):
    def init(self):
        L, M = self.shape
        self.W = self.add_weight((L, 4 * M)).randn()
        self.U = self.add_weight((M, 4 * M)).randn()
        self.b = self.add_weight((4 * M,)).zeros()
        self.h = self.add_state((M,))  # (N, M)
        self.A = None  # (N, T, 4M)
        self.c = None  # (N, T, M)
        self.h_prev = None  # (N, M)
        self.c_prev = None  # (N, M)
        self.stateful = self.add_state(True)

    def forward(self):
        L, M = self.shape
        N, T, L = self.x.d.shape
        if self.h.d is None or not self.stateful.d:
            self.h_prev = np.zeros((N, M), dtype=self.dtype)
        else:
            self.h_prev = self.h.d
        if self.c is None or not self.stateful.d:
            self.c = np.empty((N, T, M), dtype=self.dtype)
            self.c_prev = np.zeros((N, M), dtype=self.dtype)
        else:
            self.c_prev = self.c[:, -1]
        if self.A is None or self.A.shape != (N, T, 4 * M):
            self.A = np.empty((N, T, 4 * M), dtype=self.dtype)  # [f, g, i, o]

        x = self.x.d @ self.W.d
        y = np.empty((N, T, M), dtype=self.dtype)

        for t in range(T):
            h = self.h_prev if t == 0 else y[:, t - 1]
            a = x[:, t] + h @ self.U.d + self.b.d
            a[:, :M] = sigmoid(a[:, :M])  # f
            a[:, M : 2 * M] = np.tanh(a[:, M : 2 * M])  # g
            a[:, 2 * M : 3 * M] = sigmoid(a[:, 2 * M : 3 * M])  # i
            a[:, 3 * M :] = sigmoid(a[:, 3 * M :])  # o
            self.A[:, t] = a
            f, g, i, o = a[:, :M], a[:, M : 2 * M], a[:, 2 * M : 3 * M], a[:, 3 * M :]
            c = self.c_prev if t == 0 else self.c[:, t - 1]
            self.c[:, t] = f * c + g * i
            y[:, t] = o * np.tanh(self.c[:, t])

        self.y.d = y
        self.h.d = y[:, -1]

    def backward(self):
        L, M = self.shape
        N, T, L = self.x.d.shape
        dx = np.empty((N, T, L), dtype=self.dtype)
        self.h.g = 0
        dc = 0
        for t in reversed(range(T)):
            dy = self.y.g[:, t] + self.h.g
            tanh_c = np.tanh(self.c[:, t])
            a = self.A[:, t]
            f, g, i, o = a[:, :M], a[:, M : 2 * M], a[:, 2 * M : 3 * M], a[:, 3 * M :]
            ds = dc + (dy * o) * (1 - tanh_c ** 2)
            c = self.c_prev if t == 0 else self.c[:, t - 1]
            dc = ds * f
            df = ds * c
            di = ds * g
            do = dy * tanh_c
            dg = ds * i
            df *= f * (1 - f)
            di *= i * (1 - i)
            do *= o * (1 - o)
            dg *= 1 - g ** 2
            da = np.hstack((df, dg, di, do))
            self.b.g = np.sum(da, axis=0)
            self.W.g = self.x.d[:, t].T @ da
            h = self.h_prev if t == 0 else self.y.d[:, t - 1]
            self.U.g = h.T @ da
            self.h.g = da @ self.U.d.T
            dx[:, t] = da @ self.W.d.T
        self.x.g = dx

    def reset_state(self):
        self.h.d = None
        self.c = None


class Select(Layer):
    input_ndim = 0

    def init(self):
        self.index = -1

    def forward(self):
        self.y.d = self.x.d[:, self.index].copy()

    def backward(self):
        dx = np.zeros_like(self.x.d)
        dx[:, self.index] = self.y.g
        self.x.g = dx
