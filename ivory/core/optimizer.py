from dataclasses import dataclass, field
from typing import Iterable, List

from ivory.common.context import np
from ivory.core.base import get_class
from ivory.core.model import Model
from ivory.core.variable import Data, Variable


@dataclass(eq=False)
class Optimizer:
    learning_rate: float = 0.01
    variables: List[Variable] = field(default_factory=list, repr=False)
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = self.__class__.__name__

    def set_variables(self, variables: Iterable[Variable]):
        self.variables = list(variables)
        self.init()

    def set_model(self, model: Model):
        self.set_variables(model.weight_variables)

    def init(self):
        pass

    def update(self):
        pass


def get_optimizer(name: str, *args, **kwargs) -> Optimizer:
    cls = get_class(Optimizer, name)
    if cls is None:
        raise ValueError(f"Unknown optimizer: {name}.")
    return cls(*args, **kwargs)


class SGD(Optimizer):
    def update(self):
        for variable in self.variables:
            variable.data -= self.learning_rate * variable.grad


@dataclass(eq=False)
class Momentum(Optimizer):
    momentum: float = 0.9
    v: List[Data] = field(init=False)

    def init(self):
        self.v = [np.zeros_like(variable.data) for variable in self.variables]

    def update(self):
        for k, variable in enumerate(self.variables):
            self.v[k] = self.momentum * self.v[k] - self.learning_rate * variable.grad
            variable.data += self.v[k]


class Nesterov(Momentum):
    """Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)."""

    def update(self):
        for k, variable in enumerate(self.variables):
            self.v[k] *= self.momentum
            self.v[k] -= self.learning_rate * variable.grad
            variable.data += self.momentum * self.momentum * self.v[k]
            variable.data -= (1 + self.momentum) * self.learning_rate * variable.grad


@dataclass(eq=False)
class AdaGrad(Optimizer):
    h: List[Data] = field(init=False)

    def init(self):
        self.h = [np.zeros_like(variable.data) for variable in self.variables]

    def update(self):
        for k, variable in enumerate(self.variables):
            self.h[k] += variable.grad * variable.grad
            sqrt = np.sqrt(self.h[k]) + 1e-7
            variable.data -= self.learning_rate * variable.grad / sqrt


@dataclass(eq=False)
class RMSprop(AdaGrad):
    decay_rate: float = 0.99

    def update(self):
        for k, variable in enumerate(self.variables):
            self.h[k] *= self.decay_rate
            self.h[k] += (1 - self.decay_rate) * variable.grad * variable.grad
            sqrt = np.sqrt(self.h[k]) + 1e-7
            variable.data -= self.learning_rate * variable.grad / sqrt


@dataclass(eq=False)
class Adam(Optimizer):
    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    iteration: int = field(init=False)
    m: List[Data] = field(init=False)
    v: List[Data] = field(init=False)

    def init(self):
        self.iteration = 0
        self.m = [np.zeros_like(variable.data) for variable in self.variables]
        self.v = [np.zeros_like(variable.data) for variable in self.variables]

    def update(self):
        self.iteration += 1
        lr, it = self.learning_rate, self.iteration
        lr *= np.sqrt(1.0 - self.beta2 ** it) / (1.0 - self.beta1 ** it)

        for k, variable in enumerate(self.variables):
            self.m[k] += (1 - self.beta1) * (variable.grad - self.m[k])
            self.v[k] += (1 - self.beta2) * (variable.grad ** 2 - self.v[k])
            variable.data -= lr * self.m[k] / (np.sqrt(self.v[k]) + 1e-7)
