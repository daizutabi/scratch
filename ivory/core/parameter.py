from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ivory.common.context import np
from ivory.core.base import Base
from ivory.core.variable import Data, Shape, Variable


@dataclass(repr=False, eq=False)
class Parameter(Base):
    shape: Shape
    layer: Any
    name: str = ""
    transpose: bool = False
    variable: Optional[Variable] = None
    init: Optional[Callable[..., Data]] = field(default=None, init=False)

    def _repr_name_(self) -> str:
        if not isinstance(self.layer, Base):
            return self.name
        else:
            return ".".join([self.layer._repr_name_(), self.name])

    def __post_init__(self):
        if not isinstance(self.shape, tuple):
            value, self.shape = self.shape, ()

            def init(value=value):
                return value

            self.init = init

    def set_variable(self, variable: Optional[Variable] = None) -> Variable:
        variable = variable or Variable(self.shape)
        variable.add_parameter(self)
        return variable

    def share_variable(self, parameter: "Parameter", transpose=False) -> Variable:
        self.transpose = transpose
        return self.set_variable(parameter.variable)

    def ones(self) -> "Parameter":
        self.init = lambda: np.ones(self.shape, dtype=self.layer.dtype)
        return self

    def zeros(self) -> "Parameter":
        self.init = lambda: np.zeros(self.shape, dtype=self.layer.dtype)
        return self

    def randn(self, std=None) -> "Parameter":
        def init(std="he"):
            if isinstance(std, str):
                if std.lower() == "xavier":
                    std = np.sqrt(1 / self.shape[0], dtype=self.layer.dtype)
                elif std.lower() == "he":
                    std = np.sqrt(2 / self.shape[0], dtype=self.layer.dtype)
                else:
                    raise ValueError(f"Unknown std name: {std}.")

            return std * np.random.randn(*self.shape).astype(self.layer.dtype)

        self.init = init
        return self

    @property
    def d(self):
        return self.variable.data


class GradParameter(Parameter):
    @property
    def g(self):
        return self.variable.grad

    @g.setter
    def g(self, grad):
        if self.variable.grad is None or grad is None:
            self.variable.grad = grad
        else:
            self.variable.grad += grad


class Input(GradParameter):
    pass


class Weight(GradParameter):
    pass


class DataParameter(Parameter):
    @property
    def d(self):
        return self.variable.data

    @d.setter
    def d(self, data):
        if self.variable.data is None or data is None:
            self.variable.data = data
        else:
            self.variable.data += data


class Output(DataParameter):
    @property
    def g(self):
        return self.variable.grad


class Loss(DataParameter):
    pass


class State(Parameter):
    @property
    def d(self):
        return self.variable.data

    @d.setter
    def d(self, data):
        self.variable.data = data

    @property
    def g(self):
        return self.variable.grad

    @g.setter
    def g(self, grad):
        self.variable.grad = grad
