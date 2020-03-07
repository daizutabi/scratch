from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from ivory.core.base import Base

Data = Any
Shape = Tuple[int, ...]


@dataclass(repr=False, eq=False)
class Variable(Base):
    shape: Shape
    name: str = ""
    parameters: List = field(default_factory=list)
    data: Optional[Data] = field(default=None)
    grad: Optional[Data] = field(default=None)
    init: Optional[Callable[..., Data]] = None
    frozen: bool = False

    def _repr_name_(self) -> str:
        if self.name:
            return self.name
        elif self.parameters:
            name = ", ".join(f"'{p._repr_name_()}'" for p in self.parameters)
            name = f"[{name}]"
            return name
        else:
            return ""

    def add_parameter(self, parameter):
        if not match_shape(self, parameter):
            raise ValueError(f"Shape doesn't match: {self.shape} != {parameter.shape}")
        if parameter not in self.parameters:
            self.parameters.append(parameter)
        parameter.variable = self
        if parameter.init is not None:
            self.init = parameter.init
            if self.data is None:
                self.data = self.init()

        return parameter


def match_shape(variable: Variable, parameter) -> bool:
    if parameter.transpose is False:
        return variable.shape == parameter.shape
    else:
        return variable.shape == parameter.shape[::-1]
