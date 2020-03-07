from dataclasses import dataclass, field
from itertools import chain
from typing import List

from ivory.core.base import Base
from ivory.core.parameter import Input, Loss, Output, Parameter, State, Weight
from ivory.core.variable import Data, Shape, Variable


@dataclass(repr=False, eq=False)
class Layer(Base):
    shape: Shape = ()
    name: str = ""
    dtype: str = "f"
    parameters: List[Parameter] = field(default_factory=list, init=False)
    inputs: List[Input] = field(default_factory=list, init=False)
    outputs: List[Output] = field(default_factory=list, init=False)
    weights: List[Weight] = field(default_factory=list, init=False)
    states: List[State] = field(default_factory=list, init=False)

    init_input = True
    init_output = True
    input_ndim = -1
    counter = 0

    def __post_init__(self):
        if self.init_input:
            shape = self.shape[: self.input_ndim] if self.input_ndim else self.shape
            self.x = self.add_input(shape)
        if self.init_output:
            self.y = self.add_output(self.shape[self.input_ndim :])
        self.init()

        if not self.name:
            self.__class__.counter += 1
            self.name = f"{self.__class__.__name__}.{self.counter}"

        for name, param in vars(self).items():
            if param in self.parameters:
                param.name = name

    def add_parameter(self, cls: type, shape: Shape = (), params=None):
        parameter = cls(shape, layer=self)
        self.parameters.append(parameter)
        if params is not None:
            params.append(parameter)
        return parameter

    def add_input(self, shape: Shape = ()) -> Input:
        return self.add_parameter(Input, shape, self.inputs)

    def add_output(self, shape: Shape = ()) -> Output:
        return self.add_parameter(Output, shape, self.outputs)

    def add_weight(self, shape: Shape = ()) -> Weight:
        return self.add_parameter(Weight, shape, self.weights)

    def add_state(self, shape: Shape = ()) -> State:
        return self.add_parameter(State, shape, self.states)

    def init(self):
        pass

    def set_variables(self) -> List[Variable]:
        """Assign new variables to layer's parameters if `variable` is None."""
        variables = []
        for parameter in self.parameters:
            if parameter.variable:
                variables.append(parameter.variable)
            else:
                variables.append(parameter.set_variable())
        return variables

    def set_input_layer(self, layer: "Layer") -> List[Variable]:
        """Set input layer and share variables between input and output parameters."""
        variables = []
        for i, o in zip(self.inputs, layer.outputs):
            if not o.variable:
                o.set_variable()
            variables.append(i.set_variable(o.variable))
        return variables

    def share_weight_variables(self, layer: "Layer") -> List[Variable]:
        """Share weight parameters with a given layer."""
        it = zip(self.weights, layer.weights)
        return [p.set_variable(q.variable) for p, q in it]

    def set_data(self, *data: Data):
        """Set input data to input parameters."""
        for input, x in zip(self.inputs, data):
            input.variable.data = x  # type:ignore

    def clear_data(self):
        for parameter in self.outputs:
            if parameter.variable is not None:
                parameter.variable.data = None

    def clear_grad(self):
        for parameter in chain(self.inputs, self.weights):
            if parameter.variable is not None:
                parameter.variable.grad = None


class LossLayer(Layer):
    input_ndim = 0

    def init(self):
        self.t = self.add_input()
        self.loss = self.add_parameter(Loss)

    def clear_data(self):
        super().clear_data()
        if self.loss.variable is not None:
            self.loss.variable.data = None
