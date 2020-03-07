import inspect
from dataclasses import dataclass, field
from itertools import chain, product
from typing import Iterable, Iterator, List, Optional, Tuple

from ivory.common.context import np
from ivory.core.base import get_class
from ivory.core.layer import Layer
from ivory.core.parameter import Input, Loss, Output, State, Weight
from ivory.core.variable import Data, Shape, Variable
from ivory.layers.loss import LossLayer


@dataclass(repr=False)
class Model:
    losses: List[Loss]
    layers: List[Layer] = field(init=False)
    inputs: List[Input] = field(init=False)
    outputs: List[Output] = field(init=False)
    weights: List[Weight] = field(init=False)
    states: List[State] = field(init=False)
    input_variables: List[Variable] = field(init=False)
    output_variables: List[Variable] = field(init=False)
    weight_variables: List[Variable] = field(init=False)
    state_variables: List[Variable] = field(init=False)
    loss_variables: List[Variable] = field(init=False)
    data_input_variables: List[Variable] = field(init=False)
    data_output_variables: List[Variable] = field(init=False)
    frozen_input_variables: List[Variable] = field(init=False)

    def __post_init__(self):
        self.build()

    def forward(self, predict=False, start=0):
        for variable in chain(self.output_variables, self.loss_variables):
            variable.data = None
        for layer in self.layers[start:]:
            if predict and isinstance(layer, LossLayer):
                continue
            layer.forward()

    def backward(self):
        for variable in chain(self.input_variables, self.weight_variables):
            variable.grad = None
        for layer in reversed(self.layers):
            layer.backward()

    def build(self) -> "Model":
        """Collect layers or parameters structures before evaluation or training."""
        self.layers = list(self.find_layers())

        def get_parameters(attr):
            return [param for layer in self.layers for param in getattr(layer, attr)]

        self.inputs = get_parameters("inputs")
        self.outputs = get_parameters("outputs")
        self.weights = get_parameters("weights")
        self.states = get_parameters("states")

        ps = chain(self.inputs, self.outputs, self.weights, self.states, self.losses)
        for p in ps:
            if not p.variable:
                p.set_variable()

        def get_unique_variables(parameters):
            return list(dict.fromkeys(p.variable for p in parameters))

        self.input_variables = get_unique_variables(self.inputs)
        self.output_variables = get_unique_variables(self.outputs)
        self.weight_variables = get_unique_variables(self.weights)
        self.state_variables = get_unique_variables(self.states)
        self.loss_variables = get_unique_variables(self.losses)

        def get_data_variables(variables, frozen=False):
            vs = [v for v in variables if len(v.parameters) == 1 and v.frozen is frozen]
            return vs

        self.data_input_variables = get_data_variables(self.input_variables)
        self.data_output_variables = get_data_variables(self.output_variables)
        self.frozen_input_variables = get_data_variables(self.input_variables, True)

        return self

    @property
    def grad_variables(self):
        vs = [self.data_input_variables, self.weight_variables, self.state_variables]
        for v in chain(*vs):
            if v.grad is not None:
                yield v

    @property
    def loss(self) -> float:
        return sum(variable.data for variable in self.loss_variables)  # type:ignore

    @property
    def perplexity(self) -> float:
        return np.exp(self.loss)

    @property
    def accuracy(self) -> float:
        return sum(loss.layer.accuracy for loss in self.losses) / len(self.losses)

    def predict(self, *data: Data) -> Data:
        if data:
            self.set_data(*data)
            self.forward(predict=True)
        pred = [loss.layer.predict() for loss in self.losses]
        if len(self.losses) == 1:
            return pred[0]
        else:
            return pred

    def numerical_gradient(self, variable: Variable, epsilon: float = 1e-4) -> Data:
        """Return the numerical gradient tensor (partial self / partial variable)."""
        data = variable.data
        if data is None:
            return None
        grad = np.zeros_like(data)
        state_data = [v.data for v in self.state_variables]
        for index in product(*(range(x) for x in data.shape)):
            value = data[index]
            data[index] += epsilon
            self.forward()
            plus = self.loss
            data[index] -= 2 * epsilon
            for v, d in zip(self.state_variables, state_data):
                v.data = d
            self.forward()
            minus = self.loss
            grad[index] = (plus - minus) / (2 * epsilon)
            data[index] = value
            for v, d in zip(self.state_variables, state_data):
                v.data = d

        return grad

    def gradient_error(self, variable: Variable, epsilon: float = 1e-4) -> float:
        """Return gradient error."""
        if variable.grad is None:
            return np.nan
        grad = self.numerical_gradient(variable, epsilon)
        return np.average(np.abs(variable.grad - grad))

    def init(self, **kwargs) -> "Model":
        for variable in chain(self.weight_variables, self.state_variables):
            if variable.init is None:
                continue
            name = variable.parameters[0].name
            if name in kwargs.keys():
                variable.data = variable.init(kwargs[name])
            else:
                signature = inspect.signature(variable.init)
                keys = signature.parameters.keys()
                kwargs_ = {key: value for key, value in kwargs.items() if key in keys}
                variable.data = variable.init(**kwargs_)
        return self

    def set_train(self, train: bool) -> None:
        """Set `train` state for layers with `train` state."""
        for layer in self.layers:
            if hasattr(layer, "train"):
                layer.train.d = layer.train.init(train)  # type:ignore

    def set_data(self, *data: Data) -> None:
        for variable, x in zip(self.data_input_variables, data):
            variable.data = x

    def reset_state(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "reset_state"):
                layer.reset_state()  # type:ignore

    def find_layers(self) -> Iterator[Layer]:
        """Yield layers in order of forward propagation removing duplicated layers.

        `forward` method of layers can be called in this order and
        `backward` method can be called in the reversed order.
        """
        layers: List[Layer] = []
        for loss in self.losses:
            for layer in parent_layers(loss.layer, layers):
                yield layer
                layers.append(layer)

    def clip_grads(self, max_grad):
        grads = [v.grad for v in self.weight_variables]
        total_norm = 0.0
        for grad in grads:
            total_norm += np.sum(grad ** 2)  # type:ignore
        total_norm = np.sqrt(total_norm)
        rate = max_grad / (total_norm + 1e-6)
        if rate < 1:
            for grad in grads:
                grad *= rate  # type:ignore


def parent_layers(layer: Layer, layers: List[Layer]) -> Iterator[Layer]:
    layers.append(layer)
    for input in chain(layer.states, layer.inputs):
        if input.variable is None:
            continue
        for parameter in input.variable.parameters:
            if isinstance(parameter, Output):
                if parameter.layer not in layers:
                    yield from parent_layers(parameter.layer, layers)
    yield layer


def branch(net: Iterable, input_layer: Optional[Layer] = None) -> List[Layer]:
    layers = list(normalize(net))
    if input_layer is None:
        input_shape = layers[0][1]
        layers = layers[1:]
    else:
        input_shape = input_layer.y.shape

    layers_: List[Layer] = []
    for k, (name, shape) in enumerate(layers):
        cls = get_class(Layer, name)
        if cls is None:
            raise ValueError(f"Layer not found: {name}")
        layer = cls(input_shape + shape)
        if input_layer:
            layer.set_input_layer(input_layer)
        input_layer = layer
        input_shape = layer.y.shape
        layers_.append(layer)
    return layers_


def sequential(net: Iterable) -> Model:
    """Return model instance of the created sequential layers.

    Parameters
    ----------
    net
        Layer description. Element of the iterable contains layer class name and shape.
        The element can contains activation layer class name at the end.

        The first element must be ("input",  *input_shape).
        The Last element must be a loss layer which has a `loss` parameter.

    Layer description example:
        3-inputs, 3x5 affine + relu activation, 5x4 affine + softmax-cross-entropy.
        [('input', 3), ('affine', 5, 'relu'), ('affine', 4, 'softmax_cross_entropy')]
    """
    return Model([branch(net)[-1].loss])  # type:ignore


def normalize(net: Iterable) -> Iterator[Tuple[str, Shape]]:
    """Normalize layer description for creating a sequence net.

    Examples
    --------
    >>> list(normalize([("affine", 3, "relu"), ("affine", 2)]))
    [('affine', (3,)), ('relu', ()), ('affine', (2,))]
    >>> list(normalize([("affine", 3, "relu", "xxx"), ("affine", 2)]))
    [('affine', (3,)), ('relu', ()), ('xxx', ()), ('affine', (2,))]
    >>> list(normalize([(2, "affine", 3, "relu")]))
    [('affine', (3,)), ('relu', ()), ('affine', (3,)), ('relu', ())]
    >>> list(normalize([("input", 3), "softmax_cross_entropy"]))
    """
    for x in net:
        if isinstance(x, str):
            yield (x, ())
        elif isinstance(x[0], int):
            for _ in range(x[0]):
                yield from normalize([x[1:]])
        elif isinstance(x[-1], str):
            yield from normalize([x[:-1]])
            yield (x[-1], ())
        else:
            yield (x[0], tuple(x[1:]))
