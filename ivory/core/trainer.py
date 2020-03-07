from dataclasses import dataclass, field
from itertools import product as product_
from typing import (Callable, Dict, Iterable, Iterator, List, Optional, Tuple,
                    Union)

import pandas as pd

from ivory.common.context import np
from ivory.common.dataset import Dataset
from ivory.core.model import Model
from ivory.core.model import sequential as sequence_
from ivory.core.optimizer import Optimizer, get_optimizer
from ivory.core.variable import Data

EpochData = Union[Tuple[Data, ...], Dict[str, Tuple[Data, ...]]]


@dataclass(eq=False)
class Trainer:
    model: Model
    optimizer: Optimizer
    metrics: List[str] = field(default_factory=list)
    dataset: Optional[Dataset] = None
    epoch_data: EpochData = field(default_factory=tuple)
    max_grad: Optional[float] = None

    def __post_init__(self):
        self.optimizer.set_model(self.model)

    def __repr__(self):
        cls = self.__class__.__name__
        inputs = [v.shape for v in self.model.data_input_variables]
        r = f"{cls}(inputs={inputs}, optimizer='{self.optimizer.name}', "
        return r + f"metrics={self.metrics})"

    def init(self, **kwargs) -> "Trainer":
        self.model.init(**kwargs)
        return self

    def build(self):
        self.model.build()
        self.optimizer.set_model(self.model)

    def set_model(self, model: Model) -> "Trainer":
        self.model = model
        self.optimizer.set_model(self.model)
        return self

    def set_net(self, net: Iterable) -> "Trainer":
        return self.set_model(sequence_(net))

    def set_data(self, *data: Data) -> None:
        self.model.set_data(*data)

    def get_metrics(self, metrics: str) -> float:
        if metrics == "loss":
            return self.model.loss
        elif metrics in ["acc", "accuracy"]:
            return self.model.accuracy
        elif metrics in ["ppl", "perplexity"]:
            return np.exp(self.model.loss)
        else:
            raise ValueError(f"Unknown metrics: {metrics}.")

    def evaluate(self, *data: Data) -> Tuple[float, ...]:
        if data:
            self.set_data(*data)
            self.model.forward()

        return tuple(self.get_metrics(metrics) for metrics in self.metrics)

    def predict(self, *data: Data):
        return self.model.predict(*data)

    def fit(self, dataset: Dataset, epoch_data: EpochData = ()) -> "Trainer":
        self.dataset = dataset
        self.epoch_data = epoch_data
        return self

    def __iter__(self) -> Iterator[Tuple]:
        if self.dataset is None:
            raise StopIteration
        epoch = -1
        for data in self.dataset:
            self.set_data(*data)
            self.model.forward()
            self.model.backward()
            if self.max_grad is not None:
                self.model.clip_grads(self.max_grad)
            self.optimizer.update()
            if not self.epoch_data:
                yield (self.dataset.iteration,) + self.evaluate()  # type:ignore
            elif self.dataset.epoch != epoch:
                self.model.set_train(False)
                epoch = self.dataset.epoch
                if isinstance(self.epoch_data, tuple):
                    yield (epoch,) + self.evaluate(*self.epoch_data)  # type:ignore
                else:
                    for key, value in self.epoch_data.items():
                        yield (epoch, key) + self.evaluate(*value)  # type:ignore
                self.model.set_train(True)

    def to_frame(self, factory=list) -> pd.DataFrame:
        columns = to_columns(self.epoch_data) + self.metrics
        return pd.DataFrame(factory(iter(self)), columns=columns)


def to_columns(epoch_data: EpochData) -> List[str]:
    if not epoch_data:
        return ["iteration"]
    elif isinstance(epoch_data, tuple):
        return ["epoch"]
    else:
        return ["epoch", "data"]


def sequential(
    net: Iterable,
    optimizer: Union[str, Optimizer] = "sgd",
    dataset: Optional[Dataset] = None,
    metrics: Optional[List[str]] = None,
) -> Trainer:
    if isinstance(optimizer, str):
        optimizer = get_optimizer(optimizer)
    if metrics is None:
        metrics = ["loss"]

    return Trainer(sequence_(net), optimizer, metrics, dataset=dataset)


@dataclass(eq=False)
class Product:
    trainer_factory: Callable[..., Trainer]
    iterables: Tuple
    trainer: Optional[Trainer] = None
    dataset: Optional[Dataset] = None
    epoch_data: EpochData = ()

    def __repr__(self):
        cls = self.__class__.__name__
        iterables = ", ".join(f"<{len(x)}>" for x in self.iterables)
        r = f"{cls}(iterables=({iterables})"
        if self.trainer:
            r += f", trainer={self.trainer}"
        return r + ")"

    def fit(self, dataset: Dataset, epoch_data: EpochData = ()) -> "Product":
        self.dataset = dataset
        self.epoch_data = epoch_data
        return self

    def __iter__(self) -> Iterator[Tuple]:
        if self.dataset is None:
            raise StopIteration
        for args in product_(*self.iterables):
            self.trainer = self.trainer_factory(*args)
            for result in self.trainer.fit(self.dataset, epoch_data=self.epoch_data):
                yield args + result

    def to_frame(self, columns: Optional[List[str]] = None, factory=list):
        if columns is None:
            columns = [f"var_{k+1}" for k in range(len(self.iterables))]
        columns += to_columns(self.epoch_data)
        columns += self.trainer.metrics  # type:ignore

        return pd.DataFrame(factory(iter(self)), columns=columns)


def product(trainer_factory: Callable[..., Trainer], *iterables) -> Product:
    return Product(trainer_factory, iterables)
