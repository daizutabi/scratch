import collections
from dataclasses import dataclass, field
from itertools import accumulate, count
from typing import List, Tuple

from ivory.common.context import np
from ivory.common.util import create_contexts_target
from ivory.core.variable import Data


@dataclass(eq=False)
class Dataset:
    data: List[Data]
    size: Tuple = (1,)
    batch_size: int = 1
    column: int = 0
    random: bool = False
    epochs: int = 1
    name: str = ""
    index: int = field(default=0, init=False)
    epoch: int = field(default=0, init=False)
    iteration: int = field(default=0, init=False)

    def __post_init__(self):
        self.data = [np.asarray(x) for x in self.data]
        self.length = self.data[0].shape[0]

    def __repr__(self):
        name = self.name or self.__class__.__name__
        s = f"{name}(batch_size={self.batch_size}, epochs={self.epochs}, "
        return s + f"len={len(self)}, column={self.column}, size={self.size})"

    def __len__(self):
        return self.size[self.column] // self.batch_size

    def __getitem__(self, index) -> Tuple[Data, ...]:
        if isinstance(index, tuple):
            column = self.column
            self.column, index = index
            data = self[index]
            self.column = column
            return data
        if isinstance(index, slice):
            start = (index.start or 0) * self.batch_size
            start += self.start[self.column]
            stop = (index.stop or 0) * self.batch_size or self.size[self.column]
            stop += self.start[self.column]
            index = slice(start, stop)
        elif self.random:
            index = np.random.choice(self.size[self.column], self.batch_size)
            index += self.start[self.column]
        elif isinstance(index, int):
            start = self.batch_size * index + self.start[self.column]
            stop = start + self.batch_size
            index = slice(start, stop)
        return tuple(x[index] for x in self.data)

    def __iter__(self):
        length = len(self)
        epoch_iter = range(self.epochs) if self.epochs >= 0 else count()
        self.iteration = 0
        for self.epoch in epoch_iter:
            for self.index in range(length):
                yield self[self.index]
                self.iteration += 1

    @property
    def shape(self):
        return tuple((self.batch_size,) + x.shape[1:] for x in self.data)

    def shuffle(self, column=None) -> None:
        if column is not None:
            raise NotImplementedError
        index = np.random.permutation(self.data[0].shape[0])
        for k in range(len(self.data)):
            self.data[k] = self.data[k][index]

    def split(self, size: Tuple[int, ...]) -> None:
        size_ = [self.length * x // sum(size) for x in size[:-1]]
        self.size = (*size_, self.length - sum(size_))
        self.start = (0, *accumulate(self.size))

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, length):
        if length == -1:
            length = self.data[0].shape[0]
        self._length = length
        self.split(self.size)

    @property
    def state(self):
        return (self.epoch, self.index, self.iteration)


@dataclass(eq=False, repr=False)
class ContextDataset(Dataset):
    negative_sample_size: int = 0
    window_size: int = 1
    power: float = 0.75
    replace: bool = False

    def __post_init__(self):
        self.corpus = self.data
        self.sampler = UnigramSampler(self.corpus, self.power)
        self.set_window_size(self.window_size)

    def __getitem__(self, index) -> Tuple[Data, ...]:
        contexts, target = super().__getitem__(index)
        if self.negative_sample_size:
            negative_sample = self.sampler.get_negative_sample(
                target, self.negative_sample_size, self.replace
            )
            return (contexts, target) + tuple(negative_sample.T)
        else:
            return contexts, target

    def set_window_size(self, window_size):
        self.window_size = window_size
        self.data = create_contexts_target(self.corpus, self.window_size)
        self.length = self.data[0].shape[0]

    @property
    def vocab_size(self):
        return self.sampler.vocab_size


class UnigramSampler:
    def __init__(self, corpus, power=0.75):
        counts = collections.Counter(corpus)
        self.vocab_size = len(counts)
        self.probability = np.zeros(self.vocab_size)
        for i in range(self.vocab_size):
            self.probability[i] = counts[i]
        self.probability = np.power(self.probability, power)
        self.probability /= np.sum(self.probability)

    def get_negative_sample(self, target, sample_size, replace=False):
        batch_size = target.shape[0]

        if not replace:
            negative_sample = np.zeros((batch_size, sample_size), dtype=np.int32)
            for i in range(batch_size):
                p = self.probability.copy()
                p[target[i]] = 0
                p /= p.sum()
                negative_sample[i, :] = np.random.choice(
                    self.vocab_size, size=sample_size, replace=False, p=p
                )
            return negative_sample
        else:  # Fast when `replace` is True.
            size = (batch_size, sample_size)
            p = self.probability
            return np.random.choice(self.vocab_size, size, replace=True, p=p)


@dataclass(eq=False)
class TimeDataset(Dataset):
    time_size: int = 1

    def __repr__(self):
        s = super().__repr__()
        return s.replace("(batch", f"(time_size={self.time_size}, batch")

    def __len__(self):
        return self.size[self.column] // (self.batch_size * self.time_size)

    def __getitem__(self, index) -> Tuple[Data, ...]:
        if isinstance(index, int):
            size = self.size[self.column] // self.batch_size
            start = np.arange(self.batch_size) * size + self.start[self.column]
            start = start.reshape(-1, 1) + np.arange(self.time_size)
            start += index * self.time_size
            index = start
        else:
            raise NotImplementedError
        return tuple(x[index] for x in self.data)

    @property
    def shape(self):
        return tuple((self.batch_size, self.time_size) + x.shape[1:] for x in self.data)


@dataclass(eq=False)
class Seq2seqDataset(Dataset):
    def __post_init__(self):
        super().__post_init__()
        self.data = [self.data[0], self.data[1][:, :-1], self.data[1][:, 1:]]
        # self.data = [self.data[0], self.data[1][:, :-1].copy(), self.data[1][:, 1:].copy()]
