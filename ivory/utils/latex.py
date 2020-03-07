from itertools import product
from typing import Callable, List, Union

Symbol = Union[str, Callable[..., str]]


def array(symbol: Symbol, *shape: int, parenthesis: bool = True, pre=()) -> List:
    if isinstance(symbol, str):

        def func(*index):
            return "".join([symbol, subscript(*pre, *index, parenthesis=parenthesis)])

        return array(func, *shape)

    if len(shape) == 1:
        return [symbol(*pre, i + 1) for i in range(shape[0])]
    elif len(shape) == 2:
        n, m = shape
        if m:
            return [[symbol(*pre, i + 1, j + 1) for j in range(m)] for i in range(n)]
        else:
            return [[symbol(*pre, i + 1) for i in range(n)]]
    else:
        raise ValueError


def matrix(array: List) -> str:
    if not isinstance(array[0], str):
        return matrix([" & ".join(row) for row in array])

    mat = " \\\\ ".join(array)
    return r"\left[\begin{matrix}" + mat + r"\end{matrix}\right]"


def frac(a: str, b: str, fold=False) -> str:
    if fold:
        return f"{a}/{b}"
    else:
        return f"\\frac{{{a}}}{{{b}}}"


def subscript(*index: int, parenthesis=True, delim="") -> str:
    sub = delim.join(str(i) for i in index)
    return f"_{{{sub}}}" if parenthesis else f"_{sub}"


def partial(symbol: str, variable: str, fold=False, func=False) -> Symbol:
    a, b = (f"\\partial {x}" for x in [symbol, variable])

    if func:

        def expr(*args, **kwargs):
            return frac(a, b + subscript(*args, **kwargs), fold)

        return expr

    else:
        return frac(a, b, fold)


class Expr:
    def __init__(self, value: str):
        self._value = value

    @property
    def value(self) -> str:
        return self._value

    def __repr__(self):
        return self.value

    def _repr_latex_(self):
        return repr(self), {"module": "sympy"}


class Matrix(Expr):
    def __init__(self, variable, *shape):
        self.variable = variable
        self.shape = shape

    @property
    def value(self) -> str:
        def sub(*pre):
            return matrix(array(self.variable, *self.shape[-2:], pre=pre))

        if len(self.shape) == 0:
            return f"\\mathbf{{{self.variable.upper()}}}"
        elif len(self.shape) <= 2:
            return sub()
        elif len(self.shape) == 3:
            return matrix([sub(i + 1) for i in range(self.shape[0])])
        elif len(self.shape) == 4:
            n, m = self.shape[:2]
            return matrix([[sub(i + 1, j + 1) for j in range(m)] for i in range(n)])
        else:
            raise ValueError

    @property
    def symbol(self) -> Expr:
        return Expr(f"\\mathbf{{{self.variable.upper()}}}")

    @property
    def s(self) -> Expr:
        return self.symbol

    @property
    def sympy(self):
        import sympy

        return sympy.Matrix(array(self.variable, *self.shape, parenthesis=False))

    @property
    def S(self):
        return self.sympy

    def spartial(self, symbol: str, fold: bool = False) -> Expr:
        return Expr(partial(symbol, str(self.symbol), fold=fold))  # type: ignore

    def partial(self, func: str, fold: bool = True) -> Expr:
        expr = partial(func, self.variable, fold=fold, func=True)
        return Expr(matrix(array(expr, *self.shape)))

    def numpy(self, variable=None):
        import numpy as np

        variable = variable or self.variable
        prod = product(*(range(1, x + 1) for x in self.shape))
        return np.array([substr(variable, x) for x in prod]).reshape(*self.shape)

    @property
    def numpy_str(self):
        return self.numpy()

    @property
    def numpy_int(self):
        return self.numpy(int)

    def apply(self, func, *args, **kwargs):
        import sympy
        import numpy as np

        array = func(self.numpy_int, *args, **kwargs).astype(int)
        shape = array.shape
        array = ["_".join([self.variable, str(x)]) for x in array.reshape(-1)]
        array = np.array(array).reshape(*shape)
        return sympy.Matrix(array)


def substr(variable, sub):
    sub = "".join(str(x) for x in sub)
    if isinstance(variable, str):
        return "_".join([variable, sub])
    else:
        return variable(sub)


class Vector(Matrix):
    def __init__(self, variable, length):
        super().__init__(variable, length, 0)
