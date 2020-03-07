import re
from dataclasses import dataclass
from typing import Optional, Union


@dataclass(eq=False)
class Base:
    def __repr__(self):
        class_name = self.__class__.__name__
        name = self._repr_name_()
        if name.startswith("["):
            args = name
        elif name:
            args = f"'{name}'"
        else:
            args = ""
        if self.shape is not None:
            join = ", " if args else ""
            args += f"{join}{self.shape}"

        return f"<{class_name}({args}) at 0x{id(self):0x}>"

    def _repr_name_(self):
        return self.name


def get_class(base: type, cls: Union[type, str]) -> Optional[type]:
    """Return class or subclass type from str or type.

    Return None if no class found.
    """
    if base is cls:
        return base
    elif isinstance(cls, str) and is_same_class(cls, base):
        return base
    else:
        for base in base.__subclasses__():
            cls_ = get_class(base, cls)
            if cls_:
                return cls_
        else:
            return None


def is_same_class(cls: str, base: type) -> bool:
    """Return True if the class name is corresponding to the base type."""
    snake = to_snake_case(base.__name__) == to_snake_case(cls)
    lower = base.__name__.lower() == cls.lower()
    return snake or lower


def to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case.

    [Reference](https://codeday.me/jp/qa/20181222/13344.html)

    Example
    -------
    >>> to_snake_case("SoftmaxCrossEntropy")
    'softmax_cross_entropy'
    """
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
