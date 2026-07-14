from typing import Callable, Optional, TypeVar

T = TypeVar("T")
U = TypeVar("U")

# damn, rust is so much cooler




def map(x: Optional[T], f: Callable[[T], U]) -> Optional[U]:
    if x is not None:
        return f(x)
    return None


def unwrap(x: Optional[T], error: str = "") -> T:
    assert x is not None, error
    return x






