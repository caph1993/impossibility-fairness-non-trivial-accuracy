from typing import Callable, TypeVar
from typing_extensions import Protocol
from typing import Callable, TypeVar

_T = TypeVar("_T")


class _Property(Protocol[_T]):
    __get__: Callable[..., _T]


class cached_property(property, _Property[_T]):
    """
    decorator for converting a method into a cached property
    See https://stackoverflow.com/a/4037979/3671939
    This uses a modification:
    1. inherit from property, which disables setattr(instance, name, value)
        as it raises AttributeError: Can't set attribute
    2. use instance.__dict__[name] = value to fix
    """

    def __init__(self, method: Callable[..., _T]):
        self._method = method

    def __get__(self, instance, _) -> _T:
        name = self._method.__name__
        value = self._method(instance)
        instance.__dict__[name] = value
        return value
