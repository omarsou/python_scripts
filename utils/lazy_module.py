from typing import TypeVar, Generic


T = TypeVar('T')


class Lazy(Generic[T]):

    def __init__(self, ctor):
        super().__init__()
        self._ctor = ctor
        self._value = None
        self._got = False

    @property
    def value(self) -> T:
        if (self._got):
            return self._value
        self._value = self._ctor()
        self._got = True
        return self._value