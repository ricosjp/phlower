from collections.abc import Callable
from typing import TypeVar

from phlower.nn._interface_module import IPhlowerModuleAdapter
from phlower.utils.exceptions import PhlowerRunTimeError

T1 = TypeVar("T1")
T2 = TypeVar("T2")


def attach_location_to_error_message(
    func: Callable[[IPhlowerModuleAdapter, T1], T2],
) -> Callable[[IPhlowerModuleAdapter, T1], T2]:
    """Decorator to modify the error message of a function.

    Args:
        func (callable): The function to be decorated.

    Returns:
        callable: The decorated function with modified error message.
    """

    def wrapper(self: IPhlowerModuleAdapter, *args: T1, **kwargs) -> T2:
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            if isinstance(e, PhlowerRunTimeError):
                e.add_location(self.name)
                raise e
            else:
                ex = PhlowerRunTimeError()
                ex.set_message(str(e))
                ex.add_location(self.name)
                raise ex from e

    return wrapper
