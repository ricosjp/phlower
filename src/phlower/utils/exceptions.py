from typing import Callable

class DimensionIncompatibleError(ValueError):
    """This error raises when calculation of physical units failed due to
    unit incompatibility.
    """
