class InvalidDimensionError(ValueError):
    """This error raises when invalid dimension is found."""


class DimensionIncompatibleError(ValueError):
    """This error raises when calculation of physical units failed due to
    unit incompatibility.
    """


class PhlowerModuleNodeDimSizeError(ValueError):
    """This error raises when nodes setting in module is inconsistent with
    the previous module.
    """


class PhlowerModuleKeyError(ValueError):
    """This error raises when key is missed in phlower module settings."""


class PhlowerModuleDuplicateKeyError(ValueError):
    """This error raises when duplicate key is
    detected in phlower module settings.
    """


class PhlowerModuleDuplicateNameError(ValueError):
    """This error raises when duplicate name is
    detected in phlower GROUP module settings.
    """


class PhlowerModuleCycleError(ValueError):
    """This error raises when a cycle is detected in modules."""


class PhlowerMultiProcessError(ValueError): ...


class PhlowerFeatureStoreOverwriteError(ValueError):
    """This error raises when to try to overwrite feature store item"""

    ...


class PhlowerSparseRankUndefinedError(ValueError):
    """This error raises when trying to access tensor rank for sparse tensor"""
