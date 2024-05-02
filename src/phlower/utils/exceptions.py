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
