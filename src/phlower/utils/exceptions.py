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


class PhlowerIterationSolverSettingError(ValueError):
    """This error raises when an settin item in interation solver is invalid."""

    ...


class PhlowerMultiProcessError(ValueError): ...


class PhlowerFeatureStoreOverwriteError(ValueError):
    """This error raises when to try to overwrite feature store item"""

    ...


class PhlowerRestartTrainingCompletedError(ValueError):
    """This error raises when trying to restart train job
    which has already completed.
    """

    ...


class PhlowerSparseUnsupportedError(ValueError):
    """
    This error raises when trying to call methods not supported for sparse
    tensors
    """


class PhlowerUnsupportedTorchFunctionError(ValueError):
    """
    This error raises when trying to call a function not supported
    by the phlower library although torch does
    """


class PhlowerReshapeError(ValueError):
    """
    This error raises when trying to reshape a tensor in an invalid
    manner
    """


class NotFoundReferenceModuleError(ValueError): ...


class PhlowerIncompatibleTensorError(ValueError):
    """
    This error raises when trying to perform an operation for incompatible
    tensor(s)
    """


class PhlowerTypeError(TypeError):
    """This error raises when type is not compatible with what is expected."""


class PhlowerInvalidActivationError(ValueError):
    """This error raises when a set activation is invalid"""


class PhlowerDimensionRequiredError(ValueError):
    """
    This error raises when the dimension does not exist despite required.
    """


class PhlowerInvalidArgumentsError(ValueError):
    """This error raises when the arguments are invalid."""


class PhlowerNaNDetectedError(RuntimeError):
    """This error raises when NaN value is detected."""
