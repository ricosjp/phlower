from enum import Enum, StrEnum


class PhlowerFileExtType(Enum):
    NPY = ".npy"
    NPYENC = ".npy.enc"
    NPZ = ".npz"
    NPZENC = ".npz.enc"
    # PKL = ".pkl"
    # PKLENC = ".pkl.enc"
    PTH = ".pth"
    PTHENC = ".pth.enc"
    YAML = ".yaml"
    YML = ".yml"
    YAMLENC = ".yaml.enc"
    YMLENC = ".yml.enc"
    CSV = ".csv"


class ModelSelectionType(Enum):
    BEST = "best"
    LATEST = "latest"
    TRAIN_BEST = "train_best"
    SPECIFIED = "specified"


class TrainerSavedKeyType(StrEnum):
    model_state_dict = "model_state_dict"
    scheduled_optimizer = "scheduled_optimizer"


class LossType(Enum):
    MSE = "mse"


class DirectoryType(Enum):
    RAW = 0
    INTERIM = 1
    PREPROCESSED = 2


class PhlowerScalerName(Enum):
    IDENTITY = "identity"
    ISOAM_SCALE = "isoam_scale"
    MAX_ABS_POWERED = "max_abs_powered"
    MIN_MAX = "min_max"
    # SPARSE_STD = "sparse_std"
    STANDARDIZE = "standardize"
    STD_SCALE = "std_scale"


class ActivationType(StrEnum):
    gelu = "gelu"
    identity = "identity"
    inversed_leaky_relu0p5 = "inversed_leaky_relu0p5"
    inversed_smooth_leaky_relu = "inversed_smooth_leaky_relu"
    leaky_relu0p5 = "leaky_relu0p5"
    leaky_relum0p5 = "leaky_relum0p5"
    relu = "relu"
    sigmoid = "sigmoid"
    smooth_leaky_relu = "smooth_leaky_relu"
    sqrt = "sqrt"
    tanh = "tanh"
    truncated_atanh = "truncated_atanh"


class PoolingType(StrEnum):
    max = "max"
    mean = "mean"


class PhlowerIterationSolverType(StrEnum):
    none = "none"
    simple = "simple"
    bb = "bb"  # barzilai_borwein
    cg = "cg"  # conjugate gradient


class PhlowerHandlerRegisteredKey(StrEnum):
    TERMINATE = "TERMINATE"


class PhlowerHandlerTrigger(StrEnum):
    iteration_completed = "iteration_completed"
    epoch_completed = "epoch_completed"


class TrainerInitializeType(StrEnum):
    none = "none"
    pretrained = "pretrained"
    restart = "restart"
