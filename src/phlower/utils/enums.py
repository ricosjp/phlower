from enum import Enum


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


class TrainerSavedKeyType(str, Enum):
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


class ActivationType(str, Enum):
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


class PoolingType(str, Enum):
    max = "max"
    mean = "mean"


class PhysicalDimensionSymbolType(Enum):
    T = 0  # time
    L = 1  # length
    M = 2  # mass
    I = 3  # electric current  # noqa: E741
    Theta = 4  # thermodynamic temperature
    N = 5  # amount of substance
    J = 6  # luminous intensity

    @classmethod
    def is_exist(cls, name: str) -> bool:
        return name in _symbol2quntityname.keys()

    def to_quantity_name(self) -> str:
        return _symbol2quntityname[self.name]

    def __str__(self):
        return f"{self.name} ({self.to_quantity_name()})"


_symbol2quntityname = {
    PhysicalDimensionSymbolType.T.name: "time",
    PhysicalDimensionSymbolType.L.name: "length",
    PhysicalDimensionSymbolType.M.name: "mass",
    PhysicalDimensionSymbolType.I.name: "electric current",
    PhysicalDimensionSymbolType.Theta.name: "thermodynamic temperature",
    PhysicalDimensionSymbolType.N.name: "amount of substance",
    PhysicalDimensionSymbolType.J.name: "luminous intensity",
}


class PhlowerIterationSolverType(str, Enum):
    none = "none"
    simple = "simple"
    bb = "bb"  # barzilai_borwein
    cg = "cg"  # conjugate gradient


class PhlowerHandlerRegisteredKey(str, Enum):
    TERMINATE = "TERMINATE"


class PhlowerHandlerTrigger(str, Enum):
    iteration_completed = "iteration_completed"
    epoch_completed = "epoch_completed"


class TrainerInitializeType(str, Enum):
    none = "none"
    pretrained = "pretrained"
    restart = "restart"
