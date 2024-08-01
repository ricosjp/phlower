from enum import Enum

from typing_extensions import Self


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


class TrainerSavedKeyType(Enum):
    MODEL_STATE_DICT = "model_state_dict"


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


class PhysicalDimensionSymbolType(Enum):
    T = 0  # time
    L = 1  # length
    M = 2  # mass
    I = 3  # electric current
    Theta = 4  # thermodynamic temperature
    N = 5  # amount of substance
    J = 6  # luminous intensity

    @classmethod
    def is_exist(cls, name: str) -> bool:
        registered_names = [item.name for item in PhysicalDimensionSymbolType]
        return name in registered_names

    def to_quantity_name(self):
        _symbol2quntityname = {
            PhysicalDimensionSymbolType.T.name: "time",
            PhysicalDimensionSymbolType.L.name: "length",
            PhysicalDimensionSymbolType.M.name: "mass",
            PhysicalDimensionSymbolType.I.name: "electric current",
            PhysicalDimensionSymbolType.Theta.name: "thermodynamic temperature",
            PhysicalDimensionSymbolType.N.name: "amount of substance",
            PhysicalDimensionSymbolType.J.name: "luminous intensity",
        }
        return _symbol2quntityname[self.name]

    def __str__(self):
        return f"{self.name} ({self.to_quantity_name()})"
