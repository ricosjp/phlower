from enum import Enum


class PhlowerFileExtType(Enum):
    NPY = ".npy"
    NPYENC = ".npy.enc"
    NPZ = ".npz"
    NPZENC = ".npz.enc"
    PKL = ".pkl"
    PKLENC = ".pkl.enc"
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


class PhysicalDimensionType(Enum):
    mass = 0  # Ex: kilogram
    length = 1  # Ex: metre
    time = 2  # Ex: second
    thermodynamic_temperature = 3  # Ex: Kelvin
    amount_of_substance = 4  # Ex: mole
    electric_current = 5  # Ex. ampere
    luminous_intensity = 6  # Ex. candela

    @classmethod
    def is_exist(cls, name: str) -> bool:
        registered_names = [item.name for item in PhysicalDimensionType]
        return name in registered_names
