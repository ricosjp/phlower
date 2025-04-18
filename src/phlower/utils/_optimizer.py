import torch


class OptimizerSelector:
    _REGISTERED: dict[str, type[torch.optim.Optimizer]] = {
        "Adadelta": torch.optim.Adadelta,
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "SparseAdam": torch.optim.SparseAdam,
        "Adamax": torch.optim.Adamax,
        "ASGD": torch.optim.ASGD,
        "LBFGS": torch.optim.LBFGS,
        "NAdam": torch.optim.NAdam,
        "RAdam": torch.optim.RAdam,
        "RMSprop": torch.optim.RMSprop,
        "Rprop": torch.optim.Rprop,
        "SGD": torch.optim.SGD,
    }

    @staticmethod
    def get_registered_names() -> list[str]:
        return list(OptimizerSelector._REGISTERED.keys())

    @staticmethod
    def register(name: str, cls: type[torch.optim.Optimizer]) -> None:
        OptimizerSelector._REGISTERED.update({name: cls})

    @staticmethod
    def exist(name: str) -> bool:
        return name in OptimizerSelector._REGISTERED

    @staticmethod
    def select(name: str) -> type[torch.optim.Optimizer]:
        _dict = OptimizerSelector._REGISTERED
        if name not in _dict:
            raise KeyError(
                f"{name} is not implemented as an optimizer in phlower. "
                f"Registered optimizers are {list(_dict.keys())}"
            )
        return _dict[name]
