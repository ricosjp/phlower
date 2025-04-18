import torch


class SchedulerSelector:
    _REGISTERED: dict[str, type[torch.optim.lr_scheduler.LRScheduler]] = {
        "LambdaLR": torch.optim.lr_scheduler.LambdaLR,
        "MultiplicativeLR": torch.optim.lr_scheduler.MultiplicativeLR,
        "StepLR": torch.optim.lr_scheduler.StepLR,
        "MultiStepLR": torch.optim.lr_scheduler.MultiStepLR,
        "ConstantLR": torch.optim.lr_scheduler.ConstantLR,
        "LinearLR": torch.optim.lr_scheduler.LinearLR,
        "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
        "SequentialLR": torch.optim.lr_scheduler.SequentialLR,
        "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
        "ChainedScheduler": torch.optim.lr_scheduler.ChainedScheduler,
        "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "CyclicLR": torch.optim.lr_scheduler.CyclicLR,
        "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,  # NOQA
        "OneCycleLR": torch.optim.lr_scheduler.OneCycleLR,
        "PolynomialLR": torch.optim.lr_scheduler.PolynomialLR,
    }

    @staticmethod
    def get_registered_names() -> list[str]:
        return list(SchedulerSelector._REGISTERED.keys())

    @staticmethod
    def register(
        name: str, cls: type[torch.optim.lr_scheduler.LRScheduler]
    ) -> None:
        SchedulerSelector._REGISTERED.update({name: cls})

    @staticmethod
    def exist(name: str) -> bool:
        return name in SchedulerSelector._REGISTERED

    @staticmethod
    def select(name: str) -> type[torch.optim.lr_scheduler.LRScheduler]:
        _dict = SchedulerSelector._REGISTERED
        if name not in _dict:
            raise KeyError(
                f"{name} is not implemented as an scheduler in phlower. "
                f"Registered schedulers are {list(_dict.keys())}"
            )
        return _dict[name]
