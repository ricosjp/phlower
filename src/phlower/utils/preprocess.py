from phlower.utils.enums import PhlowerScalerName


def get_registered_scaler_names() -> list[str]:
    return list(PhlowerScalerName.__members__.keys())
