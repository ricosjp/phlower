from phlower.utils.enums import PhlowerScalerName


def get_registered_scaler_names() -> list[str]:
    names = [v for v in PhlowerScalerName.__members__]
    return names
