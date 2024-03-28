from phlower.logging import get_logger


def test__get_logger_handlers():
    logger = get_logger("phlower.tensors._physics_tensor")
    assert logger.hasHandlers()
