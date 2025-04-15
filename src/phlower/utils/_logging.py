import logging


class DefaultLoggerFactory:
    _is_propagate: bool = True

    @classmethod
    def _get_phlower_root_logger(cls) -> logging.Logger:
        return logging.getLogger("phlower")

    @classmethod
    def _get_library_logger(cls) -> logging.Logger:
        _phlower_root_logger = logging.getLogger("phlower")
        _phlower_root_logger.propagate = cls._is_propagate
        return _phlower_root_logger

    @classmethod
    def _initialize_handler(cls):
        _phlower_root_logger = cls._get_library_logger()
        if _phlower_root_logger.hasHandlers():
            # handlers have already set.
            return

        formatter = logging.Formatter(
            r"[%(name)s] [%(asctime)s] [%(levelname)s] "
            r"[%(funcName)s] %(message)s"
        )

        _handler = logging.StreamHandler()
        _handler.setLevel(logging.WARNING)
        _handler.setFormatter(formatter)

        _phlower_root_logger.addHandler(_handler)
        return


def get_phlower_logger() -> logging.Logger:
    return DefaultLoggerFactory._get_library_logger()


def set_debug_logger_properties():
    """
    This method is designed to set debug logger properties.
    It is used in the test code or debug code for Phlower itself.
    """
    DefaultLoggerFactory._initialize_handler()
    return


def get_logger(name: str) -> logging.Logger:
    """This is designed to get logger inside phlower library itself.
    If you are user of this library and want to get library root logger,
    use 'get_phlower_logger' instead.

    Parameters
    ----------
    name : str
        name of logger

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    return logger
