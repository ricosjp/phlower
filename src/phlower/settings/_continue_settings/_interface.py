import abc


class IReadOnlyContinueSetting(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_stop_count(self) -> int:
        """
        The number of times to stop and update the settings.
        """
        pass


class IContinueParameter(metaclass=abc.ABCMeta):
    """
    Interface for continue settings.
    """

    @abc.abstractmethod
    def validate(self, parent: IReadOnlyContinueSetting) -> None:
        """
        Validate the setting with the parent setting.
        """
        pass
