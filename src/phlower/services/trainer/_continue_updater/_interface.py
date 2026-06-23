import abc

from phlower.settings._trainer_setting import PhlowerTrainerSetting


class IContinueParameterUpdator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update_parameters(
        self,
        prev_setting: PhlowerTrainerSetting,
        continue_count: int,
    ) -> PhlowerTrainerSetting:
        """
        update trainer setting for next training.
        Args:
            prev_setting (PhlowerTrainerSetting): previous trainer setting
            continue_count (int): current continue count. Starts from 1.
        Returns:
            PhlowerTrainerSetting: updated trainer setting for next training
        """
        ...

    @abc.abstractmethod
    def get_output_directory_name_suffix(
        self, prev_setting: PhlowerTrainerSetting, continue_count: int
    ) -> str: ...
