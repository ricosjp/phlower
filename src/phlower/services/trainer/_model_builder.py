import pathlib

import torch

from phlower.io import PhlowerFileBuilder
from phlower.nn import PhlowerGroupModule
from phlower.settings import PhlowerModelSetting
from phlower.utils import get_logger

_logger = get_logger(__name__)


class ModelBuilder:
    def __init__(
        self, model_setting: PhlowerModelSetting, device: str | torch.device
    ) -> None:
        self._model_setting = model_setting
        self._device = device

    def _create_initialized(self) -> PhlowerGroupModule:
        _model = PhlowerGroupModule.from_setting(self._model_setting.network)
        _model.to(self._device)
        return _model

    def _create_loaded(
        self, checkpoint_path: pathlib.Path, decrypt_key: bytes | None = None
    ) -> PhlowerGroupModule:
        """Create network object loaded from checkpoint file

        Parameters
        ----------
        checkpoint_file : pathlib.Path
            Path to checkpoint file
        decrypt_key : Optional[bytes], optional
            decrpt byte key, by default None

        Returns
        -------
        PhlowerGroupModule
            network object
        """
        checkpoint = PhlowerFileBuilder.checkpoint_file(checkpoint_path)
        _logger.info(f"Load snapshot file: {checkpoint.file_path}")

        model = self._create_initialized()
        content = checkpoint.load(device=self._device, decrypt_key=decrypt_key)
        model.load_state_dict(content["model_state_dict"])
        return model
