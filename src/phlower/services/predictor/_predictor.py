import pathlib
from collections.abc import Iterator

from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.data import DataLoaderBuilder, LazyPhlowerDataset, LumpedTensorData
from phlower.io import PhlowerDirectory, select_snapshot_file
from phlower.nn import PhlowerGroupModule
from phlower.settings import (
    PhlowerModelSetting,
    PhlowerPredictorSetting,
    PhlowerSetting,
)


class PhlowerPredictor:
    def __init__(
        self,
        model_directory: pathlib.Path | str,
        predict_setting: PhlowerPredictorSetting,
    ):
        self._model_directory = PhlowerDirectory(model_directory)
        self._predict_setting = predict_setting

        self._model_setting = _load_model_setting(
            model_directory=self._model_directory,
            file_basename=predict_setting.saved_setting_filename,
        )
        self._model = _load_model(
            model_directory=self._model_directory,
            model_setting=self._model_setting,
            selection_mode=self._predict_setting.selection_mode,
            device=self._predict_setting.device,
        )

    def predict(
        self,
        preprocessed_directories: list[pathlib.Path],
        disable_dimensions: bool = False,
    ) -> Iterator[IPhlowerTensorCollections]:
        dataset = LazyPhlowerDataset(
            x_variable_names=self._model_setting.network.get_input_keys(),
            y_variable_names=self._model_setting.network.get_output_keys(),
            support_names=self._model_setting.network.support_names,
            directories=preprocessed_directories,
        )

        builder = DataLoaderBuilder.from_setting(self._predict_setting)
        data_loader = builder.create(
            dataset,
            variable_dimensions=self._model_setting.variable_dimensions,
            disable_dimensions=disable_dimensions,
            shuffle=False,
        )

        for batch in data_loader:
            batch: LumpedTensorData

            h = self._model.forward(
                batch.x_data, supports=batch.sparse_supports
            )
            yield h

        # HACK: Need to save h
        # HACK: Need to unbatch ?


def _load_model_setting(
    model_directory: PhlowerDirectory, file_basename: str
) -> PhlowerModelSetting:
    yaml_file = model_directory.find_yaml_file(file_base_name=file_basename)
    setting = PhlowerSetting.read_yaml(yaml_file)
    if setting.model is None:
        raise ValueError(f"model setting is not found in {yaml_file.file_path}")
    return setting.model


def _load_model(
    model_setting: PhlowerModelSetting,
    model_directory: PhlowerDirectory,
    selection_mode: str,
    device: str | None = None,
) -> PhlowerGroupModule:
    _model = PhlowerGroupModule.from_setting(model_setting.network)
    _model.load_checkpoint_file(
        checkpoint_file=select_snapshot_file(model_directory, selection_mode),
        device=device,
    )
    return _model
