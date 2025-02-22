from __future__ import annotations

import pathlib
from collections.abc import Iterator
from typing import overload

from torch.utils.data import DataLoader

from phlower._base import IPhlowerArray
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.data import DataLoaderBuilder, LazyPhlowerDataset, LumpedTensorData
from phlower.io import PhlowerDirectory, select_snapshot_file
from phlower.nn import PhlowerGroupModule
from phlower.services.preprocessing import PhlowerScalingService
from phlower.settings import (
    PhlowerModelSetting,
    PhlowerPredictorSetting,
    PhlowerSetting,
)
from phlower.utils.typing import (
    PhlowerInverseScaledPredictionResult,
    PhlowerPredictionResult,
)


class PhlowerPredictor:
    @classmethod
    def from_pathes(
        cls,
        model_directory: pathlib.Path | str,
        predict_setting_yaml: pathlib.Path,
        scaling_setting_yaml: pathlib.Path | None = None,
        decrypt_key: bytes | None = None,
    ) -> PhlowerPredictor:
        predict_setting = PhlowerSetting.read_yaml(
            predict_setting_yaml, decrypt_key=decrypt_key
        )

        if scaling_setting_yaml is None:
            return PhlowerPredictor(
                model_directory=model_directory,
                predict_setting=predict_setting.prediction,
            )

        scaling_setting = PhlowerSetting.read_yaml(
            scaling_setting_yaml, decrypt_key=decrypt_key
        )
        return PhlowerPredictor(
            model_directory=model_directory,
            predict_setting=predict_setting.prediction,
            scaling_setting=scaling_setting,
        )

    def __init__(
        self,
        model_directory: pathlib.Path | str,
        predict_setting: PhlowerPredictorSetting,
        scaling_setting: PhlowerSetting | None = None,
    ):
        self._model_directory = PhlowerDirectory(model_directory)
        self._predict_setting = predict_setting

        if scaling_setting is not None:
            self._scalers = PhlowerScalingService.from_setting(scaling_setting)
        else:
            self._scalers = None

        self._model_setting = _load_model_setting(
            model_directory=self._model_directory,
            file_basename=predict_setting.saved_setting_filename,
        )
        # NOTE: it is necessary to resolve information of modules
        #  which need reference module.
        # Resolving cost is O(N). N is the number of modules
        self._model_setting.resolve()

        self._model = _load_model(
            model_directory=self._model_directory,
            model_setting=self._model_setting,
            selection_mode=self._predict_setting.selection_mode,
            device=self._predict_setting.device,
            target_epoch=self._predict_setting.target_epoch,
        )

    @overload
    def predict(
        self,
        preprocessed_directories: list[pathlib.Path],
        perform_inverse_scaling: False,
        disable_dimensions: bool = False,
        decrypt_key: bytes | None = None,
        return_only_prediction: bool = False,
    ) -> Iterator[PhlowerPredictionResult]: ...

    @overload
    def predict(
        self,
        preprocessed_directories: list[pathlib.Path],
        perform_inverse_scaling: True,
        disable_dimensions: bool = False,
        decrypt_key: bytes | None = None,
        return_only_prediction: bool = False,
    ) -> Iterator[PhlowerInverseScaledPredictionResult]: ...

    def predict(
        self,
        preprocessed_directories: list[pathlib.Path],
        perform_inverse_scaling: bool,
        disable_dimensions: bool = False,
        decrypt_key: bytes | None = None,
    ) -> Iterator[IPhlowerTensorCollections | dict[str, IPhlowerArray]]:
        dataset = LazyPhlowerDataset(
            input_settings=self._model_setting.inputs,
            label_settings=self._model_setting.labels,
            field_settings=self._model_setting.fields,
            directories=preprocessed_directories,
            decrypt_key=decrypt_key,
        )

        builder = DataLoaderBuilder.from_setting(self._predict_setting)
        data_loader = builder.create(
            dataset,
            disable_dimensions=disable_dimensions,
            shuffle=False,
        )

        if not perform_inverse_scaling:
            yield from self._predict(data_loader)
            return

        if self._scalers is None:
            raise ValueError(
                "scaler are not defined. "
                "Please check that your scaling setting"
                " is inputted when initializing this class."
            )
        yield from self._predict_with_inverse(data_loader)
        return

    def _predict(
        self, data_loader: DataLoader, return_only_prediction: bool = False
    ) -> Iterator[PhlowerPredictionResult]:
        for batch in data_loader:
            batch: LumpedTensorData

            # HACK: Need to unbatch ?
            assert batch.n_data == 1
            h = self._model.forward(batch.x_data, field_data=batch.field_data)

            if return_only_prediction:
                yield PhlowerPredictionResult(prediction_data=h)
            else:
                yield PhlowerPredictionResult(
                    input_data=batch.x_data,
                    prediction_data=h,
                    answer_data=batch.y_data,
                )

    def _predict_with_inverse(
        self, data_loader: DataLoader, return_only_prediction: bool = False
    ) -> Iterator[dict[str, IPhlowerArray]]:
        for batch in data_loader:
            batch: LumpedTensorData

            h = self._model.forward(batch.x_data, field_data=batch.field_data)

            preds = self._scalers.inverse_transform(
                h.to_phlower_arrays_dict(), raise_missing_message=True
            )
            if return_only_prediction:
                yield PhlowerInverseScaledPredictionResult(
                    prediction_data=preds
                )
            else:
                x_data = self._scalers.inverse_transform(
                    batch.x_data.to_phlower_arrays_dict(),
                    raise_missing_message=True,
                )
                answer_data = self._scalers.inverse_transform(batch.y_data)
                yield PhlowerInverseScaledPredictionResult(
                    prediction_data=preds,
                    input_data=x_data,
                    answer_data=answer_data,
                )


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
    target_epoch: int | None = None,
) -> PhlowerGroupModule:
    _model = PhlowerGroupModule.from_setting(model_setting.network)
    _model.load_checkpoint_file(
        checkpoint_file=select_snapshot_file(
            model_directory, selection_mode, target_epoch=target_epoch
        ),
        device=device,
    )
    if device is not None:
        _model.to(device)
    return _model
