from collections.abc import Iterator

from phlower.collections import IPhlowerTensorCollections
from phlower.data import LumpedTensorData
from phlower.settings._time_series_sliding_setting import (
    SlidingWindowForStage,
    SlidingWindowParameters,
)


class SlidingWindowHelper:
    def __init__(
        self,
        lumped_data: LumpedTensorData,
        sliding_setting: SlidingWindowForStage,
    ):
        self._lumped_data = lumped_data
        self._window_setting = sliding_setting
        self._total_size = self._deterimine_total_size()

    def _deterimine_total_size(self) -> int:
        if self._window_setting.is_active is False:
            return 1

        input_strided_size = _check_length(
            self._lumped_data.x_data,
            self._window_setting.inputs,
        )
        label_strided_size = _check_length(
            self._lumped_data.y_data,
            self._window_setting.labels,
        )

        if input_strided_size != label_strided_size:
            raise ValueError(
                "Inconsistent lengths after applying sliding window: "
                f"inputs: {input_strided_size}, labels: {label_strided_size}"
            )

        return input_strided_size

    def __len__(self) -> int:
        return self._total_size

    def __iter__(self) -> Iterator[LumpedTensorData]:
        if self._window_setting.is_active is False:
            yield self._lumped_data
            return

        for i in range(self._total_size):
            yield LumpedTensorData(
                x_data={
                    k: v.slice_time(
                        self._window_setting.inputs.get_window(k).get_slice(i),
                        keep_time_series=True,
                    )
                    if v.is_time_series
                    else v
                    for k, v in self._lumped_data.x_data.items()
                },
                y_data={
                    k: v.slice_time(
                        self._window_setting.labels.get_window(k).get_slice(i),
                        keep_time_series=True,
                    )
                    if v.is_time_series
                    else v
                    for k, v in self._lumped_data.y_data.items()
                },
                field_data=self._lumped_data.field_data,
                data_directories=self._lumped_data.data_directories,
                x_batch_info=self._lumped_data.x_batch_info,
                y_batch_info=self._lumped_data.y_batch_info,
            )


def _check_length(
    collection: IPhlowerTensorCollections,
    window_parameters: SlidingWindowParameters,
) -> int:
    _lengths: set[int] = set()
    for k, v in collection.items():
        if not v.is_time_series:
            continue

        window = window_parameters.get_window(k)
        _lengths.add(window.get_total_items(v.time_series_length))

    if len(_lengths) > 1:
        raise ValueError(
            "Inconsistent lengths after applying sliding window: " f"{_lengths}"
        )

    return _lengths.pop() if _lengths else 0
