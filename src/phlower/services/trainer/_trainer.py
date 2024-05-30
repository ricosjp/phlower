import pathlib

import torch

from phlower._base import PhlowerTensor
from phlower.data import DataLoaderBuilder, LazyPhlowerDataset, LumpedTensorData
from phlower.nn import PhlowerGroupModule
from phlower.services.loss_operations import LossCalculator
from phlower.settings import PhlowerSetting
from phlower.utils import get_logger

_logger = get_logger(__name__)


class PhlowerTrainer:
    def __init__(self, setting: PhlowerSetting):
        self._setting = setting

    def train(
        self,
        preprocessed_directories: list[pathlib.Path],
        n_epoch: int,
        disable_dimensions: bool = False,
    ) -> tuple[PhlowerGroupModule, PhlowerTensor]:

        dataset = LazyPhlowerDataset(
            x_variable_names=self._setting.model.get_input_keys(),
            y_variable_names=self._setting.model.get_output_keys(),
            support_names=self._setting.model.support_names,
            directories=preprocessed_directories,
        )

        builder = DataLoaderBuilder.from_setting(self._setting.trainer)
        data_loader = builder.create(
            dataset, disable_dimensions=disable_dimensions
        )

        model = PhlowerGroupModule.from_setting(self._setting.model)

        loss_function = LossCalculator.from_setting(self._setting.trainer)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        loss: PhlowerTensor | None = None
        for _ in range(n_epoch):
            for batch in data_loader:
                batch: LumpedTensorData
                optimizer.zero_grad()

                h = model.forward(batch.x_data, supports=batch.sparse_supports)

                losses = loss_function.calculate(h, batch.y_data)
                loss = loss_function.aggregate(losses)
                loss.backward()
                optimizer.step()

        return model, loss
