from __future__ import annotations

import pathlib

import torch
from torch.utils.data import DataLoader

from phlower.data import (
    LumpedTensorData,
)
from phlower.nn import PhlowerGroupModule
from phlower.services.loss_operations import LossCalculator
from phlower.services.trainer._handlers import PhlowerHandlersRunner
from phlower.services.trainer._optimizer import PhlowerOptimizerWrapper
from phlower.services.trainer._sliding_window_helper import SlidingWindowHelper
from phlower.settings import (
    PhlowerTrainerSetting,
)
from phlower.utils import PhlowerProgressBar
from phlower.utils.enums import (
    PhlowerHandlerTrigger,
)
from phlower.utils.typing import (
    AfterEpochTrainingInfo,
)

from ._gather import (
    gather_loss_details_across_processes,
    gather_losses_across_processes,
)


class TrainingRunner:
    def __init__(
        self,
        trainer_setting: PhlowerTrainerSetting,
        loss_calculator: LossCalculator,
        handlers: PhlowerHandlersRunner,
    ):
        self._trainer_setting = trainer_setting
        self._loss_calculator = loss_calculator
        self._handlers = handlers

    def run(
        self,
        epoch: int,
        output_directory: pathlib.Path,
        *,
        model: PhlowerGroupModule,
        train_loader: DataLoader,
        scheduled_optimizer: PhlowerOptimizerWrapper,
        train_pbar: PhlowerProgressBar,
    ) -> AfterEpochTrainingInfo:
        train_losses: list[float] = []
        train_loss_details: list[dict[str, float]] = []
        for tr_batch in train_loader:
            tr_batch = tr_batch.to(
                device=self._trainer_setting.get_device(),
                non_blocking=self._trainer_setting.non_blocking,
            )
            train_last_loss, train_detail_losses = training_batch_step(
                self._trainer_setting,
                tr_batch,
                scheduled_optimizer,
                model,
                self._loss_calculator,
                self._handlers,
            )
            train_pbar.update(
                trick=self._trainer_setting.batch_size,
                desc=f"training loss: {train_last_loss:.3e}",
            )
            train_losses.append(train_last_loss)
            train_loss_details.append(train_detail_losses)

        scheduled_optimizer.step_scheduler()
        return AfterEpochTrainingInfo(
            epoch=epoch,
            train_losses=train_losses,
            train_loss_details=train_loss_details,
            output_directory=output_directory,
        )

    def parallel_run(
        self,
        rank: int,
        epoch: int,
        output_directory: pathlib.Path,
        *,
        model: PhlowerGroupModule,
        train_loader: DataLoader,
        scheduled_optimizer: PhlowerOptimizerWrapper,
        train_pbar: PhlowerProgressBar,
    ) -> AfterEpochTrainingInfo:
        _train_losses: list[float] = []
        _train_loss_details: list[dict[str, float]] = []

        # In DDP, each process should set epoch
        #  for sampler at the beginning of each epoch
        train_loader.sampler.set_epoch(epoch)
        device = self._trainer_setting.get_device(rank)
        model.train()
        for tr_batch in train_loader:
            tr_batch: LumpedTensorData
            tr_batch = tr_batch.to(
                device=device,
                non_blocking=self._trainer_setting.non_blocking,
            )
            train_last_loss, train_detail_losses = training_batch_step(
                self._trainer_setting,
                tr_batch,
                scheduled_optimizer,
                model,
                self._loss_calculator,
                self._handlers,
            )
            train_pbar.update(
                trick=self._trainer_setting.batch_size,
                desc=f"training loss (GPU Rank: {rank}): {train_last_loss:.3e}",
            )

            _train_losses.append(train_last_loss)
            _train_loss_details.append(train_detail_losses)

        train_losses = gather_losses_across_processes(_train_losses)
        train_loss_details = gather_loss_details_across_processes(
            _train_loss_details
        )

        scheduled_optimizer.step_scheduler()
        return AfterEpochTrainingInfo(
            epoch=epoch,
            train_losses=train_losses,
            train_loss_details=train_loss_details,
            output_directory=output_directory,
        )


def training_batch_step(
    trainer_setting: PhlowerTrainerSetting,
    tr_batch: LumpedTensorData,
    scheduled_optimizer: PhlowerOptimizerWrapper,
    model: torch.nn.Module,
    loss_calculator: LossCalculator,
    handlers: PhlowerHandlersRunner,
) -> tuple[float, dict[str, float]]:
    helper = SlidingWindowHelper(
        tr_batch,
        trainer_setting.time_series_sliding.training_window_settings,
    )
    assert len(helper) > 0, "No sliding windows are generated."
    for _slided_batch in helper:
        last_loss, detached_losses = _training_batch_step_w_slide(
            _slided_batch,
            scheduled_optimizer,
            model,
            loss_calculator,
        )
        handlers.run(
            last_loss, trigger=PhlowerHandlerTrigger.iteration_completed
        )
    return last_loss, detached_losses


def _training_batch_step_w_slide(
    tr_batch: LumpedTensorData,
    scheduled_optimizer: PhlowerOptimizerWrapper,
    model: torch.nn.Module,
    loss_calculator: LossCalculator,
) -> tuple[float, dict[str, float]]:
    scheduled_optimizer.zero_grad()

    h = model.forward(tr_batch.x_data, field_data=tr_batch.field_data)

    losses = loss_calculator.calculate(
        h, tr_batch.y_data, batch_info_dict=tr_batch.y_batch_info
    )
    detached_losses = {k: v.item() for k, v in losses.to_numpy().items()}
    loss = loss_calculator.aggregate(losses)
    loss.backward()

    scheduled_optimizer.step_optimizer()
    scheduled_optimizer.zero_grad()
    _last_loss = loss.detach().to_tensor().float().item()

    del loss
    # NOTE: This is necessary to use less memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return _last_loss, detached_losses
