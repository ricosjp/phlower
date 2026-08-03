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
from phlower.services.utils import SlidingWindowHelper
from phlower.settings import (
    PhlowerTrainerSetting,
)
from phlower.utils import PhlowerProgressBar
from phlower.utils.calculation_state import CalculationState
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
        self._batch_step_runner = _BatchStepRunner(
            gradient_accumulation_steps=self._trainer_setting.gradient_accumulation_steps
        )

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
        for idx, tr_batch in enumerate(train_loader):
            state = CalculationState(
                mode="training",
                current_epoch=epoch,
                current_batch_iteration=idx,
                output_directory=output_directory,
            )
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
                self._batch_step_runner,
                state=state,
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
        for idx, tr_batch in enumerate(train_loader):
            state = CalculationState(
                mode="training",
                current_epoch=epoch,
                current_batch_iteration=idx,
                output_directory=output_directory,
            )
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
                self._batch_step_runner,
                state=state,
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
    batch_step_runner: _BatchStepRunner,
    state: CalculationState | None = None,
) -> tuple[float, dict[str, float]]:
    helper = SlidingWindowHelper(
        tr_batch,
        trainer_setting.time_series_sliding.training_window_settings,
    )
    assert len(helper) > 0, "No sliding windows are generated."
    for _slided_batch in helper:
        last_loss, detached_losses = batch_step_runner.run(
            _slided_batch,
            scheduled_optimizer,
            model,
            loss_calculator,
            state=state,
        )
        handlers.run(
            last_loss, trigger=PhlowerHandlerTrigger.iteration_completed
        )
    return last_loss, detached_losses


# region Batch Step


class _UpdateTimingCounter:
    def __init__(self, n_size: int):
        self._count = 0
        assert n_size > 0, "n_size must be greater than 0."
        self._n_size = n_size

    def reset(self) -> None:
        self._count = 0

    def increment(self) -> None:
        self._count += 1
        self._count %= self._n_size

    @property
    def n_size(self) -> int:
        return self._n_size

    @property
    def count(self) -> int:
        return self._count

    @property
    def is_full(self) -> bool:
        return (self._count + 1) % self._n_size == 0


class _BatchStepRunner:
    def __init__(self, gradient_accumulation_steps: int = 1):
        self._update_counter = _UpdateTimingCounter(gradient_accumulation_steps)

    def run(
        self,
        tr_batch: LumpedTensorData,
        scheduled_optimizer: PhlowerOptimizerWrapper,
        model: PhlowerGroupModule | torch.nn.Module,
        loss_calculator: LossCalculator,
        *,
        state: CalculationState | None = None,
    ) -> tuple[float, dict[str, float]]:

        if self._update_counter.count == 0:
            scheduled_optimizer.zero_grad()

        h = model.forward(
            tr_batch.x_data, field_data=tr_batch.field_data, state=state
        )

        losses = loss_calculator.calculate(
            h, tr_batch.y_data, batch_info_dict=tr_batch.y_batch_info
        )
        loss = loss_calculator.aggregate(losses)
        (loss / self._update_counter.n_size).backward()

        if self._update_counter.is_full:
            scheduled_optimizer.step_optimizer()

            # NOTE: This is necessary to use less memory
            scheduled_optimizer.zero_grad()

        _detached_losses = {k: v.item() for k, v in losses.to_numpy().items()}
        _last_loss = loss.detach().to_tensor().float().item()

        del loss
        # NOTE: This is necessary to use less memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if isinstance(model, PhlowerGroupModule):
            # When the model is DistributedDataParallel,
            # finalize_debug cannnot be called.
            model.finalize_debug()

        self._update_counter.increment()
        return _last_loss, _detached_losses
