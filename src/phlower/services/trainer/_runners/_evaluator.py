from __future__ import annotations

from typing import overload

import numpy as np
import torch
from torch.utils.data import DataLoader

from phlower.nn import PhlowerGroupModule
from phlower.services.loss_operations import LossCalculator
from phlower.services.trainer._handlers import PhlowerHandlersRunner
from phlower.services.trainer._sliding_window_helper import SlidingWindowHelper
from phlower.settings import (
    PhlowerTrainerSetting,
)
from phlower.utils import PhlowerProgressBar, StopWatch
from phlower.utils.enums import (
    PhlowerHandlerTrigger,
)
from phlower.utils.typing import (
    AfterEpochTrainingInfo,
    AfterEvaluationOutput,
)

from ._gather import (
    gather_loss_details_across_processes,
    gather_losses_across_processes,
)


class EvaluationRunner:
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
        rank: int,
        info: AfterEpochTrainingInfo,
        *,
        model: PhlowerGroupModule,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        train_pbar: PhlowerProgressBar,
        validation_pbar: PhlowerProgressBar,
        timer: StopWatch,
    ) -> AfterEvaluationOutput:
        if not self._trainer_setting.evaluation_for_training:
            train_eval_loss = np.average(info.train_losses)
            train_loss_details = _aggregate_loss_details(
                info.train_loss_details
            )
        else:
            train_eval_loss, train_loss_details = self._evaluation(
                rank=rank,
                model=model,
                data_loader=train_loader,
                loss_function=self._loss_calculator,
                pbar=train_pbar,
                pbar_title="batch train loss",
            )

        if validation_loader is None:
            validation_eval_loss = None
            validation_loss_details = None
        else:
            validation_eval_loss, validation_loss_details = self._evaluation(
                rank=rank,
                model=model,
                data_loader=validation_loader,
                loss_function=self._loss_calculator,
                pbar=validation_pbar,
                pbar_title="batch val loss",
            )

        return AfterEvaluationOutput(
            epoch=info.epoch,
            train_eval_loss=train_eval_loss,
            validation_eval_loss=validation_eval_loss,
            elapsed_time=timer.watch() if timer is not None else None,
            output_directory=info.output_directory,
            train_loss_details=train_loss_details,
            validation_loss_details=validation_loss_details,
        )

    def parallel_run(
        self,
        rank: int,
        info: AfterEpochTrainingInfo,
        *,
        model: PhlowerGroupModule,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        train_pbar: PhlowerProgressBar,
        validation_pbar: PhlowerProgressBar,
        timer: StopWatch | None,
    ) -> AfterEvaluationOutput:
        if not self._trainer_setting.evaluation_for_training:
            train_eval_loss = np.average(info.train_losses)
            train_loss_details = _aggregate_loss_details(
                info.train_loss_details
            )
        else:
            train_eval_loss, train_loss_details = self._parallel_evaluation(
                epoch=info.epoch,
                model=model,
                data_loader=train_loader,
                loss_function=self._loss_calculator,
                pbar=train_pbar,
                pbar_title=f"batch train loss (GPU rank:{rank})",
                rank=rank,
            )

        if validation_loader is None:
            validation_eval_loss = None
            validation_loss_details = None
        else:
            validation_eval_loss, validation_loss_details = (
                self._parallel_evaluation(
                    epoch=info.epoch,
                    model=model,
                    data_loader=validation_loader,
                    loss_function=self._loss_calculator,
                    pbar=validation_pbar,
                    pbar_title=f"batch val loss (GPU rank:{rank})",
                    rank=rank,
                )
            )

        return AfterEvaluationOutput(
            epoch=info.epoch,
            train_eval_loss=train_eval_loss,
            validation_eval_loss=validation_eval_loss,
            elapsed_time=timer.watch() if timer is not None else None,
            output_directory=info.output_directory,
            train_loss_details=train_loss_details,
            validation_loss_details=validation_loss_details,
        )

    def _evaluation(
        self,
        rank: int,
        model: PhlowerGroupModule,
        data_loader: DataLoader,
        loss_function: LossCalculator,
        pbar: PhlowerProgressBar,
        pbar_title: str,
    ) -> tuple[float, dict[str, float] | None]:
        results: list[float] = []
        results_details: list[dict[str, np.ndarray]] = []

        model.eval()

        for batch in data_loader:
            batch = batch.to(
                device=self._trainer_setting.get_device(rank),
                non_blocking=self._trainer_setting.non_blocking,
            )
            helper = SlidingWindowHelper(
                batch,
                self._trainer_setting.time_series_sliding.validation_window_settings,
            )
            _loss_list, _loss_details = _evaluation_batch_step(
                helper,
                model,
                loss_function,
                self._handlers,
            )
            results.extend(_loss_list)
            results_details.extend(_loss_details)

            pbar.update(
                trick=batch.n_data,
                desc=f"{pbar_title}: {results[-1]:.3e}",
            )
        return np.average(results), _aggregate_loss_details(results_details)

    def _determine_context_manager(
        self,
    ) -> type[torch.inference_mode] | type[torch.no_grad]:
        match self._trainer_setting.evaluate_context_manager:
            case "inference_mode":
                return torch.inference_mode
            case "no_grad":
                return torch.no_grad
            case _:
                raise ValueError(
                    "Unknown evaluate_context_manager: "
                    f"{self._trainer_setting.evaluate_context_manager}"
                )

    def _parallel_evaluation(
        self,
        epoch: int,
        model: PhlowerGroupModule,
        data_loader: DataLoader,
        loss_function: LossCalculator,
        pbar: PhlowerProgressBar,
        pbar_title: str,
        rank: int,
    ) -> tuple[float, dict[str, float] | None]:
        results: list[float] = []
        results_details: list[dict[str, np.ndarray]] = []

        model.eval()

        context_manager = self._determine_context_manager()
        with context_manager():
            # It is necessary to set epoch for each process
            # when using DistributedSampler
            data_loader.sampler.set_epoch(epoch)
            device = self._trainer_setting.get_device(rank)
            for batch in data_loader:
                batch = batch.to(
                    device=device,
                    non_blocking=self._trainer_setting.non_blocking,
                )
                helper = SlidingWindowHelper(
                    batch,
                    self._trainer_setting.time_series_sliding.validation_window_settings,
                )

                _loss_list, _loss_details = _evaluation_batch_step(
                    helper,
                    model,
                    loss_function,
                    self._handlers,
                )

                pbar.update(
                    trick=batch.n_data,
                    desc=f"{pbar_title}: {_loss_list[-1]:.3e}",
                )

                # Communicate losses from all processes
                # It is implicitly assumed that all processes
                #  have the same number of time sliding windows
                gathered_loss = gather_losses_across_processes(_loss_list)
                gathered_details = gather_loss_details_across_processes(
                    _loss_details
                )

                if rank == 0:
                    results.extend(gathered_loss)
                    results_details.extend(gathered_details)

        return np.average(results), _aggregate_loss_details(results_details)


@overload
def _evaluation_batch_step(
    sliding_window_helper: SlidingWindowHelper,
    model: torch.nn.Module,
    loss_calculator: LossCalculator,
    handlers: PhlowerHandlersRunner,
) -> tuple[list[float], dict[str, float] | None]: ...


def _evaluation_batch_step(
    sliding_window_helper: SlidingWindowHelper,
    model: torch.nn.Module,
    loss_calculator: LossCalculator,
    handlers: PhlowerHandlersRunner,
) -> tuple[list[float], dict[str, float] | None]:
    results: list[float] = []
    results_details: list[dict[str, np.ndarray]] = []

    for _batch in sliding_window_helper:
        h = model.forward(_batch.x_data, field_data=_batch.field_data)
        val_losses = loss_calculator.calculate(
            h,
            _batch.y_data,
            batch_info_dict=_batch.y_batch_info,
        )
        detached_val_loss = (
            loss_calculator.aggregate(val_losses)
            .detach()
            .to_tensor()
            .float()
            .item()
        )
        handlers.run(
            detached_val_loss,
            trigger=PhlowerHandlerTrigger.iteration_completed,
        )
        results.append(detached_val_loss)
        results_details.append(val_losses.to_numpy())
    return results, results_details


def _aggregate_loss_details(
    loss_details: list[dict[str, np.ndarray]],
) -> dict[str, float]:
    """Aggregate loss details from list of loss details

    Args:
        loss_details (list[dict[str, float]]): List of loss details

    Returns:
        dict[str, float]: Aggregated loss details
    """
    if len(loss_details) == 0:
        return {}

    assert all(len(v) == len(loss_details[0]) for v in loss_details)
    keys = loss_details[0].keys()
    aggregated = {k: np.mean([v[k] for v in loss_details]).item() for k in keys}

    return aggregated
