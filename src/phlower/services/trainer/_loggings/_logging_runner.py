import pathlib

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from phlower.io import PhlowerCheckpointFile
from phlower.nn import PhlowerGroupModule
from phlower.services.trainer._handlers import PhlowerHandlersRunner
from phlower.services.trainer._loggings._record_io import LogRecordIO
from phlower.services.trainer._optimizer import PhlowerOptimizerWrapper
from phlower.utils.typing import AfterEvaluationOutput


class LoggingRunner:
    def __init__(
        self,
        output_directory: pathlib.Path,
        log_every_n_epoch: int,
        loss_keys: list[str],
        parallel_mode: str = "None",
    ):
        self._output_directory = output_directory
        self._record_io = LogRecordIO(
            file_path=output_directory / "log.csv", loss_keys=loss_keys
        )
        self._log_every_n_epoch = log_every_n_epoch
        self._parallel_mode = parallel_mode

    def get_weight_directory(self) -> pathlib.Path:
        return self._output_directory / "weights"

    def show_overview(self, device: torch.device) -> None:
        tqdm.write("\n-------  Start Training -------")
        tqdm.write(f"Device: {device}")
        tqdm.write(f"{torch.cuda.is_available()=}")
        tqdm.write(f"Log every {self._log_every_n_epoch} epoch")
        tqdm.write(f"Output directory: {self._output_directory}")
        if dist.is_initialized():
            tqdm.write(f"World size: {dist.get_world_size()}")
            tqdm.write(f"Rank: {dist.get_rank()}")
            tqdm.write(f"Parallel type: {self._parallel_mode}")
        tqdm.write("-------------------------------")
        tqdm.write(self._record_io.get_header())

    def run(
        self,
        output: AfterEvaluationOutput,
        *,
        model: PhlowerGroupModule,
        scheduled_optimizer: PhlowerOptimizerWrapper,
        handlers: PhlowerHandlersRunner,
        encrypt_key: bytes | None = None,
        fsdp_sharded: bool = False,
        rank: int = 0,
    ):
        if output.epoch % self._log_every_n_epoch != 0:
            return

        if rank == 0:
            self._show_loss_to_console(output)
            self._dump_loss(output)

        self._save_checkpoint(
            output,
            model=model,
            scheduled_optimizer=scheduled_optimizer,
            handlers=handlers,
            encrypt_key=encrypt_key,
            fsdp_sharded=fsdp_sharded,
            rank=rank,
        )

        return

    def _show_loss_to_console(
        self,
        info: AfterEvaluationOutput,
    ) -> None:
        tqdm.write(self._record_io.to_str(info))
        return

    def _dump_loss(
        self,
        info: AfterEvaluationOutput,
    ) -> None:
        if not self._record_io.file_path.exists():
            self._record_io.write_header()

        # Dumo to file
        self._record_io.write(info)
        return

    def _save_checkpoint(
        self,
        info: AfterEvaluationOutput,
        *,
        model: PhlowerGroupModule | DistributedDataParallel,
        scheduled_optimizer: PhlowerOptimizerWrapper,
        handlers: PhlowerHandlersRunner,
        encrypt_key: bytes | None = None,
        fsdp_sharded: bool = False,
        rank: int = 0,
    ) -> None:
        if isinstance(model, DistributedDataParallel):
            model = model.module

        data = {
            "epoch": info.epoch,
            "validation_loss": info.validation_eval_loss,
            "model_state_dict": _get_model_state_dict(
                model, fsdp_sharded=fsdp_sharded
            ),
            # "scheduled_optimizer": scheduled_optimizer.state_dict(),
            "elapsed_time": info.elapsed_time,
            "handler_runners": handlers.state_dict(),
        }

        if rank == 0:
            prefix = PhlowerCheckpointFile.get_fixed_prefix()
            file_basename = f"{prefix}{info.epoch}"
            PhlowerCheckpointFile.save(
                output_directory=self.get_weight_directory(),
                file_basename=file_basename,
                data=data,
                encrypt_key=encrypt_key,
            )
        return


def _get_model_state_dict(
    model: PhlowerGroupModule
    | DistributedDataParallel
    | FullyShardedDataParallel,
    fsdp_sharded: bool = True,
) -> dict[str, torch.Tensor]:
    if not fsdp_sharded:
        if isinstance(model, DistributedDataParallel):
            model = model.module
        return model.state_dict()

    model_state_dict = get_model_state_dict(
        model=model,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        ),
    )
    return model_state_dict
