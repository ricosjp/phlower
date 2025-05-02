import pathlib

import torch
from tqdm import tqdm

from phlower.io import PhlowerCheckpointFile
from phlower.nn import PhlowerGroupModule
from phlower.services.trainer._handlers import PhlowerHandlersRunner
from phlower.services.trainer._loggings._record_io import LogRecordIO
from phlower.services.trainer._optimizer import PhlowerOptimizerWrapper
from phlower.utils.typing import AfterEvaluationOutput


class LoggingRunner:
    def __init__(self, output_directory: pathlib.Path, log_every_n_epoch: int):
        self._output_directory = output_directory
        self._record_io = LogRecordIO(file_path=output_directory / "log.csv")
        self._log_every_n_epoch = log_every_n_epoch

    def get_weight_directory(self) -> pathlib.Path:
        return self._output_directory / "weights"

    def show_overview(self, device: torch.device) -> None:
        tqdm.write("\n-------  Start Training -------")
        tqdm.write(f"Device: {device}")
        tqdm.write(f"{torch.cuda.is_available()=}")
        tqdm.write(f"Log every {self._log_every_n_epoch} epoch")
        tqdm.write(f"Output directory: {self._output_directory}")
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
    ):
        if output.epoch % self._log_every_n_epoch != 0:
            return

        self._show_loss_to_console(output)
        self._dump_loss(output)

        self._save_checkpoint(
            output,
            model=model,
            scheduled_optimizer=scheduled_optimizer,
            handlers=handlers,
            encrypt_key=encrypt_key,
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
        model: PhlowerGroupModule,
        scheduled_optimizer: PhlowerOptimizerWrapper,
        handlers: PhlowerHandlersRunner,
        encrypt_key: bytes | None = None,
    ) -> None:
        data = {
            "epoch": info.epoch,
            "validation_loss": info.validation_eval_loss,
            "model_state_dict": model.state_dict(),
            "scheduled_optimizer": scheduled_optimizer.state_dict(),
            "elapsed_time": info.elapsed_time,
            "handler_runners": handlers.state_dict(),
        }
        prefix = PhlowerCheckpointFile.get_fixed_prefix()
        file_basename = f"{prefix}{info.epoch}"
        PhlowerCheckpointFile.save(
            output_directory=self.get_weight_directory(),
            file_basename=file_basename,
            data=data,
            encrypt_key=encrypt_key,
        )
        return
