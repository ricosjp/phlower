from __future__ import annotations

import pathlib

from pipe import select, where

from phlower.services.trainer.logging_items import ILoggingItem, create_logitems


class LogRecord:
    def __init__(
        self,
        *,
        epoch: int,
        train_loss: float,
        validation_loss: float,
        elapsed_time: float,
        train_loss_details: dict[str, float] | None = None,
        validation_loss_details: dict[str, float] | None = None,
    ) -> None:
        self.epoch = create_logitems(int(epoch), "epoch")
        self.train_loss = create_logitems(train_loss, "train_loss")
        self.validation_loss = create_logitems(
            validation_loss, "validation_loss"
        )
        self.elapsed_time = create_logitems(elapsed_time, "elapsed_time")
        self.train_loss_details = create_logitems(
            train_loss_details, "train_loss_details/", default_factory=dict
        )
        self.validation_loss_details = create_logitems(
            validation_loss_details,
            "validation_loss_details/",
            default_factory=dict,
        )

        self._detail_keys = []
        if train_loss_details is not None:
            self._detail_keys = list(train_loss_details.keys())

    def get_detail_keys(self) -> list[str]:
        return self._detail_keys


class LogRecordIO:
    def __init__(
        self,
        file_path: pathlib.Path,
        loss_keys: list[str] = None,
        display_margin: int = 4,
    ):
        self._file_path = file_path
        self._display_margin = display_margin
        self._loss_keys = loss_keys if loss_keys is not None else []
        self._train_details_title = r"details_tr/"
        self._validation_details_title = r"details_val/"
        self._headers = self._get_headers()

    def _get_headers(self) -> list[ILoggingItem]:
        headers = [
            "epoch",
            "train_loss",
            "validation_loss",
            "elapsed_time",
            *[f"{self._train_details_title}{k}" for k in self._loss_keys],
            *[f"{self._validation_details_title}{k}" for k in self._loss_keys],
        ]
        headers = [create_logitems(v) for v in headers]
        return headers

    def get_header(self) -> str:
        strings = [
            v.format(padding_margin=self._display_margin) for v in self._headers
        ]
        return "\n" + "".join(strings)

    def to_str(self, log_record: LogRecord) -> str:
        strings = [
            log_record.epoch.format(padding_margin=self._display_margin),
            log_record.train_loss.format(
                formatter=".5e", padding_margin=self._display_margin
            ),
            log_record.train_loss_details.format(
                formatter=".5e",
                key_orders=self._loss_keys,
                padding_margin=self._display_margin,
                title=self._train_details_title,
            ),
            log_record.validation_loss.format(
                formatter=".5e", padding_margin=self._display_margin
            ),
            log_record.validation_loss_details.format(
                formatter=".5e",
                key_orders=self._loss_keys,
                padding_margin=self._display_margin,
                title=self._validation_details_title,
            ),
            log_record.elapsed_time.format(
                formatter=".2f", padding_margin=self._display_margin
            ),
        ]
        return "".join(strings)

    def _header_strings(self) -> str:
        headers = [v.format() for v in self._headers]
        headers = [v for v in headers if len(v) > 0]
        return ", ".join(headers)

    def write_header(self) -> None:
        headers = list(
            self._headers
            | select(lambda x: x.format())
            | where(lambda x: len(x) > 0)
        )

        header_str = ",".join(headers)
        with open(self._file_path, "w") as fw:
            fw.write(header_str + "\n")

    def write(self, log_record: LogRecord) -> None:
        values = [
            log_record.epoch.format(),
            log_record.train_loss.format(formatter=".5e"),
            log_record.train_loss_details.format(
                formatter=".5e", key_orders=self._loss_keys
            ),
            log_record.validation_loss.format(formatter=".5e"),
            log_record.validation_loss_details.format(
                formatter=".5e", key_orders=self._loss_keys
            ),
            log_record.elapsed_time.format(formatter=".2f"),
        ]
        values = [v for v in values if len(v) > 0]
        with open(self._file_path, "a") as fw:
            fw.write(",".join(values) + "\n")
