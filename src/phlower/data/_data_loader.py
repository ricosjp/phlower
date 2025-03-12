from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from phlower.data._collate_fn import PhlowerCollateFn
from phlower.data._datasets import IPhlowerDataset
from phlower.settings import (
    PhlowerPredictorSetting,
    PhlowerTrainerSetting,
)


class DataLoaderBuilder:
    @classmethod
    def from_setting(
        cls, setting: PhlowerTrainerSetting | PhlowerPredictorSetting
    ) -> DataLoaderBuilder:
        return DataLoaderBuilder(
            non_blocking=setting.non_blocking,
            device=setting.device,
            random_seed=setting.random_seed,
            batch_size=setting.batch_size,
            num_workers=setting.num_workers,
        )

    def __init__(
        self,
        non_blocking: bool,
        device: str,
        random_seed: int,
        batch_size: int,
        num_workers: int,
    ) -> None:
        self._non_blocking = non_blocking
        self._device = device
        self._random_seed = random_seed
        self._batch_size = batch_size
        self._num_workers = num_workers

    def create(
        self,
        dataset: IPhlowerDataset,
        *,
        shuffle: bool = True,
        disable_dimensions: bool = False,
        drop_last: bool = False,
    ) -> DataLoader:
        _collate_fn = PhlowerCollateFn(
            device=self._device,
            non_blocking=self._non_blocking,
            disable_dimensions=disable_dimensions,
        )

        random_generator = torch.Generator()
        random_generator.manual_seed(self._random_seed)
        data_loader = DataLoader(
            dataset,
            collate_fn=_collate_fn,
            batch_size=self._batch_size,
            shuffle=shuffle,
            num_workers=self._num_workers,
            worker_init_fn=_seed_worker,
            generator=random_generator,
            drop_last=drop_last,
        )
        return data_loader


def _seed_worker(worker_id: int) -> None:
    # To ensure randomness of numpy and random module when multiprocessing
    # See https://pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading  # NOQA
    # each worker will have its PyTorch seed set to base_seed + worker_id
    # However, seeds for other libraries may be duplicated upon initializing
    # workers, causing each worker to return identical random numbers.

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
