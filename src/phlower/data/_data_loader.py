from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from phlower.data._collate_fn import PhlowerCollateFn
from phlower.data._datasets import IPhlowerDataset
from phlower.settings import (
    PhlowerPredictorSetting,
    PhlowerTrainerSetting,
)
from phlower.utils import get_logger

_logger = get_logger(__name__)


class DataLoaderBuilder:
    @classmethod
    def from_setting(
        cls, setting: PhlowerTrainerSetting | PhlowerPredictorSetting
    ) -> DataLoaderBuilder:
        return DataLoaderBuilder(
            non_blocking=setting.non_blocking,
            random_seed=setting.random_seed,
            batch_size=setting.batch_size,
            num_workers=setting.num_workers,
            pin_memory=setting.pin_memory,
        )

    def __init__(
        self,
        non_blocking: bool,
        random_seed: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
    ) -> None:
        self._non_blocking = non_blocking
        self._random_seed = random_seed
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory

    def create(
        self,
        dataset: IPhlowerDataset,
        *,
        device: str | torch.device = "cpu",
        shuffle: bool = True,
        disable_dimensions: bool = False,
        drop_last: bool = False,
        run_distributed: bool = False,
    ) -> DataLoader:
        _collate_fn = PhlowerCollateFn(
            device=device,
            non_blocking=self._non_blocking,
            disable_dimensions=disable_dimensions,
        )

        random_generator = torch.Generator()
        random_generator.manual_seed(self._random_seed)

        if not run_distributed:
            return DataLoader(
                dataset,
                collate_fn=_collate_fn,
                batch_size=self._batch_size,
                shuffle=shuffle,
                num_workers=self._num_workers,
                worker_init_fn=_seed_worker,
                generator=random_generator,
                drop_last=drop_last,
                pin_memory=self._pin_memory,
            )

        return DataLoader(
            dataset,
            collate_fn=_collate_fn,
            batch_size=self._batch_size,
            shuffle=False,  # shuffle is set by sampler
            num_workers=self._num_workers,
            worker_init_fn=_seed_worker,
            generator=random_generator,
            drop_last=drop_last,
            pin_memory=self._pin_memory,
            sampler=DistributedSampler(dataset, shuffle=shuffle),
        )


def _seed_worker(worker_id: int) -> None:
    # To ensure randomness of numpy and random module when multiprocessing
    # See https://pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading  # NOQA
    # each worker will have its PyTorch seed set to base_seed + worker_id
    # However, seeds for other libraries may be duplicated upon initializing
    # workers, causing each worker to return identical random numbers.

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
