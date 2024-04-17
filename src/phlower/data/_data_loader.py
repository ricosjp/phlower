import pathlib
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from phlower.data._collate_fn import PhlowerCollateFn
from phlower.data._datasets import IPhlowerDataset
from phlower.settings._phlower_setting import PhlowerTrainerSetting


class DataLoaderBuilder:
    def __init__(self, setting: PhlowerTrainerSetting) -> None:
        self._setting = setting
        self._collate_fn = PhlowerCollateFn(
            device=self._setting.device,
            non_blocking=self._setting.non_blocking,
            dimensions=setting.variable_dimensions
        )

    def create(
        self,
        dataset: IPhlowerDataset,
        *,
        shuffle: bool = True,
    ) -> DataLoader:

        random_generator = torch.Generator()
        random_generator.manual_seed(self._setting.random_seed)
        data_loader = DataLoader(
            dataset,
            collate_fn=self._collate_fn,
            batch_size=self._setting.batch_size,
            shuffle=shuffle,
            num_workers=self._setting.num_workers,
            worker_init_fn=_seed_worker,
            generator=random_generator,
        )
        return data_loader


def _seed_worker(worker_id: int) -> None:
    # To ensure randomness of numpy and random module when multiprocessing
    # See https://pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
