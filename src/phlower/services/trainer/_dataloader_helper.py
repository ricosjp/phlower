import pathlib

import torch
from torch.utils.data import DataLoader, Dataset

from phlower.data import (
    DataLoaderBuilder,
    LazyPhlowerDataset,
    OnMemoryPhlowerDataSet,
)
from phlower.settings import PhlowerSetting


def prepare_dataloader(
    setting: PhlowerSetting,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    disable_dimensions: bool,
    run_distributed: bool = False,
    device: str | torch.device = "cpu",
) -> tuple[DataLoader, DataLoader | None]:
    builder = DataLoaderBuilder.from_setting(setting.training)
    train_loader = builder.create(
        train_dataset,
        device=device,
        disable_dimensions=disable_dimensions,
        run_distributed=run_distributed,
    )

    if len(validation_dataset) == 0:
        return train_loader, None

    validation_loader = builder.create(
        validation_dataset,
        device=device,
        disable_dimensions=disable_dimensions,
        run_distributed=run_distributed,
    )
    return train_loader, validation_loader


def prepare_datasets(
    setting: PhlowerSetting,
    train_directories: list[pathlib.Path],
    validation_directories: list[pathlib.Path] | None = None,
    decrypt_key: bytes | None = None,
) -> tuple[Dataset, Dataset]:
    """
    Prepare training and validation datasets based on the provided settings.

    Parameters
    ----------
    setting : PhlowerSetting
        The overall Phlower settings
    train_directories : list[pathlib.Path]
        List of directories containing training data.
    validation_directories : list[pathlib.Path] | None, optional
        List of directories containing validation data.
        If None, an empty validation dataset is created.
        Default is None.
    decrypt_key : bytes | None, optional
        Key used for decrypting data files, if necessary. Default is None.

    Returns
    -------
    tuple[Dataset, Dataset]
        A tuple containing the training dataset and validation dataset.

    """

    if setting.training.lazy_load:
        train_dataset = LazyPhlowerDataset(
            input_settings=setting.model.inputs,
            label_settings=setting.model.labels,
            field_settings=setting.model.fields,
            directories=train_directories,
            decrypt_key=decrypt_key,
        )
        validation_dataset = LazyPhlowerDataset(
            input_settings=setting.model.inputs,
            label_settings=setting.model.labels,
            field_settings=setting.model.fields,
            directories=validation_directories,
            decrypt_key=decrypt_key,
        )
    else:
        train_dataset = OnMemoryPhlowerDataSet.create(
            input_settings=setting.model.inputs,
            label_settings=setting.model.labels,
            field_settings=setting.model.fields,
            directories=train_directories,
            decrypt_key=decrypt_key,
        )
        validation_dataset = OnMemoryPhlowerDataSet.create(
            input_settings=setting.model.inputs,
            label_settings=setting.model.labels,
            field_settings=setting.model.fields,
            directories=validation_directories,
            decrypt_key=decrypt_key,
        )
    return train_dataset, validation_dataset
