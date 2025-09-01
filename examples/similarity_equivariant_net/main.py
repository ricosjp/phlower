import pathlib
import shutil

import numpy as np
import pandas as pd
import phlower
from phlower.services.trainer import PhlowerTrainer


def main():
    data_directory = pathlib.Path("data")
    output_directory = data_directory / "model"
    if data_directory.exists():
        shutil.rmtree(data_directory)

    setting = phlower.settings.PhlowerSetting.read_yaml(
        "similarity_equivariant_mlp.yml"
    )

    generate_data(setting.data.training[0], setting.data.validation[0])

    trainer = PhlowerTrainer.from_setting(setting)

    trainer.train(
        output_directory=output_directory,
    )

    df = pd.read_csv(output_directory / "log.csv")

    train_loss = df.loc[:, "train_loss"].to_numpy()
    validation_loss = df.loc[:, "validation_loss"].to_numpy()

    l_scale = np.load(setting.data.validation[0] / "l_scale.npy")
    t_scale = np.load(setting.data.validation[0] / "t_scale.npy")

    # Test scaling equivariance
    scale = np.squeeze(l_scale**2 / t_scale**2)
    print(f"Train loss: {train_loss[-1]:.5e}")
    print(f"Scaled validation loss: {validation_loss[-1] / scale:.5e}")
    np.testing.assert_array_almost_equal(
        train_loss,
        validation_loss / scale,
        decimal=5,
    )


def generate_data(
    reference_directory: pathlib.Path, scaled_directory: pathlib.Path
):
    dtype = np.float32
    reference_directory.mkdir(parents=True)
    scaled_directory.mkdir(parents=True)

    n_nodes = 100

    l_scale = np.random.rand() * 10 + 0.1
    t_scale = np.random.rand() * 10 + 0.1

    nodal_initial_u = np.random.rand(n_nodes, 3, 1).astype(dtype)
    np.save(
        reference_directory / "nodal_initial_u.npy",
        nodal_initial_u,
    )
    np.save(
        scaled_directory / "nodal_initial_u.npy",
        nodal_initial_u * l_scale / t_scale,
    )

    np.save(
        reference_directory / "nodal_last_u.npy",
        nodal_initial_u,
    )
    np.save(
        scaled_directory / "nodal_last_u.npy",
        nodal_initial_u * l_scale / t_scale,
    )

    lengths = np.random.rand(n_nodes, 1).astype(dtype)
    np.save(
        reference_directory / "length.npy",
        lengths,
    )
    np.save(
        scaled_directory / "length.npy",
        lengths * l_scale,
    )

    time = np.random.rand(1, 1).astype(dtype)
    np.save(
        reference_directory / "time.npy",
        time,
    )
    np.save(
        scaled_directory / "time.npy",
        time * t_scale,
    )

    np.save(
        scaled_directory / "l_scale.npy",
        [l_scale],
    )
    np.save(
        scaled_directory / "t_scale.npy",
        [t_scale],
    )

    return


if __name__ == "__main__":
    main()
