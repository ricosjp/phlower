import pathlib
import shutil

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

import phlower
from phlower.services.predictor import (
    PhlowerPredictor,
)
from phlower.services.trainer import PhlowerTrainer


def main():
    data_directory = pathlib.Path("data")
    output_directory = data_directory / "model"
    if data_directory.exists():
        shutil.rmtree(data_directory)

    setting = phlower.settings.PhlowerSetting.read_yaml("cg.yml")

    generate_data(setting.data.training[0])

    trainer = PhlowerTrainer.from_setting(setting)
    trainer.train(output_directory=output_directory)

    # Predict
    predictor = PhlowerPredictor(
        model_directory=output_directory,
        predict_setting=phlower.settings.PhlowerPredictorSetting(
            selection_mode="train_best",
        ),
    )
    results = list(
        predictor.predict(
            setting.data.training,
            perform_inverse_scaling=False,
        )
    )[0]

    x = np.squeeze(results.input_data["x"])

    n_ts = len(results.prediction_data["phi"])
    fig, ax = plt.subplots()
    for i_ts in range(n_ts):
        ax.cla()
        # plt.plot(x, results.input_data["phi_init"], ":", label="init")
        ax.plot(
            x,
            np.squeeze(results.answer_data["phi"][i_ts]),
            "-",
            label="ans",
        )
        ax.plot(
            x,
            np.squeeze(results.prediction_data["phi"][i_ts]),
            "--",
            label="pred",
        )
        ax.legend()
        ax.set_title(f"t = {(i_ts + 1) * 0.1:0.2f}")
        fig.savefig(output_directory / f"pred.{i_ts + 1:08d}.pdf")
        plt.pause(1.0)

    fig.show()
    print(f"Figure saved in: {output_directory}")

    rmse_phi = rmse(results.prediction_data["phi"], results.answer_data["phi"])
    print(f"RMSE: {rmse_phi:5e}")
    assert rmse_phi < 1.0e-1


def generate_data(output_directory: pathlib.Path):
    dtype = np.float32

    output_directory.mkdir(parents=True, exist_ok=True)

    n_points = 10
    length = 2.0

    dx = length / n_points
    dt = 0.1
    n_t = round(1.0 / dt)
    a = 1.0
    m = 2
    k = 0.1
    phi_left = -1.5
    phi_right = -1.5 + length

    x = np.linspace(0.0, length, n_points)[:, None]
    t = np.linspace(0, 1.0, n_t)
    phi = (
        -x * (x - length) / 2
        - 1.5
        + x
        + np.einsum(
            "xf,t->txf",
            a * np.sin(m * np.pi * x / length),
            np.exp(-k * (m * np.pi / length) ** 2 * t),
        )
    )

    dirichlet = np.ones(phi.shape[1:]) * np.nan
    dirichlet[0] = phi_left
    dirichlet[-1] = phi_right

    lap = np.zeros((n_points, n_points))
    indices = np.arange(n_points)
    lap[indices, indices] = -2 / dx**2
    lap[indices[:-1] + 1, indices[:-1]] = 1 / dx**2
    lap[indices[:-1], indices[:-1] + 1] = 1 / dx**2
    slap = sp.coo_array(lap, dtype=dtype)
    # raise ValueError(phi[..., -1, 0], dirichlet[-1])

    np.save(output_directory / "x.npy", x.astype(dtype))
    np.save(output_directory / "phi_init.npy", phi[0].astype(dtype))
    np.save(output_directory / "nu.npy", np.ones((1, 1), dtype=dtype))
    np.save(output_directory / "phi.npy", phi[1:].astype(dtype))
    np.save(output_directory / "dirichlet.npy", dirichlet.astype(dtype))

    sp.save_npz(output_directory / "lap.npz", slap)
    return


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    return np.sqrt(np.mean((x - y) ** 2))


if __name__ == "__main__":
    main()
