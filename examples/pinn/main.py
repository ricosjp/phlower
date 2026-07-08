from __future__ import annotations

import argparse
import dataclasses as dc
import pathlib
import shutil

import matplotlib.pyplot as plt
import numpy as np

from phlower.services.predictor import PhlowerPredictor
from phlower.services.trainer import PhlowerTrainer
from phlower.settings import PhlowerSetting


@dc.dataclass(init=False, frozen=True)
class Constants:
    mass: float = 1.0
    damping: float = 0.2
    stiffness: float = 1.0

    def get_problem(self) -> MassSpringDamper:
        return MassSpringDamper(m=self.mass, c=self.damping, k=self.stiffness)


def setup() -> pathlib.Path:
    here = pathlib.Path(__file__).parent
    output_base_dir = here / "data"
    if output_base_dir.exists():
        shutil.rmtree(output_base_dir)
    output_base_dir.mkdir()
    return output_base_dir


class MassSpringDamper:
    def __init__(self, m: float, c: float, k: float):
        self.m = m
        self.c = c
        self.k = k

    def solve(self, t: float) -> float:
        omega = (self.k / self.m) ** 0.5
        h = self.c / (2 * self.m * omega)
        return np.exp(-h * omega * t) * np.cos(omega * (1 - h**2) ** 0.5 * t)


def create_sample_data(output_base_dir: pathlib.Path):
    """
    We simulate a simple 1D mass-spring-damper system.
    m d^2x/dt^2 + c dx/dt + k x = 0
    """

    problem = Constants().get_problem()
    data_t = np.linspace(0, 10, 10)
    data_x = problem.solve(data_t)

    residual_t = np.linspace(0, 30, 100)

    boundary_t = np.zeros(100)
    boundary_x = np.ones_like(boundary_t)

    dataset = {
        "data_t": data_t,
        "data_x": data_x,
        "residual_t": residual_t,
        "boundary_t": boundary_t,
        "boundary_x": boundary_x,
        "m": np.array([problem.m]),
        "c": np.array([problem.c]),
        "k": np.array([problem.k]),
    }
    _save_data(dataset, output_base_dir / "preprocessed/training")

    test_t = np.linspace(0, 30, 300)
    test_x = problem.solve(test_t)
    test_dataset = {
        "data_t": test_t,
        "answer_x": test_x,
        "residual_t": residual_t,
        "boundary_t": boundary_t,
        "m": np.array([problem.m]),
        "c": np.array([problem.c]),
        "k": np.array([problem.k]),
    }
    _save_data(test_dataset, output_base_dir / "preprocessed/test")


def _save_data(data: dict[str, np.ndarray], output_dir: pathlib.Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, value in data.items():
        np.save(output_dir / f"{key}.npy", value)


def train(output_base_dir: pathlib.Path, n_epoch: int):

    setting = PhlowerSetting.read_yaml(
        pathlib.Path(__file__).parent / "pinn.yml"
    )
    setting = setting.model_copy(
        update={
            "training": setting.training.model_copy(update={"n_epoch": n_epoch})
        }
    )
    setting = PhlowerSetting.model_validate(setting)

    trainer = PhlowerTrainer.from_setting(setting)

    trainer.train(
        output_directory=output_base_dir / "model",
        train_directories=[output_base_dir / "preprocessed/training"],
    )


def predict(output_base_dir: pathlib.Path):
    predictor = PhlowerPredictor.from_pathes(
        output_base_dir / "model", pathlib.Path(__file__).parent / "pinn.yml"
    )

    predictions = list(
        predictor.predict(
            preprocessed_data=[output_base_dir / "preprocessed/test"],
            perform_inverse_scaling=False,
        )
    )[0]

    output_dir = output_base_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        output_dir / "predicted_x.npy",
        predictions.prediction_data["data_x"].numpy(),
    )


def plot_results(output_base_dir: pathlib.Path):
    output_dir = output_base_dir / "predictions"
    predicted_x = np.load(output_dir / "predicted_x.npy")

    test_dir = output_base_dir / "preprocessed/test"
    answer_x = np.load(test_dir / "answer_x.npy")
    test_t = np.load(test_dir / "data_t.npy")

    train_dir = output_base_dir / "preprocessed/training"
    train_t = np.load(train_dir / "data_t.npy")
    train_x = np.load(train_dir / "data_x.npy")

    fig, ax = plt.subplots()
    ax.plot(test_t, predicted_x, label="Predicted", linestyle="-")
    ax.plot(test_t, answer_x, label="Ground Truth", linestyle="--")
    ax.scatter(train_t, train_x, color="red", label="Training Data")
    ax.set_xlabel("Time")
    ax.set_ylabel("Displacement")
    ax.set_title("Mass-Spring-Damper System Prediction")
    ax.legend()
    fig.savefig(output_dir / "prediction_plot.png")


def main():
    parser = argparse.ArgumentParser(description="Run PINN example.")
    parser.add_argument(
        "--n-epoch", type=int, default=100, help="Number of training epochs."
    )

    args = parser.parse_args()

    output_base_dir = setup()
    create_sample_data(output_base_dir)
    train(output_base_dir, n_epoch=args.n_epoch)
    output_base_dir = pathlib.Path(__file__).parent / "data"
    predict(output_base_dir)
    plot_results(output_base_dir)


if __name__ == "__main__":
    main()
