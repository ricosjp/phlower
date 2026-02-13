import pathlib
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import phlower
from phlower.services.predictor import (
    PhlowerPredictionResult,
    PhlowerPredictor,
)
from phlower.services.trainer import PhlowerTrainer


def main():
    data_directory = pathlib.Path("data")
    output_directory = data_directory / "model"
    if data_directory.exists():
        shutil.rmtree(data_directory)

    setting = phlower.settings.PhlowerSetting.read_yaml("interpolator.yml")

    generate_data(setting.data.training[0])

    trainer = PhlowerTrainer.from_setting(setting)
    trainer.train(output_directory=output_directory)

    # Predict
    predictor = PhlowerPredictor(
        model_directory=output_directory,
        predict_setting=phlower.settings.PhlowerPredictorSetting(
            selection_mode="latest",
        ),
    )
    results = list(
        predictor.predict(
            setting.data.training,
            perform_inverse_scaling=False,
        )
    )
    postprocess_results(
        output_directory / "predictions", results, setting.data.training
    )


def generate_data(output_directory: pathlib.Path):
    dtype = np.float32

    output_directory.mkdir(parents=True, exist_ok=True)

    n_points_source = 200
    n_points_face1 = 30
    n_points_face2 = 50
    scale_z = 2

    z_face1 = 0.4 * scale_z
    z_face2 = 0.8 * scale_z

    source_points = np.random.rand(n_points_source, 3, 1)
    source_points[:, -1] = source_points[:, -1] * scale_z
    face1_points = np.ones((n_points_face1, 3, 1)) * z_face1
    face1_points[:, :2, 0] = np.random.rand(n_points_face1, 2)
    face2_points = np.ones((n_points_face2, 3, 1)) * z_face2
    face2_points[:, :2, 0] = np.random.rand(n_points_face2, 2)

    # Input fields
    source_r = np.linalg.norm(source_points[:, :2], axis=1)

    face1_r = np.linalg.norm(face1_points[:, :2], axis=1)
    face2_r = np.linalg.norm(face2_points[:, :2], axis=1)

    # Answer fields
    face1_phi = answer_scalar_field(face1_r)
    face2_phi = answer_scalar_field(face2_r)
    face1_v = answer_vector_field(face1_r)
    face2_v = answer_vector_field(face2_r)

    np.save(output_directory / "source_points.npy", source_points.astype(dtype))
    np.save(output_directory / "face1_points.npy", face1_points.astype(dtype))
    np.save(output_directory / "face2_points.npy", face2_points.astype(dtype))

    np.save(output_directory / "source_r.npy", source_r.astype(dtype))

    np.save(output_directory / "face1_phi.npy", face1_phi.astype(dtype))
    np.save(output_directory / "face2_phi.npy", face2_phi.astype(dtype))
    np.save(output_directory / "face1_v.npy", face1_v.astype(dtype))
    np.save(output_directory / "face2_v.npy", face2_v.astype(dtype))
    return


def answer_scalar_field(r: np.ndarray) -> np.ndarray:
    return np.sin(r * np.pi)


def answer_vector_field(r: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            np.zeros_like(r),
            np.zeros_like(r),
            np.cos(r * np.pi),
        ],
        axis=1,
    )


def postprocess_results(
    output_directory_base: pathlib.Path,
    results: list[PhlowerPredictionResult],
    base_directories: list[pathlib.Path],
):
    loss_phi_source = 0.0
    loss_phi_face1 = 0.0
    loss_phi_face2 = 0.0
    loss_v_source = 0.0
    loss_v_face1 = 0.0
    loss_v_face2 = 0.0
    for result, base_directory in zip(results, base_directories, strict=True):
        # Source
        source_points = np.load(base_directory / "source_points.npy")
        predicted_source_phi = result.prediction_data["phi"].numpy()
        predicted_source_v = result.prediction_data["v"].numpy()
        answer_source_phi = answer_scalar_field(result.input_data["source_r"])
        answer_source_v = answer_vector_field(result.input_data["source_r"])

        # Face 1
        face1_points = np.load(base_directory / "face1_points.npy")
        predicted_face1_phi = result.prediction_data["face1_phi"].numpy()
        predicted_face1_v = result.prediction_data["face1_v"].numpy()
        answer_face1_phi = result.answer_data["face1_phi"].numpy()
        answer_face1_v = result.answer_data["face1_v"].numpy()

        # Face 2
        face2_points = np.load(base_directory / "face2_points.npy")
        predicted_face2_phi = result.prediction_data["face2_phi"].numpy()
        predicted_face2_v = result.prediction_data["face2_v"].numpy()
        answer_face2_phi = result.answer_data["face2_phi"].numpy()
        answer_face2_v = result.answer_data["face2_v"].numpy()

        # Evaluate loss
        loss_phi_source += rmse(predicted_source_phi, answer_source_phi)
        loss_phi_face1 += rmse(predicted_face1_phi, answer_face1_phi)
        loss_phi_face2 += rmse(predicted_face2_phi, answer_face2_phi)
        loss_v_source += rmse(predicted_source_v, answer_source_v)
        loss_v_face1 += rmse(predicted_face1_v, answer_face1_v)
        loss_v_face2 += rmse(predicted_face2_v, answer_face2_v)

        case_name = base_directory.name
        output_directory = output_directory_base / case_name
        output_directory.mkdir(parents=True, exist_ok=True)

        # Write vtu
        ms = pv.PointSet()
        ms.points = np.squeeze(source_points)
        ms.point_data.update(
            {
                "predicted_phi": np.squeeze(predicted_source_phi),
                "predicted_v": np.squeeze(predicted_source_v),
                "answer_phi": np.squeeze(answer_source_phi),
                "answer_v": np.squeeze(answer_source_v),
            }
        )
        ms.cast_to_unstructured_grid().save(output_directory / "source.vtu")

        m1 = pv.PointSet()
        m1.points = np.squeeze(face1_points)
        m1.point_data.update(
            {
                "predicted_phi": np.squeeze(predicted_face1_phi),
                "predicted_v": np.squeeze(predicted_face1_v),
                "answer_phi": np.squeeze(answer_face1_phi),
                "answer_v": np.squeeze(answer_face1_v),
            }
        )
        m1.cast_to_unstructured_grid().save(output_directory / "face1.vtu")

        m2 = pv.PointSet()
        m2.points = np.squeeze(face2_points)
        m2.point_data.update(
            {
                "predicted_phi": np.squeeze(predicted_face2_phi),
                "predicted_v": np.squeeze(predicted_face2_v),
                "answer_phi": np.squeeze(answer_face2_phi),
                "answer_v": np.squeeze(answer_face2_v),
            }
        )
        m2.cast_to_unstructured_grid().save(output_directory / "face2.vtu")

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        n_decimate_source = 1
        ax.scatter(
            source_points[::n_decimate_source, 0, 0],
            source_points[::n_decimate_source, 1, 0],
            source_points[::n_decimate_source, 2, 0],
            c=predicted_source_phi[::n_decimate_source, 0],
            marker=".",
            label="source (pred)",
        )
        ax.scatter(
            face1_points[:, 0, 0],
            face1_points[:, 1, 0],
            face1_points[:, 2, 0],
            c=answer_face1_phi[:, 0],
            marker="s",
            label="face1 (ans)",
        )
        ax.scatter(
            face2_points[:, 0, 0],
            face2_points[:, 1, 0],
            face2_points[:, 2, 0],
            c=answer_face2_phi[:, 0],
            marker="^",
            label="face2 (ans)",
        )
        ax.set_box_aspect((1, 1, 2))
        ax.set_title(f"{case_name}: scalar")
        ax.legend()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=-20, azim=120, roll=76)
        fig.savefig(output_directory / "scalar.pdf")

        fig_quiver = plt.figure()
        ax_quiver = fig_quiver.add_subplot(projection="3d")
        ax_quiver.quiver(
            source_points[::n_decimate_source, 0, 0],
            source_points[::n_decimate_source, 1, 0],
            source_points[::n_decimate_source, 2, 0],
            predicted_source_v[::n_decimate_source, 0, 0],
            predicted_source_v[::n_decimate_source, 1, 0],
            predicted_source_v[::n_decimate_source, 2, 0],
            alpha=0.5,
            length=0.1,
            color="red",
            normalize=False,
            label="source (pred)",
            arrow_length_ratio=0.5,
        )
        ax_quiver.quiver(
            face1_points[:, 0, 0],
            face1_points[:, 1, 0],
            face1_points[:, 2, 0],
            answer_face1_v[:, 0, 0],
            answer_face1_v[:, 1, 0],
            answer_face1_v[:, 2, 0],
            alpha=0.5,
            length=0.1,
            color="blue",
            normalize=False,
            label="face1 (ans)",
            arrow_length_ratio=0.5,
        )
        ax_quiver.quiver(
            face2_points[:, 0, 0],
            face2_points[:, 1, 0],
            face2_points[:, 2, 0],
            answer_face2_v[:, 0, 0],
            answer_face2_v[:, 1, 0],
            answer_face2_v[:, 2, 0],
            alpha=0.5,
            length=0.1,
            color="green",
            normalize=False,
            label="face2 (ans)",
            arrow_length_ratio=0.5,
        )
        ax_quiver.set_box_aspect((1, 1, 2))
        ax_quiver.set_title(f"{case_name}: vector")
        ax_quiver.legend()
        ax_quiver.set_xlabel("X")
        ax_quiver.set_ylabel("Y")
        ax_quiver.set_zlabel("Z")
        ax_quiver.view_init(elev=-20, azim=120, roll=76)
        fig_quiver.savefig(output_directory / "vector.pdf")

        print(f"Data written in: {output_directory}")

    print(f"{' ' * 29}{'phi'.rjust(9)},   {'v'.rjust(9)}")
    print(
        f"Loss at the source points: {loss_phi_source:.5e}, {loss_v_source:.5e}"
    )
    print(
        f"       Loss at the face 1: {loss_phi_face1:.5e}, {loss_v_face1:.5e}"
    )
    print(
        f"       Loss at the face 2: {loss_phi_face2:.5e}, {loss_v_face1:.5e}"
    )

    assert loss_phi_source < np.mean([loss_phi_face1, loss_phi_face2]) * 2
    assert loss_v_source < np.mean([loss_v_face1, loss_v_face2]) * 2

    plt.show()
    return


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    return np.sqrt(np.mean((x - y) ** 2))


if __name__ == "__main__":
    main()
