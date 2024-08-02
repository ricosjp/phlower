"""
.. _third:


High Level API for scaling, training and prediction
----------------------------------------------------

In this section, we will use high level API for performing machine learning process.

"""

###################################################################################################
# At First, we will prepare dummy data.
# These dummy data corresponds to feature values extracted from simultion data.
#

import pathlib
import random
import shutil

import numpy as np
import scipy.sparse as sp


def prepare_sample_interim_files():
    np.random.seed(0)
    random.seed(0)

    output_directory = pathlib.Path("out")
    if output_directory.exists():
        shutil.rmtree(output_directory)

    base_interim_dir = output_directory / "interim"
    base_interim_dir.mkdir(parents=True)

    n_cases = 5
    dtype = np.float32
    for i in range(n_cases):
        n_nodes = 100 * (i + 1)
        interim_dir = base_interim_dir / f"case_{i}"
        interim_dir.mkdir()

        nodal_initial_u = np.random.rand(n_nodes, 3, 1)
        np.save(
            interim_dir / "nodal_initial_u.npy",
            nodal_initial_u.astype(dtype),
        )

        # nodal_last_u = np.random.rand(n_nodes, 3, 1)
        np.save(interim_dir / "nodal_last_u.npy", nodal_initial_u.astype(dtype))

        sparse_array_names = [
            "nodal_nadj",
            "nodal_x_grad_hop1",
            "nodal_y_grad_hop1",
            "nodal_z_grad_hop1",
        ]
        rng = np.random.default_rng()
        for name in sparse_array_names:
            arr = sp.random(n_nodes, n_nodes, density=0.1, random_state=rng)
            sp.save_npz(interim_dir / name, arr.tocoo().astype(dtype))

        (interim_dir / "converted").touch()


prepare_sample_interim_files()

###################################################################################################
# Setting file for scaling and training can be downloaded from 
# `data.yml
# <https://github.com/ricosjp/phlower/tutorials/basic_usages/sample_data/e2e/setting.yml>`_
# we perform scaling process for data above.
# 

from phlower.services.preprocessing import PhlowerScalingService
from phlower.settings import PhlowerSetting

setting = PhlowerSetting.read_yaml("sample_data/e2e/setting.yml")

scaler = PhlowerScalingService.from_setting(setting)
scaler.fit_transform_all(
    interim_data_directories=[
        pathlib.Path("out/interim/case_0"),
        pathlib.Path("out/interim/case_1"),
        pathlib.Path("out/interim/case_2"),
        pathlib.Path("out/interim/case_3"),
        pathlib.Path("out/interim/case_4"),
    ],
    output_base_directory=pathlib.Path("out/preprocessed"),
)


###################################################################################################
# Next, we perform training by using preprocessed data.
#

from phlower.services.trainer import PhlowerTrainer

trainer = PhlowerTrainer.from_setting(setting)

loss = trainer.train(
    train_directories=[
        pathlib.Path("out/preprocessed/case_0"),
        pathlib.Path("out/preprocessed/case_1"),
        pathlib.Path("out/preprocessed/case_2"),
    ],
    validation_directories=[
        pathlib.Path("out/preprocessed/case_3"),
        pathlib.Path("out/preprocessed/case_4"),
    ],
    output_directory=pathlib.Path("out/model"),
)

###################################################################################################
# ``train`` function returns PhlowerTensor object which corresponds to last validation loss.
# Let's call print it.
#
# We can find that loss object has physical dimension and it is L^2 T^(-2)
# because we use MSE (Mean Squared Error) as a loss function.

print(loss)


###################################################################################################
# Finally, we perform predicion by using pretrained model.
# Setting file for prediction can be downloaded from 
# `data.yml
# <https://github.com/ricosjp/phlower/tutorials/basic_usages/sample_data/e2e/predict.yml>`_
#
# It is found that physical dimension is also considered properly.

from phlower.services.predictor import PhlowerPredictor

setting = PhlowerSetting.read_yaml("sample_data/e2e/predict.yml")

predictor = PhlowerPredictor(
    model_directory=pathlib.Path("out/model"),
    predict_setting=setting.prediction,
)

preprocessed_directories = [pathlib.Path("out/preprocessed/case_3")]

for result in predictor.predict(preprocessed_directories):
    for k in result.keys():
        print(f"{k}: {result[k].dimension}")
