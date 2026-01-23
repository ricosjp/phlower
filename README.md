# Phlower

![PyPI - Implementation](https://img.shields.io/pypi/implementation/phlower)
![PyPI - Version](https://img.shields.io/pypi/pyversions/phlower)

![](docs/source/_static/logo.png)


__Phlower__ is a deep learning framework based on PyTorch especially for physical phenomenon such as fluid dynamics.

For more details, please refer to user manual below.

- [User Manual](https://ricosjp.github.io/phlower/)


## Key Features


Tutorials are prepared for key features !

1. [Extended Tensor object which enables you to handle physics dimention](https://ricosjp.github.io/phlower/tutorials/basic_usages/01_phlower_tensor_basic.html)
2. [Model definition by yaml file](https://ricosjp.github.io/phlower/tutorials/basic_usages/02_model_definition_by_yaml_file.html)
3. [High Level API for scaling, training and predicion](https://ricosjp.github.io/phlower/tutorials/basic_usages/03_high_level_api_for_scaling_training_and_prediction.html)



## Installation

**Phlower** is registered at [PyPI](https://pypi.org/project/phlower/).

If you have already installed PyTorch, you can install phlower using pip. Python 3.10 or newer is supported.

```
pip install phlower
```

If you have not installed PyTorch yet, please follow the instructions on the official PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).
Or, you can install phlower along with PyTorch by specifying the appropriate extra keyword. For example, to install phlower with CPU support only, use the following command:

```
pip install phlower[cpu]
```

`cu118` and `cu124` are also available for CUDA 11.8 and CUDA 12.4, respectively.


## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)


## Development

This github repository is mirrored from the internal GitLab server of RICOS Corporation.
This is because some tests need to use GPU resources provided by RICOS.
However, the development is basically open to public and contributions are welcome.


## Publication

We have published a conference paper about Phlower.

> **Phlower: A Deep Learning Framework Supporting PyTorch Tensors with Physical Dimensions**  
> Riku Sakamoto  
> *Proceedings of the SciPy Conference*, 2025  
> DOI: https://doi.org/10.25080/vwty6796

