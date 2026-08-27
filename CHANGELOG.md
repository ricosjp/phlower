# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## UnReleased


### Breaking Change

* All errors raised in `IPhlowerModuleAdapter` and its realized classes are now wrapped in `PhlowerRunTimeError` with location information

### Fixed

* Fix to convert index tensor's device corresponding to input tensor 
* Revert Share module to inherit torch.nn.Module
* Fix to convert raw int or float values when parsing preprocess.yml
* Fix to set context manager when evaluation
* Fix the timing to update tensor during iteration. Target variables are now updated right after gradients are evaluated unless the gradient is diverged
* Fix unbatch process in `PhlowerInterpolatorPresetGroupModule`
* Fix CG solver to accept multiple input tensors
* Fix problems when restarting from the intermediate training state

### Added

* Add `time_series_sliding` item in prediction setting
* Add `NanToNum` Module
* Allow broadcasting in reducer setting
* Add `InterpolatorPresetGroupModule`
* Add logit transform in scaling functions
* Add a feature to dump intermediate outputs of each module during forward and backward pass
* Add Scaling Module
* Add activation functions (e.g. `sin`, `cos`, `exp` )
* Add CG solver module
* Add precondtioner to CG solver module
* Integrate `graphlow` object in phlower
* Add an option to get data from `field_data` in IsoGCN module
* Add `PhlowerContinueTrainer` to continue training from a checkpoint
* Add adjoint method to compute gradients in CG solver module
* Add an option to overwrite `field_data` within group or module's forward pass
* Add an option to control dim in pooling
* Add scaling functions for physical nondimensionalization
* Add and tweak internal operations to handle PINNs (Physics-Informed Neural Networks) in phlower
    * Add `FixedNorm` module
    * Add `Residual` module
    * Add "residual_mse" to compute loss for residuals in the loss calculator
    * Add several features which are necessary to perform PINN training and prediction
* Add error handlers to suppress or dump error information
* Add `Slip` module to handle slip boundary condition
* Allow to dump  en-route tensors when prediction (Only forward-path)
* Add temperature options to TransolverAttention 
* Add gradient accumulation steps 
* Handle NaN filling value in Slip module 
* Add `BiCG` solver module
* Add expand option in `Pooling` module

### Maintenance

* Extract codes and rely on `phlower_tensor` module to handle tensor with physical dimension
* Change to use `uv` in github actions
* Improve flaky tests
* Add cpu and gpu tests
* Generate documentation for all modules in `phlower.nn` and `phlower.group_module` using sphinx
* Tweak log emission timing

## [0.3.0] - 2025-12-08

### Added
* Add `attach_handler` method to `PhlowerTrainer` to add an extra handler at training process.
* Add `lazy_load` parameter to `TrainingSetting` to load data lazily.
* When `time_series_length` is -1, `PhlowerGroupModule` determines the time series length automatically from the input data.
* Add `NaNStoppingHandler` to stop training when loss becomes NaN.
* Add sliding window method for time series data in training process.
* Add distributed data parallel (DDP) training.
* Add `evaluate_context_manager` parameter to `TrainerSetting` to choose context manager during evaluation.
* Add `LayerNorm` module in `phlower.nn`
* Add `aggregation_method` parameter to `LossCalculator` to choose aggregation method of losses. (sum or mean)
* Add `cg` mode in iteration solver of `PhlowerGroupModule`.
* Add `PhlowerPresetGroupModule` as a preset group.

### Changed
* Time series tensor is splitted into each time step when forwarding with `time_series_length` in `PhlowerGroupModule`.
* Display details of losses at training process.
* Default inference mode is changed to `torch.inference_mode` from `torch.no_grad`.

### Fixed
* Fix `restart` method not to load recursively previous restarted checkpoints.
* Fix to set empty tcp port for DDP

## [0.2.2] - 2025-06-12

### Fixed
* Fix `PhlowerTensor` with physical dimension to handle `torch.stack`.

### Changed
* Decompose input members when to apply reverse transform after prediction
* Change default value of `bias` parameter in setting class of `GCN`. (False -> True)
* Change default value of `bias` parameter in setting class of `coefficient_network` in `IsoGCN`. (False -> True)


## [0.2.1] - 2025-06-04
### Added
* Add CHANGELOG.md

### Fixed
* Fix x_data to convert np.ndarray after transformed inversely in prediction stage. 
* Improve index access to PhlowerTensor in order to retain shape configuration such as timeseries or node index.
* Improve index access to return PhlowerTensor.
* Fix rearrange to retain shape configuration.

### Changed
* Change default value of `bias` parameter in setting class of `EnequivariantMLP`. (False -> True)

