# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## UnReleased
### Added
* Add `attach_handler` method to `PhlowerTrainer` to add an extra handler at training process.
* Add `lazy_load` parameter to `TrainingSetting` to load data lazily.
* When `time_series_length` is -1, `PhlowerGroupModule` determines the time series length automatically from the input data.

### Changed
* Time series tensor is splitted into each time step when forwarding with `time_series_length` in `PhlowerGroupModule`.
* Display details of losses at training process.


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

