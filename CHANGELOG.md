# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## UnReleased

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

