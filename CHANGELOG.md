# Changelog

## Unreleased

### Changed

- Added independent Python 3.9, 3.10, and 3.11 CI jobs and real pytest coverage.
- Persisted categorical encoders and corrected feature-order and missing-value handling.
- Replaced normal pickle persistence with versioned JSON and Safetensors bundles.
- Added mini-batch training, validated YAML configuration, and direct CLI entry points.
- Required tests before package publication and expanded dependency security scanning.

### Security

- Moved legacy pickle loading behind an explicit trusted-only migration API.
