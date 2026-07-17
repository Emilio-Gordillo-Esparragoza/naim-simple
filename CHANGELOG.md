# Changelog

## Unreleased

### Security

- Keep the package build backend on setuptools >=77 (Python 3.9 compatible) and
  upgrade to setuptools >=83 only in the Python 3.10 security job so
  `pip-audit` can resolve PYSEC-2026-3447 without breaking the 3.9 matrix.

### Changed

- Added independent Python 3.9, 3.10, and 3.11 CI jobs and real pytest coverage.
- Persisted categorical encoders and corrected feature-order and missing-value handling.
- Replaced normal pickle persistence with versioned JSON and Safetensors bundles.
- Added mini-batch training, validated YAML configuration, and direct CLI entry points.
- Required tests before package publication and expanded dependency security scanning.

### Security

- Moved legacy pickle loading behind an explicit trusted-only migration API.
