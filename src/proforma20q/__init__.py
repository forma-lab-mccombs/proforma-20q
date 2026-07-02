"""ProForma-20Q: a public benchmark for joint probabilistic forecasting of
complete quarterly financial statements at horizons 1-20 quarters.

The benchmark defines the *task* (data build, splits, targets, transforms,
evaluation protocol, submission format, reference baselines). It is model-free:
you build the data from your own WRDS credentials, train your model, and score a
forecast file with :func:`proforma20q.evaluate.evaluate_forecasts`.

Nothing WRDS-derived is distributed here -- code, configs, and checksums only.
"""

__version__ = "0.1.0"

from .transforms import regularize, de_regularize, transform  # noqa: F401

__all__ = ["regularize", "de_regularize", "transform", "__version__"]
