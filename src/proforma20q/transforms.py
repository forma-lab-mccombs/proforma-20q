"""The regularization transform, ported verbatim (numpy-only) from the Forma
research repo's ``src/data/transforms.py``.

The benchmark works in a *regularized target space*: every statement item is
scaled by a firm-size proxy, passed through ``asinh``, z-scored per item, and
clamped. Both the ground-truth artifact and every submitted forecast live in
this space, so metrics are unit-free and comparable across the 78 items. See
``build.py`` for how mu / sigma / k are estimated.

NOTE: the non-linear transform is ``asinh`` (inverse hyperbolic sine), NOT the
signed-log that some legacy docs mention. asinh is correct.
"""
from __future__ import annotations

import numpy as np

# A tiny epsilon guards divisions, matching the research repo bit-for-bit.
_EPS = 1e-8


def transform(v, k):
    """Non-linear transform ``asinh(k * v)``.

    ``k`` is the reciprocal firm-size proxy (so ``k * v`` is size-scaled). Works
    on scalars or numpy arrays; ``k`` may be a scalar or an array broadcastable
    against ``v``.
    """
    return np.arcsinh(np.asarray(v) * k)


def regularize(v, k, mu, sigma):
    """Standardize a raw value into regularized space:

        z = (asinh(k * v) - mu) / (sigma + 1e-8)
    """
    return (transform(v, k) - mu) / (sigma + _EPS)


def de_regularize(x, k, mu, sigma):
    """Inverse of :func:`regularize`:

        v = sinh(x * sigma + mu) / (k + 1e-8)
    """
    t = np.asarray(x) * sigma + mu
    return np.sinh(t) / (k + _EPS)


def get_derivative_s(v, k, mu, sigma):
    """d/dv of :func:`regularize` -- the Jacobian factor for change-of-variables
    density work (e.g. converting a regularized-space density back to raw units).

        d/dv regularize(v) = k / (sigma * sqrt(1 + (k v)^2))
    """
    v = np.asarray(v)
    return k / (sigma * np.sqrt(1 + np.power(k * v, 2)) + _EPS)
