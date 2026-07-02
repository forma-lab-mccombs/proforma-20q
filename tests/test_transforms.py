import numpy as np

from proforma20q.transforms import de_regularize, regularize, transform


def test_asinh_transform():
    assert np.isclose(transform(0.0, 1.0), 0.0)
    assert np.isclose(transform(np.sinh(2.0), 1.0), 2.0)


def test_regularize_round_trip():
    rng = np.random.default_rng(0)
    v = rng.standard_normal(1000) * 50.0
    k, mu, sigma = 0.3, 0.1, 1.2
    z = regularize(v, k, mu, sigma)
    v2 = de_regularize(z, k, mu, sigma)
    assert np.allclose(v, v2, atol=1e-6)


def test_regularize_is_finite_and_monotone():
    v = np.linspace(-1e6, 1e6, 501)
    z = regularize(v, 0.5, 0.0, 1.0)
    assert np.all(np.isfinite(z))
    assert np.all(np.diff(z) > 0)  # strictly increasing
