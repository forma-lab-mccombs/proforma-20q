import numpy as np
import pandas as pd
import pytest

from proforma20q.schema import (
    SubmissionError,
    normalize_columns,
    validate_forecast,
    write_forecast,
    read_forecast,
)
from fixtures import forecast_from_truth, synthetic_truth


def _valid_fc():
    return forecast_from_truth(synthetic_truth(n_firms=5, n_q=2), noise=0.1)


def test_valid_forecast_passes():
    assert validate_forecast(_valid_fc(), strict=False) == []


def test_internal_aliases_normalized():
    fc = _valid_fc().rename(columns={"firm": "firm_id", "origin": "quarter",
                                     "horizon": "forecast_horizon"})
    out = normalize_columns(fc)
    assert {"firm", "origin", "horizon"}.issubset(out.columns)
    assert validate_forecast(out, strict=False) == []


def test_missing_column_flagged():
    fc = _valid_fc().drop(columns=["prediction"])
    problems = validate_forecast(fc, strict=False)
    assert any("prediction" in p for p in problems)
    with pytest.raises(SubmissionError):
        validate_forecast(fc, strict=True)


def test_horizon_out_of_range_flagged():
    fc = _valid_fc()
    fc.loc[0, "horizon"] = 25
    assert any("range" in p for p in validate_forecast(fc, strict=False))


def test_negative_sigma_flagged():
    fc = _valid_fc()
    fc["sigma"] = 1.0
    fc.loc[0, "sigma"] = -0.5
    assert any("sigma" in p for p in validate_forecast(fc, strict=False))


def test_duplicate_keys_flagged():
    fc = _valid_fc()
    dup = pd.concat([fc, fc.iloc[[0]]], ignore_index=True)
    assert any("duplicate" in p for p in validate_forecast(dup, strict=False))


def test_write_read_round_trip(tmp_path):
    fc = _valid_fc()
    p = tmp_path / "fc.parquet"
    write_forecast(fc, p)
    back = read_forecast(p, validate=True, strict=True)
    assert len(back) == len(fc)
    assert back["prediction"].dtype == np.float32
