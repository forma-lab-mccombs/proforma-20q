import numpy as np
import pandas as pd
import pytest

from proforma20q.schema import (
    SubmissionError,
    normalize_columns,
    validate_forecast,
    write_forecast,
    write_forecast_blocks,
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


def test_all_valid_targets_pass():
    fc = _valid_fc()
    assert not any("target" in p for p in validate_forecast(fc, strict=False))


def test_fully_bogus_target_rejected():
    fc = _valid_fc()
    fc["target"] = "not_a_real_target"
    problems = validate_forecast(fc, strict=False)
    assert any("not among the 78 pf_full targets" in p for p in problems)
    with pytest.raises(SubmissionError):
        validate_forecast(fc, strict=True)


def test_partially_bogus_target_rejected():
    fc = _valid_fc()
    # Misspell the target on a subset of rows: the valid rows still validate,
    # but the misspelled ones must be flagged rather than silently dropped.
    fc.loc[fc.index[:5], "target"] = "niqq"
    problems = validate_forecast(fc, strict=False)
    tprob = [p for p in problems if "target" in p]
    assert tprob and "niqq" in tprob[0]


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


def _blocks(fc):
    """Split a forecast into (target, horizon) blocks, the streaming unit."""
    return [g for _k, g in fc.groupby(["target", "horizon"], sort=False)]


def test_block_writer_matches_single_write(tmp_path):
    """`write_forecast_blocks` is the writer a full-coverage (~550M row)
    submission needs; it must produce the same file contents as the one-shot
    writer for data small enough for both."""
    fc = _valid_fc()
    one, many = tmp_path / "one.parquet", tmp_path / "many.parquet"
    write_forecast(fc, one)
    n = write_forecast_blocks(_blocks(fc), many, rows_per_group=7)
    assert n == len(fc)

    a = read_forecast(one, validate=False).sort_values(
        ["firm", "target", "origin", "horizon"]).reset_index(drop=True)
    b = read_forecast(many, validate=False).sort_values(
        ["firm", "target", "origin", "horizon"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_block_writer_emits_multiple_row_groups(tmp_path):
    """Row-group-per-buffer is the point: peak memory is the buffer, not the
    submission."""
    import fastparquet

    fc = _valid_fc()
    p = tmp_path / "fc.parquet"
    write_forecast_blocks(_blocks(fc), p, rows_per_group=5)
    pf = fastparquet.ParquetFile(str(p))
    assert len(pf.row_groups) > 1
    assert sum(rg.num_rows for rg in pf.row_groups) == len(fc)


def test_block_writer_validates_each_block(tmp_path):
    fc = _valid_fc()
    bad = _blocks(fc)
    bad[1] = bad[1].assign(target="not_a_real_item")
    with pytest.raises(SubmissionError):
        write_forecast_blocks(bad, tmp_path / "fc.parquet")


def test_block_writer_rejects_an_empty_stream(tmp_path):
    with pytest.raises(SubmissionError):
        write_forecast_blocks([], tmp_path / "fc.parquet")


def test_block_writer_overwrites_a_stale_file(tmp_path):
    """Appending row-groups to a file left over from a previous run would
    silently double a submission."""
    fc = _valid_fc()
    p = tmp_path / "fc.parquet"
    write_forecast_blocks(_blocks(fc), p, rows_per_group=5)
    write_forecast_blocks(_blocks(fc), p, rows_per_group=5)
    assert len(read_forecast(p, validate=False)) == len(fc)
