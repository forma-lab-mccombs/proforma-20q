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


def test_submission_md_minimal_example_round_trips(tmp_path):
    """The verbatim SUBMISSION.md example, then the next documented command.

    `pd.Timestamp("2011-12-31")` is datetime64[s] under pandas >= 2.2;
    fastparquet stored it as TIMESTAMP[MILLIS] while recording
    `numpy_type: datetime64[s]`, and the read failed with
    "Cannot losslessly cast '1325289 ms' to s". Every other fixture in this
    suite builds origins via `period_range(...).to_timestamp(how="end")` --
    nanosecond EOQ, which dodges the bug -- so nothing caught it."""
    fc = pd.DataFrame({
        "firm":       ["001045", "001045", "001045"],
        "target":     ["niq",    "niq",    "revtq"],
        "origin":     pd.Timestamp("2011-12-31"),
        "horizon":    [1, 2, 1],
        "prediction": [0.42, 0.55, -1.13],
        "sigma":      [0.8, 0.9, 0.7],
    })
    # The input that broke it: a datetime resolution COARSER than nanoseconds.
    # Which one pandas picks for `pd.Timestamp("2011-12-31")` is version-
    # dependent (`[s]` on pandas 2.2, `[us]` on pandas 3) -- the bug is the
    # non-ns unit, not any particular one, so pin that.
    assert fc["origin"].dtype.kind == "M"
    assert np.datetime_data(fc["origin"].dtype)[0] != "ns"
    p = tmp_path / "my_forecasts.parquet"
    write_forecast(fc, p)

    back = read_forecast(p, validate=True, strict=True)
    assert len(back) == 3
    assert back["origin"].dtype == "datetime64[ns]"
    assert (back["origin"] == pd.Timestamp("2011-12-31")).all()

    from proforma20q.cli import main
    assert main(["validate", str(p)]) == 0


@pytest.mark.parametrize("origin", [
    pd.Timestamp("2011-12-31"),                                   # datetime64[s]
    pd.Period("2011Q4", freq="Q"),                                # Period[Q]
    pd.Timestamp("2011-12-31").as_unit("ms"),                     # millisecond
    pd.Period("2011Q4", freq="Q").to_timestamp(how="end"),        # nanosecond EOQ
])
def test_every_accepted_origin_form_round_trips(tmp_path, origin):
    fc = pd.DataFrame({"firm": ["001045"], "target": ["niq"], "origin": [origin],
                       "horizon": [1], "prediction": [0.42]})
    p = tmp_path / "fc.parquet"
    write_forecast(fc, p)
    back = read_forecast(p, validate=True, strict=True)
    assert back["origin"].dtype == "datetime64[ns]"
    assert back["origin"].iloc[0].year == 2011 and back["origin"].iloc[0].quarter == 4


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


def test_block_writer_allows_one_all_null_block(tmp_path):
    """A single (target, horizon) with no fittable training rows is legitimately
    all-NaN -- the linear baselines emit exactly that. Validating each block in
    isolation must not turn it into an aborted 1,560-block write."""
    fc = _valid_fc()
    blocks = _blocks(fc)
    blocks[1] = blocks[1].assign(prediction=np.nan)
    p = tmp_path / "fc.parquet"
    assert write_forecast_blocks(blocks, p) == len(fc)
    back = read_forecast(p, validate=False)
    assert back["prediction"].isna().sum() == len(blocks[1])
    # ... but a submission that is all-null everywhere is still an error
    with pytest.raises(SubmissionError, match="every prediction"):
        write_forecast_blocks([b.assign(prediction=np.nan) for b in blocks],
                              tmp_path / "empty.parquet")


def test_block_writer_catches_duplicates_inside_a_block(tmp_path):
    """Duplicate keys come from a repeated (firm, origin) in the origin frame,
    which duplicates the key in EVERY block -- so the intra-block scan is the one
    that matters, and it is cheap."""
    fc = _valid_fc()
    blocks = [pd.concat([b, b.iloc[[0]]], ignore_index=True) for b in _blocks(fc)]
    with pytest.raises(SubmissionError, match="duplicate"):
        write_forecast_blocks(blocks, tmp_path / "fc.parquet")


def test_block_writer_skips_an_empty_block(tmp_path):
    """A generator that yields nothing for a target with no test rows is the
    obvious user pattern; it must not abort the write."""
    fc = _valid_fc()
    blocks = _blocks(fc)
    empty = blocks[0].iloc[:0]
    p = tmp_path / "fc.parquet"
    assert write_forecast_blocks([blocks[0], empty, *blocks[1:]], p) == len(fc)


def test_a_failed_write_leaves_the_previous_submission_intact(tmp_path):
    """A run that dies at block 1,400 of 1,560 must not leave a well-formed
    parquet that scores as an intentional partial-coverage entry -- and must not
    destroy the good file from the previous run."""
    fc = _valid_fc()
    p = tmp_path / "fc.parquet"
    write_forecast_blocks(_blocks(fc), p)
    good = read_forecast(p, validate=False)

    def dies_partway():
        for i, b in enumerate(_blocks(fc)):
            if i == 2:
                raise RuntimeError("simulated OOM")
            yield b

    with pytest.raises(RuntimeError):
        write_forecast_blocks(dies_partway(), p, rows_per_group=1)
    pd.testing.assert_frame_equal(read_forecast(p, validate=False), good)
    assert (tmp_path / "fc.parquet.partial").exists()   # visible, and not a submission


def test_block_writer_normalizes_origin_across_blocks(tmp_path):
    """Mixed origin representations were coerced against the first row-group's
    schema, so a Period block read back as 1970-era nanoseconds -- silently."""
    fc = _valid_fc()
    blocks = _blocks(fc)
    blocks[1] = blocks[1].assign(
        origin=pd.PeriodIndex(pd.to_datetime(blocks[1]["origin"]), freq="Q"))
    p = tmp_path / "fc.parquet"
    write_forecast_blocks(blocks, p)
    back = read_forecast(p, validate=False)
    assert back["origin"].dt.year.min() >= 2010


def test_categorical_keys_do_not_become_the_string_nan(tmp_path):
    """`astype(str)` on a categorical with a missing value writes the literal
    'nan', which validates as a real firm id and joins to nothing."""
    fc = _valid_fc()
    firm = fc["firm"].astype("category")
    fc = fc.assign(firm=firm.cat.set_categories(sorted(set(fc["firm"]))[1:]))
    with pytest.raises(SubmissionError, match="null value"):
        write_forecast_blocks(_blocks(fc), tmp_path / "fc.parquet")


def test_sigma_underflow_in_float32_is_rejected(tmp_path):
    """A sigma that passes `> 0` on the way in and lands as 0.0 on disk makes the
    file fail the check it just passed."""
    fc = _valid_fc().assign(sigma=1e-50)
    with pytest.raises(SubmissionError, match="underflow"):
        write_forecast(fc, tmp_path / "fc.parquet")


def test_validate_warns_about_coverage_and_raw_units(tmp_path):
    """A forecast that is 40% inf shrinks the scored sample for EVERY model it is
    compared against, and a raw-dollar submission scores as noise. Both are
    schema-valid, so both have to be warnings the user actually sees."""
    from proforma20q.cli import main
    from proforma20q.schema import forecast_warnings

    fc = _valid_fc()
    holed = fc.copy()
    holed.loc[holed.index[: int(len(fc) * 0.4)], "prediction"] = np.inf
    warns = forecast_warnings(holed)
    assert any("non-finite" in w and "shrinking it for every model" in w for w in warns)

    dollars = fc.assign(prediction=fc["prediction"] * 1e6)
    assert any("regularized space" in w for w in forecast_warnings(dollars))
    assert forecast_warnings(fc) == []

    # ... and the CLI surfaces them while still exiting 0 (the file IS valid)
    p = tmp_path / "holed.parquet"
    write_forecast(holed, p, validate=False)
    assert main(["validate", str(p)]) == 0


def test_file_validation_streams_by_row_group(tmp_path):
    """`validate` must not start by loading the file: a full-coverage submission
    is ~550M rows / ~73 GB as a frame."""
    from proforma20q.schema import validate_forecast_file

    fc = _valid_fc()
    p = tmp_path / "fc.parquet"
    write_forecast_blocks(_blocks(fc), p, rows_per_group=5)
    problems, n_rows = validate_forecast_file(p)
    assert problems == [] and n_rows == len(fc)

    bad = tmp_path / "bad.parquet"
    write_forecast_blocks([b.assign(target="not_an_item") for b in _blocks(fc)],
                          bad, validate=False, rows_per_group=5)
    problems, n_rows = validate_forecast_file(bad)
    assert n_rows == len(fc)
    assert any("not among the 78 pf_full targets" in p for p in problems)
