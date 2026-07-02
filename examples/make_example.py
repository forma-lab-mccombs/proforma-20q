"""Generate the tiny SYNTHETIC example submission shipped in this folder.

Run: ``python examples/make_example.py``. Produces:
  * example_forecast.parquet  -- a valid submission (firm/target/origin/horizon/prediction/sigma)
  * example_truth.parquet     -- a matching wide truth frame you can score against

Nothing here is WRDS-derived -- it exists purely to document the file formats and
let ``proforma20q evaluate`` be exercised end-to-end without a real build.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from proforma20q.schema import write_forecast

HERE = Path(__file__).resolve().parent
TARGETS = ["niq", "revtq", "atq", "ltq"]
HORIZONS = list(range(1, 21))


def main() -> None:
    rng = np.random.default_rng(20260702)
    firms = [f"{i:06d}" for i in range(25)]
    origins = pd.period_range("2010Q1", "2011Q4", freq="Q").to_timestamp(how="end")

    truth_rows = [{"firm": f, "origin": o} for f in firms for o in origins]
    truth = pd.DataFrame(truth_rows)
    fc_rows = []
    for t in TARGETS:
        level0 = rng.standard_normal(len(truth))
        truth[f"{t}_level_0"] = level0
        for h in HORIZONS:
            actual = level0 + 0.15 * h + rng.standard_normal(len(truth)) * 0.6
            truth[f"{t}_t{h}"] = actual
            # a plausible fade-to-mean forecast with a horizon-growing sigma
            pred = 0.9 ** h * level0 + rng.standard_normal(len(truth)) * 0.3
            sigma = np.full(len(truth), 0.5 + 0.03 * h)
            for firm, origin, p, s in zip(truth["firm"], truth["origin"], pred, sigma):
                fc_rows.append({"firm": firm, "target": t, "origin": origin,
                                "horizon": h, "prediction": float(p), "sigma": float(s)})

    fc = pd.DataFrame(fc_rows)
    write_forecast(fc, HERE / "example_forecast.parquet")
    truth.to_parquet(HERE / "example_truth.parquet", engine="fastparquet", index=False)
    print(f"Wrote {len(fc):,}-row example_forecast.parquet and example_truth.parquet to {HERE}")


if __name__ == "__main__":
    main()
