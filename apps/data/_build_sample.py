"""One-off builder for the bundled Fama-French sample used by the capital-markets app.

Run manually (needs network the first time; caches under the OS temp dir) to refresh
apps/data/ff_sample.npz. Never run at app runtime — the app loads the committed .npz.

    python apps/data/_build_sample.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from _ff_data import load_fama_french

N_MONTHS = 240
N_INDUSTRIES = 49


def main() -> None:
    """Fetch the Fama--French panel and write the compressed ``ff_sample.npz`` bundle."""
    data = load_fama_french(N_MONTHS, n_industries=N_INDUSTRIES)
    out = Path(__file__).resolve().parent / "ff_sample.npz"
    np.savez_compressed(
        out,
        dates=data.dates.astype(np.int64),
        industry_excess=data.industry_excess.astype(np.float64),
        factor_returns=data.factor_returns.astype(np.float64),
        factor_names=np.array(data.factor_names),
        industry_names=np.array(data.industry_names),
    )
    kb = out.stat().st_size / 1024
    print(
        f"wrote {out} ({kb:.0f} KB): {data.industry_excess.shape} industries, "
        f"{data.factor_returns.shape} factors, {data.dates[0]}-{data.dates[-1]}"
    )


if __name__ == "__main__":
    main()
