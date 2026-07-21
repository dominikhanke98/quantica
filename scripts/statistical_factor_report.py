#!/usr/bin/env python
"""Statistical (PCA) factor model — recovery, the RMT cutoff, and statistical vs observable.

Three artifacts:

1. **Known-truth recovery** (synthetic, no network) — returns are generated from a planted
   low-rank factor structure; the Marchenko--Pastur cutoff recovers the true *number* of
   factors and the recovered loadings span the true factor *subspace* (principal-angle
   check). Pure noise yields zero factors. This is the headline correctness result.

2. **Scree and the RMT cutoff** (real data) — the eigenvalue spectrum of the industry
   correlation matrix, the variance each component explains, and how many factors the three
   selection rules keep: variance-explained, the scree elbow, and the Marchenko--Pastur
   edge (which separates signal eigenvalues from the noise bulk).

3. **Statistical vs observable** (real data) — the PCA model against the Fama--French
   observable model on the *same* universe: how much variance each explains, whether the
   top statistical factors line up with the observable ones (does PC1 look like the
   market?), and which forecasts risk better out of sample (the stage-2 comparison).

Sections 2--3 fetch Ken French data (cached via ``scripts/_ff_data.py``; never in CI).
Regenerate with::

    python scripts/statistical_factor_report.py

The README embeds a captured run.
"""

from __future__ import annotations

import io
import sys

import numpy as np
from _ff_data import load_fama_french
from quantica.factor import (
    FactorCovariance,
    FactorRiskModel,
    LedoitWolfCovariance,
    SampleCovariance,
    StatisticalFactorCovariance,
    StatisticalFactorModel,
    compare_estimators,
    generate_factor_data,
    marchenko_pastur_edges,
    marchenko_pastur_rank,
    scree_elbow_rank,
    subspace_similarity,
    variance_explained_rank,
)

_N_MONTHS = 300
_N_INDUSTRIES = 49
_TRAIN_WINDOW = 60
_TEST_WINDOW = 12
_ANNUALISE = np.sqrt(12.0)


def _known_truth_section() -> None:
    """PCA recovers the planted factor count and subspace; noise yields none."""
    print("### 1. Known-truth recovery (synthetic, no network)\n")
    print("| Universe | True k | MP-recovered k | Subspace similarity |")
    print("| --- | ---: | ---: | ---: |")
    for seed in range(4):
        rng = np.random.default_rng(seed)
        n, k_true = 40, 3
        betas = rng.uniform(-1.0, 1.0, size=(n, k_true))  # mixed sign -> clean spectral gap
        data = generate_factor_data(
            600,
            betas,
            rng,
            specific_vols=0.04,
            factor_vols=np.array([0.09] * k_true),
            factor_names=tuple(f"F{i}" for i in range(k_true)),
        )
        model = StatisticalFactorModel.fit(data.asset_returns)
        sim = subspace_similarity(model.betas, data.true_betas)
        print(f"| 40 assets, seed {seed} | {k_true} | {model.n_factors} | {sim:.4f} |")
    rng = np.random.default_rng(99)
    noise = rng.standard_normal((600, 40))
    noise_k = marchenko_pastur_rank(np.linalg.eigvalsh(np.corrcoef(noise, rowvar=False)), 40, 600)
    print(f"| 40 assets, **pure noise** | 0 | **{noise_k}** | — |")
    print(
        "\nThe Marchenko--Pastur cutoff recovers the exact planted factor count and the "
        "loadings span the true factor space (largest principal angle ~ 0); on pure noise it "
        "certifies **zero** real factors. Individual components are not identified — the "
        "*span* is, which is what the subspace-similarity check tests.\n"
    )


def _scree_section(model: StatisticalFactorModel, n_obs: int) -> None:
    """The spectrum, variance explained, and the three selection rules on real data."""
    evr = model.explained_variance_ratio
    cum = model.cumulative_variance_ratio
    _, lam_plus = marchenko_pastur_edges(len(evr), n_obs)

    print("### 2. Scree and the RMT cutoff (49 industries)\n")
    print(
        f"{_N_INDUSTRIES} industries, {n_obs} months; Marchenko--Pastur upper edge "
        f"(sigma^2 = 1) at lambda+ = {lam_plus:.2f}. Leading eigenvalues:\n"
    )
    print("| PC | Eigenvalue | Variance explained | Cumulative | Above MP edge? |")
    print("| ---: | ---: | ---: | ---: | :---: |")
    for j in range(6):
        flag = "yes" if model.eigenvalues[j] > lam_plus else "no"
        print(f"| {j + 1} | {model.eigenvalues[j]:.2f} | {evr[j]:.1%} | {cum[j]:.1%} | {flag} |")

    mp_plain = marchenko_pastur_rank(model.eigenvalues, len(evr), n_obs, adjust_variance=False)
    mp_adj = marchenko_pastur_rank(model.eigenvalues, len(evr), n_obs, adjust_variance=True)
    var_k = variance_explained_rank(model.eigenvalues, 0.9)
    scree_k = scree_elbow_rank(model.eigenvalues)
    print("\n| Selection rule | Factors kept |")
    print("| --- | ---: |")
    print(f"| Marchenko--Pastur, sigma^2 = 1 | {mp_plain} |")
    print(f"| Marchenko--Pastur, bulk refit | {mp_adj} |")
    print(f"| variance-explained >= 90% | {var_k} |")
    print(f"| scree elbow | {scree_k} |")
    print(
        f"\nThe first eigenvalue ({model.eigenvalues[0]:.0f}, ~{evr[0]:.0%} of the variance) "
        "is the **market** mode — far outside the noise bulk. The plain sigma^2 = 1 cutoff "
        f"flags {mp_plain} clean factors; because that dominant mode depletes the bulk, "
        f"refitting the bulk variance lifts the count to {mp_adj} (at the cost of some "
        "sensitivity to finite-size stragglers near the edge). Reaching 90% of the variance "
        f"takes {var_k} components — most of them noise — which is exactly why a variance "
        "target over-keeps and the RMT cutoff is the sharper, more honest rule.\n"
    )


def _comparison_section(industries: np.ndarray, factors: np.ndarray, factor_names) -> None:  # type: ignore[no-untyped-def]
    """Statistical vs observable: variance explained, factor alignment, OOS risk forecasting."""
    stat = StatisticalFactorModel.fit(industries, n_components=4)  # match the 4 observable factors
    obs = FactorRiskModel.fit(industries, factors, factor_names=tuple(factor_names))

    # Variance explained: mean systematic fraction (communality / R^2) per asset.
    stat_sys = np.mean(np.diag(stat.systematic_covariance()) / np.diag(stat.covariance()))
    obs_r2 = float(np.mean([e.r_squared for e in obs.exposures]))

    # Do the statistical factors line up with the observable ones? Correlate PC scores.
    standardized = (industries - industries.mean(0)) / industries.std(0, ddof=1)
    corr = np.atleast_2d(np.cov(standardized, rowvar=False, ddof=1))
    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    pc_scores = standardized @ eigvecs[:, order[:4]]  # (T, 4) principal-component returns

    print("### 3. Statistical vs observable factors (same 49-industry universe)\n")
    print("| Model | Factors | Mean variance explained |")
    print("| --- | ---: | ---: |")
    print(f"| Fama--French observable | 4 (MKT/SMB/HML/MOM) | {obs_r2:.1%} |")
    print(f"| PCA statistical | 4 (PC1--PC4) | {stat_sys:.1%} |")
    print(
        "\nCorrelation of each statistical factor with the observable factors "
        "(|rho|, sign of a PC is arbitrary):\n"
    )
    print("| | " + " | ".join(factor_names) + " |")
    print("| --- | " + " | ".join(["---:"] * len(factor_names)) + " |")
    for j in range(4):
        cells = [
            f"{abs(np.corrcoef(pc_scores[:, j], factors[:, m])[0, 1]):.2f}"
            for m in range(len(factor_names))
        ]
        print(f"| PC{j + 1} | " + " | ".join(cells) + " |")

    pc1_mkt = abs(np.corrcoef(pc_scores[:, 0], factors[:, 0])[0, 1])
    print(
        f"\n**PC1 is the market**: it correlates {pc1_mkt:.2f} with Mkt-RF — the dominant "
        "statistical factor is the observable market factor, discovered without being told. "
        "The lower PCs blend the style factors (no one-to-one match, since PCA maximises "
        "variance, not economic interpretability).\n"
    )

    comparison = compare_estimators(
        industries,
        (
            SampleCovariance(),
            LedoitWolfCovariance(),
            StatisticalFactorCovariance(n_components=4),
            FactorCovariance(),
        ),
        train_window=_TRAIN_WINDOW,
        test_window=_TEST_WINDOW,
        factor_returns=factors,
        rng=np.random.default_rng(0),
    )
    vols = comparison.mean_min_variance_vol()
    print(
        "Out-of-sample min-variance realised volatility (annualised), "
        f"{_TRAIN_WINDOW}-month train / {_TEST_WINDOW}-month test:\n"
    )
    print("| Estimator | Realised OOS vol |")
    print("| --- | ---: |")
    for name in ("sample", "ledoit-wolf", "statistical-factor", "factor"):
        print(f"| {name} | {vols[name] * _ANNUALISE:.1%} |")
    print(
        "\nBoth factor models forecast risk far better than the sample covariance, which "
        "inverts a near-singular matrix. **Honest finding:** on real industry data the "
        "*statistical* four-factor model (matching Ledoit--Wolf) actually edges the "
        "*observable* four-factor model out of sample — the four Fama--French factors do not "
        "fully span industry systematic risk, whereas PCA extracts the four directions that "
        "genuinely dominate the covariance. The trade-off is interpretability: the observable "
        "factors are economic, investable and stable; the statistical factors are whatever "
        "the covariance says, re-estimated each window. PCA wins the pure risk-forecasting "
        "contest it is built for; the observable model wins on economic meaning — the same "
        "'which model to trust when' verdict as the estimator comparison.\n"
    )


def main() -> None:
    """Print the recovery, the scree/RMT cutoff, and the statistical-vs-observable sections."""
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("## Statistical (PCA) factor model — recovery, RMT cutoff, and the observable tie-back\n")
    _known_truth_section()

    data = load_fama_french(_N_MONTHS, n_industries=_N_INDUSTRIES)
    full = StatisticalFactorModel.fit(data.industry_excess, n_components=_N_INDUSTRIES)
    _scree_section(full, data.industry_excess.shape[0])
    _comparison_section(data.industry_excess, data.factor_returns, data.factor_names)


if __name__ == "__main__":
    main()
