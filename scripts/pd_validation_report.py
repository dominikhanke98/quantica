#!/usr/bin/env python
"""Generate the PD-model validation report for the README.

A champion logistic regression and a gradient-boosting challenger are fitted to
the seeded synthetic credit portfolio (known true PDs, planted nonlinearity) and
pushed through the full validation battery:

1. **Discrimination** — AUC / Gini / KS with stratified-bootstrap CIs, plus the
   true-PD score as the discrimination ceiling only synthetic data can show.
2. **Calibration** (the centerpiece) — the per-grade table with exact-binomial
   and ECB Jeffreys p-values, and Hosmer--Lemeshow per model.
3. **Stability** — score PSI and per-characteristic PSI against a monitoring
   sample with a planted leverage drift.
4. **Validate the validators** — the size/power table for the calibration tests
   themselves, measured on grades with known true PDs.

Everything is seeded and deterministic; the README embeds this output verbatim.
Requires scikit-learn (a dev/demo dependency). Regenerate with::

    python scripts/pd_validation_report.py
"""

from __future__ import annotations

import io
import sys

import numpy as np
from quantica.risk.credit import (
    auc,
    binomial_test,
    characteristic_stability,
    discrimination_report,
    generate_credit_portfolio,
    grade_calibration,
    hosmer_lemeshow,
    jeffreys_test,
    psi,
)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

N_OBLIGORS = 30_000
N_TRAIN = 18_000
N_MONITOR = 10_000
DATA_SEED = 42
MONITOR_SEED = 43
BOOT_SEED = 7
N_BOOT = 500
LEVERAGE_DRIFT = 0.35
N_GRADES = 7
TEST_SIZE = 0.05


def fit_models() -> dict[str, np.ndarray]:
    sample = generate_credit_portfolio(N_OBLIGORS, np.random.default_rng(DATA_SEED))
    x_tr, y_tr = sample.features[:N_TRAIN], sample.defaults[:N_TRAIN]
    x_te = sample.features[N_TRAIN:]
    champion = LogisticRegression(max_iter=1000).fit(x_tr, y_tr)
    challenger = HistGradientBoostingClassifier(random_state=0).fit(x_tr, y_tr)
    monitor = generate_credit_portfolio(
        N_MONITOR, np.random.default_rng(MONITOR_SEED), leverage_shift=LEVERAGE_DRIFT
    )
    return {
        "y": sample.defaults[N_TRAIN:],
        "true_pd": sample.true_pd[N_TRAIN:],
        "x_test": x_te,
        "champion": champion.predict_proba(x_te)[:, 1],
        "challenger": challenger.predict_proba(x_te)[:, 1],
        "x_monitor": monitor.features,
        "champion_monitor": champion.predict_proba(monitor.features)[:, 1],
        "feature_names": np.asarray(sample.feature_names),
        "dr_test": np.array([sample.defaults[N_TRAIN:].mean()]),
    }


def discrimination_section(d: dict[str, np.ndarray]) -> None:
    print("### 1. Discrimination (out of sample, stratified bootstrap 95% CIs)\n")
    print("| Model | AUC | Gini | KS |")
    print("| --- | --- | --- | --- |")
    for label, key in (("Champion (logit)", "champion"), ("Challenger (GBM)", "challenger")):
        rep = discrimination_report(d["y"], d[key], np.random.default_rng(BOOT_SEED), n_boot=N_BOOT)
        print(
            f"| {label} | {rep.auc.point:.3f} [{rep.auc.lower:.3f}, {rep.auc.upper:.3f}] "
            f"| {rep.gini.point:.3f} [{rep.gini.lower:.3f}, {rep.gini.upper:.3f}] "
            f"| {rep.ks.point:.3f} [{rep.ks.lower:.3f}, {rep.ks.upper:.3f}] |"
        )
    print(f"| True PD (ceiling) | {auc(d['y'], d['true_pd']):.3f} | — | — |")
    print(
        "\nThe challenger recovers most of the planted nonlinearity (interaction + "
        "convexity) the linear champion cannot represent — its AUC sits close to the "
        "true-PD ceiling, a comparison only synthetic data affords.\n"
    )


def calibration_section(d: dict[str, np.ndarray]) -> None:
    print("### 2. Calibration per rating grade (the centerpiece)\n")
    for label, key in (("Champion (logit)", "champion"), ("Challenger (GBM)", "challenger")):
        hl = hosmer_lemeshow(d["y"], d[key])
        print(
            f"**{label}** — Hosmer–Lemeshow χ²({hl.dof}) = {hl.statistic:.1f}, "
            f"p = {hl.p_value:.2e} → "
            f"{'**rejected**' if hl.reject(TEST_SIZE) else 'passes'}\n"
        )
        print("| Grade | n | Defaults | Mean PD | Observed DR | Binomial p | Jeffreys p |")
        print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in grade_calibration(d["y"], d[key], n_grades=N_GRADES):
            flag = " ⚠" if r.jeffreys_p < TEST_SIZE else ""
            print(
                f"| {r.grade} | {r.n_obligors} | {r.n_defaults} | {r.mean_pd:.4f} "
                f"| {r.observed_rate:.4f} | {r.binomial_p:.3f} | {r.jeffreys_p:.3f}{flag} |"
            )
        print()
    print(
        "Both models are flagged (⚠ = Jeffreys p < 0.05). The mis-specified champion "
        "fails Hosmer–Lemeshow outright, and its *safest* grade is the most mis-rated "
        "— obligors with extreme negative leverage are assigned tiny PDs while the "
        "planted convexity makes them risky (observed DR ≈ 30× the assigned PD): a "
        "grade-level calibration table catches what a single AUC never would. The "
        "challenger — despite ranking best — still under-states PDs in specific "
        "grades. Ranking well and being calibrated are different properties; a "
        "challenger would be promoted only after recalibration.\n"
    )


def stability_section(d: dict[str, np.ndarray]) -> None:
    print("### 3. Stability (monitoring sample with a planted leverage drift)\n")
    score_psi = psi(d["champion"], d["champion_monitor"])
    print(f"Score PSI: **{score_psi.value:.3f}** → {score_psi.band}\n")
    print("| Characteristic | PSI | Band |")
    print("| --- | ---: | --- |")
    names = tuple(str(n) for n in d["feature_names"])
    for row in characteristic_stability(d["x_test"], d["x_monitor"], names):
        print(f"| {row.name} | {row.psi.value:.4f} | {row.psi.band} |")
    print(
        "\nThe characteristic view attributes the drift to the leverage distribution "
        "(the planted shift) while every other input stays stable — PSI on the score "
        "alone would say *that* the population moved, not *why*.\n"
    )


def size_power_section() -> None:
    rng = np.random.default_rng(0)
    n_trials = 5000

    def rate(defaults: np.ndarray, n: int, pd: float, which: str) -> float:
        cache = {
            int(x): (
                binomial_test(int(x), n, pd).p_value
                if which == "binomial"
                else jeffreys_test(int(x), n, pd).p_value
            )
            for x in np.unique(defaults)
        }
        return float(np.mean([cache[int(x)] < TEST_SIZE for x in defaults]))

    print("### 4. Validate the validators — size and power of the calibration tests\n")
    print(
        f"Simulated grades with known true PDs, {n_trials:,} trials, nominal size "
        f"{TEST_SIZE:.0%}; power measured against a PD understated by half:\n"
    )
    print("| Test | Grade | Size | Power (true DR = 2× PD) |")
    print("| --- | --- | ---: | ---: |")
    for n, pd0, label in ((800, 0.02, "n=800, PD=2%"), (150, 0.01, "n=150, PD=1% (low-default)")):
        d_null = rng.binomial(n, pd0, n_trials)
        d_alt = rng.binomial(n, 2 * pd0, n_trials)
        for which in ("binomial", "jeffreys"):
            name = "Exact binomial" if which == "binomial" else "Jeffreys (ECB)"
            print(
                f"| {name} | {label} | {rate(d_null, n, pd0, which):.1%} "
                f"| {rate(d_alt, n, pd0, which):.1%} |"
            )

    # Hosmer--Lemeshow: correct dof matters (true PDs -> chi^2 with G dof).
    hl_trials = 600
    size_g, size_g2, power = [], [], []
    for i in range(hl_trials):
        r = np.random.default_rng(1000 + i)
        p = 1.0 / (1.0 + np.exp(-r.normal(-3.5, 1.0, 4000)))
        y = (r.random(4000) < p).astype(float)
        size_g.append(hosmer_lemeshow(y, p, dof=10).reject(TEST_SIZE))
        size_g2.append(hosmer_lemeshow(y, p).reject(TEST_SIZE))
        power.append(hosmer_lemeshow(y, np.clip(p * 0.5, 1e-9, 1.0), dof=10).reject(TEST_SIZE))
    print(
        f"| Hosmer–Lemeshow (dof = G) | n=4000, true PDs | {np.mean(size_g):.1%} "
        f"| {np.mean(power):.1%} |"
    )
    print(f"| Hosmer–Lemeshow (dof = G−2) | n=4000, true PDs | {np.mean(size_g2):.1%} | — |")
    print(
        "\nHonest findings: the **exact binomial is conservative** (size well below "
        "nominal, collapsing in low-default grades) and pays for it in power; the "
        "**Jeffreys test holds near-nominal size and roughly doubles the power** on "
        "the low-default grade — the property for which the ECB instructions adopt "
        "it. And Hosmer–Lemeshow needs the right null: with true (non-estimated) "
        "PDs the statistic is χ²(G); the textbook G−2 convention (for models fitted "
        "on the sample) visibly over-rejects here.\n"
    )


def main() -> None:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    print("## PD-model validation report\n")
    d = fit_models()
    dr = float(d["dr_test"][0])
    print(
        f"Synthetic portfolio: {N_OBLIGORS:,} obligors ({N_TRAIN:,} development / "
        f"{N_OBLIGORS - N_TRAIN:,} validation), out-of-sample default rate {dr:.1%}. "
        f"Champion: logistic regression; challenger: gradient boosting "
        f"(scikit-learn; the validators consume only model outputs).\n"
    )
    discrimination_section(d)
    calibration_section(d)
    stability_section(d)
    size_power_section()


if __name__ == "__main__":
    main()
