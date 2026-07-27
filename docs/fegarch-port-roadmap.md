# quantica × fEGarch — porting roadmap

Scoping document for bringing the functionality of the **fEGarch** R package
(Schulz, Feng, Peitz, Ayensu et al., Paderborn University — v1.0.6, GPL-3) into
quantica's time-series pillar (Pillar V) as a dedicated long-memory-volatility module.

This is a multi-session sub-project, not one PR. The organizing principle is the
same effective-challenge discipline as the rest of quantica, with one enormous
advantage: **fEGarch itself is the reference implementation.** Every model gets
benchmarked against its R counterpart, turning this from "reimplement a paper" into
"independent Python reimplementation, validated against the published reference" —
which is exactly quantica's identity, now applied to a real package with a personal
connection (your coauthor's work).

---

## 0. Before any code — three decisions to make first

1. **Talk to your coauthor (Dominik Schulz) about attribution and positioning.**
   fEGarch is their published research (GPL-3, with an associated paper — Feng et al.
   2023 on the FIMLog-GARCH). A Python port that credits the source and benchmarks
   against fEGarch is collaborative and generous — but it's *their* models, so agree
   up front on: how it's attributed, whether it's framed as "a Python port of fEGarch"
   vs "quantica's long-memory GARCH module inspired by / validated against fEGarch,"
   and licensing (fEGarch is GPL-3; if you copy/closely-follow its code that has
   license implications — a clean-room reimplementation from the *paper and reference
   manual* is cleaner than porting the C++ line-by-line). **This could also become a
   joint thing** — a validated Python companion to their R package is a genuinely
   useful research artifact. Have this conversation before publishing anything.

2. **License check.** fEGarch is **GPL-3**. quantica is **MIT**. This matters: if you
   port fEGarch's C++/R *source*, GPL-3 is copyleft and would force the derived module
   (arguably the library) to GPL-3, which conflicts with your MIT license. The clean
   path is a **clean-room reimplementation from the mathematical specifications** (the
   reference manual and the underlying papers), not a translation of their code —
   which is both legally cleaner and a stronger validation story (independent
   implementation that still matches). Confirm this approach with your coauthor and,
   if in doubt, keep this module clearly delineated. *This is a real constraint, not a
   formality — decide it before writing code.*

3. **Benchmarking infrastructure.** The whole validation strategy rests on comparing
   to fEGarch's output. Decide how: (a) run fEGarch in R and export fitted
   parameters / conditional-variance series / forecasts as fixture data files that
   Python tests load and match against (cleanest — no R dependency in CI, deterministic
   fixtures committed to the repo), or (b) an `rpy2` bridge (heavier, fragile in CI).
   **Recommend (a):** generate reference fixtures once from fEGarch on known inputs,
   commit them, and assert Python matches them to tolerance. This keeps CI R-free.

---

## What fEGarch actually contains (the scope)

A very broad family, unified under one `fEGarch_spec()` → `fEGarch()` fitting
interface, all via **quasi-maximum-likelihood estimation** (conditioned on presample
values). Grouped by difficulty:

**Short-memory (SM) volatility models** — the foundation:
- GARCH, GJR-GARCH, TGARCH, APARCH (asymmetric power ARCH)
- EGARCH, Log-GARCH, MEGARCH (modified EGARCH), MLog-GARCH (modulus Log-GARCH)

**Long-memory (LM) / fractionally-integrated models** — the hard, distinctive core:
- FIGARCH, FIGJR-GARCH, FITGARCH, FIAPARCH (FI orders fixed at p=q=1 in fEGarch)
- FIEGARCH, FILog-GARCH, FIMLog-GARCH, FIMEGARCH (the EGARCH-family LM models — the
  package's namesake and its research contribution)

**Conditional distributions** (8, incl. skewed): normal (`norm`), Student-t (`std`),
GED (`ged`), average Laplace / ALD (`ald`), and skewed versions (`snorm`, `sstd`,
`sged`, `sald`).

**Mean models (dual modelling)**: constant, ARMA, and **FARIMA** (fractionally
integrated ARMA) in the mean — simultaneously with the variance model. Plus
GARCH-in-mean (`garchm`).

**Semiparametric extension**: a nonparametric local-polynomial step for a smooth
scale component (`locpol_spec`, `use_nonpar`) — the `smoots`/`esemifar` machinery.

**Simulation**: `*_sim` for every model (garch_sim, aparch_sim, figarch_sim, …).

**Forecasting**: `predict`, `predict_roll` (rolling forecasts without refitting).

**Risk (VaR/ES)**: `measure_risk`, `VaR_calc`, and backtests — `trafflight_test`
(Basel traffic light), loss functions, `predict_roll`-based rolling VaR/ES. **You
already have most of this in the risk pillar — it's a tie-back, not new work.**

**Diagnostics/tests**: `ljung_box_test`, `sign_bias_test`, `goodn_of_fit_test`,
`fit_test_suite`, information criteria, distribution estimation (`distr_est`).

---

## The dependency structure (why the order matters)

Everything rests on three foundations that must come first and be rock-solid, because
every later model reuses them:

1. **The conditional-distribution layer** (8 distributions with pdf/cdf/quantile/rng,
   and their skew parameterizations) — every QMLE likelihood needs these.
2. **The QMLE estimation engine** (a generic negative-log-likelihood optimizer with
   the presample-conditioning convention, parameter bounds, and a Hessian-based
   vcov/standard errors) — every model plugs into this.
3. **The fractional-differencing engine** (the `(1−L)^d` operator via its truncated
   infinite MA/binomial expansion, with a truncation-length policy) — every FI- model
   needs this, and it's the single hardest and most error-prone numerical piece.

Get those three right and validated, and the models become (relatively) mechanical
specifications on top. Rush them and every downstream model inherits the bug.

---

## Phased roadmap (each phase = 1–3 PRs, each independently validated vs fEGarch)

### Phase 0 — foundations (do first, no models yet)
- **Conditional distributions** (`quantica/timeseries/distributions.py` or similar):
  norm, std (Student-t), ged, ald, and skewed snorm/sstd/sged/sald — pdf, cdf,
  quantile, sampler, each with the exact fEGarch parameterization. Validate: pdf
  integrates to 1; moments; quantile∘cdf round-trip; **match fEGarch's density values
  on fixture points**.
- **QMLE engine**: generic conditional-likelihood maximization with presample
  conditioning, bounds, and Hessian standard errors (lean on scipy.optimize; the
  demonstrable skill is the likelihood construction + the conditioning convention,
  not the optimizer). Validate against a hand-built simple case.
- Headline: the distribution layer matches fEGarch to tolerance on fixtures.

### Phase 1 — short-memory foundation
- GARCH, GJR-GARCH, TGARCH, APARCH under all 8 distributions, via the QMLE engine.
  (GARCH/GJR you have via `arch` — but here they must go through the *unified*
  fEGarch-style interface and match fEGarch's QMLE, which uses presample conditioning
  and may differ slightly from arch's defaults — document the convention.)
- Simulation (`*_sim`) for each.
- Validate: **fitted parameters and conditional-variance series match fEGarch** on the
  bundled SP500/UKinflation data to tolerance; simulation recovers known parameters.
- Headline: the SM models reproduce fEGarch's fits.

### Phase 2 — the EGARCH family (SM)
- EGARCH, Log-GARCH, MEGARCH, MLog-GARCH. These are the package's core family (log-
  variance specifications), distinct from Phase 1's level/power specs.
- Validate vs fEGarch fits + simulation.

### Phase 3 — the fractional-differencing engine (the crux)
- Implement `(1−L)^d` truncated expansion, the FARIMA/long-memory filter machinery,
  and the truncation-length policy (`trunc`, `presample` args in fEGarch).
- Validate in isolation: the fractional-difference of a known series matches the
  analytic binomial-coefficient expansion; long-memory autocorrelation decay is
  hyperbolic (not exponential) as it should be; **match fEGarch's `close_to_lreturn`
  / lin_filters output** on fixtures. This phase ships the *engine*, tested alone,
  before any FI model uses it — because a bug here poisons everything downstream.

### Phase 4 — long-memory volatility models
- FIGARCH, FIGJR-GARCH, FITGARCH, FIAPARCH (fixed p=q=1, as fEGarch does).
- Then FIEGARCH, FILog-GARCH, FIMLog-GARCH, FIMEGARCH (the EGARCH-family LM models —
  the research namesake).
- Validate: **fitted `d` and parameters match fEGarch** to tolerance — this is the
  headline of the whole sub-project, because matching a long-memory QMLE fit to a
  published reference is genuinely hard and genuinely impressive.

### Phase 5 — dual mean modelling
- ARMA and FARIMA mean models fitted simultaneously with the variance model;
  GARCH-in-mean. Validate vs fEGarch dual fits.

### Phase 6 — forecasting, risk, diagnostics (mostly tie-back)
- `predict` / `predict_roll` (rolling forecasts without refitting).
- VaR/ES via these models + backtests — **wire into the existing risk pillar** (you
  already have Kupiec/Christoffersen/traffic-light/Acerbi–Székely; fEGarch's
  `trafflight_test` and `measure_risk` map onto them). This is where the new
  volatility models feed your existing validation machinery — the coherence payoff.
- Diagnostics: Ljung-Box, sign-bias test, goodness-of-fit, info criteria (several you
  have; add the missing ones).

### Phase 7 (optional) — semiparametric extension
- The nonparametric local-polynomial scale step (`locpol_spec`, `smoots`/`esemifar`
  machinery). Most specialized; do last or skip if scope needs trimming.

---

## Validation strategy (the spine, every phase)

1. **fEGarch fixtures are the reference.** Generate once in R on fixed inputs (SP500,
   UKinflation, simulated series with known seeds), export fitted params, conditional
   variances, forecasts, VaR/ES → commit as test fixtures. Python asserts a match to
   tolerance. No R in CI.
2. **Known-truth recovery.** Simulate from each model with known parameters; confirm
   the QMLE recovers them (the anchor that doesn't need fEGarch).
3. **Reduction anchors.** FI models → their SM counterparts as `d → 0`; skewed
   distributions → symmetric as skew → neutral; APARCH → GARCH at δ=2, etc.
4. **Cross-checks vs `arch`** where the model overlaps (plain GARCH/GJR/EGARCH), noting
   convention differences (presample conditioning).
5. **The fractional engine tested in isolation** before any FI model uses it.

Every phase keeps the gate green, docstring coverage 100%, feature-branch + PR.

---

## Honest scope assessment

This is the largest single undertaking in quantica's history — realistically **~7–12
PRs across many sessions.** Phases 0 and 3 (distributions/QMLE engine and the
fractional-differencing engine) are the hard, load-bearing ones; get them right and
the rest is disciplined specification work. The long-memory QMLE (Phase 4) matching
fEGarch to tolerance is the headline artifact and the hardest validation.

It's worth it: a **clean-room, MIT-licensed-or-appropriately-licensed, fully-validated
Python implementation of a serious long-memory GARCH family — benchmarked against the
authors' own R package — is something essentially no Python library offers**
(`arch` stops at short-memory; nothing mainstream does FIEGARCH/FIMLog-GARCH well).
That's not "another pillar"; it's a potentially field-useful contribution, and with
your coauthor's involvement, a genuinely distinctive one.

Do Phase 0 first. Don't write a single model until the distributions, the QMLE engine,
and — separately — the fractional-differencing engine are implemented and validated,
because everything downstream inherits their correctness.
