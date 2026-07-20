# PROGRESS

Running state file for `quantica`. Updated at the end of each working session
(see CLAUDE.md §"Session close-out"). Concise and factual.

**Current phase:** **All three pillars complete AND deployed — the originally-planned
scope of `quantica` is closed.** Derivatives-pricing track complete (Phase 1 core +
Phase 4 deepening). **Phase 3 (quant risk / model validation) complete** across five
families (market risk, derivatives P&L, FRTB PLA, credit/PD, ML under SR 11-7).
**Phase 2 (systematic portfolio management) complete** — construction + walk-forward
backtest + the backtest-validity layer (DSR / PBO / purged CV / MinTRL), built on the
factor track. The derivatives / risk / portfolio triad (CLAUDE.md §9) is closed, and the
deferred **thin Streamlit apps (step 8) are merged to `main` (PR #1) and LIVE** on
Streamlit Community Cloud at **https://quantica.streamlit.app/** (linked from the README
top matter). Everything the CLAUDE.md brief set out to build now exists, is validated,
and is demonstrable in one click. Next: optional depth only — see "Next".

Capital-markets roadmap: **multi-factor risk model — stage 1 ✓** (exposures +
decomposition + Σ = BFBᵀ + D) → **stage 2 ✓** (OOS estimator comparison: sample vs
Ledoit–Wolf vs factor; ill-conditioning demo; bias stats) → **portfolio construction +
backtest + validity layer ✓** → **statistical (PCA/RMT) factor model ✓** (completes the
observable-vs-statistical pair). **Capital-markets / portfolio track complete.**

Phase-4 roadmap: **American ✓** → **LSM ✓** → **exotics ✓** → **Heston pricer ✓**
→ **Heston calibration ✓** → **Merton jump-diffusion ✓** → **autocallable note ✓**.
**Derivatives-pricing deepening track complete.**

Phase-3 roadmap: **market-risk VaR/ES + backtesting ✓** → **derivatives-P&L
integration ✓** (option book revalued through the pricers as the risk P&L source)
→ **credit-risk / PD validation ✓** → **ML-model validation (SR 11-7) ✓** →
**FRTB P&L attribution ✓** (IMA-eligibility test reusing the derivatives-risk P&L).
**The risk pillar's planned model families are complete.**

## Completed

- **Project skeleton** — packaging (`pyproject.toml`), ruff + mypy + pytest
  config, CI workflow, `.gitattributes` (LF), tracked `numerical-validation`
  skill under `.claude/skills/`.
- **Step 1 — core types / instrument / process** — `OptionType`,
  `EuropeanOption` (payoff + engine seam), `BlackScholesProcess` (frozen, with
  `with_*` bump helpers, `discount_factor`, `forward`).
- **Step 2 — analytic engine + Greeks** — `AnalyticEuropeanEngine`: Black–Scholes
  closed form + delta/gamma/vega/theta/rho, validated vs bump-and-reval and
  QuantLib (price + all Greeks to `rtol≈1e-10`).
- **Step 3 — implied volatility** — `implied_volatility`: safeguarded Newton
  (via vega) inside a Brent bracket; no-arbitrage-band error handling; round-trip
  + QuantLib-solver benchmark.
- **Step 4 — CRR binomial engine** — `BinomialEngine`: backward induction,
  continuous dividends; `O(1/N)` convergence verified by log-log slope (even-N
  subsequence, sawtooth handled); QuantLib CRR benchmark.
- **Step 5 — Monte Carlo engine** — `MonteCarloEngine`: exact GBM terminal
  simulation, injected seeded `Generator`, antithetic + control variates,
  `estimate()` exposes the standard error; within ~3 SE of analytic; variance
  reduction demonstrated (VRF ~2× antithetic, ~7× control).
- **Step 6 — Crank–Nicolson PDE engine** — `FiniteDifferenceEngine`: BS PDE on a
  log-price grid, CN scheme, tridiagonal solve; second-order `O(h²)` convergence
  verified by a log-log slope of ≈ −2; parity up to discretisation; QuantLib FD
  benchmark.
- **Step 7 — four-way cross-method convergence test** —
  `tests/pricing/test_cross_method.py`: prices the same option under all four
  engines across strikes / maturities / call-put with non-zero dividend, each
  anchored to analytic to a justified per-method tolerance (CRR `O(1/N)` < 2e-3
  at N=2000; PDE `O(h²)` < 1.5e-3 at 500×500; MC within 3 SE, seeded). Completes
  the derivatives-pricing core.
- **Convergence table** — `scripts/convergence_table.py` (seeded, reproducible),
  spans analytic / CRR / MC / PDE; embedded verbatim in the README, which frames
  it as the effective-challenge centrepiece.
- **Phase 4, step 1 — American options** — `ExerciseStyle` enum; `VanillaOption`
  base with `EuropeanOption` / `AmericanOption` subclasses (shared payoff + engine
  seam). `BinomialEngine` early exercise via `max(continuation, intrinsic)`;
  `FiniteDifferenceEngine` via the LCP (projected SOR on the CN tridiagonal
  system). Analytic + MC engines reject American. Validated by tree↔PDE
  cross-agreement, QuantLib American benchmarks, and exact theorems
  (no-dividend American call = European to machine precision; premium ≥ 0).
- **Phase 4, step 2 — Longstaff–Schwartz Monte Carlo** — `LongstaffSchwartzEngine`
  (`engines/lsm.py`): full-path exact log-GBM simulation on an exercise-date grid;
  backward induction regressing discounted continuation on a monomial basis
  (strike-scaled) using in-the-money paths only; value by realized cashflows.
  Configurable `exercise_dates` / `basis_degree` (default 50 / 3), seeded
  `Generator`, antithetic, `estimate()` exposes the SE. Reuses `MCResult`; the
  terminal-only `MonteCarloEngine` fast path is untouched. Validated against the
  tree/PDE American references within ~3 SE, with the low-bias/lower-bound
  signature confirmed (mean over seeds sits ~5e-3 below reference; richer basis
  recovers more value); no-dividend call recovers European; SE ~ 1/√n; seeded
  determinism.

- **Phase 4, step 3 — path-dependent exotics** — `GBMPathSimulator`
  (`engines/_paths.py`) extracted from LSM (shared full-path GBM + antithetic;
  LSM behaviour unchanged). **Asian** (`engines/asian.py`): `geometric_asian_price`
  closed form (QuantLib-exact with aligned dates); `AsianMonteCarloEngine` prices
  arithmetic/geometric, with the geometric payoff as a control variate for the
  arithmetic price (VRF ~880×). **Barrier** (`engines/barrier.py`): `barrier_price`
  Reiner–Rubinstein closed form (QuantLib-exact, all 8 types); `BarrierMonteCarloEngine`
  with discrete monitoring and an optional Brownian-bridge correction. Validated:
  geometric MC↔closed form (3 SE); discrete-monitoring bias direction + shrinkage;
  bridge recovers the continuous price and beats discrete at fixed step count;
  in-out parity exact.
- **Phase 4, step 4a — Heston process refactor** — factored a lightweight frozen
  `Market` carrier (spot, rate, div) out of the processes; `BlackScholesProcess`
  stays flat/backward-compatible and gains `.market`/`from_market`; new
  `HestonProcess` (v0, kappa, theta, xi, rho + `feller_satisfied`). Resolved the
  ignored-vol TODO: `implied_volatility` now takes a `Market` (no placeholder vol).
- **Phase 4, step 4b — Heston pricer** — `HestonFFTEngine` (`engines/heston.py`):
  Carr–Madan FFT of the characteristic function, with the branch-cut-stable
  "little Heston trap" CF from the start; strike placed on an exact FFT node
  (no interpolation); puts via parity; configurable alpha/n_fft/eta. Validated:
  reduces to Black–Scholes as xi→0 (~2e-7, the featured anchor); CF correct at
  t=0 / u=0; put–call parity; arbitrage-free monotonicity; alpha/grid stability;
  QuantLib AnalyticHestonEngine benchmark (~1e-7, integer-day maturities).
  Short-maturity diagnostic: an apparent ~1.9e-2 "error" at T=0.1 was a day-count
  artifact (round(365·0.1)=36 days ≈ 0.0986 yr); an independent scipy-quadrature
  truth confirmed our FFT is exact to ~1e-13 there, i.e. more accurate than
  QuantLib's *default* AnalyticHestonEngine at short expiry — so benchmarks use
  integer-day maturities to align conventions.
- **Phase 4, step 4c — Heston calibration** — `calibrate_heston`
  (`pricing/calibration.py`): fits `(v0, kappa, theta, xi, rho)` to a vanilla
  implied-vol surface by `scipy.optimize.least_squares` over `HestonFFTEngine`
  prices, reusing the step-3 `implied_volatility` solver to invert model prices on
  the **OTM** option (best-conditioned vega). **Vol space is the default** (doesn't
  overweight expensive ITM quotes); `space="price"` available. `HestonParams` /
  `VolQuote` / `ParamBounds` / `DEFAULT_BOUNDS` / `vol_surface_from_grid` supporting
  types; box bounds passed straight to the optimizer; non-finite pricer corners
  (CF overflow) penalised so the solver is repelled, not crashed. **Feller**
  reported via `feller_satisfied` by default, optional soft penalty (`feller_weight`).
  **Identifiability** surfaced two ways: `profile_objective` (pin one param,
  re-optimise the rest → valley-width read-out: κ valley ≈ 3× wider than ρ) and
  multi-start `param_spread`. Multi-start seeded (deterministic; default
  `default_rng(0)`). Validated (`tests/pricing/test_calibration.py`, 17 tests):
  noise-free synthetic recovery to solver tolerance (headline); tight fit quality
  (RMSE < 1e-5 vol); noisy recovery shows v0/θ tight (~3%) vs κ loose (~20%);
  κ-valley broader than ρ-valley; Feller flag + penalty both exercised; seeded
  determinism; weights applied. QuantLib benchmark: our fit and QuantLib's
  `HestonModelHelper` + Levenberg–Marquardt both recover the truth and agree.
  Report script `scripts/heston_calibration_report.py` (synthetic recovery,
  realistic non-Heston smile fit @0.245 vol-pt RMSE, identifiability) → embedded in
  the README.
- **Phase 4, step 5 — Merton jump-diffusion** — `MertonProcess` (`processes.py`:
  σ, λ, μ_J, σ_J composing the `Market` carrier; `.compensator` κ̄; forward
  unaffected by jumps via the drift compensator). Priced **two independent ways**
  (`engines/merton.py`): `MertonClosedFormEngine` (Poisson-weighted sum of BS prices
  — conditional on n jumps it's BS with inflated variance `σ_n²=σ²+nσ_J²/T` and an
  effective dividend `q_n` so each term's forward matches while the discount stays
  `e^{-rT}`; each BS term delegated to `AnalyticEuropeanEngine`; series truncated at
  a documented tol, tail bounded by max(S,K)); and `MertonFFTEngine` (Merton CF into
  the shared Carr–Madan transform). **Refactor:** extracted the Carr–Madan transform
  into `engines/_carr_madan.py` (`carr_madan_call_price` taking a CF callable) now
  that a second model needs it; `HestonFFTEngine` refactored onto it (behaviour +
  benchmarks unchanged). Validated (`tests/pricing/engines/test_merton.py`, 58
  tests): **closed-form vs FFT agree to ~2e-7 (headline, self-anchored)**; BS limit
  (λ→0); CF at known points; Poisson-series monotone convergence + truncation error
  below stated tol; parity; arbitrage-free monotonicity; α/grid stability; the
  negative-skew jump smile. **No QuantLib benchmark**: `Merton76Process` exists but
  the `JumpDiffusionEngine` is not exposed in the QuantLib Python wrapper, so the
  closed-form-vs-FFT agreement is the rigorous check (documented). Demo
  `scripts/jump_diffusion_smile.py`: Merton vs Heston smile at the same baseline vol
  — Merton's short-dated smile is ~5× steeper than its long-dated one (jumps), vs
  Heston's ~1.2× → embedded in the README.
- **Phase 3, step 1 — market-risk core + backtesting** — new `quantica/risk/`
  package. `Portfolio` (`portfolio.py`): weights × value → a P&L / loss *series*
  (deliberately a series, not the asset matrix, so an option book revalued through
  the pricers can later replace the linear portfolio without touching risk code).
  `measures.py`: `RiskEstimate`, `normal_var_es` (Gaussian closed form — the
  analytic anchor), `empirical_var_es` (Rockafellar–Uryasev tail-mean, stable).
  Four engines (`engines.py`, shared `VaREngine` protocol): `HistoricalSimulationVaR`,
  `ParametricVaR` (variance–covariance; normality caveat documented),
  `MonteCarloVaR` (MV-normal sim, seeded; converges to parametric — cross-check),
  `FilteredHistoricalSimulationVaR` (GARCH(1,1) via `arch`, lazy import; bootstraps
  standardised residuals scaled by the 1-step vol forecast). **Backtesting layer
  (the deliverable, `backtest.py`)**: `kupiec_pof` (unconditional coverage),
  `christoffersen_independence` + `christoffersen_cc`, `basel_traffic_light`
  (green/yellow/red + multiplier add-on), and — the highlight — `acerbi_szekely`
  (2014) Z1/Z2 **ES** backtest with a Monte-Carlo null (ES is not elicitable,
  Gneiting 2011, so naive ES backtests fail). `rolling_var_forecasts` for
  out-of-sample one-step backtests. Validated (`tests/risk/`, 46 tests): analytic
  anchors (parametric == closed form on sample moments; MC → parametric; HS →
  parametric on large normal); backtest correctness on hand-checkable cases; **and
  the meta-challenge — size & power of the backtests themselves**: Kupiec size ~4%
  / power ~1.0, Acerbi–Székely size ~4.6% / power ~1.0, Christoffersen independence
  *conservative* (~2%, honest finding for rare 99% exceptions) / power ~0.76 vs
  clustering. Deps: added `arch>=6.3` (runtime, lazy-imported) + mypy overrides for
  `arch.*`/`pandas.*`. Report `scripts/risk_backtest_report.py`: the size/power
  table + a GARCH-t worked backtest where parametric-normal hits the Basel **red**
  zone while filtered-HS stays **green** → embedded in the README.
- **Phase 3, step 2 — derivatives-P&L integration** — `quantica/risk/derivatives.py`
  ties the two pillars. `OptionBook` (positions = instrument + *its own pricing
  engine* + signed quantity, plus `underlying_quantity` for hedged books) +
  `MarketScenarios` (seeded instantaneous spot returns and optional additive vol
  shifts; theta drops out by design). Three P&L methods on the *same* scenario set
  (so divergence is approximation error, not noise): `full_revaluation_pnl`
  (reprice through the engines — the risk path IS the pricing path, no drift),
  `delta_normal_pnl` (Δ·δS + ν·δσ), `delta_gamma_pnl` (+ ½Γ·δS²). Book Greeks via
  central-difference bump-and-reval through each position's engine (consistent
  with the pricing numerics; matches analytic Greeks for European). `book_var_es`
  adapter → `empirical_var_es(-pnl)`; **risk/backtest layer untouched** (the
  P&L-series seam doing its job). Validated (`tests/risk/test_derivatives.py`, 16
  tests): no-drift consistency to the last bit; bump Greeks == analytic; small-move
  full == linear; delta-hedged book isolates ½Γ·δS²; **headline divergence** —
  short-gamma book: delta-normal VaR −41% vs full (underestimates, the omitted
  −½|Γ|δS² is pure loss), long-gamma: +197% (overestimates, gamma cushions),
  near-linear: agree to 4 decimals, delta-gamma repairs to ~1%; Kupiec reused
  unchanged rejects the delta-normal forecast (43 exceptions vs 7.5 expected) and
  passes delta-gamma/full (8, p=0.856); mixed analytic+binomial-American book;
  seeded determinism. Report `scripts/derivatives_var_report.py` → both tables
  embedded in the README. Scope note: BS-process books (scenario = spot/vol shock);
  Heston/Merton books would need a scenario model for their extra params.
- **Phase 3, step 3 — credit-risk / PD validation** — new `quantica/risk/credit/`
  subpackage, organised along the three regulatory validation dimensions.
  **Model-agnostic by design**: validators consume model outputs (y, PD scores),
  never a fitted model — package stays numpy/scipy-only; scikit-learn is a *dev*
  extra used only in scripts/tests (documented in pyproject). Modules:
  `discrimination.py` (AUC via Mann–Whitney rank identity with exact tie handling,
  Gini, KS, `roc_curve`, **stratified**-bootstrap CIs — stratification keeps
  low-default resamples non-degenerate); `calibration.py` (**the centerpiece**:
  exact `binomial_test` (one-sided prudential default), ECB `jeffreys_test`
  (Beta(d+½, n−d+½) posterior), `hosmer_lemeshow` with an exposed `dof` (G−2
  fitted-model convention vs χ²(G) for true/non-estimated PDs), `assign_grades` /
  `grade_calibration` per-grade table, `calibration_curve`); `stability.py` (PSI
  with expected-sample quantile bins + 0.10/0.25 convention bands, labelled as
  convention; `characteristic_stability` CSI); `data.py` (seeded synthetic
  portfolio with **known true PDs**, planted leverage×behavioural interaction +
  leverage² convexity so a linear logit is mis-specified, `leverage_shift` for
  drift). Validated (`tests/risk/credit/`, 40 tests): AUC **three independent
  ways** (rank ≡ trapezoid-ROC ≡ sklearn to machine precision, ties included) +
  binormal analytic anchors (AUC = Φ(δ/√2), KS = 2Φ(δ/2)−1); KS vs brute force +
  scipy; Jeffreys vs direct beta posterior; HL vs hand computation; PSI vs hand
  computation + known drifts; **the meta-challenge (size/power on known-truth
  grades)**: exact binomial **conservative** (size 3.7% at n=800, collapsing to
  1.7% at n=150 low-default) and paying in power (18% vs 35% at n=150) while
  **Jeffreys holds ~nominal size (5.6–5.8%) and ~doubles low-default power** —
  the measured reason ECB adopted it; HL has correct size with dof=G on true PDs
  (4.7%) while the G−2 convention over-rejects (11.5%) — dof is part of the
  validator. Champion/challenger (seed-robust): GBM out-discriminates logit by
  ~5 AUC pts (0.922 vs 0.869, ceiling 0.929) but **both flagged by calibration**
  (champion fails HL with χ²≈2307 — its *safest* grade defaults at ~30× the
  assigned PD via the planted convexity; challenger understates PDs in specific
  grades) → "promote only after recalibration", reported honestly. Report
  `scripts/pd_validation_report.py` (discrimination CIs, per-grade tables,
  PSI/CSI drift attribution, size/power table) → embedded in the README with
  ECB/Basel/SR 11-7 framing (factual).
- **Phase 3, step 4 — ML-model validation (SR 11-7)** — new
  `quantica/risk/ml_validation/` package (numpy/scipy-only; consumes SHAP
  matrices, PD scores, or a bare `predict` callable — never model internals;
  `shap` joins scikit-learn as a *dev* extra, verified working on Python 3.14).
  Modules: `explainability.py` (`check_local_accuracy` — SHAP's additivity axiom
  asserted; `global_importance` / `driver_recovery` vs a known DGP ranking;
  `rank_stability` (pairwise Spearman across replications);
  `attribution_direction`); `robustness.py` (`prediction_stability` — |ΔPD|
  under seeded feature-scaled noise; `performance_under_shift` — AUC + HL
  (dof=G, scores external to the eval samples) dev vs shifted);
  `fairness.py` (`disparate_impact` four-fifths (labelled EEOC convention),
  `group_calibration` two-sided Jeffreys within group; impossibility trade-off
  documented); `soundness.py` (`ConceptualSoundnessReview` — per-dimension
  verdicts + transparent aggregation rule → APPROVE / APPROVE_WITH_CONDITIONS /
  REJECT). **Data:** `CreditSample` gains a protected-`group` proxy +
  `group_effect` knob, drawn *after* all prior RNG consumption so every existing
  seeded result is bit-identical (verified). Validated
  (`tests/risk/ml_validation/`, 29 tests): **local accuracy 1e-14 on TreeSHAP /
  LinearSHAP and shown to FAIL (error ≈ 5.9) on the wrong output scale**
  (probability vs log-odds — the classic silent mistake); **SHAP recovers the
  planted driver order exactly** (both models) and the planted
  leverage×behavioural interaction as the top interaction pair (>3× margin);
  direction signs match the DGP with leverage attenuated by its U-shape (honest
  nuance, asserted); refit/subsample rank stability ≥ 0.9;
  **prediction-stability metric validated against the linear closed form**
  (E|Δf| = √(2/π)·σ_Δ); the honest robustness finding — **GBM tail |ΔPD| 0.28 vs
  champion 0.019 (15×) under 1% noise** (structural step-function jumps); the
  fairness impossibility on known truth — **even the TRUE PDs are calibrated
  within group yet fail four-fifths (ratio 0.76)** (base-rate fact, not model
  defect). Report `scripts/ml_validation_report.py`: full SR 11-7 review ending
  in **APPROVE WITH CONDITIONS** (calibration, robustness-tail, fairness-policy
  conditions; discrimination/explainability/drift PASS) → embedded in the README.
- **Phase 3, step 5 — FRTB P&L Attribution (PLA)** — `quantica/risk/frtb.py`, the
  IMA-eligibility test, ties the risk and derivatives pillars under a regulatory
  frame by *reusing* the derivatives-risk P&L machinery: **HPL** = the book's
  full-revaluation P&L (`OptionBook.full_revaluation_pnl`, the pricing path
  itself), **RTPL** = the risk model's sensitivities P&L (`delta_normal_pnl` /
  `delta_gamma_pnl`). PLA is literally the full-reval-vs-sensitivities comparison
  from step 2, elevated to a pass/fail capital test. Two Basel MAR33 metrics —
  `spearman_correlation` (rank; does the model *order* P&L right?) and
  `ks_distance` (two-sample KS; do the *distributions* agree?), both hand-rolled
  and anchored to `scipy.stats` — each mapped to green/amber/red at the published
  breakpoints (Spearman green ≥ 0.80 / red < 0.70; KS green ≤ 0.09 / red > 0.12;
  overall = worse of the two). `pla_test(rtpl, hpl)` and `book_pla_test(book,
  scenarios, rtpl_method=…)` → `PLAResult` (zones, `ima_eligible`,
  `capital_consequence`). Validated (`tests/risk/test_frtb.py`, 16 tests): Spearman
  & KS == scipy (ties included); the four published thresholds asserted verbatim;
  per-metric and boundary (≥/> conventions) zone logic; **known-truth books
  reusing the gamma divergence** — near-linear/deep-ITM delta-only → GREEN;
  short-gamma delta-**gamma** under large moves → GREEN (curvature spanned);
  short-gamma delta-**only** → RED on both metrics (IMA-ineligible → SA); the same
  desk at small moves → AMBER (Spearman green, KS amber); zone worsens
  monotonically with move size; no-drift (book HPL == full-reval path); constant
  RTPL → 0 correlation → red. Report `scripts/frtb_pla_report.py`: the three-desk
  green/green/red table → embedded in the README. Headline: **a short-gamma desk
  failing PLA is the delta-normal-vs-full-reval divergence, now with a capital
  consequence** — the regulator's eligibility test and the MV "when does the
  linear approximation break?" question are formally the same. **Scope (deliberate):
  FRTB is implemented as PLA only** — the rest of the framework (liquidity-horizon
  scaling, the ES capital charge and its regulatory aggregation, the standardised
  approach) is intentionally *out of scope*: it is regulatory-plumbing breadth that
  points away from the capital-markets direction the project is now taking. PLA was
  the high-signal slice because it *reuses* the derivatives-risk P&L and closes the
  pricing↔risk loop; going further into FRTB capital mechanics would add compliance
  surface without new modelling insight.
- **Capital-markets track, stage 1 — multi-factor risk model** — new **top-level**
  `quantica/factor/` package (placed at top level, NOT under `risk/`, because it is
  the shared foundation consumed by both market-risk decomposition and the future
  portfolio track). **Scope discipline held**: no hand-rolled estimators — loadings
  from `statsmodels` OLS (lazy import; gives t-stats/R²/residual variance),
  factor covariance from `numpy.cov`; the package's own code is the assembly +
  decomposition, with the OOS estimator-comparison layer deferred to stage 2 (the
  real deliverable). Modules: `exposures.py` (`estimate_exposures` — per-asset
  time-series OLS → `FactorExposures` with alpha/betas/t-stats/R²/specific var);
  `model.py` (`FactorRiskModel.fit` → B, alphas, F, D and per-asset exposures;
  `covariance()` = symmetrised B·F·Bᵀ + D; `systematic_covariance`;
  `variance_decomposition` per asset; `portfolio_variance` /
  `portfolio_risk_decomposition` / `portfolio_factor_exposure` = Bᵀw); `data.py`
  (`generate_factor_data` — synthetic panel with **planted** betas/alphas/specific
  var, seeded; the deterministic-test path so CI never needs a network fetch).
  Interface designed so a statistical-factor (PCA) variant can slot in later.
  Deps: `statsmodels>=0.14` made an explicit runtime dep (was already transitive
  via `arch`) + mypy override. Validated (`tests/factor/`, 16 tests): **betas ==
  independent `numpy.linalg.lstsq` to 1e-10** (anchors that we call statsmodels
  correctly) and == statsmodels directly; **single-factor reduces to the CAPM beta
  cov/var** to 1e-10; **known-truth recovery** — planted betas within 4 standard
  errors, specific variances within 10%; t-stats separate a real factor (|t|>20)
  from planted-zero factors (|t|<4); Σ symmetric + PD + equals its definition;
  variance/portfolio decompositions add up; seeded determinism; validation.
  Report `scripts/factor_model_report.py`: fetches FF–Carhart factors + 10 industry
  portfolios from Ken French (cached in OS temp, **never in CI**), fits the model —
  economically sane exposures (Utils market β 0.57, HiTec 1.16; Energy HML +1.16
  value vs HiTec −0.40 growth; R² 0.35–0.92); equal-weight portfolio 15.7% ann.
  vol, **93% systematic** → embedded in the README. **STOPPED for review after
  stage 1 (per the task); stage 2 = the OOS estimator comparison is next.**
- **Capital-markets track, stage 2 — OOS estimator comparison** (the factor step's
  headline). Two new modules in `quantica/factor/`. **Scope discipline held**: no
  estimators re-implemented — `estimators.py` wraps three behind one
  `CovarianceEstimator` protocol: `SampleCovariance` (`numpy.cov`),
  `LedoitWolfCovariance` (`sklearn.covariance.LedoitWolf`, lazy import),
  `FactorCovariance` (the stage-1 Σ=BFBᵀ+D); plus `condition_number` and
  `min_variance_weights` (GMV = Σ⁻¹1 renormalised). `evaluation.py` is the
  deliverable framework: `walk_forward_windows` (strictly non-overlapping,
  train_end==test_start — **no-lookahead is a tested property**), `compare_estimators`
  (per window: fit each estimator on train, score on the *next* test window via the
  **bias** = realized/forecast vol on shared random portfolios + each estimator's own
  **min-variance** portfolio), `BiasStats` (whole distribution, not just mean),
  `frobenius_error` + `min_variance_true_loss` (known-truth losses). Deps:
  `scikit-learn>=1.4` promoted from dev → runtime (lazy) + mypy `sklearn.*` override.
  Validated (`tests/factor/test_estimators.py` + `test_evaluation.py`, 20 tests):
  each estimator == its library directly; GMV weights == closed form (diagonal
  case) and are a true minimum; **no-lookahead**; **known-truth min-var ordering
  factor < LW < sample** on a factor DGP; **ill-conditioning** (sample cond ≫ LW,
  factor as n→T; factor best-conditioned); **the headline** — sample's min-var
  portfolio worst OOS with bias > 2, while random portfolios are indistinguishable
  across estimators; determinism. Report `scripts/covariance_comparison_report.py`
  (49-industry FF universe, 60-mo window, n/T≈0.8): sample min-var **23.0%** realized
  vol (forecast bias **6.0**) vs Ledoit–Wolf 11.8% vs factor 12.9%; condition number
  100→61,000 (sample) vs bounded (LW/factor) → embedded in README. **Honest finding:
  the factor model wins on the synthetic factor DGP (correctly specified) but
  Ledoit–Wolf wins on real industry data (4 factors don't fully span it); the
  universal result is only that sample covariance is worst under matrix inversion —
  "which estimator to trust *when*", not a single winner.** FF loader refactored to
  shared `scripts/_ff_data.py` (10- or 49-industry, missing-value handling).
- **Phase 2 — systematic portfolio management (the third pillar).** New
  `quantica/portfolio/` package (new runtime dep **`cvxpy>=1.5`**, lazy-imported, +
  mypy override; verified solving on Python 3.14 via CLARABEL/OSQP/SCS). Three layers,
  headline last, **scope discipline held** (no solvers or statistics re-implemented —
  cvxpy for the QP, numpy/scipy for the validity stats, factor-step estimators for Σ):
  - **Construction** (`construction.py`): `PortfolioConstraints` (long-only,
    per-name position limits, L1 **turnover budget** vs current holdings, full-investment)
    → linear cvxpy constraints. `minimum_variance_weights` (min wᵀΣw),
    `mean_variance_weights` (max μᵀw − ½γ wᵀΣw), `risk_parity_weights` (Spinu's convex
    log-barrier ERC, long-only by construction). All use `cp.psd_wrap` for solver
    robustness. **Validated against algebra** (`test_construction.py`, 12 tests):
    budget-only GMV == closed-form Σ⁻¹𝟙 to **7e-17** (the anchor); MV == its
    budget-constrained closed form; risk-parity == inverse-vol weights for diagonal Σ
    and equalises risk contributions; long-only/position-cap/turnover budgets each
    asserted respected; MV→GMV as γ→∞; MV tilts toward higher α.
  - **Backtest engine** (`backtest.py`): `walk_forward_backtest` reuses the tested
    `factor.evaluation.walk_forward_windows` (no-lookahead), holds target weights over
    each non-overlapping window while tracking **exact weight drift**, measures one-way
    turnover vs the drifted book, and charges `ProportionalCosts` as a first-period
    return drag. `Strategy` protocol + `strategy.py` bundlers
    (`MinimumVarianceStrategy` / `RiskParityStrategy` / `MeanVarianceStrategy`, the last
    taking a pluggable alpha `Signal`) bind a `CovarianceEstimator` to a constructor —
    the stage-2 comparison plugs straight in. `BacktestResult` exposes gross/net series,
    turnover, costs, weights, Sharpe, cumulative return. **Validated for exactness**
    (`test_backtest.py` 12, `test_strategy.py` 4): zero-cost net == gross to the last
    bit; gross − net reconciles to total cost to 1e-15; opening turnover from cash == 1;
    weight drift == analytic renormalisation; **no-lookahead proven** (a data-dependent
    strategy's past weights are bit-identical after corrupting the future); min-variance
    realises lower OOS vol than equal-weight; costs strictly reduce net return.
  - **Backtest-validity layer (THE deliverable)** — `overfitting.py`:
    `probabilistic_sharpe_ratio` (PSR, Bailey–LdP 2012), `expected_maximum_sharpe`
    (multiple-testing benchmark, Euler–Mascheroni closed form), `deflated_sharpe_ratio`
    (+ `_from_trials` picking the best column), `minimum_track_record_length`,
    `probability_of_backtest_overfitting` (PBO via CSCV: partition rows into S blocks,
    over all C(S,S/2) balanced splits record the logit of the IS-best's OOS rank; PBO =
    P(logit ≤ 0)). `cv.py`: `purged_kfold_indices` (two-sided purge of `label_horizon` +
    trailing `embargo`). `data.py`: `generate_trial_returns` (planted-signal known
    truth). **Validated** (`test_overfitting.py` 20, `test_cv.py` 8): PSR == Φ of the
    independent z-score, == 0.5 at benchmark, penalises skew/kurtosis; expected-max ==
    formula, grows with trials; DSR == PSR at the emax benchmark; **MinTRL exact
    round-trip through PSR**; PBO == 0.5 for noise (averaged), < 0.05 for a dominant
    signal; **the headline known-truth — a 100-noise overfit search is flagged spurious
    (DSR never significant, PBO ≈ 0.5) while a planted signal survives both (DSR ≈ 1,
    PBO = 0)**; **purging removes overlapping-label leakage** (nearest-neighbour "skill"
    0.77 unpurged → ≈ 0 purged). Added `IntArray` to `core/types.py`.
  - Report `scripts/portfolio_backtest_report.py` (synthetic known-truth table, no
    network; + real 49-industry FF backtest net of 10 bps costs). **Honest real-data
    finding — the DSR/PBO split verdict**: the best net strategy (`minvar/sample`, net
    Sharpe 0.65) is DSR-significant (0.998) yet high-PBO (0.85) — the six near-identical
    long-only configs make the *premium* real (low cross-trial variance → low emax
    benchmark → high DSR) but the *ranking* non-repeatable (high PBO). "Trust the
    premium, not the ranking." MinTRL for that Sharpe is 79 months. → embedded in README.
  - **Gate green**: 849 tests (55 new), ruff + mypy clean.
- **README as the cold-reviewer artifact + the Jagannathan–Ma cross-pillar investigation.**
  No new features; the README was restructured to lead with the **thesis** (validation-first:
  the deliverable is the evidence each model is correct, not the model), a **three-pillar**
  table with one-line signatures, a **Headline results** section (one artifact per pillar,
  chosen for what it *proves* — the four-way convergence table; the short-gamma book that
  breaks delta-normal VaR by −41% and fails FRTB PLA for the same reason; the sample
  covariance error-maximiser at 23.0% vol / 6× optimistic; DSR/PBO overfit detection), and a
  new **"What quantica ships that other tools don't"** section promoting the gaps log with
  its measured numbers. The old chronological "Status" blockquote is gone; the detail
  sections were regrouped under three `## Pillar` dividers (H2 section heads demoted to H3)
  so the accretion reads as structure. Every README number is script/test-reproducible
  (anchors verified). **Investigation (resolved):** the apparent tension — sample covariance
  *worst* under unconstrained inversion (factor stage 2) yet `minvar/sample` the *best*
  backtest config — is Jagannathan & Ma (2003): a no-short-sale constraint is **exactly**
  equivalent to solving the unconstrained problem with a shrunk covariance
  Σ̃ = Σ − (μ𝟙ᵀ + 𝟙μᵀ), μ the KKT multiplier on w≥0, and the shrunk assets are precisely the
  high-covariance ones the unconstrained GMV would short. Evidence: `tests/portfolio/
  test_jagannathan_ma.py` (4 tests — the GMV(Σ̃)==long-only-GMV equivalence to **1.6e-14**
  from the primal alone, KKT non-negativity + complementary slackness, shrunk-set ==
  bound-hitting-set, and a synthetic OOS-outcome test) + `scripts/shortsale_shrinkage_report.py`
  (49-industry FF: long-only collapses the sample GMV from **23.0% → 11.6%**, level with
  Ledoit–Wolf's 11.5%; condition number 61,123 → 36,082). Documented as a cross-pillar
  insight in **both** the factor and portfolio README sections. **Gate green**: 853 tests
  (4 new), ruff + mypy clean.
- **Step 8 — the thin apps (Streamlit + Plotly over all three pillars).** New `apps/`
  package + `app` optional extra (`streamlit>=1.30`, `plotly>=5.18`), kept out of the
  runtime *and* dev sets so the library/tests/CI never depend on a UI. **Architecture
  rule held (CLAUDE.md §2 — zero quant logic in `apps/`)**: every number is computed by
  `quantica`; the apps only orchestrate calls, cache, and draw. Structure enforces it —
  Streamlit-free **compute** modules (`_derivatives.py`, `_risk.py`, `_capital.py`, plus
  `_data.py` loading a committed 39 KB FF sample `apps/data/ff_sample.npz`, never fetched
  at runtime) hold all the orchestration; `quantica_app.py` is presentation only
  (widgets, `st.cache_data`, Plotly). One app, sidebar pillar selector (lazy per-pillar
  render, not `st.tabs`, so an interaction recomputes only one pillar). **Derivatives**:
  live price+Greeks, Greek profiles, the four-way convergence table, a rotatable Heston
  IV surface, Heston-vs-BS + Merton-jump smiles. **Risk**: the delta-normal/delta-gamma/
  full-reval VaR divergence + scenario-P&L histogram, the live FRTB PLA verdict (delta-
  only → RED/IMA-ineligible vs delta-gamma → GREEN), the four VaR/ES engines rolled OOS
  on the FF portfolio. **Capital markets**: the OOS covariance comparison (sample GMV
  23.0% / bias 6.0), the Jagannathan–Ma 2×2 + exact equivalence (1.6e-14), the DSR/PBO
  overfit detector with a planted-signal slider. Validated: `tests/apps/test_apps_smoke.py`
  (11 tests, Streamlit-free — imports + sane shapes/directions for every compute fn, runs
  in CI under `dev`); the whole app additionally verified error-free across all three
  pillars via Streamlit's `AppTest` harness locally. Added pytest `pythonpath=["."]` (so
  the uninstalled `apps` package imports in tests) and an `apps/**` RUF001/2/3 ignore
  (Greek / typographic symbols in UI labels). **Gate green**: 864 tests (11 new), ruff +
  mypy clean. **Delivered on branch `feat/apps` as [PR #1](https://github.com/dominikhanke98/quantica/pull/1)
  (first non-trunk change; opened via the REST API since `gh` is not installed here) —
  CI green on the branch, then MERGED to `main` via a merge commit (curated multi-commit
  history preserved), and `feat/apps` deleted.**
- **Step 8b — deploy to Streamlit Community Cloud.** The app is **live at
  https://quantica.streamlit.app/** (linked from the README top matter, above the
  three-pillar table). Getting there took a dependency-packaging fix: **streamlit +
  plotly were moved from an `[app]` optional extra into the main runtime `dependencies`,
  and the (ignored) `requirements.txt` deleted** — see the deploy gotcha under "Gaps".
  The `quantica` package code still never imports the UI stack (only `apps/` does), so
  the library stays usable without a UI (CLAUDE.md §1); the cost is a heavier
  `pip install -e .` footprint (§3), accepted since the repo isn't on PyPI. Verified in a
  clean venv that a main-group-only install (`pip install .`, no extras) makes
  `import plotly.graph_objects` and the full app import chain succeed.
- **Step 9 — auto-generated API reference manual (CRAN-style).** A complete, browsable
  reference generated **from the source docstrings** so it can never drift. (1) **Docstring
  audit**: filled every public-API gap (concrete `CovarianceEstimator` / `VaREngine` /
  `Strategy` / `TransactionCostModel` protocol implementations, the `exercise` properties,
  `BiasStats`/`PBOResult`/`FamaFrenchSample` properties, engine `npv`/`greeks` Parameters,
  the apps compute layer) — **100% public docstring coverage** across `quantica` + `apps`.
  (2) **Generator = pdoc** (chosen over Sphinx: renders the existing NumPy docstrings to
  clean cross-linked HTML with zero config and no LaTeX toolchain, so it regenerates
  trivially in CI; reasoning noted in `scripts/build_docs.py`). New `docs` optional extra
  (`pdoc`, `interrogate`). (3) **One-command build** `scripts/build_docs.py` → `docs/api/`
  (50 pages, organized by module = the three pillars; `--no-search` keeps it ~3 MB;
  committed so it's browsable via raw.githack). (4) **Anti-drift gate**: `interrogate`
  (`[tool.interrogate]`, `fail-under = 100`) in a new CI `docs` job that also rebuilds the
  reference (a docstring that fails to render breaks the build) — you cannot merge an
  undocumented public function. (5) **Standing procedure** added to CLAUDE.md §6
  (docstring = single source of truth; update it in the same commit; regenerate, never
  hand-edit `docs/api/`). README top matter links the reference; Development section
  documents the workflow. Gate green: 864 tests, ruff + mypy + interrogate(100%) clean.
- **Step 10 — PDE Greeks + Rannacher start-up (closes the finitediff.py L-stability loop).**
  `FiniteDifferenceEngine` now satisfies the `GreeksEngine` protocol. **Delta/gamma** come
  off the solved value surface almost for free — central differences of adjacent nodes,
  mapped from the log-grid by the chain rule (`Δ=V_x/S`, `Γ=(V_xx−V_x)/S²`); **theta** is a
  central difference in the time direction (one extra CN step past today); **vega/rho** are
  bump-and-reval re-solves reusing `process.with_vol`/`with_rate`. The engine was refactored
  to one shared `_solve` path (grid + theta-scheme step machinery). **Rannacher start-up**:
  a new `rannacher_steps` param (default **2**, `0`=pure CN) replaces the first CN steps
  nearest expiry with backward-Euler half-step pairs (L-stable) to damp the payoff-kink
  oscillation, then resumes CN. **Validated** (`test_finitediff.py`, +~12 tests): all five
  Greeks agree with the analytic BS Greeks to O(h²) (rel 1e-3 at 400×400) and converge at
  **second order** (log-log slope −2.00 each), calls & puts; American Greeks obey bounds
  (put delta∈[−1,0], gamma>0); **the headline — Rannacher damps the gamma oscillation:
  total variation of the gamma error 1.7e-2 (pure CN) → 1.9e-4 (Rannacher) = 89× smoother,
  max error 143× lower** on a coarse-in-time (200×25) grid. QuantLib FD benchmark
  (`test_benchmark_quantlib.py`, +2): our FD delta/gamma/theta match QuantLib's
  `FdBlackScholesVanillaEngine` to an O(h²) inter-library difference. Demo
  `scripts/rannacher_gamma_demo.py` (deterministic) prints the before/after metrics + the
  gamma-vs-spot profile (plot data) → embedded in the README PDE-Greeks section. Default
  Rannacher slightly shifted the FD prices, so the README convergence table's PDE rows were
  regenerated. **The L-stability caveat the engine carried as a comment is now implemented
  and demonstrated.** Gate green: 878 tests (14 new), ruff + mypy + interrogate(100%) clean.
  Delivered on branch `feat/pde-greeks` → PR #2, merged to `main` (merge commit `bcaf59d`).
- **Step 11 — HRP + Black–Litterman (the two most-asked-about systematic-PM methods).**
  Two new construction modules in `quantica/portfolio/`, both consuming the existing
  `CovarianceEstimator` and pluggable into the walk-forward backtest + validity layer, with
  `HRPStrategy` / `BlackLittermanStrategy` added to `strategy.py`. **Scope discipline held**:
  the construction *algorithms* are hand-implemented (the demonstrable skill); only the
  clustering plumbing is leaned on (`scipy.cluster.hierarchy`). **HRP** (`hrp.py`,
  `hrp_weights` + `quasi_diagonal_order`): the three López de Prado stages — correlation-
  distance tree linkage (scipy), quasi-diagonalisation by leaf order, recursive bisection
  splitting risk inversely to cluster variance — **never inverts the covariance**.
  Validated (`test_hrp.py`, 6): weights sum to 1 / long-only; two-asset case == inverse-
  variance closed form; equal-variance == equal weight; planted cluster blocks recovered;
  **the headline OOS-robustness tie-back — on an n/T≈0.94 universe HRP realises <0.5× the
  min-variance OOS vol with ~20× less leverage** (never inverts vs the error maximiser).
  **Black–Litterman** (`black_litterman.py`, `implied_equilibrium_returns` + `black_litterman`
  → `BlackLittermanResult`): reverse-optimise benchmark weights to equilibrium π=δΣw_mkt,
  He–Litterman default Ω=diag(PτΣPᵀ), the master formula → posterior (μ_BL, Σ_BL) → MV.
  Validated (`test_black_litterman.py`, 5): **reverse-opt round-trips to 1.8e-15**; **no
  views → posterior == equilibrium** (clean known-truth); a confident view moves the
  posterior toward it; **the headline stability contrast — a 1% return-estimate shock swings
  naive unconstrained MV ~7× more than BL** (BL shrinks toward equilibrium). Reports
  `scripts/hrp_robustness_report.py` (FF 49-industry: sample min-var 28.4% vs HRP 12.7% vs
  LW 11.1% OOS, no inversion; + net-of-cost backtest) and `scripts/black_litterman_report.py`
  (7× stability contrast; + net-of-cost backtest) → embedded in README. **Honest findings:**
  (i) on the constrained long-only backtest HRP/BL land in a similar Sharpe band to the
  others — the industry premium dominates and the long-only cap already regularises (the
  Jagannathan–Ma effect), so BL's advantage is specifically the *unconstrained* case; (ii)
  reported net-of-cost. Gate green: 889 tests (11 new), ruff + mypy + interrogate(100%) clean.
  Delivered on branch `feat/hrp-blacklitterman` (PR, per the apps/PDE-Greeks workflow).
- **Step 12 — Autocallable structured note (composing path / barrier / stochastic-vol
  machinery into a traded product).** New `AutocallableNote` instrument (`instruments.py`,
  a frozen dataclass — a note, not a `VanillaOption`: observation schedule, autocall
  barrier, accrued coupon, downside barrier, all as fractions of the initial fixing) +
  `AutocallableMonteCarloEngine` (`engines/autocallable.py`) pricing it by Monte Carlo
  under **Black–Scholes, Heston and Merton** (`estimate` → `AutocallableResult`: price, SE,
  per-date first-autocall probabilities, survival & loss probabilities). **Scope discipline
  held — the note payoff is the only new logic.** Since the Heston/Merton engines are
  transform-based (FFT/closed-form) and cannot price path-dependent products, two path
  simulators were added to `engines/_paths.py` alongside the reused `GBMPathSimulator`:
  `MertonPathSimulator` (GBM diffusion + exact compound-Poisson jump per step, compensated
  drift) and `HestonPathSimulator` (full-truncation Euler on the CIR variance, `n_substeps`
  sub-stepping). **Both anchored to the existing validated pricers** — the composition claim
  made rigorous. Discrete monitoring is *genuinely* discrete (contractual dates), so — unlike
  the continuous barrier — there is no continuous-limit bias to correct (exact BS/Merton
  marginals; only Euler bias under Heston, set by `heston_substeps`). Validated
  (`tests/pricing/engines/test_autocallable.py`, 10 tests): **Merton MC == closed form** and
  **Heston MC == FFT** for a European call (within ~4 SE, the simulator anchors); **payoff
  wiring pinned exactly** — a single-observation note == cash-or-nothing digitals +
  asset-or-nothing put in closed form under BS; structural limits (autocall→0 collapses to a
  one-period coupon-bond cashflow to machine precision; full protection → zero loss, price ≥
  principal PV; autocall+survival probabilities partition to 1 within 1e-12); **the headline —
  flat vol matched to Heston's ATM vol overprices the note by ~0.85% of notional and
  understates P(loss) 3.8%→4.9%** (the note is short the down-and-in put; steep negative skew
  makes it dearer), asserted at >8 combined SE; reproducibility (same seed bit-identical,
  seeds agree within SE). Report `scripts/autocallable_smile_report.py` (BS-vs-Heston/Merton
  mispricing table + Heston autocall-probability-by-date breakdown) → embedded in README.
  **Honest finding:** the Merton (jump) smile is roughly symmetric in the wings, so its net
  mispricing is small and opposite-signed — *skew*, not tail-fatness, is what an autocallable
  is structurally short, which is why the diffusive-skew (Heston) result is the clean, material
  one. Gate green: 899 tests (10 new), ruff + mypy + interrogate(100%) clean. Delivered on
  branch `feat/autocallable` (PR #4, merged to `main` via merge commit `dc6a574`).
- **Step 13 — Statistical (PCA / RMT) factor model — completes the observable-vs-statistical
  pair.** New `quantica/factor/statistical.py`. **Scope discipline held**: the
  eigendecomposition/SVD leans on `numpy.linalg`/`scipy.linalg`; the demonstrable skill is
  the factor-model construction, the component-selection reasoning, and the validation.
  **Shared-base refactor (CLAUDE.md §2, second consumer):** extracted `LinearFactorModel`
  (the Σ=BFBᵀ+D assembly + risk decomposition) from `FactorRiskModel`; both the observable
  model and the new `StatisticalFactorModel` inherit it (observable model behaviour
  unchanged — 36 factor tests still green). **`StatisticalFactorModel.fit`** does
  correlation-PCA: standardise → eigendecompose the correlation matrix (≡ SVD of the
  standardised returns) → keep `k` PCs → `B = diag(s)·V_k·√Λ_k`, `F = I_k`,
  `D = diag(s²·(1−communality))`, so `Σ = BBᵀ + D` **preserves the sample variances exactly**
  on the diagonal and only approximates the off-diagonal. **Component selection as a real
  modelling decision** — three rules: `variance_explained_rank`, `scree_elbow_rank` (geometric
  knee), and the headline **`marchenko_pastur_rank`** (RMT: count eigenvalues above
  `λ₊ = σ²(1+√(n/T))²`; optional single-pass bulk-variance refit for a dominant market mode,
  default off — the plain σ²=1 cutoff is stable/textbook). `subspace_similarity` (cos of the
  largest principal angle, via `scipy.linalg.subspace_angles`) checks span recovery.
  **`StatisticalFactorCovariance`** is a `CovarianceEstimator` (needs *no* observable factors)
  so it plugs into the stage-2 OOS comparison. Validated (`tests/factor/test_statistical.py`,
  22 tests): **known-truth — MP recovers the exact planted factor count (3/3 across seeds) and
  the loadings span the true subspace (similarity > 0.99); pure noise → 0 factors** (headline);
  anchors (Σ symmetric PSD; EVR sums to 1; diag == sample variances exactly; k=n reproduces the
  sample covariance; **eigh == independent SVD**); selection-rule behaviour (variance monotone,
  scree elbow on a two-block spectrum, MP edge formula; the bulk refit recovers weak factors
  σ²=1 misses under a dominant market); **tie-back — the PCA covariance beats sample OOS and on
  known-truth min-var loss, competing with the observable factor model.** Report
  `scripts/statistical_factor_report.py`: synthetic recovery (no network) + real 49-industry
  scree/RMT cutoff + statistical-vs-observable comparison. **Honest findings:** (i) on real
  data PC1 carries ~57% of variance (the market) and correlates **0.96** with observable
  Mkt-RF — PCA rediscovers the market; (ii) the *statistical* 4-factor model (11.3% OOS vol,
  = Ledoit–Wolf) **edges the *observable* 4-factor model** (12.3%) — the 4 FF factors don't
  fully span industry risk, while PCA targets the covariance directly; the trade-off is
  interpretability. Gate green: 921 tests (22 new), ruff + mypy + interrogate(100%) clean.
  Delivered on branch `feat/statistical-factors` as **PR #5 — open, CI-green (py3.11,
  py3.12, benchmark, docs all pass), awaiting review** (not yet merged; the merge is left to
  the author, per the established workflow).

## Next — optional depth only (planned scope is done)

**All three pillars are complete, merged to `main`, and the app is live at
https://quantica.streamlit.app/.** The originally-planned scope of `quantica` (CLAUDE.md
§8–9 + the deferred apps) is fully delivered. Nothing remains on the critical path.

**Depth-encore build queue — essentially complete:**
- **(C1) PDE Greeks + Rannacher start-up** — **✓ (step 10, PR #2 merged).**
- **(A) HRP + Black–Litterman construction** — **✓ (step 11, PR #3 merged).**
- **(C2) Autocallable on the path machinery** — **✓ (step 12, PR #4 merged `dc6a574`):**
  composes the path/barrier/stochastic-vol machinery into a real traded product; headline
  BS-vs-Heston smile mispricing ~**0.85% of notional** (flat vol overprices — the note is
  short *skew*, not tail-fatness, hence Merton jumps net near-zero).
- **(D) Statistical PCA/RMT factor model** — **✓ (step 13, PR #5 open, CI-green, awaiting
  review):** correlation-PCA into Σ=BBᵀ+D, Marchenko–Pastur component selection (recovers
  the planted count 3/3, subspace >0.99), shared `factor/model.py` `LinearFactorModel` base
  extracted (§2 second consumer); tie-back — PC1↔market 0.96, statistical 4-factor 11.3% OOS
  vs observable 12.3% (observable wins on interpretability, not accuracy).

Remaining optional build items (none started, none blocking): swap the American PSOR for
Brennan–Schwartz (direct tridiagonal LCP solve); the FRTB expected-shortfall capital charge
at 97.5% (liquidity-horizon scaling, regulatory ES aggregation); surfacing HRP/BL/PCA in the
apps' capital-markets tab.

## Presentation backlog (pending — the encore's write-up half)

The *building* is essentially done; the **presentation** half is the remaining work.
**Seven blog posts are drafted, awaiting number-verification (re-run each source script and
check every figure against the current code) and publishing:**
1. Hosmer–Lemeshow degrees of freedom — why `dof = G−2` over-rejects on externally-supplied
   PDs (validate-the-validator size study).
2. SHAP output-scale trap — log-odds vs probability, and how `check_local_accuracy` catches
   the silent failure.
3. Gamma / Rannacher start-up — damping the Crank–Nicolson gamma oscillation at the payoff
   kink (89× smoother).
4. HRP without inversion — robustness exactly where the inverting min-variance portfolio
   blows up.
5. Autocallable skew — flat vol misprices a short-skew structured product (~0.85% of
   notional; skew, not tail-fatness).
6. *(drafted, topic TBD-in-notes)* — plus the flagship narrative post tying the
   validation-first thesis across all three pillars.

**Before publishing any figure, re-run its script and reconcile against the current code** —
the code has moved since some drafts (e.g. the FF-data reports, the Rannacher table, the
autocallable numbers) and the README embeds captured runs that are the source of truth.

## Gaps in existing tools (accumulating — portfolio-narrative material)

Findings where standard libraries are silently wrong, missing, or opaque — and
this repo's independent implementation surfaced it. Add to this list as they occur.

- **The scientific stack ships PCA, not a PCA *risk model* with a principled factor count
  (step 13).** `sklearn.decomposition.PCA` and `numpy.linalg.eigh`/`svd` give the
  eigendecomposition, but neither ships (i) the reconstruction into a well-conditioned
  Σ = BBᵀ + D risk model consistent with an observable-factor interface, nor (ii) a
  *principled* component-selection rule — `PCA(n_components=…)` wants the count handed to it,
  and `n_components=0.9` (variance-explained) demonstrably over-keeps (22 of 49 industry PCs,
  mostly noise). The Marchenko–Pastur / random-matrix cutoff that separates signal
  eigenvalues from the noise bulk is the actual modelling decision, and it lives only in
  specialist RMT code — the same "the primitives exist, the method doesn't" category as the
  HRP/BL and OOS-covariance gaps. (Also a validation-flavoured finding: the naive
  bulk-variance fixed point *spirals* on finite-size stragglers near the edge, so the stable
  choice is the plain σ²=1 cutoff with a single optional refit — documented in the code.)
- **No autocallable — and no path generator to price one — in QuantLib's Python wrapper
  (step 12).** QuantLib ships vanilla/American/Asian/barrier instruments and both a Heston
  and a Merton *process*, but (i) there is no autocallable instrument, and (ii) the generic
  Monte Carlo path-generator machinery that would let you price a bespoke path-dependent
  payoff under those processes is not exposed in the Python wrapper (it lives in C++
  `MakeMC*` engine factories bound only for specific instruments). So there is no reference
  to benchmark an autocallable against, *and* the Heston/Merton MC paths had to be
  hand-written — which is why the two new path simulators are anchored to QuantLib-validated
  transform prices (`HestonFFTEngine`, `MertonClosedFormEngine`) instead. Same "the
  primitives exist, the product doesn't" category as the HRP/BL and OOS-covariance gaps.
- **The scientific stack ships the plumbing, not the PM construction methods (step 11).**
  `scipy.cluster.hierarchy` gives the linkage/leaf-order clustering and numpy/scipy the
  linear algebra, but neither ships **HRP** or **Black–Litterman** as a construction rule —
  those live only in dedicated portfolio libraries (PyPortfolioOpt, riskfolio-lib). Same
  "the primitives exist, the method doesn't" category as the OOS covariance-estimator gap;
  implementing the three HRP stages and the BL master formula on top of the scipy plumbing
  is exactly the demonstrable skill (and lets both plug into the repo's own backtest +
  validity layer, which those libraries do not ship).
- **QuantLib's FD engine exposes no vega/rho (step 10).** `FdBlackScholesVanillaEngine`
  provides `delta()`, `gamma()` and `theta()` off its grid but raises "vega not provided" /
  "rho not provided" — the volatility- and rate-sensitivities are simply not computed by
  the FD engine (they need re-solves, which QuantLib leaves to the caller). So the PDE
  vega/rho benchmark had to fall back to the analytic Greeks as the anchor (the delta/
  gamma/theta QuantLib cross-check still stands). A concrete "the reference is missing the
  quantity, not wrong" finding — the same category as the ES-backtest / FRTB-PLA gaps.
- **Streamlit Community Cloud dependency handling (step 8b, deploy).** Cloud installed
  from `pyproject.toml` **via Poetry's main dependency group only** — it **ignored
  `requirements.txt` entirely and did not install optional extras**, so the `[app]`
  extra's `plotly` was never present and the app crashed on `import plotly.graph_objects`.
  Fix was to declare the UI deps in the main `dependencies` (not an extra) and delete the
  dead `requirements.txt`. Second gotcha: after pushing the fix, the deploy **still failed
  from a stale build container caching the old dependency set** — a **manual reboot of the
  app from the Streamlit Cloud dashboard** was required to force a clean reinstall. Lesson
  for any future redeploy: declare deploy-required deps in the pyproject main group, and
  if a dep change doesn't take, reboot the app (not just re-push) to bust the cache.
- **Backtest-validity layer absent from mainstream backtesters (Phase 2).** The
  popular open-source backtesters (`backtrader`, `vectorbt`, `zipline`, `bt`) ship the
  *engine* — signal → weights → P&L curve — but not the layer that answers *is this
  curve real?*. The Deflated Sharpe Ratio, PBO/CSCV, purged-embargoed CV and MinTRL are
  López de Prado's, and the one library that bundled them (`mlfinlab`) went
  closed-source/commercial. So the exact deliverable a model-validation reviewer cares
  about — deflating a backtest for the number of trials and testing selection
  robustness — is the "missing, not wrong" gap this pillar fills, and its own
  correctness is proved on planted known-truth (noise flagged, signal survives).
- **DSR and PBO answer different questions, and can (correctly) disagree (Phase 2).** On
  the real FF grid, the best net strategy was DSR-significant (0.998) *and* high-PBO
  (0.85). Not a bug: DSR asks whether the Sharpe *magnitude* is real after multiple
  testing (here yes — an industry premium), PBO whether the *selection* among
  near-identical configs is repeatable (here no). A tool that reports only one metric
  hides half the verdict; reporting both is the point.
- **Hosmer–Lemeshow degrees of freedom (step 3).** Many implementations hardcode
  `dof = G − 2`. That convention is derived for a model *fitted on the same
  sample*; when validating externally-supplied PDs (true/regulatory/vendor PDs —
  the standard model-validation situation) the null is χ²(G), and the G−2
  convention **over-rejects** (measured 11.5% at nominal 5%). Our
  `hosmer_lemeshow` exposes `dof` and documents both nulls; the size study proves
  the difference.
- **ES backtesting (step 1).** ES is not elicitable (Gneiting 2011), so the naive
  count-based backtest used for VaR does not transfer; most risk libraries simply
  omit ES backtests. Acerbi–Székely with a Monte-Carlo null fills the gap, with
  its own size/power measured.
- **No QuantLib Merton engine in the Python wrapper (Phase 4, step 5).**
  `Merton76Process` exists but no jump-diffusion pricing engine is exposed, so
  there is no reference to benchmark against — the closed-form-vs-FFT
  self-validation (~2e-7) had to carry the effective challenge instead.
- **SHAP output-scale ambiguity (step 4).** `shap.TreeExplainer` explains the
  **log-odds margin** for `HistGradientBoostingClassifier` while users naturally
  compare against `predict_proba` — the additivity identity then fails silently
  (nothing errors; the attributions just don't sum to anything meaningful,
  max error ≈ 5.9 on our book). `check_local_accuracy` exists precisely to make
  this loud; the report demonstrates the failure mode explicitly.
- **QuantLib's default `AnalyticHestonEngine` at short expiry (Phase 4, step 4b).**
  An apparent ~1.9e-2 benchmark "error" at T=0.1 decomposed into (i) a day-count
  artifact (`round(365·0.1)` = 36 days ≈ 0.0986yr, not 0.1) and (ii) QuantLib's
  *default* integration being the less accurate side at short maturity: an
  independent scipy-quadrature truth confirmed our Carr–Madan FFT exact to ~1e-13
  there. Benchmarks therefore use integer-day maturities — and "the reference
  disagreed because the reference was coarser" is itself a validation finding.
- **FRTB PLA absent from open-source risk tooling (step 5).** The P&L-attribution
  test is a regulatory eligibility gate banks implement in-house; no mainstream
  open-source risk library ships it (same "missing, not wrong" category as the
  ES-backtest gap). Implementing it required only reusing the derivatives-risk
  full-reval-vs-sensitivities P&L already built — the plumbing was there; what was
  missing was the regulatory framing (metrics, published thresholds, zone
  aggregation), which is exactly the demonstrable skill.
- **No OOS covariance-estimator validation in mainstream libraries (factor
  stage 1/2).** `sklearn.covariance` ships the *estimators* (LedoitWolf, OAS) and
  `statsmodels` the regressions, but neither ships the layer that answers *which
  estimator forecasts realized risk better out of sample* — the actual
  model-validation question. That layer (stage 2) is the factor package's reason to
  exist; same "missing, not wrong" category.
- **Ken French CSV parsing quirks (factor stage 1, data plumbing).** The library's
  files are deceptively hostile to naive parsing: (i) multi-line prose preambles
  that *contain commas*, so "first comma line = header" grabs a sentence; the
  header is the last comma line *before* the first data row. (ii) A single file
  concatenates several monthly blocks with the *same* `YYYYMM` date format
  (value-weighted returns, then equal-weighted, then firm counts, then average
  dollar sizes) — a YYYYMM filter alone silently mixes returns with firm-size
  dollars and produces betas of ~100 and 36000% specific vols. Fix: take only the
  first contiguous monthly block. A concrete "real data is messy; validate the
  ingest by economic smell test" finding (the fixed betas are economically sane).

## Open design notes

- **No-short-sale is covariance shrinkage (cross-pillar, Jagannathan–Ma 2003).** The
  long-only minimum-variance portfolio on any Σ equals the *unconstrained* GMV of
  Σ̃ = Σ − (μ𝟙ᵀ + 𝟙μᵀ) (μ = the KKT multiplier on w≥0, recoverable from the primal via
  λ = wᵀΣw, μ = Σw − λ𝟙). Consequence: the sample covariance's error-maximiser failure is
  an *unconstrained* phenomenon; long-only regularises it (23.0% → 11.6% OOS vol on FF),
  so `minvar/sample` winning the long-only backtest does not contradict factor stage 2.
  Pinned by `tests/portfolio/test_jagannathan_ma.py`.
- **`LinearFactorModel` base extracted on the second consumer (step 13).** The Σ=BFBᵀ+D
  assembly and risk decomposition now live in a shared `LinearFactorModel` base;
  `FactorRiskModel` (observable, regression loadings) and `StatisticalFactorModel` (PCA
  loadings) differ only in their `fit` and their extra fields. Done per CLAUDE.md §2 (extract
  on the *second* use, not up front) — the statistical model was that second consumer. All
  construction is via `.fit()` with kwargs, so the dataclass-inheritance field reordering is
  invisible to callers; the 36 pre-existing factor tests pin the observable model unchanged.
- **Correlation-PCA (not covariance-PCA) is the deliberate choice (step 13).** Standardising
  to unit variance before the eigendecomposition (i) makes the Marchenko–Pastur bulk edges
  clean (σ²=1 for i.i.d. noise), the whole point of the RMT cutoff, and (ii) makes the
  reconstruction preserve the sample variances exactly on the diagonal (kept + dropped
  communality = 1), so only the off-diagonal cross-correlations are approximated. `F = I_k`
  because the PCs are orthonormal with their variance folded into the loadings.
- **Portfolio validity layer operates on return *matrices*, decoupled from the
  backtester (Phase 2).** `overfitting.py` (DSR/PBO) takes a `(T, N)` trial-return
  matrix, not a backtest object — so it is testable on synthetic known-truth without
  running any construction, and the report can assemble the six strategy net-return
  series into the matrix and interrogate the *selection*. The return series is the seam,
  same posture as the risk pillar's P&L series.
- **Risk parity takes no general `PortfolioConstraints` (Phase 2).** Spinu's convex
  log-barrier ERC is solved unconstrained then renormalised (`w = y / 𝟙ᵀy`), which is
  intrinsic to the formulation — so position/turnover constraints can't be layered on
  without breaking the equal-risk property. Documented in the docstring; min-variance
  and mean-variance carry the full constraint set. Revisit only if a constrained-ERC
  variant is actually needed.
- **cvxpy typed as `Any` at the seam (Phase 2).** cvxpy ships no stubs (mypy override,
  ignore_missing_imports); the three private helpers that touch `cvxpy` objects
  (`_cvxpy`, `_linear_constraints`, `_solve`) annotate those params `Any` rather than
  scattering `# type: ignore`, keeping the public constructors fully typed.
- **Ignored-vol market carrier — RESOLVED (step 4a).** `implied_volatility` now
  takes a `Market` (spot, rate, div); there is no placeholder vol to ignore. The
  `Market` carrier is shared by `BlackScholesProcess` and `HestonProcess`.
- **Autocallable MC is plain (no antithetic/control variate), price-then-diagnostics (step
  12).** The engine runs plain seeded Monte Carlo and reports the SE, rather than reusing the
  antithetic/control-variate paths the BS-only engines have. Deliberate: the payoff and the
  path simulators must be *uniform* across BS/Heston/Merton, and antithetic pairing does not
  compose cleanly with the Poisson jumps (Merton) or the correlated variance draws (Heston),
  so keeping it plain keeps one code path across all three processes; the SE is small enough
  (~2e-4 at 400k paths) that the headline is >8 SE regardless. The `AutocallableResult`
  carries the autocall-probability breakdown alongside the price so the diagnostics come from
  the same simulation, not a second pass. **Heston Euler bias**: the full-truncation scheme
  has a small upward-converging discretisation bias in `heston_substeps` (0.989→0.992 over
  16→128 on the headline note); it only makes the "flat vol overprices" gap *more*
  conservative, so 64 sub-steps is used for the report and the direction is documented.
- **`estimate()` vs `npv` for MC stats.** The standard error is exposed via
  `MonteCarloEngine.estimate()` rather than threading a stats flag through the
  generic `npv`/`PricingEngine` seam. Deliberate; revisit only if other engines
  need to return stats.
- **mypy targets 3.12** (runtime is 3.11+) because current numpy stubs use the
  3.12 `type` statement; ruff `target-version = "py311"` guards 3.11 syntax.
- **Shared Carr–Madan transform (step 5).** `engines/_carr_madan.py` holds the
  model-agnostic `carr_madan_call_price(cf, ...)`; both `HestonFFTEngine` and
  `MertonFFTEngine` build a CF closure and call it. Extracted on the *second*
  consumer (CLAUDE.md §2), not up front.
- **No QuantLib Merton engine.** This QuantLib build exposes `Merton76Process` but
  not a wrapped jump-diffusion engine, so Merton has no QuantLib benchmark; the
  closed-form-vs-FFT agreement (~2e-7) is the effective challenge instead.

## How to resume

1. Run the full gate: `ruff format --check . && ruff check . && mypy quantica && pytest`
   (add `-m benchmark` for the QuantLib cross-checks; needs the `benchmark` extra).
2. Skim `git log --oneline` for the last coherent state.
3. Re-read `CLAUDE.md` (durable brief) and `.claude/skills/numerical-validation/SKILL.md`
   (the validation protocol every new numerical method must pass).
