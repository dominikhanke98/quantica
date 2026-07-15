# PROGRESS

Running state file for `quantica`. Updated at the end of each working session
(see CLAUDE.md §"Session close-out"). Concise and factual.

**Current phase:** Derivatives-pricing track complete (Phase 1 core + Phase 4
deepening). **Phase 3 (quant risk / model validation) complete** across five
families (market risk, derivatives P&L, FRTB PLA, credit/PD, ML under SR 11-7).
**Capital-markets / portfolio track open** — multi-factor risk model complete
(stage 1 exposures/decomposition + stage 2 OOS estimator comparison). Next:
**Phase 2 — systematic portfolio management** (`quantica/portfolio/`); see "Next".

Capital-markets roadmap: **multi-factor risk model — stage 1 ✓** (exposures +
decomposition + Σ = BFBᵀ + D) → **stage 2 ✓** (OOS estimator comparison: sample vs
Ledoit–Wolf vs factor; ill-conditioning demo; bias stats) → portfolio
construction (signal → construction → backtest; purged/embargoed CV) — next.

Phase-4 roadmap: **American ✓** → **LSM ✓** → **exotics ✓** → **Heston pricer ✓**
→ **Heston calibration ✓** → **Merton jump-diffusion ✓**. **Derivatives-pricing
deepening track complete.**

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

## Next — Phase 2: systematic portfolio management (the third pillar)

The factor track (stages 1 + 2) is complete and was built precisely to feed this.
**Full-budget session** — new `quantica/portfolio/`. Framing that keeps it in this
repo's identity: *everyone ships a backtester; this ships the test of whether the
backtest means anything.* Three layers, headline last:

- **Construction** — mean-variance / minimum-variance / risk-parity portfolios via
  **`cvxpy`** (new runtime dep per CLAUDE.md §3 — lean on it, don't hand-roll the
  QP), consuming a `CovarianceEstimator` from `quantica/factor/` (so the estimator
  comparison directly informs which Σ the optimiser gets). Reuse
  `min_variance_weights` as the closed-form anchor to validate the cvxpy GMV.
- **Walk-forward backtest engine** — reuse `factor.evaluation.walk_forward_windows`
  (no-lookahead already tested); add realistic **costs, turnover, capacity**. Signal
  → construction → rebalance → P&L, seeded and reproducible.
- **Backtest-validity layer (THE deliverable)** — the model-validation discipline
  applied to strategy backtests, the thing generic backtesters omit:
  - **Deflated Sharpe Ratio** (Bailey–López de Prado): deflate the observed SR for
    the number of trials, non-normality (skew/kurtosis), and sample length.
  - **Probability of Backtest Overfitting (PBO)** via **combinatorially-symmetric
    cross-validation (CSCV)**: is the in-sample best strategy median-or-worse OOS?
  - **Purged + embargoed CV** (López de Prado) — extend the walk-forward splits so
    label/feature overlap can't leak across the train/test boundary.
  - **Minimum Track Record Length** — how long a track record must be for an SR to
    be significant at a confidence level.
  - **Known-truth validation (the headline check, same discipline as the whole
    repo):** run a deliberately **overfit noise search** (pick the best of many
    random signals on noise) → must be flagged spurious (**DSR ≈ 0, PBO high**);
    plant a **real signal** → must survive (DSR > 0, PBO low). Proves the
    overfitting detector detects overfitting.

  Lean on `numpy`/`scipy`/`pandas` for the statistics; the deliverable is the
  validity framework, not the estimators. Add a Streamlit + Plotly app last.

## Later — OPEN DIRECTION DECISION (after the factor track)

Two pillars stand complete: derivatives pricing (Phase 1 + 4) and quant risk /
model validation (Phase 3, all four model families). The options, with the
trade-off framed:

- **(A) Phase 2 — systematic portfolio management** — the third and final pillar
  (signal → construction → backtest; covariance-estimation study: sample vs
  Ledoit–Wolf vs factor vs HRP; walk-forward + purged/embargoed CV; realistic
  costs/turnover/capacity; Streamlit + Plotly app). **Best serves the
  all-three-fields goal**: completes the derivatives / risk / portfolio triad
  (CLAUDE.md §9) and rounds out the buy-side-generalist profile.
- **(B) Deepen the risk pillar further** — FRTB PLA is done (step 5); remaining
  incremental options are the FRTB expected-shortfall charge at the 97.5% level
  end-to-end (liquidity-horizon scaling, the regulatory ES aggregation). Smaller;
  strengthens the model-validation-specialist story.
- **(C) The apps** — the deferred thin Streamlit + Plotly pricing app
  (`apps/pricing_app.py`: sliders → live price, Greek profiles, IV surface,
  convergence table) and/or a risk dashboard. Thin UI over the tested core, zero
  quant logic in `apps/`; makes the portfolio *demonstrable* to non-readers.
- **(D) Derivatives deepening** — PDE Greeks + Rannacher start-up (cashes in the
  documented L-stability caveat in `finitediff.py`; the American PDE's PSOR could
  also be swapped for Brennan–Schwartz); or an autocallable on the LSM/path
  machinery.

**Weigh:** (A) if the target profile is the three-pillar generalist (the stated
CLAUDE.md goal); (C) is the cheap complement once a pillar is chosen — a pricing
app after (D), a risk dashboard after (B). Nothing blocks any option technically.

## Gaps in existing tools (accumulating — portfolio-narrative material)

Findings where standard libraries are silently wrong, missing, or opaque — and
this repo's independent implementation surfaced it. Add to this list as they occur.

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

- **Ignored-vol market carrier — RESOLVED (step 4a).** `implied_volatility` now
  takes a `Market` (spot, rate, div); there is no placeholder vol to ignore. The
  `Market` carrier is shared by `BlackScholesProcess` and `HestonProcess`.
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
