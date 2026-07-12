# PROGRESS

Running state file for `quantica`. Updated at the end of each working session
(see CLAUDE.md §"Session close-out"). Concise and factual.

**Current phase:** Derivatives-pricing track complete (Phase 1 core + Phase 4
deepening). **Phase 3 (quant risk / model validation) now open** as the second
pillar — market-risk core landed.

Phase-4 roadmap: **American ✓** → **LSM ✓** → **exotics ✓** → **Heston pricer ✓**
→ **Heston calibration ✓** → **Merton jump-diffusion ✓**. **Derivatives-pricing
deepening track complete.**

Phase-3 roadmap: **market-risk VaR/ES + backtesting ✓** → **derivatives-P&L
integration ✓** (option book revalued through the pricers as the risk P&L source)
→ **credit-risk / PD validation ✓** → **ML-model validation (SR 11-7) ✓**.
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

## Next — risk pillar model families complete

Phase 3's planned families (market risk, derivatives P&L, credit/PD, ML) are all
landed. Options from here:

- **Backtest extensions** — FRTB P&L attribution; expected-shortfall at the FRTB
  97.5% level end-to-end.
- **Apps** — the deferred thin Streamlit + Plotly pricing app and/or a risk
  dashboard (thin UI over the tested core; zero quant logic in `apps/`).
- **Phase 2 — systematic portfolio management** — the third pillar (signal →
  construction → backtest; covariance-estimation study; purged/embargoed CV).
- **Derivatives deepening (prior options)** — PDE Greeks + Rannacher start-up
  (cash in the `finitediff.py` L-stability note); autocallable.

Earlier open strategic options (still valid if pivoting back to derivatives):

## Prior note — strategic options at the derivatives/risk branch point

The **derivatives-pricing deepening track is complete**: four-way European core,
American options, LSM, path-dependent exotics, Heston (pricer + calibration), and
Merton jump-diffusion — all cross-validated and (where a reference exists)
QuantLib-benchmarked. This is a natural branch point; **decide the direction before
writing code.** The choice hinges on the target role:

- **(A) Go deeper in derivatives** — sharpens a *derivatives-specialist* profile
  (pricing quant / derivatives desk / model validation of pricing models). Next
  candidate steps, in rough priority:
  - **PDE Greeks + Rannacher start-up** — cash in the documented L-stability caveat
    in `finitediff.py`: Crank–Nicolson's non-L-stable behaviour pollutes Greeks near
    the strike/at short T; two fully-implicit (Rannacher) start-up steps fix it.
    Natural next increment, self-contained, reuses the existing PDE engine, and adds
    a *Greeks-from-PDE* capability validated vs the analytic/bump-and-reval Greeks.
  - **Autocallable** (optional, bigger) — a structured exotic priced by MC (and/or
    PDE), exercising the LSM/path machinery on a real payoff; good portfolio piece
    but heavier.
- **(B) Pivot to a new track** — broadens to a *three-track generalist* profile
  (the CLAUDE.md §9 roadmap), which reads as "model validation / buy-side" breadth:
  - **Phase 2 — systematic portfolio management** (signal → construction → backtest;
    covariance-estimation study; walk-forward + purged/embargoed CV; realistic
    costs). Streamlit + Plotly app.
  - **Phase 3 — quant risk / model validation** (VaR/ES engines + backtests: Kupiec,
    Christoffersen, Acerbi–Székely, FRTB traffic-light; PD/ML model validation under
    SR 11-7). This track most directly *is* the model-validation narrative.

**Recommendation to weigh (not yet decided):** if the job search skews toward
derivatives-desk / pricing-model-validation roles, do (A) PDE Greeks next (small,
high-signal, closes an already-documented loose end). If the target is a broader
model-validation / buy-side seat, pivot to (B) Phase 3 to demonstrate range beyond
pricing. **The author (domain expert) makes this call at the top of next session.**

Also still open regardless of the above:

- **Deferred Phase-1 deliverable — thin Streamlit + Plotly app**
  (`apps/pricing_app.py`): sliders → live price, Greek profiles, implied-vol
  surface, convergence table. Thin UI over the tested core; zero pricing logic in
  `apps/`. Best done once a track is chosen (pricing app vs portfolio/risk app).

Note the documented Rannacher/L-stability caveat in `finitediff.py` if PDE Greeks
are ever added (this is exactly step (A) above); the American PDE uses PSOR
(Brennan–Schwartz would be a faster direct alternative for vanillas).

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
