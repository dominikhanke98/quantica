# PROGRESS

Running state file for `quantica`. Updated at the end of each working session
(see CLAUDE.md §"Session close-out"). Concise and factual.

**Current phase:** Phase 1 pricing core complete (European options, four ways).
Now in a **derivatives-deepening track (Phase 4, taken ahead of Phases 2–3)**.

Phase-4 roadmap: **American ✓** → **LSM ✓** → **exotics ✓** → **Heston pricer ✓**
→ **Heston calibration ✓** → **Merton jump-diffusion ✓**. **Derivatives-pricing
deepening track complete.** Next: the deferred thin Streamlit + Plotly app, or
pivot to Phase 2 (portfolio) / Phase 3 (model validation).

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

## Next — derivatives track complete

The four-way European core, American options, LSM, exotics, Heston (pricer +
calibration), and Merton are all done. Options from here:

- **Deferred Phase-1 deliverable — thin Streamlit + Plotly app**
  (`apps/pricing_app.py`): sliders → live price, Greek profiles, implied-vol
  surface, convergence table. Thin UI over the tested core; zero pricing logic in
  `apps/`.

Note the documented Rannacher/L-stability caveat in `finitediff.py` if PDE Greeks
are ever added; the American PDE uses PSOR (Brennan–Schwartz would be a faster
direct alternative for vanillas).

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
