# PROGRESS

Running state file for `quantica`. Updated at the end of each working session
(see CLAUDE.md §"Session close-out"). Concise and factual.

**Current phase:** Phase 1 pricing core complete (European options, four ways).
Now in a **derivatives-deepening track (Phase 4, taken ahead of Phases 2–3)**.

Phase-4 roadmap: **American ✓** → **LSM ✓** → exotics (path-dependent MC: Asian,
barrier) → Heston + calibration → Merton jump-diffusion.

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

## Next — Phase 4, step 3: exotics (path-dependent Monte Carlo)

- Reuse the LSM full-path GBM simulator to price **Asian** (arithmetic-average)
  and **barrier** (knock-in/out) options by Monte Carlo; validate against
  QuantLib's analytic/MC engines where available and against closed forms where
  they exist (geometric-average Asian has one; barrier options have Reiner–Rubinstein
  formulas).
- Consider extracting the GBM path simulator (currently inside `lsm.py`) into a
  shared helper once this second consumer lands (§2.6, extract on second use).
- Then (roadmap): Heston + calibration → Merton jump-diffusion.
- **Deferred Phase-1 deliverable — thin Streamlit + Plotly app**
  (`apps/pricing_app.py`): sliders → live price, Greek profiles, implied-vol
  surface, convergence table. Thin UI over the tested core; zero pricing logic in
  `apps/`.

Note the documented Rannacher/L-stability caveat in `finitediff.py` if PDE Greeks
are ever added; the American PDE uses PSOR (Brennan–Schwartz would be a faster
direct alternative for vanillas).

## Open design notes

- **Ignored-vol market carrier (TODO).** `implied_volatility` and
  `MonteCarloEngine`/tests take a `BlackScholesProcess` whose `vol` is a
  placeholder in IV's case (the unknown being solved for). It's documented and
  tested (answer independent of the passed vol), but the `vol=…` argument reads
  oddly. Consider a dedicated market/quote type carrying only `spot, rate, div`,
  or a `process.without_vol()` view — revisit if a third consumer appears.
- **`estimate()` vs `npv` for MC stats.** The standard error is exposed via
  `MonteCarloEngine.estimate()` rather than threading a stats flag through the
  generic `npv`/`PricingEngine` seam. Deliberate; revisit only if other engines
  need to return stats.
- **mypy targets 3.12** (runtime is 3.11+) because current numpy stubs use the
  3.12 `type` statement; ruff `target-version = "py311"` guards 3.11 syntax.

## How to resume

1. Run the full gate: `ruff format --check . && ruff check . && mypy quantica && pytest`
   (add `-m benchmark` for the QuantLib cross-checks; needs the `benchmark` extra).
2. Skim `git log --oneline` for the last coherent state.
3. Re-read `CLAUDE.md` (durable brief) and `.claude/skills/numerical-validation/SKILL.md`
   (the validation protocol every new numerical method must pass).
