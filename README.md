# quantica

A focused, rigorously **tested and validated** quantitative-finance library in
Python, built as a public portfolio of quant engineering: clean API design,
hand-implemented core numerics, disciplined testing, and **independent
validation against an industry reference (QuantLib)**.

The narrative is deliberately that of *model validation*: every numerical
method is cross-checked against at least one other method and, where a
reference exists, benchmarked against QuantLib within a stated tolerance.

> **Status:** Phase 1 (European derivatives pricing) core complete: the pricing
> primitives, an implied-volatility solver, and **four engines — analytic
> Black–Scholes (price + Greeks), CRR binomial tree, Monte Carlo with variance
> reduction, and a Crank–Nicolson PDE solver** — cross-validated four ways and
> benchmarked against QuantLib. Now in a **derivatives-deepening track** (Phase 4,
> ahead of the portfolio/risk phases): **American options**, priced three
> independent ways — the tree and PDE generalize to early exercise
> (`max(continuation, intrinsic)`; the PDE as a linear complementarity problem
> solved by PSOR), joined by **Longstaff–Schwartz Monte Carlo** — cross-validated
> against each other and QuantLib and by exact structural theorems, since there is
> no closed form. **Path-dependent exotics** — Asian options (with a
> geometric-average control variate) and barrier options (with the discrete-monitoring
> bias corrected by a Brownian bridge). And **Heston stochastic volatility**, priced
> by the Carr–Madan FFT of the characteristic function (with the branch-cut-stable
> "little Heston trap" formulation), anchored by its Black–Scholes limit and
> **calibrated to an implied-vol surface** by nonlinear least squares — validated by
> synthetic parameter recovery, benchmarked against QuantLib's own calibrator, with
> the model's identifiability limits and the Feller condition reported honestly.
> Finally **Merton jump-diffusion**, priced two independent ways — the closed-form
> Poisson-weighted Black–Scholes series and the Carr–Madan FFT — which cross-validate
> each other to ~10⁻⁷, completing the derivatives-pricing deepening track. The
> **second pillar is now open — Phase 3, quant risk / model validation**: VaR/ES
> engines (historical, parametric, filtered-historical with a GARCH filter, Monte
> Carlo) with a full VaR **and** ES backtesting suite (Kupiec, Christoffersen, Basel
> traffic-light, and the Acerbi–Székely ES backtest), whose **own size and power are
> validated** — and the two pillars now meet: an **option book revalued through the
> pricing engines** is a drop-in P&L source for the risk layer, with the
> delta-normal / delta-gamma / full-revaluation divergence characterised as a
> model-validation study. Newly added: **credit-risk / PD model validation** —
> discrimination (AUC/Gini/KS with bootstrap CIs), the per-grade calibration
> battery (exact binomial, ECB Jeffreys, Hosmer–Lemeshow), and PSI stability, with
> the calibration tests' own size and power measured on known-truth grades. See
> the risk sections below.

## Architecture

Pricing follows an **Instrument / Process / Engine** separation:

| Concept | Question it answers | Example |
| --- | --- | --- |
| **Instrument** | *What* is the contract? | `EuropeanOption` (strike, expiry, payoff) |
| **Process** | *How* does the market move? | `BlackScholesProcess` (spot, rate, div, vol) |
| **Engine** | *How* do we price it numerically? | `AnalyticEuropeanEngine`, `BinomialEngine`, `MonteCarloEngine`, `FiniteDifferenceEngine` |

```python
import numpy as np
from quantica import (
    EuropeanOption,
    BlackScholesProcess,
    OptionType,
    AnalyticEuropeanEngine,
    BinomialEngine,
    MonteCarloEngine,
    FiniteDifferenceEngine,
)

option = EuropeanOption(strike=100.0, expiry=1.0, option_type=OptionType.CALL)
process = BlackScholesProcess(spot=100.0, rate=0.05, div=0.0, vol=0.2)

option.set_engine(AnalyticEuropeanEngine())
option.npv(process)      # 10.4506  (Black–Scholes closed form)
option.greeks(process)   # Greeks(delta=..., gamma=..., vega=..., theta=..., rho=...)

# Re-price the same contract with a different engine — the basis of the
# cross-method convergence test:
option.set_engine(BinomialEngine(steps=5000))
option.npv(process)      # 10.4502  (CRR binomial tree)

option.set_engine(FiniteDifferenceEngine(space_steps=400, time_steps=400))
option.npv(process)      # 10.4498  (Crank–Nicolson PDE)

# Monte Carlo with a seeded generator, antithetic + control-variate variance
# reduction, and the standard error on the estimate:
mc = MonteCarloEngine(1_000_000, rng=np.random.default_rng(0),
                      antithetic=True, control_variate=True)
result = mc.estimate(option, process)   # MCResult(price=..., std_error=..., n_paths=...)
```

Recover the volatility implied by a quoted price (Brent, with a vega-driven
Newton fast path):

```python
from quantica import implied_volatility

implied_volatility(10.4506, option, process)   # -> 0.2  (process.vol is ignored)
```

Swapping the engine to re-price the *same* `option` is exactly what makes the
cross-method convergence test natural; the Crank–Nicolson PDE engine is the last
to land.

## Validation

Every numerical method is validated as its own deliverable (see the
`numerical-validation` skill). The analytic engine currently passes:

- **Analytical sanity** — known textbook values (Hull), put–call parity to
  machine precision, arbitrage bounds, monotonicity in spot and vol, and the
  `σ→0` / `T→0` / deep-ITM limits.
- **Greeks** — every analytic Greek (delta, gamma, vega, theta, rho) matches a
  central bump-and-reval finite difference.
- **QuantLib benchmark** — price *and* all five Greeks agree with QuantLib's
  `AnalyticEuropeanEngine` to `rtol≈1e-10`, using matched conventions
  (continuous compounding, `Actual/365`, `NullCalendar`). Run with
  `pytest -m benchmark`.

The **implied-volatility solver** passes a price→IV→price round-trip and known-vol
recovery across a wide grid of strikes, maturities, and vols (calls and puts),
handles the no-solution cases (below intrinsic / above the upper bound) with
clear errors, and agrees with QuantLib's implied-vol solver to `rtol≈1e-6`.

The **binomial (CRR) engine** converges to the analytic price at first order
`O(1/N)` (verified by a log-error/log-N slope of ≈ −1 along an even-N
subsequence, handling the even/odd sawtooth), satisfies put–call parity exactly
on the lattice, and matches QuantLib's own CRR engine to an `O(1/N)` grid
difference.

The **Monte Carlo engine** simulates the exact GBM terminal price (no
time-stepping bias) from an injected, seeded `Generator`, reports the standard
error, and lands within ~3 SE of the analytic price. Antithetic and control-variate
variance reduction materially shrink the SE at equal path count (measured VRF
≈ 2× and ≈ 7× respectively for the ATM call below).

The **Crank–Nicolson PDE engine** solves the Black–Scholes PDE on a log-price
grid and converges to the analytic price at second order `O(h²)` (verified by a
log-error/log-steps slope of ≈ −2 — the error quarters each time the grid
doubles), recovers put–call parity up to discretisation error, and matches
QuantLib's own FD engine to an `O(h²)` grid difference.

### Convergence table

Cross-method convergence of all four engines to the closed form, generated
reproducibly by [`scripts/convergence_table.py`](scripts/convergence_table.py)
(seeded Monte Carlo; rerun it to regenerate — do not hand-edit):

European call — S=100, K=100, r=0.05, q=0, sigma=0.2, T=1
(Monte Carlo: 1,000,000 paths, seed 20240709)

| Method | Price | Abs. error vs analytic | Note |
| --- | ---: | ---: | --- |
| Black–Scholes (analytic) | 10.45058357 | — | reference |
| Binomial CRR (N=10) | 10.25340904 | 1.97e-01 | O(1/N) |
| Binomial CRR (N=50) | 10.41069154 | 3.99e-02 | O(1/N) |
| Binomial CRR (N=100) | 10.43061166 | 2.00e-02 | O(1/N) |
| Binomial CRR (N=500) | 10.44658514 | 4.00e-03 | O(1/N) |
| Binomial CRR (N=1000) | 10.44858410 | 2.00e-03 | O(1/N) |
| Binomial CRR (N=5000) | 10.45018364 | 4.00e-04 | O(1/N) |
| Crank–Nicolson PDE (50×50) | 10.39814510 | 5.24e-02 | O(h²) |
| Crank–Nicolson PDE (100×100) | 10.43758885 | 1.30e-02 | O(h²) |
| Crank–Nicolson PDE (200×200) | 10.44734182 | 3.24e-03 | O(h²) |
| Crank–Nicolson PDE (400×400) | 10.44977356 | 8.10e-04 | O(h²) |
| Monte Carlo (naive) | 10.45100434 | 4.21e-04 | SE 1.5e-02, VRF 1.0× |
| Monte Carlo (antithetic) | 10.44977472 | 8.09e-04 | SE 1.0e-02, VRF 2.0× |
| Monte Carlo (control variate) | 10.44881692 | 1.77e-03 | SE 5.6e-03, VRF 6.9× |

The CRR error halves each time `N` doubles (`O(1/N)`) and the Crank–Nicolson
error quarters each time the grid doubles (`O(h²)`), while the Monte Carlo error
is statistical, bounded by its standard error, which the variance-reduction
techniques shrink (VRF = variance-reduction factor vs naive at equal path count).

**Why this is the effective challenge.** The four methods derive from genuinely
independent mathematical foundations — a closed-form integral (analytic), a
discrete no-arbitrage lattice (CRR tree), a finite-difference solve of the
governing PDE (Crank–Nicolson), and stochastic sampling of the terminal
distribution (Monte Carlo). They share no code path, yet they converge to the
*same* price, each at its characteristic error order. That agreement is the
evidence that the implementation is correct: a bug in any one engine would break
the consensus rather than hide in it. The
[four-way cross-method test](tests/pricing/test_cross_method.py) enforces this
across strikes, maturities, and calls/puts with a non-zero dividend yield —
anchoring CRR and PDE to the analytic price within their discretisation bounds
and Monte Carlo within ~3 standard errors — and the QuantLib benchmarks
independently reconcile each engine against an industry reference. Independent
reimplementation plus benchmarking *is* model validation; this table and test
are that argument in miniature.

### American options — validating without a closed form

American options (early exercise permitted any time before expiry) have **no
closed-form price**, so Black–Scholes can no longer be the anchor. Three
independent methods price them and are cross-validated against each other:

- **Binomial tree** — the lattice takes `max(continuation, intrinsic)` at each
  node (early exercise is one line in the existing backward induction).
- **Crank–Nicolson PDE** — the PDE becomes a linear complementarity problem
  (`V ≥ intrinsic`) solved per time step by projected SOR on the *same*
  tridiagonal system.
- **Longstaff–Schwartz Monte Carlo (LSM)** — simulate full GBM paths (exact
  log-stepping, no path bias); at each exercise date regress the discounted
  continuation value on a monomial basis of the spot using **only in-the-money
  paths**, exercise where intrinsic beats the fitted continuation, and value each
  path by its **realized cashflow** under that policy.

The validation strategy adapts to the missing analytic reference:

- **Cross-method** — tree and PDE agree to their combined discretisation
  tolerance (≈ 2 × 10⁻³ at N = 2000 / 300×300 grid); LSM agrees with them within
  a few standard errors.
- **QuantLib benchmark** — matched against QuantLib's American CRR engine
  (agreement ≈ 3 × 10⁻⁵) and its American finite-difference engine (≈ 10⁻³),
  behind `-m benchmark`.
- **Exact structural theorems** (the strongest checks available without an
  analytic price):
  - *No-dividend American call = European call.* Early exercise of a call on a
    non-dividend-paying stock is never optimal, so on a given engine the tree and
    PDE prices are **identical to machine/solver precision** — the tree never
    takes the intrinsic branch and the PDE obstacle never binds — and LSM
    recovers the European price within statistical error. A sharp correctness
    check.
  - *Early-exercise premium ≥ 0.* American ≥ European on the same engine/grid
    (no discretisation slack), for both puts and calls, with the American put
    showing a strictly positive premium.

> **Highlighted validation insight — LSM is a lower bound, by design.** LSM
> values each path with a *sub-optimal* (regression-estimated) exercise policy,
> which can only leave value on the table, so the estimator is **biased low**.
> Averaged over seeds it lands **at or just below** the tree/PDE reference
> (measured ≈ 5 × 10⁻³ below) — and a richer regression basis, giving a better
> policy, provably moves the estimate *up* toward the reference. LSM sitting
> slightly below the tree/PDE price is therefore the **expected correctness
> signature, not a failure** — reading that gap correctly is itself part of the
> effective challenge.

The analytic and terminal-only Monte Carlo engines cleanly reject American
options (the latter directing to LSM).

### Path-dependent exotics — Asian and barrier options

Two path-dependent families are priced by Monte Carlo over a shared exact-GBM
path simulator (no path-discretisation bias), each with a highlighted validation
idea.

**Asian (average-price) options.** The payoff is on the *average* of the
underlying. The **geometric** average is lognormal, so it has a closed form
([`geometric_asian_price`](quantica/pricing/engines/asian.py), matched to
QuantLib to machine precision) — the analytic anchor. The **arithmetic** average
has no closed form, so it is priced by Monte Carlo.

> **Highlighted insight — a control variate motivated by finance.** The
> arithmetic and geometric averages of the *same* path are almost perfectly
> correlated, and the geometric payoff's mean is known exactly (its closed form).
> Using it as a control variate for the arithmetic price gives a **measured
> variance-reduction factor of ~880×** — the standard error drops by ~30× at the
> same path count. Unlike a generic control, this one comes from a real
> relationship between two traded contracts.

**Barrier options (knock-in / knock-out).** The vanilla payoff is switched on or
off if the underlying touches a barrier. The continuous-monitoring price is the
Reiner–Rubinstein closed form ([`barrier_price`](quantica/pricing/engines/barrier.py),
matched to QuantLib to machine precision across all eight types), but a real
contract is *discretely* monitored.

> **Highlighted insight — the discrete-monitoring bias, named and corrected.**
> Discrete Monte Carlo misses barrier crossings *between* observation dates, so
> it under-detects hits: a knock-out is biased **high**, a knock-in **low**. The
> bias shrinks as monitoring frequency rises. Rather than only brute-forcing more
> steps, the **Brownian-bridge correction** analytically restores each step's
> continuous crossing probability — at 50 steps it cuts the knock-out bias from
> **≈ 24 standard errors to under 1**, recovering the continuous price. In-out
> parity (knock-in + knock-out = vanilla) holds exactly on shared paths as a
> structural check.

### Heston stochastic volatility — Fourier pricing

The Heston model gives the variance its own stochastic factor (a mean-reverting
CIR process, correlated with the spot). This forced the market-data abstraction
to generalize: a lightweight [`Market`](quantica/pricing/processes.py) carrier
(spot, rate, dividend) is now shared by both `BlackScholesProcess` and
`HestonProcess`, and `implied_volatility` takes just a `Market` — there is no
longer a placeholder volatility to ignore.

Heston has no simple closed form, but its *characteristic function* is known, so
European options are priced by the [Carr–Madan FFT](quantica/pricing/engines/heston.py):
the option value is a Fourier integral of the CF, evaluated by one FFT. Two
highlighted ideas:

> **Highlighted insight — the Black–Scholes limit is the free correctness anchor.**
> As the vol-of-vol `ξ → 0` with `v0 = θ = σ²`, the variance becomes deterministic
> and Heston must collapse to Black–Scholes. The FFT price matches
> `AnalyticEuropeanEngine` to **~2 × 10⁻⁷** across strikes and maturities — a sharp,
> reference-free check on the whole CF-and-transform pipeline, exactly analogous to
> the no-dividend-call theorem for American options.

> **Highlighted insight — the branch cut, handled from the start.** The Heston CF
> contains a complex square root and logarithm; the textbook-naive formulation
> crosses the logarithm's branch cut for longer maturities, producing
> discontinuities that silently corrupt the integration. We use the **"little
> Heston trap"** (Albrecher et al.) formulation — `g = (β−d)/(β+d)` with a `−dτ`
> exponent — which stays on the principal branch and is stable at all maturities.

The characteristic function is checked directly at its known values (`φ(u)=1` at
`u=0`; `φ = S₀^{iu}` at `t=0`), put–call parity holds, prices are arbitrage-free
(monotone in strike), and the price is stable across the damping factor `α` and
the FFT grid — `α` is exposed as a numerical knob, not hard-coded silently. The
FFT matches QuantLib's `AnalyticHestonEngine` to ~10⁻⁷ once day-count conventions
are aligned (behind `pytest -m benchmark`).

### Heston calibration — fitting the surface, and the honesty about it

A pricer is only half the story: the model has to be *calibrated* to the market.
[`calibrate_heston`](quantica/pricing/calibration.py) fits the five parameters
`(v0, κ, θ, ξ, ρ)` to a vanilla implied-vol surface by nonlinear least squares
(`scipy.optimize.least_squares`), pricing each quote with the `HestonFFTEngine` and
reusing the step-3 implied-vol solver to move between price and vol space. The
design decisions — objective, weighting, bounds, the Feller handling, and the
identifiability diagnostics — are the deliverable; the optimizer itself is a
library call.

**Vol space is the default.** Residuals are measured in implied-vol points, not
price, because a fixed price error is a huge vol error for a cheap OTM option and a
tiny one for an expensive ITM option — a price-space fit silently overweights deep
ITM quotes, which carry the least information about the smile. Each model price is
inverted on the *out-of-the-money* option at that strike, where vega is largest and
the inversion is best-conditioned.

> **Highlighted insight — synthetic recovery is the reference-free proof.** Generate
> a surface from *known* Heston parameters, calibrate back, and check the machinery
> returns them. Because the same FFT engine generates and fits the surface, engine
> discretisation is not a confounder — a noise-free surface is recovered to solver
> tolerance, which is the rigorous check that the calibration is correct (no
> external reference needed). On top of that, our fit and QuantLib's own
> `HestonModelHelper` + Levenberg–Marquardt calibration both recover the truth and
> agree (behind `pytest -m benchmark`).

```
| Parameter | Truth  | Recovered | Abs. error |
| --------- | -----: | --------: | ---------: |
| v0        | 0.0400 |    0.0400 |    7.8e-10 |
| kappa     | 2.0000 |    2.0000 |    2.0e-07 |
| theta     | 0.0500 |    0.0500 |    5.6e-10 |
| xi        | 0.3000 |    0.3000 |    6.6e-08 |
| rho       | -0.7000|   -0.7000 |    1.6e-07 |
```

> **Highlighted insight — identifiability, named rather than hidden.** A single
> surface does not pin all five parameters equally. `κ` (mean-reversion speed) and
> `θ` (long-run variance) trade off, so the objective is *flat* along that
> direction. [`profile_objective`](quantica/pricing/calibration.py) makes this
> concrete: pin one parameter, re-optimise the other four, and measure the width of
> the near-optimal valley. On the recovery surface **κ's valley is ≈ 3× wider (in
> relative terms) than ρ's** (±12.5% vs ±4.1%) — the surface pins the skew (ρ)
> tightly and the mean-reversion speed (κ) only loosely. Under measurement noise
> this shows up directly: `v0` and `θ` recover to ~3% while `κ` scatters ~20%. We
> report this rather than present a single-point fit as if it were unique.

> **Highlighted insight — the Feller condition is surfaced, not silently violated.**
> `2κθ ≥ ξ²` keeps the variance strictly positive. A fit that breaks it is still a
> valid Heston model, so by default the condition is *reported* via the existing
> `HestonProcess.feller_satisfied` flag rather than forced; an optional soft penalty
> (`feller_weight`) biases the fit toward the satisfying region when the caller
> wants the cleaner variance process, trading a little fit quality for it.

The compelling demo fits a hand-specified, *non-Heston* equity-index smile
(downward skew flattening with maturity) that Heston can only approximate: the fit
lands at **0.245 vol-point RMSE** with small but *structured* residuals (the model's
known short-maturity skew limitation), Feller satisfied. Regenerate the whole report
— recovery table, smile-fit slice, and identifiability — with
`python scripts/heston_calibration_report.py`.

### Merton jump-diffusion — two methods, cross-validated against each other

The last pricer in the derivatives track is Merton (1976) jump-diffusion, which
adds a compound-Poisson jump to the Black–Scholes diffusion
([`MertonProcess`](quantica/pricing/processes.py): diffusion vol `σ`, jump
intensity `λ`, jump mean `μ_J`, jump vol `σ_J`, composing the same `Market`
carrier). It is priced **two independent ways**:

- **Closed form** — a Poisson-weighted sum of Black–Scholes prices
  ([`MertonClosedFormEngine`](quantica/pricing/engines/merton.py)): conditional on
  `n` jumps the option is an ordinary Black–Scholes option with inflated variance
  and a shifted forward, so the price is `Σ_n P(n)·BS_n` — each `BS_n` delegated to
  the existing `AnalyticEuropeanEngine`. The series is truncated at a documented
  tolerance (the Poisson tail, bounded by `max(S, K)`, caps the truncation error).
- **Characteristic function + FFT** — Merton's CF (diffusion × compound-Poisson)
  plugs straight into the **same Carr–Madan transform** reused from the Heston
  engine (factored into [`_carr_madan.py`](quantica/pricing/engines/_carr_madan.py)
  now that a second model needs it).

> **Highlighted insight — the two methods are each other's validation.** The
> closed-form Poisson sum and the CF/FFT price agree to **~2 × 10⁻⁷** across strikes,
> maturities and call/put — a self-anchored cross-check that needs no external
> reference, exactly the role Heston's Black–Scholes limit plays. (QuantLib ships
> `Merton76Process` but does not expose a jump-diffusion *engine* in its Python
> wrapper, so this agreement *is* the rigorous check.) On top of it: `λ → 0`
> collapses to Black–Scholes, the CF is exact at its known points (`φ(0)=1`,
> `φ = S₀^{iu}` at `t=0`), put–call parity holds, the Poisson series converges
> geometrically with the truncation error provably under tolerance, and the FFT is
> stable across `α` and the grid.

> **Highlighted insight — jumps are the fingerprint of the short-dated smile.** With
> the *same* baseline diffusion vol (15%), Merton and Heston generate very different
> implied-vol smiles. A downward jump can move the spot far in little time, so
> Merton's short-dated smile is steep; a stochastic-vol diffusion needs time to
> spread the terminal distribution, so Heston's is milder and more persistent. This
> is the "why Black–Scholes fails" story told two ways
> (`python scripts/jump_diffusion_smile.py`):

```
| Model  | smile width T=0.1 | smile width T=1.0 | short/long ratio |
| ------ | ----------------: | ----------------: | ---------------: |
| Merton |          17.2 pts |           3.4 pts |             5.1× |
| Heston |           8.0 pts |           6.5 pts |             1.2× |
```

## Market risk — VaR/ES and the backtesting layer (Phase 3)

The second pillar of `quantica` is **quant risk / model validation**, and it is
built to the same identity: the deliverable is not the risk number but the
**evidence that the risk model is adequate**. The [`quantica.risk`](quantica/risk)
package pairs four VaR/ES engines with a full backtesting suite — and then
validates the backtests *themselves*.

The engines price the loss distribution of a [`Portfolio`](quantica/risk/portfolio.py)
four ways: **historical simulation** (empirical tail, no distributional
assumption), **parametric variance–covariance** (Gaussian closed form — fast, but
its normal tail *understates* risk, which is exactly the point of backtesting it),
**filtered historical simulation** (a GARCH(1,1) volatility filter from `arch` over
bootstrapped residuals, so the tail rides on the current conditional vol), and
**Monte Carlo** (which converges to the parametric closed form under a normal fit —
a deliberate cross-check — but can later consume the option pricers above as a
nonlinear P&L map). VaR and ES are reported together at 95 / 97.5 / 99%.

> **Highlighted insight — we validate the backtests themselves (size and power).**
> Running a backtest is not evidence unless the backtest works. Every test is put
> through a seeded Monte-Carlo study of its **size** (false-rejection rate under a
> correct model, target ≈ 5%) and **power** (rejection rate under a mis-specified
> one). This is the meta-level effective challenge, and it surfaces an honest
> finding — the Christoffersen independence test is *conservative* at 99% because
> exceptions are rare — rather than hiding it (`python scripts/risk_backtest_report.py`):

```
| Backtest                    | Size (target ≈ 5%) | Power (mis-specified) |
| --------------------------- | -----------------: | --------------------: |
| Kupiec POF (coverage)       |               3.8% |                100.0% |
| Christoffersen independence |               2.2% |                 75.6% |
| Acerbi–Székely ES (Z2)      |               4.6% |                100.0% |
```

> **Highlighted insight — a *correct* Expected-Shortfall backtest.** ES is **not
> elicitable** (Gneiting, 2011): no scoring function is minimised by the ES, so the
> count-and-compare logic that backtests VaR does not transfer, and many libraries
> simply omit ES validation. `quantica` implements the **Acerbi–Székely (2014)**
> tests, which sidestep elicitability by testing the *magnitude* of tail losses
> against the predicted ES with a Monte-Carlo null. This catches a subtle failure a
> VaR count test cannot: a heavy tail with the *right* VaR exceedance rate but an
> *understated* ES — the Z2 statistic turns clearly positive while the VaR test
> passes.

The worked example rolls all four engines out-of-sample over a fat-tailed,
volatility-clustered market (GARCH(1,1) with Student-t shocks) and lets the
backtests judge them — the parametric-normal model lands in the Basel **red** zone
while filtered historical simulation stays **green**:

```
| Engine                | Exceptions | Expected | Kupiec p | Basel zone | AS Z2 (ES) |
| --------------------- | ---------: | -------: | -------: | ---------- | ---------: |
| Historical simulation |         22 |       15 |    0.089 | yellow     |     +0.643 |
| Parametric (normal)   |         32 |       15 |    0.000 | red        |     +1.775 |
| Filtered HS (GARCH)   |         18 |       15 |    0.450 | green      |     +0.267 |
| Monte Carlo (normal)  |         32 |       15 |    0.000 | red        |     +1.774 |
```

Regulatory context, factually: the exception-count zones are the Basel supervisory
**traffic-light** framework (green / yellow / red with a capital-multiplier
add-on), and the whole "independent reimplementation + effective challenge" posture
is the SR 11-7 / model-validation discipline applied to a risk library. The
portfolio interface is deliberately a P&L-series abstraction so a book of options,
revalued through the pricers above, can later replace the linear asset portfolio
without touching the risk or backtesting code.

### Derivatives risk — the two pillars meet

That P&L-series seam is now cashed in: an [`OptionBook`](quantica/risk/derivatives.py)
holds option positions (each priced by whichever engine from the pricing pillar —
analytic European, binomial American, exotics — plus an optional underlying leg for
hedges), and turns seeded `MarketScenarios` (spot returns, optional vol shifts)
into a P&L distribution three ways:

- **Full revaluation** — reprice the whole book through the pricing engines under
  every scenario. Exact by construction: the risk path *is* the pricing path, so
  there is no drift between the two pillars (asserted to the last bit in the tests).
- **Delta-normal** — first-order P&L from the book's aggregate Greeks (computed by
  bump-and-reval through each position's own engine, so the approximation stays
  consistent with the numerics that price the book). Fast, classic, and *blind to
  gamma*.
- **Delta-gamma** — adds the ½ Γ δS² term at negligible cost.

Scenarios are instantaneous shocks (theta drops out), and all three methods are
evaluated on the *same* seeded scenario set — so any divergence is approximation
error, not sampling noise. The P&L vector feeds `empirical_var_es` and the
backtests **unchanged**.

> **Highlighted insight — when does the linear approximation break, and which way?**
> The answer is the sign of the omitted ½ Γ δS² term. For a **short-gamma** book it
> is a pure loss, so delta-normal *under*-states VaR (dangerous); for a
> **long-gamma** book it cushions losses, so delta-normal *over*-states VaR
> (expensive); for a near-linear book (Γ ≈ 0) the methods agree. Delta-gamma
> repairs the curvature error to ~1% (`python scripts/derivatives_var_report.py`):

```
| Book (99% 1-day VaR)             | Delta-normal | Delta-gamma | Full reval | DN error | DG error |
| -------------------------------- | -----------: | ----------: | ---------: | -------: | -------: |
| Deep-ITM call (near-linear)      |       295.37 |      295.36 |     295.36 |    +0.0% |    +0.0% |
| Long ATM straddle (long gamma)   |        33.22 |       11.31 |      11.18 |  +197.1% |    +1.2% |
| Short ATM straddle (short gamma) |        32.74 |       56.40 |      55.87 |   -41.4% |    +0.9% |
```

The existing Kupiec backtest, reused unchanged on realized full-revaluation P&L,
reaches the same verdict: on the short-gamma book the delta-normal VaR takes **43
exceptions where 7.5 are expected** (rejected, p ≈ 0) while the delta-gamma and
full-revaluation forecasts achieve nominal coverage (8 exceptions, p = 0.856). A
delta-hedged position is the microscope version of the same point: its linear P&L
is identically zero, and full revaluation exposes exactly the residual ½ Γ δS²
gamma P&L.

## Credit risk — PD model validation

A second model family in the risk pillar: validating **probability-of-default
models**, organised in [`quantica/risk/credit/`](quantica/risk/credit) along the
three dimensions a bank's (and the ECB's) validation framework uses —
**discrimination**, **calibration**, and **stability**. The package is
deliberately *model-agnostic*: every validator consumes model outputs (default
indicators, PD scores), never a fitted model object, so the library needs only
numpy/scipy; the champion/challenger fitting (scikit-learn) lives in the demo
script and tests. That separation *is* the model-validation posture — the
validator doesn't own the model.

Regulatory context, factually: per-grade PD calibration testing with the
**Jeffreys test** follows the ECB's *Instructions for reporting the validation
results of internal models* (the IRB validation-reporting regime under the
CRR/Basel IRB framework); the discrimination/calibration/stability triad and the
"effective challenge" posture are the SR 11-7 discipline applied to credit
models.

- **Discrimination** — AUC via the Mann–Whitney rank identity, Gini, KS, all with
  **stratified-bootstrap CIs** (resampling within class keeps low-default
  resamples non-degenerate). Validated three independent ways: rank form ≡
  trapezoidal ROC integration ≡ scikit-learn, to machine precision *including
  ties*, plus the analytic binormal anchors AUC = Φ(δ/√2), KS = 2Φ(δ/2) − 1.
- **Calibration (the centerpiece)** — per-grade **exact binomial** and **ECB
  Jeffreys** tests, **Hosmer–Lemeshow**, the per-grade validation table, and a
  reliability curve. On the demo book the mis-specified champion's *safest* grade
  is its most mis-rated (observed default rate ≈ 30× the assigned PD — the
  planted convexity): a grade-level table catches what a single AUC never would.
- **Stability** — PSI on the score and per-characteristic (CSI), with the
  0.10/0.25 bands labelled as the industry convention they are. On a monitoring
  sample with a planted leverage drift, the characteristic view attributes the
  shift to leverage alone — *why*, not just *that*.

> **Highlighted insight — validate the validators (again).** As with
> Kupiec/Acerbi–Székely on the market-risk side, the calibration tests are
> themselves put through a size/power study on simulated grades with known true
> PDs (`python scripts/pd_validation_report.py`):

```
| Test                        | Grade                | Size  | Power (true DR = 2× PD) |
| --------------------------- | -------------------- | ----: | ----------------------: |
| Exact binomial              | n=800, PD=2%         |  3.7% |                   94.0% |
| Jeffreys (ECB)              | n=800, PD=2%         |  5.6% |                   96.1% |
| Exact binomial              | n=150, PD=1% (LDP)   |  1.7% |                   18.2% |
| Jeffreys (ECB)              | n=150, PD=1% (LDP)   |  5.8% |                   35.4% |
| Hosmer–Lemeshow (dof = G)   | n=4000, true PDs     |  4.7% |                  100.0% |
| Hosmer–Lemeshow (dof = G−2) | n=4000, true PDs     | 11.5% |                       — |
```

> The honest findings: the **exact binomial test is conservative** — its
> discreteness pushes the size far below nominal and collapses it in low-default
> grades, costing real detection power; the **Jeffreys test holds near-nominal
> size and roughly doubles the power** on the low-default grade — measured here,
> and precisely the property for which the ECB instructions adopt it. And
> **Hosmer–Lemeshow needs the right null**: with true (non-estimated) PDs the
> statistic is χ²(G), while the textbook G−2 convention — meant for models fitted
> on the sample — over-rejects at 11.5%. Degrees of freedom are part of the
> validator, too.

> **Highlighted insight — ranking well and being calibrated are different
> properties.** On the synthetic book (known truth, planted nonlinearity) the
> gradient-boosting challenger out-discriminates the logistic champion by ~5 AUC
> points (0.922 vs 0.869, near the true-PD ceiling of 0.929) — yet *both* models
> are flagged by the calibration battery: the champion fails Hosmer–Lemeshow
> outright (χ² ≈ 2307), and the challenger still under-states PDs in specific
> grades (Jeffreys p < 0.01). The realistic validation verdict is not "the ML
> model wins" but "the challenger may be promoted only after recalibration" —
> reported as found, not smoothed over.

## Development

```bash
pip install -e ".[dev]"      # install with the dev toolchain

pytest                        # run the test suite
pytest --cov=quantica --cov-report=term-missing
pytest -m benchmark           # QuantLib cross-checks (needs the 'benchmark' extra)

ruff format . && ruff check . # format + lint
mypy quantica                 # type-check
```

CI (ruff · mypy · pytest) runs on every push; the QuantLib benchmark suite runs
as a separate job.
