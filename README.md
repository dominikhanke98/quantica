# quantica

A focused, rigorously **tested and validated** quantitative-finance library in
Python, built as a public portfolio of quant engineering: clean API design,
hand-implemented core numerics, disciplined testing, and **independent
validation against an industry reference (QuantLib)**.

The narrative is deliberately that of *model validation*: every numerical
method is cross-checked against at least one other method and, where a
reference exists, benchmarked against QuantLib within a stated tolerance.

> **Status:** Phase 1 (derivatives pricing) in progress. Implemented so far: the
> pricing primitives (option contract, Black–Scholes market process, engine
> interface), an implied-volatility solver, and **all four engines — the analytic
> Black–Scholes pricer (price + Greeks), the CRR binomial tree, a Monte Carlo
> engine with variance reduction, and a Crank–Nicolson PDE solver** — cross-validated
> for convergence and benchmarked against QuantLib. The four-way cross-method
> convergence test is next.

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
