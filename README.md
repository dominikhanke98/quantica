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
> interface) and the **first engine — the closed-form Black–Scholes analytic
> pricer with price and Greeks**, validated against QuantLib. The remaining
> engines (binomial tree, Monte Carlo, Crank–Nicolson PDE) and the cross-method
> convergence table follow.

## Architecture

Pricing follows an **Instrument / Process / Engine** separation:

| Concept | Question it answers | Example |
| --- | --- | --- |
| **Instrument** | *What* is the contract? | `EuropeanOption` (strike, expiry, payoff) |
| **Process** | *How* does the market move? | `BlackScholesProcess` (spot, rate, div, vol) |
| **Engine** | *How* do we price it numerically? | `AnalyticEuropeanEngine`, `BinomialEngine` *(MC, PDE next)* |

```python
from quantica import (
    EuropeanOption,
    BlackScholesProcess,
    OptionType,
    AnalyticEuropeanEngine,
    BinomialEngine,
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
```

Recover the volatility implied by a quoted price (Brent, with a vega-driven
Newton fast path):

```python
from quantica import implied_volatility

implied_volatility(10.4506, option, process)   # -> 0.2  (process.vol is ignored)
```

The same `option` will be re-priced by swapping the engine once the tree, MC,
and PDE engines land — which is what makes the cross-method convergence test
natural.

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

### Convergence table

Cross-method convergence of the CRR tree to the closed form, generated
reproducibly by [`scripts/convergence_table.py`](scripts/convergence_table.py)
(rerun it to regenerate — do not hand-edit):

European call — S=100, K=100, r=0.05, q=0, sigma=0.2, T=1

| Method | Price | Abs. error vs analytic |
| --- | ---: | ---: |
| Black–Scholes (analytic) | 10.45058357 | — (reference) |
| Binomial CRR (N=10) | 10.25340904 | 1.97e-01 |
| Binomial CRR (N=50) | 10.41069154 | 3.99e-02 |
| Binomial CRR (N=100) | 10.43061166 | 2.00e-02 |
| Binomial CRR (N=500) | 10.44658514 | 4.00e-03 |
| Binomial CRR (N=1000) | 10.44858410 | 2.00e-03 |
| Binomial CRR (N=5000) | 10.45018364 | 4.00e-04 |

The absolute error halves each time `N` doubles — the signature of `O(1/N)`
convergence.

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
