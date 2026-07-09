---
name: numerical-validation
description: >-
  Validate any numerical / quantitative-finance implementation the way a model-validation
  team would (effective challenge). Use this skill WHENEVER you add or modify a pricer,
  Greek, Monte Carlo routine, PDE/tree solver, VaR/ES engine, or any numerical model in
  quantica — and whenever you write or review its tests. Covers cross-method convergence
  tests, benchmarking against QuantLib, analytical sanity checks (put-call parity, limiting
  cases), Greeks validation via bump-and-reval, Monte Carlo standard-error discipline, and
  the convergence-table format for the README. Trigger this even if the user only says
  "add tests", "check this pricer", "validate", "benchmark", or "does this converge" — the
  validation layer is the deliverable, not an afterthought.
---

# Numerical validation (effective challenge)

The purpose of `quantica` is not just to compute prices — it is to demonstrate that each
implementation is **independently correct and validated**. A number without validation is
not a deliverable. This skill defines the standard protocol every numerical model must pass
before it is considered done, and the reusable test patterns for each check.

Apply the checks in the order below. Not every model needs every check, but you must
consciously decide which apply and state why any is skipped.

---

## The validation checklist

For a new pricing method, work through:

1. **Analytical sanity checks** — known closed forms, put–call parity, arbitrage bounds, limiting cases.
2. **Greeks validation** — analytic Greeks vs bump-and-reval finite differences.
3. **Cross-method convergence** — the method agrees with at least one *other* method as its discretisation is refined.
4. **QuantLib benchmark** — agreement with the industry reference within a stated tolerance (marked `@pytest.mark.benchmark`).
5. **Monte Carlo discipline** (MC only) — report the standard error; assert agreement within a few SE, not an arbitrary absolute tolerance.
6. **Convergence table** — record the order-of-convergence / final agreement in a form the README can display.

State tolerances explicitly and justify them (machine precision for analytics, discretisation
error for trees/PDE, statistical error for MC). A tolerance with no rationale is a red flag.

---

## 1. Analytical sanity checks

The cheapest, most convincing tests. Always include the ones that apply.

- **Known closed-form values.** Hard-code a textbook example (e.g. Hull) and assert to ~1e-8.
- **Put–call parity.** `C - P == S*exp(-qT) - K*exp(-rT)` to machine precision for European options.
- **Arbitrage bounds.** `max(S - K e^{-rT}, 0) <= C <= S`; option value non-negative; monotone in spot and vol.
- **Limiting cases.** `vol -> 0` gives the discounted intrinsic value; very deep ITM call → forward minus discounted strike; `T -> 0` → intrinsic.

```python
import numpy as np
from quantica.pricing import EuropeanOption, BlackScholesProcess, OptionType, AnalyticEuropeanEngine

def test_put_call_parity():
    proc = BlackScholesProcess(spot=100, rate=0.05, div=0.02, vol=0.2)
    engine = AnalyticEuropeanEngine()
    call = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)
    put  = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.PUT)
    call.set_engine(engine); put.set_engine(engine)
    lhs = call.npv(proc) - put.npv(proc)
    rhs = proc.spot * np.exp(-proc.div * 1.0) - 100 * np.exp(-proc.rate * 1.0)
    assert lhs == np.isclose(lhs, rhs, atol=1e-10)  # machine-precision identity
```

---

## 2. Greeks validation (bump-and-reval)

Validate every analytic Greek against a central finite difference of the *price*. This
catches sign errors and calculus mistakes that price-only tests miss.

```python
def bump_and_reval(price_fn, param, h):
    """Central difference of a scalar price function w.r.t. one parameter."""
    return (price_fn(param + h) - price_fn(param - h)) / (2 * h)

def test_delta_matches_finite_difference():
    proc = BlackScholesProcess(spot=100, rate=0.05, div=0.0, vol=0.2)
    opt = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)
    opt.set_engine(AnalyticEuropeanEngine())
    price_of_spot = lambda s: opt.npv(proc.with_spot(s))   # noqa: E731
    fd_delta = bump_and_reval(price_of_spot, proc.spot, h=1e-4)
    assert np.isclose(opt.greeks(proc).delta, fd_delta, rtol=1e-5)
```

Notes: use a **central** difference (second-order accurate); pick `h` around `1e-4`–`1e-5`
for smooth payoffs; for MC Greeks prefer pathwise/likelihood-ratio estimators and validate
those against analytic BS Greeks rather than bumping (bump-and-reval on MC is noisy unless
you reuse the random stream).

---

## 3. Cross-method convergence

The headline evidence that the numerics are right: price *the same instrument* with every
engine and assert they agree as the discretisation is refined. Trees and PDE converge
deterministically; MC converges in the statistical sense (see §5).

```python
import pytest

@pytest.mark.parametrize("engine_factory, tol", [
    (lambda: BinomialEngine(steps=5000),       1e-2),   # O(1/N) CRR convergence
    (lambda: FiniteDifferenceEngine(nx=800, nt=800), 1e-2),
])
def test_engine_converges_to_analytic(engine_factory, tol):
    proc = BlackScholesProcess(spot=100, rate=0.05, div=0.0, vol=0.2)
    opt = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)
    opt.set_engine(AnalyticEuropeanEngine())
    reference = opt.npv(proc)
    opt.set_engine(engine_factory())
    assert abs(opt.npv(proc) - reference) < tol
```

Where feasible, also assert the **order of convergence** (error roughly halves/quarters as
steps double) — that is stronger evidence than a single tolerance and reads well in the
README. Fit `log(error)` vs `log(N)` and check the slope.

---

## 4. QuantLib benchmark (the effective challenge)

Independently reimplementing and then benchmarking against the industry reference *is* model
validation — make it explicit. Keep these in a dedicated group so they can be run or skipped
with `-m benchmark` / `-m 'not benchmark'`, and so a missing QuantLib install degrades
gracefully.

```python
import pytest

ql = pytest.importorskip("QuantLib")   # skip cleanly if benchmark extra not installed

@pytest.mark.benchmark
def test_european_call_matches_quantlib():
    # quantica
    proc = BlackScholesProcess(spot=100, rate=0.05, div=0.0, vol=0.2)
    opt = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)
    opt.set_engine(AnalyticEuropeanEngine())
    ours = opt.npv(proc)

    # QuantLib reference (build the analytic European engine)
    # ... construct ql.EuropeanOption + ql.BlackScholesProcess + ql.AnalyticEuropeanEngine ...
    theirs = ...  # ql_option.NPV()

    assert np.isclose(ours, theirs, rtol=1e-10)
```

When you benchmark, **record any divergence and its cause** (day-count, calendar,
compounding, continuous vs discrete dividends). "We match QuantLib except under X, because
Y" is a stronger validation narrative than silent agreement — surface it in the README.

---

## 5. Monte Carlo discipline

MC results are random variables. Do not test them with a fixed absolute tolerance.

- Always compute and expose the **standard error**: `se = sample_std / sqrt(n_paths)`.
- Assert the estimate is within **~3 SE** of the analytic/reference value (≈99.7% band).
- **Seed explicitly** via a passed-in `numpy.random.Generator` so tests are deterministic.
- Demonstrate **variance reduction** works: antithetic and/or control variates should shrink
  the SE materially versus naive MC at the same path count — assert the SE ratio.

```python
def test_mc_price_within_three_standard_errors():
    rng = np.random.default_rng(42)
    proc = BlackScholesProcess(spot=100, rate=0.05, div=0.0, vol=0.2)
    opt = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)
    opt.set_engine(AnalyticEuropeanEngine()); reference = opt.npv(proc)
    opt.set_engine(MonteCarloEngine(n_paths=200_000, antithetic=True, rng=rng))
    result = opt.npv(proc, return_stats=True)   # -> price, std_error
    assert abs(result.price - reference) < 3 * result.std_error

def test_control_variate_reduces_variance():
    rng = np.random.default_rng(0)
    # SE with control variate should be a fraction of naive SE at equal path count.
    ...
    assert se_cv < 0.5 * se_naive
```

---

## 6. Convergence table for the README

The README is a validation report. For each instrument, present a table the reviewer can
read at a glance — the artifact that proves the effective challenge happened.

| Method                 | Price      | Abs. error vs analytic | Notes                         |
|------------------------|-----------:|-----------------------:|-------------------------------|
| Black–Scholes (exact)  | 10.4506    | —                      | reference                     |
| Binomial (CRR, 5000)   | 10.4510    | 4e-4                   | O(1/N)                        |
| Crank–Nicolson PDE     | 10.4507    | 1e-4                   | 800×800 grid                  |
| Monte Carlo (200k, AV) | 10.4498    | 8e-4 (SE 2e-3)         | within 1 SE; antithetic       |
| **QuantLib**           | 10.4506    | 2e-11                  | industry reference            |

Generate this from a small reproducible script in `scripts/` (seeded), so `make table` or a
one-line command regenerates it. Never hand-type the numbers.

---

## Anti-patterns to reject

- A price with no validating test → not done.
- A fixed absolute tolerance on a Monte Carlo estimate → statistically meaningless; use SE.
- Global `np.random.*` in a model or test → non-reproducible; inject a `Generator`.
- Greeks tested only for sign or plausibility → validate against bump-and-reval.
- QuantLib imported into the `quantica` package → benchmark-only, tests only.
- Hiding a benchmark discrepancy → the discrepancy-plus-explanation is the deliverable.
