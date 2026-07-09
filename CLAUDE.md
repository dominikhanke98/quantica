# CLAUDE.md — `quantica`

> Persistent brief for Claude Code. Read at the start of every session. Keep this file dense: durable decisions, standards, and structure only. Detailed per-project specs live in `docs/`.

---

## 1. What this project is

`quantica` is a focused, rigorously **tested and validated** quantitative-finance library in Python, plus a set of thin interactive apps built on top of it. It is being built by a **PhD in Banking & Finance** as a public coding portfolio to support a move from academia into a quant role (model validation / buy-side systematic PM / derivatives desk).

**The audience is a hiring manager or senior quant reviewing the GitHub repo.** That reframes the goal: the *code quality, test discipline, and validation are themselves the deliverable*, as much as the finance. Optimise every decision for "would an experienced quant read this and think *this person can be trusted to ship correct, well-engineered models*."

**This is NOT a QuantLib rival.** It is a *personal toolkit* that demonstrates the author can design a clean API, hand-implement core numerics correctly, test them, and **independently validate them against an industry reference**. That "independent reimplementation + benchmarking" narrative is the point — it is literally what model validation is.

---

## 2. Core principles (non-negotiable)

1. **Thin UI over a tested core.** All quant logic lives in the `quantica` package and is fully usable + unit-tested *without* any UI. Apps (`apps/`) only import and call it — they contain zero pricing/portfolio logic. Build the UI **last**, once the core is correct.
2. **Every numerical method is validated against something.** A pricer must agree with at least one other method (cross-method convergence) and, where a reference exists, be benchmarked against **QuantLib** within a stated tolerance. The convergence/benchmark table *is* a headline artifact.
3. **Reproducibility is mandatory.** One command reproduces every headline result and figure. RNG is always an explicit, seeded `numpy.random.Generator` passed in — **never** global `np.random.*`. No large data or secrets in the repo; data pulls are scripted/cached.
4. **Negative results and failure modes are deliverables.** Sections titled "where this breaks / conditions under which it fails" reflect the effective-challenge mindset and are encouraged, not hidden.
5. **Hand-write what demonstrates skill; lean on libraries for plumbing.** See §6.
6. **No premature abstraction.** Extract shared code into `core/` only when a *second* consumer actually needs it. Don't architect a grand framework up front — grow it as tracks are added.

---

## 3. Hand-write vs. lean-on

**Hand-implement (the implementation IS the demonstration):**
- Option pricers: Black–Scholes analytics + Greeks, binomial/trinomial trees, Monte Carlo, Crank–Nicolson PDE.
- Implied-vol solver, Greeks (analytic + pathwise/bump-and-reval).
- (Later) VaR/ES engines, validation test statistics (Kupiec, Christoffersen, Acerbi–Székely), backtester core, portfolio-construction estimators studied for their own sake (shrinkage, HRP).

**Lean on established libraries (reinventing adds no signal):**
- `numpy` / `scipy` — linear algebra, `scipy.stats`, `scipy.optimize`, `scipy.interpolate`, FFT.
- `pandas` (or `polars`) — data handling.
- `cvxpy` — convex optimisation (portfolio track).
- `arch` — GARCH/vol filtering (risk track).
- **`QuantLib`** — used ONLY as a test/benchmark dependency (dev group), never a runtime dependency. Benchmarking against it = effective challenge.

If a new runtime dependency is proposed, note the justification in the PR/commit; keep the runtime dependency set small.

---

## 4. Architecture

Pricing follows an **Instrument / Process / Engine** separation (the QuantLib pattern, deliberately, because it's clean and extensible):

- **Instrument** — the contract/payoff (`EuropeanOption`, later `AmericanOption`, `AsianOption`, ...). Knows strike, expiry, option type, payoff. Knows nothing about *how* it's priced.
- **Process** — the market dynamics (`BlackScholesProcess`: spot, risk-free rate, dividend yield, vol; later `HestonProcess`, `SABR`).
- **Engine** — a numerical method that prices an Instrument under a Process (`AnalyticEuropeanEngine`, `BinomialEngine`, `MonteCarloEngine`, `FiniteDifferenceEngine`).

Usage shape:
```python
option = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)
process = BlackScholesProcess(spot=100, rate=0.05, div=0.0, vol=0.2)
option.set_engine(AnalyticEuropeanEngine())
price = option.npv(process)
greeks = option.greeks(process)   # delta, gamma, vega, theta, rho
```
The same `option` can be re-priced by swapping the engine — which is exactly what makes the convergence test natural: price one instrument with all four engines and assert agreement.

---

## 5. Repository layout

```
quantica/
├── quantica/                     # the importable package
│   ├── __init__.py
│   ├── core/                     # shared primitives (grow lazily)
│   │   ├── types.py              # enums (OptionType), type aliases
│   │   └── math/
│   │       ├── rootfinding.py    # Brent/Newton for implied vol
│   │       └── interpolation.py
│   └── pricing/
│       ├── instruments.py        # payoffs, EuropeanOption, ...
│       ├── processes.py          # BlackScholesProcess, ...
│       ├── volatility.py         # implied-vol solver
│       └── engines/
│           ├── analytic.py       # Black–Scholes closed form + Greeks
│           ├── binomial.py       # CRR / Jarrow–Rudd trees
│           ├── montecarlo.py     # MC + variance reduction (antithetic, control variate)
│           └── finitediff.py     # Crank–Nicolson PDE
│   # portfolio/ validation/ backtest/  ← added in later phases
├── tests/                        # mirrors package layout
├── notebooks/                    # exploration & research writeups ONLY
├── apps/                         # thin Streamlit/Gradio front ends
├── docs/                         # detailed per-project specs, math notes
├── scripts/                      # data pulls, reproduce-results entrypoints
├── pyproject.toml
├── README.md                     # written as a research/validation report
├── CLAUDE.md                     # this file
└── .github/workflows/ci.yml      # ruff + mypy + pytest
```

---

## 6. Engineering standards

- **Python 3.11+.**
- **Type hints everywhere**, checked with **mypy** (strict-ish). Public functions fully annotated.
- **Lint + format: `ruff`** (`ruff format` + `ruff check`) — replaces black/isort/flake8.
- **Docstrings: NumPy style.** Every public class/function documents params, returns, and the reference (paper/textbook) for the method.
- **Tests: `pytest` + `pytest-cov`.** High coverage on the numerical core is expected; the core is the product.
- **No magic numbers.** Financial and numerical parameters (grid sizes, tolerances, step counts) are explicit, named, and documented.
- **Deterministic randomness.** Pass a seeded `Generator`; tests set fixed seeds.
- **Small, meaningful commits** telling a coherent story — never one "initial commit" dump. CI (ruff + mypy + pytest) must pass on every push; keep the green badge.

### Testing philosophy for numerical finance
- **Cross-method convergence**: analytic vs tree vs MC vs PDE agree within tolerance (MC within a few standard errors).
- **Benchmark/regression vs QuantLib**: separate test group (e.g. `-m benchmark`), asserting agreement within tolerance. This is the effective-challenge evidence.
- **Analytical sanity checks**: put–call parity, known closed-form edge cases, deep ITM/OTM limits, zero-vol limit.
- **Greeks**: validate analytic Greeks against bump-and-reval finite differences.
- Optional: property-based tests with `hypothesis` (e.g. price monotonic in vol/spot).

---

## 7. Common commands

```bash
# install (editable, with dev + benchmark deps)
pip install -e ".[dev]"

# run the full test suite
pytest

# coverage
pytest --cov=quantica --cov-report=term-missing

# run only benchmark-vs-QuantLib tests
pytest -m benchmark

# lint / format / typecheck
ruff check .
ruff format .
mypy quantica

# launch the pricing app (built last)
streamlit run apps/pricing_app.py
```

---

## 8. Current phase — Derivatives pricing (Phase 1)

**Goal of Phase 1:** price European options **four ways** (Black–Scholes analytic, binomial tree, Monte Carlo with variance reduction, Crank–Nicolson PDE), show they converge to each other, benchmark against QuantLib, add an implied-vol solver and Greeks, then wrap it in a thin Streamlit + Plotly explorer (sliders → live price, Greek profiles, a rotatable implied-vol surface, a convergence-table figure).

**Suggested build sequence (small tested increments):**
1. `core/types.py` (`OptionType`), `pricing/instruments.py` (`EuropeanOption` + payoff), `pricing/processes.py` (`BlackScholesProcess`).
2. `engines/analytic.py`: Black–Scholes price + analytic Greeks. Tests: known values, put–call parity, Greeks vs bump-and-reval.
3. `pricing/volatility.py`: implied vol via Brent/Newton. Test: round-trip (price → IV → price).
4. `engines/binomial.py` (CRR): converges to analytic as steps → ∞. Test the convergence.
5. `engines/montecarlo.py`: antithetic + control-variate variance reduction; report standard error. Test: within a few SE of analytic.
6. `engines/finitediff.py`: Crank–Nicolson. Test convergence to analytic under grid refinement.
7. Cross-method convergence test (all four agree) + QuantLib benchmark test group.
8. `apps/pricing_app.py`: thin Streamlit + Plotly UI over the above. **Last.**

**Deliverable narrative for the README:** "European options implemented independently four ways, cross-validated for convergence and benchmarked against QuantLib — here is the convergence table and the conditions under which each method degrades." That framing does double duty for the model-validation positioning.

---

## 9. Roadmap (later phases — do not build yet)

- **Phase 2 — Portfolio management:** signal → construction → backtest pipeline; covariance-estimation study (sample vs Ledoit–Wolf vs factor vs HRP); walk-forward + purged/embargoed CV; realistic costs/turnover/capacity.
- **Phase 3 — Model validation:** VaR/ES engines + backtests (Kupiec, Christoffersen, Acerbi–Székely, FRTB traffic-light & P&L attribution); PD model validation (AUC/Gini/KS, calibration, PSI); ML-model validation under SR 11-7 (SHAP, robustness, fairness).
- **Phase 4 — Financial engineering depth:** exotics (Asian, barrier, autocallable) via MC + Longstaff–Schwartz; stochastic vol (Heston via Carr–Madan FFT, SABR); XVA/exposure (EPE/PFE, CVA) reusing the MC core; AAD for Greeks.

Apps by track: **Streamlit + Plotly** (pricing, portfolio), **Gradio on Hugging Face Spaces** (ML-model-validation demo).

---

## 10. How to work in this repo (guidance for Claude Code)

- Prefer **small, tested increments**: implement one thing, write its test alongside, keep CI green.
- **Write the test with the code**, not after. For any method with a reference, add the QuantLib benchmark.
- **Don't touch later-phase modules** unless asked; stay within the current phase (§8).
- **Don't add runtime dependencies** casually — justify them; keep the runtime set lean.
- **Keep quant logic out of `apps/`** — if you're tempted to compute in the UI, it belongs in the package.
- When a design decision has trade-offs, **surface them briefly** rather than silently picking — the author is a domain expert and wants to make the call.
- Update this file when a durable architectural decision changes; keep it dense.
