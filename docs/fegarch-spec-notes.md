# fEGarch clean-room — specification notes

> Working notes recording the **mathematical specifications** (from the cited papers and the
> `fEGarch` reference manual) that the clean-room port is implemented against, and the open
> convention questions to be resolved against the committed `fEGarch` **output fixtures**
> (`tests/fixtures/fegarch/`). This file is a *permitted* clean-room input: it records published
> mathematics and our own derivations only — **never** anything read from the `fEGarch` source
> (CLAUDE.md §12). Both Phase-0 items below (the Fernández–Steel skew convention and the
> average-Laplace form) are **RESOLVED** — locked against the committed fixtures; any future entry
> that is not yet pinned carries a **RECONCILE** marker.

---

## 1. Fernández–Steel (1998) skewing — the `snorm` / `sstd` / `sged` / `sald` variants

**Source.** Fernández, C. & Steel, M. F. J. (1998), "On Bayesian Modeling of Fat Tails and
Skewness," *JASA* 93(441). This is the skewing mechanism the `fEGarch` reference manual states its
skewed distributions use; the equations below are from the paper, not from any source code.

### 1.1 The skewed density

Given a **symmetric** unimodal base density `f` (in the QMLE context, the standardized base with
mean 0 and variance 1) and a skewness parameter `γ > 0`, the Fernández–Steel skewed density is

```
p(ε | γ) = [ 2 / (γ + 1/γ) ] · { f(ε/γ)·1_{ε ≥ 0} + f(γε)·1_{ε < 0} }.
```

`γ = 1` recovers the symmetric base exactly. `γ > 1` stretches the right tail (positive skew);
`γ < 1` stretches the left tail (negative skew). The density is continuous at `ε = 0`.

### 1.2 Raw moments of the (un-standardized) skewed variable

For the density above, the `r`-th raw moment has the closed form

```
E(ε^r | γ) = M_r · [ γ^{r+1} + (−1)^r / γ^{r+1} ] / (γ + 1/γ),

with   M_r = ∫₀^∞ s^r · 2 f(s) ds       (the r-th absolute moment of the symmetric base).
```

`M_r` depends only on the base (not on `γ`). In particular, with the base standardized so
`M_0 = 1` and `M_2 = 1` (unit variance):

- **Mean:**  `E(ε | γ) = M_1 · (γ − 1/γ)`  — using `[γ² − 1/γ²]/(γ + 1/γ) = γ − 1/γ`.
- **Second raw moment:** `E(ε² | γ) = (γ³ + 1/γ³)/(γ + 1/γ) = (γ² + 1/γ²) − 1`.
- **Variance:**  `Var(ε | γ) = (γ² + 1/γ²) − 1 − M_1²(γ − 1/γ)² = (1 − M_1²)(γ² + 1/γ²) + 2M_1² − 1`.

### 1.3 The standardization step (the part most likely to drift from fEGarch)

The QMLE convention is that the innovation `z_t` has **mean 0 and variance 1**. The raw skewed
variable `ε` above does **not** — its mean is `μ = M_1(γ − 1/γ)` and its standard deviation is
`σ = sqrt[(1 − M_1²)(γ² + 1/γ²) + 2M_1² − 1]`. So the standardized skewed innovation is the
location–scale transform

```
z = (ε − μ) / σ,     i.e.     f_z(z) = σ · p(σz + μ | γ),
```

with `μ`, `σ` as above. These are **fEGarch's standardization constants** — WP 2026-04 App. C.1
Eqs. 38–41, where `C_E = μ` and `C_V = σ` (with `s = γ`). At `γ = 1`, `C_E = 0` and `C_V = 1`.

### 1.4 Consistency with the `quantica` implementation

`quantica.timeseries.fegarch.distributions.FernandezSteelSkew` implements exactly §1.3:
`μ = M_1·(ξ − 1/ξ)` and `σ² = (1 − M_1²)(ξ² + ξ⁻²) + 2M_1² − 1`, with `M_1 = E|z_base|` the base's
first absolute moment, and standardizes `z = (ε − μ)/σ`. Substituting `M_2 = 1` into the moment
formula (§1.2) reproduces this `σ²` term-for-term, and these constants **equal App. C.1 Eqs. 39–40**
(they are algebraically the Lambert–Laurent form). So the standardization is confirmed against the
specification, not merely a plausible default.

### 1.5 Argument convention — RESOLVED (`skew = γ` directly)

**`fEGarch`'s `skew` argument equals `γ` directly** (no reparameterization). Confirmed against the
committed fixtures by `test_fernandez_steel_constants_match_fegarch_fixture`:

- The analytic skewness of `FernandezSteelSkew` at `xi = skew` matches `fEGarch`'s empirical
  skewness (from `rsnorm_s(n, skew=…)` etc.) within Monte-Carlo tolerance for every skewed label.
- The orientation matches: `skew < 1` ⇒ left-skew (negative), `skew > 1` ⇒ right-skew (positive)
  — e.g. `skew = 0.8` ⇒ skewness ≈ −0.34, `skew = 1.3` ⇒ ≈ +0.39.
- The full 13-point quantile grid of each skewed variant matches the fixtures to ~0.02.

No further reconciliation needed for the skew: the `xi` parameter **is** fEGarch's `skew`.

---

## 2. Average-Laplace (`ald`) — RESOLVED (scaled average-Laplace / Sargan)

**Source.** WP 2026-04 App. C.1, Eqs. 31–33 and 37. The `ald` is the **scaled average-Laplace
(Sargan)** density, *not* the plain Laplace: a symmetric density with exponential tails but a
degree-`P` polynomial shoulder, which becomes lighter-tailed (toward normal) as `P` grows. `P` is a
**fixed integer construction parameter** (`P ≥ 1`), profiled by `fEGarch` over a discrete grid
(Eq. 59) rather than continuously optimized — so in `quantica` it is a construction argument
(`AverageLaplace(P)`), **not** an estimated shape parameter (`param_names = ()`).

### 2.1 Standardized density (mean 0, variance 1)

With `ι = √(2(P+1))`, `B = 2^(−2P)·C(2P, P)`, and coefficients

```
c₀ = c₁ = 1,     c_j = [2(P − j + 1)] / [j(2P − j + 1)] · c_{j−1},   j = 2..P,
```

the density and CDF are (Eqs. 31–33)

```
f(z) = (ι·B/2)·exp(−ι|z|)·Σ_{j=0}^{P} c_j (ι|z|)^j,

F(z) = ½ + (B/2)·Σ_{j=0}^{P} c_j·j!·P(j+1, ι z)   for z ≥ 0,   F(z) = 1 − F(−z)   for z < 0,
```

where `P(a, x)` is the regularized lower incomplete gamma (`scipy.special.gammainc`). The quantile
has no closed form and is obtained by numeric inversion (bisection). `F(0) = ½`.

### 2.2 Absolute moments and the kurtosis identity (Eq. 37)

```
a(K) = B · [2(P+1)]^(−K/2) · Σ_{j=0}^{P} c_j · Γ(j + K + 1).
```

This gives `a(0) = 1` (normalized), `a(2) = 1` (unit variance), `a(1) = E|z|` (feeds the FS skew at
`K = 1`), and the **raw-kurtosis identity**

```
a(4) = 3 + 3/(P+1).
```

### 2.3 Fixture confirmation

`test_ald_form_matches_fegarch_fixture` locks this against the committed fixtures: for `P ∈ {2, 8}`
the exact `a(4) = 3 + 3/(P+1)` (= 4.0, 3.3̄) matches `fEGarch`'s empirical kurtosis (`ald_P2` ≈ 4.02,
`ald_P8` ≈ 3.33), decisively **not** the Laplace (kurtosis 6), and the 13-point quantile grid matches
to ~0.02. This replaces the earlier placeholder standardized-Laplace `AverageLaplace`.

---

*Add further specification derivations here as later phases (SM models, the fractional-differencing
operator, LM models, dual mean) are implemented — always from the papers/manual, never the source.*
