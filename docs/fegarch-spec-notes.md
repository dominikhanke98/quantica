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

## 3. GARCH(1,1) recursion + QMLE conditioning — RESOLVED (Phase 1)

**Source.** Bollerslev (1986) for the recursion; WP 2026-04 App. C.3 for the QMLE conditioning.

### 3.1 The model

Constant mean (fEGarch default, orders `P=Q=D=0`) and the plain GARCH(1,1) variance:

```
ε_t = r_t − μ,     σ²_t = ω + α·ε²_{t−1} + β·σ²_{t−1},
```

with `ω > 0`, `α ≥ 0`, `β ≥ 0`, `α + β < 1` (stationarity). The QMLE log-likelihood is the shared
engine's `Σ_t [−ln σ_t + ln f_z((r_t − μ)/σ_t)]`, maximized jointly over `(μ, ω, α, β)` and any
distribution shape parameters. fEGarch reports `mu`, `omega`, `phi1` (= `α`), `beta1` (= `β`), the
log-likelihood, and per-observation AIC/BIC = `(2k − 2ℓ)/n`, `(k·ln n − 2ℓ)/n`.

### 3.2 Pre-sample conditioning — confirmed by the fixture

The recursion needs `σ²_0`, `ε²_0`. Reconstructing the committed fixture's conditional-SD series
(`fit_garch11_norm_sigma.csv`) from fEGarch's reported parameters under each candidate:

| Pre-sample convention | max abs σ deviation | verdict |
| --- | --- | --- |
| `σ²_0 = ε²_0 = Var(r)` (unbiased, `ddof=1`) | **1.0e-17** | **match** (machine precision) |
| `σ²_0 = ε²_0 = mean(ε²)` (biased) | 2.3e-6 (~1.8e-4 rel) | no |
| `σ²_0 = ε²_0 = mean(ε²)` over first 50 | 2.2e-3 (~0.17 rel) | no |
| `σ²_0 = ω/(1−α−β)`, `ε²_0 = 0` | 8.3e-4 (~0.066 rel) | no |
| `σ²_0 = ε²_0 = ω/(1−α−β)` | 3.1e-4 (~0.025 rel) | no |

So fEGarch seeds with the **unbiased sample variance of the returns** (`Var(r)`, `ddof=1`),
mean-invariant since `Var(r−μ) = Var(r)`. fEGarch's `presample=50` argument does **not** change this
output (the full-sample unbiased variance reproduces the series exactly). Only `σ²_1` depends on the
seed directly; `t ≥ 2` use observed residuals, so an exact `σ²_1` fixes the whole series.

### 3.3 Fixture confirmation

Fitting `fit_garch(synthetic_returns, "norm")` reproduces fEGarch's fit: parameters to ≤ 3.4e-5
relative, log-likelihood to 6.7e-9, AIC/BIC to 1e-7, and the full conditional-SD series to a max
relative deviation of 7.6e-6 — machine-order (these are *exact* fit fixtures, unlike the
Monte-Carlo distribution fixtures of §1–2). The fit is done on internally rescaled returns (the MLE
is scale-equivariant) for numerical conditioning of the small-magnitude `ω`.

---

## 4. Asymmetric SM models GJR-GARCH / TGARCH / APARCH — RESOLVED (Phase 1)

**Sources.** Glosten, Jagannathan & Runkle (1993) for GJR; Zakoïan (1994) for TGARCH; Ding, Granger &
Engle (1993) for APARCH; WP 2026-04 App. C.3 for the QMLE conditioning.

### 4.1 One recursion, three models

Reconciling the committed fixtures (`fit_{gjrgarch,tgarch,aparch}11_norm_*`) against candidate
recursions shows that `fEGarch`'s `gjrgarch`, `tgarch` and `aparch` are the **single APARCH power
recursion** (Ding-Granger-Engle 1993) evaluated at three powers `δ`:

```
σ_t^δ = ω + φ₁·(|ε_{t-1}| − γ₁·ε_{t-1})^δ + β₁·σ_{t-1}^δ,     ε_t = r_t − μ,
```

with `ω, φ₁, β₁ ≥ 0`, `|γ₁| < 1`, `δ > 0`, and the same asymmetry kernel `(|ε| − γ₁ε)` throughout:

| model | power `δ` | recursion on | intercept `ω` scale (fixture) |
| --- | --- | --- | --- |
| **GJR-GARCH** | `δ = 2` | variance `σ²` | `~3.0e-6` (`σ²`-units) |
| **TGARCH** | `δ = 1` | std. dev. `σ` | `~2.3e-4` (`σ`-units) |
| **APARCH** | `δ` free (fitted `~2.41`) | `σ^δ` | `~4.8e-7` (`σ^δ`-units) |

The **two-order-of-magnitude `ω` gap between GJR and TGARCH is the fixture fingerprint of the
`σ²`-vs-`σ` recursion** — the intercept lives in different units. This was decisive: GJR's Glosten
*indicator* form `σ² = ω + (φ₁ + γ₁·𝟙[ε<0])·ε² + β₁·σ²` does **not** reproduce the GJR fixture (max
σ deviation `~1e-2`, both sign conventions), whereas the APARCH-at-`δ=2` kernel
`σ² = ω + φ₁(|ε| − γ₁ε)² + β₁σ²` matches it to machine precision. So `fEGarch`'s `gjrgarch` uses the
APARCH `δ=2` parameterization (equivalently: slope `φ₁(1−γ₁)²` on good news, `φ₁(1+γ₁)²` on bad
news), not the textbook indicator.

`δ` for APARCH is a **free continuously-estimated QMLE parameter** (`fEGarch`'s default
`fix_delta = NA`, fitted `≈ 2.41`), bounded `δ ∈ (0, 4]` — contrast the ALD's discrete profiled `P`
(§2), which is a fixed construction argument.

### 4.2 Recursion form is machine-exact; the pre-sample carries the residual

Seeding each recursion with the **fixture's own `σ_0`** and stepping forward with the reported
parameters reproduces the whole tail `σ_{1:}` to `≤ 1e-15` for all three — so the kernel form is
exactly `fEGarch`'s. The only open quantity is the pre-sample `σ_0`. The recursion needs `σ_0^δ` and
a pre-sample news-impact `kernel_0`; the reconciled convention is

```
σ_0^δ = Var(r)^{δ/2}   (unbiased, ddof=1),      kernel_0 = (1/n)·Σ_t |ε_t|^δ   (δ-th absolute moment),
```

i.e. the `σ^δ` state from the unbiased sample variance and the news-impact from the **expected
symmetric news impact** `E|ε|^δ` (the leverage term `−γ₁ε` has zero pre-sample mean by symmetry).
Only the combination `φ₁·kernel_0 + β₁·σ_0^δ` is identifiable from `σ_0`, so `kernel_0` cannot be
separated further from a single fixture. Realized σ-series deviations under this convention:

| model | σ max abs dev | σ max rel dev | note |
| --- | --- | --- | --- |
| GJR (`δ=2`) | `~3e-9` | `~3e-7` | `E|ε|² = mean(ε²)` |
| TGARCH (`δ=1`) | `~1e-8` | `~1e-6` | `E|ε|` (the 1st absolute moment, ≪ `sd`) |
| APARCH (`δ≈2.41`) | `~9e-5` | `~7e-3` | see below |

**APARCH pre-sample is a flagged residual.** At the fitted `δ ≈ 2.41` the `δ`-th absolute moment
`E|ε|^δ` does not reproduce `fEGarch`'s (unpublished) pre-sample state — the implied `kernel_0` sits
closer to `(mean ε²)^{δ/2}` than to `E|ε|^δ`, and no single closed-form moment matches all three
models to machine precision (TGARCH wants `E|ε|`, APARCH wants `(mean ε²)^{δ/2}`; they coincide only
at `δ=2`). This is an **open reconcile item** resolvable only with more fixtures or the (forbidden)
source; it affects **only `σ_0`** (decaying thereafter), the recursion form is exact, and it is
surfaced honestly in the APARCH fixture-match tolerances (loglik dev `~6e-3`, σ rel `~7e-3`). We use
the principled `E|ε|^δ` because it is theoretically the expected symmetric news impact and is
near-exact for GJR/TGARCH.

### 4.3 Fixture confirmation + reduction anchors

`fit_gjr` / `fit_tgarch` match `fEGarch` to GARCH-level tolerance (parameters `≤ ~2e-4` relative,
log-likelihood `≤ ~3e-6`, σ-series `≤ 1.2e-5` relative); `fit_aparch` matches `δ` to `1.4e-3`
relative and the well-identified `β₁`/`γ₁` tightly, with the larger `ω`/loglik/σ residuals above from
the pre-sample. Reduction anchors hold structurally: `γ₁ = 0` collapses GJR to the plain GARCH news
impact `ε²`, and `δ = 2` makes `aparch_recursion` identical to `gjr_recursion` (max dev `0`). Fits are
done on internally rescaled returns (the MLE is scale-equivariant; `ω` scales as `scale^δ`).

This completes the Phase-1 short-memory family (GARCH / GJR / TGARCH / APARCH).

---

*Add further specification derivations here as later phases (the EGARCH family, the
fractional-differencing operator, LM models, dual mean) are implemented — always from the
papers/manual, never the source.*
