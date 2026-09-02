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

### 4.2 Recursion form is machine-exact; pre-sample seeded per recursion — RESOLVED (reconciled)

Seeding each recursion with the **fixture's own `σ_0`** and stepping forward with the reported
parameters reproduces the whole tail `σ_{1:}` to `≤ 1e-15` for all three — so the kernel **form** is
exactly `fEGarch`'s. Only `σ_0` depends on the pre-sample. The recursion needs a `σ^δ` **state** seed
and a pre-sample **news-impact** `kernel_0`; only the combination `φ₁·kernel_0 + β₁·σ_0^δ` is
identifiable from a single fixture. The reconciled convention (proved by a candidate sweep across all
four fixtures):

```
σ_0^δ  = Var(r)^{δ/2}            (unbiased, ddof=1)                — all recursions
kernel_0 = Var(r)^{δ/2}         (the variance-power)              — σ² / σ^δ recursions: GARCH, GJR, APARCH
kernel_0 = (1/n)·Σ_t |ε_t|      (the first absolute sample moment) — σ-recursion: TGARCH (δ = 1)
```

Both `kernel_0` forms are `fEGarch`'s estimate of the expected news impact `E|ε|^δ`; the
**variance-power** `(Var r)^{δ/2}` and the **first absolute moment** `E|ε|` **coincide at `δ = 2`**
(GARCH, GJR) and **fork for `δ ≠ 2`**. The fork is real, not a modelling choice: the two models that
can distinguish the forms give **opposite verdicts** —

| model | δ | needed `kernel_0` | variance-power `Var^{δ/2}` | abs-moment `E|ε|^δ` |
| --- | --- | --- | --- | --- |
| GARCH | 2 | `Var₁` (exact) | ✓ `1.7e-18` | `≈` (mean ε² = `Var₀`) |
| GJR | 2 | `≈ Var` | ✓ `~2e-7` | ✓ `~3e-9` |
| **TGARCH** | **1** | `E|ε| = 9.63e-3` | ✗ `sd = 1.26e-2` → **`2.4e-4`** | ✓ `mean|ε|` → **`1.5e-8`** |
| **APARCH** | **2.41** | `≈ Var^{δ/2} = 2.62e-5` | ✓ → **`1.5e-7`** | ✗ `mean|ε|^δ = 3.27e-5` → **`9e-5`** |

(deviations are realized `σ_0` errors). Each single form nails **three of four**; TGARCH demands the
abs-moment, APARCH the variance-power. We therefore seed **per recursion**: the σ²/σ^δ recursions use
the variance-power, the TGARCH σ-recursion uses the first absolute moment. This is legible in
`asymmetric.py` (`_SEED_VARIANCE_POWER` vs `_SEED_ABS_MOMENT`), not a hidden branch.

**Epistemic status (honest).** This is an **empirical reconciliation against the committed output**,
not a proven internal identity: the `fEGarch` source is never consulted (CLAUDE.md §12). It is the
convention that *reproduces `fEGarch`'s output* to fixture tolerance, and it is the simplest such
convention consistent with all four fixtures. A **corroborating** sign that the pre-sample is
genuinely per-model (not one universal formula) is a `ddof` split even at `δ = 2`: GARCH is exact
only with the **unbiased** `Var₁` kernel seed (`1.7e-18`), whereas GJR is exact only with the
**biased** `Var₀ = mean(ε²)` (`1.7e-18`) — a `~2e-7` difference, comfortably below tolerance, so the
code uses the clean unbiased `Var₁` throughout.

### 4.3 Fixture confirmation + reduction anchors

Under the per-recursion seed all four match at the **same tier**: parameters `≤ ~1.4e-3` relative
(the weakly-identified `ω` is the loosest), log-likelihood `≤ ~2e-5`, information criteria `≤ 1e-6`,
and the conditional-SD series to `≤ ~1e-4` relative:

| model | params (worst) | loglik dev | σ max rel dev |
| --- | --- | --- | --- |
| GARCH | `ω 3.4e-5` | `6.7e-9` | `7.6e-6` |
| GJR | `μ 5.6e-5` | `1.9e-5` | `1.6e-5` |
| TGARCH | `μ 1.8e-4` | `2.7e-6` | `1.2e-5` |
| APARCH | `ω 1.4e-3`, `δ 1.5e-4` | `1.3e-5` | `9.8e-5` |

Reduction anchors hold: `γ₁ = 0` collapses GJR to the plain GARCH news impact `ε²` — and because GJR
now seeds the pre-sample at the same `Var₁` as `garch_recursion`, this collapse is **machine-exact**
(`< 1e-15`), not merely tolerance-close; `δ = 2` makes `aparch_recursion` identical to
`gjr_recursion` (max dev `0`). Fits are on internally rescaled returns (the MLE is scale-equivariant;
`ω` scales as `scale^δ`).

This completes the Phase-1 short-memory family (GARCH / GJR / TGARCH / APARCH), all fixture-validated
at the same tolerance tier.

---

## 5. EGARCH(1,1) — the first EGF model — RESOLVED (Phase 2)

**Sources.** Nelson (1991) for the original EGARCH; WP 2026-04 (Schulz et al.) §2.1 + App. C.3 for the
EGF spec and QMLE conditioning; the EGF papers WP175 (Ayensu et al. 2026) and WP173 (Peitz et al.
2026). fEGarch source never consulted.

### 5.1 The Type-I log-variance recursion

EGARCH is the **Type-I** EGF model (an explicit asymmetry term; output stated in WP 2026-04's
representation (2)). For orders `(1, 1)` — representation (5) with `p = 1`, `q = 1` (so `q-1 = 0`,
i.e. no `ψ` term and the news-impact coefficient is `ψ₀ = 1`):

```
r_t = μ + σ_t·η_t,     ln σ²_t = ω + g(η_{t-1}) + ϕ₁·ln σ²_{t-1},
g(η) = κ·η + γ·(|η| − E|η|).
```

`κ` weights the **asymmetry** term (on `η`), `γ` the **magnitude** term (on `|η| − E|η|`). The
`E(η)=0` term drops out (η is standardized), leaving only the `E|η|` centering — which makes
`E[g(η)] = 0` and hence `ωσ = E[ln σ²]`.

**⚠ κ/γ orientation (a parameterization trap).** WP 2026-04 and WP173 write `g = κη + γ(|η|−E|η|)`
(**κ = asymmetry, γ = magnitude**); **WP175 swaps the letters** (`g_eg = γη + κ(|η|−E|η|)`). fEGarch
follows the WP 2026-04 / WP173 orientation — confirmed by the fixture signs: `κ = −0.0235`
(leverage: bad news raises vol) and `γ = +0.1573` (magnitude). We use `κ` on `η`.

### 5.2 Reported intercept: `ωσ` vs `ω`

fEGarch reports `omega_sig = ωσ = E[ln σ²_t]` (the **unconditional mean** of the log-variance),
**not** the recursion intercept. They are linked by `ω = ωσ·ϕ(1) = ωσ·(1 − ϕ₁)`, applied internally.
Numerically the fixture's `ωσ = −8.957` ⇒ `ω = ωσ(1−ϕ₁) ≈ −0.155`; `exp(ωσ) = 1.29e-4` is the model's
unconditional σ². The `(1,1)` parameter vector is `{μ, ωσ, ϕ₁, κ, γ}` (+ shape params) — no `ψ`.

### 5.3 Pre-sample conditioning (App. C.3) — confirmed to machine precision

Per App. C.3, the pre-sample **news-impact history is zero** (`g(η_t) = 0` for `t ≤ 0`) and the
pre-sample **log-variance is the log of the unbiased sample variance**:

```
ln σ²[0] = ω + ϕ₁·ln(Var(r)),   ddof = 1;   then   ln σ²[t] = ω + g(η[t-1]) + ϕ₁·ln σ²[t-1],
```

with `η[t-1] = (r[t-1] − μ)/σ[t-1]`, `σ = exp(ln σ²/2)`. Reconstructing the fixture's σ-series from
its reported parameters under this convention matches to **~4e-17** (`ddof=0` gives `~2e-6`). Note the
whole series — not just `σ_0` — is reproduced, since `ln Var(r)` is *exactly* fEGarch's seed (unlike
the short-memory σ₀ presample, which carried a small residual).

### 5.4 The distribution seam (E|η|) + a deferred follow-up

`g(η)` needs `E|η|`, the first absolute moment of the standardized innovation, sourced from the
Phase-0 distribution layer's `abs_moment` (`norm` → `√(2/π) ≈ 0.79788`) and **passed into the
recursion as a captured value — never hard-coded**. This makes the recursion↔distribution seam
general. Two items are the **documented Phase-2 follow-up** (norm is all that is validated here):
`abs_moment` was added to the `ConditionalDistribution` base (raising by default) and is implemented
on the four symmetric bases, but **not yet on the Fernández-Steel skew wrapper** (skewed-EGARCH needs
`E|η|` of the *standardized skewed* variable); and a **jointly-estimated shape** (`std`/`ged`) needs
`E|η|` recomputed at the current shape each iteration rather than captured once.

### 5.5 Fixture confirmation + checks

`fit_egarch(synthetic_returns, "norm")` reproduces fEGarch: parameters to **≤ 2.3e-5** relative
(`μ 1.4e-5, ωσ 4.5e-6, ϕ₁ 8.6e-7, κ 1.7e-5, γ 2.3e-5`), log-likelihood **9e-8**, AIC/BIC **7e-11**,
σ-series max **1.8e-5** relative. Reduction: **`κ = 0` makes `g` even** in `η` (reflecting the
residuals leaves σ unchanged to `~1e-19`), a non-zero `κ` breaks it. Known-truth simulation recovers
the planted `{ωσ, ϕ₁, κ, γ}` within a few SE. **Scale behaviour (as predicted):** a return rescale
`r → c·r` shifts `ωσ` **additively** by `ln(c²)` (it is a log-variance intercept), `μ` scales by `c`,
and `ϕ₁, κ, γ` are invariant — confirmed to `1e-4`. Fits are on internally rescaled returns.

Phase-2 remaining: Log-GARCH (Type-II, done — §6), MEGARCH and MLog-GARCH (Type-I, the generalized
`g_asy`/`g_mag` of App. C.1 Eqs. 7–9).

---

## 6. Log-GARCH(1,1) — the Type-II EGF model — RESOLVED (Phase 2)

**Sources.** WP 2026-04 §2.1 representations (10)-(13) + App. C.3; the EGF papers WP175 / WP173; the
Log-GARCH of Geweke (1986) / Milhøj (1987) / Pantula (1986). fEGarch source never consulted.

### 6.1 The Type-II recursion — no asymmetry, an ARMA on `ln σ²`

Type-II replaces Type-I's `g(η)` with the **log-square innovation** `ξ_t = ln(η²_t) − E[ln η²_t]`
(WP 2026-04 Eq. 10-11): `ln σ²_t = ωσ + γ(B)ξ_t`, `γ(B) = ϕ⁻¹(B)ψ(B) − 1`. Multiplying by `ϕ(B)`
gives the ARMA representation (Eq. 13); for orders `(1, 1)`:

```
ln σ²_t = ω + ϕ₁·ln σ²_{t-1} + (ψ₁ + ϕ₁)·ξ_{t-1},     ω = ωσ(1 − ϕ₁),   ξ_{t-1} = ln η²_{t-1} − E[ln η²].
```

Two structural points confirmed by the fixture:
- **The news-impact loading is the combined `(ψ₁ + ϕ₁)`** — not `ψ₁` alone. `ϕ₁` (AR on `ln σ²`) and
  `ψ₁` (MA on `ξ`) are reported *separately* but enter the recursion only through their sum.
- **`ψ` runs to lag `q` for Type-II** (`ψ(B) = 1 + Σ_{j=1}^{q}ψ_j Bʲ`), vs `q−1` for Type-I. So
  Log-GARCH(1,1) has a **free `ψ₁`** where EGARCH(1,1) had none — exactly the fixture's extra `psi1`.
- **No asymmetry term** — Type-II is structurally symmetric (it is Type-I with `κ = p_mag = M_mag =
  0`, WP 2026-04 line 349), so there is no `κ`-reduction check; the symmetry is by construction.

Parameter map: `mu` → μ, `omega_sig` → `ωσ = E[ln σ²]` (recursion intercept `ω = ωσ(1−ϕ₁)`),
`phi1` → ϕ₁, `psi1` → ψ₁. Vector `{mu, omega_sig, phi1, psi1}` (no `κ`/`γ`).

### 6.2 Log-square centering `E[ln η²]` — a new distribution moment

`ξ` centers on **`E[ln η²]`**, a *log*-moment (unlike EGARCH's `E|η|`). Phase-0 exposed only
`abs_moment`, so a **new `mean_log_sq` method** was added to the distribution base (raising by
default, like `abs_moment`), implemented on `norm` in closed form:

```
E[ln η²] = ψ(½) + ln 2 = −γ_Euler − ln 2 = −1.2703628      (η ~ N(0,1) ⇒ η² ~ χ²₁).
```

It feeds through the **same closure seam** as `E|η|` — the recursion receives it, never hard-coded.
`std`/`ged` (and the skewed variants) are deferred exactly like `abs_moment`; only `norm` is
validated, and `loggarch_sim`/`fit_loggarch` under non-norm raise until then.

### 6.3 Pre-sample conditioning — ddof=1 confirmed to machine precision

Per App. C.3 — identical to EGARCH — the pre-sample `ξ` history is zero and
`ln σ²[0] = ω + ϕ₁·ln(Var(r))` with `ddof=1`. Reconstructing the fixture's σ-series from its reported
parameters matches to **~9e-16** (`ddof=0` gives ~2e-6).

### 6.4 Near-common-root identification + the optimizer

On this series the fit sits at `ϕ₁ = 0.989`, `ψ₁ = −0.954` — the ARMA polynomials **nearly cancel**,
so the loading `(ψ₁ + ϕ₁) ≈ 0.035` is small and the likelihood is **flat/multimodal along the
`ϕ₁ ≈ −ψ₁` ridge** (WP 2026-04 §6's "least-stable family member"). Consequences, handled honestly:
- The default gradient optimizer (**L-BFGS-B**) **stalls** on the ridge (lands ~0.4-1.3 below
  fEGarch's log-likelihood). The QMLE engine gained a minimal, backward-compatible `method`/`options`
  argument (default unchanged); Log-GARCH uses the **derivative-free Nelder-Mead** simplex with a
  Log-GARCH-typical start (`ϕ₁=0.95, ψ₁=−0.9`) to land in fEGarch's basin.
- **fEGarch's fixture is a local, not global, optimum** — a different start finds a marginally
  *higher*-likelihood point (~0.4-0.6) with different `ϕ₁`/`ψ₁`. So `ϕ₁` and `ψ₁` *individually* are
  weakly identified (start/platform-sensitive), while the **σ-series, the log-likelihood and the
  combined `(ψ₁ + ϕ₁)` are strongly pinned**. Tests assert the pinned quantities **tight** (EGARCH
  tier) and the individual coefficients **looser** — the split is the diagnostic (a loose σ-series
  would be a real discrepancy, not the ridge). This is a data/model property, not a port defect.

### 6.5 Fixture confirmation

From the fEGarch-basin start, `fit_loggarch(synthetic_returns, "norm")` reproduces fEGarch:
σ-series max **6.4e-8** (rel 5.8e-6), log-likelihood **2.3e-9**, AIC/BIC **1.9e-12**, combined
`(ψ₁+ϕ₁)` to **8.6e-7** rel, and even `ϕ₁`/`ψ₁` individually to **~1e-7** (well inside the looser
guard). Scale behaviour is **additive in `ωσ`** by `ln(scale²)` (μ scales, ϕ₁/ψ₁ invariant), as for
EGARCH. This completes the Type-II branch; MEGARCH / MLog-GARCH (Type-I) remain.

---

## 7. Generalized Type-I EGF (MEGARCH + MLog-GARCH) — RESOLVED (Phase 2)

**Sources.** WP 2026-04 §2.1 Eqs. (7)–(9); WP173 §2.2.1–2.2.3 Eqs. (21), (25) (Peitz et al. —
MEGARCH/MLog-GARCH); the modulus-log transform is John & Draper (1980). fEGarch source never
consulted.

### 7.1 One recursion, three constant-sets

MEGARCH and MLog-GARCH are the same **Type-I** log-variance recursion as EGARCH
(`ln σ²_t = ω + g(η_{t-1}) + ϕ₁ ln σ²_{t-1}`, §5), differing only in the news-impact transformation
(WP 2026-04 Eqs. 7–9):

```
g(η)      = κ·{g_asy(η) − E[g_asy]} + γ·{g_mag(η) − E[g_mag]},
g_asy(η)  = sgn(η)·{ ln(|η|+M) if p=0 ;  [(|η|+M)^p − M]/p if p>0 },
g_mag(η)  =         { ln(|η|+M) if p=0 ;  [(|η|+M)^p − M]/p if p>0 },
```

with **fixed construction constants** `(M_asy, p_asy)`, `(M_mag, p_mag)` (chosen per model, **not**
fitted). The three models are constant-sets, using the modulus-log transform
`ζ(η) = sgn(η)·ln(|η|+1)` (John–Draper 1980):

| model | `(M,p)_asy` | `(M,p)_mag` | `g_asy` | `g_mag` | fitted γ |
| --- | --- | --- | --- | --- | --- |
| **EGARCH** | `(0,1)` | `(0,1)` | `η` | `|η|` | 0.157 |
| **MEGARCH** | `(1,0)` | `(0,1)` | `ζ(η)` | `|η|` | 0.158 |
| **MLog-GARCH** | `(1,0)` | `(1,0)` | `ζ(η)` | `ln(|η|+1)` | **0.284** |

The parameter vector is `{mu, omega_sig, phi1, kappa, gamma}` for all three (WP173 Eqs. 25/21 state
`g_me = κ{ζ−E ζ}+γ{|η|−E|η|}`, `g_ml = κ{ζ−E ζ}+γ{|ζ|−E|ζ|}`, with `|ζ(η)| = ln(|η|+1)`).

### 7.2 Centering — one new distribution moment; zero asymmetry centering for symmetric

`E[g(η)] = 0` (keeping `ωσ = E[ln σ²]`) needs each term centered:
- **Asymmetry** `E[g_asy] = E[sgn(η)·ln(|η|+1)] = 0` for **symmetric** innovations (`g_asy` odd,
  density even) — implemented as `0` on the symmetric bases; nonzero only under the FS-skew wrapper
  (deferred). Confirmed: reconstructing with `E_asy = 0` matches to ~6e-17.
- **Magnitude** `E[g_mag]`: `E|η|` (`abs_moment`) for EGARCH/MEGARCH; **`E[ln(|η|+1)]`** for
  MLog-GARCH — a **new distribution moment** `mean_log_modulus`, numerical for the normal (no
  elementary closed form): `E[ln(|η|+1)] = 0.5348222957` (vs `E|η| = 0.7978845608`). Same closure
  seam as `abs_moment` / `mean_log_sq`; `std`/`ged`/skewed deferred. (MEGARCH needs **no** new moment
  — it reuses `abs_moment`; MLog-GARCH's magnitude is the only place `mean_log_modulus` is required.)

### 7.3 Refactor + EGARCH regression guard (bit-for-bit)

EGARCH, MEGARCH and MLog-GARCH are built as one generalized Type-I recursion parameterized by the
`(M_asy, p_asy, M_mag, p_mag)` constant-set (`_type1_variance` in `egarch.py`); `fit_egarch` /
`fit_megarch` / `fit_mloggarch` are thin wrappers. The `p = 1` branch returns `|η|` directly (the
`[(|η|+0)^1 − 0]/1 = |η|` linear form, kept exact), so the **EGARCH `(0,1,0,1)` instance reproduces
the old hand-written recursion bit-for-bit** — verified by `np.array_equal` and by the unchanged
EGARCH fixture test (a hard regression guard: the refactor did not shift the validated path even at
1e-16).

### 7.4 Fixture confirmation — the γ-magnitude tell

Both reconstruct their σ-series from the fixture params to machine precision (MEGARCH ~6e-17,
MLog-GARCH ~4e-17). Being **well-identified** (not a near-common-root ridge like Log-GARCH), all
parameters match tight (EGARCH tier): parameters ≤ ~9e-5 relative, log-likelihood ≤ ~3e-8, σ-series
≤ ~1.6e-5 relative. The **γ magnitude is the parameterization tell**: MEGARCH's `γ ≈ 0.158` (it
keeps EGARCH's `|η|` magnitude) vs MLog-GARCH's **`γ ≈ 0.284`** (~1.8× larger, because `ln(|η|+1)` is
a compressed regressor). A mis-wired MLog-GARCH magnitude (using `|η|`) would return `γ ≈ 0.16` — the
fixture catches it, the same role the `ω`-units played for GJR/TGARCH. Pre-sample and scale
behaviour are identical to EGARCH (§5): App. C.3 conditioning (`ddof=1`), additive `ωσ` under
rescaling.

**This generalized Type-I seam is what Phase 4's fractionally-integrated variants (FIEGARCH,
FIMEGARCH, FIMLog-GARCH) will extend** — the same `g_asy`/`g_mag` transformation composed with the
`(1−B)^d` fractional-differencing operator. This completes the four Phase-2 EGF models
(EGARCH / Log-GARCH / MEGARCH / MLog-GARCH), all `(1,1)`/`norm` fixture-validated.

---

## 8. Fractional-differencing operator `(1−L)^d` — RESOLVED (Phase 3)

**Sources.** The binomial expansion of `(1−L)^d` (Hosking 1981; Bollerslev-Mikkelsen 1996, WP 2026-04
§2.1 Eq. 6); WP 2026-04 App. C.3 for the truncation / pre-sample policy, which cites Nielsen-Noël
(2021) for the FFT. fEGarch source never consulted. This is an **internal filter, not a fitted
model**, so it has **no output fixture** — validation is analytic + cross-method (§8.4), not
fixture-matching.

### 8.1 The operator and its coefficients

```
(1−L)^d = Σ_{i=0}^∞ b_i L^i,     b_0 = 1,     b_i = b_{i-1}·(i−1−d)/i = (−1)^i·C(d,i),
```

for any real `d` (`d > 0` fractional **differencing**, `d < 0` fractional **integration**). Applied
to a series `(x_t)`, the filtered value is the causal convolution `y_t = Σ_{i=0}^{L} b_i x_{t-i}`.
The coefficients are computed by the exact `O(L)` cumulative-product recursion; the recursion is the
`(−1)^i C(d,i)` binomial to machine precision. Limits: `d=0` → all `b_i=0` (`i≥1`), the identity;
`d=1` → `b=[1,−1,0,…]`, the ordinary first difference. Asymptotically `b_i ~ i^{-d-1}/Γ(-d)`.

### 8.2 fEGarch truncation / pre-sample policy (App. C.3) — ⚠ Phase 4 inherits this

The infinite series is truncated; **WP 2026-04 App. C.3** fixes the convention (verbatim: *"the
default for long-memory EGF models is L = n−1 with pre-sample values g(η_t) = ξ_t = 0 for
t = …,−1, 0, so that the infinite-length coefficient series are always practically truncated as far
back as needed so that the first observation time point is included"*):

- **Default truncation length `L = n − 1`** (the full within-sample history; values are capped at
  `n−1`, since under the pre-sample-0 convention there are no more lags to use).
- **Pre-sample of the filtered quantity = 0** for `t ≤ 0`.

This is **load-bearing even without a fixture**: every fractionally-integrated Phase-4 model
(FIEGARCH / FIMEGARCH / FIMLog-GARCH via the generalized Type-I seam §7; FIGARCH / FIAPARCH / FITGARCH
/ FIGJR via the SM recursions; FILog-GARCH via Type-II) composes its recursion with this operator
under **exactly this convention** — a wrong truncation policy here would propagate to all of them.
The `quantica` API exposes `trunc` (default `None` → `n−1`) and `presample` (default `0.0`) matching
this.

### 8.3 Two filter paths (the FFT is the performance path)

The filter is a causal convolution of `x` with the coefficient vector `b`, provided two ways: a
**direct** convolution (`O(nL)`) and an **FFT** convolution (`O(n log n)`) — the latter per the
Nielsen-Noël (2021) FFT approach App. C.3 cites, and the path the long-memory models will use for
efficiency. They agree to FFT round-off (~1e-15).

### 8.4 Validation (analytic + cross-method, in place of a fixture)

Reproducible seeded artifact: `scripts/fracdiff_convergence.py` (the phase's validation table).
Realized:
- **Coefficient accuracy** vs analytic `(−1)^i C(d,i)`: `~1e-16` (machine precision) for
  `d ∈ {0.2, 0.3, 0.45}`.
- **`d=0` identity / `d=1` first difference**: exact to `~1e-15`.
- **FFT-vs-direct cross-method**: agree to `~1e-15` (tolerance 1e-10 with platform margin) — the
  numerical-validation skill's cross-method check at fixed `(d, L)`.
- **Truncation error `‖y_L − y_full‖_∞`**: decays monotonically with `L` (`~1e-1` at `L=10` →
  `~1e-3` at `L=1000` → `0` at `L=n−1`), tracking the coefficient tail `|b_L| ~ L^{-d-1}`.
- **Long-memory ACF**: fractionally integrating white noise by `(1−L)^{-d}` gives a hyperbolic ACF
  `ρ(k) ~ k^{2d-1}` (fitted log-log slope negative, loosely around `2d−1` — finite-sample biased, so
  asserted in a wide band), with mid-lag autocorrelation ~100× a white-noise control — decisively
  long memory, not geometric.

---

## 9. FIEGARCH(1,d,1) — the first long-memory model — RESOLVED (Phase 4)

**Sources.** WP 2026-04 §2.1 Eqs. 4–6 (the fractionally-integrated EGF), App. C.3 Eq. 50 (the
truncated-MA(∞) recursion + the L=n−1 / pre-sample-0 policy inherited from §8.2). fEGarch source
never consulted; validated against the committed output fixture
`fit_fiegarch11_norm_{params.json,sigma.csv}` (generated once in R on the seeded
`synthetic_returns.csv`).

### 9.1 The model — composition, not new numerics

FIEGARCH replaces EGARCH's AR log-variance recursion with a **truncated MA(∞)** in the same Type-I
news impact `g(η) = κη + γ(|η| − E|η|)` (the EGARCH `(0,1,0,1)` constant-set, §7):

```
ln σ²_t = ωσ + Σ_{i=0}^{L−1} θ_i · g(η_{t−1−i}),     θ(B) = φ⁻¹(B)(1−B)^{−d}ψ(B) = Σ θ_i B^i,  θ_0 = 1,
```

which for orders `(1,d,1)` (`p=q=1` ⇒ `ψ(B)=1`) is `θ(B) = (1 − φ₁B)⁻¹(1 − B)^{−d}`. So the whole
model is built by **composition of two existing engines** — nothing new is hand-rolled:

- the **Phase-2 Type-I `g`** is reused verbatim through the `type1_news_impact(…, constants=
  EGARCH_CONSTANTS, mean_asy=0, mean_mag=E|η|)` seam (§7.3);
- the **`θ` coefficients** are the causal convolution of the geometric `φ⁻¹(B) = Σ_k φ₁^k B^k` series
  with the **Phase-3 fractional-integration** coefficients `b_i(−d)` of `(1−B)^{−d}` —
  i.e. `fracdiff_coeffs(−d, L)` at **negative** exponent (fractional *integration*, §8.1) — computed
  as `θ_i = Σ_{k=0}^{i} φ₁^k b_{i−k}(−d)`, FFT-accelerated (the Nielsen-Noël 2021 product App. C.3
  cites). `theta_coefficients(φ₁, d, L)` returns these with `θ_0 = 1`.

Parameter vector: `(mu, omega_sig, phi1, kappa, gamma, d)` — EGARCH's five plus the fractional `d`.

### 9.2 Pre-sample — the one real divergence from EGARCH (fixture-pinned)

The MA(∞) form has **no `ln σ²_{t−i}` feedback** and **no `ln Var(r)` seed** (unlike EGARCH's AR
form, §5.3). Two consequences, both confirmed against the fixture to `~1e-16`:

- the intercept is `ωσ` **directly** — *not* the AR-form `ωσ(1−φ₁)` (which reconstructs the fixture
  σ-series at relative error `4.9`, decisively wrong);
- the `g`-history is `0` for `t ≤ 0`, so `ln σ²_0 = ωσ` and **`σ_0 = exp(ωσ/2)` exactly** — asserted
  as the presample *tell* in the tests.

### 9.3 The fractional order `d ∈ (0, 1)` — bound NOT clamped at 0.5

`d = 0` collapses to short-memory EGARCH; `d ∈ (0, 0.5)` is weakly stationary long memory; and
`d ∈ (0.5, 1)` is mean-reverting but **non-stationary** long memory. The fEGarch fit lands
`d = 0.744` (upper regime) with a small `φ₁ = 0.39`: the high-persistence GARCH-simulated data is
captured by a large `d` and a small `φ₁`, the persistence **reparameterized from the AR term into the
slow θ tail**. The QMLE bound is therefore `d ∈ (1e-6, 0.9999)` — **not** clamped at 0.5 — and the
truncation is the full `L = n−1` (§8.2), so the whole slowly-decaying θ tail is load-bearing at large
`d`. Scale-equivariance is the EGARCH one (additive `ωσ ← ωσ + ln c²`, `μ ← cμ`, everything else
invariant), exact at the recursion level.

### 9.4 Fixture confirmation + checks

`fit_fiegarch(returns, cond_dist="norm")` matches the fEGarch fixture at **EGARCH tier** (the model is
well-identified despite the extra `d`): all six params to relative `≤ 3.4e-4` (`d` to `2.8e-5`),
log-likelihood to `7e-7`, AIC/BIC to `6e-10`, and the σ-series to `1.8e-6` (relative `6.3e-5`). The
recursion at the reported params reproduces the fixture σ to `1.4e-16`. Additional checks: the
`σ_0 = exp(ωσ/2)` presample tell; the **d→0 reduction** (`theta_coefficients(φ₁,0,L)` collapses to
EGARCH's geometric `θ_i = φ₁^i` to `~1e-16`); exact recursion-level scale-equivariance; and
known-truth recovery of all six params incl. `d` within a few SE (the `d`/`φ₁` SEs are wider — the
persistence trades off between them on a flatter surface). The fit recursion is `O(n²)` (g feeds back
through σ); simulation is `O(n log n)` (innovations given ⇒ g precomputed, one FFT convolution).

---

## 10. FIMEGARCH(1,d,1) + FIMLog-GARCH(1,d,1) — long-memory Type-I analogues — RESOLVED (Phase 4)

**Sources.** As FIEGARCH (§9): WP 2026-04 §2.1 Eqs. 4–6 + App. C.3 Eq. 50, plus the Type-I
constant-sets of §7 (Eqs. 7–9). fEGarch source never consulted; validated against the committed
fixtures `fit_fimegarch11_norm_*` and `fit_fimloggarch11_norm_*` (R spec functions `fimegarch_spec` /
`fimloggarch_spec` confirmed via `ls("package:fEGarch")` + `args()`, both spec-first
`function(orders=c(1,1), cond_dist=...)` identical to `fiegarch_spec`).

### 10.1 The clean analogue — same θ(B), swap the constant-set

These two are the fractionally-integrated **MEGARCH** and **MLog-GARCH**: the *exact* long-memory
analogue of the §7 relationship. They reuse **all** of FIEGARCH's machinery — the same
`theta_coefficients(φ₁, d, L)` composition `θ(B) = (1−φ₁B)⁻¹(1−B)^{−d}` (§9.1) and the same MA(∞)
pre-sample (intercept `ωσ` directly, `σ_0 = exp(ωσ/2)`, §9.2) — and differ **only** in the Type-I
`g`-transformation constant-set (and hence the magnitude-centering moment):

| model | constant-set `(M_asy,p_asy,M_mag,p_mag)` | asymmetry | magnitude | centering |
|---|---|---|---|---|
| FIEGARCH | `(0,1,0,1)` | `η` | `\|η\|` | `E\|η\|` (`abs_moment`) |
| **FIMEGARCH** | `(1,0,0,1)` | `sgn(η)ln(\|η\|+1)` | `\|η\|` | `E\|η\|` (`abs_moment`) |
| **FIMLog-GARCH** | `(1,0,1,0)` | `sgn(η)ln(\|η\|+1)` | `ln(\|η\|+1)` | `E[ln(\|η\|+1)]` (`mean_log_modulus`) |

Implementation is therefore **thin wrappers**, not reimplementations: `fiegarch.py` factors a private
`_fiegarch_variance(…, constants, mean_asy, mean_mag)` / `_fit_fiegarch(…, constants,
log_modulus_magnitude)` / `_sim_fiegarch(…)` — exactly mirroring `egarch.py`'s `_type1_variance` /
`_fit_type1` / `_sim_type1` seam — and the six public functions (`fimegarch_recursion` / `fit_fimegarch`
/ `fimegarch_sim` and the `fimloggarch_*` trio) are one-liners over it. The FIEGARCH public path is
unchanged (its 11 tests still pass bit-identically).

### 10.2 The gate that replaced spec-extraction

Because the composition was a known analogue, the build was gated on a **read-only reconstruction**
(before any model code): reconstruct each fixture's σ-series from its committed params using the
existing `theta_coefficients` + `type1_news_impact` at the model's constant-set. Both reconstructed to
**machine precision** — FIMEGARCH `1.28e-16`, FIMLog-GARCH `1.25e-16`, both with `σ_0 = exp(ωσ/2)` —
confirming the clean-analogue assumption (had either missed `~1e-15`, it would have signalled some
constant-set × θ(B) interaction requiring full spec extraction).

### 10.3 The γ-tell + fixture confirmation

The **γ-magnitude tell** carries over from §7.4 under fractional persistence: FIMEGARCH keeps the
`|η|` magnitude so `γ = 0.172` (≈ FIEGARCH/MEGARCH's ~0.17), while FIMLog-GARCH's compressed
`ln(|η|+1)` magnitude gives `γ = 0.321` — a **1.87×** ratio, the same log-modulus signature as the
short-memory pair. Both fitted `d ≈ 0.74` (upper regime), `φ₁ ≈ 0.39`, as FIEGARCH. `fit_fimegarch` /
`fit_fimloggarch` match their fixtures at **FIEGARCH tier**: all six params to relative `≤ 2e-4`,
log-likelihood to `≤ 2.4e-7`, AIC/BIC to `≤ 2e-10`, σ-series to `≤ 8.4e-7` (relative `≤ 8.2e-5`); the
recursion at reported params reproduces σ to `~1e-16`. Additional checks (per model): the
`σ_0 = exp(ωσ/2)` presample tell; the **d→0 reduction** to short-memory MEGARCH / MLog-GARCH (the FI
MA-form and the AR-form are one stationary process with different pre-sample seeds, converging
exponentially — agreement `< 1e-9` after `t = 100`, realized `~1e-11`); exact recursion-level
scale-equivariance; and known-truth recovery of all six params incl. `d`. FIMLog-GARCH under non-norm
is **deferred** (its `mean_log_modulus` centering is implemented only for the normal, as in §7).

---

## 11. FIGARCH(1,d,1) — the variance-recursion long-memory seam — RESOLVED (Phase 4)

**Sources.** Baillie-Bollerslev-Mikkelsen (1996) for the FIGARCH form; Conrad-Haag (2006) for the
ARCH(∞) non-negativity conditions; **WP175 §2.1 Eqs. 2.3-2.4** for the ω-direct parameterization.
fEGarch source never consulted; validated against the committed fixture `fit_figarch11_norm_*` (R
function `figarch()` confirmed via `ls`/`args` — a **data-first** function, there is no
`figarch_spec`). The pre-sample convention was resolved by matching the fixture to machine precision.

### 11.1 A new seam — the operator in the *variance* recursion (ω-direct, no feedback)

FIGARCH is the first **non-EGF** long-memory model: the `(1−L)^d` operator enters the
conditional-**variance** polynomial (WP175 Eq. 2.4), not the log-variance θ(B):

```
σ²_t = ω + Σ_{i=1}^{∞} θ_i ε²_{t−i},   ε_t = r_t − μ,   θ(B) = 1 − (1−φ₁B)(1−B)^d / (1−β₁B),
```

with `θ_1 = d + φ₁ − β₁` and all `θ_i ≥ 0` (Conrad-Haag). Three properties distinguish this seam:

- **ω-direct intercept.** The intercept is the **variance** `ω` (WP175 `ω*`) used *directly* — **not**
  the Baillie-Bollerslev-Mikkelsen / `arch`-package ARCH(∞) intercept `(1−β₁)⁻¹ω`. This was pinned
  empirically (`ω + Var ≈ mean h_fix`) and confirmed against WP175; the `arch` package's `(1−β)⁻¹ω`
  form is why it diverges at ~2e-3 (§11.4).
- **No feedback.** `σ²_t` is a **pure linear filter of observed `ε²`** — there is no η/σ² feedback
  (unlike every prior model). So the fit recursion is a single causal convolution
  (`figarch_variance_filter`), not a coupled step-by-step recursion, and the fit is fast (O(n log n),
  ~0.6 s). Simulation *does* couple (there `ε²=σ²η²` is generated), so it is a sequential recursion.
- **Reuses Phase-3 `fracdiff_coeffs` at +d.** `θ(B)` = `fracdiff_coeffs(+d)` ⊛ `(1−φ₁B)` ⊛ geometric
  `(1−β₁B)⁻¹` (FFT-accelerated) — the *positive* exponent `(1−B)^d` is fractional **differencing**,
  vs FIEGARCH's `−d` integration. Factored into public `figarch_coefficients` +
  `figarch_variance_filter`, the reusable primitives the SM-FI group inherits (§11.5).

Parameter vector `(mu, omega, phi1, beta1, d)`; `φ₁`/`β₁` the ARCH/GARCH lags, `d ∈ (0,1)`
(figarch's `drange`; fitted `d = 0.678` is upper-regime). ω scales **multiplicatively** by `c²` under
a return rescale (a variance intercept), unlike the EGF `ωσ` additive shift.

### 11.2 Pre-sample = 50 terms at Var(ddof=1) — resolved to machine precision

App. C.3's EGF convention (`L=n−1`, g/ξ=0) does **not** apply to FIGARCH. `figarch()`'s own defaults
are `trunc="none"`, `presample=50`, which WP171/WP175 do not fully specify. Resolved against the
fixture by exhaustive candidate testing: the pre-sample `ε²_{t−i}` (`t−i ≤ 0`) are seeded at the
**unbiased sample variance `Var(r)` (ddof=1)** for exactly the first **50** lags, `0` beyond;
`trunc="none"` = the θ(B) series runs full length. **Both parameters are decisive and unique:**

| convention | max\|σ − fixture\| |
|---|---|
| **50 terms, Var(ddof=1)** | **4.86e-17** (machine-exact) |
| 50 terms, Var(ddof=0) | 2.33e-6 |
| 49 or 51 terms, Var(ddof=1) | ~8e-6 |
| all pre-sample = 0 | 9.25e-3 |
| unconditional ω/(1−Σθ) (degenerate, Σθ→1) | 3.71e-2 |

Confirmed by an independent direct (non-FFT) recomputation (~1e-17 at every sampled `t`). This is a
**complete resolution** — no irreducible-from-output fork.

### 11.3 Fixture confirmation + checks

`figarch_recursion` at the reported params reproduces the fixture σ to **4.86e-17** (5.9e-17 with the
FFT coefficient path). `fit_figarch` matches at ~1e-5: all five params to relative `≤ 3.6e-5`,
log-likelihood to `7.8e-9`, AIC/BIC to `6.3e-12`, σ-series to `8.7e-8` (relative `7.3e-5`); the fit
converges in ~0.6 s. Additional checks: the 50-term/ddof=1 uniqueness (§11.2); the **d→0 reduction**
(`figarch_coefficients(φ₁,β₁,0,L)` collapses to GARCH's ARCH(∞) `θ_i = (φ₁−β₁)β₁^{i−1}` to ~1e-16);
exact recursion-level scale-equivariance (ω → c²ω); and known-truth recovery of all five params incl.
`d` within a few SE.

### 11.4 The `arch` cross-check (documented-divergence anchor)

`arch` 8.0.0's `FIGARCH` uses the BBM `(1−β)⁻¹ω` intercept and an EWMA backcast, so at the fEGarch
params it reproduces the fixture only to **2.1e-3** — a **rough sanity anchor** confirming the general
BBM form, **not** a machine-precision benchmark. The fEGarch fixture (matched to 4.86e-17) is the
primary anchor; the ω-direct + 50-term-Var convention is what distinguishes fEGarch's FIGARCH.

### 11.5 The short-memory FI group inherits this seam

`figarch_coefficients` (the `fracdiff(+d) ⊛ ARMA` θ-composition) and `figarch_variance_filter` (the
ω-direct linear filter with the 50-term `Var(ddof=1)` pre-sample) are factored as reusable primitives.
**FIAPARCH / FITGARCH / FIGJR** compose the *same* θ(B) and filter, differing only in the *news* term
fed to the filter (a power/asymmetry transform of `ε` instead of `ε²`) — exactly as the EGF FI group
shared `type1_news_impact`. This is the last major new seam of the long-memory family.

---

## 12. FIAPARCH(1,d,1) — the δ-power seam + a documented bounded-limit seed — RESOLVED (Phase 4)

**Sources.** Ding-Granger-Engle (1993) APARCH; Tse (1998) FIAPARCH; BBM (1996) + Conrad-Haag (2006)
for the FIGARCH ARCH(∞); **WP175 §2.1 Eqs. 2.9-2.10** for the ω-direct σ^δ form. fEGarch source never
consulted; validated against **two** committed fixtures — the boundary `fit_fiaparch11_norm_*` (d≈1)
and the interior `fit_fiaparch11_norm_interior_*` (d≈0, on the additive low-persistence series). R
function `fiaparch()` confirmed data-first via `ls`/`args` (`fix_delta=c(NA,1,2)` ⇒ δ estimated).

### 12.1 The δ-power composition — proven to machine precision

WP175 Eq. 2.10 gives FIAPARCH as FIGARCH's **same ARCH(∞) sequence** `{θ_i}` applied to the APARCH
power-asymmetry news:

```
σ^δ_t = ω + Σ_{i=1}^{∞} θ_i (|ε_{t−i}| − γ ε_{t−i})^δ,   ε = r − μ,   θ(B) = 1 − (1−φ₁B)(1−B)^d/(1−β₁B),
```

then `σ = (σ^δ)^{1/δ}`. So it is built **entirely by reuse** — `figarch_coefficients` **unchanged**
(WP175 confirms the identical θ(B); θ₁ = d+φ₁−β₁) and `figarch_variance_filter` with the **news term
swapped** from `ε²` to `(|ε|−γε)^δ`. ω is in σ^δ units, used directly; no η/σ² feedback. **The seam is
machine-exact:** the recursion at each fixture's own *empirical* seed reproduces its σ-series to
`1.1e-16` (boundary) and `2.3e-17` (interior). Param vector `(mu, omega, phi1, beta1, gamma, delta,
d)` — 7, the largest. **Reductions:** δ=2, γ=0 → FIGARCH (`(|ε|)²=ε²`, verified to `~6e-8`, the
seed-convention residual §12.3); d→0 → short-memory APARCH.

### 12.2 The d≈1 boundary is forced (Conrad-Haag) and weakly identified

Non-negativity requires `θ₁ = d+φ₁−β₁ ≥ 0`, i.e. **`d ≥ β₁−φ₁`**. The high-persistence fixture
(β₁=0.913) therefore **forces `d = 0.99999987`** (≥ 0.912); the low-persistence series (α+β=0.60)
lands `d ≈ 5e-7`. At the boundary the parameters are **weakly identified** (a near-flat likelihood
ridge with φ₁≈0): the fit reaches fEGarch's log-likelihood to `0.06/7599` but at a *different* ridge
point (ω off 58%, δ off 9.5%), while d and β₁ recover. Interior d is well-identified — the fit
recovers all seven params (ω to ~4%, the rest to ~1e-3), σ to `3e-5`.

### 12.3 The presample seed — the SECOND irreducible-from-output limit (after APARCH σ₀)

FIGARCH seeds its 50-term presample at the sample mean of *its* news (`Var(r,ddof=1)` = mean of `ε²`).
FIAPARCH **generalizes that rule**: seed at **`mean[(|ε|−γε)^δ]`** (which reduces to FIGARCH's
`Var(ddof=1)` at δ=2,γ=0 up to the μ-vs-r̄/ddof choice). This is **not** fEGarch's exact value. Two
fixtures at d≈0 and d≈1 **disconfirmed every closed form**:

| candidate | interior d≈0 | boundary d≈1 |
|---|---|---|
| `Var^{δ/2}·E[(\|z\|−γz)^δ]` (the FIGARCH-consistent family) | 1.6% off | 44% off |
| `mean(news)` (the chosen convention) | **0.07% off** | 49% off |

The empirical seed is **above the structural floor** (Σθ≤1 ⇒ seed ≥ σ^δ[0]−ω) that every natural
candidate sits below, and it carries a **d-dependent inflation** (factor `1.0 → 1.49` as `d: 0 → 1`)
that no closed form from two points reproduces — the exact value is an **internal fEGarch backcast**,
the **second irreducible-from-output limit in the port** (after the short-memory APARCH σ₀ presample,
§4.2 / line 327). It is handled the same way: `mean(news)` is the principled convention, the **seam is
machine-exact** (proven separately), and only the seed carries a σ-residual that grows from **~1e-6
(interior d) to ~1e-3 (boundary d≈1)** where the long-memory transient decays slowly. This bound is
**asserted honestly** in the tests (`test_seam_is_machine_exact_at_the_empirical_seed` proves the
seam; the fixture-match tests assert the achievable bounded tolerances with comments), **not hidden**
by loose tolerances.

### 12.4 Two synthetic inputs

The identification needed a second `(d, δ, γ)` point away from the boundary, so an **additive**
low-persistence series `synthetic_returns_lowpersist.csv` (GARCH(1,1), α+β=0.60, seed 20240902) was
committed — the original `synthetic_returns.csv` and all its fixtures stay byte-identical. The
interior fit lands `d≈0, δ=2.596, γ=0.032`, well separated from the boundary `d≈1, δ=1.582, γ=0.112`.

---

## 13. FITGARCH(1,d,1) + FIGJR(1,d,1) — the δ-fixed FI variants — RESOLVED (Phase 4)

**Sources.** Zakoian (1994) TGARCH (δ=1); Glosten-Jagannathan-Runkle (1993) GJR (δ=2); Tse (1998) +
WP175 for the FI form. fEGarch source never consulted; validated against `fit_fitgarch11_norm_*` and
`fit_figjr11_norm_*`. R functions confirmed data-first via `ls`/`args`: **`fitgarch()`** and
**`figjrgarch()`** (the GJR function is `figjrgarch`, not `figjr`) — **neither has a `fix_delta` arg**,
so δ is *fixed* (not fitted), unlike FIAPARCH.

### 13.1 FITGARCH / FIGJR = FIAPARCH with δ fixed (thin wrappers)

FITGARCH is FIAPARCH at **δ=1** (the Zakoian σ-recursion) and FIGJR at **δ=2**; both fit the 6-vector
`{mu, omega, phi1, beta1, gamma, d}` (no δ). They reuse `fiaparch_recursion` with δ pinned —
`fitgarch_recursion ≡ fiaparch_recursion(δ=1)` and `figjr_recursion ≡ fiaparch_recursion(δ=2)`
bit-for-bit — and the shared `_fit_fi_power(delta_fixed=…)` QMLE (the Phase-1 `asymmetric.py` pattern,
where GJR/TGARCH/APARCH shared one power recursion). `omega` is a σ^δ intercept (σ-units for FITGARCH,
σ²-units for FIGJR), scaling by `scale^δ`.

### 13.2 The FIGJR kernel — `(|ε|−γε)²`, NOT the Glosten indicator (confirmed)

The reconstruction gate settled the GJR news kernel at the empirical seed:

| kernel | max\|σ−fixture\| |
|---|---|
| **`(|ε|−γε)²`** (APARCH-δ=2) | **4.16e-17** (machine-exact) |
| Glosten `ε²(1+γ·1[ε<0])` | 1.43e-3 |
| Glosten `ε²+γ·min(ε,0)²` | 1.43e-3 |

So fEGarch's FIGJR uses the **APARCH `(|ε|−γε)²` kernel**, carrying the **Phase-1 short-memory GJR
finding** (§4.1) into the FI form — the fixture confirms it, not an assumption. FITGARCH's news is the
same kernel at δ=1, `(|ε|−γε)`, machine-exact at `9.37e-17`.

### 13.3 Fixture confirmation + the boundary difference

Both are **machine-exact at their empirical seed** (FITGARCH `9.37e-17`, FIGJR `4.16e-17` — the seam
proven separately from the seed). Both fits **recover tight**: FITGARCH all params to `≤1e-2` (σ to
`1.3e-6`), FIGJR all six to `≤1.2e-3` (σ to `4.5e-5`). Notably FITGARCH lands `d=1` (β₁=0.921 forces
it, Conrad-Haag §12.2) yet is **well-identified** — the *fixed* δ removes the flat likelihood ridge
that made FIAPARCH's d≈1 boundary weakly identified, so the seed's bounded residual stays small
(`~1e-6`) and the params recover. FIGJR lands `d=0.66` (interior). Both inherit the FIAPARCH
`mean(news)` bounded-limit seed.

### 13.4 Reduction anchors

`fitgarch_recursion`/`figjr_recursion` ≡ `fiaparch_recursion(δ=1/2)` exactly (wrapper correctness).
The **d→0** reduction to short-memory TGARCH / GJR holds *after the pre-sample transient* (`<1e-10`
past `t=200`), under the parameterization map — the FI `phi1` is WP175's `φ₁ = α+β`, so the
short-memory ARCH coefficient is `α = φ₁−β₁` and the SM intercept is `ω(1−β₁)` (the FI ARCH(∞) form
and the SM AR-form are the same process with different pre-sample seeds; `φ₁>β₁` keeps the reduced
coefficients non-negative). **This completes the variance-recursion FI family** (FIGARCH / FIAPARCH /
FITGARCH / FIGJR); only FILog-GARCH (Type-II) remains in Phase 4.

---

## 14. FILog-GARCH(1,d,1) — the Type-II fractional model + a dominated-fixture finding — RESOLVED (Phase 4)

**Sources.** Geweke (1986) / Pantula (1986) / Milhøj (1987) Log-GARCH; Feng et al. (2020a)
FILog-GARCH; **WP171 §2.1 Eqs. 10-13** for the Type-II fractional form. fEGarch source never
consulted; validated against `fit_filoggarch11_norm_*`. R function `filoggarch_spec` confirmed
spec-first via `ls`/`args` (like `loggarch_spec` / `fiegarch_spec`). This is the **last Phase-4
model** — the Type-II counterpart of FIEGARCH.

### 14.1 The Type-II fractional recursion — machine-exact

WP171 Eqs. 10-11 give the truncated MA(∞) log-variance recursion loading the **log-square news**
`ξ_t = ln(η²_t) − E[ln η²]` (Log-GARCH's news, not FIEGARCH's Type-I `g(η)`):

```
ln σ²_t = ωσ + γ(B) ξ_t,   γ(B) = ϕ⁻¹(B)(1−B)^{−d}ψ(B) − 1 = Σ_{i≥1} γ_i B^i,
```

which for (1,d,1) is `γ(B) = (1−ϕ₁B)⁻¹(1−B)^{−d}(1+ψ₁B) − 1`, with `γ_0 = 0` (the `−1` drops the
constant, so the news loads from lag 1) and `γ_1 = d+ϕ₁+ψ₁`. So it is **built by composition**,
reusing FIEGARCH's `theta_coefficients` (geometric `ϕ⁻¹` ⊛ `fracdiff_coeffs(−d)`, fractional
*integration*) convolved with the Type-II MA factor **`(1+ψ₁B)`** — `ψ₁` enters the γ(B) numerator.
It reuses the Log-GARCH `mean_log_sq` moment (norm: `E[ln η²]=−γ_E−ln2=−1.27036`) and the **FIEGARCH
MA(∞) presample** (ωσ intercept direct, ξ-history=0, `σ[0]=exp(ωσ/2)`). Like FIEGARCH the fit couples
(ξ_t depends on σ_t) → O(n²); simulation does not (ξ from drawn η) → one convolution. Param vector
`(mu, omega_sig, phi1, psi1, d)`. **The recursion at fEGarch's own params reproduces the fixture σ to
`1.05e-15`** — the seam is machine-exact. The **d→0 reduction** collapses γ_i to short-memory
Log-GARCH's `(ψ₁+ϕ₁)ϕ₁^{i-1}` (~1e-12).

### 14.2 The ridge relaxes — d absorbs the persistence

Short-memory Log-GARCH sits on a near-common-root ridge (`ϕ₁≈−ψ₁`, `ϕ₁=0.989, ψ₁=−0.954`, weakly
identified). FILog-GARCH fits `ϕ₁=0.306, ψ₁=−0.556` (separation `|ϕ₁+ψ₁|=0.25`, well apart) with
`d=0.289` **interior** (weakly-stationary regime) — the fractional d **absorbs the persistence**
(ϕ₁ dropped from ~0.99, the FIEGARCH pattern), so all coefficients are identified and recover tight
(no ridge tolerance, no boundary handling).

### 14.3 An effective-challenge finding — fEGarch's fixture is a strictly-dominated local optimum

The fixture reports `mu=−0.003054, loglik=7435.98` — a **strictly-dominated local optimum**. A
multi-start check settles the mechanism: an optimizer *placed* at `mu=−0.003` stays there (loglik
7435.9, ≈ the fixture — so it is a genuine local optimum, not a stopped-mid-climb point), **but no
data-driven start reaches it**. The data mean → 7555.85 (`+119.87`), three random starts →
7553.3–7559.2 (`+117` to `+123`), and **even a start from fEGarch's own reported params escapes** to
7536.8 (`+101`). So every sensible start lands in a basin `~+120` higher (`mu` near the data mean,
`ϕ₁≈0.39, ψ₁≈−0.63, d≈0.29`, separated). Since `filoggarch_recursion` at fEGarch's params reproduces
the fixture σ to `~1e-15`, it is fEGarch's **optimizer** landing in a poor local optimum, not the
model. (At the dominant optimum the loglik ~7556 is only ~40 below the family — mild genuine
Type-II-vs-GARCH-data misfit; the ~120 gap to the fixture is the local-optimum artifact.)

**Validated honestly in two parts** (`test_filoggarch.py`, 11 tests): (1) the **seam is machine-exact
at fEGarch's params** (`test_seam_is_machine_exact_at_fegarch_params`, ~1e-15 — the model is correct);
(2) the **fit strictly dominates the fixture from every sensible start**
(`test_fit_strictly_dominates_the_fegarch_local_optimum` — the data-mean fit plus two random starts
plus a start from fEGarch's own params all beat the fixture by >100 loglik; μ near the data mean, d
interior, coefficients separated). We do **not** assert a param-by-param match to the dominated
fixture — that would validate a bad optimum. This is the port's clearest **effective-challenge
result**: an independent reimplementation catching the reference package's optimizer landing in a
strictly-dominated local optimum. Additional checks: the `σ[0]=exp(ωσ/2)` presample tell,
γ composition (`γ_0=0`, `γ_1=d+ϕ₁+ψ₁`), exact scale-equivariance, known-truth recovery incl. d, and
the non-norm deferral (`mean_log_sq` is norm-only, as MLog-GARCH).

**Phase 4 is complete — all eight fractionally-integrated models** (the Type-I EGF family FIEGARCH /
FIMEGARCH / FIMLog-GARCH; the variance-recursion family FIGARCH / FIAPARCH / FITGARCH / FIGJR; and the
Type-II FILog-GARCH) are implemented and validated.

---

## 15. The joint shape-parameter QMLE path — GARCH×std, and a weakly-identified `nu` — RESOLVED (Phase 5)

**Sources.** Bollerslev (1987) for the standardized Student-t; the shape-parameter joint-estimation
convention is fEGarch's own default (shape/skew parameters optimized *jointly* with the model
parameters, not profiled). fEGarch source never consulted; validated against `fit_garch11_std_*`.
This is the **first distribution-breadth** work — the same GARCH(1,1) recursion under a non-normal
conditional law, exercising the shape parameter carried in the fitted vector.

### 15.1 The mechanism — the engine already carries shape parameters

The Phase-0 QMLE engine (`quasi_max_likelihood`) already appends a distribution's shape parameters to
the fitted vector: `start = [*var_start, *dist.param_start]`, the negative-log-likelihood splits
`theta[:n_var]` (variance recursion) from `theta[n_var:]` (fed to `dist.logpdf`), and the reported
`param_names = (*var_names, *dist.param_names)`. So GARCH×std needs **no engine change** — `fit_garch`
just routes `cond_dist="std"` through, and the fitted vector becomes `(mu, omega, alpha, beta, nu)`.
The standardized Student-t density is `z = T·√((ν−2)/ν)` with `ν > 2` for finite unit variance; our
layer names the parameter `nu`, the fEGarch fixture reports it as `df` (a naming reconcile, no
behavioural difference). **The seam is machine-exact**: at fEGarch's reported params (incl.
`df = 341.89`) the GARCH σ series reproduces to `~1e-17` (σ does not depend on `nu`) and the
Student-t log-likelihood to `~1e-10`.

### 15.2 The optimizer seam — a flat shape ridge defeats L-BFGS-B

On the **Gaussian** synthetic returns the Student-t MLE is `ν → ∞` (the normal limit): the
log-likelihood rises **monotonically** in `nu` toward the GARCH×norm supremum `7601.11` and is
*unreachable* at any finite `nu` (it is the `ν=∞` limit). The ridge is near-flat at large `nu`
(`ll(100)=7600.04`, `ll(342)=7600.83`, `ll(5000)=7601.09`, `ll(∞)=7601.1096`), and L-BFGS-B's
projected-gradient step **stalls** there — it stops barely past its start (loglik ~7600.3, *below*
the fixture). The fix: **shape-parameter fits switch to derivative-free Nelder-Mead**, which climbs
the flat ridge to the `nu` bound (`1e6`) and reaches `7601.10996` — within `~1e-4` of the supremum.
The `nu` upper bound was raised to `1e6` (from a low cap) so this limit is reachable. The **norm path
keeps L-BFGS-B** (no shape parameter) and is therefore **bit-identical** to the validated Phase-1
fixture (`< 1e-8`).

### 15.3 An effective-challenge finding — the std fixture is strictly dominated (same pattern as §14)

fEGarch's `df = 341.89` fixture has loglik `7600.83` — **0.28 below** the GARCH×norm optimum
`7601.11`, since std **nests** norm as `ν → ∞`. It is a strictly-dominated point on the flat `nu`
ridge (fEGarch's optimizer stopping short of the `ν=∞` limit), exactly the honest pattern of the
FILog-GARCH local optimum (§14). Our fit **dominates** it: loglik `7601.10996 ≥ 7600.83` and within
`~1e-3` of the normal supremum. So the validation is **identification-structured**:

* **well-identified pieces tight** — `mu` and `omega/alpha/beta` recover to `< 1%` (`mu 0.09%`,
  `omega 0.66%`, `alpha 0.08%, beta 0.02%`);
* **`nu` by regime only** — asserted `> 100` (large), **never** param-exact against the dominated
  `df = 341.89`; its standard error is (correctly) non-finite — the likelihood is flat in `nu` — so
  only the well-identified SEs are asserted finite;
* **σ at the ridge level** — the two fits sit at different `nu` (`1e6` vs `341.89`), so their variance
  parameters and thus σ differ at the `~1e-3` relative level (`max|dev| ~1.5e-5`); the machine-exact
  σ agreement is the seam test at *identical* params, not the two-fit comparison.

We do **not** assert `loglik ≥ 7601.11` literally — that is the `ν=∞` supremum, unreachable at finite
`nu`; the honest assertion is `≥` the dominated fixture *and* within `~1e-2` of the supremum.

### 15.4 Reduction anchor + known-truth

**Reduction**: `std.logpdf(z, ν=1e6)` collapses to `norm.logpdf(z)` (`< 1e-3`), so GARCH×std recovers
GARCH×norm. **Known-truth**: simulating GARCH×std with a genuine `df = 6` (identified heavy tails),
QMLE recovers `nu = 6.01` (SE `0.36`, `|dev|/SE = 0.03`) — the shape parameter is sharply identified
when the data actually has it, in deliberate contrast to the near-normal fixture where it flies to the
bound. Validated in `test_garch_std.py` (6 tests).

## 16. GARCH×ged (sharply-identified shape) + GARCH×ald (the P-profiling fork) — RESOLVED (Phase 5)

**Sources.** Nelson (1991) GED standardization; the scaled-average-Laplace equations WP 2026-04 App.
C.1 Eqs. 31–33, 37 (already the basis of the Phase-0 `AverageLaplace`); the Prange P-profiling is
fEGarch's own convention. fEGarch source never consulted; validated against `fit_garch11_ged_*` and
`fit_garch11_ald_*`. These are the two structurally-different follow-ups to GARCH×std (§15): ged
reuses the *continuous* shape path, ald introduces the *discrete integer-grid* path.

### 16.1 GARCH×ged — the same path as std, but sharply identified (the contrast case)

GED reuses the §15 machinery verbatim (shape param in the fitted vector, Nelder-Mead), and needs **no
code change** — `fit_garch(cond_dist="ged")` already routes through. But the *outcome is the opposite
regime*: on the synthetic returns the GED shape has a genuine interior peak (`shape = 2.15`,
`SE ≈ 0.10`), so it is **sharply identified**, and the whole fit — the shape included — reproduces the
fixture to machine order (`shape` rel `2.8e-7`, loglik dev `5e-10`, σ dev `3.5e-8`), exactly like the
norm fit. Our layer names the shape `nu`; the fixture names it `shape` (same `gennorm` β; `shape = 2`
the normal). **No dominated-reference issue here** (unlike std): GED nests norm at `shape = 2`, and
the fitted `shape = 2.15` is a *real* finite-sample improvement, so the fixture loglik `7602.31` sits
legitimately **above** norm `7601.11` — nothing to challenge, just confirm we reach it.
**Reduction**: `ged.logpdf(z, shape=2)` collapses to `norm.logpdf(z)` (`3.5e-15`, machine).
**Known-truth**: simulated `shape = 1.2` (fat-tailed) recovers `nu = 1.21` (`|dev|/SE = 0.48`).
`test_garch_ged.py` (5 tests). So std and ged bracket the two identification regimes of the *same*
continuous shape path: flat ridge (std, `nu` by regime) vs sharp peak (ged, `nu` param-exact).

### 16.2 GARCH×ald — the discrete P-profiling fork (the new mechanism)

The ALD's degree `P` is **not** a QMLE parameter: `AverageLaplace.param_names = ()`, and `P` is a
construction attribute (`AverageLaplace(p=…)`). fEGarch *profiles* it over the integer grid
`Prange = c(1, 5)`. So `fit_garch(cond_dist="ald")` takes a **new outer-grid-search branch**
(`_fit_garch_ald`): for each `P ∈ {1,2,3,4,5}` fit the four continuous params `(mu, omega, alpha,
beta)` at that fixed `P` (a fixed-shape ALD → gradient-based L-BFGS-B, no flat ridge), then select the
best log-likelihood. `P` never enters the continuous optimizer. The `(P, loglik)` grid is surfaced on
the result (`GarchFit.profile`). **AIC/BIC count `P` in the penalty (`k = 5`)** even though it is
profiled, not optimized — confirmed against the fixture (`k = 5` reproduces `aic = −6.06632`, `k = 4`
does not).

**The profile + selection reproduce the fixture** (empirical confirmation of the profiling
convention, since the source is never read): the loglik rises monotonically `P1 7547.11 → P2 7569.14
→ P3 7579.00 → P4 7584.47 → P5 7587.90`, peaking at the **boundary `P = 5`** — exactly the fixture's
selection and its `P=5` loglik. **Honest misfit (documented, not a defect):** even at its thinnest
tail (`P = 5`, raw kurtosis `3 + 3/(P+1) = 3.5`) the ALD is `~13` loglik **below** norm — the ALD is a
fat-tailed family and Gaussian data has no excess kurtosis, so it cannot match the normal, and that is
*why* the profile pins at the boundary rather than an interior optimum. **Seam machine-exact** at the
fixture's `P = 5` (σ `7e-18`, loglik `0`). **Known-truth**: simulating GARCH×ALD at an *in-grid*
`P = 2` (drawing ALD(2) innovations directly, since `garch_sim` exposes only the default-P ALD) gives
an **interior** profile max, so the grid search recovers `P = 2` and the continuous params (`beta`
within `0.3%`) — the in-grid contrast to the boundary pin on Gaussian data. `test_garch_ald.py`
(7 tests). This is the port's first **discrete/profiled** parameter — the structural counterpoint to
the continuous shape params of std/ged.

## 17. GARCH×{snorm,sstd,sged,sald} — the Fernández–Steel `skew` param, completing the set — RESOLVED (Phase 5)

**Sources.** Fernández–Steel (1998) skew split; the mean-0/variance-1 re-standardization WP 2026-04
App. C.1 Eqs. 38–41 (already the basis of the Phase-0/2 `FernandezSteelSkew` wrapper). fEGarch source
never consulted; validated against `fit_garch11_{snorm,sstd,sged,sald}_*`. This **completes the GARCH
distribution set (all 8)**.

### 17.1 The `skew` param + the `xi`↔`skew` reconcile

The FS wrapper adds **one** parameter — the skew `s` — on top of a symmetric base, splitting the
density `h(x) = 2/(s+1/s)·f(x·s^{−sign x})` and re-standardizing to mean 0 / var 1, so `s = 1`
recovers the base. The joint QMLE vector carries `skew` alongside the base's own shape params:
snorm `{…, skew}`, sstd `{…, nu, skew}`, sged `{…, nu, skew}`, sald `{…, skew}` (+ profiled P). **The
`xi`↔`skew` reconcile:** the wrapper's internal math variable is `ξ` and `s = ξ` **applied directly**
(the short-memory Phase-2 finding, `s<1` left / `s>1` right / `s=1` symmetric), so the public
parameter is **renamed `xi`→`skew`** at the boundary (`get_distribution("snorm").param_names ==
("skew",)`) to match fEGarch's argument and the fixtures. The rename is name-only (the equations keep
`ξ`); nothing depended on the old public name (`test_distributions.py` uses positional tuples).
Bounds `(0.1, 10.0)`, start `1.0` (symmetric).

### 17.2 Mechanism reuse — snorm/sstd/sged for free, sald the compound case

snorm/sstd/sged need **no new code**: `param_names` is non-empty, so `fit_garch` already routes them
through the joint Nelder-Mead shape path (§15). **sald is the compound case** — the FS skew is a
continuous inner param but `P` is still the profiled integer grid — so `_fit_garch_ald` was
generalized: for each `P ∈ {1..5}` it builds `FernandezSteelSkew(AverageLaplace(p=P))` and fits
`{mu,omega,alpha,beta,skew}` jointly (Nelder-Mead, flat skew ridge), then selects the best `P`. The
**AIC/BIC penalty counts both the profiled P and the skew** (`k = 6`; snorm `k = 5`) — confirmed
against every fixture's `aic` to `~1e-13`.

### 17.3 Identification — three sharp, one dominated (the sstd inheritance)

On the symmetric Gaussian returns the **skew is nonetheless identified** (a finite sample carries a
detectable asymmetry signal), so **snorm / sged / sald reproduce their fixtures — skew included — to
`~1e-6`** (like ged), and each sits legitimately **at or above its symmetric base** (snorm `7601.21 >`
norm `7601.11`; sged `7602.35 >` ged `7602.31`; sald `7588.13 >` ald `7587.90`). Each skew is `< 1`
(the fixtures' left-skew). **The one dominated case is `sstd`**, which inherits Student-t's flat `nu`
ridge (MLE `ν→∞`): its fixture (`df = 340.8, loglik 7600.93`) sits **below** norm `7601.11`, so it is
strictly dominated; our Nelder-Mead fit climbs `ν` to the bound and reaches **snorm's** optimum
`7601.21`, **dominating the fixture by `+0.28`** (the exact std pattern of §15). So sstd is validated
like std — seam machine-exact at the fixture's own params (proving the joint `df+skew` likelihood is
right), `nu` by regime, `skew` still identified (within `~1e-3` of the fixture), fit `≥` the dominated
reference. The skew is the *identified* direction even in sstd; only `nu` is the flat ridge.

### 17.4 Reduction anchor + known-truth

**Reduction (FS correctness proof):** each skewed density at `skew = 1` equals its symmetric base to
**floating-point zero** — snorm@1→norm, sstd@1→std, sged@1→ged, sald@1→ald all `0.0e+00`.
**Known-truth:** simulating GARCH×sstd at a genuine `skew = 0.85` (`df = 6`) recovers `skew = 0.847`
(`|dev|/SE = 0.32`) and `nu = 5.8` (`|dev|/SE = 0.66`) — the positive control that the skew estimation
pins down a real asymmetry when present, in contrast to the near-symmetric fixture. `test_garch_skewed.py`
(11 tests). **The GARCH distribution set is complete — all eight conditional laws fit and validated.**

## 18. GJR-GARCH / TGARCH / APARCH under all 8 distributions — the compressed inheritance build — RESOLVED (Phase 5)

**Sources.** Glosten–Jagannathan–Runkle (1993) GJR, Zakoian (1994) TGARCH, Ding–Granger–Engle (1993)
APARCH (already the Phase-1 recursions); the distribution machinery is the proven GARCH×8 stack. No
new spec extraction — this is a **composition** of two already-validated layers. fEGarch source never
consulted; validated against `fit_{gjrgarch,tgarch,aparch}11_*`.

### 18.1 The reconstruction gate — the go/no-go, and the presample-seed subtlety

Because the three recursions are proven on norm (§4) and the eight distribution likelihoods on
GARCH×8 (§15–17), the compressed build's correctness reduces to a **reconstruction gate**: at each of
the 21 fixtures' own params, does recursion σ + distribution log-likelihood reproduce the fixture? A
*naive* full-series reconstruction gives only `~1e-7` σ / `~1e-5` loglik — but this is **not** a
distribution interaction: it is identical for **norm** (I verified), and is the asymmetric family's
**known presample-seed** discrepancy (§4: the news-impact `kernel_0` seed differs from fEGarch's by
`~1e-7`, unlike GARCH's exact `Var(r)` seed). Seeding `σ[0]` from the fixture (isolating the recursion
*form*, exactly as the Phase-1 norm reconstruction test does) makes the gate **machine-exact for all
21**: `σ[1:]` worst `1.73e-17`, loglik worst `4.39e-10`. So the composition is clean — no unexpected
model×distribution interaction — and the `~1e-7` is the same presample floor the norm fixtures already
tolerate at `<1e-5`.

### 18.2 Machinery reuse — one shared upgrade

`_fit_aparch_family` inherited the GARCH shape/skew/P-profiling **wholesale**: the same Nelder-Mead
branch for near-flat shape/skew ridges, the same `_ALD_PRANGE` P-profiling fork (generalized to
`_fit_aparch_ald`, reused for `ald` and the compound `sald`), and the same `GarchFit.profile`. The one
family-specific piece is the **δ-dependent unscaling** (`omega ~ scale^δ`, extracted into
`_build_asym_fit`), because APARCH's `δ` is jointly estimated. GJR/TGARCH add `gamma1` to the base;
APARCH adds `gamma1 + delta` — the **7-param** `{mu,omega,phi1,beta1,gamma1,delta,df}` (×std) and
**8-param** `{…,delta,df,skew}` (×sstd) compounds, the largest short-memory vectors in the port.

### 18.3 Identification-structured validation (identical pattern to GARCH×8)

* **Identified** (ged/ald/snorm/sged/sald): reproduce loglik (`~1e-5`, within `<1e-4`), σ (`~2e-7`,
  the presample floor), AIC/BIC (`~1e-8`), the asymmetry `gamma1` and shape/skew (sharply identified,
  `~1e-6`). **δ (APARCH)** recovers tight and **interior** (`2.24–2.47`, co-estimated with the shape,
  never pinned at a δ∈{1,2} seed). **P (ald/sald)** profiles to the boundary and selects `5`, every
  model.
* **Dominated std / sstd**: inherit Student-t's flat `df` ridge, so each fixture sits **below** its
  symmetric-tailed sibling (gjr×std `7602.75` < gjr×norm `7603.04`; sstd < snorm), strictly dominated.
  Our Nelder-Mead fit climbs `df` to the bound and **reaches the sibling's optimum**, dominating the
  fixture by `+0.28` (the exact std pattern). `df` validated by regime, `skew` still identified.
* **Reductions**: skew→1 recovers the symmetric base, composed with each recursion at a common σ path
  (`<1e-9`, per model×dist).
* **Known-truth — the 8-param positive control**: simulating APARCH×sstd with an identified
  `δ = 1.6`, fat tails `df = 6` and left-skew `0.85` recovers **all** of `δ` (`|dev|/SE = 0.12–0.39`),
  `df` (`1.28–1.45`) and `skew` (`0.19–0.35`) — proving the full 8-way joint estimation works when the
  data exercises every parameter, in deliberate contrast to the near-symmetric Gaussian fixture.

`test_asymmetric_distributions.py` (73 tests). **This completes the short-memory variance-recursion
family (GARCH / GJR / TGARCH / APARCH) under all eight conditional distributions.**

## 19. FIGARCH / FIAPARCH / FITGARCH / FIGJR under all 8 distributions + the tail-persistence d-shift — RESOLVED (Phase 5)

**Sources.** Baillie–Bollerslev–Mikkelsen 1996 / Conrad–Haag 2006 FIGARCH; Tse 1998 FIAPARCH; the
Phase-4 recursions (§11–13); the distribution machinery proven on GARCH×8. A composition of two
proven layers — **no spec extraction** (§12). fEGarch source never consulted; validated against
`fit_{figarch,fiaparch,fitgarch,figjr}11_*`. This completes the **FI variance-recursion family** under
all eight laws.

### 19.1 The reconstruction gate — machine-exact for all 28

The go/no-go: FIGARCH's presample (50 terms, `Var` ddof=1) matches fEGarch exactly, so its recursion
reconstructs σ directly (~1e-16); the δ-power family (FIAPARCH/FITGARCH/FIGJR) carries the documented
presample-seed offset (§12), so σ[0] is backed out of the fixture (isolating the recursion *form*, as
the norm FI tests do). Seeded that way, **all 28 model×dist compose machine-exactly** — σ[1:] worst
`1.4e-16`, loglik worst `1.4e-10`. The recursion + fractional operator + distribution likelihood
compose cleanly; no unexpected interaction. GO.

### 19.2 Machinery reuse

`fit_figarch` (extracted `_figarch_fit_from_result` + `_fit_figarch_ald`) and `_fit_fi_power`
(extracted `_build_fi_power_fit` + `_fit_fi_power_ald`) inherited the GARCH shape/skew Nelder-Mead
branch and the `_ALD_PRANGE` P-profiling wholesale; only the model-specific unscaling differs (FIGARCH
`omega ~ scale^2`; the δ-power family `omega ~ scale^delta`). FIGARCH adds `d`; FIAPARCH adds
`gamma + delta + d` (the **9-param** `×sstd` compound); FITGARCH/FIGJR add `gamma + d` (δ fixed 1/2).

### 19.3 The tail-absorbs-persistence d-shift (the headline finding)

**The distribution shifts the fractional order `d`.** The Conrad–Haag non-negativity floor
`d ≥ β₁ − φ₁` forces `d → 1` when `β₁` is high, so FIAPARCH/FITGARCH pin `d = 1.000` under
norm/ged/snorm/sged/ald/sald. But a **heavy-tailed law (std/sstd) lets `β₁` drop**, lowering the floor,
so `d` lands **interior**: FIAPARCH std/sstd `d = 0.72/0.71` (vs 1.0), FITGARCH std/sstd `0.91/0.64`,
FIGARCH sstd `0.54` (vs norm 0.68), FIGJR std `0.53`. So `d` is boundary-identified under some
distributions and interior under others *for the same model* — the tolerance is structured **per
fixture**, not per model. `delta` (FIAPARCH) settles at **~1.5–1.7**, below short-memory APARCH's
~2.4: with a fractional `d` now carrying the persistence, the power `δ` no longer has to.

### 19.4 Seam-centric validation — the fit is comparable-or-better (effective challenge)

The `d`-boundary is a **near-flat ridge**, so both fEGarch and our optimizer land at different local
optima. For **std/sstd our fit strictly dominates every fixture** — the heavy tail pushes `df → ∞`
*and* `d →` boundary, a higher optimum than fEGarch's interior-`d`, finite-`df` stop (FITGARCH×sstd
`+3.83` loglik; FIAPARCH×std `+1.56`). A few `ald`/`sald` boundary cases land slightly lower (our
optimizer's own weak-identification, worst `−0.31`). So the **seam is the correctness proof**; the fit
is validated as *reaches-or-beats* the fixture (std/sstd domination asserted; tight fixture-match only
for the well-identified interior-`d` cases: FIGARCH/FIGJR × ged). This is the effective-challenge
pattern of §14, now across a whole family: an independent reimplementation routinely finding higher
optima than the reference on a flat ridge.

**Reductions**: skew→1 recovers the symmetric base, composed with each FI recursion (`<1e-9`).
**The 9-param known-truth** (`test_fi_distributions.py`, the widest joint fit in the port): simulating
FIAPARCH×sstd with an identified `δ = 1.5`, interior `d = 0.35`, fat tails `df = 6` and skew `0.85`
recovers **all four** substitutable parameters — `δ` (`|dev|/SE ≤ 1.4`), `d` (`≤ 1.5`), `df` (`≤ 0.7`),
`skew` (`≤ 0.7`). If four partially-substitutable parameters co-recover on data that separates them,
the joint machinery is sound. `test_fi_distributions.py` (83 tests). **This completes the FI
variance-recursion family under all eight conditional distributions.**

## 20. The EGF distribution infrastructure — skew-wrapper moments + per-iteration centering, on EGARCH — RESOLVED (Phase 5)

**Sources.** Fernández–Steel (1998) skew density (Eqs. 38–41, already in the distribution layer);
Nelson (1991) EGARCH `g(η)`; the base-moment definitions. fEGarch source never consulted; validated
against the five converging `fit_egarch11_*` fixtures + known-truth for std/sstd. **This is the EGF
family's shared distribution support** — every Type-I EGF model (EGARCH/MEGARCH/MLog-GARCH + their FI
variants) inherits it; it is *proven on EGARCH first*.

Unlike the variance-recursion families, EGF distribution breadth is **new mechanism, not inheritance**,
because `g(η)`'s centering `E[g(η)]` is distribution-dependent (`E|η|` for EGARCH/MEGARCH,
`E[ln(|η|+1)]` for MLog-GARCH). Two pieces:

### 20.1 The skew-wrapper moments — quadrature over `f_skew`

The symmetric bases expose `abs_moment` (closed-form) but the log-moments (`mean_log_sq`,
`mean_log_modulus`) only on `norm`; the **FS-skew wrapper exposed none**. All are now implemented by
**adaptive quadrature over the standardized density** `E[G(z)] = ∫ G(z) f(z) dz`, split at 0 for the
`ln z²` endpoint singularity — the same path `norm.mean_log_modulus` already used, filled uniformly
for `std/ged/ald` (log-moments) and for the FS wrapper (all three). **Correctness anchor: at
`skew = 1` every skewed moment equals the base moment** (`snorm→norm`, `sged→ged`, `sald→ald`, for
`abs_moment`/`mean_log_sq`/`mean_log_modulus`) to `< 1e-9` (quad-exact). E\|η\| for the skewed law has
no elementary closed form (the mean-shift `μ_FS` sits inside the `|·|`), so quadrature is the uniform
choice; the log-moments need it regardless.

### 20.2 The per-iteration centering — a QMLE hook

For a jointly-estimated continuous shape, `E[g(η)]` is a **strong function of the shape** (E\|η\|(df):
`df=3→0.637, 6→0.750, 100→0.796, ∞→0.798`), so the centering must be **re-computed each optimizer
iteration** — not the precomputed constant the norm path uses. The engine gained a
`recursion_uses_dist_params` hook: when set, `quasi_max_likelihood` calls the recursion
`variance_recursion(var_params, returns, dist_params)`, so `_fit_type1` re-evaluates the magnitude
moment from the *current* shape each call. Shape fits use **Nelder-Mead + a restart** (the flat shape
ridge collapses the simplex — `snorm` first landed 1.10 loglik below its fixture, and a single restart
recovered it to `0.000`); the ALD profiles `P` over the grid; **norm keeps the precomputed constant +
L-BFGS-B, bit-identical** (egarch/megarch/mloggarch norm fixtures unchanged).

### 20.3 Validation — 5 fixtures + the κ/γ-shift confirmation + a robustness finding

**Reconstruction seam machine-exact** at each fixture's params (ged/ald closed-form-moment `~1e-15`,
skewed quadrature-moment `~1e-13/1e-12`). **All five converging fixtures match** (loglik `3e-10`–`2e-5`,
σ `~1e-7`); `P = 5` for ald/sald. The **κ/γ shift is explained by E\|η\|**: ald/sald have the E\|η\| that
deviates most from norm (`0.781` vs `0.798`, −2.1%) and show the largest κ/γ re-fit (κ −6%, γ +1%),
while ged/snorm/sged have E\|η\|≈norm and barely move κ/γ — the centering is confirmed against the
fixtures.

**std/sstd — the robustness finding.** fEGarch's own optimizer **failed** to fit egarch×std/sstd
("Error during optimization") — the df-dependent E\|η\| centering + the near-unit-root EGF ARMA is the
hardest EGF case, so *no fixture exists* and we do not fabricate one. Validated by **known-truth**
instead: simulating egarch×std at `df=6` recovers `df` (`|dev|/SE ≤ 1.2`) and **converges**; egarch×sstd
at `df=6, skew=0.85` recovers both. **Our Nelder-Mead + restart + stable closed-form E\|η\|(df) converges
where the reference's optimizer failed** — an independent reimplementation fitting the reference's
hardest EGF case. `test_egarch_distributions.py` (25 tests).

> **Framing correction (see §21).** EGARCH exercised only `abs_moment` (its `g_asy = η` has
> `E[η] = 0`). The **full EGF centering-moment set is four**: `abs_moment` (E\|η\|), `mean_log_sq`
> (E[ln η²]), `mean_log_modulus` (E[ln(\|η\|+1)]) and `mean_signed_log_modulus`
> (E[sgn(η)·ln(\|η\|+1)], the modulus-log *asymmetry* centering). The 4th is skew-sensitive and only
> the MEGARCH/MLog-GARCH build (§21) exercised it; the EGARCH step above proved the first, not all four.

## 21. The 4th EGF moment + MEGARCH/MLog-GARCH/Log-GARCH under all 8 distributions — RESOLVED (Phase 5)

**Sources.** Fernández–Steel skew density; the John–Draper (1980) modulus-log transform; the EGF
constant-sets (§20). fEGarch source never consulted; validated against the 16 converging
`fit_{megarch,mloggarch,loggarch}11_*` fixtures + known-truth for the 5 failures. This **completes the
short-memory EGF family** under all eight distributions.

### 21.1 The 4th EGF moment — the modulus-log asymmetry centering (the gate caught it)

The reconstruction gate on the fixtures **failed** for megarch/mloggarch under *skew* (`~1e-6`) while
symmetric + Log-GARCH + all of EGARCH were exact — exactly the "skewed-moment issue the EGARCH proof
didn't cover" the STOP condition named. Diagnosis: MEGARCH/MLog-GARCH have a **modulus-log asymmetry**
`g_asy = ζ(η) = sgn(η)·ln(|η|+1)` (`M_asy = 1, p_asy = 0`), whose centering `E[ζ(η)]` is an
**odd-function expectation** — identically **0 for every symmetric base**, but **nonzero under skew**
(`~0.001–0.002`). EGARCH's `g_asy = η` uses `E[η] = 0` (skew included); Log-GARCH's Type-II news is the
even `ln η²` (no odd term). So only MEGARCH/MLog-GARCH need it.

Fix: a **4th distribution moment** `mean_signed_log_modulus = E[sgn(z)·ln(|z|+1)]` — the base returns
`0.0` (odd integrand, even density), the FS-skew wrapper computes it by the same quadrature over
`f_skew`. **Odd-function anchor:** it vanishes at `skew = 1` (`< 1e-9`) and is exactly `0` for
norm/std/ged/ald. Wiring it as the `mean_asy` for the two modulus-log-asymmetry models (per-iteration
for continuous shape, in both fit *and* sim) makes the 6 skewed cases **machine-exact (~4e-17)**. The
full EGF centering-moment set is now **four**.

### 21.2 The compressed build — 16 fixtures + the γ tell + Log-GARCH's ridge

Reconstruction gate machine-exact for **all 16** (megarch/mloggarch skewed `~4e-17`, loggarch `~1e-12`).
**MEGARCH/MLog-GARCH are well identified** and match tightly (loglik `~1e-10`, σ `~1e-8`); **the γ tell
carries under distributions** — MLog-GARCH's `E[ln(|η|+1)]` magnitude gives `γ ≈ 0.28`, MEGARCH's `E|η|`
gives `≈ 0.16`, a **~1.80× ratio held across every law** (norm→sald). `P = 5` for all ald/sald. norm
paths bit-identical.

**Log-GARCH sits on its near-common-root `φ₁ ≈ −ψ₁` ridge**, weakly identified: with the (now
per-iteration) `mean_log_sq` centering our fit **reaches-or-beats** each fixture (ged `−7e-4`; ald `+1.3`,
sstd `+4.9` — strictly dominating), validated by the machine-exact seam, not a param match (the §14/§19
effective-challenge pattern, now under distributions).

### 21.3 The 5 fEGarch-optimizer-failure cases — robustness extended

megarch/mloggarch std/sstd and loggarch std failed fEGarch's optimizer (continuous-df fragility); *no
fixture exists*, none fabricated. Known-truth at `df = 6` (+`skew = 0.85` for sstd, which exercises the
per-iteration `mean_asy` recompute in sim **and** fit): **our optimizer converges on all 5** and recovers
`df` (+`skew`). The **loggarch std-fails/sstd-converges** split in fEGarch confirms these are flat-ridge
*starting-point* solver failures (non-deterministic), not model failures — which is why our
Nelder-Mead + restart is more robust to them. `test_egf_family_distributions.py` (46 tests). **The
short-memory EGF family (EGARCH/MEGARCH/MLog-GARCH/Log-GARCH) is complete under all eight distributions.**

---

*Add further specification derivations here as later phases (the dual mean, forecasting/risk tie-back)
are implemented — always from the papers/manual, never the source.*
