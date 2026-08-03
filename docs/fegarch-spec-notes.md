# fEGarch clean-room — specification notes

> Working notes recording the **mathematical specifications** (from the cited papers and the
> `fEGarch` reference manual) that the clean-room port is implemented against, and the open
> convention questions to be resolved against the committed `fEGarch` **output fixtures**
> (`tests/fixtures/fegarch/`). This file is a *permitted* clean-room input: it records published
> mathematics and our own derivations only — **never** anything read from the `fEGarch` source
> (CLAUDE.md §12). Each entry that is not yet pinned carries a **RECONCILE** marker.

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

with `μ`, `σ` as above (this is the Lambert & Laurent 2001 standardization of the FS skew). **This
standardization is the step most likely to differ from `fEGarch` in a subtle way** — e.g. a
different `M_1` convention, standardizing to a different target, or applying the mean/scale removal
in a different order. It is the prime suspect if the skewed-distribution fixtures fail to match.

### 1.4 Consistency with the current `quantica` implementation

`quantica.timeseries.fegarch.distributions.FernandezSteelSkew` already implements exactly §1.3:
`μ = M_1·(ξ − 1/ξ)` and `σ² = (1 − M_1²)(ξ² + ξ⁻²) + 2M_1² − 1`, with `M_1 = E|z_base|` the base's
first absolute moment, and standardizes `z = (ε − μ)/σ`. Substituting `M_2 = 1` into the paper's
moment formula reproduces this `σ²` term-for-term (see §1.2), so the *derivation* is confirmed
against the paper. What remains open is the **argument convention** (§1.5).

### 1.5 Open convention question — RECONCILE

**Does `fEGarch`'s `skew` argument equal `γ` directly, or a transform of it?** Some GARCH packages
pass `γ` itself; others pass a re-parameterized skew (e.g. a `λ ∈ (−1, 1)`, or `log γ`, or an
inverse convention where `skew < 1` means *right* skew). Our `FernandezSteelSkew` currently treats
its `xi` parameter as `γ` directly.

- **Resolution path:** the committed distribution fixtures
  (`tests/fixtures/fegarch/distribution_quantiles.csv`, `distribution_moments.csv`) were generated
  from `fEGarch`'s public samplers `rsnorm_s(n, skew=…)` etc. Matching our analytic quantiles /
  skewness at a given `xi` against `fEGarch`'s empirical values at the same `skew` value (within
  Monte-Carlo tolerance) pins the mapping.
- **Preliminary indication from the fixtures (not yet a formal test):** `fEGarch`'s `skew = 0.8`
  gives negative empirical skewness (≈ −0.34) and `skew = 1.3` gives positive (≈ +0.39). That
  orientation — `skew < 1` ⇒ left-skew, `skew > 1` ⇒ right-skew — is consistent with **`skew = γ`
  directly** under §1.1. Whether the *magnitude* matches exactly (ruling out a magnitude-preserving
  transform) is confirmed once the skipped `test_fernandez_steel_constants_match_fegarch_fixture`
  stub is wired to the fixtures. **RECONCILE: confirm `skew = γ` (and the exact standardization)
  against the fixtures before relying on the skewed variants.**

---

## 2. Average-Laplace (`ald`) form — RECONCILE (open)

Separate from the skew convention, the exact **average-Laplace** base density is still unpinned. The
`fEGarch` sampler `rald_s(n, P = …)` takes a parameter `P` (default 8); the committed fixtures show
that at `P = 8` the distribution is **near-normal** (raw kurtosis ≈ 3.3), so the current
standardized-**Laplace** default (raw kurtosis 6) in `AverageLaplace` is **not** the right form. The
name and the `P` parameter suggest an *average of `P`* Laplace-type components (which would approach
normality as `P` grows, matching the fixture), but the exact density must come from the cited paper —
**not** the source. **RECONCILE: obtain the ALD density definition from the specification paper, then
lock it against `distribution_moments.csv` / `distribution_quantiles.csv` (labels `ald_P8`,
`ald_P2`).**

---

*Add further specification derivations here as later phases (SM models, the fractional-differencing
operator, LM models, dual mean) are implemented — always from the papers/manual, never the source.*
