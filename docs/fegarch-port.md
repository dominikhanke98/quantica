# The fEGarch clean-room port

An independent **clean-room reimplementation of the models described in the R package
[`fEGarch`](https://cran.r-project.org/package=fEGarch)** (Feng–Peitz–Schulz and co-authors) — a broad
EGARCH-type volatility family with long-memory / fractionally-integrated members that no mainstream
Python library offers — built from published specifications and validated against committed `fEGarch`
output fixtures. It is the flagship of Pillar V and the longest single arc in the repository
(Phases 0–7).

Every number on this page traces to a committed source: a section of
[`fegarch-spec-notes.md`](fegarch-spec-notes.md) (the equation-level derivation log, §1–§32), a named
test under [`tests/timeseries/fegarch/`](../tests/timeseries/fegarch), or
[`PROGRESS.md`](../PROGRESS.md). Where a figure is not in the record it is not stated — cite, don't
assert.

## What it is

A reimplementation of the whole `fEGarch` model family, in
[`quantica/timeseries/fegarch/`](../quantica/timeseries/fegarch):

- **Short-memory** — GARCH, GJR-GARCH, TGARCH, APARCH (the three asymmetric ones reconciled as a single
  APARCH-power recursion at δ = 2 / 1 / free).
- **The EGF family** — EGARCH, MEGARCH, MLog-GARCH (one generalized Type-I `g`-transformation engine,
  WP171 Eqs. 7–9) and Log-GARCH (Type-II).
- **Eight fractionally-integrated / long-memory models** — FIEGARCH, FIMEGARCH, FIMLog-GARCH (Type-I);
  FIGARCH, FIAPARCH, FITGARCH, FIGJR (variance-recursion); FILog-GARCH (Type-II) — composed with a
  hand-written `(1−L)^d` fractional-differencing operator.
- **All eight conditional distributions** — `norm`, `std`, `ged`, `ald` and their Fernández–Steel
  skewed variants `snorm` / `sstd` / `sged` / `sald`, each standardized to mean 0 / variance 1.
- **Dual mean** — ARMA-in-mean and FARIMA-in-mean fits coupled to the variance in one joint QMLE
  (`fEGarch` has no GARCH-in-mean; the dual mean is ARMA + FARIMA only — spec-notes §25).
- **Forecasting** — `predict_roll` (no-refit rolling one-step) and `predict` (multi-step), validated
  across the conditional-distribution surface (PR #28): GARCH across **all eight** laws (rolling *and*
  multi-step), and EGARCH multi-step across **six of the eight** — all but `std` / `sstd`, because
  `fEGarch`'s optimizer errors on `egarch×{std, sstd}` (the continuous-df Type-I EGF cases of finding 3
  below / spec-notes §20), so there is no reference forecast output to reproduce for those two.
- **Risk** — conditional VaR/ES (WP171 Eqs. 61–62) that **feed the repository's existing risk-pillar
  backtests**.
- **Diagnostics** — the weighted Ljung-Box, Engle–Ng sign-bias, and PIT goodness-of-fit fit-tests.
- **A semiparametric scale extension** — local-polynomial trend estimation (the `smoots`/`esemifar`
  procedure) wired into a fixed-bandwidth semiparametric EGF fit.

**Clean-room means clean-room** (the binding method, CLAUDE.md §12): the code is implemented purely
from the published mathematics — WP171/WP173/WP175, the cited underlying papers, and the
`smoots`/`esemifar` papers — and **never** from the `fEGarch` / `smoots` / `esemifar` source, which is
never read, cloned, or translated. Correctness is checked **only** against committed R *output*
fixtures ([`tests/fixtures/fegarch/`](../tests/fixtures/fegarch), `fEGarch` 1.0.6 / R 4.6.1, generated
once by running the packages' public functions; R is never a dependency of `quantica` or its CI).
`fEGarch` and the papers are credited as the specification source; `quantica` stays MIT because the
work is original, from specifications, and validated against outputs.

## The validation methodology

This is the model-validation spine, and it is the point of the exercise: *effective challenge* —
reimplement the reference independently from its specification, then measure where the two agree and
where they diverge.

**Two tolerance classes, kept distinct.** Every model is validated at two levels, and the write-up
never conflates them:

- **Seam-exact (machine precision, ~1e-15).** The *recursion* — the variance / log-variance update
  itself — is reconstructed at the reference's **own** reported parameters and matched to the fixture's
  conditional-SD series. This isolates the model's correctness from the optimizer. Examples: the
  FILog-GARCH recursion reproduces the fixture σ to `1.05e-15` at `fEGarch`'s params (spec-notes §14.1);
  the GARCH rolling-forecast reconstruction pins to `6.939e-18` (§27.1); the semiparametric scale
  reconstructs `fEGarch`'s `scale_fun` to `~6e-16` (§32). One refinement of what "seam-exact" means for
  the EGF family: the four EGF centering moments (the `E|η|`-type constants the log-variance recursion is
  built on) are seam-exact to machine epsilon (`~1e-15`) for the **symmetric** laws, which have closed
  forms, but to the **quadrature floor** (`~1e-13`) for the **Fernández–Steel skew** laws, whose moments
  are integrated numerically over the skew wrapper. That is a precision boundary — the honest meaning of
  seam-exact for skew EGF centering — not a defect (PR #28).
- **Fit-tolerance (~1e-5 / 1e-6).** The *fitted parameters* are optimizer-dependent, so an independent
  optimizer lands a few `1e-6` away — the same category as the port's GARCH/EGARCH parameter matches.
  Stated as a tolerance, never as machine precision.

**The reconstruction gate is the pre-build correctness proof.** Before a model is built, its recursion
is reconstructed in Python from the fixture's parameters and checked against the fixture series. A
machine-exact gate (e.g. the predict-roll σ̂ at `6.939e-18`, §27.1) proves the recursion and the
pre-sample conditioning are right *before* any fitting code exists, so a later fit discrepancy can be
localized to the optimizer rather than the model.

**Tolerances are structured by identification, not flattened to one number.** Where a parameter is
sharply identified the test is tight; where it sits on a flat or near-common-root ridge (weakly
identified) the tolerance is set by the ridge's actual cross-platform behavior and the *seam* carries
the correctness assertion instead. The platform-sensitive ALD ridges are re-classified this way rather
than tolerance-widened to green (spec-notes §26): the seam stays the platform-invariant proof, the fit
is validated as a converged ridge optimum.

## The effective-challenge findings

The centerpiece: cases where reimplementing the models independently surfaced that the reference is
doing something subtle, non-textbook, or — in several cases — **wrong**. Each is a finding with a number
and a committed source.

### 1. `fEGarch`'s committed FILog-GARCH fit is a strictly-dominated local optimum — beaten by ~120 log-likelihood

The FILog-GARCH×norm fixture reports `mu = −0.003054, loglik = 7435.98`. An independent fit from the
data mean reaches `7555.85` — **+119.87 log-likelihood higher** — at an economically sensible optimum
(`mu` near the data mean, `d ≈ 0.29` interior, the coefficients well separated). A multi-start check
settles the mechanism: an optimizer *placed* at `mu = −0.003` stays there (it is a genuine local
optimum, not a stopped-mid-climb point), but **no data-driven start reaches it** — three random starts
land `+117` to `+123` higher, and even a start from `fEGarch`'s own reported params escapes to `+101`.
Because the recursion at `fEGarch`'s own params reproduces the fixture σ to `~1e-15` (the seam is
machine-exact — **the model is right**), this is `fEGarch`'s *optimizer* landing in a strictly-dominated
local optimum, not a model defect. Validated honestly in two parts
([`test_filoggarch.py`](../tests/timeseries/fegarch/test_filoggarch.py)): the seam is asserted
machine-exact at `fEGarch`'s params, and the fit is asserted to *strictly dominate* the fixture from
every sensible start — we do **not** assert a parameter match to the dominated point, which would
validate a bad optimum. This is the port's clearest effective-challenge result (spec-notes §14.3).

### 2. `fEGarch`'s P-profiling selected a dominated interior P — the same pattern on the discrete P-axis

The average-Laplace degree `P` is profiled over an integer grid. FILog-GARCH×sald is the **only** fixture
in the whole port where `fEGarch` selected an *interior* `P` (`P = 3`) rather than the boundary — the
natural discriminating test for whether our profiling genuinely compares per-`P` likelihoods or merely
defaults to the boundary. Read-only per-`P` verification gives a smooth monotone-concave profile
`P=3: 7550.13 → P=4: 7554.35 → P=5: 7556.80`, with `P = 4/5` clean converged optima
(`success=True`, `d ≈ 0.305` interior, nothing at a bound; the `P=5` series recomputes its loglik from
scratch to `0.0`). **Our `P = 3` fit reproduces `fEGarch`'s `P = 3` exactly (`7550.13` = the fixture)**,
so it is not a different objective — yet our profiling moves off `P = 3` and selects `P = 5`, dominating
the fixture by `+6.67`. `fEGarch`'s interior `P = 3` was an optimizer stall on the flat ridge — the §14
pattern on the discrete P-axis — and because our profiling reproduces each converged `P` value and still
moves, it *validates* that the port's other `P = 5` selections are genuine maxima, not a return-the-
boundary artifact. ([`test_fi_egf_family_distributions.py`](../tests/timeseries/fegarch/test_fi_egf_family_distributions.py);
spec-notes §22.4.)

### 3. `fEGarch`'s optimizer fails outright on the continuous-df Type-I cases where ours converges

Six continuous-df Type-I EGF fits — EGARCH/MEGARCH/MLog-GARCH under `std`/`sstd`, plus Log-GARCH under
`std` — make `fEGarch`'s optimizer error out ("Error during optimization"): the df-dependent `E|η|`
centering combined with the near-unit-root EGF ARMA is a flat-ridge starting-point fragility. **No
fixture exists for these; none was fabricated.** Instead they are validated by known-truth: simulating
at `df = 6` (`+skew = 0.85` for the skewed cases), our Nelder-Mead-plus-restart optimizer **converges on
all of them** and recovers the planted `df` (`|dev|/SE ≤ 1.2`). The `fEGarch` split where `loggarch×std`
fails but `loggarch×sstd` converges confirms these are non-deterministic solver failures, not model
failures. An independent reimplementation succeeding on the reference's hardest cases is itself an
effective-challenge result.
([`test_egf_family_distributions.py`](../tests/timeseries/fegarch/test_egf_family_distributions.py);
spec-notes §20.3, §21.3.)

### 4. `fEGarch`'s GJR uses the APARCH `(|ε|−γε)²` kernel, not the textbook Glosten indicator

Reconciling the GJR fixture forced the kernel question. The textbook Glosten–Jagannathan–Runkle
*indicator* form `σ² = ω + (φ₁ + γ₁·𝟙[ε<0])·ε² + β₁·σ²` does **not** reproduce the committed GJR series
(max σ deviation `~1e-2`, under both sign conventions). The APARCH-at-δ=2 kernel `(|ε| − γ₁ε)²` does —
so `fEGarch`'s GJR is the APARCH-power news, not the textbook indicator. The finding carries into the
fractional case: the FIGJR kernel is `(|ε| − γε)²` as well, not the Glosten indicator (the Phase-1 GJR
finding in FI form). (spec-notes §4; Phase 1 — [`PROGRESS.md`](../PROGRESS.md).)

### 5. Two pre-sample seeds are irreducible from the reference's public output — documented as bounded limits, not faked

A clean-room port validated against *output only* has an honest boundary: a few internal quantities
cannot be recovered from the published series because the published series does not pin them. The port
reaches exactly **two** such points — the APARCH `σ₀` pre-sample seed (the first irreducible-from-output
limit) and the FIAPARCH pre-sample seed (the second, §12.3). In each case the seam is otherwise
machine-exact (the δ-power recursion reproduces at the reference's params), the seed is set to a
principled value (`mean(news)` for FIAPARCH) and the exact residual is **documented as a bounded limit**
— not reverse-engineered from the forbidden source, and not silently papered over. The honest-boundary
discipline is part of the quality story: the port states precisely what it cannot reduce from available
output. (spec-notes §12.3.)

### 6. The ARMA(1,1)-in-mean optimum `fEGarch`'s plain-ARMA optimizer missed — triply corroborated

Our ARMA(1,1)-in-mean fit **dominates** the `arma11` fixture by `+0.063` log-likelihood, at a different
`(ar1, ma1)` on a near-common-root ridge. The domination is corroborated three independent ways: (a) our
own ARMA fit reaches it; (b) `fEGarch`'s *own* FARIMA optimizer — given the extra fractional-`D`
dimension — reaches the same near-common-root optimum its plain-ARMA optimizer missed; and (c) our
FARIMA fit reaches it too (`ar1 = −0.107, ma1 = +0.136`, `D ≈ 0`). Three optimizers converge on the
point `fEGarch`'s plain-ARMA optimizer did not. (spec-notes §23.3, §24.4.)

### 7. Convention catches the fixtures forced out — because the fixtures expose only p-values

The diagnostic fit-tests expose **only p-values** (no raw statistics), which turns reproduction into a
convention probe. Two conventions came out that the obvious implementation gets wrong:

- The **weighted Ljung-Box** is the Fisher–Gallagher (2012) weighted portmanteau with a **Gamma**
  approximation to the null, not the classic equal-weight Ljung-Box with a χ² null — the classic χ²₂₀
  p-value differs from the fixture by `>1e-3`, while the Gamma-weighted form matches to `~1e-15`.
- The **sign-bias** test's size regressors use the **raw** residual (not the standardized one), and its
  joint test is the **LM statistic `m·R² ~ χ²₃`, not the OLS F-test** — the LM form reproduces the
  fixture p-value `0.62320` to `1.3e-14`, while the F-test gives `0.62360`, off by `4e-4`, so the
  p-value alone discriminates the two.

Because the residuals are reconstructed machine-exact, the p-value match tests the null-distribution
computation itself; the scipy Gamma/χ²/Student-t CDFs agree with R's to machine precision.
([`test_diagnostics.py`](../tests/timeseries/fegarch/test_diagnostics.py); spec-notes §28.)

## The coherence payoff

The ported models connect to validation machinery already in the repository. The semiparametric-free
VaR/ES forecasts (`measure_risk`, WP171 Eqs. 61–62) are sign-mapped into the **existing risk-pillar
backtests** — Kupiec, Christoffersen, Basel traffic-light, Acerbi–Székely — via a thin
`backtest_return_forecasts` adapter, with no new backtest mathematics. On the GARCH(1,1) forecast over
250 days this produces a clean coherence demonstration: at the 99% level, 1 exception (2.5 expected),
Kupiec p `0.278`, Christoffersen conditional-coverage p `0.553`, a green Basel zone, and an
Acerbi–Székely Z2 of `−0.650` (realized tail losses below the predicted ES). A time-series model from
Pillar V validated through the risk pillar's backtests — one pillar's output as another's validated
input — is the cross-pillar payoff. (spec-notes §27.4.)

## The honestly-bounded scope

The port is complete through the entire roadmap (Phases 0–7), each phase done **to the extent
reproducible from available literature**. There is exactly one documented limitation, and it is stated
precisely because the honest boundary is itself part of the quality argument.

**The single gap is the semiparametric *automatic* bandwidth selector**, and it decomposes into two
specific missing papers:

- The short-memory data-driven lag-window width (`L0.opt`) is the nested **Bühlmann (1996)**
  "Locally Adaptive Lag-Window Spectral Estimation" iterative-plug-in; Feng–Gries–Fritz (2020) cites
  but does not reproduce its defining equations, and the Bühlmann paper is not in
  [`literature/`](../literature).
- The AMISE bandwidth constants (the equivalent-kernel integrals and the enlargement factor) are
  tabulated in **Feng–Heiler (2009)**, also cited-not-reproduced and absent.

Everything *around* the gap is built and validated: the local-polynomial regression is machine-exact
against `smoots::gsmooth` across all 16 (polynomial-order × kernel × boundary) conventions (spec-notes
§30); the derivative local polynomial, the integrated squared derivative, and the short-memory Bartlett
variance factor (given the window) are machine-exact components (§31); and the **fixed-bandwidth
semiparametric EGF fit is fully reproducible** — the six-step scale procedure (WP171 §2.2.1, with the
empirical `Ĉ_T` retransform correction) is machine-exact (`~6e-16`) and the parametric stage is at the
established fit tolerance (§32). So the semiparametric model is reproducible **at a user-specified
bandwidth**; `semiparametric_fit(bwidth=None)` raises `NotImplementedError` pointing at the gap rather
than guessing an unvalidated selector.

Two precisely-attributed missing-paper gaps, both inside the one auto-bandwidth selector, each of which
closes in an afternoon with library access — against a port that is otherwise complete across every
model, every conditional distribution, the dual mean, forecasting, the risk tie-back, the diagnostics,
and the semiparametric scale. The boundary is drawn where the evidence runs out, and it is drawn in ink.

## Where to read more

- [`docs/fegarch-spec-notes.md`](fegarch-spec-notes.md) — the equation-level derivation and
  effective-challenge log (§1–§32), the source for every number here.
- [`PROGRESS.md`](../PROGRESS.md) — the phase-by-phase build history.
- [`docs/fegarch-port-roadmap.md`](fegarch-port-roadmap.md) — the clean-room method and phase plan.
- [`tests/timeseries/fegarch/`](../tests/timeseries/fegarch) — the validations, one test module per
  model family.
- [API reference](https://raw.githack.com/dominikhanke98/quantica/main/docs/api/index.html) — every
  public symbol, generated from the docstrings.
