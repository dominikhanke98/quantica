r"""VaR/ES engines: historical, parametric, filtered-historical, and Monte Carlo.

Each engine consumes an asset-return matrix and a :class:`~quantica.risk.portfolio.Portfolio`
and returns a :class:`~quantica.risk.measures.RiskEstimate` at a confidence level.
They share the :class:`VaREngine` protocol so a backtest can roll any of them over
a window without knowing which method it holds.

The four methods trade off different assumptions:

* **Historical simulation** — no distributional assumption; the empirical quantile
  of the realised portfolio losses. Captures fat tails and skew present in the
  sample, but is blind to anything the sample has not yet shown and reacts slowly
  to volatility changes.
* **Parametric (variance--covariance)** — assumes multivariate-normal returns and
  reads VaR/ES off the closed form using the mean vector and covariance matrix.
  Fast and smooth, but the **normality assumption understates tail risk** — real
  return tails are heavier, so parametric-normal VaR is typically optimistic. This
  caveat is the point of backtesting it.
* **Filtered historical simulation (FHS)** — fits a GARCH(1,1) volatility filter
  (via :mod:`arch`), standardises the returns by the conditional volatility,
  bootstraps the standardised residuals, and rescales by the *current* volatility
  forecast. Combines a nonparametric tail with a dynamic volatility, so it reacts
  to clustering while keeping the empirical residual shape.
* **Monte Carlo** — simulates returns from the fitted multivariate normal and takes
  the empirical quantile. With a normal fit it converges to the parametric closed
  form (a validation cross-check); its value is the freedom to plug in a richer
  P\&L map later (e.g. full option revaluation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from quantica.risk.measures import RiskEstimate, empirical_var_es, normal_var_es

if TYPE_CHECKING:
    from quantica.core.types import FloatArray
    from quantica.risk.portfolio import Portfolio

__all__ = [
    "FilteredHistoricalSimulationVaR",
    "HistoricalSimulationVaR",
    "MonteCarloVaR",
    "ParametricVaR",
    "VaREngine",
]

# arch's optimiser is happier on percent-scaled returns; we scale in and out.
_ARCH_SCALE = 100.0


@runtime_checkable
class VaREngine(Protocol):
    """A method that estimates VaR/ES for a portfolio from an asset-return history."""

    def estimate(
        self, asset_returns: FloatArray, portfolio: Portfolio, *, level: float
    ) -> RiskEstimate:
        """Return the VaR/ES of ``portfolio`` given the ``asset_returns`` window."""
        ...


class HistoricalSimulationVaR:
    """Empirical-quantile VaR/ES of the realised portfolio losses (no distribution)."""

    def estimate(
        self, asset_returns: FloatArray, portfolio: Portfolio, *, level: float
    ) -> RiskEstimate:
        losses = portfolio.losses(asset_returns)
        est = empirical_var_es(losses, level, method="historical")
        return est


class ParametricVaR:
    r"""Variance--covariance VaR/ES under a multivariate-normal return assumption.

    Uses the sample mean vector :math:`\mu` and covariance matrix :math:`\Sigma` to
    form the portfolio P\&L moments :math:`\mu_p = V\,w^\top\mu`,
    :math:`\sigma_p = V\sqrt{w^\top \Sigma w}`, then the Gaussian closed form. The
    normality assumption is documented and is exactly what the backtests challenge.
    """

    def estimate(
        self, asset_returns: FloatArray, portfolio: Portfolio, *, level: float
    ) -> RiskEstimate:
        R = np.asarray(asset_returns, dtype=np.float64)
        if R.ndim == 1:
            R = R[:, None]
        w = np.asarray(portfolio.weights, dtype=np.float64)
        mean_vec = R.mean(axis=0)
        cov = np.cov(R, rowvar=False)
        cov = np.atleast_2d(cov)
        mean_p = portfolio.value * float(w @ mean_vec)
        var_p = portfolio.value**2 * float(w @ cov @ w)
        sigma_p = float(np.sqrt(max(var_p, 0.0)))
        return normal_var_es(mean_p, sigma_p, level, method="parametric-normal")


class MonteCarloVaR:
    r"""Monte-Carlo VaR/ES: simulate returns from the fitted multivariate normal.

    Parameters
    ----------
    n_sims : int
        Number of simulated return draws.
    rng : numpy.random.Generator
        Seeded generator, injected for reproducibility (CLAUDE.md §3).

    Notes
    -----
    With a normal fit this converges to :class:`ParametricVaR` as ``n_sims`` grows —
    a deliberate cross-check. Its real value is that the P\&L map can be swapped for
    a nonlinear one (option revaluation) without touching the risk/backtest layer.
    """

    def __init__(self, n_sims: int, *, rng: np.random.Generator) -> None:
        if n_sims < 2:
            raise ValueError(f"n_sims must be at least 2, got {n_sims}")
        self.n_sims = n_sims
        self._rng = rng

    def estimate(
        self, asset_returns: FloatArray, portfolio: Portfolio, *, level: float
    ) -> RiskEstimate:
        R = np.asarray(asset_returns, dtype=np.float64)
        if R.ndim == 1:
            R = R[:, None]
        mean_vec = R.mean(axis=0)
        cov = np.atleast_2d(np.cov(R, rowvar=False))
        draws = self._rng.multivariate_normal(mean_vec, cov, size=self.n_sims)
        losses = -portfolio.value * (draws @ np.asarray(portfolio.weights, dtype=np.float64))
        return empirical_var_es(losses, level, method="monte-carlo")


class FilteredHistoricalSimulationVaR:
    r"""FHS VaR/ES: a GARCH(1,1) volatility filter over bootstrapped residuals.

    Fits a constant-mean GARCH(1,1) to the portfolio return series, standardises by
    the conditional volatility, and rebuilds a one-step-ahead loss sample as
    :math:`\hat\mu + \hat\sigma_{T+1}\, z_i` over the standardised residuals
    :math:`z_i`. The empirical quantile of that sample is the FHS VaR/ES — a
    nonparametric tail riding on a dynamic volatility (Barone-Adesi et al., 1999).

    Parameters
    ----------
    p, q : int, optional
        GARCH orders (default 1, 1).

    Notes
    -----
    Requires the optional :mod:`arch` dependency; the import is deferred so the rest
    of :mod:`quantica.risk` works without it.
    """

    def __init__(self, *, p: int = 1, q: int = 1) -> None:
        self.p = p
        self.q = q

    def estimate(
        self, asset_returns: FloatArray, portfolio: Portfolio, *, level: float
    ) -> RiskEstimate:
        try:
            from arch import arch_model
        except ImportError as exc:  # pragma: no cover - exercised only without arch
            raise ImportError(
                "FilteredHistoricalSimulationVaR needs the optional 'arch' dependency; "
                "install it with `pip install arch`."
            ) from exc

        returns = portfolio.portfolio_returns(asset_returns) * _ARCH_SCALE
        model = arch_model(returns, mean="Constant", vol="GARCH", p=self.p, q=self.q, dist="normal")
        fit = model.fit(disp="off")

        # Standardised residuals (the empirical shock distribution) and the
        # one-step-ahead conditional volatility forecast.
        z = np.asarray(fit.std_resid, dtype=np.float64)
        z = z[np.isfinite(z)]
        forecast = fit.forecast(horizon=1, reindex=False)
        sigma_next = float(np.sqrt(forecast.variance.to_numpy()[-1, 0]))
        mu = float(fit.params["mu"])

        # Rebuild next-day returns from the current vol and the residual shape, undo
        # the percent scaling, and take the empirical loss quantile.
        simulated_returns = (mu + sigma_next * z) / _ARCH_SCALE
        losses = -portfolio.value * simulated_returns
        return empirical_var_es(losses, level, method="filtered-historical")
