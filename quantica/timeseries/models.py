r"""GARCH-family volatility models and out-of-sample forecasting (thin ``arch`` wrappers).

The GARCH *estimation* is plumbing that :mod:`arch` already does well, so this module leans on it
(CLAUDE.md §3) rather than re-deriving the maximum-likelihood recursions. What it adds is the
thin, typed surface the rest of the pillar needs:

* :func:`fit_volatility_model` — fit ``GARCH(1,1)``, ``GJR-GARCH`` (an asymmetry/leverage term)
  or ``EGARCH`` to a return series and expose the parameters, log-likelihood and information
  criteria as plain Python types.
* :func:`rolling_forecast` — the genuinely useful piece the libraries do *not* ship as a
  one-liner: a **rolling / expanding-window, one-step-ahead out-of-sample** variance forecast,
  refitting periodically, paired with the squared-return proxy it will be scored against.

Scale convention: pass returns in **percent** units (e.g. ``100 * log-returns``). GARCH
likelihoods are ill-conditioned on raw daily returns (variance ``~1e-4``), so the whole pillar
works in percent, and every variance forecast and proxy is therefore in **percent-squared**.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from quantica.core.types import FloatArray

__all__ = [
    "ForecastResult",
    "VolatilityFit",
    "VolatilityModel",
    "fit_volatility_model",
    "rolling_forecast",
]

#: The GARCH-family specifications this pillar supports.
VolatilityModel = Literal["GARCH", "GJR", "EGARCH"]


def _build_model(returns: FloatArray, model: VolatilityModel):  # type: ignore[no-untyped-def]
    """Construct the :mod:`arch` mean/volatility model for a specification (constant mean, normal).

    ``GJR-GARCH`` is a GARCH with an asymmetry order ``o=1``; ``EGARCH`` uses the log-variance
    recursion. The optional :mod:`arch` dependency is imported here so the pillar degrades to a
    clear ``ImportError`` when it is absent.
    """
    try:
        from arch import arch_model
    except ImportError as exc:  # pragma: no cover - exercised only without arch
        raise ImportError(
            "the timeseries pillar needs the 'arch' dependency; install it with `pip install arch`."
        ) from exc

    data = np.asarray(returns, dtype=np.float64)
    if model == "GARCH":
        return arch_model(data, mean="Constant", vol="GARCH", p=1, o=0, q=1, dist="normal")
    if model == "GJR":
        return arch_model(data, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="normal")
    if model == "EGARCH":
        return arch_model(data, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="normal")
    raise ValueError(f"unknown model {model!r}")


@dataclass(frozen=True)
class VolatilityFit:
    """A fitted GARCH-family model, reduced to plain types.

    Attributes
    ----------
    model : str
        The specification (``"GARCH"``, ``"GJR"`` or ``"EGARCH"``).
    params : dict of str to float
        The estimated parameters (e.g. ``mu``, ``omega``, ``alpha[1]``, ``gamma[1]``,
        ``beta[1]``), named as :mod:`arch` names them.
    loglikelihood : float
        The maximised log-likelihood.
    aic, bic : float
        Akaike / Bayesian information criteria (lower is better).
    conditional_volatility : ndarray
        The in-sample conditional volatility :math:`\\sigma_t` (percent).
    """

    model: str
    params: dict[str, float]
    loglikelihood: float
    aic: float
    bic: float
    conditional_volatility: FloatArray


def fit_volatility_model(returns: FloatArray, model: VolatilityModel = "GARCH") -> VolatilityFit:
    """Fit a GARCH-family volatility model to a return series via :mod:`arch`.

    Parameters
    ----------
    returns : ndarray
        The return series, in **percent** units (see the module note).
    model : {"GARCH", "GJR", "EGARCH"}, optional
        The specification to fit (default ``"GARCH"``).

    Returns
    -------
    VolatilityFit
        The fitted parameters, log-likelihood, information criteria and conditional volatility.

    Raises
    ------
    ImportError
        If the optional :mod:`arch` dependency is not installed.
    """
    result = _build_model(returns, model).fit(disp="off")
    return VolatilityFit(
        model=model,
        params={k: float(v) for k, v in result.params.items()},
        loglikelihood=float(result.loglikelihood),
        aic=float(result.aic),
        bic=float(result.bic),
        conditional_volatility=np.asarray(result.conditional_volatility, dtype=np.float64),
    )


@dataclass(frozen=True)
class ForecastResult:
    """A one-step-ahead out-of-sample variance forecast and the proxy it is scored against.

    Attributes
    ----------
    model : str
        The specification that produced the forecast.
    variance_forecast : ndarray
        The one-step-ahead conditional-variance forecasts :math:`\\hat\\sigma_t^2` over the
        out-of-sample window (percent-squared).
    realized_proxy : ndarray
        The squared-return proxy for the (latent) realised variance at each forecast target,
        aligned with ``variance_forecast`` (percent-squared).
    actual_returns : ndarray
        The realised returns at each forecast target (percent).
    """

    model: str
    variance_forecast: FloatArray
    realized_proxy: FloatArray
    actual_returns: FloatArray


def rolling_forecast(
    returns: FloatArray,
    model: VolatilityModel = "GARCH",
    *,
    first_forecast: int,
    refit: int = 25,
    expanding: bool = True,
) -> ForecastResult:
    """Produce rolling/expanding one-step-ahead out-of-sample variance forecasts.

    For each target ``t`` in ``[first_forecast, len(returns))`` the model is estimated on the
    history **before** ``t`` (an expanding window from the start, or a rolling window of fixed
    length ``first_forecast``) and used to forecast :math:`\\hat\\sigma_t^2`. Re-estimating every
    step is expensive, so the fit is refreshed every ``refit`` steps and reused in between — the
    standard practitioner compromise. The forecast is genuinely out-of-sample: the target return
    never enters the estimation window.

    Parameters
    ----------
    returns : ndarray
        The full return series in **percent** units.
    model : {"GARCH", "GJR", "EGARCH"}, optional
        The specification to forecast with (default ``"GARCH"``).
    first_forecast : int
        Index of the first out-of-sample target; also the rolling-window length when
        ``expanding=False``. Must satisfy ``0 < first_forecast < len(returns)``.
    refit : int, optional
        Re-estimate the model every ``refit`` targets (default 25).
    expanding : bool, optional
        ``True`` for an expanding window from the start (default); ``False`` for a fixed-length
        rolling window.

    Returns
    -------
    ForecastResult
        The variance forecasts, the squared-return proxy, and the realised returns.

    Raises
    ------
    ValueError
        If ``first_forecast`` is out of range or ``refit`` is not positive.
    ImportError
        If the optional :mod:`arch` dependency is not installed.
    """
    r = np.asarray(returns, dtype=np.float64)
    n = r.size
    if not 0 < first_forecast < n:
        raise ValueError("require 0 < first_forecast < len(returns)")
    if refit <= 0:
        raise ValueError("refit must be positive")

    targets = range(first_forecast, n)
    forecasts = np.empty(n - first_forecast, dtype=np.float64)
    fitted = None
    for i, t in enumerate(targets):
        if i % refit == 0:
            lo = 0 if expanding else t - first_forecast
            fitted = _build_model(r[lo:t], model).fit(disp="off")
        assert fitted is not None  # set on i == 0
        forecast = fitted.forecast(horizon=1, reindex=False)
        forecasts[i] = float(forecast.variance.to_numpy()[-1, 0])

    actual = r[first_forecast:]
    return ForecastResult(
        model=model,
        variance_forecast=forecasts,
        realized_proxy=actual * actual,
        actual_returns=actual,
    )
