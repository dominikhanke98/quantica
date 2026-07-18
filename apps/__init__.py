"""Thin Streamlit + Plotly presentation layer over the ``quantica`` library.

CLAUDE.md §2 (non-negotiable): **zero quant logic lives here.** Every module in this
package imports from ``quantica`` and calls the already-tested core; the apps only
orchestrate those calls and shape the results for display. The separation is itself
part of what the portfolio demonstrates — a validated library, and a UI that is a
pure consumer of it.

Layout:

* ``_data`` / ``_derivatives`` / ``_risk`` / ``_capital`` — **compute** modules. Pure
  functions that call ``quantica`` and return plot-ready data (arrays, DataFrames,
  dataclasses). They import no Streamlit and no Plotly, so their smoke tests run under
  the plain ``dev`` install and CI catches a break without a UI dependency.
* ``quantica_app`` — the Streamlit entry point (``streamlit run apps/quantica_app.py``):
  widgets, caching, and Plotly rendering only.
"""

from __future__ import annotations
