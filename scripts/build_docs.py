#!/usr/bin/env python
"""Build the quantica API reference with pdoc — one command, from the docstrings.

Regenerates ``docs/api/`` straight from the source NumPy-style docstrings, so the
reference always matches the code. **Never hand-edit the generated HTML** — change the
docstring and rerun this script (the ``interrogate`` gate guarantees every public
symbol has a docstring to render). Run::

    python scripts/build_docs.py            # or: pip install -e ".[docs]" first

The reference is organized by module, which mirrors the three pillars:

* ``quantica.pricing``   → derivatives pricing (engines, instruments, processes, ...)
* ``quantica.risk``      → risk & model validation (VaR/ES, credit, ML, FRTB)
* ``quantica.factor`` / ``quantica.portfolio`` → capital markets (factor model + portfolios)

Why pdoc (not Sphinx): pdoc renders the existing NumPy docstrings into a clean,
browsable, cross-linked HTML reference with **zero config files** and a single command,
and needs no LaTeX toolchain — so the reference regenerates trivially in CI and can
never drift. A Sphinx + LaTeX PDF would be closer to a printed CRAN manual but adds a
heavy, fragile toolchain for a portfolio repo; the HTML reference carries the same
content (every public signature, parameters, returns, raises).
"""

from __future__ import annotations

import pkgutil
import shutil
import subprocess
import sys
from pathlib import Path

import quantica

_ROOT = Path(__file__).resolve().parent.parent
_OUT = _ROOT / "docs" / "api"


def _all_modules() -> list[str]:
    """Every *public* ``quantica`` module, so pdoc emits a page for each (future-proof).

    Private (``_``-prefixed) submodules are internal implementation and are excluded
    from the reference, matching the docstring-coverage gate's private exemption.
    """
    modules = ["quantica"]
    for info in pkgutil.walk_packages(quantica.__path__, prefix="quantica."):
        if any(part.startswith("_") for part in info.name.split(".")):
            continue
        modules.append(info.name)
    return modules


def main() -> None:
    """Regenerate ``docs/api/`` from the ``quantica`` package docstrings via pdoc."""
    if _OUT.exists():
        shutil.rmtree(_OUT)
    _OUT.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pdoc",
        *_all_modules(),
        "--output-directory",
        str(_OUT),
        "--docformat",
        "numpy",
        "--math",  # render the :math:`...` LaTeX in the docstrings via MathJax
        "--no-show-source",  # a clean reference manual, not a source browser
        "--no-search",  # drop the ~5 MB search index; the module nav is enough (keeps docs/ lean)
    ]
    subprocess.run(cmd, check=True, cwd=_ROOT)
    pages = sorted(p.relative_to(_OUT).as_posix() for p in _OUT.rglob("*.html"))
    print(f"\nAPI reference written to {_OUT.relative_to(_ROOT).as_posix()}/ ({len(pages)} pages):")
    for page in pages:
        print(f"  {page}")


if __name__ == "__main__":
    main()
