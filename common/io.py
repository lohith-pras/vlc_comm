"""Shared helpers: locate project root, manage the ``outputs/`` directory, save figures."""
from __future__ import annotations

import pathlib

import matplotlib

# Use a non-interactive backend so modules run headless (CI, __main__ smoke runs).
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def project_root() -> pathlib.Path:
    """Walk upward from this file until the directory containing ``pyproject.toml``."""
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[1]


def outputs_dir() -> pathlib.Path:
    """Return (creating if needed) the project-level ``outputs/`` directory."""
    out = project_root() / "outputs"
    out.mkdir(exist_ok=True)
    return out


def save_fig(fig: "plt.Figure", name: str, dpi: int = 120) -> pathlib.Path:
    """Save *fig* to ``outputs/<name>`` and return the path. Closes the figure."""
    path = outputs_dir() / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
