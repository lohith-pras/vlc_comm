"""Doppler / mobility model for V2X-VLC OFDM.

An EV approaching a charger has low relative speed, but residual motion makes
the channel time-varying. We model that as a normalized carrier frequency
offset (CFO) on the OFDM grid, which rotates the wanted subcarrier and leaks
energy into neighbours (inter-carrier interference, ICI). This is an
instructive equivalent model -- in IM/DD VLC the optical-carrier Doppler shift
itself does not survive envelope detection; the mobility penalty enters through
channel time-variation, captured here as an effective CFO.
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

import matplotlib.pyplot as plt

from common.io import save_fig

C_LIGHT = 3e8  # m/s


def doppler_shift(velocity_ms: float, equiv_carrier_hz: float) -> float:
    """Doppler frequency shift f_d = v/c * f_c for an equivalent RF-like carrier."""
    return velocity_ms / C_LIGHT * equiv_carrier_hz


def normalized_cfo(velocity_ms: float, equiv_carrier_hz: float,
                   subcarrier_spacing_hz: float) -> float:
    """CFO normalized to subcarrier spacing, epsilon = f_d / df."""
    return doppler_shift(velocity_ms, equiv_carrier_hz) / subcarrier_spacing_hz


def apply_cfo(time_signal: np.ndarray, epsilon: float, n_fft: int) -> np.ndarray:
    """Apply a normalized CFO to a time-domain OFDM signal (phase ramp)."""
    n = np.arange(len(time_signal))
    return time_signal * np.exp(1j * 2 * np.pi * epsilon * n / n_fft)


def ici_sinr_db(epsilon: float) -> float:
    """Approximate SINR (dB) from self-ICI for a normalized CFO epsilon.

    Uses the classic small-CFO approximation SINR ~ 1 / ( (pi*eps)^2 / 3 ).
    """
    eps = abs(epsilon)
    if eps < 1e-9:
        return np.inf
    sinr = 3.0 / (np.pi * eps) ** 2
    return 10 * np.log10(sinr)


def plot_doppler(equiv_carrier_hz: float = 1e9,
                 subcarrier_spacing_hz: float = 78.125e3) -> pathlib.Path:
    velocities = np.linspace(0.0, 30.0, 100)  # 0..30 m/s (~108 km/h)
    eps = np.array([normalized_cfo(v, equiv_carrier_hz, subcarrier_spacing_hz)
                    for v in velocities])
    sinr = np.array([ici_sinr_db(e) for e in eps])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(velocities * 3.6, eps)
    ax1.set_xlabel("Relative speed (km/h)")
    ax1.set_ylabel("Normalized CFO  $\\epsilon$")
    ax1.set_title("Doppler -> normalized CFO")
    ax1.grid(alpha=0.3)

    finite = np.isfinite(sinr)
    ax2.plot(velocities[finite] * 3.6, sinr[finite], color="tab:red")
    ax2.set_xlabel("Relative speed (km/h)")
    ax2.set_ylabel("ICI-limited SINR (dB)")
    ax2.set_title("Mobility-induced ICI penalty")
    ax2.grid(alpha=0.3)
    return save_fig(fig, "doppler.png")


def main() -> None:
    equiv_carrier = 1e9       # equivalent RF carrier for the mobility model (Hz)
    df = 78.125e3             # subcarrier spacing (Hz)
    print("V2X-VLC Doppler / CFO model")
    for v in (1.0, 5.0, 15.0, 30.0):
        eps = normalized_cfo(v, equiv_carrier, df)
        print(f"  v={v:5.1f} m/s : f_d={doppler_shift(v, equiv_carrier):.1f} Hz, "
              f"eps={eps:.4f}, ICI-SINR={ici_sinr_db(eps):.1f} dB")
    path = plot_doppler(equiv_carrier, df)
    print(f"Saved Doppler figure -> {path}")


if __name__ == "__main__":
    main()
