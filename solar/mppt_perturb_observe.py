"""Perturb & Observe MPPT for a single-diode PV panel model.

Models a small PV panel (Ns series cells) with the explicit single-diode
equation (series resistance neglected for an analytic I(V)), then tracks the
maximum power point by perturbing the operating voltage and observing power.
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dataclasses import dataclass

import numpy as np

import matplotlib.pyplot as plt

from common.io import save_fig

# Physical / panel constants.
Q = 1.602176634e-19  # C
K_BOLTZMANN = 1.380649e-23  # J/K
N_CELLS = 36
IDEALITY = 1.3
ISC_STC = 5.0  # short-circuit current at 1000 W/m^2 (A)
I0 = 1e-9  # diode saturation current (A)


@dataclass
class PanelParams:
    n_cells: int = N_CELLS
    ideality: float = IDEALITY
    isc_stc: float = ISC_STC
    i0: float = I0


def thermal_voltage(temp_k: float) -> float:
    return K_BOLTZMANN * temp_k / Q


def photo_current(irradiance: float, params: PanelParams) -> float:
    """Light-generated current, linear in irradiance (W/m^2)."""
    return params.isc_stc * irradiance / 1000.0


def pv_current(v: np.ndarray, irradiance: float, temp_k: float,
               params: PanelParams | None = None) -> np.ndarray:
    """Panel output current at terminal voltage *v* (single-diode, explicit)."""
    params = params or PanelParams()
    vt = thermal_voltage(temp_k)
    iph = photo_current(irradiance, params)
    i = iph - params.i0 * (np.exp(v / (params.n_cells * params.ideality * vt)) - 1.0)
    return np.clip(i, 0.0, None)


def pv_power(v: np.ndarray, irradiance: float, temp_k: float,
             params: PanelParams | None = None) -> np.ndarray:
    return v * pv_current(v, irradiance, temp_k, params)


def open_circuit_voltage(irradiance: float, temp_k: float,
                         params: PanelParams | None = None) -> float:
    """Voc where current crosses zero (analytic from the diode equation)."""
    params = params or PanelParams()
    vt = thermal_voltage(temp_k)
    iph = photo_current(irradiance, params)
    return params.n_cells * params.ideality * vt * np.log(iph / params.i0 + 1.0)


@dataclass
class MpptTrace:
    voltage: np.ndarray
    power: np.ndarray
    mpp_voltage: float
    mpp_power: float


def perturb_and_observe(irradiance: float, temp_k: float, *, step: float = 0.3,
                        n_steps: int = 90, v_start: float | None = None,
                        params: PanelParams | None = None) -> MpptTrace:
    """Run P&O MPPT. Returns the operating-point trajectory and the tracked MPP.

    ``v_start`` defaults to ~0.6*Voc, a typical cold-start guess below the MPP.
    """
    params = params or PanelParams()
    voc = open_circuit_voltage(irradiance, temp_k, params)
    if v_start is None:
        v_start = 0.6 * voc
    v = float(np.clip(v_start, 0.5, voc - 0.1))
    direction = 1.0
    prev_power = pv_power(np.array(v), irradiance, temp_k, params).item()

    v_hist = [v]
    p_hist = [prev_power]
    for _ in range(n_steps):
        v_new = float(np.clip(v + direction * step, 0.3, voc - 0.05))
        p_new = pv_power(np.array(v_new), irradiance, temp_k, params).item()
        if p_new < prev_power:
            direction = -direction  # power dropped: reverse perturbation
        v, prev_power = v_new, p_new
        v_hist.append(v)
        p_hist.append(p_new)

    # Reference MPP from a dense sweep.
    v_grid = np.linspace(0.1, voc - 0.05, 2000)
    p_grid = pv_power(v_grid, irradiance, temp_k, params)
    idx = int(np.argmax(p_grid))
    return MpptTrace(np.array(v_hist), np.array(p_hist),
                     float(v_grid[idx]), float(p_grid[idx]))


def plot_mppt(irradiance: float, temp_k: float, trace: MpptTrace) -> pathlib.Path:
    params = PanelParams()
    voc = open_circuit_voltage(irradiance, temp_k, params)
    v_grid = np.linspace(0.0, voc, 500)
    p_grid = pv_power(v_grid, irradiance, temp_k, params)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(v_grid, p_grid, label="P-V curve")
    ax1.plot(trace.mpp_voltage, trace.mpp_power, "r*", ms=14, label="True MPP")
    ax1.plot(trace.voltage, trace.power, ".", color="orange", alpha=0.5,
             label="P&O operating points")
    ax1.set_xlabel("Voltage (V)")
    ax1.set_ylabel("Power (W)")
    ax1.set_title(f"P-V curve @ {irradiance:.0f} W/m^2")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(trace.power, "-o", ms=3)
    ax2.axhline(trace.mpp_power, color="r", ls="--", label="MPP power")
    ax2.set_xlabel("P&O iteration")
    ax2.set_ylabel("Tracked power (W)")
    ax2.set_title("MPPT convergence")
    ax2.grid(alpha=0.3)
    ax2.legend()
    return save_fig(fig, "mppt.png")


def main() -> None:
    irradiance = 1000.0
    temp_k = 298.15
    trace = perturb_and_observe(irradiance, temp_k)
    final_power = trace.power[-5:].mean()
    print("Perturb & Observe MPPT")
    print(f"  irradiance            : {irradiance:.0f} W/m^2")
    print(f"  true MPP              : {trace.mpp_power:.2f} W @ {trace.mpp_voltage:.2f} V")
    print(f"  tracked power (final) : {final_power:.2f} W")
    print(f"  tracking efficiency   : {100 * final_power / trace.mpp_power:.1f} %")
    path = plot_mppt(irradiance, temp_k, trace)
    print(f"Saved MPPT figure -> {path}")


if __name__ == "__main__":
    main()
