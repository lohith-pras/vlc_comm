"""Indoor VLC channel: Lambertian line-of-sight DC gain and electrical SNR.

Single LED source, single photodiode receiver, LOS link up to ~5 m. Implements
the standard generalised-Lambertian optical DC gain and a shot+thermal noise
model to derive the received electrical SNR.
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from dataclasses import dataclass

import numpy as np

import matplotlib.pyplot as plt

from common.io import save_fig

Q = 1.602176634e-19  # C
K_BOLTZMANN = 1.380649e-23  # J/K


@dataclass
class IndoorLinkParams:
    tx_power: float = 1.0           # transmitted optical power (W)
    half_power_angle_deg: float = 30.0   # LED semi-angle at half power
    detector_area: float = 1e-4    # photodiode active area (m^2)
    fov_deg: float = 60.0          # receiver field of view (half-angle)
    responsivity: float = 0.5      # A/W
    bandwidth: float = 10e6        # Hz
    background_current: float = 5.1e-3  # ambient-induced PD current (A)
    temperature: float = 295.0     # K
    tia_resistance: float = 1e4    # transimpedance feedback resistor (Ohm)


def lambertian_order(half_power_angle_deg: float) -> float:
    """Mode number m of the generalised Lambertian radiation pattern."""
    phi_half = np.radians(half_power_angle_deg)
    return -np.log(2.0) / np.log(np.cos(phi_half))


def dc_gain(distance: np.ndarray, irradiance_angle_deg: float,
            incidence_angle_deg: float, params: IndoorLinkParams | None = None
            ) -> np.ndarray:
    """LOS optical DC channel gain H(0).

    ``irradiance_angle_deg`` (phi) is measured at the transmitter, and
    ``incidence_angle_deg`` (psi) at the receiver. Gain is zero outside the FOV.
    """
    params = params or IndoorLinkParams()
    m = lambertian_order(params.half_power_angle_deg)
    phi = np.radians(irradiance_angle_deg)
    psi = np.radians(incidence_angle_deg)
    d = np.asarray(distance, dtype=float)
    gain = ((m + 1) * params.detector_area / (2 * np.pi * d ** 2)
            * np.cos(phi) ** m * np.cos(psi))
    inside_fov = psi <= np.radians(params.fov_deg)
    return np.where(inside_fov, np.clip(gain, 0.0, None), 0.0)


def received_power(distance: np.ndarray, irradiance_angle_deg: float = 0.0,
                   incidence_angle_deg: float = 0.0,
                   params: IndoorLinkParams | None = None) -> np.ndarray:
    """Received optical power (W)."""
    params = params or IndoorLinkParams()
    return params.tx_power * dc_gain(distance, irradiance_angle_deg,
                                     incidence_angle_deg, params)


def electrical_snr(distance: np.ndarray, irradiance_angle_deg: float = 0.0,
                   incidence_angle_deg: float = 0.0,
                   params: IndoorLinkParams | None = None) -> np.ndarray:
    """Electrical SNR (linear) of the received OOK signal."""
    params = params or IndoorLinkParams()
    p_rx = received_power(distance, irradiance_angle_deg, incidence_angle_deg, params)
    i_sig = params.responsivity * p_rx

    # Shot noise (signal + background) and thermal (Johnson) noise.
    shot = 2 * Q * (i_sig + params.background_current) * params.bandwidth
    thermal = (4 * K_BOLTZMANN * params.temperature / params.tia_resistance
               * params.bandwidth)
    noise_var = shot + thermal
    return i_sig ** 2 / noise_var


def electrical_snr_db(distance: np.ndarray, **kwargs) -> np.ndarray:
    snr = electrical_snr(distance, **kwargs)
    return 10 * np.log10(np.clip(snr, 1e-12, None))


def plot_channel(params: IndoorLinkParams | None = None) -> pathlib.Path:
    params = params or IndoorLinkParams()
    d = np.linspace(0.5, 5.0, 200)
    p_rx = received_power(d, params=params)
    snr_db = electrical_snr_db(d, params=params)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(d, p_rx * 1e3)
    ax1.set_xlabel("Distance (m)")
    ax1.set_ylabel("Received optical power (mW)")
    ax1.set_title("Indoor VLC received power vs distance")
    ax1.grid(alpha=0.3)

    ax2.plot(d, snr_db, color="tab:red")
    ax2.set_xlabel("Distance (m)")
    ax2.set_ylabel("Electrical SNR (dB)")
    ax2.set_title("Indoor VLC SNR vs distance")
    ax2.grid(alpha=0.3)
    return save_fig(fig, "indoor_channel.png")


def main() -> None:
    params = IndoorLinkParams()
    print("Indoor VLC Lambertian channel")
    print(f"  Lambertian order m : {lambertian_order(params.half_power_angle_deg):.2f}")
    for d in (1.0, 2.5, 5.0):
        p = float(received_power(np.array(d), params=params))
        snr = float(electrical_snr_db(np.array(d), params=params))
        print(f"  d={d:.1f} m : Prx={p * 1e3:.4f} mW, SNR={snr:.1f} dB")
    path = plot_channel(params)
    print(f"Saved channel figure -> {path}")


if __name__ == "__main__":
    main()
