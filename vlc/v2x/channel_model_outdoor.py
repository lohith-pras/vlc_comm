"""Outdoor V2X VLC channel: EV headlight/charger LED over 5-30 m.

Extends the Lambertian LOS model with atmospheric (Beer-Lambert) extinction and
a strong daylight-induced background photocurrent, which dominates the shot
noise outdoors. Produces received power and electrical SNR vs distance.
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from dataclasses import dataclass

import numpy as np

import matplotlib.pyplot as plt

from common.io import save_fig

Q = 1.602176634e-19
K_BOLTZMANN = 1.380649e-23


@dataclass
class OutdoorLinkParams:
    tx_power: float = 10.0          # high-power LED head/charger lamp (W)
    half_power_angle_deg: float = 20.0
    detector_area: float = 1e-3     # larger collecting lens (m^2)
    fov_deg: float = 45.0
    responsivity: float = 0.5       # A/W
    bandwidth: float = 5e6          # Hz
    background_current: float = 5e-2  # daylight-induced PD current (A) -- large
    temperature: float = 300.0
    tia_resistance: float = 1e4
    atmos_attenuation: float = 0.02  # extinction coefficient (1/m), clear-ish air
    concentrator_gain: float = 3.0   # optical concentrator + filter gain


def lambertian_order(half_power_angle_deg: float) -> float:
    phi_half = np.radians(half_power_angle_deg)
    return -np.log(2.0) / np.log(np.cos(phi_half))


def dc_gain(distance: np.ndarray, irradiance_angle_deg: float = 0.0,
            incidence_angle_deg: float = 0.0,
            params: OutdoorLinkParams | None = None) -> np.ndarray:
    """LOS optical DC gain including atmospheric extinction and concentrator gain."""
    params = params or OutdoorLinkParams()
    m = lambertian_order(params.half_power_angle_deg)
    phi = np.radians(irradiance_angle_deg)
    psi = np.radians(incidence_angle_deg)
    d = np.asarray(distance, dtype=float)
    geom = ((m + 1) * params.detector_area / (2 * np.pi * d ** 2)
            * np.cos(phi) ** m * np.cos(psi))
    atmos = np.exp(-params.atmos_attenuation * d)
    gain = geom * atmos * params.concentrator_gain
    inside_fov = psi <= np.radians(params.fov_deg)
    return np.where(inside_fov, np.clip(gain, 0.0, None), 0.0)


def received_power(distance: np.ndarray, irradiance_angle_deg: float = 0.0,
                   incidence_angle_deg: float = 0.0,
                   params: OutdoorLinkParams | None = None) -> np.ndarray:
    params = params or OutdoorLinkParams()
    return params.tx_power * dc_gain(distance, irradiance_angle_deg,
                                     incidence_angle_deg, params)


def electrical_snr(distance: np.ndarray, params: OutdoorLinkParams | None = None,
                   **angles) -> np.ndarray:
    params = params or OutdoorLinkParams()
    p_rx = received_power(distance, params=params, **angles)
    i_sig = params.responsivity * p_rx
    shot = 2 * Q * (i_sig + params.background_current) * params.bandwidth
    thermal = (4 * K_BOLTZMANN * params.temperature / params.tia_resistance
               * params.bandwidth)
    return i_sig ** 2 / (shot + thermal)


def electrical_snr_db(distance: np.ndarray, **kwargs) -> np.ndarray:
    return 10 * np.log10(np.clip(electrical_snr(distance, **kwargs), 1e-12, None))


def plot_channel(params: OutdoorLinkParams | None = None) -> pathlib.Path:
    params = params or OutdoorLinkParams()
    d = np.linspace(5.0, 30.0, 200)
    p_rx = received_power(d, params=params)
    snr_db = electrical_snr_db(d, params=params)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(d, p_rx * 1e6)
    ax1.set_xlabel("Distance (m)")
    ax1.set_ylabel("Received optical power (uW)")
    ax1.set_title("Outdoor V2X VLC received power")
    ax1.grid(alpha=0.3)

    ax2.plot(d, snr_db, color="tab:red")
    ax2.set_xlabel("Distance (m)")
    ax2.set_ylabel("Electrical SNR (dB)")
    ax2.set_title("Outdoor V2X VLC SNR (5-30 m)")
    ax2.grid(alpha=0.3)
    return save_fig(fig, "v2x_channel.png")


def main() -> None:
    params = OutdoorLinkParams()
    print("Outdoor V2X VLC channel (EV <-> charger)")
    print(f"  Lambertian order m : {lambertian_order(params.half_power_angle_deg):.2f}")
    for d in (5.0, 15.0, 30.0):
        p = float(received_power(np.array(d), params=params))
        snr = float(electrical_snr_db(np.array(d), params=params))
        print(f"  d={d:5.1f} m : Prx={p * 1e6:.3f} uW, SNR={snr:.1f} dB")
    path = plot_channel(params)
    print(f"Saved outdoor channel figure -> {path}")


if __name__ == "__main__":
    main()
