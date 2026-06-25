"""Optical link budget for the outdoor V2X EV-to-charger VLC link.

Computes transmitted/received optical power, the SNR delivered by the channel,
and the link margin against the SNR required to hit a target BER over 5-30 m.
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dataclasses import dataclass

import numpy as np

import matplotlib.pyplot as plt

from common.io import save_fig
from vlc.v2x.channel_model_outdoor import (
    OutdoorLinkParams,
    electrical_snr_db,
    received_power,
)


def watt_to_dbm(power_w: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.clip(power_w, 1e-15, None) * 1e3)


@dataclass
class LinkBudget:
    distance: np.ndarray
    tx_dbm: float
    rx_dbm: np.ndarray
    snr_db: np.ndarray
    required_snr_db: float
    margin_db: np.ndarray

    @property
    def max_range(self) -> float:
        ok = self.distance[self.margin_db >= 0]
        return float(ok.max()) if ok.size else float("nan")


def compute_link_budget(required_snr_db: float = 10.0,
                        params: OutdoorLinkParams | None = None,
                        d_min: float = 5.0, d_max: float = 30.0) -> LinkBudget:
    params = params or OutdoorLinkParams()
    d = np.linspace(d_min, d_max, 200)
    rx_dbm = watt_to_dbm(received_power(d, params=params))
    snr_db = electrical_snr_db(d, params=params)
    margin = snr_db - required_snr_db
    return LinkBudget(d, float(watt_to_dbm(np.array(params.tx_power))),
                      rx_dbm, snr_db, required_snr_db, margin)


def plot_link_budget(lb: LinkBudget) -> pathlib.Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(lb.distance, lb.rx_dbm, label="Received power")
    ax1.axhline(lb.tx_dbm, color="gray", ls=":", label=f"Tx = {lb.tx_dbm:.0f} dBm")
    ax1.set_xlabel("Distance (m)")
    ax1.set_ylabel("Optical power (dBm)")
    ax1.set_title("Optical power vs distance")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(lb.distance, lb.margin_db, color="tab:purple", label="Link margin")
    ax2.axhline(0, color="r", ls="--", label="0 dB (sensitivity)")
    ax2.set_xlabel("Distance (m)")
    ax2.set_ylabel("Margin above required SNR (dB)")
    ax2.set_title(f"Link margin (req. SNR = {lb.required_snr_db:.0f} dB)")
    ax2.grid(alpha=0.3)
    ax2.legend()
    return save_fig(fig, "link_budget.png")


def main() -> None:
    lb = compute_link_budget(required_snr_db=10.0)
    print("V2X EV-charger optical link budget")
    print(f"  Tx power              : {lb.tx_dbm:.1f} dBm")
    for d in (5.0, 15.0, 30.0):
        i = int(np.argmin(np.abs(lb.distance - d)))
        print(f"  d={d:5.1f} m : Prx={lb.rx_dbm[i]:6.1f} dBm, "
              f"SNR={lb.snr_db[i]:5.1f} dB, margin={lb.margin_db[i]:5.1f} dB")
    print(f"  max usable range      : {lb.max_range:.1f} m")
    path = plot_link_budget(lb)
    print(f"Saved link-budget figure -> {path}")


if __name__ == "__main__":
    main()
