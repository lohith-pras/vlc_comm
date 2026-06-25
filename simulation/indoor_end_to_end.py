"""End-to-end indoor VLC link: turbo + OOK + Lambertian channel.

Chain: info bits -> turbo encode -> OOK -> optical channel + receiver noise ->
OOK soft-LLR demod -> turbo decode -> BER. Compares coded vs uncoded at equal
per-symbol electrical SNR (peak-power-limited IM/DD framing: the LED on/off
levels are fixed, so coding spends throughput, not per-symbol SNR). A second
panel maps the channel SNR to physical distance to read off the usable range.
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

import matplotlib.pyplot as plt

from common.io import save_fig
from turbo.decoder import turbo_decode
from turbo.encoder import Interleaver, turbo_encode
from vlc.indoor.channel_model import IndoorLinkParams, electrical_snr_db
from vlc.indoor.modulation_ook import (
    ook_awgn_channel,
    ook_demodulate_hard,
    ook_demodulate_llr,
    ook_modulate,
)


def _snr_to_noise_std(snr_db: float) -> float:
    """Receiver-noise std for unit OOK level separation at the given SNR."""
    return 1.0 / np.sqrt(10 ** (snr_db / 10))


def coded_ber(snr_db: float, k: int, n_blocks: int, iterations: int,
              interleaver: Interleaver, rng: np.random.Generator) -> float:
    noise_std = _snr_to_noise_std(snr_db)
    bit_errors = 0
    total = 0
    for _ in range(n_blocks):
        bits = rng.integers(0, 2, k)
        cw = turbo_encode(bits, interleaver)

        def channel_llr(coded_bits: np.ndarray) -> np.ndarray:
            rx = ook_awgn_channel(ook_modulate(coded_bits), noise_std, 1.0, rng)
            return ook_demodulate_llr(rx, noise_std, gain=1.0)

        sys_llr = channel_llr(cw.systematic)
        par1_llr = channel_llr(cw.parity1)
        par2_llr = channel_llr(cw.parity2)
        decoded, _ = turbo_decode(sys_llr, par1_llr, par2_llr, interleaver,
                                  iterations=iterations)
        bit_errors += int(np.sum(decoded != bits))
        total += k
    return bit_errors / total


def uncoded_ber(snr_db: float, n_bits: int, rng: np.random.Generator) -> float:
    noise_std = _snr_to_noise_std(snr_db)
    bits = rng.integers(0, 2, n_bits)
    rx = ook_awgn_channel(ook_modulate(bits), noise_std, 1.0, rng)
    decoded = ook_demodulate_hard(rx)
    return float(np.mean(decoded != bits))


def run(snr_db_list: np.ndarray, k: int = 512, n_blocks: int = 12,
        iterations: int = 6) -> tuple[list[float], list[float]]:
    interleaver = Interleaver(k, seed=3)
    coded, uncoded = [], []
    for i, snr in enumerate(snr_db_list):
        rng = np.random.default_rng(500 + i)
        c = coded_ber(float(snr), k, n_blocks, iterations, interleaver, rng)
        u = uncoded_ber(float(snr), k * n_blocks, rng)
        coded.append(c)
        uncoded.append(u)
        print(f"  SNR={snr:4.1f} dB  coded BER={c:.3e}  uncoded BER={u:.3e}")
    return coded, uncoded


def plot(snr_db_list: np.ndarray, coded: list[float], uncoded: list[float]
         ) -> pathlib.Path:
    params = IndoorLinkParams()
    d = np.linspace(0.5, 5.0, 200)
    snr_d = electrical_snr_db(d, params=params)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.semilogy(snr_db_list, np.clip(uncoded, 1e-6, 1), "s--", label="Uncoded OOK")
    ax1.semilogy(snr_db_list, np.clip(coded, 1e-6, 1), "o-", label="Turbo-coded OOK")
    ax1.set_xlabel("Per-symbol electrical SNR (dB)")
    ax1.set_ylabel("BER")
    ax1.set_title("Indoor VLC: coded vs uncoded")
    ax1.set_ylim(1e-6, 1)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    ax2.plot(d, snr_d, color="tab:green")
    ax2.set_xlabel("Distance (m)")
    ax2.set_ylabel("Electrical SNR (dB)")
    ax2.set_title("Channel SNR vs distance (link budget)")
    ax2.grid(alpha=0.3)
    return save_fig(fig, "indoor_e2e_ber.png")


def main() -> None:
    snr_db_list = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    print("Indoor end-to-end VLC (turbo + OOK)")
    coded, uncoded = run(snr_db_list)
    path = plot(snr_db_list, coded, uncoded)
    print(f"Saved end-to-end figure -> {path}")


if __name__ == "__main__":
    main()
