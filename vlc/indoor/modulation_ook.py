"""On-Off Keying (OOK) intensity modulation for indoor VLC.

OOK maps each bit to an LED intensity level (bit 0 -> high/on, bit 1 -> low/off
to stay consistent with the decoder LLR convention L = log P(0)/P(1)).
Provides hard-decision and soft-LLR demodulation.
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

import matplotlib.pyplot as plt

from common.io import save_fig

# Intensity levels for the two bit values.
LEVEL_BIT0 = 1.0  # LED on
LEVEL_BIT1 = 0.0  # LED off


def ook_modulate(bits: np.ndarray) -> np.ndarray:
    """Map bits to normalised optical intensity levels."""
    bits = np.asarray(bits, dtype=int)
    return np.where(bits == 0, LEVEL_BIT0, LEVEL_BIT1).astype(float)


def ook_demodulate_hard(samples: np.ndarray) -> np.ndarray:
    """Threshold detection at the midpoint of the two levels."""
    threshold = 0.5 * (LEVEL_BIT0 + LEVEL_BIT1)
    # Sample above threshold -> closer to LEVEL_BIT0 -> bit 0.
    return np.where(samples >= threshold, 0, 1).astype(np.int64)


def ook_demodulate_llr(samples: np.ndarray, noise_std: float,
                       gain: float = 1.0) -> np.ndarray:
    """Soft demod: LLR = log P(bit=0)/P(bit=1) for AWGN on the received samples.

    ``gain`` scales the nominal transmit levels to received levels. For Gaussian
    noise this reduces to a linear function of the received sample.
    """
    m0 = gain * LEVEL_BIT0
    m1 = gain * LEVEL_BIT1
    var = noise_std ** 2
    return ((samples - m1) ** 2 - (samples - m0) ** 2) / (2.0 * var)


def ook_awgn_channel(intensity: np.ndarray, noise_std: float, gain: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Apply optical gain and additive Gaussian receiver noise."""
    return gain * intensity + rng.normal(0.0, noise_std, size=intensity.shape)


def _demo_ber(snr_db_list: list[float], n_bits: int = 200_000,
              seed: int = 0) -> tuple[list[float], list[float]]:
    """Uncoded OOK BER vs electrical SNR (hard decision)."""
    rng = np.random.default_rng(seed)
    bers = []
    for snr_db in snr_db_list:
        bits = rng.integers(0, 2, n_bits)
        tx = ook_modulate(bits)
        snr = 10 ** (snr_db / 10)
        # Signal amplitude (level separation) = 1; set noise from SNR.
        noise_std = 1.0 / np.sqrt(snr)
        rx = ook_awgn_channel(tx, noise_std, gain=1.0, rng=rng)
        decoded = ook_demodulate_hard(rx)
        bers.append(float(np.mean(decoded != bits)))
    return snr_db_list, bers


def plot_ook(snr_db_list: list[float], bers: list[float]) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.semilogy(snr_db_list, np.clip(bers, 1e-6, 1), "o-")
    ax.set_xlabel("Electrical SNR (dB)")
    ax.set_ylabel("Uncoded OOK BER")
    ax.set_title("OOK BER vs SNR (hard decision)")
    ax.grid(True, which="both", alpha=0.3)
    return save_fig(fig, "ook_ber.png")


def main() -> None:
    print("OOK modulation (bit0 -> on, bit1 -> off)")
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    tx = ook_modulate(bits)
    print(f"  bits      : {bits.tolist()}")
    print(f"  intensity : {tx.tolist()}")
    llr = ook_demodulate_llr(tx, noise_std=0.3)
    print(f"  clean LLR sign matches bits : {np.array_equal((llr < 0).astype(int), bits)}")

    snr_list = [0.0, 3.0, 6.0, 9.0, 12.0, 15.0]
    snr_list, bers = _demo_ber(snr_list)
    for s, b in zip(snr_list, bers):
        print(f"  SNR={s:4.1f} dB -> BER={b:.3e}")
    path = plot_ook(snr_list, bers)
    print(f"Saved OOK figure -> {path}")


if __name__ == "__main__":
    main()
