"""Monte-Carlo BER vs Eb/N0 for the turbo code over BPSK/AWGN.

Shows the coded waterfall for several decoder-iteration counts against the
analytic uncoded BPSK reference, illustrating turbo iterative gain.
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dataclasses import dataclass

import numpy as np
from scipy.special import erfc

import matplotlib.pyplot as plt

from common.io import save_fig
from turbo.decoder import _bits_to_llr, turbo_decode
from turbo.encoder import Interleaver, turbo_encode


def q_function(x: np.ndarray) -> np.ndarray:
    """Gaussian Q-function via the complementary error function."""
    return 0.5 * erfc(x / np.sqrt(2.0))


def uncoded_bpsk_ber(ebn0_db: np.ndarray) -> np.ndarray:
    """Analytic uncoded BPSK BER over AWGN."""
    ebn0 = 10 ** (ebn0_db / 10)
    return q_function(np.sqrt(2.0 * ebn0))


@dataclass
class BerPoint:
    ebn0_db: float
    iterations: int
    ber: float
    bit_errors: int
    bits: int


def simulate_point(ebn0_db: float, iterations: int, k: int, n_blocks: int,
                   seed: int) -> BerPoint:
    """Simulate one (Eb/N0, iterations) operating point."""
    rng = np.random.default_rng(seed)
    interleaver = Interleaver(k, seed=seed)
    ebn0 = 10 ** (ebn0_db / 10)

    bit_errors = 0
    total_bits = 0
    for _ in range(n_blocks):
        bits = rng.integers(0, 2, k)
        cw = turbo_encode(bits, interleaver)
        sigma2 = 1.0 / (2.0 * cw.code_rate * ebn0)
        sys_llr = _bits_to_llr(cw.systematic, sigma2, rng)
        par1_llr = _bits_to_llr(cw.parity1, sigma2, rng)
        par2_llr = _bits_to_llr(cw.parity2, sigma2, rng)
        decoded, _ = turbo_decode(sys_llr, par1_llr, par2_llr, interleaver,
                                  iterations=iterations)
        bit_errors += int(np.sum(decoded != bits))
        total_bits += k
    return BerPoint(ebn0_db, iterations, bit_errors / total_bits, bit_errors, total_bits)


def run_ber_curves(ebn0_db: np.ndarray, iteration_list: list[int], k: int,
                   n_blocks: int) -> dict[int, list[BerPoint]]:
    """Run the full sweep. Returns {iterations: [BerPoint, ...]}."""
    results: dict[int, list[BerPoint]] = {}
    for iters in iteration_list:
        pts = []
        for i, ebn0 in enumerate(ebn0_db):
            pt = simulate_point(float(ebn0), iters, k, n_blocks, seed=1000 + i)
            pts.append(pt)
            print(f"  iters={iters:2d}  Eb/N0={ebn0:4.1f} dB  "
                  f"BER={pt.ber:.3e}  ({pt.bit_errors} errs / {pt.bits} bits)")
        results[iters] = pts
    return results


def plot_ber(results: dict[int, list[BerPoint]], ebn0_db: np.ndarray) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    fine = np.linspace(ebn0_db.min(), ebn0_db.max(), 200)
    ax.semilogy(fine, uncoded_bpsk_ber(fine), "k--", label="Uncoded BPSK (analytic)")
    for iters, pts in results.items():
        ber = np.array([max(p.ber, 1e-7) for p in pts])
        ax.semilogy(ebn0_db, ber, "o-", label=f"Turbo, {iters} iter")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Bit Error Rate")
    ax.set_title("Turbo code BER vs Eb/N0 (rate ~1/3, RSC (7,5), Log-MAP)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(1e-6, 1)
    ax.legend()
    return save_fig(fig, "turbo_ber.png")


def main() -> None:
    # Modest defaults keep the __main__ run fast; scale up for smoother curves.
    k = 512
    n_blocks = 12
    ebn0_db = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    iteration_list = [1, 3, 6]

    print(f"Turbo BER sweep: K={k}, blocks/point={n_blocks}")
    results = run_ber_curves(ebn0_db, iteration_list, k, n_blocks)
    path = plot_ber(results, ebn0_db)
    print(f"Saved BER curve -> {path}")


if __name__ == "__main__":
    main()
