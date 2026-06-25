"""Iterative turbo decoder: two soft-input/soft-output BCJR (Log-MAP) decoders
exchanging extrinsic LLRs through the interleaver.

LLR convention: L = log( P(bit=0) / P(bit=1) ), so a positive LLR favours bit 0.
BPSK map used elsewhere: x = 1 - 2*bit (bit 0 -> +1, bit 1 -> -1), channel LLR
for an AWGN sample y is 2*y/sigma^2.
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from turbo.encoder import (
    MEMORY,
    NEXT_STATE,
    N_STATES,
    PARITY,
    Interleaver,
    TurboCodeword,
    turbo_encode,
)

_NEG = -1e9


def _max_star(a: np.ndarray, b: np.ndarray, log_map: bool) -> np.ndarray:
    """Elementwise max* operator. Log-MAP adds the Jacobian correction term."""
    m = np.maximum(a, b)
    if not log_map:
        return m
    return m + np.log1p(np.exp(-np.abs(a - b)))


def bcjr(sys_llr: np.ndarray, par_llr: np.ndarray, apriori: np.ndarray,
         terminated: bool, log_map: bool = True) -> np.ndarray:
    """Single SISO BCJR pass. Returns the a-posteriori LLR for every position.

    All inputs are arrays of the same length L. ``apriori`` is the a-priori LLR
    on the (systematic) bits.
    """
    length = len(sys_llr)

    # Branch metric gamma[k, s, u] = 0.5*(1-2u)*(La+Lsys) + 0.5*(1-2c)*Lpar.
    sign_u = np.array([1.0, -1.0])  # (1-2u) for u in {0,1}
    parity_sign = 1.0 - 2.0 * PARITY  # shape (N_STATES, 2)
    info_term = 0.5 * (apriori + sys_llr)  # length L

    # gamma per position: shape (L, N_STATES, 2)
    gamma = (info_term[:, None, None] * sign_u[None, None, :]
             + 0.5 * par_llr[:, None, None] * parity_sign[None, :, :])

    # Forward recursion (alpha).
    alpha = np.full((length + 1, N_STATES), _NEG)
    alpha[0, 0] = 0.0
    for k in range(length):
        nxt = np.full(N_STATES, _NEG)
        for s in range(N_STATES):
            for u in (0, 1):
                ns = NEXT_STATE[s, u]
                cand = alpha[k, s] + gamma[k, s, u]
                nxt[ns] = _max_star(np.array(nxt[ns]), np.array(cand), log_map)
        alpha[k + 1] = nxt - np.max(nxt)  # normalise for stability

    # Backward recursion (beta).
    beta = np.full((length + 1, N_STATES), _NEG)
    if terminated:
        beta[length, 0] = 0.0
    else:
        beta[length, :] = 0.0
    for k in range(length - 1, -1, -1):
        cur = np.full(N_STATES, _NEG)
        for s in range(N_STATES):
            for u in (0, 1):
                ns = NEXT_STATE[s, u]
                cand = beta[k + 1, ns] + gamma[k, s, u]
                cur[s] = _max_star(np.array(cur[s]), np.array(cand), log_map)
        beta[k] = cur - np.max(cur)

    # A-posteriori LLR per position: max* over u=0 transitions minus u=1.
    posterior = np.zeros(length)
    for k in range(length):
        m0 = _NEG
        m1 = _NEG
        for s in range(N_STATES):
            for u in (0, 1):
                ns = NEXT_STATE[s, u]
                val = alpha[k, s] + gamma[k, s, u] + beta[k + 1, ns]
                if u == 0:
                    m0 = float(_max_star(np.array(m0), np.array(val), log_map))
                else:
                    m1 = float(_max_star(np.array(m1), np.array(val), log_map))
        posterior[k] = m0 - m1
    return posterior


def turbo_decode(sys_llr: np.ndarray, par1_llr: np.ndarray, par2_llr: np.ndarray,
                 interleaver: Interleaver, iterations: int = 6,
                 log_map: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively decode. ``sys_llr``/``par1_llr`` have length K+MEMORY (terminated
    encoder 1); ``par2_llr`` has length K. Returns (decoded_bits, final_info_llr)."""
    k = interleaver.size
    le_dec1_to_2 = np.zeros(k)  # extrinsic from decoder 1 -> decoder 2 (info order)
    le_dec2_to_1 = np.zeros(k)  # extrinsic from decoder 2 -> decoder 1 (info order)

    posterior2 = np.zeros(k)
    for _ in range(iterations):
        # --- Decoder 1 (terminated, length K+MEMORY) ---
        apriori1 = np.zeros(len(sys_llr))
        apriori1[:k] = le_dec2_to_1
        post1 = bcjr(sys_llr, par1_llr, apriori1, terminated=True, log_map=log_map)
        le_dec1_to_2 = post1[:k] - apriori1[:k] - sys_llr[:k]

        # --- Decoder 2 (unterminated, length K, interleaved domain) ---
        sys_llr2 = interleaver.forward(sys_llr[:k])
        apriori2 = interleaver.forward(le_dec1_to_2)
        post2 = bcjr(sys_llr2, par2_llr, apriori2, terminated=False, log_map=log_map)
        le_dec2_to_1 = interleaver.inverse(post2 - apriori2 - sys_llr2)
        posterior2 = post2

    final_llr = interleaver.inverse(posterior2)
    decoded = (final_llr < 0).astype(np.int64)  # LLR>0 -> bit 0
    return decoded, final_llr


def _bits_to_llr(bits: np.ndarray, sigma2: float, rng: np.random.Generator) -> np.ndarray:
    """BPSK-modulate bits, add AWGN, return channel LLR (= 2y/sigma^2)."""
    x = 1.0 - 2.0 * bits  # bit0 -> +1, bit1 -> -1
    y = x + rng.normal(0.0, np.sqrt(sigma2), size=x.shape)
    return 2.0 * y / sigma2


def main() -> None:
    rng = np.random.default_rng(7)
    k = 400
    bits = rng.integers(0, 2, k)
    cw = turbo_encode(bits)

    # Transmit over BPSK/AWGN at a moderate Eb/N0.
    ebn0_db = 1.0
    rate = cw.code_rate
    ebn0 = 10 ** (ebn0_db / 10)
    sigma2 = 1.0 / (2.0 * rate * ebn0)

    sys_llr = _bits_to_llr(cw.systematic, sigma2, rng)
    par1_llr = _bits_to_llr(cw.parity1, sigma2, rng)
    par2_llr = _bits_to_llr(cw.parity2, sigma2, rng)

    # Hard-decision (uncoded-style) errors on the systematic bits for reference.
    raw_errors = int(np.sum((sys_llr[:k] < 0).astype(int) != bits))
    print(f"Turbo decoder (Log-MAP BCJR), Eb/N0 = {ebn0_db} dB, K = {k}")
    print(f"  raw systematic errors before decoding : {raw_errors}")
    for iters in (1, 2, 4, 6, 8):
        decoded, _ = turbo_decode(sys_llr, par1_llr, par2_llr, cw.interleaver,
                                  iterations=iters)
        errs = int(np.sum(decoded != bits))
        print(f"  iterations = {iters:2d} -> bit errors = {errs:4d}  "
              f"(BER = {errs / k:.4e})")


if __name__ == "__main__":
    main()
