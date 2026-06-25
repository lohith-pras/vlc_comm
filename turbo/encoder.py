"""Parallel-concatenated turbo encoder (PCCC).

Two identical recursive systematic convolutional (RSC) encoders, constraint
length K=3 (memory M=2), generator polynomials (7,5) in octal with feedback 7.
Encoder 1 is trellis-terminated; encoder 2 runs over the interleaved info bits
without termination. Overall code is rate ~1/3.
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dataclasses import dataclass

import numpy as np

# --- RSC definition: generators (7,5) octal, feedback = 7 ---------------------
MEMORY = 2
N_STATES = 1 << MEMORY  # 4 states


def _rsc_step(state: int, u: int) -> tuple[int, int]:
    """One RSC step. ``state`` packs (s1<<1)|s2 with s1 the most-recent bit.

    Feedback polynomial 7 = 1+D+D^2 -> a = u ^ s1 ^ s2.
    Feedforward parity 5 = 1+D^2   -> p = a ^ s2.
    """
    s1, s2 = state >> 1, state & 1
    a = u ^ s1 ^ s2
    p = a ^ s2
    next_state = ((a << 1) | s1) & (N_STATES - 1)
    return next_state, p


def build_trellis() -> tuple[np.ndarray, np.ndarray]:
    """Return (next_state, parity) lookup tables indexed [state, input]."""
    next_state = np.zeros((N_STATES, 2), dtype=np.int64)
    parity = np.zeros((N_STATES, 2), dtype=np.int64)
    for s in range(N_STATES):
        for u in (0, 1):
            ns, p = _rsc_step(s, u)
            next_state[s, u] = ns
            parity[s, u] = p
    return next_state, parity


NEXT_STATE, PARITY = build_trellis()


def rsc_encode(bits: np.ndarray, terminate: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode *bits* through the RSC.

    Returns (systematic, parity, tail_inputs). When ``terminate`` is True, MEMORY
    extra steps drive the encoder back to the zero state; the forced input bits
    are appended to the systematic/parity streams and returned as ``tail_inputs``.
    """
    state = 0
    sysb: list[int] = []
    par: list[int] = []
    for u in bits.astype(int):
        ns, p = _rsc_step(state, int(u))
        sysb.append(int(u))
        par.append(p)
        state = ns
    tail: list[int] = []
    if terminate:
        for _ in range(MEMORY):
            s1, s2 = state >> 1, state & 1
            u = s1 ^ s2  # forces feedback a = 0 -> drives toward zero state
            ns, p = _rsc_step(state, u)
            sysb.append(u)
            par.append(p)
            tail.append(u)
            state = ns
    return np.array(sysb, dtype=np.int64), np.array(par, dtype=np.int64), np.array(tail, dtype=np.int64)


class Interleaver:
    """Fixed pseudo-random interleaver of size K (reproducible via seed)."""

    def __init__(self, size: int, seed: int = 0) -> None:
        self.size = size
        rng = np.random.default_rng(seed)
        self.perm = rng.permutation(size)
        self.inv = np.argsort(self.perm)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x[self.perm]

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x[self.inv]


@dataclass
class TurboCodeword:
    """Encoder output. ``systematic``/``parity1`` have length K+MEMORY (terminated);
    ``parity2`` has length K (encoder 2 unterminated)."""

    systematic: np.ndarray  # info bits + enc1 tail
    parity1: np.ndarray
    parity2: np.ndarray
    info_len: int
    interleaver: Interleaver

    @property
    def code_rate(self) -> float:
        n = len(self.systematic) + len(self.parity1) + len(self.parity2)
        return self.info_len / n


def turbo_encode(bits: np.ndarray, interleaver: Interleaver | None = None) -> TurboCodeword:
    """Turbo-encode an info-bit vector. Creates a default interleaver if none given."""
    bits = np.asarray(bits, dtype=np.int64)
    k = len(bits)
    if interleaver is None:
        interleaver = Interleaver(k)
    if interleaver.size != k:
        raise ValueError(f"interleaver size {interleaver.size} != info length {k}")

    sys1, par1, _tail = rsc_encode(bits, terminate=True)
    bits_i = interleaver.forward(bits)
    _sys2, par2, _ = rsc_encode(bits_i, terminate=False)
    return TurboCodeword(systematic=sys1, parity1=par1, parity2=par2,
                         info_len=k, interleaver=interleaver)


def main() -> None:
    rng = np.random.default_rng(42)
    k = 100
    bits = rng.integers(0, 2, k)
    cw = turbo_encode(bits)
    print("Turbo encoder (PCCC, RSC (7,5) octal, M=2)")
    print(f"  info bits K           : {k}")
    print(f"  systematic len (K+M)  : {len(cw.systematic)}")
    print(f"  parity1 len (K+M)     : {len(cw.parity1)}")
    print(f"  parity2 len (K)       : {len(cw.parity2)}")
    total = len(cw.systematic) + len(cw.parity1) + len(cw.parity2)
    print(f"  total coded bits      : {total}")
    print(f"  effective code rate   : {cw.code_rate:.4f}  (~1/3)")
    # Confirm encoder 1 terminates in the zero state.
    _, _, tail = rsc_encode(bits, terminate=True)
    print(f"  enc1 tail input bits  : {tail.tolist()}")


if __name__ == "__main__":
    main()
