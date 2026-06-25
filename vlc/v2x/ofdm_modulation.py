"""DCO-OFDM for outdoor V2X VLC (real, non-negative intensity signal).

QPSK-mapped data subcarriers, Hermitian symmetry so the IFFT output is real, a
DC bias plus zero-clipping (Direct-Current-biased Optical OFDM), and a cyclic
prefix. QPSK keeps per-bit LLRs simple (I/Q each behave like BPSK), which feeds
the turbo decoder in the V2X simulation.
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from dataclasses import dataclass

import numpy as np

import matplotlib.pyplot as plt

from common.io import save_fig

_SQRT2 = np.sqrt(2.0)


def qpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """Gray-mapped QPSK. 2 bits/symbol -> unit-energy complex symbol."""
    bits = np.asarray(bits, dtype=int)
    if len(bits) % 2 != 0:
        raise ValueError("QPSK needs an even number of bits")
    b = bits.reshape(-1, 2)
    # bit 0 -> +1, bit 1 -> -1 (matches L = log P0/P1 convention)
    i = (1 - 2 * b[:, 0]) / _SQRT2
    qd = (1 - 2 * b[:, 1]) / _SQRT2
    return i + 1j * qd


def qpsk_llr(symbols: np.ndarray, noise_var: float, scale: float = 1.0
             ) -> np.ndarray:
    """Per-bit LLR (log P0/P1) for QPSK over AWGN. Returns 2 LLRs per symbol."""
    # For Gray QPSK the I and Q components are independent BPSK.
    # mean component magnitude = scale/sqrt2; LLR = 2*scale/sqrt2 * comp / noise_var
    factor = 2.0 * (scale / _SQRT2) / noise_var
    llr = np.empty(2 * len(symbols))
    llr[0::2] = factor * symbols.real
    llr[1::2] = factor * symbols.imag
    return llr


def qpsk_demodulate_hard(symbols: np.ndarray) -> np.ndarray:
    bits = np.empty(2 * len(symbols), dtype=np.int64)
    bits[0::2] = (symbols.real < 0).astype(int)
    bits[1::2] = (symbols.imag < 0).astype(int)
    return bits


@dataclass
class OfdmParams:
    n_fft: int = 64
    cp_len: int = 8
    bias_db: float = 7.0  # DC bias relative to signal std


@dataclass
class OfdmFrame:
    signal: np.ndarray         # real, non-negative time-domain samples
    n_symbols: int             # number of OFDM symbols
    bias: float
    data_carriers: np.ndarray  # indices of data subcarriers
    bits_per_frame: int


def _data_carrier_indices(n_fft: int) -> np.ndarray:
    """Subcarriers 1..N/2-1 carry data (0 and N/2 are nulled for a real signal)."""
    return np.arange(1, n_fft // 2)


def dco_ofdm_modulate(bits: np.ndarray, params: OfdmParams | None = None) -> OfdmFrame:
    params = params or OfdmParams()
    carriers = _data_carrier_indices(params.n_fft)
    n_data = len(carriers)
    bits_per_symbol = 2 * n_data  # QPSK

    pad = (-len(bits)) % bits_per_symbol
    bits_p = np.concatenate([bits, np.zeros(pad, dtype=int)])
    n_symbols = len(bits_p) // bits_per_symbol

    time_blocks = []
    for s in range(n_symbols):
        chunk = bits_p[s * bits_per_symbol:(s + 1) * bits_per_symbol]
        qsym = qpsk_modulate(chunk)
        spectrum = np.zeros(params.n_fft, dtype=complex)
        spectrum[carriers] = qsym
        # Hermitian symmetry -> real IFFT output.
        spectrum[params.n_fft - carriers] = np.conj(qsym)
        t = np.fft.ifft(spectrum) * np.sqrt(params.n_fft)
        time_blocks.append(t.real)

    time_signal = np.concatenate(time_blocks) if time_blocks else np.array([])

    # DC bias + zero clipping (DCO-OFDM).
    sigma = np.std(time_signal) if time_signal.size else 1.0
    bias = sigma * 10 ** (params.bias_db / 20)
    biased = time_signal.reshape(n_symbols, params.n_fft) + bias
    biased = np.clip(biased, 0.0, None)

    # Add cyclic prefix per symbol.
    with_cp = np.concatenate([biased[:, -params.cp_len:], biased], axis=1)
    return OfdmFrame(signal=with_cp.reshape(-1), n_symbols=n_symbols, bias=bias,
                     data_carriers=carriers, bits_per_frame=bits_per_symbol * n_symbols)


def dco_ofdm_demodulate(signal: np.ndarray, frame: OfdmFrame,
                        params: OfdmParams | None = None,
                        noise_var: float = 1e-6, soft: bool = True) -> np.ndarray:
    """Recover bits (hard) or per-bit LLRs (soft) from a received DCO-OFDM signal."""
    params = params or OfdmParams()
    sym_len = params.n_fft + params.cp_len
    blocks = signal.reshape(frame.n_symbols, sym_len)
    no_cp = blocks[:, params.cp_len:]

    out = []
    for s in range(frame.n_symbols):
        spectrum = np.fft.fft(no_cp[s]) / np.sqrt(params.n_fft)
        qsym = spectrum[frame.data_carriers]
        if soft:
            out.append(qpsk_llr(qsym, noise_var))
        else:
            out.append(qpsk_demodulate_hard(qsym))
    return np.concatenate(out) if out else np.array([])


def papr_db(signal: np.ndarray) -> float:
    """Peak-to-average power ratio (dB) of the time-domain signal."""
    power = signal ** 2
    return 10 * np.log10(np.max(power) / np.mean(power))


def plot_ofdm(params: OfdmParams | None = None) -> pathlib.Path:
    params = params or OfdmParams()
    rng = np.random.default_rng(0)
    n_bits = 2 * len(_data_carrier_indices(params.n_fft)) * 12
    bits = rng.integers(0, 2, n_bits)
    frame = dco_ofdm_modulate(bits, params)

    # Clean round-trip constellation (small noise) for the scatter plot.
    sym_len = params.n_fft + params.cp_len
    noisy = frame.signal + rng.normal(0, 0.02 * np.std(frame.signal), frame.signal.shape)
    blocks = noisy.reshape(frame.n_symbols, sym_len)[:, params.cp_len:]
    rx_syms = []
    for s in range(frame.n_symbols):
        spec = np.fft.fft(blocks[s]) / np.sqrt(params.n_fft)
        rx_syms.append(spec[frame.data_carriers])
    rx_syms = np.concatenate(rx_syms)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(frame.signal[:3 * sym_len])
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_title(f"DCO-OFDM time signal (PAPR={papr_db(frame.signal):.1f} dB)")
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("Intensity (biased, clipped >= 0)")
    ax1.grid(alpha=0.3)

    ax2.scatter(rx_syms.real, rx_syms.imag, s=8, alpha=0.4)
    ax2.set_title("Recovered QPSK constellation")
    ax2.set_xlabel("In-phase")
    ax2.set_ylabel("Quadrature")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.axvline(0, color="k", lw=0.5)
    ax2.grid(alpha=0.3)
    return save_fig(fig, "ofdm.png")


def main() -> None:
    params = OfdmParams()
    rng = np.random.default_rng(1)
    n_bits = 2 * len(_data_carrier_indices(params.n_fft)) * 10
    bits = rng.integers(0, 2, n_bits)
    frame = dco_ofdm_modulate(bits, params)

    # Noiseless round trip.
    recovered = dco_ofdm_demodulate(frame.signal, frame, params, soft=False)
    n = min(len(bits), len(recovered))
    errs = int(np.sum(bits[:n] != recovered[:n]))
    print("DCO-OFDM (QPSK, Hermitian symmetry)")
    print(f"  N_FFT={params.n_fft}, CP={params.cp_len}, data carriers="
          f"{len(frame.data_carriers)}")
    print(f"  OFDM symbols          : {frame.n_symbols}")
    print(f"  DC bias               : {frame.bias:.3f}")
    print(f"  PAPR                  : {papr_db(frame.signal):.2f} dB")
    print(f"  noiseless round-trip errors (incl. clipping): {errs}/{n}")
    path = plot_ofdm(params)
    print(f"Saved OFDM figure -> {path}")


if __name__ == "__main__":
    main()
