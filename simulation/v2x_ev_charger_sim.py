"""End-to-end V2X EV-charger VLC simulation.

Chain: ISO 15118 message bytes -> bits -> turbo encode -> DCO-OFDM (QPSK) ->
outdoor channel SNR + Doppler/ICI -> soft demod -> turbo decode -> reconstruct
messages. Reports coded vs uncoded BER and message-frame success vs distance,
plus BER vs relative velocity at a marginal range.
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dataclasses import dataclass

import numpy as np

import matplotlib.pyplot as plt

from common.io import save_fig
from ev_charger.iso15118_mock import (
    build_session_sequence,
    deserialize_sequence,
    serialize_sequence,
)
from turbo.decoder import turbo_decode
from turbo.encoder import Interleaver, turbo_encode
from vlc.v2x.channel_model_outdoor import OutdoorLinkParams, electrical_snr
from vlc.v2x.doppler_model import ici_sinr_db, normalized_cfo
from vlc.v2x.ofdm_modulation import (
    OfdmParams,
    dco_ofdm_demodulate,
    dco_ofdm_modulate,
    qpsk_demodulate_hard,
)

# Lower-power link so the 5-30 m sweep crosses the decoding waterfall.
SIM_CHANNEL = OutdoorLinkParams(tx_power=0.11)
EQUIV_CARRIER_HZ = 1e9
SUBCARRIER_SPACING_HZ = 78.125e3


def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    n = (len(bits) // 8) * 8
    return np.packbits(bits[:n].astype(np.uint8)).tobytes()


def combined_sinr(distance: float, velocity: float) -> float:
    """Channel SNR combined with Doppler-ICI SINR (parallel noise sources)."""
    snr = float(electrical_snr(np.array(distance), params=SIM_CHANNEL))
    eps = normalized_cfo(velocity, EQUIV_CARRIER_HZ, SUBCARRIER_SPACING_HZ)
    ici_db = ici_sinr_db(eps)
    if np.isinf(ici_db):
        return snr
    ici = 10 ** (ici_db / 10)
    return 1.0 / (1.0 / snr + 1.0 / ici)


@dataclass
class LinkResult:
    coded_ber: float
    uncoded_ber: float
    messages_ok: bool


def _ofdm_roundtrip_llr(coded_bits: np.ndarray, sinr_lin: float,
                        ofdm: OfdmParams, rng: np.random.Generator) -> np.ndarray:
    """OFDM-modulate, add AWGN at the target SINR, return per-coded-bit LLRs."""
    frame = dco_ofdm_modulate(coded_bits, ofdm)
    noise_std = np.sqrt(1.0 / sinr_lin)
    rx = frame.signal + rng.normal(0.0, noise_std, frame.signal.shape)
    llr = dco_ofdm_demodulate(rx, frame, ofdm, noise_var=1.0 / sinr_lin, soft=True)
    return llr[:len(coded_bits)]


def simulate_link(info_bits: np.ndarray, distance: float, velocity: float,
                  ofdm: OfdmParams, iterations: int, rng: np.random.Generator
                  ) -> LinkResult:
    sinr_lin = combined_sinr(distance, velocity)
    k = len(info_bits)
    interleaver = Interleaver(k, seed=11)

    # --- Coded path ---
    cw = turbo_encode(info_bits, interleaver)
    n_sys = len(cw.systematic)
    coded = np.concatenate([cw.systematic, cw.parity1, cw.parity2])
    llr = _ofdm_roundtrip_llr(coded, sinr_lin, ofdm, rng)
    sys_llr = llr[:n_sys]
    par1_llr = llr[n_sys:2 * n_sys]
    par2_llr = llr[2 * n_sys:2 * n_sys + k]
    decoded, _ = turbo_decode(sys_llr, par1_llr, par2_llr, interleaver,
                              iterations=iterations)
    coded_ber = float(np.mean(decoded != info_bits))

    # --- Uncoded path (raw bits over the same OFDM/SINR) ---
    frame_u = dco_ofdm_modulate(info_bits, ofdm)
    noise_std = np.sqrt(1.0 / sinr_lin)
    rx_u = frame_u.signal + rng.normal(0.0, noise_std, frame_u.signal.shape)
    hard = dco_ofdm_demodulate(rx_u, frame_u, ofdm, soft=False)[:k]
    uncoded_ber = float(np.mean(hard.astype(int) != info_bits))

    # Message reconstruction from the coded path.
    recovered = deserialize_sequence(bits_to_bytes(decoded))
    messages_ok = recovered == build_session_sequence()
    return LinkResult(coded_ber, uncoded_ber, messages_ok)


def sweep_distance(distances: np.ndarray, info_bits: np.ndarray, ofdm: OfdmParams,
                   iterations: int) -> tuple[list[float], list[float], list[bool]]:
    coded, uncoded, ok = [], [], []
    for i, d in enumerate(distances):
        rng = np.random.default_rng(2000 + i)
        res = simulate_link(info_bits, float(d), velocity=0.0, ofdm=ofdm,
                            iterations=iterations, rng=rng)
        coded.append(res.coded_ber)
        uncoded.append(res.uncoded_ber)
        ok.append(res.messages_ok)
        print(f"  d={d:5.1f} m : coded BER={res.coded_ber:.3e}  "
              f"uncoded BER={res.uncoded_ber:.3e}  msg_ok={res.messages_ok}")
    return coded, uncoded, ok


def sweep_velocity(velocities: np.ndarray, distance: float, info_bits: np.ndarray,
                   ofdm: OfdmParams, iterations: int) -> list[float]:
    coded = []
    for i, v in enumerate(velocities):
        rng = np.random.default_rng(3000 + i)
        res = simulate_link(info_bits, distance, float(v), ofdm, iterations, rng)
        coded.append(res.coded_ber)
        print(f"  v={v:5.1f} m/s : coded BER={res.coded_ber:.3e}")
    return coded


def plot(distances: np.ndarray, coded: list[float], uncoded: list[float],
         ok: list[bool], velocities: np.ndarray, coded_v: list[float],
         v_distance: float) -> pathlib.Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.semilogy(distances, np.clip(uncoded, 1e-5, 1), "s--", label="Uncoded")
    ax1.semilogy(distances, np.clip(coded, 1e-5, 1), "o-", label="Turbo-coded")
    ok_d = distances[np.array(ok)]
    if ok_d.size:
        ax1.axvspan(distances.min(), ok_d.max(), color="green", alpha=0.08,
                    label="ISO 15118 msgs intact")
    ax1.set_xlabel("Distance (m)")
    ax1.set_ylabel("BER")
    ax1.set_title("V2X VLC BER vs distance")
    ax1.set_ylim(1e-5, 1)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.semilogy(velocities * 3.6, np.clip(coded_v, 1e-5, 1), "o-")
    ax2.set_xlabel("Relative speed (km/h)")
    ax2.set_ylabel("Coded BER")
    ax2.set_title(f"BER vs velocity @ {v_distance:.0f} m")
    ax2.set_ylim(1e-5, 1)
    ax2.grid(True, which="both", alpha=0.3)
    return save_fig(fig, "v2x_sim.png")


def main() -> None:
    ofdm = OfdmParams()
    iterations = 6
    info_bits = bytes_to_bits(serialize_sequence(build_session_sequence()))
    print("V2X EV-charger end-to-end simulation")
    print(f"  ISO 15118 payload : {len(info_bits)} bits "
          f"({len(info_bits) // 8} bytes)")

    distances = np.linspace(5.0, 30.0, 8)
    print(" Distance sweep (velocity = 0):")
    coded, uncoded, ok = sweep_distance(distances, info_bits, ofdm, iterations)

    v_distance = 26.0  # marginal range to expose any mobility penalty
    velocities = np.array([0.0, 5.0, 10.0, 20.0, 30.0])
    print(f" Velocity sweep (distance = {v_distance} m):")
    coded_v = sweep_velocity(velocities, v_distance, info_bits, ofdm, iterations)

    path = plot(distances, coded, uncoded, ok, velocities, coded_v, v_distance)
    print(f"Saved V2X figure -> {path}")


if __name__ == "__main__":
    main()
