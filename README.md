# vlc-turbo-solar

Pure-Python (numpy / scipy / matplotlib) simulation suite for a **solar-powered
Visible Light Communication (VLC) link using turbo codes**, in two phases:

1. **BE project recreation** — solar-powered *indoor* VLC, 5 m range, OOK
   intensity modulation, turbo-coded.
2. **Extension** — VLC for *V2X EV-to-charger* communication: outdoor link,
   5–30 m, DCO-OFDM, Doppler/mobility model, and an ISO 15118 Plug & Charge
   message exchange carried over the coded optical link.

No hardware dependencies — everything is simulation. Every module is
independently runnable and writes its figure(s) to `outputs/`.

## Setup

Uses [uv](https://docs.astral.sh/uv/) for environment management:

```bash
uv sync                 # install deps from pyproject.toml / uv.lock
# or, from scratch:
uv add numpy scipy matplotlib jupyter
```

## Repository layout

```
solar/            PV MPPT (perturb & observe) + PWM LED driver
turbo/            PCCC encoder, Log-MAP BCJR iterative decoder, BER analysis
vlc/indoor/       Lambertian LOS channel + OOK modulation
vlc/v2x/          outdoor channel, DCO-OFDM, Doppler/CFO model
ev_charger/       optical link budget, charging state machine, ISO 15118 mock
simulation/       indoor and V2X end-to-end pipelines
notebooks/        results.ipynb — stitches every figure with narrative
common/           shared figure-saving helper
outputs/          generated PNGs
```

## Running

Each module runs standalone and saves a figure to `outputs/`:

```bash
# Phase 1 — turbo + solar + indoor VLC
uv run python turbo/encoder.py
uv run python turbo/decoder.py
uv run python turbo/ber_analysis.py            # -> outputs/turbo_ber.png
uv run python solar/mppt_perturb_observe.py    # -> outputs/mppt.png
uv run python solar/pwm_led_control.py         # -> outputs/pwm_led.png
uv run python vlc/indoor/channel_model.py      # -> outputs/indoor_channel.png
uv run python vlc/indoor/modulation_ook.py     # -> outputs/ook_ber.png
uv run python simulation/indoor_end_to_end.py  # -> outputs/indoor_e2e_ber.png

# Phase 2 — V2X EV charger
uv run python vlc/v2x/channel_model_outdoor.py # -> outputs/v2x_channel.png
uv run python vlc/v2x/ofdm_modulation.py       # -> outputs/ofdm.png
uv run python vlc/v2x/doppler_model.py         # -> outputs/doppler.png
uv run python ev_charger/link_budget.py        # -> outputs/link_budget.png
uv run python ev_charger/charging_protocol.py  # -> outputs/charging_session.png
uv run python ev_charger/iso15118_mock.py
uv run python simulation/v2x_ev_charger_sim.py # -> outputs/v2x_sim.png

# Stitch everything
uv run jupyter nbconvert --to notebook --execute notebooks/results.ipynb \
    --output results.ipynb
```

## Technical notes

- **Turbo code:** rate ~1/3 PCCC, two RSC encoders (generators (7,5) octal,
  feedback 7, memory 2), pseudo-random interleaver, encoder 1 trellis-terminated.
  Decoder: iterative Log-MAP BCJR exchanging extrinsic LLRs (default 6 iterations;
  Max-Log-MAP available via `log_map=False`).
- **LLR convention:** `L = log P(bit=0)/P(bit=1)` throughout; BPSK/OOK map
  bit 0 → high level.
- **Indoor channel:** generalised-Lambertian LOS DC gain with shot + thermal
  noise. Coded vs uncoded compared at equal per-symbol SNR (peak-power-limited
  IM/DD).
- **V2X:** DCO-OFDM with Hermitian symmetry (real IFFT), DC bias + clipping,
  QPSK subcarriers; outdoor channel adds atmospheric extinction and a large
  daylight background current. Doppler is modelled as an equivalent normalized
  CFO → ICI penalty (negligible at charging-approach speeds — a finding the
  simulation reproduces).
- **ISO 15118:** mock Plug & Charge message sequence, length-prefixed and
  serialized to bytes, transmitted over the coded OFDM link and reconstructed.

## Key results

- Turbo code shows a clear waterfall: ~0 BER by ≈1 dB Eb/N0 over BPSK/AWGN.
- Indoor turbo-coded OOK beats uncoded by several dB.
- V2X: turbo coding keeps ISO 15118 messages intact and roughly doubles the
  usable range (≈13 m uncoded → ≈26 m coded) under the modelled low-power link.
# vlc_comm
