"""EV charging session state machine with a power-ramp profile.

A minimal, hardware-free model of a charging session lifecycle that the VLC
link carries control messages for. Drives through plug-in, authentication,
authorization, a ramped charging phase, and clean shutdown.
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import enum
from dataclasses import dataclass, field

import numpy as np

import matplotlib.pyplot as plt

from common.io import save_fig


class ChargeState(enum.Enum):
    IDLE = "idle"
    PLUGGED = "plugged"
    AUTHENTICATING = "authenticating"
    AUTHORIZED = "authorized"
    CHARGING = "charging"
    COMPLETE = "complete"


@dataclass
class SessionConfig:
    target_power_kw: float = 50.0
    ramp_kw_per_s: float = 10.0
    battery_capacity_kwh: float = 0.8  # remaining capacity to fill (toy, completes fast)
    taper_power_kw: float = 10.0  # constant reduced power once near full
    dt: float = 1.0  # simulation timestep (s)
    auth_time_s: float = 3.0


@dataclass
class SessionLog:
    time: list[float] = field(default_factory=list)
    state: list[ChargeState] = field(default_factory=list)
    power_kw: list[float] = field(default_factory=list)
    energy_kwh: list[float] = field(default_factory=list)


def run_session(config: SessionConfig | None = None, max_time_s: float = 120.0
                ) -> SessionLog:
    """Step the charging state machine until completion or timeout."""
    config = config or SessionConfig()
    log = SessionLog()

    state = ChargeState.IDLE
    t = 0.0
    power = 0.0
    energy = 0.0
    auth_elapsed = 0.0

    while t <= max_time_s:
        # Log the current state before advancing the machine.
        log.time.append(t)
        log.state.append(state)
        log.power_kw.append(power)
        log.energy_kwh.append(energy)
        if state == ChargeState.COMPLETE:
            break

        # State transitions / power update for the next step.
        if state == ChargeState.IDLE:
            state = ChargeState.PLUGGED
        elif state == ChargeState.PLUGGED:
            state = ChargeState.AUTHENTICATING
        elif state == ChargeState.AUTHENTICATING:
            auth_elapsed += config.dt
            if auth_elapsed >= config.auth_time_s:
                state = ChargeState.AUTHORIZED
        elif state == ChargeState.AUTHORIZED:
            state = ChargeState.CHARGING
        elif state == ChargeState.CHARGING:
            power = min(config.target_power_kw, power + config.ramp_kw_per_s * config.dt)
            if energy >= 0.9 * config.battery_capacity_kwh:  # taper near full
                power = min(power, config.taper_power_kw)
            energy += power * config.dt / 3600.0
            if energy >= config.battery_capacity_kwh:
                state = ChargeState.COMPLETE
                power = 0.0
        t += config.dt

    return log


def plot_session(log: SessionLog) -> pathlib.Path:
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(log.time, log.power_kw, "b-", label="Power (kW)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Charging power (kW)", color="b")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(log.time, log.energy_kwh, "g-", label="Energy (kWh)")
    ax2.set_ylabel("Delivered energy (kWh)", color="g")

    # Shade the state regions.
    states = [s for s in log.state]
    changes = [0] + [i for i in range(1, len(states)) if states[i] != states[i - 1]]
    for idx in changes:
        ax1.axvline(log.time[idx], color="gray", ls=":", alpha=0.4)
        ax1.text(log.time[idx], ax1.get_ylim()[1] * 0.95, states[idx].value,
                 rotation=90, va="top", fontsize=7, alpha=0.7)
    ax1.set_title("EV charging session (state machine + power ramp)")
    return save_fig(fig, "charging_session.png")


def main() -> None:
    log = run_session()
    print("EV charging session")
    seen = []
    for s in log.state:
        if not seen or seen[-1] != s:
            seen.append(s)
    print(f"  state sequence : {' -> '.join(s.value for s in seen)}")
    print(f"  duration       : {log.time[-1]:.0f} s")
    print(f"  peak power     : {max(log.power_kw):.1f} kW")
    print(f"  energy delivered: {log.energy_kwh[-1]:.2f} kWh")
    path = plot_session(log)
    print(f"Saved charging-session figure -> {path}")


if __name__ == "__main__":
    main()
