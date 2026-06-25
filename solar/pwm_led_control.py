"""PWM LED driver: convert harvested PV power into an LED bias / duty cycle.

The MPPT stage delivers an available electrical power; this module maps it to a
PWM duty cycle that sets the average LED drive current (and hence emitted
optical power, which the VLC transmitter modulates with OOK data on top).
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dataclasses import dataclass

import numpy as np

import matplotlib.pyplot as plt

from common.io import save_fig


@dataclass
class LedParams:
    forward_voltage: float = 3.2  # V
    max_current: float = 0.35  # A (350 mA power LED)
    efficacy: float = 1.1  # optical W per electrical A (toy luminous model)

    @property
    def max_electrical_power(self) -> float:
        return self.forward_voltage * self.max_current


def duty_from_power(p_available: np.ndarray, params: LedParams | None = None) -> np.ndarray:
    """Map available electrical power to a PWM duty cycle in [0, 1]."""
    params = params or LedParams()
    duty = p_available / params.max_electrical_power
    return np.clip(duty, 0.0, 1.0)


def led_drive_current(duty: np.ndarray, params: LedParams | None = None) -> np.ndarray:
    """Average LED current for a given duty cycle."""
    params = params or LedParams()
    return duty * params.max_current


def led_optical_power(duty: np.ndarray, params: LedParams | None = None) -> np.ndarray:
    """Average emitted optical power for a given duty cycle."""
    params = params or LedParams()
    return led_drive_current(duty, params) * params.efficacy


def pwm_waveform(duty: float, n_periods: int = 5, samples_per_period: int = 100
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Generate a normalised PWM waveform (time, level) for visualisation."""
    t = np.linspace(0, n_periods, n_periods * samples_per_period, endpoint=False)
    phase = np.mod(t, 1.0)
    level = (phase < duty).astype(float)
    return t, level


def plot_pwm_led(params: LedParams | None = None) -> pathlib.Path:
    params = params or LedParams()
    p_avail = np.linspace(0, 1.4 * params.max_electrical_power, 300)
    duty = duty_from_power(p_avail, params)
    optical = led_optical_power(duty, params)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(p_avail, duty, label="Duty cycle")
    ax1.set_xlabel("Available power (W)")
    ax1.set_ylabel("PWM duty cycle")
    ax1.axvline(params.max_electrical_power, color="r", ls="--",
                label="LED rated power")
    ax1.set_title("Duty cycle vs harvested power")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2b = ax2.twinx()
    ax2.plot(duty, led_drive_current(duty, params) * 1000, "b-", label="Avg current")
    ax2b.plot(duty, optical, "g-", label="Optical power")
    ax2.set_xlabel("Duty cycle")
    ax2.set_ylabel("Avg LED current (mA)", color="b")
    ax2b.set_ylabel("Optical power (W)", color="g")
    ax2.set_title("LED drive vs duty cycle")
    ax2.grid(alpha=0.3)
    return save_fig(fig, "pwm_led.png")


def main() -> None:
    params = LedParams()
    print("PWM LED control")
    print(f"  LED rated electrical power : {params.max_electrical_power:.2f} W")
    for p in (0.3, 0.6, 1.0, 1.2):
        d = float(duty_from_power(np.array(p), params))
        print(f"  P_avail={p:.2f} W -> duty={d:.2f}, "
              f"I_avg={1000 * led_drive_current(np.array(d), params):.0f} mA, "
              f"optical={led_optical_power(np.array(d), params):.3f} W")
    path = plot_pwm_led(params)
    print(f"Saved PWM/LED figure -> {path}")


if __name__ == "__main__":
    main()
