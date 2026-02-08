"""
Fatigue System

Simulates human fatigue over extended play sessions:
- Gradual delay multiplier increase over time
- Randomized fatigue curves (each session is unique)
- Attention fluctuation (focus/unfocus cycles)
- Debug mode with graph data for visualization

Design principle: Humans slow down over time. A bot that
maintains perfect consistency for hours is detectable.
"""

import time
import math
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple
from enum import Enum

from .rng import RNG


class FatigueCurve(Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    STEPPED = "stepped"
    WAVE = "wave"


@dataclass
class FatigueConfig:
    base_multiplier: float = 1.0
    max_multiplier: float = 1.2
    full_fatigue_hours: float = 2.0
    curve: FatigueCurve = FatigueCurve.LINEAR
    curve_variance: float = 0.15
    attention_cycle_minutes: float = 15.0
    attention_variance: float = 0.1


@dataclass
class AttentionState:
    focus_level: float = 1.0
    cycle_position: float = 0.0
    cycle_duration: float = 900.0
    is_focused: bool = True


class FatigueManager:
    """
    Manages fatigue simulation for human-like behavior degradation.

    Usage:
        fatigue = FatigueManager(seed=42)
        fatigue.start_session()
        multiplier = fatigue.get_multiplier()
        delay = base_delay * multiplier
    """

    def __init__(
        self,
        config: Optional[FatigueConfig] = None,
        seed: Optional[int] = None,
        on_log: Optional[Callable[[str], None]] = None,
        debug: bool = False,
    ):
        self._config = config or FatigueConfig()
        self._rng = RNG(seed=seed)
        self._log_callback = on_log
        self._debug = debug

        self._session_start: Optional[float] = None
        self._is_running = False
        self._curve_steepness = 1.0
        self._curve_offset = 0.0
        self._attention = AttentionState()
        self._last_attention_update = 0.0
        self._history: List[Tuple[float, float, float]] = []
        self._max_history = 1000

    def _log(self, message: str) -> None:
        if self._log_callback:
            self._log_callback(f"[Fatigue] {message}")
        elif self._debug:
            print(f"[Fatigue] {message}")

    @property
    def config(self) -> FatigueConfig:
        return self._config

    @property
    def attention(self) -> AttentionState:
        return self._attention

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def debug(self) -> bool:
        return self._debug

    @debug.setter
    def debug(self, value: bool) -> None:
        self._debug = value

    def start_session(self) -> None:
        self._session_start = time.time()
        self._is_running = True
        self._history.clear()

        variance = self._config.curve_variance
        self._curve_steepness = self._rng.vary_value(1.0, variance)
        self._curve_offset = self._rng.uniform(-variance * 0.5, variance * 0.5)

        self._attention = AttentionState(
            focus_level=1.0,
            cycle_position=0.0,
            cycle_duration=self._config.attention_cycle_minutes * 60 * self._rng.vary_value(1.0, 0.2),
            is_focused=True,
        )
        self._last_attention_update = self._session_start

        self._log(
            f"Session started - curve: {self._config.curve.value}, "
            f"steepness: {self._curve_steepness:.2f}, "
            f"target: {self._config.max_multiplier:.2f}x over {self._config.full_fatigue_hours:.1f}h"
        )

    def stop_session(self) -> None:
        if self._is_running:
            duration = self.get_session_duration()
            final_mult = self.get_multiplier()
            self._log(f"Session stopped after {duration/3600:.2f}h, final multiplier: {final_mult:.3f}")
        self._is_running = False
        self._session_start = None

    def get_session_duration(self) -> float:
        if not self._is_running or self._session_start is None:
            return 0.0
        return time.time() - self._session_start

    def get_session_hours(self) -> float:
        return self.get_session_duration() / 3600.0

    def get_fatigue_progress(self) -> float:
        hours = self.get_session_hours()
        return hours / self._config.full_fatigue_hours

    def _calculate_base_fatigue(self, progress: float) -> float:
        p = min(progress, 1.5)
        curve = self._config.curve

        if curve == FatigueCurve.LINEAR:
            return p
        elif curve == FatigueCurve.EXPONENTIAL:
            return (math.exp(p * 2) - 1) / (math.exp(2) - 1)
        elif curve == FatigueCurve.LOGARITHMIC:
            if p <= 0:
                return 0.0
            return math.log(1 + p * (math.e - 1)) / math.log(math.e)
        elif curve == FatigueCurve.STEPPED:
            steps = 4
            step = int(p * steps)
            step_progress = (p * steps) % 1.0
            if step_progress < 0.2:
                return (step + step_progress / 0.2 * 0.3) / steps
            return (step + 0.3 + (step_progress - 0.2) / 0.8 * 0.7) / steps
        elif curve == FatigueCurve.WAVE:
            base = p * 0.8
            wave = math.sin(p * math.pi * 4) * 0.1 * (1 - p * 0.5)
            return max(0, base + wave)

        return p

    def get_fatigue_multiplier(self) -> float:
        if not self._is_running:
            return self._config.base_multiplier

        progress = self.get_fatigue_progress()
        adjusted_progress = progress * self._curve_steepness + self._curve_offset
        adjusted_progress = max(0, adjusted_progress)
        fatigue = self._calculate_base_fatigue(adjusted_progress)
        multiplier_range = self._config.max_multiplier - self._config.base_multiplier
        multiplier = self._config.base_multiplier + fatigue * multiplier_range
        return max(self._config.base_multiplier, min(self._config.max_multiplier * 1.1, multiplier))

    def _update_attention(self) -> None:
        if not self._is_running:
            return

        now = time.time()
        elapsed = now - self._last_attention_update
        self._last_attention_update = now

        cycle_advance = elapsed / self._attention.cycle_duration
        self._attention.cycle_position += cycle_advance

        if self._attention.cycle_position >= 1.0:
            self._attention.cycle_position = 0.0
            self._attention.is_focused = not self._attention.is_focused
            base_duration = self._config.attention_cycle_minutes * 60
            self._attention.cycle_duration = base_duration * self._rng.vary_value(1.0, 0.3)
            if self._debug:
                state = "focused" if self._attention.is_focused else "unfocused"
                self._log(f"Attention cycle: now {state} for {self._attention.cycle_duration/60:.1f}min")

        cycle_pos = self._attention.cycle_position
        transition_zone = 0.15

        if cycle_pos < transition_zone:
            transition_progress = cycle_pos / transition_zone
            if self._attention.is_focused:
                base_focus = 0.85 + transition_progress * 0.15
            else:
                base_focus = 1.0 - transition_progress * 0.15
        elif cycle_pos > (1.0 - transition_zone):
            transition_progress = (cycle_pos - (1.0 - transition_zone)) / transition_zone
            if self._attention.is_focused:
                base_focus = 1.0 - transition_progress * 0.15
            else:
                base_focus = 0.85 + transition_progress * 0.15
        else:
            base_focus = 1.0 if self._attention.is_focused else 0.85

        variance = self._config.attention_variance
        fluctuation = self._rng.gauss(0, variance * 0.5)
        self._attention.focus_level = max(0.7, min(1.15, base_focus + fluctuation))

    def get_attention_multiplier(self) -> float:
        self._update_attention()
        return 2.0 - self._attention.focus_level

    def get_multiplier(self) -> float:
        """Get the combined fatigue + attention multiplier."""
        fatigue_mult = self.get_fatigue_multiplier()
        attention_mult = self.get_attention_multiplier()
        combined = fatigue_mult * attention_mult

        if self._is_running and len(self._history) < self._max_history:
            self._history.append((
                self.get_session_duration(),
                fatigue_mult,
                self._attention.focus_level,
            ))

        if self._debug and len(self._history) % 100 == 0:
            self._log(
                f"t={self.get_session_hours():.2f}h, "
                f"fatigue={fatigue_mult:.3f}, "
                f"attention={self._attention.focus_level:.2f}, "
                f"combined={combined:.3f}"
            )

        return combined

    def get_history(self) -> List[Tuple[float, float, float]]:
        return self._history.copy()

    def get_status(self) -> dict:
        return {
            "is_running": self._is_running,
            "session_hours": self.get_session_hours(),
            "fatigue_progress": self.get_fatigue_progress(),
            "fatigue_multiplier": self.get_fatigue_multiplier(),
            "attention_focus": self._attention.focus_level,
            "attention_is_focused": self._attention.is_focused,
            "combined_multiplier": self.get_multiplier() if self._is_running else 1.0,
            "curve": self._config.curve.value,
            "curve_steepness": self._curve_steepness,
        }

    def simulate_duration(self, hours: float, samples: int = 100) -> List[Tuple[float, float]]:
        was_running = self._is_running
        old_start = self._session_start
        old_history = self._history.copy()

        self._session_start = time.time() - (hours * 3600)
        self._is_running = True

        results = []
        for i in range(samples):
            t = (i / (samples - 1)) * hours
            self._session_start = time.time() - (t * 3600)
            mult = self.get_fatigue_multiplier()
            results.append((t, mult))

        self._is_running = was_running
        self._session_start = old_start
        self._history = old_history

        return results


FATIGUE_CONFIGS = {
    "default": FatigueConfig(),
    "quick_fatigue": FatigueConfig(
        max_multiplier=1.25, full_fatigue_hours=1.5, curve=FatigueCurve.EXPONENTIAL,
    ),
    "slow_fatigue": FatigueConfig(
        max_multiplier=1.15, full_fatigue_hours=3.0, curve=FatigueCurve.LOGARITHMIC,
    ),
    "variable": FatigueConfig(
        max_multiplier=1.2, full_fatigue_hours=2.0, curve=FatigueCurve.WAVE, curve_variance=0.2,
    ),
    "stepped": FatigueConfig(
        max_multiplier=1.2, full_fatigue_hours=2.0, curve=FatigueCurve.STEPPED,
    ),
}
