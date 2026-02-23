"""
Unified Delay System

Combines all delay generation components into a single interface:
- Base timing from TimingProfile/DelayGenerator
- Fatigue multiplier from FatigueManager
- Random "thinking pause" injection
- Micro-stutter injection (tiny hesitations)
- Test harness with histogram visualization

This is the main delay interface for scripts to use.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple, Dict
from .rng import RNG, DistributionParams
from .timing import (
    TimingProfile,
    DelayGenerator,
    DelayStats,
    TIMING_PROFILES,
    FAST_ACTION,
    NORMAL_ACTION,
    THINKING_PAUSE,
)
from .fatigue import FatigueManager, FatigueConfig


@dataclass
class ThinkingPauseConfig:
    probability: float = 0.03
    min_duration: float = 1.5
    max_duration: float = 5.0
    skew: float = 1.2


@dataclass
class MicroStutterConfig:
    probability: float = 0.15
    min_duration: float = 0.02
    max_duration: float = 0.08
    max_stutters: int = 3


@dataclass
class DelayConfig:
    session_variance: float = 0.12
    thinking_pause: ThinkingPauseConfig = field(default_factory=ThinkingPauseConfig)
    micro_stutter: MicroStutterConfig = field(default_factory=MicroStutterConfig)
    fatigue: Optional[FatigueConfig] = None
    use_fatigue: bool = True


@dataclass
class DelayBreakdown:
    base_delay: float
    fatigue_multiplier: float
    thinking_pause: float
    micro_stutters: List[float]
    total: float
    profile_name: str

    def format(self) -> str:
        parts = [f"base={self.base_delay:.3f}s"]
        if self.fatigue_multiplier != 1.0:
            parts.append(f"fatigue=x{self.fatigue_multiplier:.2f}")
        if self.thinking_pause > 0:
            parts.append(f"pause={self.thinking_pause:.2f}s")
        if self.micro_stutters:
            stutter_sum = sum(self.micro_stutters)
            parts.append(f"stutters={stutter_sum:.3f}s({len(self.micro_stutters)})")
        return f"[{self.profile_name}] {' + '.join(parts)} = {self.total:.3f}s"


class Delay:
    """
    Unified delay generator combining timing, fatigue, pauses, and stutters.

    Usage:
        delay = Delay(seed=42)
        delay.start_session()
        wait = delay.delay(FAST_ACTION)
        time.sleep(wait)
    """

    def __init__(
        self,
        config: Optional[DelayConfig] = None,
        seed: Optional[int] = None,
        on_log: Optional[Callable[[str], None]] = None,
        debug: bool = False,
        stop_flag: Optional[threading.Event] = None,
    ):
        self._config = config or DelayConfig()
        self._log_callback = on_log
        self._debug = debug
        self._rng = RNG(seed=seed)
        self._stop_flag = stop_flag

        self._timing = DelayGenerator(
            seed=self._rng.seed,
            on_log=on_log if debug else None,
            debug=debug,
        )

        self._fatigue: Optional[FatigueManager] = None
        if self._config.use_fatigue:
            fatigue_config = self._config.fatigue or FatigueConfig()
            self._fatigue = FatigueManager(
                config=fatigue_config,
                seed=self._rng.seed + 1,
                on_log=on_log if debug else None,
                debug=debug,
            )

        self._is_running = False
        self._last_breakdown: Optional[DelayBreakdown] = None
        self._pause_count = 0
        self._stutter_count = 0
        self._total_pause_time = 0.0
        self._total_stutter_time = 0.0

    def _log(self, message: str) -> None:
        if self._log_callback:
            self._log_callback(f"[Delay] {message}")
        elif self._debug:
            print(f"[Delay] {message}")

    @property
    def seed(self) -> int:
        return self._rng.seed

    @property
    def debug(self) -> bool:
        return self._debug

    @debug.setter
    def debug(self, value: bool) -> None:
        self._debug = value
        self._timing.debug = value
        if self._fatigue:
            self._fatigue.debug = value

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def last_breakdown(self) -> Optional[DelayBreakdown]:
        return self._last_breakdown

    @property
    def fatigue(self) -> Optional[FatigueManager]:
        return self._fatigue

    @property
    def timing(self) -> DelayGenerator:
        return self._timing

    def set_stop_flag(self, stop_flag: threading.Event) -> None:
        """Set the stop flag for interruptible sleeps."""
        self._stop_flag = stop_flag

    def start_session(self, variance: Optional[float] = None) -> None:
        self._is_running = True
        var = variance if variance is not None else self._config.session_variance
        self._timing.vary_all_profiles(variance=var)
        if self._fatigue:
            self._fatigue.start_session()
        self._pause_count = 0
        self._stutter_count = 0
        self._total_pause_time = 0.0
        self._total_stutter_time = 0.0
        self._timing.reset_stats()
        self._log(f"Session started (variance={var*100:.0f}%)")

    def stop_session(self) -> None:
        if self._fatigue:
            self._fatigue.stop_session()
        self._is_running = False
        self._log("Session stopped")

    def _generate_thinking_pause(self) -> float:
        cfg = self._config.thinking_pause
        if not self._rng.chance(cfg.probability):
            return 0.0
        pause = self._rng.skewed_gauss(
            mean=(cfg.min_duration + cfg.max_duration) / 2,
            stddev=(cfg.max_duration - cfg.min_duration) / 4,
            skew=cfg.skew,
            min_val=cfg.min_duration,
            max_val=cfg.max_duration,
        )
        self._pause_count += 1
        self._total_pause_time += pause
        if self._debug:
            self._log(f"Thinking pause: {pause:.2f}s")
        return pause

    def _generate_micro_stutters(self) -> List[float]:
        cfg = self._config.micro_stutter
        if not self._rng.chance(cfg.probability):
            return []
        num_stutters = 1
        while num_stutters < cfg.max_stutters and self._rng.chance(0.3):
            num_stutters += 1
        stutters = []
        for _ in range(num_stutters):
            stutter = self._rng.truncated_gauss(
                mean=(cfg.min_duration + cfg.max_duration) / 2,
                stddev=(cfg.max_duration - cfg.min_duration) / 3,
                min_val=cfg.min_duration,
                max_val=cfg.max_duration,
            )
            stutters.append(stutter)
        self._stutter_count += len(stutters)
        self._total_stutter_time += sum(stutters)
        if self._debug:
            self._log(f"Micro-stutters: {len(stutters)}x, total={sum(stutters):.3f}s")
        return stutters

    def delay(
        self,
        profile: TimingProfile,
        include_pauses: bool = True,
        include_stutters: bool = True,
        include_fatigue: bool = True,
    ) -> float:
        fatigue_mult = 1.0
        if include_fatigue and self._fatigue and self._fatigue.is_running:
            fatigue_mult = self._fatigue.get_multiplier()

        base_delay = self._timing.delay(
            profile,
            use_session_variation=self._is_running,
            multiplier=fatigue_mult,
        )

        thinking_pause = 0.0
        if include_pauses:
            thinking_pause = self._generate_thinking_pause()

        micro_stutters: List[float] = []
        if include_stutters:
            micro_stutters = self._generate_micro_stutters()

        total = base_delay + thinking_pause + sum(micro_stutters)

        self._last_breakdown = DelayBreakdown(
            base_delay=base_delay / fatigue_mult if fatigue_mult != 0 else base_delay,
            fatigue_multiplier=fatigue_mult,
            thinking_pause=thinking_pause,
            micro_stutters=micro_stutters,
            total=total,
            profile_name=profile.name,
        )

        if self._debug:
            self._log(self._last_breakdown.format())

        return total

    def delay_range(
        self,
        min_delay: float,
        max_delay: float,
        include_pauses: bool = True,
        include_stutters: bool = True,
        include_fatigue: bool = True,
    ) -> float:
        fatigue_mult = 1.0
        if include_fatigue and self._fatigue and self._fatigue.is_running:
            fatigue_mult = self._fatigue.get_multiplier()

        base_delay = self._timing.delay_range(min_delay, max_delay) * fatigue_mult
        thinking_pause = self._generate_thinking_pause() if include_pauses else 0.0
        micro_stutters = self._generate_micro_stutters() if include_stutters else []
        total = base_delay + thinking_pause + sum(micro_stutters)

        self._last_breakdown = DelayBreakdown(
            base_delay=base_delay / fatigue_mult if fatigue_mult != 0 else base_delay,
            fatigue_multiplier=fatigue_mult,
            thinking_pause=thinking_pause,
            micro_stutters=micro_stutters,
            total=total,
            profile_name="custom",
        )

        return total

    def _interruptible_sleep(self, duration: float) -> bool:
        """Sleep in small chunks, checking stop flag. Returns True if interrupted."""
        if self._stop_flag is None:
            time.sleep(duration)
            return False
        # Sleep in 50ms chunks so stop flag is checked frequently
        remaining = duration
        while remaining > 0:
            if self._stop_flag.is_set():
                return True
            chunk = min(remaining, 0.05)
            time.sleep(chunk)
            remaining -= chunk
        return False

    def sleep(
        self,
        profile: TimingProfile,
        include_pauses: bool = True,
        include_stutters: bool = True,
        include_fatigue: bool = True,
    ) -> float:
        d = self.delay(profile, include_pauses=include_pauses, include_stutters=include_stutters, include_fatigue=include_fatigue)
        self._interruptible_sleep(d)
        return d

    def sleep_range(
        self,
        min_delay: float,
        max_delay: float,
        include_pauses: bool = True,
        include_stutters: bool = True,
        include_fatigue: bool = True,
    ) -> float:
        d = self.delay_range(min_delay, max_delay, include_pauses=include_pauses, include_stutters=include_stutters, include_fatigue=include_fatigue)
        self._interruptible_sleep(d)
        return d

    def get_stats(self) -> Dict:
        timing_stats = self._timing.get_stats()
        stats = {
            "is_running": self._is_running,
            "seed": self._rng.seed,
            "total_delays": timing_stats.count,
            "total_delay_time": timing_stats.total,
            "mean_delay": timing_stats.mean,
            "stddev_delay": timing_stats.stddev,
            "min_delay": timing_stats.min_delay if timing_stats.count > 0 else 0,
            "max_delay": timing_stats.max_delay if timing_stats.count > 0 else 0,
            "pause_count": self._pause_count,
            "total_pause_time": self._total_pause_time,
            "stutter_count": self._stutter_count,
            "total_stutter_time": self._total_stutter_time,
        }
        if self._fatigue:
            stats["fatigue"] = self._fatigue.get_status()
        return stats

    def format_stats(self) -> str:
        stats = self.get_stats()
        lines = [
            "=== Delay System Statistics ===",
            f"Session: {'Active' if stats['is_running'] else 'Inactive'}",
            f"Seed: {stats['seed']}",
            "",
            "--- Delays ---",
            f"Count: {stats['total_delays']}",
            f"Total Time: {stats['total_delay_time']:.2f}s",
            f"Mean: {stats['mean_delay']:.3f}s",
            f"Stddev: {stats['stddev_delay']:.3f}s",
            f"Range: [{stats['min_delay']:.3f}s - {stats['max_delay']:.3f}s]",
            "",
            "--- Modifiers ---",
            f"Thinking Pauses: {stats['pause_count']} ({stats['total_pause_time']:.2f}s total)",
            f"Micro-Stutters: {stats['stutter_count']} ({stats['total_stutter_time']:.3f}s total)",
        ]
        if "fatigue" in stats:
            f = stats["fatigue"]
            lines.extend([
                "",
                "--- Fatigue ---",
                f"Session Duration: {f['session_hours']:.2f}h",
                f"Fatigue Multiplier: {f['fatigue_multiplier']:.3f}",
                f"Attention: {f['attention_focus']:.2f} ({'focused' if f['attention_is_focused'] else 'unfocused'})",
                f"Combined Multiplier: {f['combined_multiplier']:.3f}",
            ])
        return "\n".join(lines)

    def format_histogram(self, bins: int = 20) -> str:
        return self._timing.format_histogram()

    def test_harness(
        self,
        count: int = 1000,
        profile: Optional[TimingProfile] = None,
        include_pauses: bool = True,
        include_stutters: bool = True,
        simulate_fatigue_hours: float = 0.0,
    ) -> str:
        profile = profile or NORMAL_ACTION
        was_running = self._is_running
        self.start_session()

        if simulate_fatigue_hours > 0 and self._fatigue:
            self._fatigue._session_start = time.time() - (simulate_fatigue_hours * 3600)

        delays = []
        pauses_triggered = 0
        stutters_triggered = 0

        for _ in range(count):
            d = self.delay(profile, include_pauses=include_pauses, include_stutters=include_stutters)
            delays.append(d)
            if self._last_breakdown:
                if self._last_breakdown.thinking_pause > 0:
                    pauses_triggered += 1
                if self._last_breakdown.micro_stutters:
                    stutters_triggered += 1

        import statistics as stat
        mean = stat.mean(delays)
        stddev = stat.stdev(delays) if len(delays) > 1 else 0
        min_d = min(delays)
        max_d = max(delays)

        lines = [
            f"=== Test Harness Results ===",
            f"Profile: {profile.name}",
            f"Samples: {count}",
            f"",
            f"--- Configuration ---",
            f"Pauses: {'Enabled' if include_pauses else 'Disabled'} ({self._config.thinking_pause.probability*100:.1f}% chance)",
            f"Stutters: {'Enabled' if include_stutters else 'Disabled'} ({self._config.micro_stutter.probability*100:.1f}% chance)",
            f"Simulated Fatigue: {simulate_fatigue_hours:.1f}h",
            f"",
            f"--- Results ---",
            f"Mean: {mean:.3f}s (profile mean: {profile.mean:.3f}s)",
            f"Stddev: {stddev:.3f}s (profile stddev: {profile.stddev:.3f}s)",
            f"Range: [{min_d:.3f}s - {max_d:.3f}s]",
            f"Pauses Triggered: {pauses_triggered} ({pauses_triggered/count*100:.1f}%)",
            f"Stutters Triggered: {stutters_triggered} ({stutters_triggered/count*100:.1f}%)",
            f"",
            self.format_histogram(),
        ]

        if not was_running:
            self.stop_session()

        return "\n".join(lines)

    def get_session_profiles(self) -> Dict[str, TimingProfile]:
        return self._timing.get_session_profiles()


DELAY_CONFIGS = {
    "default": DelayConfig(),
    "conservative": DelayConfig(
        session_variance=0.15,
        thinking_pause=ThinkingPauseConfig(probability=0.05, min_duration=2.0, max_duration=6.0),
        micro_stutter=MicroStutterConfig(probability=0.20),
    ),
    "aggressive": DelayConfig(
        session_variance=0.08,
        thinking_pause=ThinkingPauseConfig(probability=0.02, min_duration=1.0, max_duration=3.0),
        micro_stutter=MicroStutterConfig(probability=0.10, max_stutters=2),
    ),
    "no_modifiers": DelayConfig(
        thinking_pause=ThinkingPauseConfig(probability=0.0),
        micro_stutter=MicroStutterConfig(probability=0.0),
        use_fatigue=False,
    ),
}
