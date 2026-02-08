"""
Timing System

Human-like delay generation with:
- Named timing profiles
- Session variation (randomize at start)
- Debug mode with statistics tracking
- Histogram generation for visualization

Design principle: Humans don't produce uniform delays.
Use Gaussian distributions with meta-randomization.
"""

import time
import statistics
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Tuple

from .rng import RNG, DistributionParams


@dataclass
class TimingProfile:
    """A named timing profile with distribution parameters."""
    name: str
    label: str
    mean: float
    stddev: float
    min_val: float
    max_val: float
    skew: float = 0.0
    description: str = ""

    def to_distribution_params(self) -> DistributionParams:
        return DistributionParams(
            mean=self.mean,
            stddev=self.stddev,
            min_val=self.min_val,
            max_val=self.max_val,
            skew=self.skew,
        )

    def copy(self) -> "TimingProfile":
        return TimingProfile(
            name=self.name,
            label=self.label,
            mean=self.mean,
            stddev=self.stddev,
            min_val=self.min_val,
            max_val=self.max_val,
            skew=self.skew,
            description=self.description,
        )

    def with_variation(self, rng: RNG, variance: float = 0.1) -> "TimingProfile":
        """Create a varied copy for session-level personality."""
        varied_params = rng.vary_params(self.to_distribution_params(), variance)
        return TimingProfile(
            name=f"{self.name}_varied",
            label=f"{self.label} (Varied)",
            mean=varied_params.mean,
            stddev=varied_params.stddev,
            min_val=varied_params.min_val if varied_params.min_val else self.min_val,
            max_val=varied_params.max_val if varied_params.max_val else self.max_val,
            skew=varied_params.skew,
            description=f"Session-varied version of {self.name}",
        )


# Pre-built timing profiles
FAST_ACTION = TimingProfile(
    name="fast_action", label="Fast Action",
    mean=0.15, stddev=0.05, min_val=0.08, max_val=0.35, skew=0.3,
    description="Quick actions like clicking nearby objects",
)

NORMAL_ACTION = TimingProfile(
    name="normal_action", label="Normal Action",
    mean=0.4, stddev=0.12, min_val=0.2, max_val=0.8, skew=0.2,
    description="Standard interaction delays",
)

CAREFUL_ACTION = TimingProfile(
    name="careful_action", label="Careful Action",
    mean=0.8, stddev=0.25, min_val=0.4, max_val=1.6, skew=0.4,
    description="Deliberate actions requiring attention",
)

THINKING_PAUSE = TimingProfile(
    name="thinking_pause", label="Thinking Pause",
    mean=2.0, stddev=0.8, min_val=1.0, max_val=5.0, skew=1.0,
    description="Simulates human thinking/deciding",
)

INSTANT = TimingProfile(
    name="instant", label="Instant",
    mean=0.05, stddev=0.02, min_val=0.02, max_val=0.12, skew=0.0,
    description="Near-instant reactions (use sparingly)",
)

SLOW_ACTION = TimingProfile(
    name="slow_action", label="Slow Action",
    mean=1.2, stddev=0.35, min_val=0.6, max_val=2.5, skew=0.5,
    description="Slow, careful movements",
)

MENU_NAVIGATION = TimingProfile(
    name="menu_navigation", label="Menu Navigation",
    mean=0.3, stddev=0.1, min_val=0.15, max_val=0.6, skew=0.2,
    description="Moving through menus and interfaces",
)

TYPING_DELAY = TimingProfile(
    name="typing_delay", label="Typing Delay",
    mean=0.08, stddev=0.03, min_val=0.04, max_val=0.2, skew=0.5,
    description="Inter-key delay when typing",
)

TIMING_PROFILES: Dict[str, TimingProfile] = {
    "instant": INSTANT,
    "fast_action": FAST_ACTION,
    "normal_action": NORMAL_ACTION,
    "careful_action": CAREFUL_ACTION,
    "slow_action": SLOW_ACTION,
    "thinking_pause": THINKING_PAUSE,
    "menu_navigation": MENU_NAVIGATION,
    "typing_delay": TYPING_DELAY,
}


@dataclass
class DelayStats:
    """Statistics for generated delays."""
    count: int = 0
    total: float = 0.0
    min_delay: float = float('inf')
    max_delay: float = 0.0
    samples: List[float] = field(default_factory=list)
    max_samples: int = 1000

    def record(self, delay: float) -> None:
        self.count += 1
        self.total += delay
        self.min_delay = min(self.min_delay, delay)
        self.max_delay = max(self.max_delay, delay)
        self.samples.append(delay)
        if len(self.samples) > self.max_samples:
            self.samples.pop(0)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0

    @property
    def stddev(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return statistics.stdev(self.samples)

    def get_histogram(self, bins: int = 20) -> List[Tuple[float, float, int]]:
        if not self.samples:
            return []
        min_val = min(self.samples)
        max_val = max(self.samples)
        if min_val == max_val:
            return [(min_val, max_val, len(self.samples))]
        bin_width = (max_val - min_val) / bins
        histogram = []
        for i in range(bins):
            bin_start = min_val + i * bin_width
            bin_end = bin_start + bin_width
            count = sum(1 for s in self.samples if bin_start <= s < bin_end)
            if i == bins - 1:
                count += sum(1 for s in self.samples if s == max_val)
            histogram.append((bin_start, bin_end, count))
        return histogram

    def format_histogram(self, width: int = 40, bins: int = 15) -> str:
        hist = self.get_histogram(bins)
        if not hist:
            return "No samples recorded"
        max_count = max(h[2] for h in hist)
        if max_count == 0:
            return "No samples recorded"
        lines = [
            f"Delay Distribution (n={self.count}, mean={self.mean:.3f}s, stddev={self.stddev:.3f}s)",
            f"Range: [{self.min_delay:.3f}s - {self.max_delay:.3f}s]",
            ""
        ]
        for bin_start, bin_end, count in hist:
            bar_len = int((count / max_count) * width) if max_count > 0 else 0
            bar = "#" * bar_len
            lines.append(f"{bin_start:6.3f}s |{bar:<{width}} {count}")
        return "\n".join(lines)

    def reset(self) -> None:
        self.count = 0
        self.total = 0.0
        self.min_delay = float('inf')
        self.max_delay = 0.0
        self.samples.clear()


class DelayGenerator:
    """Generates human-like delays using timing profiles."""

    def __init__(
        self,
        seed: Optional[int] = None,
        on_log: Optional[Callable[[str], None]] = None,
        debug: bool = False,
    ):
        self._rng = RNG(seed=seed)
        self._log_callback = on_log
        self._debug = debug
        self._session_profiles: Dict[str, TimingProfile] = {}
        self._stats: Dict[str, DelayStats] = {}
        self._global_stats = DelayStats()

    def _log(self, message: str) -> None:
        if self._log_callback:
            self._log_callback(f"[Timing] {message}")
        elif self._debug:
            print(f"[Timing] {message}")

    @property
    def seed(self) -> int:
        return self._rng.seed

    @property
    def debug(self) -> bool:
        return self._debug

    @debug.setter
    def debug(self, value: bool) -> None:
        self._debug = value

    def vary_all_profiles(self, variance: float = 0.1) -> None:
        """Create session-varied versions of all profiles."""
        self._session_profiles.clear()
        for name, profile in TIMING_PROFILES.items():
            varied = profile.with_variation(self._rng, variance)
            self._session_profiles[name] = varied
            if self._debug:
                self._log(
                    f"Varied {name}: mean {profile.mean:.3f} -> {varied.mean:.3f}, "
                    f"stddev {profile.stddev:.3f} -> {varied.stddev:.3f}"
                )
        self._log(f"Session profiles varied with {variance*100:.0f}% variance")

    def get_profile(self, profile: TimingProfile, use_session_variation: bool = True) -> TimingProfile:
        if use_session_variation and profile.name in self._session_profiles:
            return self._session_profiles[profile.name]
        return profile

    def delay(
        self,
        profile: TimingProfile,
        use_session_variation: bool = True,
        multiplier: float = 1.0,
    ) -> float:
        actual_profile = self.get_profile(profile, use_session_variation)
        params = actual_profile.to_distribution_params()
        base_delay = self._rng.from_params(params)
        delay = base_delay * multiplier
        delay = max(actual_profile.min_val, min(actual_profile.max_val * multiplier, delay))
        self._record_delay(profile.name, delay)
        if self._debug:
            self._log(f"{profile.name}: {delay:.3f}s (base={base_delay:.3f}, mult={multiplier:.2f})")
        return delay

    def delay_range(self, min_delay: float, max_delay: float, skew: float = 0.0) -> float:
        mean = (min_delay + max_delay) / 2
        stddev = (max_delay - min_delay) / 4
        params = DistributionParams(mean=mean, stddev=stddev, min_val=min_delay, max_val=max_delay, skew=skew)
        delay = self._rng.from_params(params)
        self._record_delay("custom", delay)
        if self._debug:
            self._log(f"custom[{min_delay:.2f}-{max_delay:.2f}]: {delay:.3f}s")
        return delay

    def _record_delay(self, profile_name: str, delay: float) -> None:
        if profile_name not in self._stats:
            self._stats[profile_name] = DelayStats()
        self._stats[profile_name].record(delay)
        self._global_stats.record(delay)

    def get_stats(self, profile_name: Optional[str] = None) -> DelayStats:
        if profile_name is None:
            return self._global_stats
        return self._stats.get(profile_name, DelayStats())

    def format_stats(self, profile_name: Optional[str] = None) -> str:
        lines = ["=== Delay Statistics ==="]
        if profile_name:
            stats = self.get_stats(profile_name)
            lines.append(f"\nProfile: {profile_name}")
            lines.append(stats.format_histogram())
        else:
            lines.append(f"\nGlobal ({self._global_stats.count} total delays):")
            if self._global_stats.count > 0:
                lines.append(f"  Mean: {self._global_stats.mean:.3f}s")
                lines.append(f"  Stddev: {self._global_stats.stddev:.3f}s")
                lines.append(f"  Range: [{self._global_stats.min_delay:.3f}s - {self._global_stats.max_delay:.3f}s]")
            for name, stats in self._stats.items():
                if stats.count > 0:
                    lines.append(f"\n{name} (n={stats.count}):")
                    lines.append(f"  Mean: {stats.mean:.3f}s, Stddev: {stats.stddev:.3f}s")
        return "\n".join(lines)

    def format_histogram(self, profile_name: Optional[str] = None) -> str:
        stats = self.get_stats(profile_name)
        return stats.format_histogram()

    def reset_stats(self) -> None:
        self._stats.clear()
        self._global_stats.reset()
        self._log("Statistics reset")

    def sleep(
        self,
        profile: TimingProfile,
        use_session_variation: bool = True,
        multiplier: float = 1.0,
    ) -> float:
        delay = self.delay(profile, use_session_variation, multiplier)
        time.sleep(delay)
        return delay

    def get_session_profiles(self) -> Dict[str, TimingProfile]:
        return self._session_profiles.copy()

    def get_all_profiles(self) -> Dict[str, TimingProfile]:
        return TIMING_PROFILES.copy()
