"""
Random Number Generation

Provides human-like random distributions for timing and behavior.
Key features:
- Seed management for reproducibility in testing
- Gaussian distribution with configurable parameters
- Truncated Gaussian (bounded values)
- Skewed distributions (usually fast, sometimes slow)
- Meta-randomization (randomize the randomness parameters)

Design principle: "If we're following a random formula,
we have to randomize the randomness."
"""

import random
import math
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple, List


@dataclass
class DistributionParams:
    """
    Parameters for a distribution that can itself be varied.

    Attributes:
        mean: Center of the distribution
        stddev: Standard deviation (spread)
        min_val: Minimum allowed value (for truncation)
        max_val: Maximum allowed value (for truncation)
        skew: Skew factor (0 = symmetric, positive = right-skewed, negative = left-skewed)
    """
    mean: float
    stddev: float
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    skew: float = 0.0

    def copy(self) -> "DistributionParams":
        """Create a copy of these parameters."""
        return DistributionParams(
            mean=self.mean,
            stddev=self.stddev,
            min_val=self.min_val,
            max_val=self.max_val,
            skew=self.skew,
        )


class RNG:
    """
    Random number generator with human-like distributions.

    Usage:
        rng = RNG(seed=42)  # Reproducible
        rng = RNG()  # Random seed

        # Basic Gaussian
        delay = rng.gauss(mean=1.5, stddev=0.3)

        # Truncated Gaussian (bounded)
        delay = rng.truncated_gauss(mean=1.5, stddev=0.3, min_val=0.5, max_val=3.0)

        # Skewed distribution (right-skewed = usually fast, sometimes slow)
        delay = rng.skewed_gauss(mean=1.0, stddev=0.3, skew=2.0)

        # Meta-random: vary the parameters themselves
        varied_params = rng.vary_params(base_params, variance=0.15)
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        on_log: Optional[Callable[[str], None]] = None,
        debug: bool = False,
    ):
        self._seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self._random = random.Random(self._seed)
        self._log_callback = on_log
        self._debug = debug
        self._sample_count = 0

    def _log(self, message: str) -> None:
        if self._log_callback:
            self._log_callback(f"[RNG] {message}")
        elif self._debug:
            print(f"[RNG] {message}")

    @property
    def seed(self) -> int:
        return self._seed

    def reseed(self, seed: Optional[int] = None) -> int:
        self._seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self._random.seed(self._seed)
        self._sample_count = 0
        self._log(f"Reseeded with {self._seed}")
        return self._seed

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        value = self._random.uniform(low, high)
        self._sample_count += 1
        if self._debug:
            self._log(f"uniform({low}, {high}) = {value:.4f}")
        return value

    def gauss(self, mean: float = 0.0, stddev: float = 1.0) -> float:
        value = self._random.gauss(mean, stddev)
        self._sample_count += 1
        if self._debug:
            self._log(f"gauss({mean}, {stddev}) = {value:.4f}")
        return value

    def truncated_gauss(
        self,
        mean: float,
        stddev: float,
        min_val: float,
        max_val: float,
        max_attempts: int = 100,
    ) -> float:
        for _ in range(max_attempts):
            value = self._random.gauss(mean, stddev)
            if min_val <= value <= max_val:
                self._sample_count += 1
                if self._debug:
                    self._log(
                        f"truncated_gauss({mean}, {stddev}, [{min_val}, {max_val}]) = {value:.4f}"
                    )
                return value

        value = max(min_val, min(max_val, self._random.gauss(mean, stddev)))
        self._sample_count += 1
        if self._debug:
            self._log(
                f"truncated_gauss({mean}, {stddev}, [{min_val}, {max_val}]) = {value:.4f} (clamped)"
            )
        return value

    def skewed_gauss(
        self,
        mean: float,
        stddev: float,
        skew: float,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> float:
        if abs(skew) < 0.001:
            value = self._random.gauss(mean, stddev)
        else:
            base = self._random.gauss(0, 1)
            exp_component = self._random.expovariate(1.0 / abs(skew)) if skew != 0 else 0

            if skew > 0:
                value = mean + stddev * base + exp_component
            else:
                value = mean + stddev * base - exp_component

        if min_val is not None:
            value = max(min_val, value)
        if max_val is not None:
            value = min(max_val, value)

        self._sample_count += 1
        if self._debug:
            self._log(
                f"skewed_gauss({mean}, {stddev}, skew={skew}) = {value:.4f}"
            )
        return value

    def from_params(self, params: DistributionParams) -> float:
        if abs(params.skew) > 0.001:
            return self.skewed_gauss(
                mean=params.mean,
                stddev=params.stddev,
                skew=params.skew,
                min_val=params.min_val,
                max_val=params.max_val,
            )
        elif params.min_val is not None and params.max_val is not None:
            return self.truncated_gauss(
                mean=params.mean,
                stddev=params.stddev,
                min_val=params.min_val,
                max_val=params.max_val,
            )
        else:
            value = self.gauss(mean=params.mean, stddev=params.stddev)
            if params.min_val is not None:
                value = max(params.min_val, value)
            if params.max_val is not None:
                value = min(params.max_val, value)
            return value

    def vary_params(
        self,
        params: DistributionParams,
        variance: float = 0.1,
        vary_skew: bool = True,
    ) -> DistributionParams:
        """
        Create a varied copy of distribution parameters.

        This is the META-RANDOM function: it randomizes the randomness.
        Call this at session start to create a unique "personality" that
        persists through the session.
        """
        varied = params.copy()

        mean_factor = self._random.gauss(1.0, variance)
        varied.mean = params.mean * mean_factor

        stddev_factor = self._random.gauss(1.0, variance)
        varied.stddev = max(0.001, params.stddev * stddev_factor)

        if params.min_val is not None:
            varied.min_val = params.min_val * mean_factor
        if params.max_val is not None:
            varied.max_val = params.max_val * mean_factor

        if vary_skew and abs(params.skew) > 0.001:
            skew_factor = self._random.gauss(1.0, variance)
            varied.skew = params.skew * skew_factor

        if self._debug:
            self._log(
                f"vary_params: mean {params.mean:.3f} -> {varied.mean:.3f}, "
                f"stddev {params.stddev:.3f} -> {varied.stddev:.3f}"
            )

        return varied

    def vary_value(self, value: float, variance: float = 0.1) -> float:
        factor = self._random.gauss(1.0, variance)
        result = value * factor
        if self._debug:
            self._log(f"vary_value({value:.3f}, {variance}) = {result:.3f}")
        return result

    def choice(self, options: List) -> any:
        return self._random.choice(options)

    def weighted_choice(self, options: List, weights: List[float]) -> any:
        return self._random.choices(options, weights=weights, k=1)[0]

    def chance(self, probability: float) -> bool:
        return self._random.random() < probability

    def sample_many(self, params: DistributionParams, count: int) -> List[float]:
        return [self.from_params(params) for _ in range(count)]

    def get_stats(self) -> dict:
        return {
            "seed": self._seed,
            "sample_count": self._sample_count,
            "debug": self._debug,
        }


TIMING_PROFILES = {
    "instant": DistributionParams(mean=0.05, stddev=0.02, min_val=0.02, max_val=0.15),
    "fast": DistributionParams(mean=0.15, stddev=0.05, min_val=0.08, max_val=0.3),
    "normal": DistributionParams(mean=0.4, stddev=0.15, min_val=0.2, max_val=0.8),
    "careful": DistributionParams(mean=0.8, stddev=0.25, min_val=0.4, max_val=1.5),
    "slow": DistributionParams(mean=1.5, stddev=0.4, min_val=0.8, max_val=3.0),
    "fast_sometimes_slow": DistributionParams(
        mean=0.3, stddev=0.1, min_val=0.15, max_val=2.0, skew=1.5
    ),
    "thinking_pause": DistributionParams(
        mean=2.0, stddev=0.8, min_val=1.0, max_val=5.0, skew=0.5
    ),
}
