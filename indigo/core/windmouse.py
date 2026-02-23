"""
WindMouse Algorithm

Physics-based mouse path generation for human-like trajectories.
Uses gravity (pulls toward target) and wind (random drift) to create
natural, non-reducible paths that evade detection.

Two-phase behavior:
- Far from target (dist >= D_0): Wind has random perturbations, creates wandering
- Close to target (dist < D_0): Wind dampens, controlled convergence

Design principle: "If we're following a random formula, we have to randomize the randomness."
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Tuple


@dataclass
class Point:
    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def as_int_tuple(self) -> Tuple[int, int]:
        return (int(round(self.x)), int(round(self.y)))


@dataclass
class WindMouseConfig:
    gravity: float = 9.0
    wind: float = 3.0
    max_velocity: float = 15.0
    distance_threshold: float = 12.0
    variance: float = 0.1

    def copy(self) -> "WindMouseConfig":
        return WindMouseConfig(
            gravity=self.gravity,
            wind=self.wind,
            max_velocity=self.max_velocity,
            distance_threshold=self.distance_threshold,
            variance=self.variance,
        )


@dataclass
class PathStats:
    total_distance: float
    direct_distance: float
    point_count: int
    curvature_ratio: float
    max_deviation: float


@dataclass
class Path:
    points: List[Point]
    start: Point
    end: Point
    stats: PathStats
    config: WindMouseConfig
    timings: Optional[List[float]] = None  # normalized [0..1] per point
    duration: Optional[float] = None       # suggested total duration in seconds

    def get_points_as_tuples(self) -> List[Tuple[float, float]]:
        return [p.as_tuple() for p in self.points]

    def get_points_as_int_tuples(self) -> List[Tuple[int, int]]:
        return [p.as_int_tuple() for p in self.points]


WINDMOUSE_CONFIGS: Dict[str, WindMouseConfig] = {
    "default": WindMouseConfig(gravity=7.0, wind=5.0, max_velocity=15.0, distance_threshold=12.0),
    "direct": WindMouseConfig(gravity=10.0, wind=2.5, max_velocity=18.0, distance_threshold=10.0),
    "wandering": WindMouseConfig(gravity=5.0, wind=7.0, max_velocity=12.0, distance_threshold=15.0),
    "precise": WindMouseConfig(gravity=8.0, wind=3.0, max_velocity=10.0, distance_threshold=8.0),
    "tired": WindMouseConfig(gravity=5.5, wind=5.0, max_velocity=10.0, distance_threshold=14.0, variance=0.15),
    "caffeinated": WindMouseConfig(gravity=9.0, wind=5.0, max_velocity=20.0, distance_threshold=10.0, variance=0.12),
}


class WindMouse:
    """WindMouse path generator with human-like trajectories."""

    def __init__(
        self,
        config: Optional[WindMouseConfig] = None,
        seed: Optional[int] = None,
        on_log: Optional[Callable[[str], None]] = None,
        debug: bool = False,
    ):
        self._default_config = config.copy() if config else WINDMOUSE_CONFIGS["default"].copy()
        self._seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self._random = random.Random(self._seed)
        self._log_callback = on_log
        self._debug = debug

        self._session_active = False
        self._session_config: Optional[WindMouseConfig] = None

        self._paths_generated = 0
        self._total_curvature_sum = 0.0
        self._total_points_sum = 0
        self._max_curvature_seen = 0.0
        self._min_curvature_seen = float('inf')

    def _log(self, message: str) -> None:
        if self._log_callback:
            self._log_callback(f"[WindMouse] {message}")
        elif self._debug:
            print(f"[WindMouse] {message}")

    @property
    def seed(self) -> int:
        return self._seed

    def reseed(self, seed: Optional[int] = None) -> int:
        self._seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self._random.seed(self._seed)
        self._log(f"Reseeded with {self._seed}")
        return self._seed

    def start_session(self, variance: float = 0.12) -> WindMouseConfig:
        base = self._default_config
        self._session_config = WindMouseConfig(
            gravity=max(1.0, base.gravity * self._random.gauss(1.0, variance)),
            wind=max(0.5, base.wind * self._random.gauss(1.0, variance)),
            max_velocity=max(3.0, base.max_velocity * self._random.gauss(1.0, variance)),
            distance_threshold=max(5.0, base.distance_threshold * self._random.gauss(1.0, variance)),
            variance=max(0.01, base.variance * self._random.gauss(1.0, variance / 2)),
        )
        self._session_active = True
        self._log(
            f"Session started: G={self._session_config.gravity:.2f}, "
            f"W={self._session_config.wind:.2f}, "
            f"M={self._session_config.max_velocity:.2f}, "
            f"D={self._session_config.distance_threshold:.2f}"
        )
        return self._session_config

    def stop_session(self) -> None:
        self._session_active = False
        self._session_config = None
        self._log("Session stopped")

    def _get_effective_config(self, config: Optional[WindMouseConfig] = None) -> WindMouseConfig:
        if config is not None:
            return config
        if self._session_active and self._session_config is not None:
            return self._session_config
        return self._default_config

    def _apply_per_movement_variation(self, config: WindMouseConfig) -> WindMouseConfig:
        v = config.variance
        return WindMouseConfig(
            gravity=config.gravity * self._random.gauss(1.0, v * 0.5),
            wind=config.wind * self._random.gauss(1.0, v * 0.5),
            max_velocity=config.max_velocity * self._random.gauss(1.0, v * 0.3),
            distance_threshold=config.distance_threshold * self._random.gauss(1.0, v * 0.3),
            variance=config.variance,
        )

    def _calculate_deviation(self, point: Point, start: Point, end: Point) -> float:
        dx = end.x - start.x
        dy = end.y - start.y
        line_length = math.sqrt(dx * dx + dy * dy)
        if line_length < 0.001:
            return point.distance_to(start)
        dx /= line_length
        dy /= line_length
        px = point.x - start.x
        py = point.y - start.y
        proj_length = px * dx + py * dy
        closest_x = start.x + proj_length * dx
        closest_y = start.y + proj_length * dy
        return math.sqrt((point.x - closest_x) ** 2 + (point.y - closest_y) ** 2)

    def _generate_path(
        self,
        start_x: float, start_y: float,
        end_x: float, end_y: float,
        config: WindMouseConfig,
    ) -> List[Point]:
        G_0 = max(1.0, config.gravity)
        W_0 = max(0.1, config.wind)
        M_0 = max(3.0, config.max_velocity)
        D_0 = max(5.0, config.distance_threshold)

        sqrt_3 = math.sqrt(3)
        sqrt_5 = math.sqrt(5)

        points: List[Point] = []
        x, y = start_x, start_y
        v_x, v_y = 0.0, 0.0
        w_x, w_y = 0.0, 0.0

        dist = math.sqrt((end_x - x) ** 2 + (end_y - y) ** 2)
        points.append(Point(x, y))

        if dist < 1:
            return points

        # Arc bias: perpendicular force that curves the path consistently.
        # ~70% of paths get an arc, 30% go straight (variety).
        arc_px, arc_py = 0.0, 0.0
        if self._random.random() < 0.70 and dist > 30:
            # Perpendicular to the direct line
            dx = end_x - start_x
            dy = end_y - start_y
            line_len = math.sqrt(dx * dx + dy * dy)
            if line_len > 0.001:
                perp_x = -dy / line_len
                perp_y = dx / line_len
                # Magnitude scales with distance: longer moves get more arc
                arc_strength = self._random.gauss(0, 0.3) * min(dist / 200.0, 1.0)
                arc_px = perp_x * arc_strength
                arc_py = perp_y * arc_strength

        max_iterations = max(100, min(5000, int(dist / M_0 * 20)))
        iteration = 0

        while dist > 1 and iteration < max_iterations:
            iteration += 1
            w_mag = min(W_0, dist)

            if dist >= D_0:
                w_x = w_x / sqrt_3 + (2 * self._random.random() - 1) * w_mag / sqrt_5
                w_y = w_y / sqrt_3 + (2 * self._random.random() - 1) * w_mag / sqrt_5
            else:
                # Convergence zone: dampen wind and velocity proportional to closeness
                dampen = dist / D_0  # 1.0 at edge, 0.0 at target
                w_x *= dampen * 0.5
                w_y *= dampen * 0.5
                v_x *= 0.6 + 0.4 * dampen  # scale from 1.0 (edge) to 0.6 (target)
                v_y *= 0.6 + 0.4 * dampen

            dir_x = end_x - x
            dir_y = end_y - y
            dir_len = math.sqrt(dir_x * dir_x + dir_y * dir_y)

            if dir_len > 0.001:
                dir_x /= dir_len
                dir_y /= dir_len

            g_mag = min(G_0, dist)
            # Arc bias fades as we approach target (only active in far phase)
            arc_fade = min(dist / (D_0 * 2), 1.0) if dist >= D_0 else 0.0
            v_x += w_x + g_mag * dir_x + arc_px * arc_fade
            v_y += w_y + g_mag * dir_y + arc_py * arc_fade

            v_mag = math.sqrt(v_x * v_x + v_y * v_y)
            min_velocity = max(1.0, M_0 * 0.1)
            if v_mag < min_velocity and dist > D_0:
                v_x = dir_x * min_velocity
                v_y = dir_y * min_velocity
                v_mag = min_velocity

            # Clamp max speed, scale down near target
            effective_max = M_0 if dist >= D_0 else M_0 * (0.3 + 0.7 * dist / D_0)
            if v_mag > effective_max:
                random_clamp = effective_max / 2 + self._random.random() * effective_max / 2
                v_x = (v_x / v_mag) * random_clamp
                v_y = (v_y / v_mag) * random_clamp

            x += v_x
            y += v_y

            dist = math.sqrt((end_x - x) ** 2 + (end_y - y) ** 2)

            points.append(Point(x, y))

        if points[-1].x != end_x or points[-1].y != end_y:
            points.append(Point(end_x, end_y))

        return points

    def _maybe_add_overshoot(self, points: List[Point],
                             end_x: float, end_y: float) -> List[Point]:
        """~20% chance to overshoot the target and correct back on longer moves."""
        if len(points) < 5:
            return points

        direct_dist = points[0].distance_to(Point(end_x, end_y))
        if direct_dist < 80 or self._random.random() > 0.20:
            return points

        # Remove the final snap-to-target point if present
        if points[-1].x == end_x and points[-1].y == end_y:
            points = points[:-1]

        # Approach direction from the last few points
        last = points[-1]
        prev = points[-3] if len(points) >= 3 else points[0]
        dx = last.x - prev.x
        dy = last.y - prev.y
        mag = math.sqrt(dx * dx + dy * dy)
        if mag < 0.1:
            points.append(Point(end_x, end_y))
            return points
        dx /= mag
        dy /= mag

        # Overshoot: 5-15px past target, with slight perpendicular drift
        overshoot_dist = self._random.gauss(10, 3)
        overshoot_dist = max(5, min(20, overshoot_dist))
        perp_drift = self._random.gauss(0, 3)

        over_x = end_x + dx * overshoot_dist + (-dy) * perp_drift
        over_y = end_y + dy * overshoot_dist + dx * perp_drift

        # Add overshoot point
        points.append(Point(over_x, over_y))

        # Correction: 1-3 intermediate points curving back
        corrections = self._random.randint(1, 3)
        for i in range(corrections):
            t = (i + 1) / (corrections + 1)
            cx = over_x + (end_x - over_x) * t + self._random.gauss(0, 1.5)
            cy = over_y + (end_y - over_y) * t + self._random.gauss(0, 1.5)
            points.append(Point(cx, cy))

        # Final snap to target
        points.append(Point(end_x, end_y))
        return points

    def generate(
        self,
        start_x: float, start_y: float,
        end_x: float, end_y: float,
        config: Optional[WindMouseConfig] = None,
    ) -> Path:
        base_config = self._get_effective_config(config)
        movement_config = self._apply_per_movement_variation(base_config)
        points = self._generate_path(start_x, start_y, end_x, end_y, movement_config)
        points = self._maybe_add_overshoot(points, end_x, end_y)

        start = Point(start_x, start_y)
        end = Point(end_x, end_y)
        direct_distance = start.distance_to(end)

        total_distance = 0.0
        for i in range(1, len(points)):
            total_distance += points[i - 1].distance_to(points[i])

        max_deviation = 0.0
        for point in points:
            deviation = self._calculate_deviation(point, start, end)
            max_deviation = max(max_deviation, deviation)

        curvature_ratio = total_distance / direct_distance if direct_distance > 0 else 1.0

        stats = PathStats(
            total_distance=total_distance,
            direct_distance=direct_distance,
            point_count=len(points),
            curvature_ratio=curvature_ratio,
            max_deviation=max_deviation,
        )

        self._paths_generated += 1
        self._total_curvature_sum += curvature_ratio
        self._total_points_sum += len(points)
        self._max_curvature_seen = max(self._max_curvature_seen, curvature_ratio)
        if curvature_ratio < self._min_curvature_seen:
            self._min_curvature_seen = curvature_ratio

        if self._debug:
            self._log(
                f"Path generated: {len(points)} points, "
                f"curvature={curvature_ratio:.3f}, "
                f"deviation={max_deviation:.1f}px"
            )

        return Path(points=points, start=start, end=end, stats=stats, config=movement_config)

    def get_stats(self) -> dict:
        avg_curvature = self._total_curvature_sum / self._paths_generated if self._paths_generated > 0 else 0.0
        avg_points = self._total_points_sum / self._paths_generated if self._paths_generated > 0 else 0
        return {
            "seed": self._seed,
            "session_active": self._session_active,
            "paths_generated": self._paths_generated,
            "avg_curvature": avg_curvature,
            "avg_points_per_path": avg_points,
            "min_curvature": self._min_curvature_seen if self._paths_generated > 0 else 0,
            "max_curvature": self._max_curvature_seen,
        }

    def format_stats(self) -> str:
        stats = self.get_stats()
        return (
            f"WindMouse Stats:\n"
            f"  Seed: {stats['seed']}\n"
            f"  Session active: {stats['session_active']}\n"
            f"  Paths generated: {stats['paths_generated']}\n"
            f"  Avg curvature: {stats['avg_curvature']:.3f}\n"
            f"  Curvature range: [{stats['min_curvature']:.3f}, {stats['max_curvature']:.3f}]\n"
            f"  Avg points/path: {stats['avg_points_per_path']:.1f}"
        )

    def reset_stats(self) -> None:
        self._paths_generated = 0
        self._total_curvature_sum = 0.0
        self._total_points_sum = 0
        self._max_curvature_seen = 0.0
        self._min_curvature_seen = float('inf')

    def test_harness(self, count: int = 100, min_distance: float = 100.0, max_distance: float = 500.0) -> str:
        self.reset_stats()
        curvatures = []
        point_counts = []

        for _ in range(count):
            start_x = self._random.uniform(0, 1000)
            start_y = self._random.uniform(0, 700)
            distance = self._random.uniform(min_distance, max_distance)
            angle = self._random.uniform(0, 2 * math.pi)
            end_x = start_x + distance * math.cos(angle)
            end_y = start_y + distance * math.sin(angle)

            path = self.generate(start_x, start_y, end_x, end_y)
            curvatures.append(path.stats.curvature_ratio)
            point_counts.append(path.stats.point_count)

        bins = 10
        min_c = min(curvatures)
        max_c = max(curvatures)
        bin_width = (max_c - min_c) / bins if max_c > min_c else 0.1

        histogram = [0] * bins
        for c in curvatures:
            bin_idx = min(int((c - min_c) / bin_width), bins - 1) if bin_width > 0 else 0
            histogram[bin_idx] += 1

        output = [
            f"=== WindMouse Test Harness ({count} paths) ===",
            "",
            self.format_stats(),
            "",
            f"Curvature Distribution:",
        ]

        max_bar = 40
        max_count = max(histogram) if histogram else 1
        for i, h in enumerate(histogram):
            low = min_c + i * bin_width
            high = min_c + (i + 1) * bin_width
            bar_len = int((h / max_count) * max_bar)
            bar = "#" * bar_len
            output.append(f"  {low:.2f}-{high:.2f}: {bar} ({h})")

        output.extend([
            "",
            f"Point Count Range: [{min(point_counts)}, {max(point_counts)}]",
            f"Avg Points: {sum(point_counts) / len(point_counts):.1f}",
        ])

        return "\n".join(output)
