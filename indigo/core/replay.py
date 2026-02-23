"""
Replay Path Generation

Processes raw mouse recordings into normalized movement templates,
then generates contextual variations at runtime. Duck-types with WindMouse:
has generate(), start_session(), stop_session().
"""

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable

from .windmouse import Path, Point, PathStats, WindMouseConfig
from .recorder import RawRecording, Recorder, RECORDINGS_DIR


@dataclass
class MovementTemplate:
    points: List[Tuple[float, float]]   # normalized (0,0)->(~1,0)
    timings: List[float]                # [0.0 ... 1.0] per point
    duration: float                     # original duration in seconds
    distance: float                     # original displacement in pixels
    has_click: bool


@dataclass
class PathLibrary:
    script: str
    templates: Dict[str, List[MovementTemplate]]  # "short"/"medium"/"long" bins
    recording_count: int
    template_count: int


# Distance bin thresholds
SHORT_MAX = 100
MEDIUM_MAX = 300

# Segmentation parameters
REST_GAP = 0.150          # 150ms no movement = segment boundary
MIN_DISPLACEMENT = 5      # pixels
MIN_POINTS = 3


def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def _bin_name(distance: float) -> str:
    if distance < SHORT_MAX:
        return "short"
    elif distance < MEDIUM_MAX:
        return "medium"
    else:
        return "long"


def _smooth_points(
    points: List[Tuple[float, float]],
    timings: List[float],
    window: int = 3,
) -> Tuple[List[Tuple[float, float]], List[float]]:
    """Moving-average smooth, preserving first and last points."""
    n = len(points)
    if n <= window:
        return points, timings

    smoothed = [points[0]]
    smooth_t = [timings[0]]
    half = window // 2

    for i in range(1, n - 1):
        lo = max(1, i - half)
        hi = min(n - 1, i + half + 1)
        avg_x = sum(p[0] for p in points[lo:hi]) / (hi - lo)
        avg_y = sum(p[1] for p in points[lo:hi]) / (hi - lo)
        avg_t = sum(timings[lo:hi]) / (hi - lo)
        smoothed.append((avg_x, avg_y))
        smooth_t.append(avg_t)

    smoothed.append(points[-1])
    smooth_t.append(timings[-1])
    return smoothed, smooth_t


def _downsample(
    points: List[Tuple[float, float]],
    timings: List[float],
    max_points: int = 80,
) -> Tuple[List[Tuple[float, float]], List[float]]:
    """Evenly downsample to max_points, always keeping first and last."""
    n = len(points)
    if n <= max_points:
        return points, timings

    # Pick evenly-spaced indices, always including 0 and n-1
    step = (n - 1) / (max_points - 1)
    indices = [round(i * step) for i in range(max_points)]
    indices[-1] = n - 1  # ensure last point included

    return [points[i] for i in indices], [timings[i] for i in indices]


def process_recording(recording: RawRecording) -> List[MovementTemplate]:
    """Process a raw recording into normalized movement templates."""
    events = recording.events
    if not events:
        return []

    # Step 1: Segment by rest gaps
    segments = []
    current_segment = [events[0]]

    for i in range(1, len(events)):
        gap = events[i].t - events[i - 1].t
        if gap > REST_GAP and events[i].kind == "move":
            if current_segment:
                segments.append(current_segment)
            current_segment = [events[i]]
        else:
            current_segment.append(events[i])

    if current_segment:
        segments.append(current_segment)

    # Step 2-4: Filter, normalize, tag
    templates = []
    for seg in segments:
        # Extract move events for the path
        moves = [e for e in seg if e.kind == "move"]
        if len(moves) < MIN_POINTS:
            continue

        # Compute displacement
        start = (moves[0].x, moves[0].y)
        end = (moves[-1].x, moves[-1].y)
        displacement = _distance(start, end)
        if displacement < MIN_DISPLACEMENT:
            continue

        # Check if segment contains a click
        has_click = any(e.kind == "press" and e.button == "left" for e in seg)

        # Original duration
        duration = moves[-1].t - moves[0].t
        if duration <= 0:
            continue

        # Normalize: translate to origin
        raw_points = [(m.x - start[0], m.y - start[1]) for m in moves]
        raw_times = [m.t - moves[0].t for m in moves]

        # Direction vector to endpoint
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        angle = math.atan2(dy, dx)

        # Rotate so path points along +X axis
        cos_a = math.cos(-angle)
        sin_a = math.sin(-angle)
        rotated = [
            (px * cos_a - py * sin_a, px * sin_a + py * cos_a)
            for px, py in raw_points
        ]

        # Scale so displacement = 1.0
        scaled = [(px / displacement, py / displacement) for px, py in rotated]

        # Normalize timestamps to [0, 1]
        norm_times = [t / duration for t in raw_times]

        # Smooth out hand tremor and downsample dense paths
        scaled, norm_times = _smooth_points(scaled, norm_times)
        scaled, norm_times = _downsample(scaled, norm_times)

        templates.append(MovementTemplate(
            points=scaled,
            timings=norm_times,
            duration=duration,
            distance=displacement,
            has_click=has_click,
        ))

    return templates


def build_library(script_name: str, on_log: Optional[Callable[[str], None]] = None) -> Optional[PathLibrary]:
    """Build a path library from all raw recordings for a script."""
    recordings = Recorder.list_recordings(script_name)
    raw_files = recordings.get(script_name, [])

    if not raw_files:
        if on_log:
            on_log(f"[Replay] No raw recordings found for '{script_name}'")
        return None

    all_templates: List[MovementTemplate] = []
    for path in raw_files:
        recording = Recorder.load_raw(path)
        templates = process_recording(recording)
        all_templates.extend(templates)
        if on_log:
            on_log(f"[Replay] Processed {path}: {len(templates)} templates")

    if not all_templates:
        if on_log:
            on_log(f"[Replay] No valid templates extracted from {len(raw_files)} recordings")
        return None

    # Bin by distance
    bins: Dict[str, List[MovementTemplate]] = {"short": [], "medium": [], "long": []}
    for t in all_templates:
        bins[_bin_name(t.distance)].append(t)

    library = PathLibrary(
        script=script_name,
        templates=bins,
        recording_count=len(raw_files),
        template_count=len(all_templates),
    )

    if on_log:
        on_log(
            f"[Replay] Library built: {library.template_count} templates "
            f"(short={len(bins['short'])}, medium={len(bins['medium'])}, long={len(bins['long'])})"
        )

    return library


def save_library(library: PathLibrary) -> str:
    """Save a path library to disk. Returns the file path."""
    script_dir = os.path.join(RECORDINGS_DIR, library.script)
    os.makedirs(script_dir, exist_ok=True)
    path = os.path.join(script_dir, "library.json")

    data = {
        "script": library.script,
        "recording_count": library.recording_count,
        "template_count": library.template_count,
        "templates": {},
    }

    for bin_name, templates in library.templates.items():
        data["templates"][bin_name] = [
            {
                "points": t.points,
                "timings": t.timings,
                "duration": t.duration,
                "distance": t.distance,
                "has_click": t.has_click,
            }
            for t in templates
        ]

    with open(path, "w") as f:
        json.dump(data, f)

    return path


def load_library(script_name: str) -> Optional[PathLibrary]:
    """Load a path library from disk. Returns None if not found."""
    path = os.path.join(RECORDINGS_DIR, script_name, "library.json")
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        data = json.load(f)

    bins: Dict[str, List[MovementTemplate]] = {}
    for bin_name, templates in data["templates"].items():
        bins[bin_name] = [
            MovementTemplate(
                points=[tuple(p) for p in t["points"]],
                timings=t["timings"],
                duration=t["duration"],
                distance=t["distance"],
                has_click=t["has_click"],
            )
            for t in templates
        ]

    return PathLibrary(
        script=data["script"],
        templates=bins,
        recording_count=data["recording_count"],
        template_count=data["template_count"],
    )


class ReplayGenerator:
    """Generates mouse paths from recorded human movement templates.

    Duck-types with WindMouse: has generate(), start_session(), stop_session().
    """

    def __init__(
        self,
        library: PathLibrary,
        seed: Optional[int] = None,
        on_log: Optional[Callable[[str], None]] = None,
    ):
        self._library = library
        self._seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self._random = random.Random(self._seed)
        self._log_callback = on_log

        self._session_active = False
        self._speed_factor = 1.0
        self._jitter_scale = 1.0

        self._paths_generated = 0

    def _log(self, message: str) -> None:
        if self._log_callback:
            self._log_callback(f"[Replay] {message}")

    @property
    def template_count(self) -> int:
        return self._library.template_count

    def start_session(self, variance: float = 0.15) -> None:
        """Meta-randomize session-level speed and jitter."""
        self._speed_factor = max(0.7, min(1.3, self._random.gauss(1.0, variance)))
        self._jitter_scale = max(0.5, min(1.5, self._random.gauss(1.0, variance)))
        self._session_active = True
        self._log(
            f"Session started: speed_factor={self._speed_factor:.3f}, "
            f"jitter_scale={self._jitter_scale:.3f}, "
            f"templates={self._library.template_count}"
        )

    def stop_session(self) -> None:
        self._session_active = False
        self._log("Session stopped")

    def _pick_template(self, distance: float) -> Optional[MovementTemplate]:
        """Pick a random template from the appropriate distance bin."""
        primary_bin = _bin_name(distance)
        templates = self._library.templates.get(primary_bin, [])

        if templates:
            return self._random.choice(templates)

        # Fallback: try adjacent bins in order of proximity
        fallback_order = {
            "short": ("medium", "long"),
            "medium": ("short", "long"),
            "long": ("medium", "short"),
        }
        for fallback in fallback_order.get(primary_bin, ()):
            templates = self._library.templates.get(fallback, [])
            if templates:
                return self._random.choice(templates)

        return None

    def generate(
        self,
        start_x: float, start_y: float,
        end_x: float, end_y: float,
        config=None,  # ignored, for API compatibility
    ) -> Optional[Path]:
        """Generate a path from recorded templates.

        Returns a Path with timings/duration set, or None if no templates available.
        """
        dx = end_x - start_x
        dy = end_y - start_y
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < 1:
            p = Point(end_x, end_y)
            return Path(
                points=[p], start=p, end=p,
                stats=PathStats(0, 0, 1, 1.0, 0),
                config=WindMouseConfig(),
                timings=[0.0], duration=0.0,
            )

        template = self._pick_template(distance)
        if template is None:
            return None

        # Target angle
        angle = math.atan2(dy, dx)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        # Denormalize: scale by distance, rotate by angle, translate to start
        points: List[Point] = []

        for i, (nx, ny) in enumerate(template.points):
            # Scale
            sx = nx * distance
            sy = ny * distance

            # Rotate
            rx = sx * cos_a - sy * sin_a
            ry = sx * sin_a + sy * cos_a

            # Translate
            points.append(Point(start_x + rx, start_y + ry))

        # Snap endpoint exactly
        if points:
            points[-1] = Point(end_x, end_y)

        # Compute timings with session variation
        duration = template.duration * self._speed_factor
        # Add per-movement timing jitter
        timings = list(template.timings)
        for i in range(1, len(timings) - 1):
            jitter = self._random.gauss(0, 0.005 * self._jitter_scale)
            timings[i] = max(0.0, min(1.0, timings[i] + jitter))
        # Ensure monotonic
        for i in range(1, len(timings)):
            if timings[i] < timings[i - 1]:
                timings[i] = timings[i - 1]

        # Compute stats
        total_distance = 0.0
        for i in range(1, len(points)):
            total_distance += points[i - 1].distance_to(points[i])

        max_deviation = 0.0
        start_pt = Point(start_x, start_y)
        end_pt = Point(end_x, end_y)
        line_dx = end_x - start_x
        line_dy = end_y - start_y
        line_len = math.sqrt(line_dx * line_dx + line_dy * line_dy)

        if line_len > 0.001:
            ndx = line_dx / line_len
            ndy = line_dy / line_len
            for p in points:
                px = p.x - start_x
                py = p.y - start_y
                proj = px * ndx + py * ndy
                cx = start_x + proj * ndx
                cy = start_y + proj * ndy
                dev = math.sqrt((p.x - cx) ** 2 + (p.y - cy) ** 2)
                max_deviation = max(max_deviation, dev)

        curvature = total_distance / distance if distance > 0 else 1.0

        stats = PathStats(
            total_distance=total_distance,
            direct_distance=distance,
            point_count=len(points),
            curvature_ratio=curvature,
            max_deviation=max_deviation,
        )

        self._paths_generated += 1

        return Path(
            points=points,
            start=start_pt,
            end=end_pt,
            stats=stats,
            config=WindMouseConfig(),
            timings=timings,
            duration=duration,
        )
