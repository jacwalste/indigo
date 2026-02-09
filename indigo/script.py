"""
Script Engine

Base class for bot scripts. Scripts implement on_start/loop/on_stop
and run in a background thread with a stop flag.
"""

from __future__ import annotations

import math
import time
import threading
from dataclasses import dataclass
from typing import Optional, Callable, List, Set, Tuple, TYPE_CHECKING

from .vision import Vision, Color, ColorCluster, GameRegions
from .input import Input
from .core.delay import Delay
from .core.rng import RNG
from .core.timing import FAST_ACTION, NORMAL_ACTION, CAREFUL_ACTION

if TYPE_CHECKING:
    from .idle import IdleBehavior


@dataclass
class ScriptConfig:
    """Configuration for a script."""
    name: str
    max_runtime_hours: float = 6.0


@dataclass
class ScriptContext:
    """Runtime context passed to scripts."""
    vision: Vision
    input: Input
    delay: Delay
    rng: RNG
    stop_flag: threading.Event
    idle: Optional[IdleBehavior] = None


def _build_drop_orders() -> List[List[int]]:
    """Build all drop traversal patterns."""
    orders = []

    # Columns: top-to-bottom, left-to-right
    cols = []
    for c in range(4):
        for r in range(7):
            cols.append(r * 4 + c)
    orders.append(cols)

    # Rows: left-to-right, top-to-bottom
    orders.append(list(range(28)))

    # Snake columns: top-to-bottom then bottom-to-top alternating
    snake_cols = []
    for c in range(4):
        col_slots = [r * 4 + c for r in range(7)]
        if c % 2 == 1:
            col_slots.reverse()
        snake_cols.extend(col_slots)
    orders.append(snake_cols)

    # Snake rows: left-to-right then right-to-left alternating
    snake_rows = []
    for r in range(7):
        row_slots = [r * 4 + c for c in range(4)]
        if r % 2 == 1:
            row_slots.reverse()
        snake_rows.extend(row_slots)
    orders.append(snake_rows)

    return orders


_DROP_ORDERS = _build_drop_orders()


class Script:
    """Base class for bot scripts."""

    def __init__(
        self,
        config: ScriptConfig,
        ctx: ScriptContext,
        on_log: Optional[Callable[[str], None]] = None,
    ):
        self.config = config
        self.ctx = ctx
        self._log_callback = on_log
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._error: Optional[Exception] = None

    def _log(self, message: str) -> None:
        if self._log_callback:
            self._log_callback(f"[{self.config.name}] {message}")
        else:
            print(f"[{self.config.name}] {message}")

    @property
    def should_stop(self) -> bool:
        """Check stop flag and max runtime."""
        if self.ctx.stop_flag.is_set():
            return True
        if self._start_time is not None:
            elapsed_hours = (time.time() - self._start_time) / 3600.0
            if elapsed_hours >= self.config.max_runtime_hours:
                self._log(f"Max runtime reached ({self.config.max_runtime_hours}h)")
                return True
        return False

    def elapsed_str(self) -> str:
        """Format elapsed time as 'Xm XXs'."""
        elapsed = (time.time() - self._start_time) if self._start_time else 0
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        return f"{mins}m{secs:02d}s"

    def click_target(self, x: int, y: int) -> None:
        """Click a game object with occasional human-like multi-clicks.

        ~18% chance of 2+ clicks with variable timing and slight position
        offsets, simulating impatient or imprecise clicking.
        """
        rng = self.ctx.rng
        inp = self.ctx.input

        # Always do the first click
        inp.click(x, y)

        # ~18% chance of extra clicks
        if rng.chance(0.18):
            # 1-3 extra clicks
            extras = int(rng.truncated_gauss(1.5, 0.8, 1, 4))
            for _ in range(extras):
                # Variable gap: sometimes rapid, sometimes a quick pause
                if rng.chance(0.4):
                    # Rapid — barely any gap
                    gap = rng.truncated_gauss(0.06, 0.02, 0.03, 0.12)
                else:
                    # Quick pause between clicks
                    gap = rng.truncated_gauss(0.25, 0.1, 0.1, 0.5)
                time.sleep(gap)

                # Slight position offset — not exact same pixel
                ox = int(rng.truncated_gauss(0, 4, -10, 10))
                oy = int(rng.truncated_gauss(0, 4, -10, 10))
                inp.click(x + ox, y + oy)

    def find_target(self, color: Color, tolerance: int = 15,
                    min_area: int = 40) -> Optional[ColorCluster]:
        """Find the largest color cluster in the game view."""
        clusters = self.ctx.vision.find_color_clusters(
            GameRegions.GAME_VIEW, color,
            tolerance=tolerance, min_area=min_area,
        )
        return clusters[0] if clusters else None

    def is_target_near(self, color: Color, point: Tuple[int, int],
                       max_distance: int = 60, tolerance: int = 15,
                       min_area: int = 40) -> bool:
        """Check if a color cluster still exists near a screen coordinate.

        Used to detect if the specific object we clicked is still there,
        even when similar objects exist nearby.

        Args:
            color: The target color to look for.
            point: Screen coordinates (x, y) to check proximity to.
            max_distance: Max pixel distance from point to cluster center.
            tolerance: Color matching tolerance.
            min_area: Minimum cluster area.

        Returns:
            True if a matching cluster center is within max_distance of point.
        """
        clusters = self.ctx.vision.find_color_clusters(
            GameRegions.GAME_VIEW, color,
            tolerance=tolerance, min_area=min_area,
        )
        px, py = point
        for cluster in clusters:
            cx, cy = cluster.center
            dist = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            if dist <= max_distance:
                return True
        return False

    def search_for_target(self, color: Color, tolerance: int = 15,
                         min_area: int = 40) -> Optional[ColorCluster]:
        """Zoom out and rotate camera to search for a target color.

        Used when find_target() fails repeatedly. Zooms out a bit for wider
        view, then rotates camera in steps checking for the target after each.

        Returns the first ColorCluster found, or None after a full rotation.
        """
        rng = self.ctx.rng
        inp = self.ctx.input

        # Zoom out 2-7 ticks for a wider view
        zoom_ticks = int(rng.truncated_gauss(4.5, 1.5, 2, 7))
        inp.scroll(dy=-zoom_ticks)
        self._log(f"Search: zoomed out {zoom_ticks} ticks")
        self.ctx.delay.sleep_range(0.2, 0.6)

        # Rotate camera in steps, checking after each
        key = rng.choice(['left', 'right'])
        for step in range(8):
            if self.should_stop:
                return None

            # Hold arrow key for a rotation step (~45 degrees)
            hold = rng.truncated_gauss(0.6, 0.15, 0.35, 0.9)
            inp.key_hold(key, hold)
            self.ctx.delay.sleep_range(0.15, 0.5)

            # Check for target
            cluster = self.find_target(color, tolerance=tolerance, min_area=min_area)
            if cluster:
                self._log(f"Search: found target after {step + 1} rotation steps")
                return cluster

        self._log("Search: target not found after full rotation")
        return None

    def click_region_jittered(self, region: "GameRegions") -> None:
        """Click a random point within a Region, Gaussian-biased toward center."""
        rng = self.ctx.rng
        cx, cy = region.center
        margin = 1
        jx = int(rng.truncated_gauss(0, region.width * 0.25,
                                      -(region.width // 2 - margin), region.width // 2 - margin))
        jy = int(rng.truncated_gauss(0, region.height * 0.25,
                                      -(region.height // 2 - margin), region.height // 2 - margin))
        sx, sy = self.ctx.vision.to_screen(cx + jx, cy + jy)
        self.ctx.input.click(sx, sy)

    def deposit_all(self) -> int:
        """Click Deposit All in an open deposit box, verify, close.

        Assumes the deposit box interface is already open.
        Returns the number of items deposited.
        """
        vision = self.ctx.vision
        before = vision.count_inventory_items()

        # Click Deposit All button
        self.click_region_jittered(GameRegions.DEPOSIT_ALL_BUTTON)
        self.ctx.delay.sleep(NORMAL_ACTION)

        if self.should_stop:
            return 0

        self.ctx.delay.sleep_range(0.5, 1.0)

        # Verify inventory is empty, retry once if needed
        remaining = vision.count_inventory_items()
        if remaining > 0 and not self.should_stop:
            self._log(f"Still {remaining} items after deposit, retrying")
            self.click_region_jittered(GameRegions.DEPOSIT_ALL_BUTTON)
            self.ctx.delay.sleep(NORMAL_ACTION)
            remaining = vision.count_inventory_items()

        # Close deposit box with Escape
        self.ctx.delay.sleep_range(0.3, 0.6)
        self.ctx.input.key_tap('esc')
        self.ctx.delay.sleep(NORMAL_ACTION)

        return max(0, before - remaining)

    def randomize_drop_threshold(self, mean: float = 14, stddev: float = 3,
                                 min_val: float = 10, max_val: float = 20) -> int:
        """Pick a randomized drop threshold and return it."""
        threshold = int(self.ctx.rng.truncated_gauss(
            mean=mean, stddev=stddev, min_val=min_val, max_val=max_val,
        ))
        self._log(f"Drop threshold: {threshold}")
        return threshold

    def drop_inventory(self, skip_slots: Optional[Set[int]] = None,
                       expected: Optional[int] = None) -> int:
        """Drop inventory items with human-like imperfections.

        Randomizes traversal pattern, drop speed, and occasionally makes
        mistakes (misclicks, wrong click type, brief pauses).

        Args:
            skip_slots: Slot indices to never drop (e.g. {0} for equipped tool).
            expected: Max items to drop. Prevents over-dropping from vision bleed.

        Returns:
            Number of items dropped.
        """
        rng = self.ctx.rng

        # Pick a random traversal pattern
        order = rng.choice(_DROP_ORDERS)

        # Session-varied base speed: some drops faster, some slower
        speed_mult = rng.truncated_gauss(1.0, 0.15, 0.7, 1.4)

        dropped = 0
        for slot_idx in order:
            if self.should_stop:
                break
            if expected is not None and dropped >= expected:
                break
            if skip_slots and slot_idx in skip_slots:
                continue
            if not self.ctx.vision.slot_has_item(slot_idx):
                continue

            cx, cy = self.ctx.vision.slot_screen_click_point(slot_idx)

            # ~4% chance: misclick (off by some pixels), then correct
            if rng.chance(0.04):
                ox = int(rng.truncated_gauss(0, 12, -20, 20))
                oy = int(rng.truncated_gauss(0, 12, -20, 20))
                self.ctx.input.shift_click(cx + ox, cy + oy)
                self.ctx.delay.sleep(FAST_ACTION, include_pauses=False)
                # Re-click correctly
                self.ctx.input.shift_click(cx, cy)

            # ~3% chance: left-click instead of shift-click, then correct
            elif rng.chance(0.03):
                self.ctx.input.click(cx, cy)
                pause = rng.truncated_gauss(0.4, 0.15, 0.2, 0.8)
                self.ctx.delay.sleep_range(pause * 0.8, pause * 1.2)
                # Correct with shift-click
                self.ctx.input.shift_click(cx, cy)

            else:
                self.ctx.input.shift_click(cx, cy)

            # Variable delay between drops
            # ~8% chance: brief pause mid-drop (distracted/thinking)
            if rng.chance(0.08):
                pause = rng.truncated_gauss(0.8, 0.3, 0.4, 1.8)
                self.ctx.delay.sleep_range(pause * 0.8, pause * 1.2)
            else:
                # Per-item speed variation around FAST_ACTION range
                item_speed = speed_mult * rng.truncated_gauss(1.0, 0.1, 0.8, 1.3)
                base_min = FAST_ACTION.min_val * item_speed
                base_max = FAST_ACTION.max_val * item_speed
                self.ctx.delay.sleep_range(base_min, base_max,
                                           include_pauses=False)

            dropped += 1
        return dropped

    def on_start(self) -> None:
        """Called once before the loop starts. Override in subclass."""
        pass

    def loop(self) -> None:
        """Called repeatedly. Override in subclass."""
        raise NotImplementedError("Subclasses must implement loop()")

    def on_stop(self) -> None:
        """Called once after the loop ends. Override in subclass."""
        pass

    def _run(self) -> None:
        """Internal run method for the background thread."""
        self._start_time = time.time()
        self._log("Starting")
        try:
            self.on_start()
            while not self.should_stop:
                self.loop()
        except Exception as e:
            self._log(f"Error: {e}")
            self._error = e
        finally:
            self.on_stop()
            elapsed = time.time() - self._start_time
            self._log(f"Stopped after {elapsed:.1f}s")

    def start(self) -> None:
        """Start the script in a background thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the script to stop."""
        self.ctx.stop_flag.set()

    def wait(self, timeout: Optional[float] = None) -> None:
        """Wait for the script thread to finish."""
        if self._thread:
            self._thread.join(timeout=timeout)
