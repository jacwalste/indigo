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

from .vision import Vision, Color, ColorCluster, GameRegions, Region
from .input import Input
from .core.delay import Delay
from .core.rng import RNG
from .core.timing import NORMAL_ACTION

# XP drop detection defaults — configure RuneLite XP Drop plugin to magenta (FF00FF).
# Region calibrated for Fixed Mode: drops appear at x≈505, float from y≈160 up to y≈50.
XP_DROP_REGION = Region(495, 35, 25, 140)
XP_DROP_HUE_LOW = 140       # Magenta hue range in OpenCV HSV (0-180)
XP_DROP_HUE_HIGH = 165
XP_DROP_SAT_MIN = 30        # Low thresholds to catch anti-aliased pixels
XP_DROP_VAL_MIN = 30
XP_DROP_PIXEL_THRESHOLD = 10

if TYPE_CHECKING:
    from .idle import IdleBehavior


@dataclass
class ScriptConfig:
    """Configuration for a script."""
    name: str
    max_runtime_hours: float = 6.0
    reserved_slots: int = 0  # slots that won't be deposited (e.g., axe in inventory)


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
    def inv_full_count(self) -> int:
        """Number of items when inventory is 'full' (28 minus reserved slots)."""
        return 28 - self.config.reserved_slots

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

    def cancel_selection(self) -> None:
        """Cancel any active 'Use X ->' item selection by tab-switching.

        Clicks a nearby tab then back to inventory. Tab switches reliably
        clear the selection cursor in OSRS. Safe to call when nothing is
        selected — just a brief visual tab flicker.
        """
        rng = self.ctx.rng
        inp = self.ctx.input
        vision = self.ctx.vision

        # Click a neighbor tab (equipment or stats — adjacent to inventory)
        away_tab = rng.choice([GameRegions.TAB_EQUIPMENT, GameRegions.TAB_STATS])
        tx, ty = vision.to_screen(*away_tab.center)
        inp.click(tx + int(rng.truncated_gauss(0, 3, -6, 6)),
                  ty + int(rng.truncated_gauss(0, 3, -6, 6)))
        self.ctx.delay.sleep_range(0.08, 0.20)

        # Click back to inventory
        tx, ty = vision.to_screen(*GameRegions.TAB_INVENTORY.center)
        inp.click(tx + int(rng.truncated_gauss(0, 3, -6, 6)),
                  ty + int(rng.truncated_gauss(0, 3, -6, 6)))
        self.ctx.delay.sleep_range(0.08, 0.20)

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

    def check_xp_drop(self) -> bool:
        """Check if an XP drop is visible using HSV hue matching.

        Scans XP_DROP_REGION for magenta-hued pixels. Robust against
        anti-aliasing because HSV hue is preserved when text blends
        with varying backgrounds.

        Configure RuneLite XP Drop plugin to use magenta (FF00FF).
        """
        pixels = self.ctx.vision.detect_hsv_pixels(
            XP_DROP_REGION,
            XP_DROP_HUE_LOW, XP_DROP_HUE_HIGH,
            XP_DROP_SAT_MIN, XP_DROP_VAL_MIN,
        )
        return pixels >= XP_DROP_PIXEL_THRESHOLD

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

    def _ensure_mouse_in_game_view(self) -> None:
        """Move mouse into the game view if it's currently outside.

        Required before scrolling — OSRS only processes scroll/zoom when the
        mouse is hovering over the game viewport.
        """
        gv = GameRegions.GAME_VIEW
        mx, my = self.ctx.input.get_mouse_position()
        sx, sy = self.ctx.vision.to_screen(gv.x, gv.y)
        ex, ey = sx + gv.width, sy + gv.height

        if sx <= mx <= ex and sy <= my <= ey:
            return

        rng = self.ctx.rng
        tx = int(rng.truncated_gauss(gv.width / 2, gv.width * 0.15, 30, gv.width - 30))
        ty = int(rng.truncated_gauss(gv.height / 2, gv.height * 0.15, 30, gv.height - 30))
        target_x, target_y = self.ctx.vision.to_screen(gv.x + tx, gv.y + ty)
        self.ctx.input.move_to(target_x, target_y)

    def _rotate_camera_search(self, key: str, check_fn: Callable):
        """Rotate camera with varied human-like styles, scanning continuously.

        Picks a rotation style randomly each invocation to avoid predictable
        burst-pause-burst patterns. Each style covers roughly one full rotation.

        Long holds are broken into short sub-holds (~0.3s) with vision checks
        between each, so targets are detected mid-rotation instead of only
        after each full rotation step.

        Returns whatever check_fn returns when truthy, or None.
        """
        rng = self.ctx.rng
        inp = self.ctx.input

        style = rng.weighted_choice(
            ['fluid', 'burst', 'sweep', 'tap'],
            [25, 30, 25, 20],
        )
        self._log(f"Search rotation: {style}")

        # Session-varied scan interval — how often we check mid-rotation
        scan_interval = rng.truncated_gauss(0.30, 0.05, 0.20, 0.40)

        # Build (hold_duration, pause_after) segments per style
        segments = []
        if style == 'fluid':
            steps = int(rng.truncated_gauss(4, 1, 3, 6))
            for _ in range(steps):
                hold = rng.truncated_gauss(2.0, 0.7, 1.0, 4.0)
                pause = rng.truncated_gauss(0.15, 0.05, 0.08, 0.25)
                segments.append((hold, pause))

        elif style == 'burst':
            steps = int(rng.truncated_gauss(8, 2, 5, 12))
            for _ in range(steps):
                hold = rng.truncated_gauss(0.5, 0.2, 0.2, 1.0)
                pause = rng.truncated_gauss(0.35, 0.15, 0.1, 0.7)
                segments.append((hold, pause))

        elif style == 'sweep':
            steps = int(rng.truncated_gauss(6, 1.5, 4, 9))
            for _ in range(steps):
                hold = rng.truncated_gauss(1.2, 0.5, 0.4, 2.5)
                pause = rng.truncated_gauss(0.2, 0.1, 0.05, 0.5)
                segments.append((hold, pause))

        elif style == 'tap':
            steps = int(rng.truncated_gauss(12, 3, 8, 18))
            for _ in range(steps):
                hold = rng.truncated_gauss(0.18, 0.06, 0.08, 0.35)
                pause = rng.truncated_gauss(0.5, 0.2, 0.2, 1.0)
                segments.append((hold, pause))

        for step, (hold, pause) in enumerate(segments):
            if self.should_stop:
                return None

            # Break hold into sub-steps, scanning between each
            remaining = hold
            while remaining > 0:
                sub = min(remaining, scan_interval)
                inp.key_hold(key, sub)
                remaining -= sub

                if remaining > 0:
                    result = check_fn()
                    if result:
                        self._log(f"Found mid-rotation step {step + 1} ({style})")
                        return result

            # Post-hold pause + check
            self.ctx.delay.sleep_range(pause * 0.9, pause * 1.1)
            result = check_fn()
            if result:
                self._log(f"Found after {step + 1} rotations ({style})")
                return result

        return None

    def _progressive_search(self, check_fn: Callable):
        """Progressively zoom out and rotate to find a target.

        Zooms out in phases, rotating camera at each zoom level.
        Final phase zooms to max distance as a guaranteed fallback.

        Returns whatever check_fn returns when truthy, or None.
        """
        rng = self.ctx.rng
        inp = self.ctx.input

        self._ensure_mouse_in_game_view()
        key = rng.choice(['left', 'right'])
        total_zoomed = 0

        # Check current view first — maybe just need to rotate
        result = check_fn()
        if result:
            return result

        # Build zoom-out phases with randomization
        # ~65% start with a gentle zoom, then medium, then max-out fallback
        phases = []
        if rng.chance(0.65):
            phases.append(int(rng.truncated_gauss(3, 1, 2, 5)))
        phases.append(int(rng.truncated_gauss(6, 2, 3, 9)))
        phases.append(int(rng.truncated_gauss(15, 4, 10, 22)))

        for i, zoom_ticks in enumerate(phases):
            if self.should_stop:
                return None

            self._ensure_mouse_in_game_view()
            inp.scroll(dy=-zoom_ticks)
            total_zoomed += zoom_ticks
            self._log(f"Search phase {i + 1}/{len(phases)}: "
                       f"zoomed out {zoom_ticks} ticks (total: {total_zoomed})")
            self.ctx.delay.sleep_range(0.2, 0.5)

            # Quick check at new zoom level before rotating
            result = check_fn()
            if result:
                return result

            # Rotate and check
            result = self._rotate_camera_search(key, check_fn)
            if result:
                return result

        return None

    def zoom_out_find(self, color: Color, tolerance: int = 15,
                      min_area: int = 40, max_zoom: int = 25) -> Optional[ColorCluster]:
        """Zoom out incrementally until a color target becomes visible.

        Simpler than search_for_target — no camera rotation, just zoom.
        Best for fixed-position objects like banks/deposit boxes.

        Returns the first ColorCluster found, or None.
        """
        self._ensure_mouse_in_game_view()
        total_zoomed = 0

        while total_zoomed < max_zoom and not self.should_stop:
            ticks = int(self.ctx.rng.truncated_gauss(3, 1, 2, 5))
            self._ensure_mouse_in_game_view()
            self.ctx.input.scroll(dy=-ticks)
            total_zoomed += ticks
            self.ctx.delay.sleep_range(0.3, 0.6)

            target = self.find_target(color, tolerance=tolerance, min_area=min_area)
            if target:
                self._log(f"Found target after zooming out {total_zoomed} ticks")
                return target

        self._log(f"Target not found after zooming out {total_zoomed} ticks")
        return None

    def verify_xp_drop(self, timeout: float = 8.0) -> bool:
        """Wait for an XP drop to confirm an action succeeded.

        Polls check_xp_drop() with randomized intervals up to timeout.
        Returns True if a drop was seen, False on timeout.
        """
        start = time.time()
        while (time.time() - start) < timeout:
            if self.should_stop:
                return True  # don't interfere with shutdown
            if self.check_xp_drop():
                return True
            time.sleep(self.ctx.rng.truncated_gauss(0.4, 0.08, 0.25, 0.55))
        return False

    def search_for_target(self, color: Color, tolerance: int = 15,
                         min_area: int = 40) -> Optional[ColorCluster]:
        """Progressively zoom out and rotate to search for a target color.

        Starts at current zoom, zooms out in phases with camera rotation
        at each level. Final phase zooms to max distance as a fallback.

        Returns the first ColorCluster found, or None after exhaustive search.
        """
        def check():
            return self.find_target(color, tolerance=tolerance, min_area=min_area)

        cluster = self._progressive_search(check)
        if cluster:
            return cluster

        self._log("Search: target not found after full search")
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

    def tap_region_jittered(self, region: "GameRegions") -> None:
        """Tap a random point within a Region — minimal hold to avoid drag.

        Same targeting as click_region_jittered but uses Input.tap() for
        near-instant press+release. Use for bank/deposit interface buttons
        where hold + mouse drift causes OSRS to interpret clicks as drags.
        """
        rng = self.ctx.rng
        cx, cy = region.center
        margin = 1
        jx = int(rng.truncated_gauss(0, region.width * 0.25,
                                      -(region.width // 2 - margin), region.width // 2 - margin))
        jy = int(rng.truncated_gauss(0, region.height * 0.25,
                                      -(region.height // 2 - margin), region.height // 2 - margin))
        sx, sy = self.ctx.vision.to_screen(cx + jx, cy + jy)
        self.ctx.input.tap(sx, sy)

    def deposit_all(self) -> int:
        """Click Deposit All in an open deposit box, verify, close.

        Assumes the deposit box interface is already open.
        Returns the number of items deposited.
        """
        vision = self.ctx.vision
        before = vision.count_inventory_items()

        # Click Deposit All button — tap to avoid drag-scroll in deposit interface
        self.tap_region_jittered(GameRegions.DEPOSIT_ALL_BUTTON)
        self.ctx.delay.sleep(NORMAL_ACTION)

        if self.should_stop:
            return 0

        self.ctx.delay.sleep_range(0.5, 1.0)

        # Verify inventory is empty (accounting for reserved slots like wielded axe)
        expected = self.config.reserved_slots
        remaining = vision.count_inventory_items()
        if remaining > expected and not self.should_stop:
            self._log(f"Still {remaining} items after deposit (expected {expected}), retrying")
            self.tap_region_jittered(GameRegions.DEPOSIT_ALL_BUTTON)
            self.ctx.delay.sleep(NORMAL_ACTION)
            remaining = vision.count_inventory_items()

        # Close deposit box with Escape
        self.ctx.delay.sleep_range(0.3, 0.6)
        self.ctx.input.key_tap('esc')
        self.ctx.delay.sleep(NORMAL_ACTION)

        return max(0, before - remaining)

    def wait_for_bank_open(self, template_path: str, region: Region,
                           max_wait: float, threshold: float = 0.8) -> bool:
        """Poll for bank/deposit box interface to open via template matching.

        Returns True if detected before max_wait, False if timed out.
        Adds a small human-like reaction delay after detection.
        """
        import os
        if not os.path.exists(template_path):
            # No template — fall back to blind sleep
            self.ctx.delay.sleep_range(max_wait * 0.9, max_wait * 1.1)
            return True

        start = time.time()
        while (time.time() - start) < max_wait:
            if self.should_stop:
                return True
            if self.ctx.vision.template_match_region(
                region, template_path, threshold=threshold,
            ):
                # Human reaction delay — don't act instantly when interface opens
                self.ctx.delay.sleep_range(0.4, 1.2)
                return True
            time.sleep(self.ctx.rng.truncated_gauss(0.5, 0.1, 0.3, 0.7))

        return False

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

        Randomly picks a drop style each invocation:
        - per_item: shift-click each slot individually (press/release shift each time)
        - hold_shift: hold shift the whole time, just click each slot
        - chunked: hold shift for a group, release, re-hold for next group

        Also randomizes traversal pattern, drop speed, and occasionally makes
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

        # Build set of slots where misclicks are suppressed.
        # Grid is 4 cols x 7 rows; slot pitch is 42x36px, gap only 6x4px.
        #
        # Danger zones:
        # - Adjacent to protected (skip) slots — could drop rod/bait
        # - Top row (0-3) — misclick up enters game view
        # - Bottom row (24-27) — misclick down hits menu tabs
        _no_misclick: Set[int] = set()

        # Slots adjacent to protected items
        if skip_slots:
            for s in skip_slots:
                if s % 4 != 0:
                    _no_misclick.add(s - 1)  # left
                if s % 4 != 3:
                    _no_misclick.add(s + 1)  # right
                if s >= 4:
                    _no_misclick.add(s - 4)  # above
                if s < 24:
                    _no_misclick.add(s + 4)  # below
            _no_misclick -= skip_slots

        # Edge rows — misclicks could leave inventory area
        _no_misclick |= set(range(4))       # top row
        _no_misclick |= set(range(24, 28))  # bottom row

        # Choose drop style
        style = rng.weighted_choice(
            ['per_item', 'hold_shift', 'chunked'],
            [30, 45, 25],
        )

        if style == 'hold_shift':
            return self._drop_hold_shift(
                order, skip_slots, expected, speed_mult, _no_misclick,
            )
        elif style == 'chunked':
            return self._drop_chunked(
                order, skip_slots, expected, speed_mult, _no_misclick,
            )
        else:
            return self._drop_per_item(
                order, skip_slots, expected, speed_mult, _no_misclick,
            )

    def _drop_delay(self, speed_mult: float) -> None:
        """Variable delay between drops, shared across styles.

        Tuned for spam-dropping: short delays with occasional brief pauses.
        """
        rng = self.ctx.rng
        # ~6% chance: brief pause mid-drop (distracted/thinking)
        if rng.chance(0.06):
            pause = rng.truncated_gauss(0.5, 0.2, 0.25, 1.0)
            self.ctx.delay.sleep_range(pause * 0.8, pause * 1.2)
        else:
            # Quick inter-drop delay — spammy feel
            item_speed = speed_mult * rng.truncated_gauss(1.0, 0.1, 0.8, 1.3)
            base_min = 0.03 * item_speed
            base_max = 0.12 * item_speed
            self.ctx.delay.sleep_range(base_min, base_max, include_pauses=False)

    def _drop_per_item(self, order: List[int], skip_slots: Optional[Set[int]],
                       expected: Optional[int], speed_mult: float,
                       no_misclick: Set[int]) -> int:
        """Original style: shift-click each item individually."""
        rng = self.ctx.rng
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

            # ~8% chance: misclick (off by some pixels), then correct
            if slot_idx not in no_misclick and rng.chance(0.08):
                ox = int(rng.truncated_gauss(0, 6, -12, 12))
                oy = int(rng.truncated_gauss(0, 6, -12, 12))
                self.ctx.input.shift_click(cx + ox, cy + oy)
                self._drop_delay(speed_mult)
                self.ctx.input.shift_click(cx, cy)

            # ~3% chance: left-click instead of shift-click, then correct
            elif rng.chance(0.03):
                self.ctx.input.click(cx, cy)
                pause = rng.truncated_gauss(0.3, 0.1, 0.15, 0.5)
                self.ctx.delay.sleep_range(pause * 0.8, pause * 1.2)
                self.ctx.input.shift_click(cx, cy)

            else:
                self.ctx.input.shift_click(cx, cy)

            self._drop_delay(speed_mult)
            dropped += 1
        return dropped

    def _drop_hold_shift(self, order: List[int], skip_slots: Optional[Set[int]],
                         expected: Optional[int], speed_mult: float,
                         no_misclick: Set[int]) -> int:
        """Hold shift for the entire drop sequence, just click each slot."""
        rng = self.ctx.rng
        inp = self.ctx.input

        # Small pre-delay before pressing shift
        time.sleep(rng.truncated_gauss(0.02, 0.01, 0.01, 0.05))
        inp.shift_down()

        dropped = 0
        try:
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

                # ~8% chance: misclick, then correct (shift still held)
                if slot_idx not in no_misclick and rng.chance(0.08):
                    ox = int(rng.truncated_gauss(0, 6, -12, 12))
                    oy = int(rng.truncated_gauss(0, 6, -12, 12))
                    inp.click(cx + ox, cy + oy)
                    self._drop_delay(speed_mult)
                    inp.click(cx, cy)
                else:
                    inp.click(cx, cy)

                self._drop_delay(speed_mult)
                dropped += 1
        finally:
            time.sleep(rng.truncated_gauss(0.02, 0.01, 0.01, 0.05))
            inp.shift_up()

        return dropped

    def _drop_chunked(self, order: List[int], skip_slots: Optional[Set[int]],
                      expected: Optional[int], speed_mult: float,
                      no_misclick: Set[int]) -> int:
        """Hold shift for chunks of items, release between chunks."""
        rng = self.ctx.rng
        inp = self.ctx.input

        # Build list of slot indices to drop (in traversal order)
        to_drop: List[int] = []
        for slot_idx in order:
            if skip_slots and slot_idx in skip_slots:
                continue
            to_drop.append(slot_idx)

        if not to_drop:
            return 0

        # Split into chunks of 3-8 items
        chunks: List[List[int]] = []
        i = 0
        while i < len(to_drop):
            chunk_size = int(rng.truncated_gauss(5, 1.5, 3, 8))
            chunks.append(to_drop[i:i + chunk_size])
            i += chunk_size

        dropped = 0
        for chunk_idx, chunk in enumerate(chunks):
            if self.should_stop:
                break
            if expected is not None and dropped >= expected:
                break

            # Press shift for this chunk
            time.sleep(rng.truncated_gauss(0.02, 0.01, 0.01, 0.05))
            inp.shift_down()

            try:
                for slot_idx in chunk:
                    if self.should_stop:
                        break
                    if expected is not None and dropped >= expected:
                        break
                    if not self.ctx.vision.slot_has_item(slot_idx):
                        continue

                    cx, cy = self.ctx.vision.slot_screen_click_point(slot_idx)

                    # ~8% chance: misclick, then correct
                    if slot_idx not in no_misclick and rng.chance(0.08):
                        ox = int(rng.truncated_gauss(0, 6, -12, 12))
                        oy = int(rng.truncated_gauss(0, 6, -12, 12))
                        inp.click(cx + ox, cy + oy)
                        self._drop_delay(speed_mult)
                        inp.click(cx, cy)
                    else:
                        inp.click(cx, cy)

                    self._drop_delay(speed_mult)
                    dropped += 1
            finally:
                time.sleep(rng.truncated_gauss(0.02, 0.01, 0.01, 0.05))
                inp.shift_up()

            # Pause between chunks — brief "re-grip" / distraction
            if chunk_idx < len(chunks) - 1 and not self.should_stop:
                gap = rng.truncated_gauss(0.35, 0.15, 0.15, 0.7)
                self.ctx.delay.sleep_range(gap * 0.8, gap * 1.2)

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
