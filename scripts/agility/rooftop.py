"""
Rooftop Agility Script

Runs rooftop agility courses by clicking color-coded obstacles in sequence.
Each obstacle is marked with a unique color via Object Markers plugin.

Course sequence:
    1. Cyan → 2. Green → 3. Blue → 4. Yellow → 5. Purple → 6. Orange → (run back) → repeat

State machine:
    FIND_OBSTACLE → CLICK_OBSTACLE → TRAVERSING → FIND_OBSTACLE (next)

If the bot fails to find the next obstacle after several tries, it assumes
the player fell and resets to look for the start obstacle (cyan).
"""

import time
import threading
from enum import Enum, auto
from typing import Optional, Callable, List, Tuple

from indigo.script import Script, ScriptConfig, ScriptContext
from indigo.vision import Color, ColorCluster, GameRegions
from indigo.core.timing import NORMAL_ACTION


class State(Enum):
    FIND_OBSTACLE = auto()
    CLICK_OBSTACLE = auto()
    TRAVERSING = auto()


XP_MAX_WAIT = 18.0                              # Seconds before declaring misclick
XP_MAX_RETRIES = 2                              # Retries before falling back to camera search

# Ground item highlight (Mark of Grace, etc.) — detected via text label color.
# Text "Mark of grace" renders as multiple small clusters (area ~10-11 each).
GROUND_ITEM_COLOR = Color.from_hex("FF00A4FF")
GROUND_ITEM_MIN_CLUSTERS = 2      # Need 2+ clusters to confirm it's text, not noise
GROUND_ITEM_MAX_DISTANCE = 100.0  # Only pick up when close (px from game view center)
GROUND_ITEM_COOLDOWN = 30.0       # Seconds to wait before trying another pickup

# Wall entrance color — unique so we can detect "on the ground" vs "on the roof"
WALL_COLOR = Color.from_hex("FFC0FF00")  # Yellow-green — only visible from ground level

# Course obstacles in order — each marked with a unique color via Object Markers.
# Colors are AARRGGBB hex from RuneLite.
COURSE_OBSTACLES: List[Tuple[str, Color]] = [
    ("Wall",       WALL_COLOR),                   # Yellow-green - climb wall to roof
    ("Obstacle 1", Color.from_hex("FF00FFFF")),  # Cyan  - first rooftop obstacle
    ("Obstacle 2", Color.from_hex("FF00FF00")),  # Green
    ("Obstacle 3", Color.from_hex("FF0000FF")),  # Blue
    ("Obstacle 4", Color.from_hex("FFFFFF00")),  # Yellow
    ("Obstacle 5", Color.from_hex("FF095E13")),  # Dark green
    ("Obstacle 6", Color.from_hex("FFFFAD00")),  # Orange
]


class RooftopScript(Script):
    """Run rooftop agility courses."""

    def __init__(self, ctx: ScriptContext, max_hours: float = 6.0,
                 on_log: Optional[Callable[[str], None]] = None):
        super().__init__(
            config=ScriptConfig(name="Rooftop", max_runtime_hours=max_hours),
            ctx=ctx,
            on_log=on_log,
        )
        self._state = State.FIND_OBSTACLE
        self._obstacle_idx = 0
        self._last_target: Optional[ColorCluster] = None
        self._find_failures = 0
        self._traverse_start = 0.0
        self._post_click_done = False
        self._idle_phase_done = False
        self._wake_up_done = False
        self._laps = 0
        self._obstacles_completed = 0
        self._items_picked_up = 0
        self._last_pickup_attempt = 0.0
        self._xp_retries = 0
        self._xp_poll_count = 0
        self._xp_detected = threading.Event()
        self._xp_watcher_stop = threading.Event()
        self._xp_detected_time = 0.0

    def on_start(self) -> None:
        self._log("Starting rooftop agility - ensure Object Markers configured")
        self._log(f"Course: {' -> '.join(n for n, _ in COURSE_OBSTACLES)} -> (run back)")

        # Zoom in for larger click targets
        self._ensure_mouse_in_game_view()
        ticks = int(self.ctx.rng.truncated_gauss(8.0, 1.5, 6, 10))
        self.ctx.input.scroll(dy=ticks)
        self.ctx.delay.sleep_range(0.4, 0.8)
        self._log(f"Zoomed in {ticks} ticks")

    def _current_color(self) -> Color:
        return COURSE_OBSTACLES[self._obstacle_idx][1]

    def _current_name(self) -> str:
        return COURSE_OBSTACLES[self._obstacle_idx][0]

    def _is_last_obstacle(self) -> bool:
        return self._obstacle_idx == len(COURSE_OBSTACLES) - 1

    def _advance_obstacle(self) -> None:
        """Move to the next obstacle in the course."""
        self._obstacles_completed += 1
        if self._is_last_obstacle():
            self._obstacle_idx = 0
            self._laps += 1
            self._log(
                f"Lap {self._laps} complete! "
                f"({self._obstacles_completed} obstacles, {self.elapsed_str()})"
            )
        else:
            self._obstacle_idx += 1
        self._find_failures = 0

    def loop(self) -> None:
        if self._state == State.FIND_OBSTACLE:
            self._do_find()
        elif self._state == State.CLICK_OBSTACLE:
            self._do_click()
        elif self._state == State.TRAVERSING:
            self._do_traverse()

    def _do_find(self) -> None:
        rng = self.ctx.rng

        # Check for nearby ground items BEFORE zooming (wider view)
        # Cooldown prevents getting stuck retrying a far-away mark
        if time.time() - self._last_pickup_attempt > GROUND_ITEM_COOLDOWN:
            ground_click = self._find_nearby_ground_item()
            if ground_click:
                self._last_pickup_attempt = time.time()
                self._pickup_ground_item(ground_click)
                return

        # Always scroll in a few ticks to maintain zoomed-in baseline.
        # OSRS has a max zoom cap so we can't over-zoom. This counteracts
        # any drift from idle zoom_adjust, zoom_fidget, or search zoom-outs.
        self._ensure_mouse_in_game_view()
        ticks = int(rng.truncated_gauss(3.0, 1.0, 2, 5))
        self.ctx.input.scroll(dy=ticks)
        self.ctx.delay.sleep_range(0.15, 0.3)

        color = self._current_color()
        target = self.find_target(color)

        if target:
            self._last_target = target
            self._find_failures = 0
            self._state = State.CLICK_OBSTACLE
            self._log(f"Found {self._current_name()} at {target.click_point} (area={target.area})")
        else:
            self._find_failures += 1

            # After 6 failures: full search — zoom out, spin, look for anything
            if self._find_failures >= 6:
                self._log(f"Can't find {self._current_name()} after {self._find_failures} tries, searching...")
                found = self._search_for_obstacle()
                if found:
                    return
                self._find_failures = 0

            self.ctx.delay.sleep_range(1.0, 2.0)
            if self.ctx.idle:
                self.ctx.idle.maybe_idle()

    def _safe_click_point(self, cluster: ColorCluster) -> Tuple[int, int]:
        """Get a click point biased toward the center of the cluster bounding box.

        Uses Gaussian jitter constrained to the inner 40% of the bounding box
        (30% margin on each side), so clicks cluster near center and never
        land anywhere close to the edge.
        """
        bx, by, bw, bh = cluster.bounding_box
        cx = bx + bw // 2
        cy = by + bh // 2

        # 30% margin on each side — clicks stay in the inner 40%
        margin_x = max(3, int(bw * 0.30))
        margin_y = max(3, int(bh * 0.30))
        half_w = bw // 2 - margin_x
        half_h = bh // 2 - margin_y

        if half_w > 0:
            jx = int(self.ctx.rng.truncated_gauss(0, bw * 0.08, -half_w, half_w))
        else:
            jx = 0
        if half_h > 0:
            jy = int(self.ctx.rng.truncated_gauss(0, bh * 0.08, -half_h, half_h))
        else:
            jy = 0

        return (cx + jx, cy + jy)

    def _find_nearby_ground_item(self) -> Optional[Tuple[int, int]]:
        """Detect a nearby ground item via its text label color.

        The RuneLite Ground Items plugin renders "Mark of grace" as colored text,
        which shows up as multiple small clusters (individual letter groups).
        We detect 2+ clusters of the highlight color, average their positions,
        and only return a click point if the item is close to the player
        (near the center of the game view).

        Returns (x, y) screen coords to click (below the text), or None.
        """
        clusters = self.ctx.vision.find_color_clusters(
            GameRegions.GAME_VIEW, GROUND_ITEM_COLOR,
            tolerance=15, min_area=8,
        )

        if len(clusters) < GROUND_ITEM_MIN_CLUSTERS:
            return None

        # Average all cluster centers
        avg_x = sum(c.center[0] for c in clusters) / len(clusters)
        avg_y = sum(c.center[1] for c in clusters) / len(clusters)

        # Distance from game view center (screen coords)
        gv = GameRegions.GAME_VIEW
        gv_center_x, gv_center_y = self.ctx.vision.to_screen(
            gv.x + gv.width // 2, gv.y + gv.height // 2,
        )
        dist = ((avg_x - gv_center_x) ** 2 + (avg_y - gv_center_y) ** 2) ** 0.5

        if dist > GROUND_ITEM_MAX_DISTANCE:
            return None

        # Click below the text to hit the actual tile (~20-25px down)
        rng = self.ctx.rng
        offset_y = int(rng.truncated_gauss(10.0, 2.0, 6.0, 14.0))
        click_x = int(avg_x + rng.truncated_gauss(0, 3, -6, 6))
        click_y = int(avg_y + offset_y)

        self._log(
            f"Mark of Grace nearby ({len(clusters)} clusters, "
            f"dist={dist:.0f}px, click=({click_x},{click_y}))"
        )
        return (click_x, click_y)

    def _pickup_ground_item(self, click_point: Tuple[int, int]) -> None:
        """Click a ground item and wait for pickup."""
        x, y = click_point
        self.ctx.input.click(x, y)
        self.ctx.delay.sleep(NORMAL_ACTION)

        # Wait for character to walk over and pick it up
        pickup_wait = self.ctx.rng.truncated_gauss(3.0, 0.8, 2.0, 5.0)
        self.ctx.delay.sleep_range(pickup_wait * 0.9, pickup_wait * 1.1)

        self._items_picked_up += 1
        self._log(f"Picked up item (total: {self._items_picked_up})")

    def _search_for_obstacle(self) -> bool:
        """Progressively zoom out and spin camera to find the obstacle or wall.

        Uses _progressive_search to zoom out in phases (small → medium → max)
        with varied rotation at each level. Checks for:
          1. The current obstacle color → found it, continue course
          2. The wall entrance color → we fell, restart from wall

        Returns True if we found something and updated state, False otherwise.
        """
        color = self._current_color()

        def check():
            target = self.find_target(color)
            if target:
                return ('target', target)
            if self._obstacle_idx != 0:
                wall = self.find_target(WALL_COLOR)
                if wall:
                    return ('wall', wall)
            return None

        result = self._progressive_search(check)
        if result:
            kind, cluster = result
            if kind == 'target':
                self._log(f"Search: found {self._current_name()}")
                self._last_target = cluster
                self._find_failures = 0
                self._state = State.CLICK_OBSTACLE
            else:
                self._log(f"Search: spotted wall entrance — fell off, restarting course")
                self._last_target = cluster
                self._obstacle_idx = 0
                self._find_failures = 0
                self._state = State.CLICK_OBSTACLE
            self._rezoom_in()
            return True

        self._log("Search: nothing found after full search")
        self._rezoom_in()
        return False

    def _rezoom_in(self) -> None:
        """Zoom back in to a comfortable level for clicking."""
        self._ensure_mouse_in_game_view()
        rng = self.ctx.rng
        ticks = int(rng.truncated_gauss(8.0, 1.5, 6, 10))
        self.ctx.input.scroll(dy=ticks)
        self.ctx.delay.sleep_range(0.3, 0.6)

    def _start_xp_watcher(self) -> None:
        """Start background thread that continuously watches for XP drops."""
        self._xp_detected.clear()
        self._xp_watcher_stop.clear()
        self._xp_detected_time = 0.0
        self._xp_poll_count = 0

        def watch():
            while not self._xp_watcher_stop.is_set():
                if self.check_xp_drop():
                    self._xp_detected_time = time.time()
                    self._xp_poll_count += 1
                    self._xp_detected.set()
                    return
                self._xp_poll_count += 1
                self._xp_watcher_stop.wait(0.3)

        t = threading.Thread(target=watch, daemon=True)
        t.start()

    def _stop_xp_watcher(self) -> None:
        """Stop the background XP watcher."""
        self._xp_watcher_stop.set()

    def _do_click(self) -> None:
        if not self._last_target:
            return

        x, y = self._safe_click_point(self._last_target)
        self.ctx.input.click(x, y)
        self.ctx.delay.sleep(NORMAL_ACTION)

        self._traverse_start = time.time()
        self._post_click_done = False
        self._idle_phase_done = False
        self._wake_up_done = False
        self._start_xp_watcher()
        self._state = State.TRAVERSING

        self._log(
            f"Clicked {self._current_name()}, watching for XP "
            f"(attempt {self._xp_retries + 1}/{XP_MAX_RETRIES + 1}) "
            f"[lap {self._laps + 1}, {self._obstacles_completed + 1} total]"
        )

    def _zoom_fidget(self) -> None:
        """Human-like zoom exploration: zoom out to look around, then back in.

        ~15% chance per AFK phase tick. Zooms out a variable amount, pauses
        like the player is looking around, then zooms back in to baseline.
        """
        rng = self.ctx.rng
        inp = self.ctx.input

        self._ensure_mouse_in_game_view()
        out_ticks = int(rng.truncated_gauss(5.0, 2.0, 3, 10))
        inp.scroll(dy=-out_ticks)
        self.ctx.delay.sleep_range(1.0, 3.0)

        if self.should_stop:
            # Zoom back in before bailing
            self._ensure_mouse_in_game_view()
            inp.scroll(dy=out_ticks)
            return

        # Maybe spin camera while zoomed out
        if self.ctx.idle and rng.chance(0.40):
            self.ctx.idle._camera_nudge()
            self.ctx.delay.sleep_range(0.5, 1.5)

        # Zoom back in (re-check mouse position — idle action may have moved it)
        self._ensure_mouse_in_game_view()
        inp.scroll(dy=out_ticks)
        self.ctx.delay.sleep_range(0.2, 0.5)
        self._log(f"Zoom fidget (out {out_ticks}, back in)")

    def _handle_xp_detected(self) -> None:
        """React to XP drop with human-like delay, then advance."""
        self._stop_xp_watcher()
        rng = self.ctx.rng
        elapsed = self._xp_detected_time - self._traverse_start

        # Human reaction delay
        reaction = rng.truncated_gauss(0.6, 0.25, 0.25, 1.5)
        self.ctx.delay.sleep_range(reaction * 0.9, reaction * 1.1)

        self._log(
            f"XP detected after {elapsed:.1f}s ({self._xp_poll_count} polls), "
            f"advancing from {self._current_name()}"
        )

        # Last obstacle requires run-back wait before looking for wall
        if self._is_last_obstacle():
            run_back = rng.truncated_gauss(5.0, 1.5, 3.0, 8.0)
            self._log(f"Run-back wait ~{run_back:.0f}s")
            if self.ctx.idle and rng.chance(0.50):
                self.ctx.idle._do_burst()
            self.ctx.delay.sleep_range(run_back * 0.9, run_back * 1.1)

        self._xp_retries = 0
        self._advance_obstacle()
        self._state = State.FIND_OBSTACLE

    def _handle_xp_timeout(self) -> None:
        """No XP detected within timeout — assume misclick, retry or search."""
        self._stop_xp_watcher()
        self._xp_retries += 1

        if self._xp_retries > XP_MAX_RETRIES:
            self._log(
                f"No XP after {XP_MAX_RETRIES + 1} attempts on {self._current_name()}, "
                f"searching..."
            )
            self._xp_retries = 0
            found = self._search_for_obstacle()
            if not found:
                self._find_failures = 0
                self._state = State.FIND_OBSTACLE
        else:
            self._log(
                f"No XP within {XP_MAX_WAIT:.0f}s on {self._current_name()} "
                f"(attempt {self._xp_retries}/{XP_MAX_RETRIES + 1}), retrying"
            )
            self._state = State.FIND_OBSTACLE

    def _do_traverse_idle(self, elapsed: float) -> None:
        """Human-like idle behavior while background thread watches for XP.

        Full idle behaviors are safe here — the watcher thread handles
        XP detection independently, so blocking actions can't miss drops.
        """
        rng = self.ctx.rng
        idle = self.ctx.idle

        # Phase 1: Post-click fidget (~1-3s after clicking)
        if not self._post_click_done and elapsed > rng.truncated_gauss(1.5, 0.5, 0.8, 2.5):
            self._post_click_done = True
            if idle and not self.should_stop and rng.chance(0.75):
                idle._do_burst()
            return

        # Phase 3: Wake up (~2-3s before timeout)
        remaining = XP_MAX_WAIT - elapsed
        wake_up_window = rng.truncated_gauss(2.5, 0.5, 1.5, 3.5)
        if not self._wake_up_done and remaining < wake_up_window:
            self._wake_up_done = True
            if idle and not self.should_stop:
                if rng.chance(0.50):
                    idle._camera_spin()
                elif rng.chance(0.50):
                    idle._camera_nudge()
                else:
                    idle._fidget_mouse()
            return

        # Phase 2: AFK zone (middle of the wait)
        if rng.chance(0.15) and remaining > wake_up_window + 3.0:
            self._zoom_fidget()
        else:
            chunk = rng.truncated_gauss(2.5, 0.5, 1.5, 3.5)
            if chunk > 0:
                self.ctx.delay.sleep_range(chunk * 0.9, chunk * 1.1)

    def _do_traverse(self) -> None:
        """Wait for XP drop confirmation from background watcher.

        The watcher thread polls for XP drops every 0.3s independently.
        Main thread just checks the flag and does idle behaviors.
        """
        elapsed = time.time() - self._traverse_start

        # Hard timeout — no XP detected
        if elapsed >= XP_MAX_WAIT:
            self._handle_xp_timeout()
            return

        # Check if watcher found XP
        if self._xp_detected.is_set():
            self._handle_xp_detected()
            return

        # Human-like behavior while waiting
        self._do_traverse_idle(elapsed)
