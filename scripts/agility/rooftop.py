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
from enum import Enum, auto
from typing import Optional, Callable, List, Tuple

from indigo.script import Script, ScriptConfig, ScriptContext
from indigo.vision import Color, ColorCluster
from indigo.core.timing import NORMAL_ACTION


class State(Enum):
    FIND_OBSTACLE = auto()
    CLICK_OBSTACLE = auto()
    TRAVERSING = auto()


# Ground item highlight (Mark of Grace, etc.) — light blue square outline
GROUND_ITEM_COLOR = Color.from_hex("FF00A4FF")

# Course obstacles in order — each marked with a unique color via Object Markers.
# Colors are AARRGGBB hex from RuneLite.
# The wall climb and first rooftop obstacle are both cyan (two consecutive clicks).
COURSE_OBSTACLES: List[Tuple[str, Color]] = [
    ("Wall",       Color.from_hex("FF00FFFF")),  # Cyan  - climb wall to roof
    ("Obstacle 1", Color.from_hex("FF00FFFF")),  # Cyan  - first rooftop obstacle
    ("Obstacle 2", Color.from_hex("FF00FF00")),  # Green
    ("Obstacle 3", Color.from_hex("FF0000FF")),  # Blue
    ("Obstacle 4", Color.from_hex("FFFFFF00")),  # Yellow
    ("Obstacle 5", Color.from_hex("FFAD00FF")),  # Purple
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
        self._target_wait = 0.0
        self._post_click_done = False
        self._wake_up_done = False
        self._zoom_ticks = 0
        self._laps = 0
        self._obstacles_completed = 0
        self._items_picked_up = 0

    def on_start(self) -> None:
        self._log("Starting rooftop agility - ensure Object Markers configured")
        self._log(f"Course: {' -> '.join(n for n, _ in COURSE_OBSTACLES)} -> (run back)")

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

        # Zoom in for a larger click target (once per find cycle)
        if self._zoom_ticks == 0:
            ticks = int(rng.truncated_gauss(4.0, 1.0, 3, 6))
            self.ctx.input.scroll(dy=ticks)
            self._zoom_ticks = ticks
            self.ctx.delay.sleep_range(0.4, 0.8)

        # Check for highlighted ground items (Mark of Grace, etc.) before obstacles
        ground_item = self.find_target(GROUND_ITEM_COLOR, min_area=30)
        if ground_item:
            self._pickup_ground_item(ground_item)
            return

        color = self._current_color()
        target = self.find_target(color)

        if target:
            self._last_target = target
            self._find_failures = 0
            self._state = State.CLICK_OBSTACLE
            self._log(f"Found {self._current_name()} at {target.click_point} (area={target.area})")
        else:
            self._find_failures += 1

            if self._find_failures >= 5:
                self._log(f"Can't find {self._current_name()} after {self._find_failures} tries, searching...")
                # search_for_target zooms out on its own — reset our zoom tracking
                self._zoom_ticks = 0
                target = self.search_for_target(color)
                if target:
                    self._last_target = target
                    self._find_failures = 0
                    self._state = State.CLICK_OBSTACLE
                    return

                # Might have fallen off — reset to looking for start (cyan)
                if self._obstacle_idx != 0:
                    self._log("May have fallen, resetting to look for start")
                    self._obstacle_idx = 0
                    self._find_failures = 0
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

    def _pickup_ground_item(self, cluster: ColorCluster) -> None:
        """Click a highlighted ground item and wait for pickup."""
        x, y = self._safe_click_point(cluster)
        self._log(f"Ground item found at ({x}, {y}), picking up (area={cluster.area})")

        self.ctx.input.click(x, y)
        self.ctx.delay.sleep(NORMAL_ACTION)

        # Wait for character to walk over and pick it up
        pickup_wait = self.ctx.rng.truncated_gauss(3.0, 0.8, 2.0, 5.0)
        self.ctx.delay.sleep_range(pickup_wait * 0.9, pickup_wait * 1.1)

        self._items_picked_up += 1
        self._log(f"Picked up item (total: {self._items_picked_up})")

    def _do_click(self) -> None:
        if not self._last_target:
            return

        x, y = self._safe_click_point(self._last_target)
        self.ctx.input.click(x, y)
        self.ctx.delay.sleep(NORMAL_ACTION)

        # Set wait time — longer for run back after last obstacle
        is_run_back = self._is_last_obstacle()
        if is_run_back:
            self._target_wait = self.ctx.rng.truncated_gauss(14.0, 2.0, 11.0, 18.0)
        else:
            self._target_wait = self.ctx.rng.truncated_gauss(12.0, 2.0, 9.0, 16.0)

        self._traverse_start = time.time()
        self._post_click_done = False
        self._wake_up_done = False
        self._state = State.TRAVERSING

        label = " (run back)" if is_run_back else ""
        self._log(
            f"Clicked {self._current_name()}{label}, "
            f"waiting ~{self._target_wait:.0f}s "
            f"[lap {self._laps + 1}, {self._obstacles_completed + 1} total]"
        )

    def _do_traverse(self) -> None:
        """Wait for the character to traverse the obstacle.

        Three phases that mimic a real player's attention pattern:
          1. Post-click fidget (~1-3s in): attention burst right after clicking
          2. AFK zone (middle): zone out, just wait
          3. Wake up (~2-3s before end): camera nudge/spin, get ready for next click
        """
        elapsed = time.time() - self._traverse_start
        remaining = self._target_wait - elapsed

        if remaining <= 0:
            self._advance_obstacle()
            self._state = State.FIND_OBSTACLE
            return

        rng = self.ctx.rng
        idle = self.ctx.idle

        # Phase 1: Post-click fidget (~1-3s after clicking)
        # Player just clicked, zooms back out and fidgets while character moves
        if not self._post_click_done and elapsed > rng.truncated_gauss(1.5, 0.5, 0.8, 2.5):
            self._post_click_done = True
            # Zoom back out to baseline so we have a wide view for next find
            if self._zoom_ticks > 0:
                self.ctx.input.scroll(dy=-self._zoom_ticks)
                self._zoom_ticks = 0
                self.ctx.delay.sleep_range(0.2, 0.4)
            if idle and not self.should_stop and rng.chance(0.75):
                idle._do_burst()
            return

        # Phase 3: Wake up (~2-3s before the wait ends)
        # Player snaps back to attention, adjusts camera to spot next obstacle
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
            self.ctx.delay.sleep_range(remaining * 0.8, remaining * 1.0)
            return

        # Phase 2: AFK zone (middle of the wait) — just sleep
        chunk = min(remaining - wake_up_window, rng.truncated_gauss(2.5, 0.5, 1.5, 3.5))
        if chunk > 0:
            self.ctx.delay.sleep_range(chunk * 0.9, chunk * 1.1)
        return
