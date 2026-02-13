"""
Barbarian Fishing Script

Barbarian fishes at Otto's Grotto (or any barb fishing spot), drops all
fish when inventory is full, and repeats.  Slot 0 = rod, slot 1 = feathers
— both are protected during drops.

State machine:
    FIND_SPOT -> CLICK_SPOT -> FISHING -> DROPPING -> FIND_SPOT

Barbarian fishing spots move frequently (~15-20s), so the idle timeout
is short.  Uses dual-track detection: XP drops + inventory gain to
decide when the spot has moved.

Requires:
- Object Markers: green (#43FF00 / FF43FF00) on fishing spot
- RuneLite XP Drop plugin set to magenta (FF00FF)
- Fishing rod in slot 0, feathers/bait in slot 1
"""

import time
from enum import Enum, auto
from typing import Optional, Callable

from indigo.script import Script, ScriptConfig, ScriptContext
from indigo.vision import Color, ColorCluster
from indigo.core.timing import NORMAL_ACTION


class State(Enum):
    FIND_SPOT = auto()
    CLICK_SPOT = auto()
    FISHING = auto()
    DROPPING = auto()


# Green from Object Markers on fishing spot: #43FF00
SPOT_COLOR = Color(r=67, g=255, b=0)

# XP drop verification timeout after clicking spot
FISH_XP_TIMEOUT = 20.0


class BarbarianScript(Script):
    """Barbarian fish, drop everything, repeat."""

    def __init__(self, ctx: ScriptContext, max_hours: float = 6.0,
                 on_log: Optional[Callable[[str], None]] = None):
        super().__init__(
            config=ScriptConfig(
                name="Barbarian", max_runtime_hours=max_hours,
                reserved_slots=2,  # rod + feathers
            ),
            ctx=ctx,
            on_log=on_log,
        )
        self._state = State.FIND_SPOT
        self._last_target: Optional[ColorCluster] = None
        self._find_failures = 0
        self._wait_checks = 0

        # Fishing tracking (dual-track: XP drops + inventory gain)
        self._last_activity_time = 0.0
        self._last_inv_count = 0
        self._activity_timeout = 15.0  # session-varied in on_start

        # Camera recovery: on misclick, pitch overhead and retry once
        self._overhead_mode = False

        # AFK break cap for barb fishing (~4 min max)
        self._afk_max_duration = 240.0

        # Stats
        self._fish_caught = 0
        self._drop_cycles = 0

    def on_start(self) -> None:
        rng = self.ctx.rng

        # Session-vary the activity timeout (~12-18s)
        self._activity_timeout = rng.truncated_gauss(15.0, 2.0, 12.0, 18.0)

        # Vary AFK cap slightly per session (~3-4.5 min)
        self._afk_max_duration = rng.truncated_gauss(240.0, 30.0, 180.0, 270.0)

        self._log(
            f"Barbarian fishing (activity timeout={self._activity_timeout:.0f}s, "
            f"afk cap={self._afk_max_duration:.0f}s, "
            f"inv full at {self.inv_full_count}) — "
            f"ensure green on spot, XP drops magenta"
        )

    def loop(self) -> None:
        if self._state == State.FIND_SPOT:
            self._do_find_spot()
        elif self._state == State.CLICK_SPOT:
            self._do_click_spot()
        elif self._state == State.FISHING:
            self._do_fishing()
        elif self._state == State.DROPPING:
            self._do_dropping()

    # ── FIND_SPOT ────────────────────────────────────────────

    def _do_find_spot(self) -> None:
        spot = self.find_target(SPOT_COLOR)
        if spot:
            self._last_target = spot
            self._find_failures = 0
            self._state = State.CLICK_SPOT
            self._log(f"Found fishing spot at {spot.click_point} (area={spot.area})")
        else:
            self._find_failures += 1
            if self._find_failures >= 3:
                self._log("No spot after 3 tries, searching...")
                spot = self.search_for_target(SPOT_COLOR)
                if spot:
                    self._last_target = spot
                    self._find_failures = 0
                    self._state = State.CLICK_SPOT
                    return
                self._find_failures = 0
            self._log("No fishing spot found, waiting...")
            self.ctx.delay.sleep_range(1.5, 3.0)
            if self.ctx.idle:
                self.ctx.idle.maybe_idle()

    # ── CLICK_SPOT ───────────────────────────────────────────

    def _do_click_spot(self) -> None:
        if not self._last_target:
            self._state = State.FIND_SPOT
            return

        x, y = self._last_target.click_point

        # Verify target still exists (spot may have moved)
        if not self.is_target_near(SPOT_COLOR, (x, y)):
            self._log("Spot gone before click (stale target), re-finding")
            self._state = State.FIND_SPOT
            return

        self.click_target(x, y)
        self.ctx.delay.sleep(NORMAL_ACTION)

        self._wait_checks = 0

        # Verify fishing started via XP drop
        inv_count = self.ctx.vision.count_inventory_items(skip_slots=[0, 1])
        self._log(f"Clicked spot (inv={inv_count}), verifying fish...")
        if not self.verify_xp_drop(timeout=FISH_XP_TIMEOUT):
            if not self._overhead_mode:
                # First failure — pitch camera overhead and retry
                self._log("No XP drop — pitching camera overhead to clear obstructions")
                self._pitch_camera_overhead()
                self._overhead_mode = True
                self._state = State.FIND_SPOT
                return
            else:
                # Already overhead and still failing — give up, go idle
                self._log("No XP drop even from overhead — going idle")
                self._overhead_mode = False
                self._restore_camera()
                self.ctx.delay.sleep_range(3.0, 6.0)
                if self.ctx.idle:
                    self.ctx.idle.maybe_idle()
                self._state = State.FIND_SPOT
                return

        # Fishing confirmed
        if self._overhead_mode:
            self._log("Fishing confirmed from overhead — restoring camera")
            self._overhead_mode = False
            self._restore_camera()

        self._last_activity_time = time.time()
        self._last_inv_count = inv_count
        self._log(f"Fishing confirmed (inv={inv_count})")

        # Activity burst after clicking — natural fidgeting
        if self.ctx.idle and self.ctx.rng.chance(0.6):
            self.ctx.idle.force_burst()

        self._state = State.FISHING

    # ── CAMERA RECOVERY ──────────────────────────────────────

    def _pitch_camera_overhead(self) -> None:
        """Hold up arrow to pitch camera to full overhead view."""
        rng = self.ctx.rng
        inp = self.ctx.input

        key = rng.choice(['up', 'w']) if rng.chance(0.5) else 'up'
        hold = rng.truncated_gauss(3.5, 0.5, 2.8, 4.5)
        self._log(f"Pitching camera up ({key}, {hold:.1f}s)")
        inp.key_hold(key, hold)
        self.ctx.delay.sleep_range(0.3, 0.6)

    def _restore_camera(self) -> None:
        """Pitch camera back down and zoom in to a normal playing angle."""
        rng = self.ctx.rng
        inp = self.ctx.input

        # Pitch back down to a comfortable angle
        key = rng.choice(['down', 's']) if rng.chance(0.5) else 'down'
        hold = rng.truncated_gauss(1.8, 0.4, 1.2, 2.5)
        self._log(f"Restoring camera ({key}, {hold:.1f}s)")
        inp.key_hold(key, hold)
        self.ctx.delay.sleep_range(0.2, 0.4)

        # Zoom back in a bit
        self._ensure_mouse_in_game_view()
        zoom_ticks = int(rng.truncated_gauss(8, 2, 5, 12))
        inp.scroll(dy=zoom_ticks)
        self.ctx.delay.sleep_range(0.2, 0.5)

    # ── FISHING ──────────────────────────────────────────────

    def _do_fishing(self) -> None:
        # AFK breaks during fishing (capped at session-varied duration)
        if self.ctx.idle and self.ctx.idle.maybe_afk_break(
            max_duration=self._afk_max_duration
        ):
            # Reset activity timer after AFK so we don't immediately re-find
            self._last_activity_time = time.time()
            return

        # Quick XP drop check
        if self.check_xp_drop():
            self._last_activity_time = time.time()

        # Periodic inventory check
        self._wait_checks += 1
        check_inv_now = self._wait_checks % 4 == 0

        if check_inv_now:
            inv_count = self.ctx.vision.count_inventory_items(skip_slots=[0, 1])

            # Did we gain fish?
            if inv_count > self._last_inv_count:
                self._last_activity_time = time.time()
                self._last_inv_count = inv_count

            # Inventory full → drop
            if inv_count >= self.inv_full_count:
                self._log(f"Inventory full ({inv_count}), dropping")
                self._state = State.DROPPING
                return

            gap = time.time() - self._last_activity_time

            # Log progress
            self._log(
                f"Fishing... inv={inv_count}/{self.inv_full_count} "
                f"gap={gap:.0f}s/{self._activity_timeout:.0f}s "
                f"caught={self._fish_caught} drops={self._drop_cycles} "
                f"time={self.elapsed_str()}"
            )

            # No activity for too long → spot moved, re-find
            if gap >= self._activity_timeout:
                self._log(f"No activity for {gap:.0f}s, spot moved — re-finding")
                self._state = State.FIND_SPOT
                return

        # Idle behaviors between polls
        if self.ctx.idle and self.ctx.rng.chance(0.08):
            self.ctx.idle.maybe_idle()
        else:
            time.sleep(self.ctx.rng.truncated_gauss(0.4, 0.08, 0.25, 0.55))

    # ── DROPPING ─────────────────────────────────────────────

    def _do_dropping(self) -> None:
        rng = self.ctx.rng

        # Variable reaction time before dropping — not always instant
        reaction = rng.weighted_choice(
            ['instant', 'quick', 'slow', 'afk'],
            [25, 40, 25, 10],
        )
        if reaction == 'quick':
            # Noticed pretty fast, brief pause
            wait = rng.truncated_gauss(1.0, 0.4, 0.3, 2.0)
            self._log(f"Inventory full — dropping in {wait:.1f}s")
            self.ctx.delay.sleep_range(wait * 0.9, wait * 1.1)
        elif reaction == 'slow':
            # Took a moment to notice
            wait = rng.truncated_gauss(4.0, 1.5, 2.0, 8.0)
            self._log(f"Inventory full — noticed after {wait:.1f}s")
            self.ctx.delay.sleep_range(wait * 0.9, wait * 1.1)
        elif reaction == 'afk':
            # Was looking away, longer delay + maybe idle
            wait = rng.truncated_gauss(12.0, 4.0, 6.0, 25.0)
            self._log(f"Inventory full — AFK for {wait:.1f}s before dropping")
            self.ctx.delay.sleep_range(wait * 0.9, wait * 1.1)
            if self.ctx.idle:
                self.ctx.idle.force_burst()
        # else instant — drop immediately

        dropped = self.drop_inventory(skip_slots={0, 1})

        # Verify inventory is empty (only rod + feathers remain)
        remaining = self.ctx.vision.count_inventory_items(skip_slots=[0, 1])
        if remaining > 0:
            self._log(f"Still {remaining} items after drop, retrying stragglers")
            extra = self.drop_inventory(skip_slots={0, 1})
            dropped += extra

        self._fish_caught += dropped
        self._drop_cycles += 1
        self._find_failures = 0
        self._log(
            f"Dropped {dropped} fish "
            f"(total: {self._fish_caught}, cycle #{self._drop_cycles})"
        )

        # Activity burst after dropping — natural post-action fidgeting
        if self.ctx.idle and self.ctx.rng.chance(0.45):
            self.ctx.idle.force_burst()

        self._state = State.FIND_SPOT
