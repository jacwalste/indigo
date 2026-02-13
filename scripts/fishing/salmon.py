"""
Salmon/Trout Fly Fishing + Cooking Script

Fly fishes salmon and trout, cooks them on a nearby fire, drops the
cooked fish, and repeats.  Slot 0 = fishing rod, slot 1 = feathers —
both are protected during drops.

State machine:
    FIND_SPOT -> CLICK_SPOT -> FISHING ->
    FIND_FIRE -> CLICK_FIRE -> CONFIRM_ALL -> COOKING ->
    (pass 0: back to CLICK_FIRE for second fish type) ->
    CONFIRM_ALL -> COOKING -> DROPPING -> FIND_SPOT

Since salmon and trout are caught together, the cooking interface only
cooks one type at a time.  After the first type finishes, we click the
fire again, confirm all, and cook the second type before dropping.

Requires:
- Object Markers: green (#43FF00 / FF43FF00) on fishing spot
- Object Markers: blue (#0013FF / FF0013FF) on fire
- RuneLite XP Drop plugin set to magenta (FF00FF)
- Fishing rod in slot 0, feathers in slot 1
"""

import time
from enum import Enum, auto
from typing import Optional, Callable, List

from indigo.script import Script, ScriptConfig, ScriptContext
from indigo.vision import Color, ColorCluster
from indigo.core.timing import NORMAL_ACTION


class State(Enum):
    FIND_SPOT = auto()
    CLICK_SPOT = auto()
    FISHING = auto()
    FIND_FIRE = auto()
    CLICK_FIRE = auto()
    CONFIRM_ALL = auto()
    COOKING = auto()
    DROPPING = auto()


# Green from Object Markers on fishing spot: #43FF00
SPOT_COLOR = Color(r=67, g=255, b=0)

# Blue from Object Markers on fire: #0013FF
FIRE_COLOR = Color(r=0, g=19, b=255)

# XP drop verification timeout after clicking spot
FISH_XP_TIMEOUT = 20.0

# Time to wait for first cooking XP drop before assuming fire click failed
COOK_FIRST_DROP_TIMEOUT = 8.0

# Cooking XP poll interval
COOK_POLL_INTERVAL = 0.3

# If gap between cooking drops exceeds avg * this, cooking is done
COOK_DONE_MULTIPLIER = 2.0

# Absolute safety cap for cooking gap
COOK_DONE_MAX_GAP = 15.0


class SalmonScript(Script):
    """Fly fish salmon/trout, cook on fire, drop, repeat."""

    def __init__(self, ctx: ScriptContext, max_hours: float = 6.0,
                 on_log: Optional[Callable[[str], None]] = None):
        super().__init__(
            config=ScriptConfig(
                name="Salmon", max_runtime_hours=max_hours,
                reserved_slots=2,  # rod + feathers
            ),
            ctx=ctx,
            on_log=on_log,
        )
        self._state = State.FIND_SPOT
        self._last_target: Optional[ColorCluster] = None
        self._find_failures = 0
        self._wait_checks = 0

        # Fishing tracking
        self._last_drop_time = 0.0
        self._fish_gap_timeout = 35.0  # session-varied in on_start
        self._just_cooked = False

        # Cooking tracking (two passes: salmon then trout, or vice versa)
        self._cook_pass = 0  # 0 = first fish type, 1 = second
        self._drop_times: List[float] = []
        self._cook_confirmed = False
        self._cook_start_count = 0
        self._confirm_time = 0.0
        self._last_poll_was_drop = False

        # Stats
        self._fish_cooked = 0
        self._cook_cycles = 0

    def on_start(self) -> None:
        self._fish_gap_timeout = self.ctx.rng.truncated_gauss(35.0, 5.0, 25.0, 45.0)
        self._log(
            f"Fly fishing salmon/trout (gap timeout={self._fish_gap_timeout:.0f}s, "
            f"inv full at {self.inv_full_count}) — "
            f"ensure green on spot, blue on fire, XP drops magenta"
        )

    def loop(self) -> None:
        if self._state == State.FIND_SPOT:
            self._do_find_spot()
        elif self._state == State.CLICK_SPOT:
            self._do_click_spot()
        elif self._state == State.FISHING:
            self._do_fishing()
        elif self._state == State.FIND_FIRE:
            self._do_find_fire()
        elif self._state == State.CLICK_FIRE:
            self._do_click_fire()
        elif self._state == State.CONFIRM_ALL:
            self._do_confirm_all()
        elif self._state == State.COOKING:
            self._do_cooking()
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
        timeout = FISH_XP_TIMEOUT
        if self._just_cooked:
            timeout += self.ctx.rng.truncated_gauss(8.0, 2.0, 5.0, 12.0)
            self._just_cooked = False

        inv_count = self.ctx.vision.count_inventory_items(skip_slots=[0, 1])
        self._log(f"Clicked spot (inv={inv_count}), verifying fish (timeout={timeout:.0f}s)...")
        if not self.verify_xp_drop(timeout=timeout):
            self._log("No XP drop after clicking, assumed misclick")
            self._state = State.FIND_SPOT
            return

        self._last_drop_time = time.time()
        self._log(f"Fishing confirmed (inv={inv_count})")

        # Activity burst after clicking — natural fidgeting
        if self.ctx.idle and self.ctx.rng.chance(0.6):
            self.ctx.idle.force_burst()

        self._state = State.FISHING

    # ── FISHING ──────────────────────────────────────────────

    def _do_fishing(self) -> None:
        # AFK breaks during fishing (natural)
        if self.ctx.idle and self.ctx.idle.maybe_afk_break():
            return

        # Quick XP drop check
        if self.check_xp_drop():
            self._last_drop_time = time.time()

        gap = time.time() - self._last_drop_time

        # No XP drop for too long — done fishing
        if gap >= self._fish_gap_timeout:
            inv_count = self.ctx.vision.count_inventory_items(skip_slots=[0, 1])
            if inv_count >= self.inv_full_count:
                self._log(f"Inventory full ({inv_count}), going to cook")
                self._cook_pass = 0
                self._state = State.FIND_FIRE
            else:
                self._log(f"No XP drop for {gap:.0f}s (inv={inv_count}), spot moved — re-finding")
                self._state = State.FIND_SPOT
            return

        # Periodic inventory check
        self._wait_checks += 1
        if self._wait_checks % 5 == 0:
            inv_count = self.ctx.vision.count_inventory_items(skip_slots=[0, 1])
            if inv_count >= self.inv_full_count:
                self._log(f"Inventory full ({inv_count}), going to cook")
                self._cook_pass = 0
                self._state = State.FIND_FIRE
                return
            self._log(
                f"Fishing... inv={inv_count}/{self.inv_full_count} "
                f"gap={gap:.0f}s/{self._fish_gap_timeout:.0f}s "
                f"cooked={self._fish_cooked} cycles={self._cook_cycles} "
                f"time={self.elapsed_str()}"
            )

        # Idle behaviors between polls
        if self.ctx.idle and self.ctx.rng.chance(0.08):
            self.ctx.idle.maybe_idle()
        else:
            time.sleep(self.ctx.rng.truncated_gauss(0.4, 0.08, 0.25, 0.55))

    # ── FIND_FIRE ────────────────────────────────────────────

    def _do_find_fire(self) -> None:
        fire = self.find_target(FIRE_COLOR)
        if fire:
            self._last_target = fire
            self._find_failures = 0
            self._state = State.CLICK_FIRE
            self._log(f"Found fire at {fire.click_point} (area={fire.area})")
        else:
            self._find_failures += 1
            if self._find_failures >= 3:
                self._log("No fire after 3 tries, searching...")
                fire = self.search_for_target(FIRE_COLOR)
                if fire:
                    self._last_target = fire
                    self._find_failures = 0
                    self._state = State.CLICK_FIRE
                    return
                self._find_failures = 0
            self._log("No fire found, waiting...")
            self.ctx.delay.sleep_range(1.5, 3.0)
            # No idle during cooking states — zoom/scroll can interfere

    # ── CLICK_FIRE ───────────────────────────────────────────

    def _do_click_fire(self) -> None:
        if not self._last_target:
            self._state = State.FIND_FIRE
            return

        x, y = self._last_target.click_point
        self.click_target(x, y)

        if self._cook_pass == 0:
            # First cook: walk to fire + interface open
            self.ctx.delay.sleep_range(2.0, 4.0)
        else:
            # Second cook: already standing at fire
            self.ctx.delay.sleep_range(0.8, 1.5)

        self._state = State.CONFIRM_ALL

    # ── CONFIRM_ALL ──────────────────────────────────────────

    def _do_confirm_all(self) -> None:
        # Wait for the cooking dialog
        self.ctx.delay.sleep_range(1.0, 2.0)

        # Press space to confirm "all"
        self.ctx.input.key_tap('space')
        self.ctx.delay.sleep(NORMAL_ACTION)

        self._cook_start_count = self.ctx.vision.count_inventory_items(skip_slots=[0, 1])
        self._drop_times = []
        self._last_drop_time = 0.0
        self._cook_confirmed = False
        self._confirm_time = time.time()
        self._last_poll_was_drop = False
        which = "first" if self._cook_pass == 0 else "second"
        self._log(f"Confirmed all — {which} fish type (inv={self._cook_start_count}), watching for XP drops")
        self._state = State.COOKING

    # ── COOKING ──────────────────────────────────────────────

    def _do_cooking(self) -> None:
        now = time.time()

        # Poll for XP drop — debounce so each drop only counts once
        raw_drop = self.check_xp_drop()
        new_drop = raw_drop and not self._last_poll_was_drop
        self._last_poll_was_drop = raw_drop

        if new_drop:
            if not self._cook_confirmed:
                self._cook_confirmed = True
                self._log("Cooking confirmed (first XP drop)")
                # Activity burst after cooking starts
                if self.ctx.idle and self.ctx.rng.chance(0.6):
                    self.ctx.idle.force_burst()
            self._drop_times.append(now)
            self._last_drop_time = now

        # Phase 1: Waiting for first drop — verify cooking actually started
        if not self._cook_confirmed:
            elapsed = now - self._confirm_time
            if elapsed >= COOK_FIRST_DROP_TIMEOUT:
                self._log(f"No XP drop after {elapsed:.1f}s — fire click failed, retrying")
                fire = self.search_for_target(FIRE_COLOR)
                if fire:
                    self._last_target = fire
                self._state = State.FIND_FIRE
                return

            time.sleep(COOK_POLL_INTERVAL)
            return

        # Phase 2: Cooking confirmed — track intervals and detect completion
        drop_count = len(self._drop_times)
        gap_since_last = now - self._last_drop_time

        # Running average of intervals between drops
        avg_interval = 0.0
        if drop_count >= 2:
            intervals = [self._drop_times[i] - self._drop_times[i - 1]
                         for i in range(1, drop_count)]
            avg_interval = sum(intervals) / len(intervals)

        # Need enough drops before considering done
        min_drops = max(3, self._cook_start_count // 2)

        done = False
        if drop_count >= min_drops and avg_interval > 0 and gap_since_last > avg_interval * COOK_DONE_MULTIPLIER:
            done = True
            self._log(
                f"No cooking XP for {gap_since_last:.1f}s "
                f"(avg interval {avg_interval:.1f}s, {drop_count} drops) — cooking complete"
            )
        elif gap_since_last > COOK_DONE_MAX_GAP:
            done = True
            self._log(f"No cooking XP for {gap_since_last:.1f}s (safety cap) — cooking complete")

        if done:
            if self._cook_pass == 0:
                # First fish type done — cook the second type
                # Re-find fire (camera may have rotated during idle)
                self._cook_pass = 1
                self._log("First fish type done, cooking second type")
                self._state = State.FIND_FIRE
            else:
                # Both types cooked — drop everything
                self._state = State.DROPPING
            return

        # Log progress periodically
        if drop_count > 0 and drop_count % 7 == 0 and new_drop:
            self._log(
                f"Cooking... drops={drop_count}/{self._cook_start_count} "
                f"avg={avg_interval:.1f}s "
                f"cooked={self._fish_cooked} cycles={self._cook_cycles} "
                f"time={self.elapsed_str()}"
            )

        # Idle early in cooking
        if self.ctx.idle and drop_count <= 10 and new_drop and self.ctx.rng.chance(0.15):
            self.ctx.idle.maybe_idle()
        else:
            time.sleep(COOK_POLL_INTERVAL)

    # ── DROPPING ─────────────────────────────────────────────

    def _do_dropping(self) -> None:
        dropped = self.drop_inventory(skip_slots={0, 1})

        # Verify inventory is empty (only rod + feathers remain)
        remaining = self.ctx.vision.count_inventory_items(skip_slots=[0, 1])
        if remaining > 0:
            self._log(f"Still {remaining} items after drop, retrying stragglers")
            extra = self.drop_inventory(skip_slots={0, 1})
            dropped += extra

        self._fish_cooked += dropped  # includes burnt fish
        self._cook_cycles += 1
        self._just_cooked = True
        self._find_failures = 0
        self._log(
            f"Dropped {dropped} fish "
            f"(total: {self._fish_cooked}, cycle #{self._cook_cycles})"
        )
        self._state = State.FIND_SPOT
