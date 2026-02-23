"""
Wintertodt Minigame Script

Chops bruma roots, fletches them into kindling, feeds the lit brazier,
drinks rejuvenation potions when cold, and makes new potions between rounds.

State machine:
    FIND_ROOTS -> CLICK_ROOTS -> CHOPPING -> FLETCHING ->
    FIND_FIRE -> CLICK_FIRE -> FEEDING ->
    (inv empty -> partial chop cycle | WT energy 0% -> INTERMISSION -> FIND_ROOTS)

Requires:
- Object Markers: red (FFFF0000) on bruma roots
- Object Markers: yellow (FFE7FF00) on lit brazier
- Object Markers: blue (FF00A4FF) on unlit brazier
- NPC Indicators / Object Markers: cyan (#00FFFF) on potion ingredient 1
- Object Markers: green (FF00FF00) on potion ingredient 2
- RuneLite XP Drop plugin set to magenta (FF00FF)
- Potion in inventory slot 0, tinderbox in slot 2, knife in slot 27
- Start bot at beginning of a round
"""

import os
import time
from enum import Enum, auto
from typing import Optional, Callable

import cv2
import numpy as np

import indigo
from indigo.script import Script, ScriptConfig, ScriptContext
from indigo.vision import Color, ColorCluster, GameRegions, Region
from indigo.core.timing import NORMAL_ACTION


class State(Enum):
    FIND_ROOTS = auto()
    CLICK_ROOTS = auto()
    CHOPPING = auto()
    FLETCHING = auto()
    FIND_FIRE = auto()
    CLICK_FIRE = auto()
    FEEDING = auto()
    INTERMISSION = auto()


# ── Object Markers colors ────────────────────────────────────────────

ROOTS_COLOR = Color.from_hex("FFFF0000")         # Red bruma roots
LIT_BRAZIER_COLOR = Color.from_hex("FFE7FF00")   # Yellow lit brazier
UNLIT_BRAZIER_COLOR = Color.from_hex("FF00A4FF")  # Blue unlit brazier

INGREDIENT_CYAN = Color(r=0, g=255, b=255)        # Potion ingredient 1
INGREDIENT_GREEN = Color.from_hex("FF00FF00")     # Potion ingredient 2

# ── HUD bar regions (game-relative) ─────────────────────────────────

WARMTH_BAR = Region(7, 30, 198, 13)
WARMTH_COLOR = Color.from_hex("ff5300")

WT_ENERGY_BAR = Region(8, 46, 197, 11)
ENERGY_GREEN = Color.from_hex("00d000")
ENERGY_RED = Color.from_hex("df0000")

BAR_TOLERANCE = 30
MIN_COL_PIXELS = 3

# ── Inventory layout ────────────────────────────────────────────────

POTION_SLOT = 0
TINDERBOX_SLOT = 2
KNIFE_SLOT = 27
RESERVED_SLOTS = {POTION_SLOT, TINDERBOX_SLOT, KNIFE_SLOT}
MAX_ROOT_SLOTS = 25  # 28 - 3 reserved

# ── Timing constants ────────────────────────────────────────────────

FEED_STALL_BASE = 1.8
CHOP_STALL_TIMEOUT = 10.0
CHOP_CONFIRM_TIMEOUT = 3.0

# Fletching — template matching on bruma root icon
LOG_TEMPLATE_PATH = os.path.join(
    os.path.dirname(indigo.__file__), "templates", "bruma_log.png"
)
INVENTORY_REGION = Region(
    GameRegions.INV_START_X, GameRegions.INV_START_Y,
    4 * (GameRegions.INV_SLOT_W + GameRegions.INV_GAP_X) - GameRegions.INV_GAP_X,
    7 * (GameRegions.INV_SLOT_H + GameRegions.INV_GAP_Y) - GameRegions.INV_GAP_Y,
)
SLOT_PITCH_X = GameRegions.INV_SLOT_W + GameRegions.INV_GAP_X   # 42
SLOT_PITCH_Y = GameRegions.INV_SLOT_H + GameRegions.INV_GAP_Y   # 36
ROOT_MATCH_THRESHOLD = 0.80
FLETCH_CONFIRM_TIMEOUT = 4.0   # seconds to confirm fletching started
FLETCH_STALL_TIMEOUT = 5.0     # seconds without root count decrease = cold flinch
FLETCH_POLL_INTERVAL = 0.5     # template matching is heavier than XP drop check

# Brazier monitoring during feeding
BRAZIER_GRACE_PERIOD = 2.0
FEED_POLL_MIN = 0.15
FEED_POLL_MAX = 0.25


class WintertodtScript(Script):
    """Wintertodt minigame: chop roots, fletch kindling, feed brazier, manage potions."""

    def __init__(self, ctx: ScriptContext, max_hours: float = 6.0,
                 fletch: bool = True,
                 on_log: Optional[Callable[[str], None]] = None):
        super().__init__(
            config=ScriptConfig(name="Wintertodt", max_runtime_hours=max_hours,
                                reserved_slots=3),
            ctx=ctx,
            on_log=on_log,
        )
        self._fletch = fletch
        self._state = State.FIND_ROOTS
        self._last_target: Optional[ColorCluster] = None
        self._find_failures = 0

        # Session-varied parameters (set in on_start)
        self._warmth_threshold: float = 35.0
        self._feed_stall_base: float = 1.8

        # Chop cycle tracking
        self._chop_target: int = MAX_ROOT_SLOTS  # full inv first cycle
        self._chop_cycle: int = 0  # 0=first, 1+=partial
        self._chop_last_count = 0
        self._chop_last_increase = 0.0
        self._chop_confirmed = False

        # Fletching tracking — template matching on log icon
        self._log_template = None           # cv2 image, loaded in on_start
        self._fletch_initial_logs = 0       # log count when fletching started
        self._fletch_prev_count = 0         # previous poll's count (for stall detection)
        self._fletch_last_change_time = 0.0 # when count last decreased from prev poll
        self._fletch_confirmed = False
        self._fletch_start_time = 0.0

        # Feeding tracking
        self._feed_last_count = 0
        self._feed_last_decrease = 0.0
        self._feed_start_count = 0
        self._brazier_visible = False
        self._brazier_lost_time: Optional[float] = None

        # Intermission
        self._potion_made = False
        self._intermission_start = 0.0
        self._intermission_logged = False

        # Stats
        self._rounds_completed = 0
        self._kindling_fed = 0
        self._potions_drunk = 0
        self._potions_made = 0

    # ── Lifecycle ────────────────────────────────────────────────────

    def _warmth_critical(self) -> bool:
        """Returns True if warmth is dangerously low (for idle interruption)."""
        warmth = self._read_warmth()
        return warmth >= 0 and warmth <= self._warmth_threshold

    def _click_inventory_tab(self) -> None:
        """Click inventory tab to ensure it's active before slot interactions."""
        tx, ty = self.ctx.vision.to_screen(*GameRegions.TAB_INVENTORY.center)
        jx = int(self.ctx.rng.truncated_gauss(0, 4, -8, 8))
        jy = int(self.ctx.rng.truncated_gauss(0, 4, -8, 8))
        self.ctx.input.click(tx + jx, ty + jy)
        self.ctx.delay.sleep_range(0.15, 0.35)

    def on_start(self) -> None:
        if not os.path.exists(LOG_TEMPLATE_PATH):
            raise RuntimeError(
                "Bruma log template not captured — run `indigo test capture-log` "
                "with bruma logs in inventory"
            )
        self._log_template = cv2.imread(LOG_TEMPLATE_PATH, cv2.IMREAD_COLOR)
        self._log(f"Loaded log template ({self._log_template.shape[1]}x{self._log_template.shape[0]})")

        self._warmth_threshold = self.ctx.rng.truncated_gauss(35.0, 3.0, 30.0, 40.0)
        self._feed_stall_base = self.ctx.rng.truncated_gauss(1.8, 0.3, 1.2, 2.5)
        self._log(f"Warmth threshold: {self._warmth_threshold:.0f}%, feed reaction: {self._feed_stall_base:.1f}s")
        self._log(f"Mode: {'fletch' if self._fletch else 'no-fletch (raw logs)'}")
        self._log("Ensure: red=roots, yellow=lit brazier, blue=unlit brazier")
        self._log("Ensure: potion slot 0, tinderbox slot 2, knife slot 27")

        if self.ctx.idle:
            self.ctx.idle.set_interrupt_check(self._warmth_critical)

        warmth = self._read_warmth()
        if warmth < 0:
            self._log("WARNING: Cannot read warmth bar")
        else:
            self._log(f"Warmth: {warmth:.0f}%")

        energy = self._read_wt_energy()
        if energy < 0:
            self._log("WARNING: Cannot read WT energy bar")
        else:
            self._log(f"WT Energy: {energy:.0f}%")

    def on_stop(self) -> None:
        feed_type = "kindling" if self._fletch else "logs"
        self._log(
            f"Session: {self._rounds_completed} rounds, "
            f"{self._kindling_fed} {feed_type} fed, "
            f"{self._potions_drunk} potions drunk, "
            f"{self._potions_made} potions made, "
            f"time={self.elapsed_str()}"
        )

    # ── Bar Reading ──────────────────────────────────────────────────

    def _read_bar_fill(self, region: Region, color: Color) -> float:
        """Read a horizontal bar's fill percentage via column scanning.

        Returns 0.0-100.0, or -1.0 if no bar pixels detected.
        """
        frame = self.ctx.vision.grab(region)
        bgr = np.array(color.to_bgr(), dtype=np.uint8)
        lower = np.clip(bgr.astype(np.int16) - BAR_TOLERANCE, 0, 255).astype(np.uint8)
        upper = np.clip(bgr.astype(np.int16) + BAR_TOLERANCE, 0, 255).astype(np.uint8)
        mask = cv2.inRange(frame, lower, upper)

        col_counts = np.sum(mask > 0, axis=0)
        filled_cols = np.where(col_counts >= MIN_COL_PIXELS)[0]
        if len(filled_cols) == 0:
            return -1.0

        rightmost = int(filled_cols[-1]) + 1
        return rightmost / region.width * 100.0

    def _read_warmth(self) -> float:
        """Read warmth bar fill percentage (0.0-100.0, or -1.0 if unreadable)."""
        return self._read_bar_fill(WARMTH_BAR, WARMTH_COLOR)

    def _read_wt_energy(self) -> float:
        """Read Wintertodt energy bar fill percentage (green portion).

        Returns 0.0-100.0, or -1.0 if bar not visible.
        """
        green_pct = self._read_bar_fill(WT_ENERGY_BAR, ENERGY_GREEN)
        if green_pct >= 0:
            return green_pct

        red_pct = self._read_bar_fill(WT_ENERGY_BAR, ENERGY_RED)
        if red_pct >= 0:
            return 0.0

        return -1.0

    # ── Global Checks ────────────────────────────────────────────────

    def _check_warmth(self) -> bool:
        """Drink potion if warmth is below threshold. Returns True if drank."""
        warmth = self._read_warmth()
        if warmth < 0 or warmth > self._warmth_threshold:
            return False

        # Ensure inventory tab is active — idle may have left us on stats/etc
        self._click_inventory_tab()

        if not self.ctx.vision.slot_has_item(POTION_SLOT):
            self._log(f"Warmth low ({warmth:.0f}%) but no potion!")
            return False

        cx, cy = self.ctx.vision.slot_screen_click_point(POTION_SLOT)
        self.ctx.input.click(cx, cy)
        self.ctx.delay.sleep(NORMAL_ACTION)
        self.ctx.delay.sleep_range(0.5, 1.0)
        self._potions_drunk += 1
        self._log(f"Drank potion (warmth={warmth:.0f}%, threshold={self._warmth_threshold:.0f}%)")
        return True

    def _check_round_end(self) -> bool:
        """Check if WT energy hit 0%. Transitions to INTERMISSION if so."""
        energy = self._read_wt_energy()
        if energy < 0:
            return False
        if energy > 1.0:
            return False

        self._log(f"Round over (WT energy={energy:.0f}%)")
        self._intermission_start = time.time()
        self._potion_made = False
        self._intermission_logged = False
        self._chop_target = MAX_ROOT_SLOTS
        self._chop_cycle = 0
        self._state = State.INTERMISSION
        return True

    # ── Inventory Helpers ────────────────────────────────────────────

    def _root_count(self) -> int:
        """Count inventory items, skipping reserved slots."""
        return self.ctx.vision.count_inventory_items(skip_slots=list(RESERVED_SLOTS))

    # ── Main Loop ────────────────────────────────────────────────────

    def loop(self) -> None:
        # Warmth check — ALWAYS, every state, every loop iteration
        drank = self._check_warmth()
        if drank and self._state == State.FEEDING:
            self._log("Drank during feeding, re-clicking fire")
            self._find_failures = 0
            self._state = State.FIND_FIRE
            return
        if drank and self._state == State.FLETCHING:
            # Drinking interrupts fletching animation — restart knife click
            self._log("Drank during fletching, restarting knife")
            self._fletch_last_change_time = time.time()
            self._fletch_start_time = time.time()
            self._click_knife_on_log()
            return

        # Round-end check (not during intermission)
        if self._state != State.INTERMISSION:
            if self._check_round_end():
                return

        if self._state == State.FIND_ROOTS:
            self._do_find_roots()
        elif self._state == State.CLICK_ROOTS:
            self._do_click_roots()
        elif self._state == State.CHOPPING:
            self._do_chopping()
        elif self._state == State.FLETCHING:
            self._do_fletching()
        elif self._state == State.FIND_FIRE:
            self._do_find_fire()
        elif self._state == State.CLICK_FIRE:
            self._do_click_fire()
        elif self._state == State.FEEDING:
            self._do_feeding()
        elif self._state == State.INTERMISSION:
            self._do_intermission()

    # ── FIND_ROOTS ───────────────────────────────────────────────────

    def _do_find_roots(self) -> None:
        roots = self.find_target(ROOTS_COLOR)
        if roots:
            self._last_target = roots
            self._find_failures = 0
            self._state = State.CLICK_ROOTS
            self._log(f"Found roots at {roots.click_point} (area={roots.area})")
        else:
            self._find_failures += 1
            if self._find_failures >= 3:
                self._log("No roots after 3 tries, searching...")
                roots = self.search_for_target(ROOTS_COLOR)
                if roots:
                    self._last_target = roots
                    self._find_failures = 0
                    self._state = State.CLICK_ROOTS
                    return
                self._find_failures = 0
            self._log("No roots found, waiting...")
            self.ctx.delay.sleep_range(1.5, 3.0)
            if self.ctx.idle:
                self.ctx.idle.maybe_idle()

    # ── CLICK_ROOTS ──────────────────────────────────────────────────

    def _do_click_roots(self) -> None:
        if not self._last_target:
            self._state = State.FIND_ROOTS
            return

        x, y = self._last_target.click_point
        self.click_target(x, y)
        self.ctx.delay.sleep(NORMAL_ACTION)

        self._chop_last_count = self._root_count()
        self._chop_last_increase = time.time()
        self._chop_confirmed = False
        self._log(f"Clicked roots (inv={self._chop_last_count}, target={self._chop_target})")
        self._state = State.CHOPPING

    # ── CHOPPING ─────────────────────────────────────────────────────

    def _do_chopping(self) -> None:
        now = time.time()
        count = self._root_count()

        if count > self._chop_last_count:
            if not self._chop_confirmed:
                self._chop_confirmed = True
                self._log("Chopping confirmed")
                if self.ctx.idle and self.ctx.rng.chance(0.6):
                    self.ctx.idle.force_burst()
            self._chop_last_count = count
            self._chop_last_increase = now

        # Check if we've reached our chop target
        if count >= self._chop_target:
            if self._fletch and self._chop_cycle == 0:
                # First cycle: fletch logs into kindling for max points
                self._log(f"Chop target reached ({count}/{self._chop_target}), fletching")
                self._init_fletching(count)
                self._state = State.FLETCHING
            else:
                # No-fletch mode or partial cycles: feed logs directly
                self._log(f"Chop target reached ({count}/{self._chop_target}), feeding logs directly")
                self._find_failures = 0
                self._state = State.FIND_FIRE
            return

        # Stall detection
        if not self._chop_confirmed:
            if now - self._chop_last_increase > CHOP_CONFIRM_TIMEOUT:
                self._log("Chopping not confirmed, retrying")
                self._find_failures = 0
                self._state = State.FIND_ROOTS
                return
        else:
            if now - self._chop_last_increase > CHOP_STALL_TIMEOUT:
                self._log(f"Chopping stalled at {count} roots, retrying")
                self._find_failures = 0
                self._state = State.FIND_ROOTS
                return

        # Poll tempo — tighter in no-fletch mode (need to feed sooner)
        if self._fletch:
            poll_wait = self.ctx.rng.truncated_gauss(2.5, 0.5, 1.5, 3.5)
        else:
            poll_wait = self.ctx.rng.truncated_gauss(2.0, 0.4, 1.5, 2.5)
        self.ctx.delay.sleep_range(poll_wait * 0.9, poll_wait * 1.1)

        if self._fletch and self._chop_confirmed and self.ctx.idle and self.ctx.rng.chance(0.08):
            self.ctx.idle.maybe_idle()

    # ── FLETCHING ────────────────────────────────────────────────────

    def _find_log_slots(self) -> set:
        """Find inventory slots still containing bruma logs via template matching.

        Single screen grab of the full inventory, one matchTemplate call,
        then maps match positions to the slot grid.

        Returns set of slot indices, or empty set if none found.
        Returns {-1} if template not loaded.
        """
        if self._log_template is None:
            return {-1}

        frame = self.ctx.vision.grab(INVENTORY_REGION)
        result = cv2.matchTemplate(frame, self._log_template, cv2.TM_CCOEFF_NORMED)
        locs = np.where(result >= ROOT_MATCH_THRESHOLD)

        if len(locs[0]) == 0:
            return set()

        # Map match positions to inventory slot grid
        th, tw = self._log_template.shape[:2]
        matched_slots = set()
        for y, x in zip(locs[0], locs[1]):
            cx = x + tw // 2
            cy = y + th // 2
            col = cx // SLOT_PITCH_X
            row = cy // SLOT_PITCH_Y
            if col < 4 and row < 7:
                slot = row * 4 + col
                if slot not in RESERVED_SLOTS:
                    matched_slots.add(slot)

        return matched_slots

    def _init_fletching(self, log_count: int) -> None:
        """Reset tracking and start knife interaction."""
        self._fletch_initial_logs = log_count
        self._fletch_prev_count = log_count
        self._fletch_confirmed = False
        self._fletch_start_time = time.time()
        self._fletch_last_change_time = time.time()
        self._log(f"Fletching {log_count} logs")
        self._click_knife_on_log()

    def _click_knife_on_log(self) -> bool:
        """Click knife then log (or log then knife) to start fletching."""
        self.cancel_selection()
        log_slots = self._find_log_slots()
        if not log_slots or log_slots == {-1}:
            self._log("No logs to fletch")
            return False

        slot = self.ctx.rng.choice(list(log_slots))
        lx, ly = self.ctx.vision.slot_screen_click_point(slot)
        kx, ky = self.ctx.vision.slot_screen_click_point(KNIFE_SLOT)

        # ~35% chance: click log first, then knife
        if self.ctx.rng.chance(0.35):
            self.ctx.input.click(lx, ly)
            self.ctx.delay.sleep(NORMAL_ACTION)
            self.ctx.delay.sleep_range(0.2, 0.5)
            self.ctx.input.click(kx, ky)
        else:
            self.ctx.input.click(kx, ky)
            self.ctx.delay.sleep(NORMAL_ACTION)
            self.ctx.delay.sleep_range(0.2, 0.5)
            self.ctx.input.click(lx, ly)

        self.ctx.delay.sleep(NORMAL_ACTION)

        # Bruma logs auto-fletch all — no confirm dialog
        self.ctx.delay.sleep_range(0.5, 1.0)

        # Move mouse out of inventory so hover tooltip doesn't interfere
        # with template matching during the polling phase
        self._ensure_mouse_in_game_view()

        self._fletch_last_change_time = time.time()
        return True

    def _do_fletching(self) -> None:
        now = time.time()

        # Count remaining logs via template matching
        log_slots = self._find_log_slots()
        if -1 in log_slots:
            time.sleep(FLETCH_POLL_INTERVAL)
            return

        logs = len(log_slots)

        # Track decreases from PREVIOUS poll (not all-time low).
        # Template noise can cause ±1 oscillation — using prev-poll
        # means any dip resets the stall timer, preventing false stalls.
        if logs < self._fletch_prev_count:
            self._fletch_last_change_time = now
            if not self._fletch_confirmed:
                self._fletch_confirmed = True
                self._log(f"Fletching confirmed (logs: {self._fletch_initial_logs} -> {logs})")
        self._fletch_prev_count = logs

        # Completion: no logs left
        if logs == 0:
            self._log(f"Fletching done ({self._fletch_initial_logs} logs), finding fire")
            self._find_failures = 0
            self._state = State.FIND_FIRE
            return

        # Confirm timeout — fletching never started
        if not self._fletch_confirmed:
            if now - self._fletch_start_time > FLETCH_CONFIRM_TIMEOUT:
                self._log(f"Fletching not confirmed after {FLETCH_CONFIRM_TIMEOUT:.0f}s, retrying knife")
                self._click_knife_on_log()
                self._fletch_start_time = time.time()
                return
        else:
            # Stall detection (cold flinch) — log count stopped decreasing
            if now - self._fletch_last_change_time > FLETCH_STALL_TIMEOUT:
                self._log(f"Fletch stall ({logs} logs remaining), re-knifing")
                self._click_knife_on_log()
                return

        # Light idle during fletching (~6%)
        if self._fletch_confirmed and self.ctx.idle and self.ctx.rng.chance(0.06):
            self.ctx.idle.maybe_idle()
        else:
            time.sleep(FLETCH_POLL_INTERVAL)

    # ── FIND_FIRE ────────────────────────────────────────────────────

    def _do_find_fire(self) -> None:
        # Check for unlit brazier first — player auto-uses tinderbox on it
        unlit = self.find_target(UNLIT_BRAZIER_COLOR)
        if unlit:
            self._last_target = unlit
            self._find_failures = 0
            self._state = State.CLICK_FIRE
            self._log(f"Found UNLIT brazier at {unlit.click_point} (area={unlit.area})")
            return

        fire = self.find_target(LIT_BRAZIER_COLOR)
        if fire:
            self._last_target = fire
            self._find_failures = 0
            self._state = State.CLICK_FIRE
            self._log(f"Found brazier at {fire.click_point} (area={fire.area})")
        else:
            self._find_failures += 1
            if self._find_failures >= 3:
                # Search for unlit first, then lit
                self._log("No brazier after 3 tries, searching...")
                unlit = self.search_for_target(UNLIT_BRAZIER_COLOR)
                if unlit:
                    self._last_target = unlit
                    self._find_failures = 0
                    self._state = State.CLICK_FIRE
                    self._log("Found unlit brazier via search")
                    return
                fire = self.search_for_target(LIT_BRAZIER_COLOR)
                if fire:
                    self._last_target = fire
                    self._find_failures = 0
                    self._state = State.CLICK_FIRE
                    return
                self._find_failures = 0
            self._log("No brazier found, waiting...")
            self.ctx.delay.sleep_range(0.8, 1.5)

    # ── CLICK_FIRE ───────────────────────────────────────────────────

    def _do_click_fire(self) -> None:
        if not self._last_target:
            self._state = State.FIND_FIRE
            return

        x, y = self._last_target.click_point
        self.click_target(x, y)

        # Wait for walk + start feeding animation
        walk_wait = self.ctx.rng.truncated_gauss(2.5, 0.5, 1.5, 3.5)
        self.ctx.delay.sleep_range(walk_wait * 0.9, walk_wait * 1.1)

        # Initialize feeding tracking
        self._feed_start_count = self._root_count()
        self._feed_last_count = self._feed_start_count
        self._feed_last_decrease = time.time()
        self._brazier_visible = True
        self._brazier_lost_time = None
        self._log(f"Clicked brazier (inv kindling={self._feed_start_count})")
        self._state = State.FEEDING

    # ── FEEDING ──────────────────────────────────────────────────────

    def _do_feeding(self) -> None:
        now = time.time()

        # 1. Check for blue (unlit) — highest priority, immediate react
        unlit = self.find_target(UNLIT_BRAZIER_COLOR)
        if unlit:
            self._log("Brazier went out (blue detected), re-finding fire")
            self._find_failures = 0
            self._state = State.FIND_FIRE
            return

        # 2. Check for yellow (lit) — dual detection with grace period
        lit = self.find_target(LIT_BRAZIER_COLOR)
        if lit:
            # Yellow visible — all good, clear any grace period
            if self._brazier_lost_time is not None:
                self._log("Brazier yellow reappeared (was broken, another player fixed)")
                self._brazier_lost_time = None
            self._brazier_visible = True
        else:
            # Yellow not visible
            if self._brazier_visible and self._brazier_lost_time is None:
                # Just disappeared — enter grace period
                self._brazier_lost_time = now
                self._brazier_visible = False
            elif self._brazier_lost_time is not None:
                # In grace period — check if exceeded
                grace_elapsed = now - self._brazier_lost_time
                if grace_elapsed > BRAZIER_GRACE_PERIOD:
                    self._log(f"Brazier lost for {grace_elapsed:.1f}s (no blue/yellow), re-finding fire")
                    self._find_failures = 0
                    self._state = State.FIND_FIRE
                    return

        # 3. Inventory count — track feeding progress
        count = self._root_count()

        if count < self._feed_last_count:
            self._feed_last_count = count
            self._feed_last_decrease = now

        # Feeding complete — inventory depleted
        if count <= 0:
            fed = self._feed_start_count
            self._kindling_fed += fed
            self._log(
                f"Feeding done ({fed} kindling, {self._kindling_fed} total), "
                f"time={self.elapsed_str()}"
            )
            self._transition_after_feeding()
            return

        # 4. Stall detection — cold hit (brazier still lit but we stopped feeding)
        stall_time = now - self._feed_last_decrease
        stall_threshold = self.ctx.rng.truncated_gauss(
            self._feed_stall_base, 0.3, self._feed_stall_base * 0.6, self._feed_stall_base * 1.8,
        )
        if stall_time > stall_threshold:
            self._log(f"Cold hit ({stall_time:.1f}s), re-clicking fire")
            self._find_failures = 0
            self._state = State.FIND_FIRE
            return

        # Tight poll — monitor brazier + warmth + inv
        self.ctx.delay.sleep_range(FEED_POLL_MIN, FEED_POLL_MAX)

    # ── Transition After Feeding ─────────────────────────────────────

    def _transition_after_feeding(self) -> None:
        """Decide next action after feeding empties inventory."""
        energy = self._read_wt_energy()

        # If energy is very low, let the round-end check handle it
        if energy >= 0 and energy <= 5.0:
            self._log(f"WT energy low ({energy:.0f}%), going to roots (round-end imminent)")
            self._find_failures = 0
            self._state = State.FIND_ROOTS
            return

        # No-fletch: always full chop — raw logs need more volume for points
        if not self._fletch:
            self._chop_cycle += 1
            self._chop_target = MAX_ROOT_SLOTS
            self._log(f"Full chop cycle {self._chop_cycle} (target={MAX_ROOT_SLOTS}, no-fletch)")
            self._find_failures = 0
            self._state = State.FIND_ROOTS
            return

        # Partial chop — scale target to remaining energy
        if energy > 0:
            base_target = int(energy / 2)
            jitter = int(self.ctx.rng.truncated_gauss(0, 1.5, -3, 3))
            target = max(3, min(MAX_ROOT_SLOTS, base_target + jitter))
        else:
            # Can't read energy bar — default to moderate chop
            target = int(self.ctx.rng.truncated_gauss(12, 3, 8, MAX_ROOT_SLOTS))

        self._chop_cycle += 1
        self._chop_target = target
        self._log(f"Partial chop cycle {self._chop_cycle} (target={target})")
        self._find_failures = 0
        self._state = State.FIND_ROOTS

    # ── INTERMISSION ─────────────────────────────────────────────────

    def _do_intermission(self) -> None:
        if not self._intermission_logged:
            self._intermission_logged = True
            self._rounds_completed += 1
            self._log(
                f"Round {self._rounds_completed} complete — "
                f"{self._kindling_fed} kindling fed, "
                f"{self._potions_drunk} potions drunk, "
                f"time={self.elapsed_str()}"
            )

        if not self._potion_made:
            self._try_make_potion()

        energy = self._read_wt_energy()
        if energy > 5.0:
            self._log(f"New round starting (WT energy={energy:.0f}%)")
            self._chop_target = MAX_ROOT_SLOTS
            self._chop_cycle = 0
            self._find_failures = 0
            self._state = State.FIND_ROOTS
            return

        self.ctx.delay.sleep_range(1.5, 2.5)
        if self.ctx.idle:
            if self.ctx.rng.chance(0.12):
                self.ctx.idle.maybe_idle()
            self.ctx.idle.maybe_afk_break(max_duration=45.0)

    # ── Potion Making ────────────────────────────────────────────────

    def _wait_inv_change(self, before_count: int, timeout: float = 3.0) -> bool:
        """Fast-poll until inventory count changes. Returns True if changed."""
        start = time.time()
        while time.time() - start < timeout:
            if self.should_stop:
                return False
            current = self.ctx.vision.count_inventory_items()
            if current != before_count:
                return True
            time.sleep(0.05)
        return False

    def _find_new_item_slot(self, before: set) -> Optional[int]:
        """Find which slot a new item appeared in (already detected by count change)."""
        for i in range(28):
            if i in before:
                continue
            if self.ctx.vision.slot_has_item(i):
                return i
        return None

    def _occupied_slots(self) -> set:
        """Return set of currently occupied inventory slot indices."""
        return {i for i in range(28) if self.ctx.vision.slot_has_item(i)}

    def _try_make_potion(self) -> None:
        """Make a rejuvenation potion during intermission if slot 0 is empty.

        Picks up one of each ingredient, interrupting the dispenser immediately
        to avoid getting multiple items. Uses the next action (clicking the other
        ingredient, or a misclick) as the interrupt.
        """
        if self._potion_made:
            return
        if self.ctx.vision.slot_has_item(POTION_SLOT):
            self._potion_made = True
            return

        self._log("Potion slot empty, making a new one")

        # Find BOTH ingredients before clicking anything
        cyan = self.find_target(INGREDIENT_CYAN)
        if not cyan:
            cyan = self.search_for_target(INGREDIENT_CYAN)
        if not cyan:
            self._log("Can't find cyan ingredient")
            return

        green = self.find_target(INGREDIENT_GREEN)
        if not green:
            green = self.search_for_target(INGREDIENT_GREEN)
        if not green:
            self._log("Can't find green ingredient")
            return

        # Step 1: Click cyan, fast-poll for pickup, immediately click green to interrupt
        before = self._occupied_slots()
        before_count = len(before)
        self.click_target(*cyan.click_point)

        if not self._wait_inv_change(before_count):
            self._log("Cyan ingredient didn't appear in inventory")
            return

        # Immediately click green — interrupts cyan dispenser AND starts green pickup
        before2 = self._occupied_slots()
        before2_count = len(before2)
        cyan_slot = self._find_new_item_slot(before)
        self.click_target(*green.click_point)

        if cyan_slot is not None:
            self._log(f"Got cyan ingredient in slot {cyan_slot}")

        # Step 2: Fast-poll for green pickup, then interrupt
        if not self._wait_inv_change(before2_count):
            self._log("Green ingredient didn't appear in inventory")
            return

        green_slot = self._find_new_item_slot(before2)

        # Interrupt green dispenser — click on the ground near it
        gx, gy = green.click_point
        ox = int(self.ctx.rng.truncated_gauss(0, 15, -30, 30))
        oy = int(self.ctx.rng.truncated_gauss(20, 8, 10, 40))
        self.ctx.input.click(gx + ox, gy + oy)

        if green_slot is not None:
            self._log(f"Got green ingredient in slot {green_slot}")

        # Step 3: Use ingredients on each other
        if cyan_slot is None or green_slot is None:
            self._log("Couldn't identify ingredient slots")
            return

        self.cancel_selection()
        self.ctx.delay.sleep_range(0.3, 0.6)

        first, second = (cyan_slot, green_slot) if self.ctx.rng.chance(0.5) else (green_slot, cyan_slot)

        cx, cy = self.ctx.vision.slot_screen_click_point(first)
        self.ctx.input.click(cx, cy)
        self.ctx.delay.sleep(NORMAL_ACTION)
        self.ctx.delay.sleep_range(0.3, 0.8)

        cx, cy = self.ctx.vision.slot_screen_click_point(second)
        self.ctx.input.click(cx, cy)
        self.ctx.delay.sleep(NORMAL_ACTION)
        self.ctx.delay.sleep_range(1.0, 2.0)

        if self.ctx.vision.slot_has_item(POTION_SLOT):
            self._potion_made = True
            self._potions_made += 1
            self._log("Potion created")
        else:
            self._log("Potion creation may have failed")
