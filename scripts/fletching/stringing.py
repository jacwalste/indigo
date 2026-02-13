"""
Bow Stringing Script (Fletching)

Strings bows at a bank booth. Withdraws 14 unstrung bows and 14
bowstrings, uses one on the other, confirms "Make All" with space,
watches XP drops to know when stringing finishes, then banks for more.

State machine:
    FIND_BANK -> CLICK_BANK -> WITHDRAW_ITEMS -> CLOSE_BANK ->
    USE_ITEM -> CONFIRM_ALL -> STRINGING -> FIND_BANK

Requires:
- Object Markers: red (#FF0000) on bank booth
- RuneLite XP Drop plugin set to magenta (FF00FF)
- Bank interface slot 1 calibrated (GameRegions.BANK_LOG_SLOT) — unstrung bows
- Bank interface slot 2 calibrated (GameRegions.BANK_BOWSTRING_SLOT) — bowstrings
- Unstrung bows in bank slot 1, bowstrings in bank slot 2
"""

import os
import time
from enum import Enum, auto
from typing import Optional, Callable

import indigo
from indigo.script import Script, ScriptConfig, ScriptContext
from indigo.vision import Color, ColorCluster, GameRegions
from indigo.core.timing import NORMAL_ACTION


class State(Enum):
    FIND_BANK = auto()
    CLICK_BANK = auto()
    WITHDRAW_ITEMS = auto()
    CLOSE_BANK = auto()
    USE_ITEM = auto()
    CONFIRM_ALL = auto()
    STRINGING = auto()


# Red from Object Markers on bank booth: #FF0000
BANK_COLOR = Color(r=255, g=0, b=0)

# Template for bank interface verification (same deposit button image)
BANK_TEMPLATE = os.path.join(
    os.path.dirname(indigo.__file__), "templates", "deposit_box.png"
)

# Time to wait for first XP drop after confirming — if none, interaction failed
FIRST_DROP_TIMEOUT = 8.0

# XP drop poll interval
XP_POLL_INTERVAL = 0.3

# Gap since last visible XP drop before we consider stringing done.
# XP text lingers ~1-2s on screen, so after the last bow is strung
# the text fades and then this gap elapses — total ~4-5s reaction.
DONE_GAP_BASE = 3.5

# Absolute safety cap — if no XP drop for this long, definitely done
# (protects against edge cases where detection fails mid-batch)
DONE_MAX_GAP = 45.0


class StringingScript(Script):
    """String bows at a bank booth."""

    def __init__(self, ctx: ScriptContext, max_hours: float = 6.0,
                 on_log: Optional[Callable[[str], None]] = None):
        super().__init__(
            config=ScriptConfig(name="Stringing", max_runtime_hours=max_hours),
            ctx=ctx,
            on_log=on_log,
        )
        self._state = State.FIND_BANK
        self._last_target: Optional[ColorCluster] = None
        self._find_failures = 0
        self._bank_fail_streak = 0
        self._bank_trips = 0
        self._bows_strung = 0

        # XP drop tracking — simple visibility-based (no debounce needed)
        self._last_xp_visible = 0.0  # last time check_xp_drop() returned True
        self._string_confirmed = False
        self._confirm_time = 0.0
        self._done_gap = DONE_GAP_BASE  # randomized per cycle

    def on_start(self) -> None:
        if GameRegions.BANK_LOG_SLOT.x == 0 and GameRegions.BANK_LOG_SLOT.y == 0:
            raise RuntimeError(
                "BANK_LOG_SLOT not calibrated - run `indigo test bankslot` and update "
                "GameRegions.BANK_LOG_SLOT in indigo/vision.py"
            )
        if GameRegions.BANK_BOWSTRING_SLOT.x == 0 and GameRegions.BANK_BOWSTRING_SLOT.y == 0:
            raise RuntimeError(
                "BANK_BOWSTRING_SLOT not calibrated - run `indigo test bankslot` and update "
                "GameRegions.BANK_BOWSTRING_SLOT in indigo/vision.py"
            )
        self._log("Starting stringing - ensure red on bank, XP drops magenta, bows slot 1, strings slot 2")

    def loop(self) -> None:
        if self._state == State.FIND_BANK:
            self._do_find_bank()
        elif self._state == State.CLICK_BANK:
            self._do_click_bank()
        elif self._state == State.WITHDRAW_ITEMS:
            self._do_withdraw_items()
        elif self._state == State.CLOSE_BANK:
            self._do_close_bank()
        elif self._state == State.USE_ITEM:
            self._do_use_item()
        elif self._state == State.CONFIRM_ALL:
            self._do_confirm_all()
        elif self._state == State.STRINGING:
            self._do_stringing()

    # ── FIND_BANK ──────────────────────────────────────────────

    def _do_find_bank(self) -> None:
        bank = self.find_target(BANK_COLOR)
        if bank:
            self._last_target = bank
            self._find_failures = 0
            self._state = State.CLICK_BANK
            self._log(f"Found bank at {bank.click_point} (area={bank.area})")
        else:
            self._find_failures += 1
            if self._find_failures >= 3:
                self._log("No bank after 3 tries, zooming out...")
                bank = self.zoom_out_find(BANK_COLOR)
                if bank:
                    self._last_target = bank
                    self._find_failures = 0
                    self._state = State.CLICK_BANK
                    return
                self._log("Zoom-out failed, searching with camera rotation...")
                bank = self.search_for_target(BANK_COLOR)
                if bank:
                    self._last_target = bank
                    self._find_failures = 0
                    self._state = State.CLICK_BANK
                    return
                self._find_failures = 0
            self._log("No bank found, waiting...")
            self.ctx.delay.sleep_range(1.5, 3.0)
            # No idle during banking — zoom/scroll can interfere with bank interface

    # ── CLICK_BANK ─────────────────────────────────────────────

    def _do_click_bank(self) -> None:
        if not self._last_target:
            self._state = State.FIND_BANK
            return

        x, y = self._last_target.click_point
        self.ctx.input.click(x, y)

        # Scale walk time up on successive failures
        streak = min(self._bank_fail_streak, 5)
        base_walk = self.ctx.rng.truncated_gauss(2.0, 0.4, 1.5, 3.0)
        extra = streak * self.ctx.rng.truncated_gauss(1.5, 0.5, 0.8, 2.5)
        walk_time = base_walk + extra

        if self._bank_fail_streak > 0:
            self._log(f"Clicked bank, walking over (max ~{walk_time:.1f}s, +{extra:.1f}s for retry #{self._bank_fail_streak})")
        else:
            self._log(f"Clicked bank, walking over (max ~{walk_time:.1f}s)")

        # Poll for bank interface to open
        # No idle during banking — zoom/scroll can scroll inside the bank interface
        opened = self.wait_for_bank_open(BANK_TEMPLATE, GameRegions.BANK_DEPOSIT_BUTTON, max_wait=walk_time)
        if opened:
            self._log("Bank opened")
        else:
            self._log("Walk timer expired, checking bank anyway")

        self._state = State.WITHDRAW_ITEMS

    # ── WITHDRAW_ITEMS ─────────────────────────────────────────

    def _do_withdraw_items(self) -> None:
        # Verify bank interface is actually open
        if os.path.exists(BANK_TEMPLATE):
            bank_open = self.ctx.vision.template_match_region(
                GameRegions.BANK_DEPOSIT_BUTTON, BANK_TEMPLATE, threshold=0.8,
            )
            if not bank_open:
                self._bank_fail_streak += 1
                self._log(
                    f"Bank not visible (streak={self._bank_fail_streak}), "
                    f"bank click missed — closing and rotating camera"
                )
                # Dismiss any open interface before rotating — a scroll during
                # camera rotation would scroll the bank item list, not the camera.
                self.ctx.input.key_tap('esc')
                self.ctx.delay.sleep(NORMAL_ACTION)
                # Banker NPC may be blocking the booth click box.
                # Spin camera ~180 degrees to get a clear angle.
                key = self.ctx.rng.choice(['left', 'right'])
                hold = self.ctx.rng.truncated_gauss(2.5, 0.5, 1.8, 3.5)
                self.ctx.input.key_hold(key, hold)
                self.ctx.delay.sleep_range(0.3, 0.6)
                self._state = State.FIND_BANK
                return

        # Deposit any leftover items first (in case inventory isn't empty)
        inv_before = self.ctx.vision.count_inventory_items()
        if inv_before > 0:
            self._log(f"Inventory not empty ({inv_before} items), depositing first")
            self.click_region_jittered(GameRegions.BANK_DEPOSIT_BUTTON)
            self.ctx.delay.sleep(NORMAL_ACTION)
            self.ctx.delay.sleep_range(0.4, 0.8)

        # Click first bank slot (unstrung bows)
        self.click_region_jittered(GameRegions.BANK_LOG_SLOT)
        self.ctx.delay.sleep(NORMAL_ACTION)

        # Human-like pause between withdrawals
        self.ctx.delay.sleep_range(0.15, 0.4)

        # Click second bank slot (bowstrings)
        self.click_region_jittered(GameRegions.BANK_BOWSTRING_SLOT)
        self.ctx.delay.sleep(NORMAL_ACTION)

        # Wait for inventory to fill
        self.ctx.delay.sleep_range(0.6, 1.2)

        inv_count = self.ctx.vision.count_inventory_items()
        if inv_count < 28:
            self._log(f"Withdraw incomplete (inv={inv_count}), retrying...")
            self.click_region_jittered(GameRegions.BANK_LOG_SLOT)
            self.ctx.delay.sleep(NORMAL_ACTION)
            self.ctx.delay.sleep_range(0.15, 0.4)
            self.click_region_jittered(GameRegions.BANK_BOWSTRING_SLOT)
            self.ctx.delay.sleep(NORMAL_ACTION)
            self.ctx.delay.sleep_range(0.6, 1.2)
            inv_count = self.ctx.vision.count_inventory_items()

        if inv_count < 28:
            self._bank_fail_streak += 1
            self._log(f"Withdraw still failed (inv={inv_count}, streak={self._bank_fail_streak}), closing bank")
            self.ctx.input.key_tap('esc')
            self.ctx.delay.sleep(NORMAL_ACTION)
            self._state = State.FIND_BANK
            return

        self._bank_fail_streak = 0
        self._bank_trips += 1
        self._log(f"Withdrew items (inv={inv_count}, trip #{self._bank_trips})")

        self._state = State.CLOSE_BANK

    # ── CLOSE_BANK ─────────────────────────────────────────────

    def _is_bank_open(self) -> bool:
        """Check if bank interface is still visible."""
        if not os.path.exists(BANK_TEMPLATE):
            return False
        return self.ctx.vision.template_match_region(
            GameRegions.BANK_DEPOSIT_BUTTON, BANK_TEMPLATE, threshold=0.8,
        )

    def _do_close_bank(self) -> None:
        self.ctx.delay.sleep_range(0.3, 0.6)

        # Humanization: 85% Escape, 15% click X button
        if self.ctx.rng.chance(0.85):
            self.ctx.input.key_tap('esc')
        else:
            self.click_region_jittered(GameRegions.BANK_CLOSE_BUTTON)
            self._log("Closed bank via X button")

        self.ctx.delay.sleep(NORMAL_ACTION)

        # Verify bank actually closed — Escape/X can fail to register
        if self._is_bank_open():
            self._log("Bank still open after close attempt, pressing Escape again")
            self.ctx.input.key_tap('esc')
            self.ctx.delay.sleep(NORMAL_ACTION)

        self._find_failures = 0

        # Might pause after banking before starting — "got supplies, zones out"
        if self.ctx.idle:
            self.ctx.idle.maybe_afk_break(max_duration=90.0)

        self._state = State.USE_ITEM

    # ── USE_ITEM ───────────────────────────────────────────────

    def _pick_random_bow_slot(self) -> Optional[int]:
        """Pick a random occupied slot from first 14 (slots 0-13)."""
        occupied = [i for i in range(14) if self.ctx.vision.slot_has_item(i)]
        if not occupied:
            return None
        return self.ctx.rng.choice(occupied)

    def _pick_random_string_slot(self) -> Optional[int]:
        """Pick a random occupied slot from last 14 (slots 14-27)."""
        occupied = [i for i in range(14, 28) if self.ctx.vision.slot_has_item(i)]
        if not occupied:
            return None
        return self.ctx.rng.choice(occupied)

    def _do_use_item(self) -> None:
        bow_slot = self._pick_random_bow_slot()
        string_slot = self._pick_random_string_slot()

        if bow_slot is None or string_slot is None:
            self._log("Missing items in inventory, going to bank")
            self._state = State.FIND_BANK
            return

        # Click first item (unstrung bow)
        cx, cy = self.ctx.vision.slot_screen_click_point(bow_slot)
        self.ctx.input.click(cx, cy)
        self.ctx.delay.sleep(NORMAL_ACTION)

        # Human-like pause before clicking second item
        self.ctx.delay.sleep_range(0.3, 0.8)

        # Click second item (bowstring) — use-on pattern
        cx, cy = self.ctx.vision.slot_screen_click_point(string_slot)
        self.ctx.input.click(cx, cy)
        self.ctx.delay.sleep(NORMAL_ACTION)

        self._log(f"Used bow (slot {bow_slot}) on string (slot {string_slot})")
        self._state = State.CONFIRM_ALL

    # ── CONFIRM_ALL ────────────────────────────────────────────

    def _do_confirm_all(self) -> None:
        # Wait for the "Make X" dialog to appear
        self.ctx.delay.sleep_range(1.0, 2.0)

        # Press space to confirm "all"
        self.ctx.input.key_tap('space')
        self.ctx.delay.sleep(NORMAL_ACTION)

        self._last_xp_visible = 0.0
        self._string_confirmed = False
        self._confirm_time = time.time()
        # Randomize the done-gap per cycle — sometimes we notice fast, sometimes slow
        self._done_gap = self.ctx.rng.truncated_gauss(DONE_GAP_BASE, 0.8, 2.5, 5.5)
        self._log(f"Confirmed all, watching for XP drops (gap threshold={self._done_gap:.1f}s)")
        self._state = State.STRINGING

    # ── STRINGING ──────────────────────────────────────────────

    def _do_stringing(self) -> None:
        # Safety: if bank interface is somehow still open (Escape failed, etc.),
        # dismiss it before doing anything — idle's zoom_adjust would scroll inside it.
        if self._is_bank_open():
            self._log("Bank still open during stringing — dismissing")
            self.ctx.input.key_tap('esc')
            self.ctx.delay.sleep(NORMAL_ACTION)
            return

        # AFK break only after stringing confirmed — cap at 60s (stringing is fast)
        if self._string_confirmed and self.ctx.idle and self.ctx.idle.maybe_afk_break(max_duration=60.0):
            return

        now = time.time()

        # Simple visibility-based tracking: just check if XP drop text is on screen.
        # Bow stringing is fast (~1.8s/bow) so XP text overlaps continuously.
        # No debounce needed — we only care when drops STOP appearing.
        if self.check_xp_drop():
            self._last_xp_visible = now
            if not self._string_confirmed:
                self._string_confirmed = True
                self._log("Stringing confirmed (XP drop visible)")
                if self.ctx.idle and self.ctx.rng.chance(0.6):
                    self.ctx.idle.force_burst()

        # Phase 1: Waiting for first drop — verify stringing actually started
        if not self._string_confirmed:
            elapsed = now - self._confirm_time
            if elapsed >= FIRST_DROP_TIMEOUT:
                self._log(f"No XP drop after {elapsed:.1f}s — stringing failed, retrying")
                self._state = State.USE_ITEM
                return
            time.sleep(XP_POLL_INTERVAL)
            return

        # Phase 2: Stringing confirmed — detect when XP drops stop appearing
        gap = now - self._last_xp_visible

        if gap >= self._done_gap:
            self._bows_strung += 14
            self._log(
                f"XP drops stopped ({gap:.1f}s gap) — stringing complete "
                f"({self._bows_strung} total, trips={self._bank_trips}, time={self.elapsed_str()})"
            )
            # Randomized reaction delay — sometimes alert, sometimes AFK
            # This varies how quickly we go to bank after noticing we're done
            reaction = self.ctx.rng.truncated_gauss(3.0, 2.5, 0.5, 12.0)
            self._log(f"Reacting in {reaction:.1f}s")
            self.ctx.delay.sleep_range(reaction * 0.9, reaction * 1.1)
            # Might zone out after finishing — "done, not paying attention"
            if self.ctx.idle:
                self.ctx.idle.maybe_afk_break(max_duration=120.0)
            self._state = State.FIND_BANK
            return

        # Safety cap — something went wrong, bail
        total_elapsed = now - self._confirm_time
        if total_elapsed >= DONE_MAX_GAP:
            self._bows_strung += 14
            self._log(f"Safety cap ({total_elapsed:.0f}s) — stringing complete (forced)")
            self._state = State.FIND_BANK
            return

        # Idle while waiting — player would fidget during stringing
        if self.ctx.idle and self.ctx.rng.chance(0.08):
            self.ctx.idle.maybe_idle()
        else:
            time.sleep(XP_POLL_INTERVAL)
