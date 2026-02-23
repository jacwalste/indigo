"""
Bonfire Firemaking Script

Burns logs at a bonfire near a bank booth. Uses inventory logs on a
cyan-highlighted bonfire, confirms "all" with space, watches XP drops
to know when burning finishes, then banks for more logs.

State machine:
    FIND_FIRE -> USE_LOG -> CONFIRM_ALL -> BURNING ->
    FIND_BANK -> CLICK_BANK -> WITHDRAW_LOGS -> CLOSE_BANK -> FIND_FIRE

Requires:
- NPC Indicators or Object Markers: cyan (#00FFFF) on bonfire
- Object Markers: red (#FF0000) on bank booth
- RuneLite XP Drop plugin set to magenta (FF00FF)
- Bank interface log slot calibrated (GameRegions.BANK_LOG_SLOT)
- Logs in the first visible bank slot
"""

import os
import time
from enum import Enum, auto
from typing import Optional, Callable, List

import indigo
from indigo.script import Script, ScriptConfig, ScriptContext
from indigo.vision import Color, ColorCluster, GameRegions
from indigo.core.timing import NORMAL_ACTION


class State(Enum):
    FIND_FIRE = auto()
    USE_LOG = auto()
    CONFIRM_ALL = auto()
    BURNING = auto()
    FIND_BANK = auto()
    CLICK_BANK = auto()
    WITHDRAW_LOGS = auto()
    CLOSE_BANK = auto()


# Cyan from NPC Indicators / Object Markers on bonfire: #00FFFF
FIRE_COLOR = Color(r=0, g=255, b=255)

# Red from Object Markers on bank booth: #FF0000
BANK_COLOR = Color(r=255, g=0, b=0)

# Template for bank interface verification (same deposit button image as deposit box)
BANK_TEMPLATE = os.path.join(
    os.path.dirname(indigo.__file__), "templates", "deposit_box.png"
)

# Time to wait for first XP drop after confirming — if none, bonfire interaction failed
FIRST_DROP_TIMEOUT = 8.0

# XP drop poll interval (fast enough to catch every drop)
XP_POLL_INTERVAL = 0.3

# Safety multiplier: if gap between drops exceeds avg * this, assume done
DONE_MULTIPLIER = 2.0

# Absolute safety cap — if no drop for this long, definitely done
DONE_MAX_GAP = 15.0


class BonfireScript(Script):
    """Burn logs at a bonfire and bank for more."""

    def __init__(self, ctx: ScriptContext, max_hours: float = 6.0,
                 on_log: Optional[Callable[[str], None]] = None):
        super().__init__(
            config=ScriptConfig(name="Bonfire", max_runtime_hours=max_hours),
            ctx=ctx,
            on_log=on_log,
        )
        self._state = State.FIND_FIRE
        self._last_target: Optional[ColorCluster] = None
        self._find_failures = 0
        self._burn_start_count = 0
        self._logs_burned = 0
        self._bank_trips = 0
        self._bank_fail_streak = 0  # consecutive failed withdrawals

        # XP drop tracking
        self._drop_times: List[float] = []
        self._last_drop_time = 0.0
        self._burn_confirmed = False
        self._confirm_time = 0.0
        self._last_poll_was_drop = False  # debounce: only count rising edges

    def on_start(self) -> None:
        if GameRegions.BANK_LOG_SLOT.x == 0 and GameRegions.BANK_LOG_SLOT.y == 0:
            raise RuntimeError(
                "BANK_LOG_SLOT not calibrated - run `indigo test bankslot` and update "
                "GameRegions.BANK_LOG_SLOT in indigo/vision.py"
            )
        self._log("Starting bonfire - ensure cyan on fire, red on bank, XP drops magenta")

    def loop(self) -> None:
        if self._state == State.FIND_FIRE:
            self._do_find_fire()
        elif self._state == State.USE_LOG:
            self._do_use_log()
        elif self._state == State.CONFIRM_ALL:
            self._do_confirm_all()
        elif self._state == State.BURNING:
            self._do_burning()
        elif self._state == State.FIND_BANK:
            self._do_find_bank()
        elif self._state == State.CLICK_BANK:
            self._do_click_bank()
        elif self._state == State.WITHDRAW_LOGS:
            self._do_withdraw_logs()
        elif self._state == State.CLOSE_BANK:
            self._do_close_bank()

    # ── FIND_FIRE ──────────────────────────────────────────────

    def _do_find_fire(self) -> None:
        fire = self.find_target(FIRE_COLOR)
        if fire:
            self._last_target = fire
            self._find_failures = 0
            self._state = State.USE_LOG
            self._log(f"Found bonfire at {fire.click_point} (area={fire.area})")
        else:
            self._find_failures += 1
            if self._find_failures >= 3:
                self._log("No bonfire after 3 tries, searching...")
                fire = self.search_for_target(FIRE_COLOR)
                if fire:
                    self._last_target = fire
                    self._find_failures = 0
                    self._state = State.USE_LOG
                    return
                self._find_failures = 0
            self._log("No bonfire found, waiting...")
            self.ctx.delay.sleep_range(1.5, 3.0)
            if self.ctx.idle:
                self.ctx.idle.maybe_idle()

    # ── USE_LOG ────────────────────────────────────────────────

    def _pick_random_log_slot(self) -> Optional[int]:
        """Pick a random occupied inventory slot."""
        occupied = [i for i in range(28) if self.ctx.vision.slot_has_item(i)]
        if not occupied:
            return None
        return self.ctx.rng.choice(occupied)

    def _do_use_log(self) -> None:
        if not self._last_target:
            self._state = State.FIND_FIRE
            return

        slot = self._pick_random_log_slot()
        if slot is None:
            self._log("No logs in inventory, going to bank")
            self._state = State.FIND_BANK
            return

        # Click the inventory log slot
        cx, cy = self.ctx.vision.slot_screen_click_point(slot)
        self.ctx.input.click(cx, cy)
        self.ctx.delay.sleep(NORMAL_ACTION)

        # Short human-like pause before clicking fire
        self.ctx.delay.sleep_range(0.3, 0.8)

        # Click the bonfire
        x, y = self._last_target.click_point
        self.ctx.input.click(x, y)
        self.ctx.delay.sleep(NORMAL_ACTION)

        self._log(f"Used log (slot {slot}) on bonfire")
        self._state = State.CONFIRM_ALL

    # ── CONFIRM_ALL ────────────────────────────────────────────

    def _do_confirm_all(self) -> None:
        # Wait for the "how many" dialog to appear
        self.ctx.delay.sleep_range(1.0, 2.0)

        # Press space to confirm "all"
        self.ctx.input.key_tap('space')
        self.ctx.delay.sleep(NORMAL_ACTION)

        self._burn_start_count = self.ctx.vision.count_inventory_items()
        self._drop_times = []
        self._last_drop_time = 0.0
        self._burn_confirmed = False
        self._confirm_time = time.time()
        self._last_poll_was_drop = False
        self._log(f"Confirmed all (inv={self._burn_start_count}), watching for XP drops")
        self._state = State.BURNING

    # ── BURNING ────────────────────────────────────────────────

    def _do_burning(self) -> None:
        # AFK break only after burning is confirmed (don't interrupt verification).
        # Fire goes out after 180s — cap AFK at 120s to leave buffer for banking/restarting.
        if self._burn_confirmed and self.ctx.idle and self.ctx.idle.maybe_afk_break(max_duration=120.0):
            return

        now = time.time()

        # Poll for XP drop — debounce so each drop only counts once.
        # XP text lingers on screen ~1-2s, so multiple polls see the same drop.
        # Only count the rising edge (no-drop → drop transition).
        raw_drop = self.check_xp_drop()
        new_drop = raw_drop and not self._last_poll_was_drop
        self._last_poll_was_drop = raw_drop

        if new_drop:
            if not self._burn_confirmed:
                self._burn_confirmed = True
                self._log("Burning confirmed (first XP drop)")
                # Activity burst right after burning starts — most natural time to fidget
                if self.ctx.idle and self.ctx.rng.chance(0.6):
                    self.ctx.idle.force_burst()
            self._drop_times.append(now)
            self._last_drop_time = now

        # Phase 1: Waiting for first drop — verify burning actually started
        if not self._burn_confirmed:
            elapsed = now - self._confirm_time
            if elapsed >= FIRST_DROP_TIMEOUT:
                self._log(f"No XP drop after {elapsed:.1f}s — bonfire failed, retrying")
                fire = self.search_for_target(FIRE_COLOR)
                if fire:
                    self._last_target = fire
                self._state = State.FIND_FIRE
                return

            time.sleep(XP_POLL_INTERVAL)
            return

        # Phase 2: Burning confirmed — track intervals and detect completion
        drop_count = len(self._drop_times)
        gap_since_last = now - self._last_drop_time

        # Build running average of intervals between drops
        avg_interval = 0.0
        if drop_count >= 2:
            intervals = [self._drop_times[i] - self._drop_times[i - 1]
                         for i in range(1, drop_count)]
            avg_interval = sum(intervals) / len(intervals)

        # Need at least half the inventory worth of drops before considering done.
        # This ensures a solid sample of the actual burn rate.
        min_drops = max(3, self._burn_start_count // 2)

        # Done conditions:
        # 1. Enough drops collected and gap exceeds 2x the average
        # 2. Or absolute safety cap exceeded (always active)
        done = False
        if drop_count >= min_drops and avg_interval > 0 and gap_since_last > avg_interval * DONE_MULTIPLIER:
            done = True
            self._log(
                f"No XP drop for {gap_since_last:.1f}s "
                f"(avg interval {avg_interval:.1f}s, {drop_count} drops) — burning complete"
            )
        elif gap_since_last > DONE_MAX_GAP:
            done = True
            self._log(f"No XP drop for {gap_since_last:.1f}s (safety cap) — burning complete")

        if done:
            self._logs_burned += self._burn_start_count
            self._log(
                f"Burned {self._burn_start_count} logs "
                f"({drop_count} drops, {self._logs_burned} total)"
            )
            self._state = State.FIND_BANK
            return

        # Log progress periodically
        if drop_count > 0 and drop_count % 7 == 0 and new_drop:
            self._log(
                f"Burning... drops={drop_count}/{self._burn_start_count} "
                f"avg={avg_interval:.1f}s "
                f"burned={self._logs_burned} trips={self._bank_trips} "
                f"time={self.elapsed_str()}"
            )

        # Idle early in the burn — player would fidget after clicking, not mid-burn
        if self.ctx.idle and drop_count <= 10 and new_drop and self.ctx.rng.chance(0.15):
            self.ctx.idle.maybe_idle()
        else:
            time.sleep(XP_POLL_INTERVAL)

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
                # First try zoom-out (bank is usually nearby, just off-screen)
                self._log("No bank after 3 tries, zooming out...")
                bank = self.zoom_out_find(BANK_COLOR)
                if bank:
                    self._last_target = bank
                    self._find_failures = 0
                    self._state = State.CLICK_BANK
                    return
                # Zoom didn't help — rotate camera too
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

        # Wait for character to walk to bank + interface to open.
        # Scale walk time up on successive failures — misclicks may leave us further away.
        streak = min(self._bank_fail_streak, 5)
        base_walk = self.ctx.rng.truncated_gauss(2.0, 0.4, 1.5, 3.0)
        extra = streak * self.ctx.rng.truncated_gauss(1.5, 0.5, 0.8, 2.5)
        walk_time = base_walk + extra

        if self._bank_fail_streak > 0:
            self._log(f"Clicked bank, walking over (max ~{walk_time:.1f}s, +{extra:.1f}s for retry #{self._bank_fail_streak})")
        else:
            self._log(f"Clicked bank, walking over (max ~{walk_time:.1f}s)")

        # Poll for bank interface to open instead of blind sleep
        # No idle during banking — zoom/scroll can scroll inside the bank interface
        opened = self.wait_for_bank_open(BANK_TEMPLATE, GameRegions.BANK_DEPOSIT_BUTTON, max_wait=walk_time)
        if opened:
            self._log("Bank opened")
        else:
            self._log("Walk timer expired, checking bank anyway")

        self._state = State.WITHDRAW_LOGS

    # ── WITHDRAW_LOGS ──────────────────────────────────────────

    def _do_withdraw_logs(self) -> None:
        # Verify bank interface is actually open (if template available)
        if os.path.exists(BANK_TEMPLATE):
            bank_open = self.ctx.vision.template_match_region(
                GameRegions.BANK_DEPOSIT_BUTTON, BANK_TEMPLATE, threshold=0.8,
            )
            if not bank_open:
                self._bank_fail_streak += 1
                self._log(
                    f"Bank not visible (streak={self._bank_fail_streak}), "
                    f"bank click missed — re-finding"
                )
                self._state = State.FIND_BANK
                return

        # Click the log slot in the bank interface — tap to avoid drag-scroll
        self.tap_region_jittered(GameRegions.BANK_LOG_SLOT)
        self.ctx.delay.sleep(NORMAL_ACTION)

        # Wait for inventory to fill
        self.ctx.delay.sleep_range(0.5, 1.0)

        inv_count = self.ctx.vision.count_inventory_items()
        if inv_count == 0:
            self._log("Withdraw failed (inv empty), retrying click")
            self.tap_region_jittered(GameRegions.BANK_LOG_SLOT)
            self.ctx.delay.sleep(NORMAL_ACTION)
            self.ctx.delay.sleep_range(0.5, 1.0)
            inv_count = self.ctx.vision.count_inventory_items()

        if inv_count == 0:
            self._bank_fail_streak += 1
            self._log(f"Withdraw still failed (streak={self._bank_fail_streak}), closing bank and retrying")
            self.ctx.input.key_tap('esc')
            self.ctx.delay.sleep(NORMAL_ACTION)
            self._state = State.FIND_BANK
            return

        self._bank_fail_streak = 0
        self._bank_trips += 1
        self._log(f"Withdrew logs (inv={inv_count}, trip #{self._bank_trips})")

        self._state = State.CLOSE_BANK

    # ── CLOSE_BANK ─────────────────────────────────────────────

    def _do_close_bank(self) -> None:
        self.ctx.delay.sleep_range(0.3, 0.6)
        self.ctx.input.key_tap('esc')
        self.ctx.delay.sleep(NORMAL_ACTION)

        self._find_failures = 0
        self._state = State.FIND_FIRE
        self._log("Bank closed, finding fire")
