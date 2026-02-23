"""
Yew Trees Script (with Deposit Box Banking)

Chops yew trees using NPC Indicators (cyan highlight).
Banks logs via deposit box (red highlight) when inventory is full.

Yews are very slow — long gaps between XP drops are normal.
Uses XP drop gap tracking to detect when a tree has fallen.

State machine:
    FIND_TREE -> CLICK_TREE -> WAITING -> FIND_BANK -> CLICK_BANK -> DEPOSITING -> FIND_TREE

Requires:
- NPC Indicators: cyan (#00FFFF) on yew trees
- Object Markers or similar: red (#FF0000 / FFFF0000) on deposit box
- RuneLite XP Drop plugin set to magenta (FF00FF)
- Axe wielded (all 28 inventory slots available for logs)
- Optional: indigo/templates/deposit_box.png for deposit box verification
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
    FIND_TREE = auto()
    CLICK_TREE = auto()
    WAITING = auto()
    FIND_BANK = auto()
    CLICK_BANK = auto()
    DEPOSITING = auto()


# Cyan from NPC Indicators plugin: #00FFFF
TREE_COLOR = Color(r=0, g=255, b=255)

# Red from Object Markers on deposit box: #FF0000
DEPOSIT_BOX_COLOR = Color(r=255, g=0, b=0)

# Template for deposit box interface verification
DEPOSIT_BOX_TEMPLATE = os.path.join(
    os.path.dirname(indigo.__file__), "templates", "deposit_box.png"
)

# XP drop verification timeout — yews are slow, need extra time for walk + first chop
CHOP_XP_TIMEOUT = 25.0

# Max seconds between XP drops before assuming tree fell — yews chop very slowly
DROP_GAP_TIMEOUT = 45.0


class YewsScript(Script):
    """Chop yew trees and bank via deposit box."""

    def __init__(self, ctx: ScriptContext, max_hours: float = 6.0,
                 axe_in_inventory: bool = False,
                 on_log: Optional[Callable[[str], None]] = None):
        super().__init__(
            config=ScriptConfig(
                name="Yews", max_runtime_hours=max_hours,
                reserved_slots=1 if axe_in_inventory else 0,
            ),
            ctx=ctx,
            on_log=on_log,
        )
        self._state = State.FIND_TREE
        self._last_target: Optional[ColorCluster] = None
        self._wait_checks = 0
        self._last_drop_time = 0.0
        self._find_failures = 0
        self._logs_banked = 0
        self._bank_trips = 0
        self._bank_fail_streak = 0
        self._just_banked = False

    def on_start(self) -> None:
        mode = f"bank mode, inv full at {self.inv_full_count}"
        if self.config.reserved_slots > 0:
            mode += f" ({self.config.reserved_slots} reserved slot)"
        self._log(f"Chopping yews ({mode}) - ensure NPC Indicators (cyan) + deposit box (red)")

    def loop(self) -> None:
        if self._state == State.FIND_TREE:
            self._do_find_tree()
        elif self._state == State.CLICK_TREE:
            self._do_click_tree()
        elif self._state == State.WAITING:
            self._do_waiting()
        elif self._state == State.FIND_BANK:
            self._do_find_bank()
        elif self._state == State.CLICK_BANK:
            self._do_click_bank()
        elif self._state == State.DEPOSITING:
            self._do_depositing()

    # -- FIND_TREE -------------------------------------------------

    def _do_find_tree(self) -> None:
        tree = self.find_target(TREE_COLOR)
        if tree:
            self._last_target = tree
            self._find_failures = 0
            self._state = State.CLICK_TREE
            self._log(f"Found yew at {tree.click_point} (area={tree.area})")
        else:
            self._find_failures += 1
            if self._find_failures >= 3:
                self._log("No yew after 3 tries, searching...")
                tree = self.search_for_target(TREE_COLOR)
                if tree:
                    self._last_target = tree
                    self._find_failures = 0
                    self._state = State.CLICK_TREE
                    return
                self._find_failures = 0
            self._log("No yew tree found, waiting...")
            self.ctx.delay.sleep_range(1.5, 3.0)
            if self.ctx.idle:
                self.ctx.idle.maybe_idle()

    # -- CLICK_TREE ------------------------------------------------

    def _do_click_tree(self) -> None:
        if not self._last_target:
            self._state = State.FIND_TREE
            return

        x, y = self._last_target.click_point

        # Verify target still exists before clicking (tree may have fallen)
        if not self.is_target_near(TREE_COLOR, (x, y)):
            self._log("Yew gone before click (stale target), re-finding")
            self._state = State.FIND_TREE
            return

        self.click_target(x, y)
        self.ctx.delay.sleep(NORMAL_ACTION)

        self._wait_checks = 0

        # Verify chopping actually started via XP drop
        # Extra time after banking to account for walking back to trees
        timeout = CHOP_XP_TIMEOUT
        if self._just_banked:
            timeout += self.ctx.rng.truncated_gauss(8.0, 2.0, 5.0, 12.0)
            self._just_banked = False

        inv_count = self.ctx.vision.count_inventory_items()
        self._log(f"Clicked yew (inv={inv_count}), verifying chop (timeout={timeout:.0f}s)...")
        if not self.verify_xp_drop(timeout=timeout):
            self._log("No XP drop after clicking, assumed misclick")
            self._state = State.FIND_TREE
            return

        self._last_drop_time = time.time()
        self._log(f"Chopping confirmed (inv={inv_count})")

        # Activity burst after clicking
        if self.ctx.idle and self.ctx.rng.chance(0.6):
            self.ctx.idle.force_burst()

        self._state = State.WAITING

    # -- WAITING ---------------------------------------------------

    def _do_waiting(self) -> None:
        # Check for AFK break (if triggered, gap will naturally detect tree fell)
        if self.ctx.idle and self.ctx.idle.maybe_afk_break():
            return

        # Quick XP drop check first
        if self.check_xp_drop():
            self._last_drop_time = time.time()

        gap = time.time() - self._last_drop_time

        # No XP drop for too long -> tree fell, find next one
        if gap >= DROP_GAP_TIMEOUT:
            inv_count = self.ctx.vision.count_inventory_items()
            if inv_count >= self.inv_full_count:
                self._log("Inventory full, banking")
                self._state = State.FIND_BANK
            else:
                self._log(f"No XP drop for {gap:.0f}s, tree fell -- finding next")
                self._state = State.FIND_TREE
            return

        # Periodic inventory check for full inventory
        self._wait_checks += 1
        if self._wait_checks % 5 == 0:
            inv_count = self.ctx.vision.count_inventory_items()
            if inv_count >= self.inv_full_count:
                self._log("Inventory full, banking")
                self._state = State.FIND_BANK
                return
            self._log(
                f"Chopping... inv={inv_count}/{self.inv_full_count} "
                f"gap={gap:.0f}s/{DROP_GAP_TIMEOUT:.0f}s "
                f"banked={self._logs_banked} trips={self._bank_trips} "
                f"time={self.elapsed_str()}"
            )

        # Idle behaviors between polls
        if self.ctx.idle and self.ctx.rng.chance(0.08):
            self.ctx.idle.maybe_idle()
        else:
            time.sleep(self.ctx.rng.truncated_gauss(0.4, 0.08, 0.25, 0.55))

    # -- FIND_BANK -------------------------------------------------

    def _do_find_bank(self) -> None:
        bank = self.find_target(DEPOSIT_BOX_COLOR)
        if bank:
            self._last_target = bank
            self._find_failures = 0
            self._state = State.CLICK_BANK
            self._log(f"Found deposit box at {bank.click_point} (area={bank.area})")
        else:
            self._find_failures += 1
            if self._find_failures >= 3:
                self._log("No deposit box after 3 tries, zooming out...")
                bank = self.zoom_out_find(DEPOSIT_BOX_COLOR)
                if bank:
                    self._last_target = bank
                    self._find_failures = 0
                    self._state = State.CLICK_BANK
                    return
                self._log("Zoom-out failed, searching with camera rotation...")
                bank = self.search_for_target(DEPOSIT_BOX_COLOR)
                if bank:
                    self._last_target = bank
                    self._find_failures = 0
                    self._state = State.CLICK_BANK
                    return
                self._find_failures = 0
            self._log("No deposit box found, waiting...")
            self.ctx.delay.sleep_range(1.5, 3.0)

    # -- CLICK_BANK ------------------------------------------------

    def _do_click_bank(self) -> None:
        if not self._last_target:
            self._state = State.FIND_BANK
            return

        x, y = self._last_target.click_point
        self.click_target(x, y)

        # Scale walk time up on successive failures
        streak = min(self._bank_fail_streak, 5)
        base_walk = self.ctx.rng.truncated_gauss(12.0, 2.5, 8.0, 18.0)
        extra = streak * self.ctx.rng.truncated_gauss(1.5, 0.5, 0.8, 2.5)
        walk_time = base_walk + extra

        if self._bank_fail_streak > 0:
            self._log(f"Clicked deposit box, walking over (max ~{walk_time:.0f}s, +{extra:.0f}s for retry #{self._bank_fail_streak})")
        else:
            self._log(f"Clicked deposit box, walking over (max ~{walk_time:.0f}s)")

        opened = self.wait_for_bank_open(DEPOSIT_BOX_TEMPLATE, GameRegions.DEPOSIT_ALL_BUTTON, max_wait=walk_time)
        if opened:
            self._log("Deposit box opened")
        else:
            self._log("Walk timer expired, checking deposit box anyway")

        self._state = State.DEPOSITING

    # -- DEPOSITING ------------------------------------------------

    def _do_depositing(self) -> None:
        if os.path.exists(DEPOSIT_BOX_TEMPLATE):
            box_open = self.ctx.vision.template_match_region(
                GameRegions.DEPOSIT_ALL_BUTTON, DEPOSIT_BOX_TEMPLATE, threshold=0.8,
            )
            if not box_open:
                self._bank_fail_streak += 1
                self._log(
                    f"Deposit box not visible (streak={self._bank_fail_streak}), "
                    f"bank click missed -- re-finding"
                )
                self._state = State.FIND_BANK
                return

        deposited = self.deposit_all()

        remaining = self.ctx.vision.count_inventory_items()
        if remaining > self.config.reserved_slots:
            self._bank_fail_streak += 1
            self._log(
                f"Deposit incomplete: {remaining} items remain "
                f"(streak={self._bank_fail_streak}), retrying"
            )
            self._state = State.FIND_BANK
            return

        self._bank_fail_streak = 0
        self._just_banked = True
        self._logs_banked += deposited
        self._bank_trips += 1
        self._log(
            f"Deposited {deposited} logs "
            f"(total banked: {self._logs_banked}, trip #{self._bank_trips})"
        )

        self._find_failures = 0
        self._state = State.FIND_TREE
