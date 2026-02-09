"""
Willow Trees Script (with Deposit Box Banking)

Chops willow trees using NPC Indicators (cyan highlight).
Banks logs via deposit box (red highlight) when inventory is full.

Willows last longer than oaks — more logs per tree before falling,
so the idle timeout is higher (~35s).

State machine:
    FIND_TREE -> CLICK_TREE -> WAITING -> FIND_BANK -> CLICK_BANK -> DEPOSITING -> FIND_TREE

Requires:
- NPC Indicators: cyan (#00FFFF) on willow trees
- Object Markers or similar: red (#FF0000 / FFFF0000) on deposit box
- Axe wielded (all 28 inventory slots available for logs)
"""

import time
from enum import Enum, auto
from typing import Optional, Callable

from indigo.script import Script, ScriptConfig, ScriptContext
from indigo.vision import Color, ColorCluster
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


class WillowsScript(Script):
    """Chop willow trees and bank via deposit box."""

    IDLE_TIMEOUT_MIN = 8.0
    IDLE_TIMEOUT_MAX = 16.0

    def __init__(self, ctx: ScriptContext, max_hours: float = 6.0,
                 on_log: Optional[Callable[[str], None]] = None):
        super().__init__(
            config=ScriptConfig(name="Willows", max_runtime_hours=max_hours),
            ctx=ctx,
            on_log=on_log,
        )
        self._state = State.FIND_TREE
        self._last_target: Optional[ColorCluster] = None
        self._wait_checks = 0
        self._last_inv_count = 0
        self._last_gain_time = 0.0
        self._idle_timeout = 12.0
        self._find_failures = 0
        self._logs_banked = 0
        self._bank_trips = 0

    def on_start(self) -> None:
        self._log("Chopping willows (bank mode) - ensure NPC Indicators (cyan) + deposit box (red)")

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

    def _do_find_tree(self) -> None:
        tree = self.find_target(TREE_COLOR)
        if tree:
            self._last_target = tree
            self._find_failures = 0
            self._state = State.CLICK_TREE
            self._log(f"Found willow at {tree.click_point} (area={tree.area})")
        else:
            self._find_failures += 1
            if self._find_failures >= 3:
                self._log("No willow after 3 tries, searching...")
                tree = self.search_for_target(TREE_COLOR)
                if tree:
                    self._last_target = tree
                    self._find_failures = 0
                    self._state = State.CLICK_TREE
                    return
                self._find_failures = 0
            self._log("No willow tree found, waiting...")
            self.ctx.delay.sleep_range(1.5, 3.0)
            if self.ctx.idle:
                self.ctx.idle.maybe_idle()

    def _do_click_tree(self) -> None:
        if self._last_target:
            x, y = self._last_target.click_point
            self.click_target(x, y)
            self.ctx.delay.sleep(NORMAL_ACTION)

            self._last_inv_count = self.ctx.vision.count_inventory_items()
            self._last_gain_time = time.time()
            self._idle_timeout = self.ctx.rng.truncated_gauss(
                mean=12.0, stddev=2.0,
                min_val=self.IDLE_TIMEOUT_MIN, max_val=self.IDLE_TIMEOUT_MAX,
            )
            self._wait_checks = 0
            self._state = State.WAITING
            self._log(f"Clicked willow (inv={self._last_inv_count})")

    def _do_waiting(self) -> None:
        self.ctx.delay.sleep_range(2.0, 4.0)
        self._wait_checks += 1

        if self.ctx.idle:
            self.ctx.idle.maybe_idle()

        inv_count = self.ctx.vision.count_inventory_items()

        # Did we gain a log?
        gained = inv_count > self._last_inv_count
        if gained:
            self._last_inv_count = inv_count
            self._last_gain_time = time.time()

        idle_secs = time.time() - self._last_gain_time

        if self._wait_checks % 4 == 0:
            self._log(
                f"Chopping... inv={inv_count}/28 "
                f"idle={idle_secs:.0f}s/{self._idle_timeout:.0f}s "
                f"banked={self._logs_banked} trips={self._bank_trips} "
                f"time={self.elapsed_str()}"
            )

        # Inventory full → bank
        if inv_count >= 28:
            self._log("Inventory full, banking")
            self._state = State.FIND_BANK
            return

        # No new logs for a while → tree fell, find next one
        if idle_secs >= self._idle_timeout:
            self._log(f"No logs for {idle_secs:.0f}s, finding next tree")
            self._state = State.FIND_TREE
            return

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
                self._log("No deposit box after 3 tries, searching...")
                bank = self.search_for_target(DEPOSIT_BOX_COLOR)
                if bank:
                    self._last_target = bank
                    self._find_failures = 0
                    self._state = State.CLICK_BANK
                    return
                self._find_failures = 0
            self._log("No deposit box found, waiting...")
            self.ctx.delay.sleep_range(1.5, 3.0)
            if self.ctx.idle:
                self.ctx.idle.maybe_idle()

    def _do_click_bank(self) -> None:
        if self._last_target:
            x, y = self._last_target.click_point
            self.click_target(x, y)

            # Wait for character to run to deposit box + interface to open
            walk_time = self.ctx.rng.truncated_gauss(12.0, 2.5, 8.0, 18.0)
            self._log(f"Clicked deposit box, walking over (~{walk_time:.0f}s)")
            self.ctx.delay.sleep_range(walk_time * 0.9, walk_time * 1.1)

            # Idle while running over
            if self.ctx.idle and not self.should_stop:
                self.ctx.idle.maybe_idle()

            self._state = State.DEPOSITING

    def _do_depositing(self) -> None:
        deposited = self.deposit_all()
        self._logs_banked += deposited
        self._bank_trips += 1
        self._log(f"Deposited {deposited} logs (total banked: {self._logs_banked}, trip #{self._bank_trips})")

        self._find_failures = 0
        self._state = State.FIND_TREE
