"""
Normal Trees Script (Lumbridge)

Chops normal trees at Lumbridge using NPC Indicators (cyan highlight).

Without --light:
    Drops logs via shift-click in column order. Axe is wielded, so all 28 slots are droppable.
    FIND_TREE -> CLICK_TREE -> WAITING -> DROPPING -> FIND_TREE

With --light:
    Tinderbox in slot 0, axe wielded. Light each log as it appears in slot 1.
    FIND_TREE -> CLICK_TREE -> WAITING -> LIGHTING -> FIND_TREE
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
    DROPPING = auto()
    LIGHTING = auto()


# Cyan from NPC Indicators plugin: #00FFFF
TREE_COLOR = Color(r=0, g=255, b=255)



class TreesScript(Script):
    """Chop normal trees and drop logs."""

    def __init__(self, ctx: ScriptContext, max_hours: float = 6.0,
                 light: bool = False,
                 on_log: Optional[Callable[[str], None]] = None):
        super().__init__(
            config=ScriptConfig(name="Trees", max_runtime_hours=max_hours),
            ctx=ctx,
            on_log=on_log,
        )
        self._light = light
        self._state = State.FIND_TREE
        self._last_tree: Optional[ColorCluster] = None
        self._drop_threshold = 20
        self._wait_checks = 0
        self._last_inv_count = 0
        self._last_gain_time = 0.0

    def on_start(self) -> None:
        mode = "light" if self._light else "drop"
        self._log(f"Chopping trees ({mode} mode) - ensure NPC Indicators is on (cyan)")
        if not self._light:
            self._drop_threshold = self.randomize_drop_threshold(mean=14, stddev=3, min_val=10, max_val=20)
        self._logs_chopped = 0
        self._drop_cycles = 0

    def loop(self) -> None:
        if self._state == State.FIND_TREE:
            tree = self.find_target(TREE_COLOR)
            if tree:
                self._last_tree = tree
                self._state = State.CLICK_TREE
                self._log(f"Found tree at {tree.click_point} (area={tree.area})")
            else:
                self._log("No tree found, waiting...")
                self.ctx.delay.sleep_range(1.5, 3.0)
                if self.ctx.idle:
                    self.ctx.idle.maybe_idle()

        elif self._state == State.CLICK_TREE:
            if self._last_tree:
                x, y = self._last_tree.click_point
                self.click_target(x, y)
                self.ctx.delay.sleep(NORMAL_ACTION)

                # Record inventory baseline
                self._last_inv_count = self.ctx.vision.count_inventory_items()
                self._last_gain_time = time.time()
                self._wait_checks = 0
                self._state = State.WAITING
                self._log(f"Clicked tree (inv={self._last_inv_count})")

        elif self._state == State.WAITING:
            # Poll every 2-4 seconds — just watch inventory
            self.ctx.delay.sleep_range(2.0, 4.0)
            self._wait_checks += 1

            # Maybe do something human-like
            if self.ctx.idle:
                self.ctx.idle.maybe_idle()

            if self._light:
                # Light mode: watch slot 1 for a log
                has_log = self.ctx.vision.slot_has_item(1)

                if self._wait_checks % 4 == 0:
                    self._log(
                        f"Chopping... slot1={'LOG' if has_log else 'empty'} "
                        f"lit={self._logs_chopped} "
                        f"time={self.elapsed_str()}"
                    )

                if has_log:
                    self._state = State.LIGHTING
                    return
            else:
                # Drop mode: watch inventory count
                inv_count = self.ctx.vision.count_inventory_items()

                if self._wait_checks % 4 == 0:
                    self._log(
                        f"Chopping... inv={inv_count}/{self._drop_threshold} "
                        f"chopped={self._logs_chopped} drops={self._drop_cycles} "
                        f"time={self.elapsed_str()}"
                    )

                # Got a log — drop if full, otherwise find next tree
                if inv_count > self._last_inv_count:
                    self._last_inv_count = inv_count
                    if inv_count >= self._drop_threshold:
                        self._log(f"Inventory has {inv_count} items, dropping")
                        self._state = State.DROPPING
                    else:
                        self._log(f"Got log (inv={inv_count}), finding next tree")
                        self._state = State.FIND_TREE
                    return

        elif self._state == State.LIGHTING:
            self._light_log()
            # Variable pause before next tree — sometimes quick, sometimes leisurely
            pause = self.ctx.rng.truncated_gauss(mean=1.5, stddev=1.0, min_val=0.3, max_val=4.0)
            self.ctx.delay.sleep_range(pause * 0.8, pause * 1.2)
            self._state = State.FIND_TREE

        elif self._state == State.DROPPING:
            dropped = self.drop_inventory(expected=self._last_inv_count)
            self._logs_chopped += dropped
            self._drop_cycles += 1
            self._log(f"Dropped {dropped} items (total chopped: {self._logs_chopped}, cycle #{self._drop_cycles})")
            self._drop_threshold = self.randomize_drop_threshold(mean=14, stddev=3, min_val=10, max_val=20)
            self._state = State.FIND_TREE

    def _light_log(self) -> None:
        """Click tinderbox (slot 0) then click log (slot 1) to light it."""
        from indigo.core.timing import FAST_ACTION

        # Click tinderbox
        tx, ty = self.ctx.vision.slot_screen_click_point(0)
        self.ctx.input.click(tx, ty)
        self.ctx.delay.sleep(FAST_ACTION)

        # Click log
        lx, ly = self.ctx.vision.slot_screen_click_point(1)
        self.ctx.input.click(lx, ly)
        self.ctx.delay.sleep(NORMAL_ACTION)

        self._logs_chopped += 1
        self._log(f"Lit log (total: {self._logs_chopped})")
