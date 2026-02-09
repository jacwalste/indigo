"""
Shrimp Fishing Script

Fishes shrimp at Lumbridge using NPC Indicators (cyan highlight).
Drops fish via shift-click in column order, keeping slot 0 (net).

State machine:
    FIND_SPOT -> CLICK_SPOT -> WAITING -> DROPPING -> FIND_SPOT
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
    WAITING = auto()
    DROPPING = auto()


# Cyan from NPC Indicators plugin: #00FFFF
FISHING_SPOT_COLOR = Color(r=0, g=255, b=255)



class ShrimpScript(Script):
    """Fish shrimp and drop them."""

    # How long (seconds) with no new fish before we assume we stopped fishing
    IDLE_TIMEOUT_MIN = 45.0
    IDLE_TIMEOUT_MAX = 75.0

    def __init__(self, ctx: ScriptContext, max_hours: float = 6.0,
                 on_log: Optional[Callable[[str], None]] = None):
        super().__init__(
            config=ScriptConfig(name="Shrimp", max_runtime_hours=max_hours),
            ctx=ctx,
            on_log=on_log,
        )
        self._state = State.FIND_SPOT
        self._last_spot: Optional[ColorCluster] = None
        self._drop_threshold = 20
        self._wait_checks = 0
        self._last_inv_count = 0
        self._last_gain_time = 0.0
        self._idle_timeout = 60.0

    def on_start(self) -> None:
        self._log("Fishing shrimp - ensure NPC Indicators is on (cyan)")
        self._drop_threshold = self.randomize_drop_threshold(mean=12, stddev=3, min_val=8, max_val=15)
        self._fish_caught = 0
        self._drop_cycles = 0

    def loop(self) -> None:
        if self._state == State.FIND_SPOT:
            spot = self.find_target(FISHING_SPOT_COLOR)
            if spot:
                self._last_spot = spot
                self._state = State.CLICK_SPOT
                self._log(f"Found spot at {spot.click_point} (area={spot.area})")
            else:
                self._log("No fishing spot found, waiting...")
                self.ctx.delay.sleep_range(1.5, 3.0)

        elif self._state == State.CLICK_SPOT:
            if self._last_spot:
                x, y = self._last_spot.click_point
                self.click_target(x, y)
                self.ctx.delay.sleep(NORMAL_ACTION)

                # Record inventory baseline and start idle timer
                self._last_inv_count = self.ctx.vision.count_inventory_items(skip_slots=[0])
                self._last_gain_time = time.time()
                self._idle_timeout = self.ctx.rng.truncated_gauss(
                    mean=60.0, stddev=8.0,
                    min_val=self.IDLE_TIMEOUT_MIN, max_val=self.IDLE_TIMEOUT_MAX,
                )
                self._wait_checks = 0
                self._state = State.WAITING
                self._log(f"Clicked fishing spot (inv={self._last_inv_count})")

        elif self._state == State.WAITING:
            # Poll every 2-4 seconds — just watch inventory
            self.ctx.delay.sleep_range(2.0, 4.0)
            self._wait_checks += 1

            # Maybe do something human-like
            if self.ctx.idle:
                self.ctx.idle.maybe_idle()

            inv_count = self.ctx.vision.count_inventory_items(skip_slots=[0])

            # Did we gain fish?
            if inv_count > self._last_inv_count:
                self._last_inv_count = inv_count
                self._last_gain_time = time.time()

            idle_secs = time.time() - self._last_gain_time

            # Periodic status every ~4 checks (~10-15s)
            if self._wait_checks % 4 == 0:
                self._log(
                    f"Fishing... inv={inv_count}/{self._drop_threshold} "
                    f"idle={idle_secs:.0f}s/{self._idle_timeout:.0f}s "
                    f"caught={self._fish_caught} drops={self._drop_cycles} "
                    f"time={self.elapsed_str()}"
                )

            # Inventory full enough → drop
            if inv_count >= self._drop_threshold:
                self._log(f"Inventory has {inv_count} items, dropping")
                self._state = State.DROPPING
                return

            # No new fish for a while → stopped fishing, re-click
            if idle_secs >= self._idle_timeout:
                self._log(f"No fish for {idle_secs:.0f}s, re-clicking")
                self._state = State.FIND_SPOT
                return

        elif self._state == State.DROPPING:
            dropped = self.drop_inventory(skip_slots={0}, expected=self._last_inv_count)
            self._fish_caught += dropped
            self._drop_cycles += 1
            self._log(f"Dropped {dropped} items (total caught: {self._fish_caught}, cycle #{self._drop_cycles})")
            self._drop_threshold = self.randomize_drop_threshold(mean=12, stddev=3, min_val=8, max_val=15)
            self._state = State.FIND_SPOT
