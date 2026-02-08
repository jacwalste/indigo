"""
Shrimp Fishing Script

Fishes shrimp at Lumbridge using NPC Indicators (cyan highlight).
Drops fish via shift-click in column order, keeping slot 0 (net).

State machine:
    FIND_SPOT -> CLICK_SPOT -> WAITING -> DROPPING -> FIND_SPOT
"""

from enum import Enum, auto
from typing import Optional, Callable

from indigo.script import Script, ScriptConfig, ScriptContext
from indigo.vision import Color, GameRegions, ColorCluster
from indigo.core.timing import FAST_ACTION, NORMAL_ACTION


class State(Enum):
    FIND_SPOT = auto()
    CLICK_SPOT = auto()
    WAITING = auto()
    DROPPING = auto()


# Cyan from NPC Indicators plugin: #00FFFF
FISHING_SPOT_COLOR = Color(r=0, g=255, b=255)

# Drop slots in column order (skip slot 0 = fishing net)
# Columns: [0,4,8,12,16,20,24], [1,5,9,13,17,21,25], [2,6,10,14,18,22,26], [3,7,11,15,19,23,27]
DROP_ORDER = []
for col in range(4):
    for row in range(7):
        slot = row * 4 + col
        if slot != 0:
            DROP_ORDER.append(slot)


class ShrimpScript(Script):
    """Fish shrimp and drop them."""

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

    def on_start(self) -> None:
        self._log("Fishing shrimp - ensure NPC Indicators is on (cyan)")
        self._randomize_drop_threshold()

    def _randomize_drop_threshold(self) -> None:
        """Randomize when to start dropping (anti-detection)."""
        self._drop_threshold = int(self.ctx.rng.truncated_gauss(
            mean=12, stddev=3, min_val=8, max_val=15,
        ))
        self._log(f"Drop threshold: {self._drop_threshold}")

    def _find_spot(self) -> Optional[ColorCluster]:
        """Find the largest cyan cluster in the game view."""
        clusters = self.ctx.vision.find_color_clusters(
            GameRegions.GAME_VIEW,
            FISHING_SPOT_COLOR,
            tolerance=15,
            min_area=40,
        )
        return clusters[0] if clusters else None

    def _spot_drifted(self, old: ColorCluster, new: ColorCluster) -> bool:
        """Check if spot moved significantly."""
        dx = abs(old.click_point[0] - new.click_point[0])
        dy = abs(old.click_point[1] - new.click_point[1])
        return (dx + dy) > 80

    def loop(self) -> None:
        if self._state == State.FIND_SPOT:
            spot = self._find_spot()
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
                self.ctx.input.click(x, y)
                self.ctx.delay.sleep(NORMAL_ACTION)
                self._wait_checks = 0
                self._state = State.WAITING
                self._log("Clicked fishing spot")

        elif self._state == State.WAITING:
            # Check every 1-3 seconds
            self.ctx.delay.sleep_range(1.0, 3.0)
            self._wait_checks += 1

            # Check inventory
            inv_count = self.ctx.vision.count_inventory_items(skip_slots=[0])
            if inv_count >= self._drop_threshold:
                self._log(f"Inventory has {inv_count} items, dropping")
                self._state = State.DROPPING
                return

            # Check if spot moved or disappeared
            spot = self._find_spot()
            if spot is None:
                self._log("Spot disappeared, re-finding")
                self._state = State.FIND_SPOT
                return
            if self._last_spot and self._spot_drifted(self._last_spot, spot):
                self._log("Spot moved, re-clicking")
                self._last_spot = spot
                self._state = State.CLICK_SPOT
                return

            # Update spot position
            self._last_spot = spot

        elif self._state == State.DROPPING:
            self._drop_inventory()
            self._randomize_drop_threshold()
            self._state = State.FIND_SPOT

    def _drop_inventory(self) -> None:
        """Shift-click drop all items in column order, skip slot 0."""
        dropped = 0
        for slot_idx in DROP_ORDER:
            if self.should_stop:
                break
            if not self.ctx.vision.slot_has_item(slot_idx):
                continue
            cx, cy = self.ctx.vision.slot_screen_center(slot_idx)
            self.ctx.input.shift_click(cx, cy)
            self.ctx.delay.sleep(FAST_ACTION, include_pauses=False)
            dropped += 1
        self._log(f"Dropped {dropped} items")
