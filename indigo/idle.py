"""
Idle Behavior System

Human-like idle behaviors that fire during script downtime (fishing waits, etc.).

Uses an "attention burst" model: long stretches of doing nothing, interrupted by
short bursts of 2-4 actions (like a player glancing at their screen, checking a
few things, then zoning out again).
"""

import math
import time
from typing import Optional, Callable, List

from .script import ScriptContext
from .vision import GameRegions


# Individual actions available during attention bursts.
# Weights control how likely each action is to be picked within a burst.
_ACTIONS = [
    ("fidget_mouse", 15),
    ("mouse_wander", 15),
    ("camera_nudge", 20),
    ("check_stats", 12),
    ("browse_tabs", 10),
    ("hover_item", 12),
    ("camera_spin", 8),
    ("camera_circle", 8),
    ("zoom_adjust", 10),
]


class IdleBehavior:
    """Human-like idle behaviors using attention burst model.

    Most of the time: nothing. Occasionally: a burst of 2-4 actions in
    quick succession, then quiet again. Mimics a player glancing at their
    screen between doing something else.
    """

    def __init__(
        self,
        ctx: ScriptContext,
        on_log: Optional[Callable[[str], None]] = None,
    ):
        self._ctx = ctx
        self._log_callback = on_log

        # Session-varied state (set in start_session)
        self._burst_chance = 0.08
        self._weights: List[float] = [w for _, w in _ACTIONS]
        self._afk_min = 8.0
        self._afk_max = 30.0
        self._last_burst_time = 0.0
        self._min_burst_gap = 20.0  # minimum seconds between bursts

    def _log(self, message: str) -> None:
        if self._log_callback:
            self._log_callback(f"[Idle] {message}")
        else:
            print(f"[Idle] {message}")

    def _stopped(self) -> bool:
        return self._ctx.stop_flag.is_set()

    def _safe_sleep(self, seconds: float) -> None:
        """Sleep in small increments, checking stop flag."""
        end = time.time() + seconds
        while time.time() < end:
            if self._stopped():
                return
            time.sleep(min(0.25, end - time.time()))

    def start_session(self) -> None:
        """Vary parameters for session personality."""
        rng = self._ctx.rng
        self._burst_chance = rng.truncated_gauss(0.08, 0.02, 0.05, 0.12)
        self._weights = [rng.vary_value(w, 0.20) for _, w in _ACTIONS]
        self._afk_min = rng.vary_value(8.0, 0.15)
        self._afk_max = rng.vary_value(30.0, 0.15)
        self._min_burst_gap = rng.truncated_gauss(25.0, 5.0, 15.0, 40.0)
        self._last_burst_time = time.time()
        self._log(
            f"Session: burst_chance={self._burst_chance:.0%}, "
            f"min_gap={self._min_burst_gap:.0f}s"
        )

    def maybe_idle(self) -> bool:
        """Maybe trigger an attention burst. Returns True if a burst ran."""
        if self._stopped():
            return False

        # Enforce minimum gap between bursts
        since_last = time.time() - self._last_burst_time
        if since_last < self._min_burst_gap:
            return False

        if not self._ctx.rng.chance(self._burst_chance):
            return False

        self._do_burst()
        return True

    def _do_burst(self) -> None:
        """Run an attention burst: 2-4 actions chained together."""
        rng = self._ctx.rng

        action_count = int(rng.truncated_gauss(2.5, 0.8, 2, 4))
        self._log(f"Attention burst ({action_count} actions)")

        names = [n for n, _ in _ACTIONS]
        last_action = None

        for i in range(action_count):
            if self._stopped():
                break

            # Pick an action (avoid repeating the same one twice in a row)
            for _ in range(5):
                name = rng.weighted_choice(names, self._weights)
                if name != last_action:
                    break
            last_action = name

            method = getattr(self, f"_{name}")
            method()

            # Brief pause between actions within a burst (0.3-1.5s)
            if i < action_count - 1 and not self._stopped():
                gap = rng.truncated_gauss(0.8, 0.3, 0.3, 1.5)
                self._safe_sleep(gap)

        self._last_burst_time = time.time()

    # --- Individual Actions ---

    def _fidget_mouse(self) -> None:
        """Move mouse 10-50px in a random direction, pause, drift back."""
        if self._stopped():
            return
        inp = self._ctx.input
        rng = self._ctx.rng

        ox, oy = inp.get_mouse_position()

        angle = rng.uniform(0, 2 * math.pi)
        dist = rng.truncated_gauss(30.0, 10.0, 10.0, 50.0)
        dx = int(dist * math.cos(angle))
        dy = int(dist * math.sin(angle))

        inp.move_to(ox + dx, oy + dy)
        self._safe_sleep(rng.truncated_gauss(0.5, 0.2, 0.2, 0.8))

        if self._stopped():
            return

        jx = int(rng.truncated_gauss(0, 2, -3, 3))
        jy = int(rng.truncated_gauss(0, 2, -3, 3))
        inp.move_to(ox + jx, oy + jy)

        self._log(f"Fidget ({int(dist)}px)")

    def _mouse_wander(self) -> None:
        """Move to random spot in game view, pause, return."""
        if self._stopped():
            return
        inp = self._ctx.input
        rng = self._ctx.rng
        gv = GameRegions.GAME_VIEW

        ox, oy = inp.get_mouse_position()

        tx = int(rng.uniform(gv.x, gv.x + gv.width))
        ty = int(rng.uniform(gv.y, gv.y + gv.height))

        sx, sy = self._ctx.vision.to_screen(tx, ty)
        inp.move_to(sx, sy)
        self._safe_sleep(rng.truncated_gauss(1.2, 0.4, 0.5, 2.0))

        if self._stopped():
            return

        jx = int(rng.truncated_gauss(0, 3, -5, 5))
        jy = int(rng.truncated_gauss(0, 3, -5, 5))
        inp.move_to(ox + jx, oy + jy)

        self._log("Mouse wander")

    def _camera_nudge(self) -> None:
        """Tap camera keys — single or diagonal combo."""
        if self._stopped():
            return
        rng = self._ctx.rng
        inp = self._ctx.input

        hold = rng.truncated_gauss(0.15, 0.08, 0.05, 0.4)

        if rng.chance(0.30):
            keys = rng.choice([
                ['left', 'up'], ['left', 'down'],
                ['right', 'up'], ['right', 'down'],
            ])
            inp.keys_hold(keys, hold)
            label = '+'.join(keys)
        elif rng.chance(0.75):
            key = rng.choice(['w', 'a', 's', 'd'])
            inp.key_hold(key, hold)
            label = key
        else:
            key = rng.choice(['left', 'right', 'up', 'down'])
            inp.key_hold(key, hold)
            label = key

        self._safe_sleep(rng.truncated_gauss(0.3, 0.1, 0.15, 0.5))

        self._log(f"Camera nudge ({label})")

    def _click_tab(self, tab_region) -> None:
        """Click a tab region with jitter."""
        rng = self._ctx.rng
        tx, ty = self._ctx.vision.to_screen(*tab_region.center)
        tx += int(rng.truncated_gauss(0, 4, -8, 8))
        ty += int(rng.truncated_gauss(0, 4, -8, 8))
        self._ctx.input.click(tx, ty)

    def _return_to_inventory(self) -> None:
        """Click inventory tab to return."""
        self._click_tab(GameRegions.TAB_INVENTORY)
        self._safe_sleep(self._ctx.rng.truncated_gauss(0.3, 0.1, 0.15, 0.5))

    def _check_stats(self) -> None:
        """Open stats tab, hover/click some skills, return to inventory."""
        if self._stopped():
            return
        rng = self._ctx.rng
        inp = self._ctx.input
        vision = self._ctx.vision

        self._click_tab(GameRegions.TAB_STATS)
        self._safe_sleep(rng.truncated_gauss(0.8, 0.2, 0.4, 1.2))

        if self._stopped():
            self._return_to_inventory()
            return

        hover_count = int(rng.truncated_gauss(1.5, 0.8, 1, 3))
        for _ in range(hover_count):
            if self._stopped():
                break
            skill_idx = int(rng.uniform(0, GameRegions.STATS_SKILL_COUNT))
            col = skill_idx % GameRegions.STATS_COLS
            row = skill_idx // GameRegions.STATS_COLS
            skill_x = GameRegions.STATS_START_X + col * GameRegions.STATS_SKILL_W + GameRegions.STATS_SKILL_W // 2
            skill_y = GameRegions.STATS_START_Y + row * GameRegions.STATS_SKILL_H + GameRegions.STATS_SKILL_H // 2
            sx, sy = vision.to_screen(skill_x, skill_y)
            sx += int(rng.truncated_gauss(0, 6, -12, 12))
            sy += int(rng.truncated_gauss(0, 4, -8, 8))
            inp.move_to(sx, sy)
            self._safe_sleep(rng.truncated_gauss(0.8, 0.3, 0.4, 1.5))

            if rng.chance(0.20) and not self._stopped():
                inp.click(sx, sy)
                self._safe_sleep(rng.truncated_gauss(1.2, 0.4, 0.6, 2.0))

        self._return_to_inventory()
        self._log(f"Check stats (hovered {hover_count} skills)")

    def _browse_tabs(self) -> None:
        """Open 1-2 random tabs, look around briefly, return to inventory."""
        if self._stopped():
            return
        rng = self._ctx.rng
        inp = self._ctx.input

        tab_count = 1 if rng.chance(0.6) else 2
        available = list(GameRegions.BROWSE_TABS)
        tab_names = []
        for _ in range(tab_count):
            pick = rng.choice(available)
            tab_names.append(pick)
            available.remove(pick)

        for tab_name in tab_names:
            if self._stopped():
                break

            tab_region = getattr(GameRegions, tab_name)
            self._click_tab(tab_region)
            self._safe_sleep(rng.truncated_gauss(1.2, 0.4, 0.6, 2.0))

            if rng.chance(0.5) and not self._stopped():
                px = int(rng.uniform(554, 750))
                py = int(rng.uniform(210, 460))
                sx, sy = self._ctx.vision.to_screen(px, py)
                inp.move_to(sx, sy)
                self._safe_sleep(rng.truncated_gauss(0.8, 0.3, 0.4, 1.5))

        self._return_to_inventory()
        self._log(f"Browse tabs ({', '.join(t.replace('TAB_', '').lower() for t in tab_names)})")

    def _hover_item(self) -> None:
        """Hover over a random occupied inventory slot briefly."""
        if self._stopped():
            return
        rng = self._ctx.rng
        inp = self._ctx.input
        vision = self._ctx.vision

        occupied = [i for i in range(28) if vision.slot_has_item(i)]
        if not occupied:
            return

        slot = rng.choice(occupied)
        cx, cy = vision.slot_screen_click_point(slot)
        inp.move_to(cx, cy)
        self._safe_sleep(rng.truncated_gauss(0.6, 0.2, 0.3, 1.0))

        if self._stopped():
            return

        inp.move_to(cx + int(rng.uniform(-15, 15)), cy + int(rng.uniform(-15, 15)))

        self._log(f"Hover item (slot {slot})")

    def _camera_spin(self) -> None:
        """Hold a random arrow key to spin the camera for a bit."""
        if self._stopped():
            return
        rng = self._ctx.rng
        inp = self._ctx.input

        key = rng.choice(['left', 'right'])
        duration = rng.truncated_gauss(1.2, 0.4, 0.5, 2.0)
        inp.key_hold(key, duration)

        self._safe_sleep(rng.truncated_gauss(0.3, 0.1, 0.15, 0.5))

        self._log(f"Camera spin ({key}, {duration:.1f}s)")

    def _camera_circle(self) -> None:
        """Hold two keys at once for diagonal camera movement, cycle through combos."""
        if self._stopped():
            return
        rng = self._ctx.rng
        inp = self._ctx.input

        combos = [
            ['left', 'up'], ['up', 'right'], ['right', 'down'], ['down', 'left'],
        ]

        start = int(rng.uniform(0, 4))
        reverse = rng.chance(0.5)
        steps = int(rng.truncated_gauss(3, 1, 2, 5))

        for i in range(steps):
            if self._stopped():
                break
            idx = (start + (i if not reverse else -i)) % 4
            keys = combos[idx]
            hold = rng.truncated_gauss(0.35, 0.12, 0.15, 0.6)
            inp.keys_hold(keys, hold)
            self._safe_sleep(rng.truncated_gauss(0.08, 0.03, 0.03, 0.15))

        self._safe_sleep(rng.truncated_gauss(0.3, 0.1, 0.15, 0.5))
        self._log(f"Camera circle ({steps} steps)")

    def _zoom_adjust(self) -> None:
        """Scroll to adjust zoom level randomly."""
        if self._stopped():
            return
        rng = self._ctx.rng
        inp = self._ctx.input

        ticks = int(rng.truncated_gauss(3.5, 1.5, 1, 6))
        direction = rng.choice([1, -1])
        inp.scroll(dy=ticks * direction)

        self._safe_sleep(rng.truncated_gauss(0.3, 0.1, 0.15, 0.5))

        label = "in" if direction > 0 else "out"
        self._log(f"Zoom adjust ({label}, {ticks} ticks)")
