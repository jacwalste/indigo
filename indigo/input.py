"""
Input System

Mouse movement and clicking via pynput, using WindMouse for human-like paths.
Click hold durations use Gaussian distributions with session-level personality.
"""

import math
import time
from typing import Optional, Callable, Tuple

from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController, Key

from .core.windmouse import WindMouse
from .core.delay import Delay
from .core.rng import RNG


class Input:
    """Mouse and keyboard input with human-like behavior."""

    def __init__(
        self,
        delay: Delay,
        windmouse: WindMouse,
        seed: Optional[int] = None,
        on_log: Optional[Callable[[str], None]] = None,
    ):
        self._delay = delay
        self._windmouse = windmouse
        self._rng = RNG(seed=seed)
        self._log_callback = on_log

        self._mouse: Optional[MouseController] = None
        self._keyboard: Optional[KeyboardController] = None

        # Click hold params (varied per session)
        self._click_hold_mean = 0.085
        self._click_hold_stddev = 0.025
        self._click_hold_min = 0.045
        self._click_hold_max = 0.160

    def _log(self, message: str) -> None:
        if self._log_callback:
            self._log_callback(f"[Input] {message}")
        else:
            print(f"[Input] {message}")

    def _ensure_controllers(self) -> None:
        if self._mouse is None:
            self._mouse = MouseController()
        if self._keyboard is None:
            self._keyboard = KeyboardController()

    def start_session(self) -> None:
        """Vary click hold parameters for session personality."""
        self._ensure_controllers()
        self._click_hold_mean = self._rng.vary_value(0.085, 0.15)
        self._click_hold_stddev = self._rng.vary_value(0.025, 0.15)
        self._click_hold_min = self._rng.vary_value(0.045, 0.1)
        self._click_hold_max = self._rng.vary_value(0.160, 0.1)
        self._log(
            f"Session started: hold_mean={self._click_hold_mean:.3f}s, "
            f"hold_std={self._click_hold_stddev:.3f}s"
        )

    def get_mouse_position(self) -> Tuple[int, int]:
        """Get current mouse position."""
        self._ensure_controllers()
        pos = self._mouse.position
        return (int(pos[0]), int(pos[1]))

    def move_to(self, x: int, y: int) -> None:
        """Move mouse to (x, y) using WindMouse path with variable speed."""
        self._ensure_controllers()
        cx, cy = self.get_mouse_position()
        path = self._windmouse.generate(cx, cy, x, y)
        points = path.get_points_as_int_tuples()

        n = len(points)
        for i, (px, py) in enumerate(points):
            self._mouse.position = (px, py)

            if i < n - 1:
                # Distance to next point
                nx, ny = points[i + 1]
                step_dist = math.sqrt((nx - px) ** 2 + (ny - py) ** 2)

                # Progress through path (0.0 = start, 1.0 = end)
                t = i / max(n - 1, 1)

                # Speed curve: slow start, fast middle, gentle decel at end
                # Bell-ish shape peaking around t=0.4
                speed_factor = 0.3 + 0.7 * math.sin(min(t * 1.3, 1.0) * math.pi)

                # Base delay scales with step distance (bigger step = more time)
                # but inversely with speed factor (faster = less delay)
                base_ms = 0.5 + step_dist * 0.3
                delay = (base_ms / max(speed_factor, 0.3)) / 1000.0

                # Random jitter +/- 20%
                jitter = self._rng.truncated_gauss(1.0, 0.1, 0.8, 1.2)
                delay *= jitter

                # Clamp to reasonable range
                delay = max(0.0005, min(delay, 0.008))

                time.sleep(delay)

    def _hold_duration(self) -> float:
        """Generate a Gaussian click hold duration."""
        return self._rng.truncated_gauss(
            mean=self._click_hold_mean,
            stddev=self._click_hold_stddev,
            min_val=self._click_hold_min,
            max_val=self._click_hold_max,
        )

    def click(self, x: int, y: int) -> None:
        """Move to position and click with human-like hold duration."""
        self._ensure_controllers()
        self.move_to(x, y)
        hold = self._hold_duration()
        self._mouse.press(Button.left)
        time.sleep(hold)
        self._mouse.release(Button.left)

    def shift_click(self, x: int, y: int) -> None:
        """Hold shift, click at position, release shift."""
        self._ensure_controllers()
        # Small pre-delay before pressing shift
        pre_delay = self._rng.truncated_gauss(0.02, 0.01, 0.01, 0.05)
        self._keyboard.press(Key.shift)
        time.sleep(pre_delay)

        self.click(x, y)

        # Small post-delay before releasing shift
        post_delay = self._rng.truncated_gauss(0.02, 0.01, 0.01, 0.05)
        time.sleep(post_delay)
        self._keyboard.release(Key.shift)

    def key_tap(self, key: str, hold: Optional[float] = None) -> None:
        """Press and release a key with human-like hold duration.

        Args:
            key: Single character (e.g. 'w', 'a') or Key attribute name.
            hold: Override hold duration. If None, uses Gaussian.
        """
        self._ensure_controllers()
        from pynput.keyboard import KeyCode
        k = KeyCode.from_char(key) if len(key) == 1 else getattr(Key, key)
        if hold is None:
            hold = self._rng.truncated_gauss(
                self._click_hold_mean, self._click_hold_stddev,
                self._click_hold_min, self._click_hold_max,
            )
        self._keyboard.press(k)
        time.sleep(hold)
        self._keyboard.release(k)

    def key_hold(self, key: str, duration: float) -> None:
        """Hold a key for a specific duration.

        Args:
            key: Single character (e.g. 'w', 'a').
            duration: How long to hold in seconds.
        """
        self._ensure_controllers()
        from pynput.keyboard import KeyCode
        k = KeyCode.from_char(key) if len(key) == 1 else getattr(Key, key)
        self._keyboard.press(k)
        time.sleep(duration)
        self._keyboard.release(k)

    def keys_hold(self, keys: list, duration: float) -> None:
        """Hold multiple keys simultaneously for a duration.

        Args:
            keys: List of key strings (e.g. ['w', 'a'] or ['left', 'up']).
            duration: How long to hold in seconds.
        """
        self._ensure_controllers()
        from pynput.keyboard import KeyCode
        resolved = []
        for key in keys:
            k = KeyCode.from_char(key) if len(key) == 1 else getattr(Key, key)
            resolved.append(k)

        for k in resolved:
            self._keyboard.press(k)
            # Tiny stagger between key presses — humans don't press simultaneously
            time.sleep(self._rng.truncated_gauss(0.02, 0.008, 0.008, 0.04))

        time.sleep(duration)

        for k in resolved:
            self._keyboard.release(k)
            time.sleep(self._rng.truncated_gauss(0.015, 0.006, 0.005, 0.03))

    def scroll(self, dx: int = 0, dy: int = 0) -> None:
        """Scroll the mouse wheel with human-like per-tick delays.

        Args:
            dx: Horizontal scroll ticks (not used in OSRS, included for completeness).
            dy: Vertical scroll ticks. Positive = scroll up (zoom in), negative = scroll down (zoom out).
        """
        self._ensure_controllers()
        total = abs(dy)
        direction = 1 if dy > 0 else -1
        for _ in range(total):
            self._mouse.scroll(dx, direction)
            tick_delay = self._rng.truncated_gauss(0.05, 0.015, 0.02, 0.08)
            time.sleep(tick_delay)

    def middle_drag(self, x: int, y: int, dx: int, dy: int) -> None:
        """Middle-click drag from (x, y) by (dx, dy). Used for camera rotation."""
        self._ensure_controllers()
        self.move_to(x, y)
        self._mouse.press(Button.middle)
        time.sleep(self._rng.truncated_gauss(0.05, 0.015, 0.03, 0.08))
        self.move_to(x + dx, y + dy)
        self._mouse.release(Button.middle)

    def stop_session(self) -> None:
        """Cleanup."""
        self._log("Session stopped")

    def get_status(self) -> dict:
        return {
            "click_hold_mean": self._click_hold_mean,
            "click_hold_stddev": self._click_hold_stddev,
        }
