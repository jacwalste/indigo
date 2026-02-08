"""
Input System

Mouse movement and clicking via pynput, using WindMouse for human-like paths.
Click hold durations use Gaussian distributions with session-level personality.
"""

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
        """Move mouse to (x, y) using WindMouse path."""
        self._ensure_controllers()
        cx, cy = self.get_mouse_position()
        path = self._windmouse.generate(cx, cy, x, y)
        points = path.get_points_as_int_tuples()

        for px, py in points:
            self._mouse.position = (px, py)
            time.sleep(0.001)  # ~1ms between points for smooth movement

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

    def stop_session(self) -> None:
        """Cleanup."""
        self._log("Session stopped")

    def get_status(self) -> dict:
        return {
            "click_hold_mean": self._click_hold_mean,
            "click_hold_stddev": self._click_hold_stddev,
        }
