"""
Script Engine

Base class for bot scripts. Scripts implement on_start/loop/on_stop
and run in a background thread with a stop flag.
"""

import time
import threading
from dataclasses import dataclass
from typing import Optional, Callable

from .vision import Vision
from .input import Input
from .core.delay import Delay
from .core.rng import RNG


@dataclass
class ScriptConfig:
    """Configuration for a script."""
    name: str
    max_runtime_hours: float = 6.0


@dataclass
class ScriptContext:
    """Runtime context passed to scripts."""
    vision: Vision
    input: Input
    delay: Delay
    rng: RNG
    stop_flag: threading.Event


class Script:
    """Base class for bot scripts."""

    def __init__(
        self,
        config: ScriptConfig,
        ctx: ScriptContext,
        on_log: Optional[Callable[[str], None]] = None,
    ):
        self.config = config
        self.ctx = ctx
        self._log_callback = on_log
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._error: Optional[Exception] = None

    def _log(self, message: str) -> None:
        if self._log_callback:
            self._log_callback(f"[{self.config.name}] {message}")
        else:
            print(f"[{self.config.name}] {message}")

    @property
    def should_stop(self) -> bool:
        """Check stop flag and max runtime."""
        if self.ctx.stop_flag.is_set():
            return True
        if self._start_time is not None:
            elapsed_hours = (time.time() - self._start_time) / 3600.0
            if elapsed_hours >= self.config.max_runtime_hours:
                self._log(f"Max runtime reached ({self.config.max_runtime_hours}h)")
                return True
        return False

    def on_start(self) -> None:
        """Called once before the loop starts. Override in subclass."""
        pass

    def loop(self) -> None:
        """Called repeatedly. Override in subclass."""
        raise NotImplementedError("Subclasses must implement loop()")

    def on_stop(self) -> None:
        """Called once after the loop ends. Override in subclass."""
        pass

    def _run(self) -> None:
        """Internal run method for the background thread."""
        self._start_time = time.time()
        self._log("Starting")
        try:
            self.on_start()
            while not self.should_stop:
                self.loop()
        except Exception as e:
            self._log(f"Error: {e}")
            self._error = e
        finally:
            self.on_stop()
            elapsed = time.time() - self._start_time
            self._log(f"Stopped after {elapsed:.1f}s")

    def start(self) -> None:
        """Start the script in a background thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the script to stop."""
        self.ctx.stop_flag.set()

    def wait(self, timeout: Optional[float] = None) -> None:
        """Wait for the script thread to finish."""
        if self._thread:
            self._thread.join(timeout=timeout)
