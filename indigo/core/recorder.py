"""
Mouse Event Recorder

Records raw mouse events via pynput while the user manually plays.
Events are stored as game-relative coordinates for position-independent replay.
"""

import json
import os
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Tuple, Optional, Callable


@dataclass
class MouseEvent:
    x: int
    y: int
    t: float          # seconds since recording start
    kind: str          # "move", "press", "release"
    button: str = ""   # "left", "right", "middle" (empty for moves)


@dataclass
class RawRecording:
    script: str
    timestamp: str     # ISO 8601 start time
    duration: float
    game_origin: Tuple[int, int]
    events: List[MouseEvent] = field(default_factory=list)


RECORDINGS_DIR = os.path.expanduser("~/.indigo/recordings")
MIN_MOVE_INTERVAL = 0.005  # 5ms throttle between move events


class Recorder:
    """Records mouse events while the user manually plays an activity."""

    def __init__(
        self,
        script_name: str,
        game_origin: Tuple[int, int],
        on_log: Optional[Callable[[str], None]] = None,
    ):
        self._script = script_name
        self._game_origin = game_origin
        self._log_callback = on_log

        self._events: List[MouseEvent] = []
        self._recording = False
        self._start_time: float = 0.0
        self._last_move_time: float = 0.0
        self._listener = None
        self._stop_event = threading.Event()

    def _log(self, message: str) -> None:
        if self._log_callback:
            self._log_callback(f"[Recorder] {message}")
        else:
            print(f"[Recorder] {message}")

    def start(self) -> None:
        """Start recording mouse events."""
        from pynput.mouse import Listener as MouseListener

        self._events = []
        self._start_time = time.monotonic()
        self._last_move_time = 0.0
        self._recording = True
        self._stop_event.clear()

        gx, gy = self._game_origin

        def on_move(x, y):
            if not self._recording:
                return
            now = time.monotonic()
            t = now - self._start_time
            # Throttle move events
            if t - self._last_move_time < MIN_MOVE_INTERVAL:
                return
            self._last_move_time = t
            self._events.append(MouseEvent(
                x=int(x) - gx, y=int(y) - gy, t=t, kind="move",
            ))

        def on_click(x, y, button, pressed):
            if not self._recording:
                return
            t = time.monotonic() - self._start_time
            kind = "press" if pressed else "release"
            btn = button.name if hasattr(button, 'name') else str(button)
            self._events.append(MouseEvent(
                x=int(x) - gx, y=int(y) - gy, t=t, kind=kind, button=btn,
            ))

        self._listener = MouseListener(on_move=on_move, on_click=on_click)
        self._listener.start()
        self._log(f"Recording started ({len(self._events)} events)")

    def stop(self) -> RawRecording:
        """Stop recording and return the raw recording."""
        self._recording = False
        if self._listener:
            self._listener.stop()
            self._listener = None

        duration = time.monotonic() - self._start_time if self._start_time else 0.0
        self._log(f"Recording stopped: {len(self._events)} events, {duration:.1f}s")

        return RawRecording(
            script=self._script,
            timestamp=datetime.now().isoformat(),
            duration=duration,
            game_origin=self._game_origin,
            events=list(self._events),
        )

    def save(self, recording: RawRecording) -> str:
        """Save a raw recording to disk. Returns the file path."""
        script_dir = os.path.join(RECORDINGS_DIR, recording.script)
        os.makedirs(script_dir, exist_ok=True)

        # Timestamp-based filename (safe for filesystems)
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        path = os.path.join(script_dir, f"raw_{ts}.json")

        data = {
            "script": recording.script,
            "timestamp": recording.timestamp,
            "duration": recording.duration,
            "game_origin": list(recording.game_origin),
            "events": [
                {
                    "x": e.x, "y": e.y, "t": round(e.t, 4),
                    "kind": e.kind, "button": e.button,
                }
                for e in recording.events
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f)

        self._log(f"Saved {len(recording.events)} events to {path}")
        return path

    @staticmethod
    def load_raw(path: str) -> RawRecording:
        """Load a raw recording from disk."""
        with open(path, "r") as f:
            data = json.load(f)

        events = [
            MouseEvent(
                x=e["x"], y=e["y"], t=e["t"],
                kind=e["kind"], button=e.get("button", ""),
            )
            for e in data["events"]
        ]

        return RawRecording(
            script=data["script"],
            timestamp=data["timestamp"],
            duration=data["duration"],
            game_origin=tuple(data["game_origin"]),
            events=events,
        )

    @staticmethod
    def list_recordings(script_name: Optional[str] = None) -> dict:
        """List available recordings. Returns {script: [file_paths]}."""
        result = {}
        if not os.path.exists(RECORDINGS_DIR):
            return result

        scripts = [script_name] if script_name else os.listdir(RECORDINGS_DIR)
        for name in scripts:
            script_dir = os.path.join(RECORDINGS_DIR, name)
            if not os.path.isdir(script_dir):
                continue
            raw_files = sorted([
                os.path.join(script_dir, f)
                for f in os.listdir(script_dir)
                if f.startswith("raw_") and f.endswith(".json")
            ])
            if raw_files:
                result[name] = raw_files

        return result
