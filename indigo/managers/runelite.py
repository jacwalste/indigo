"""
RuneLite Process Manager

Handles RuneLite application lifecycle:
- Launch with isolated home directory (no shared credentials)
- Position window at (0,0) using AppleScript
- Graceful close and emergency force-kill
"""

import os
import subprocess
import shutil
import time
from typing import Optional, Callable


class RuneLiteManager:
    """
    Manages RuneLite application lifecycle on macOS.

    Uses isolated home directory to separate bot credentials from main install.
    Credentials stored at ~/.indigo/runelite/.runelite/
    """

    DEFAULT_APP_LOCATIONS = [
        "/Applications/RuneLite.app",
        os.path.expanduser("~/Applications/RuneLite.app"),
    ]

    DEFAULT_BOT_HOME = os.path.expanduser("~/.indigo/runelite")

    def __init__(
        self,
        bot_home: Optional[str] = None,
        profile: str = "bot",
        window_x: int = 0,
        window_y: int = 0,
        on_log: Optional[Callable[[str], None]] = None
    ):
        self.bot_home = bot_home or self.DEFAULT_BOT_HOME
        self.profile = profile
        self.window_x = window_x
        self.window_y = window_y
        self._log_callback = on_log

        self.app_path = self._find_app()
        self.executable = self._find_executable()

    def _log(self, message: str) -> None:
        if self._log_callback:
            self._log_callback(f"[RuneLite] {message}")
        else:
            print(f"[RuneLite] {message}")

    def _find_app(self) -> Optional[str]:
        for path in self.DEFAULT_APP_LOCATIONS:
            if os.path.exists(path):
                return path
        return None

    def _find_executable(self) -> Optional[str]:
        if not self.app_path:
            return None
        exe = os.path.join(self.app_path, "Contents", "MacOS", "RuneLite")
        return exe if os.path.exists(exe) else None

    def is_available(self) -> bool:
        return self.executable is not None

    def is_running(self) -> bool:
        try:
            result = subprocess.run(
                ["pgrep", "-f", "RuneLite"],
                capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def has_credentials(self) -> bool:
        creds_path = os.path.join(self.bot_home, ".runelite", "credentials.properties")
        if not os.path.exists(creds_path):
            return False
        try:
            with open(creds_path, 'r') as f:
                content = f.read()
                return 'JX_DISPLAY_NAME=' in content and len(content) > 100
        except Exception:
            return False

    def get_account_name(self) -> Optional[str]:
        creds_path = os.path.join(self.bot_home, ".runelite", "credentials.properties")
        try:
            with open(creds_path, 'r') as f:
                for line in f:
                    if line.startswith('JX_DISPLAY_NAME='):
                        name = line.split('=', 1)[1].strip()
                        return name if name else None
        except Exception:
            pass
        return None

    def _ensure_bot_home(self) -> None:
        runelite_dir = os.path.join(self.bot_home, ".runelite")
        repo_dir = os.path.join(runelite_dir, "repository2")

        os.makedirs(runelite_dir, exist_ok=True)

        if not os.path.exists(repo_dir):
            main_repo = os.path.expanduser("~/.runelite/repository2")
            if os.path.exists(main_repo):
                self._log("Copying JAR cache from main installation...")
                shutil.copytree(main_repo, repo_dir)
                self._log(f"Copied {len(os.listdir(repo_dir))} JAR files")

    def launch(self, require_credentials: bool = True) -> bool:
        if not self.is_available():
            self._log("RuneLite not found")
            return False

        if self.is_running():
            self._log("RuneLite already running")
            return True

        if require_credentials and not self.has_credentials():
            self._log("No credentials found!")
            self._log(f"Expected: {self.bot_home}/.runelite/credentials.properties")
            self._log("Run credential setup first")
            return False

        self._ensure_bot_home()

        self._log(f"Launching RuneLite (profile: {self.profile})")
        if self.has_credentials():
            account = self.get_account_name() or "unknown"
            self._log(f"Credentials found for: {account}")

        try:
            cmd = [
                self.executable,
                "--launch-mode=FORK",
                f"-J-Duser.home={self.bot_home}",
                "-p", self.profile
            ]

            subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(3)

            if self.is_running():
                self._log("RuneLite launched")
                return True

            self._log("Launch may have failed")
            return False

        except Exception as e:
            self._log(f"Launch failed: {e}")
            return False

    def wait_for_window(self, timeout: int = 60) -> bool:
        self._log(f"Waiting for window (timeout: {timeout}s)")
        start = time.time()

        while time.time() - start < timeout:
            if self._window_exists():
                elapsed = time.time() - start
                self._log(f"Window detected after {elapsed:.1f}s")
                time.sleep(0.5)
                self._position_window()
                return True

            if time.time() - start > 10 and self.is_running():
                self._log("Window detection timed out, but process running")
                self._position_window()
                return True

            time.sleep(2)

        if self.is_running():
            self._log("Proceeding despite window detection failure")
            self._position_window()
            return True

        self._log("Window not detected")
        return False

    def _window_exists(self) -> bool:
        script = '''
        tell application "System Events"
            set runningApps to name of every process
            if runningApps contains "RuneLite" then
                tell process "RuneLite"
                    return (count of windows) > 0
                end tell
            else
                return false
            end if
        end tell
        '''
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip().lower() == "true"
        except Exception:
            return False

    def _position_window(self) -> bool:
        self._log(f"Positioning window at ({self.window_x}, {self.window_y})")
        script = f'''
        tell application "System Events"
            tell process "RuneLite"
                if (count of windows) > 0 then
                    set position of window 1 to {{{self.window_x}, {self.window_y}}}
                    return true
                else
                    return false
                end if
            end tell
        end tell
        '''
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=5
            )
            success = result.stdout.strip().lower() == "true"
            if success:
                self._log(f"Window positioned at ({self.window_x}, {self.window_y})")
            return success
        except Exception as e:
            self._log(f"Failed to position window: {e}")
            return False

    def close(self, timeout: int = 10) -> bool:
        if not self.is_running():
            self._log("RuneLite not running")
            return True

        self._log("Closing RuneLite...")
        try:
            subprocess.run(
                ["osascript", "-e", 'tell application "RuneLite" to quit'],
                capture_output=True, timeout=5
            )
        except Exception:
            pass

        start = time.time()
        while time.time() - start < timeout:
            if not self.is_running():
                self._log("RuneLite closed")
                return True
            time.sleep(0.5)

        self._log("Force killing...")
        try:
            subprocess.run(["pkill", "-f", "RuneLite"], capture_output=True, timeout=5)
            time.sleep(1)
            return not self.is_running()
        except Exception:
            return False

    def force_kill(self) -> bool:
        if not self.is_running():
            return True
        self._log("Force killing RuneLite (emergency stop)")
        try:
            subprocess.run(["pkill", "-9", "-f", "RuneLite"], capture_output=True, timeout=5)
            time.sleep(0.5)
            return not self.is_running()
        except Exception:
            return False

    def get_status(self) -> dict:
        return {
            "available": self.is_available(),
            "running": self.is_running(),
            "has_credentials": self.has_credentials(),
            "account": self.get_account_name(),
            "app_path": self.app_path,
            "bot_home": self.bot_home,
        }
