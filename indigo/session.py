"""
Session Manager

Orchestrates the session lifecycle:
- VPN connection (ensure Chicago)
- RuneLite launch
- Status updates
- Emergency stop handling
"""

import threading
from typing import Optional, Callable
from enum import Enum


class SessionState(Enum):
    IDLE = "idle"
    CONNECTING_VPN = "connecting_vpn"
    LAUNCHING_RUNELITE = "launching_runelite"
    WAITING_WINDOW = "waiting_window"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class SessionManager:
    """
    Manages the bot session lifecycle.

    Orchestrates VPN connection, RuneLite launch, and provides
    callbacks for status updates.
    """

    def __init__(
        self,
        on_log: Optional[Callable[[str], None]] = None,
        on_vpn_status: Optional[Callable[[str], None]] = None,
        on_runelite_status: Optional[Callable[[str], None]] = None,
        on_session_status: Optional[Callable[[str], None]] = None,
        on_state_change: Optional[Callable[[SessionState], None]] = None,
    ):
        self._on_log = on_log
        self._on_vpn_status = on_vpn_status
        self._on_runelite_status = on_runelite_status
        self._on_session_status = on_session_status
        self._on_state_change = on_state_change

        self._state = SessionState.IDLE
        self._stop_flag = threading.Event()
        self._emergency_stop_flag = threading.Event()
        self._current_thread: Optional[threading.Thread] = None

        # Managers (lazy-loaded)
        self._vpn = None
        self._runelite = None

    def _log(self, message: str) -> None:
        if self._on_log:
            self._on_log(message)
        else:
            print(f"[Session] {message}")

    def _set_state(self, state: SessionState) -> None:
        self._state = state
        if self._on_state_change:
            self._on_state_change(state)

    def _update_vpn_status(self, status: str) -> None:
        if self._on_vpn_status:
            self._on_vpn_status(status)

    def _update_runelite_status(self, status: str) -> None:
        if self._on_runelite_status:
            self._on_runelite_status(status)

    def _update_session_status(self, status: str) -> None:
        if self._on_session_status:
            self._on_session_status(status)

    def _get_vpn(self):
        if self._vpn is None:
            from .managers.vpn import MullvadVPN
            self._vpn = MullvadVPN(on_log=self._log)
        return self._vpn

    def _get_runelite(self):
        if self._runelite is None:
            from .managers.runelite import RuneLiteManager
            self._runelite = RuneLiteManager(on_log=self._log)
        return self._runelite

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state == SessionState.RUNNING

    def start_session(self, skip_vpn: bool = False) -> None:
        """
        Start a new session in a background thread.

        Flow: VPN -> RuneLite -> Wait for window -> Running
        """
        if self._state != SessionState.IDLE:
            self._log("[Session] Session already active")
            return

        self._stop_flag.clear()
        self._emergency_stop_flag.clear()

        def run():
            try:
                self._run_session(skip_vpn=skip_vpn)
            except Exception as e:
                self._log(f"[Session] Error: {e}")
                self._set_state(SessionState.ERROR)
                self._update_session_status("error")

        self._current_thread = threading.Thread(target=run, daemon=True)
        self._current_thread.start()

    def _run_session(self, skip_vpn: bool = False) -> None:
        # Step 1: VPN
        if not skip_vpn:
            self._set_state(SessionState.CONNECTING_VPN)
            self._update_vpn_status("connecting")
            self._update_session_status("connecting")

            vpn = self._get_vpn()

            if not vpn.is_available():
                self._log("[Session] Mullvad VPN not available")
                self._update_vpn_status("error")
                self._set_state(SessionState.ERROR)
                return

            success, msg = vpn.ensure_chicago()
            self._log(f"[VPN] {msg}")

            if not success:
                self._update_vpn_status("error")
                self._set_state(SessionState.ERROR)
                return

            self._update_vpn_status("connected")

            if self._stop_flag.is_set():
                self._log("[Session] Cancelled")
                self._set_state(SessionState.IDLE)
                return

        # Step 2: Launch RuneLite
        self._set_state(SessionState.LAUNCHING_RUNELITE)
        self._update_runelite_status("connecting")

        runelite = self._get_runelite()

        if not runelite.is_available():
            self._log("[Session] RuneLite not found")
            self._update_runelite_status("error")
            self._set_state(SessionState.ERROR)
            return

        if not runelite.launch(require_credentials=False):
            self._update_runelite_status("error")
            self._set_state(SessionState.ERROR)
            return

        if self._stop_flag.is_set():
            self._log("[Session] Cancelled")
            runelite.close()
            self._set_state(SessionState.IDLE)
            return

        # Step 3: Wait for window
        self._set_state(SessionState.WAITING_WINDOW)

        if not runelite.wait_for_window(timeout=60):
            self._log("[Session] Failed to detect RuneLite window")
            self._update_runelite_status("error")
            self._set_state(SessionState.ERROR)
            return

        self._update_runelite_status("connected")

        # Step 4: Running
        self._set_state(SessionState.RUNNING)
        self._update_session_status("running")
        self._log("[Session] Started successfully")

    def stop_session(self, disconnect_vpn: bool = False) -> None:
        if self._state == SessionState.IDLE:
            return

        self._set_state(SessionState.STOPPING)
        self._stop_flag.set()
        self._update_session_status("stopping")

        def run():
            try:
                self._stop_session_impl(disconnect_vpn)
            finally:
                self._set_state(SessionState.IDLE)
                self._update_session_status("stopped")
                self._update_runelite_status("disconnected")
                if disconnect_vpn:
                    self._update_vpn_status("disconnected")

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

    def _stop_session_impl(self, disconnect_vpn: bool) -> None:
        runelite = self._get_runelite()

        if runelite.is_running():
            self._log("[Session] Closing RuneLite...")
            runelite.close()

        if disconnect_vpn:
            vpn = self._get_vpn()
            if vpn.is_connected():
                self._log("[Session] Disconnecting VPN...")
                vpn.disconnect()

        self._log("[Session] Stopped")

    def emergency_stop(self) -> None:
        """Emergency stop - immediately terminate RuneLite."""
        self._emergency_stop_flag.set()
        self._stop_flag.set()
        self._log("[Session] EMERGENCY STOP")

        runelite = self._get_runelite()
        runelite.force_kill()

        self._set_state(SessionState.IDLE)
        self._update_session_status("stopped")
        self._update_runelite_status("disconnected")

        self._log("[Session] Emergency stop complete")

    def get_status(self) -> dict:
        vpn = self._get_vpn()
        runelite = self._get_runelite()
        return {
            "state": self._state.value,
            "vpn": vpn.get_status() if vpn else {},
            "runelite": runelite.get_status() if runelite else {},
        }

    def cleanup(self) -> None:
        self._stop_flag.set()
        runelite = self._get_runelite()
        if runelite and runelite.is_running():
            self._log("[Session] Cleaning up: closing RuneLite")
            runelite.close()
