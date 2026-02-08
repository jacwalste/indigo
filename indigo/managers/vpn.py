"""
Mullvad VPN Manager

Handles VPN lifecycle:
- Check connection status
- Connect/disconnect
- Verify relay location (us-chi)
"""

import subprocess
import time
import re
from typing import Optional, Callable, Tuple


class MullvadVPN:
    """
    Manages Mullvad VPN connection via CLI.

    Requires Mullvad CLI: brew install mullvad-vpn
    """

    EXPECTED_RELAY = "us-chi"

    def __init__(self, on_log: Optional[Callable[[str], None]] = None):
        self._log_callback = on_log
        self._original_ip: Optional[str] = None

    def _log(self, message: str) -> None:
        if self._log_callback:
            self._log_callback(f"[VPN] {message}")
        else:
            print(f"[VPN] {message}")

    def _run(self, *args: str, timeout: int = 10) -> Tuple[bool, str]:
        try:
            result = subprocess.run(
                ["mullvad", *args],
                capture_output=True, text=True, timeout=timeout
            )
            output = result.stdout.strip() or result.stderr.strip()
            return result.returncode == 0, output
        except FileNotFoundError:
            return False, "Mullvad CLI not found"
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    def is_available(self) -> bool:
        success, _ = self._run("version")
        return success

    def is_connected(self) -> bool:
        success, output = self._run("status")
        if not success:
            return False
        return "Connected" in output and "Disconnected" not in output

    def get_relay_location(self) -> Optional[str]:
        success, output = self._run("relay", "get")
        if not success:
            return None

        lines = output.lower()

        match = re.search(r'city\s+(\w+),?\s*(\w{2})\b', lines)
        if match:
            city, country = match.groups()
            return f"{country}-{city}"

        match = re.search(r'country\s+(\w+),?\s*city\s+(\w+)', lines)
        if match:
            country, city = match.groups()
            return f"{country}-{city}"

        match = re.search(r'\b([a-z]{2}-[a-z]{3,4})\b', lines)
        if match:
            return match.group(1)

        return None

    def is_chicago(self) -> bool:
        if not self.is_connected():
            return False
        relay = self.get_relay_location()
        return relay is not None and self.EXPECTED_RELAY in relay.lower()

    def get_ip(self) -> Optional[str]:
        try:
            import urllib.request
            with urllib.request.urlopen("https://api.ipify.org", timeout=10) as response:
                return response.read().decode('utf-8').strip()
        except Exception:
            return None

    def connect(self, timeout: int = 30) -> bool:
        if self._original_ip is None:
            self._original_ip = self.get_ip()

        if self.is_connected():
            self._log("Already connected")
            return True

        self._log("Connecting...")
        success, output = self._run("connect")

        if not success:
            self._log(f"Connect failed: {output}")
            return False

        start = time.time()
        while time.time() - start < timeout:
            if self.is_connected():
                new_ip = self.get_ip()
                self._log(f"Connected (IP: {new_ip})")
                return True
            time.sleep(1)

        self._log("Connection timed out")
        return False

    def disconnect(self) -> bool:
        if not self.is_connected():
            self._log("Already disconnected")
            return True

        self._log("Disconnecting...")
        success, _ = self._run("disconnect")

        time.sleep(2)
        if not self.is_connected():
            self._log("Disconnected")
            return True

        return False

    def set_relay(self, location: str) -> bool:
        self._log(f"Setting relay to {location}")

        if "-" in location:
            parts = location.split("-", 1)
        else:
            parts = location.split(None, 1)

        if len(parts) == 2:
            country, city = parts
            success, output = self._run("relay", "set", "location", country, city)
        else:
            success, output = self._run("relay", "set", "location", location)

        if success:
            self._log(f"Relay set to {location}")
        else:
            self._log(f"Failed to set relay: {output}")
        return success

    def ensure_chicago(self) -> Tuple[bool, str]:
        """Ensure VPN is connected to Chicago. Main entry point."""
        if not self.is_available():
            return False, "Mullvad CLI not found. Install: brew install mullvad-vpn"

        if self.is_chicago():
            ip = self.get_ip()
            return True, f"VPN connected to Chicago (IP: {ip})"

        if self.is_connected():
            current = self.get_relay_location()
            self._log(f"Connected to {current}, need Chicago")
            self.disconnect()
            time.sleep(1)

        if not self.set_relay(self.EXPECTED_RELAY):
            return False, "Failed to set relay to Chicago"

        if not self.connect():
            return False, "Failed to connect to VPN"

        if self.is_chicago():
            ip = self.get_ip()
            return True, f"VPN connected to Chicago (IP: {ip})"

        actual = self.get_relay_location()
        return False, f"VPN connected but not to Chicago (got: {actual})"

    def get_status(self) -> dict:
        connected = self.is_connected()
        return {
            "available": self.is_available(),
            "connected": connected,
            "relay": self.get_relay_location(),
            "is_chicago": self.is_chicago(),
            "ip": self.get_ip() if connected else None,
        }
