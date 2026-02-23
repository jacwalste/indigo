"""
Vision System

Screen capture and color detection for OSRS Fixed Mode.
Uses mss for fast screen grabs and OpenCV for color matching.

GameRegions coordinates are game-relative (assume content area at 0,0).
Vision uses game_origin to offset these to absolute screen coordinates.
"""

import random
import sys
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple

import cv2
import mss
import threading
import numpy as np

# macOS standard title bar height (content starts below this)
TITLE_BAR_HEIGHT_MACOS = 28

# XFCE/Linux default title bar height (fallback if detection fails)
TITLE_BAR_HEIGHT_LINUX_DEFAULT = 30


def get_title_bar_height(on_log: Optional[Callable[[str], None]] = None) -> int:
    """Get the title bar height for the current platform.

    On macOS: returns the fixed 28px value.
    On Linux: attempts to detect via _NET_FRAME_EXTENTS on the RuneLite window
    using RuneLiteManager._detect_title_bar_height_linux(). Falls back to 30px.
    """
    if sys.platform == "darwin":
        return TITLE_BAR_HEIGHT_MACOS

    if sys.platform.startswith("linux"):
        try:
            from .managers.runelite import RuneLiteManager
            rl = RuneLiteManager(on_log=on_log)
            height = rl._detect_title_bar_height_linux()
            if on_log:
                on_log(f"[Vision] Detected Linux title bar height: {height}px")
            return height
        except Exception:
            pass
        return TITLE_BAR_HEIGHT_LINUX_DEFAULT

    # Unknown platform — use macOS default
    return TITLE_BAR_HEIGHT_MACOS


@dataclass
class Region:
    """A rectangular screen region."""
    x: int
    y: int
    width: int
    height: int

    def to_mss_monitor(self) -> dict:
        return {
            "left": self.x,
            "top": self.y,
            "width": self.width,
            "height": self.height,
        }

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class Color:
    """An RGB color."""
    r: int
    g: int
    b: int

    @classmethod
    def from_rgb(cls, r: int, g: int, b: int) -> "Color":
        return cls(r=r, g=g, b=b)

    @classmethod
    def from_hex(cls, hex_str: str) -> "Color":
        """Parse hex color string. Supports #RRGGBB or #AARRGGBB (alpha discarded)."""
        hex_str = hex_str.lstrip("#")
        if len(hex_str) == 8:
            hex_str = hex_str[2:]  # strip alpha prefix: AARRGGBB -> RRGGBB
        if len(hex_str) != 6:
            raise ValueError(f"Expected 6 or 8 hex digits, got {len(hex_str)}: {hex_str}")
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        return cls(r=r, g=g, b=b)

    def to_bgr(self) -> Tuple[int, int, int]:
        return (self.b, self.g, self.r)


@dataclass
class ColorCluster:
    """A cluster of pixels matching a color."""
    center: Tuple[int, int]
    click_point: Tuple[int, int]
    area: int
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h


class GameRegions:
    """Fixed-mode OSRS screen regions (765x503, window at 0,0)."""

    GAME_VIEW = Region(4, 4, 512, 334)
    MINIMAP = Region(548, 4, 210, 167)
    INVENTORY = Region(554, 206, 196, 261)
    CHATBOX = Region(4, 345, 512, 142)

    # Inventory grid: 4 columns x 7 rows
    # Slot size: 36x32px, gap: 6x4px, starts at (563, 213)
    INV_START_X = 563
    INV_START_Y = 213
    INV_SLOT_W = 36
    INV_SLOT_H = 32
    INV_GAP_X = 6
    INV_GAP_Y = 4
    INV_COLS = 4
    INV_ROWS = 7

    # Interface tabs (bottom of side panel, Fixed Classic layout)
    # Approximate positions — verify with `indigo test vision`
    TAB_COMBAT = Region(527, 168, 33, 36)
    TAB_STATS = Region(560, 168, 33, 36)
    TAB_QUESTS = Region(593, 168, 33, 36)
    TAB_INVENTORY = Region(626, 168, 33, 36)
    TAB_EQUIPMENT = Region(659, 168, 33, 36)
    TAB_PRAYER = Region(692, 168, 33, 36)
    TAB_MAGIC = Region(725, 168, 33, 36)

    # Browsable tabs (everything except inventory — used by idle browse behavior)
    BROWSE_TABS = ["TAB_COMBAT", "TAB_STATS", "TAB_QUESTS",
                   "TAB_EQUIPMENT", "TAB_PRAYER", "TAB_MAGIC"]

    # Stats panel grid: 3 columns x 8 rows (23 skills)
    # Each skill box is roughly 63x32, starting at ~(554, 210) in the side panel
    STATS_START_X = 554
    STATS_START_Y = 210
    STATS_SKILL_W = 63
    STATS_SKILL_H = 32
    STATS_COLS = 3
    STATS_ROWS = 8
    STATS_SKILL_COUNT = 23

    # Deposit box interface — calibrate with `indigo test coords`
    DEPOSIT_ALL_BUTTON = Region(124, 285, 34, 26)

    # Bank booth interface — deposit inventory button, calibrate with `indigo test coords`
    BANK_DEPOSIT_BUTTON = Region(425, 296, 33, 29)

    # Bank interface log slot — calibrated with `indigo test bankslot`
    BANK_LOG_SLOT = Region(410, 264, 29, 25)

    # Bank interface second slot (bowstrings) — calibrated with `indigo test bankslot`
    BANK_BOWSTRING_SLOT = Region(362, 265, 31, 22)

    # Bank interface close button (X) — for humanized bank closing
    BANK_CLOSE_BUTTON = Region(477, 14, 18, 19)

    @classmethod
    def get_inventory_slot(cls, index: int) -> Region:
        """Get the region for an inventory slot (0-27)."""
        col = index % cls.INV_COLS
        row = index // cls.INV_COLS
        x = cls.INV_START_X + col * (cls.INV_SLOT_W + cls.INV_GAP_X)
        y = cls.INV_START_Y + row * (cls.INV_SLOT_H + cls.INV_GAP_Y)
        return Region(x, y, cls.INV_SLOT_W, cls.INV_SLOT_H)

    @classmethod
    def get_all_inventory_slots(cls) -> List[Region]:
        """Get regions for all 28 inventory slots."""
        return [cls.get_inventory_slot(i) for i in range(28)]


class Vision:
    """Screen capture and color detection."""

    def __init__(
        self,
        game_origin: Tuple[int, int] = (0, 0),
        on_log: Optional[Callable[[str], None]] = None,
    ):
        self._game_origin = game_origin
        self._log_callback = on_log
        self._sct_local = threading.local()

    @property
    def _sct(self):
        """Get thread-local mss instance (X11 requires per-thread connections)."""
        if not hasattr(self._sct_local, 'instance'):
            self._sct_local.instance = mss.mss()
        return self._sct_local.instance

    def _log(self, message: str) -> None:
        if self._log_callback:
            self._log_callback(f"[Vision] {message}")
        else:
            print(f"[Vision] {message}")

    def to_screen(self, x: int, y: int) -> Tuple[int, int]:
        """Convert game-relative coords to absolute screen coords."""
        return (x + self._game_origin[0], y + self._game_origin[1])

    def _offset_region(self, region: Region) -> Region:
        """Offset a game-relative Region to absolute screen coords for mss capture."""
        return Region(
            x=region.x + self._game_origin[0],
            y=region.y + self._game_origin[1],
            width=region.width,
            height=region.height,
        )

    def grab(self, region: Region) -> np.ndarray:
        """Capture a screen region (game-relative), returns BGR numpy array."""
        screen_region = self._offset_region(region)
        screenshot = self._sct.grab(screen_region.to_mss_monitor())
        # mss returns BGRA, drop alpha channel
        frame = np.array(screenshot)[:, :, :3]
        return frame

    def find_color_clusters(
        self,
        region: Region,
        color: Color,
        tolerance: int = 10,
        min_area: int = 50,
    ) -> List[ColorCluster]:
        """Find clusters of a specific color in a region.

        Region is game-relative. Returned coords are absolute screen coords.
        """
        frame = self.grab(region)

        bgr = np.array(color.to_bgr(), dtype=np.uint8)
        lower = np.clip(bgr.astype(np.int16) - tolerance, 0, 255).astype(np.uint8)
        upper = np.clip(bgr.astype(np.int16) + tolerance, 0, 255).astype(np.uint8)

        mask = cv2.inRange(frame, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        clusters = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = x + w // 2
                cy = y + h // 2

            # Convert local contour coords to absolute screen coords
            screen_cx, screen_cy = self.to_screen(region.x + cx, region.y + cy)
            screen_x, screen_y = self.to_screen(region.x + x, region.y + y)

            clusters.append(ColorCluster(
                center=(screen_cx, screen_cy),
                click_point=(screen_cx, screen_cy),
                area=area,
                bounding_box=(screen_x, screen_y, w, h),
            ))

        # Sort by area descending (largest first)
        clusters.sort(key=lambda c: c.area, reverse=True)
        return clusters

    def slot_screen_center(self, slot_index: int) -> Tuple[int, int]:
        """Get absolute screen center of an inventory slot (exact)."""
        slot = GameRegions.get_inventory_slot(slot_index)
        return self.to_screen(*slot.center)

    def slot_screen_click_point(self, slot_index: int) -> Tuple[int, int]:
        """Get a jittered click point within an inventory slot.

        Gaussian spread around center, heavier toward the middle,
        clamped to slot bounds so we never miss.
        """
        slot = GameRegions.get_inventory_slot(slot_index)
        cx, cy = slot.center

        # Stddev ~1/4 of slot dimension: most clicks land in the inner half
        jitter_x = random.gauss(0, slot.width * 0.25)
        jitter_y = random.gauss(0, slot.height * 0.25)

        # Clamp to slot bounds (leave 1px margin)
        margin = 1
        x = int(max(slot.x + margin, min(slot.x + slot.width - margin, cx + jitter_x)))
        y = int(max(slot.y + margin, min(slot.y + slot.height - margin, cy + jitter_y)))

        return self.to_screen(x, y)

    def slot_has_item(self, slot_index: int) -> bool:
        """Check if an inventory slot has an item (brightness + variance check)."""
        slot = GameRegions.get_inventory_slot(slot_index)
        frame = self.grab(slot)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = float(np.mean(gray))
        std = float(np.std(gray))
        return mean > 40 and std > 15

    def count_inventory_items(self, skip_slots: Optional[List[int]] = None) -> int:
        """Count occupied inventory slots."""
        skip = set(skip_slots) if skip_slots else set()
        count = 0
        for i in range(28):
            if i in skip:
                continue
            if self.slot_has_item(i):
                count += 1
        return count

    def detect_hsv_pixels(
        self,
        region: Region,
        hue_low: int,
        hue_high: int,
        sat_min: int = 30,
        val_min: int = 30,
    ) -> int:
        """Count pixels matching an HSV hue range in a region.

        HSV matching is far more robust than BGR for anti-aliased text:
        hue is preserved even when pixels blend with the background.

        Args:
            region: Game-relative region to scan.
            hue_low: Minimum hue (OpenCV 0-180 scale).
            hue_high: Maximum hue (OpenCV 0-180 scale).
            sat_min: Minimum saturation (0-255).
            val_min: Minimum value/brightness (0-255).

        Returns:
            Number of matching pixels.
        """
        frame = self.grab(region)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([hue_low, sat_min, val_min])
        upper = np.array([hue_high, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        return int(cv2.countNonZero(mask))

    def template_match_region(
        self,
        region: Region,
        template_path: str,
        threshold: float = 0.8,
        padding: int = 40,
    ) -> bool:
        """Check if a template image appears near a screen region.

        Uses OpenCV normalized cross-correlation (TM_CCOEFF_NORMED).
        Grabs a padded area around the region so the template can slide
        and tolerate small coordinate offsets.

        Handles transparent PNGs by using the alpha channel as a mask.

        Args:
            region: Game-relative region indicating where to look.
            template_path: Path to template image file (PNG/JPG).
            threshold: Minimum match score (0.0-1.0). Default 0.8.
            padding: Extra pixels around the region to search. Default 40.

        Returns:
            True if the best match score >= threshold.
        """
        # Load template with alpha if present
        template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
        if template is None:
            return False

        mask = None
        if len(template.shape) == 3 and template.shape[2] == 4:
            mask = template[:, :, 3]
            template = template[:, :, :3]

        th, tw = template.shape[:2]

        # Grab a padded region larger than the template for sliding
        grab_region = Region(
            x=max(0, region.x - padding),
            y=max(0, region.y - padding),
            width=region.width + padding * 2,
            height=region.height + padding * 2,
        )
        frame = self.grab(grab_region)

        if template.shape[0] > frame.shape[0] or template.shape[1] > frame.shape[1]:
            return False

        if mask is not None:
            result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED, mask=mask)
        else:
            result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return max_val >= threshold

    @staticmethod
    def detect_game_origin(on_log: Optional[Callable[[str], None]] = None) -> Tuple[int, int]:
        """Position RuneLite window at (0,0) and detect game content origin.

        Returns (x, y) where game content starts (window_x, window_y + title_bar).
        """
        import time
        from .managers.runelite import RuneLiteManager

        def log(msg: str) -> None:
            if on_log:
                on_log(f"[Vision] {msg}")
            else:
                print(f"[Vision] {msg}")

        rl = RuneLiteManager(on_log=on_log)

        # Position at (0,0) and let window manager settle
        ok = rl._position_window()
        if not ok:
            log("WARNING: Window positioning may have failed")
        time.sleep(0.5)

        # Read actual position
        pos = rl.get_window_position()
        if pos is None:
            log("Could not read window position, assuming (0, 0)")
            pos = (0, 0)

        if sys.platform.startswith("linux"):
            # xdotool reports client area position (below title bar already)
            origin = (pos[0], pos[1])
            log(f"Window at {pos}, game_origin = {origin} (Linux: no title bar offset)")
        else:
            # macOS reports window frame position (includes title bar)
            title_bar = get_title_bar_height(on_log=on_log)
            origin = (pos[0], pos[1] + title_bar)
            log(f"Window at {pos}, title_bar={title_bar}px, game_origin = {origin}")
        return origin

    @staticmethod
    def verify_window_size(
        on_log: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """Verify RuneLite window matches Fixed Classic layout (~765x503 + title bar)."""
        from .managers.runelite import RuneLiteManager

        def log(msg: str) -> None:
            if on_log:
                on_log(f"[Vision] {msg}")
            else:
                print(f"[Vision] {msg}")

        rl = RuneLiteManager(on_log=on_log)
        size = rl.get_window_size()
        if size is None:
            log("Could not read window size")
            return False

        title_bar = get_title_bar_height(on_log=on_log)
        expected_w, expected_h = 765, 503 + title_bar
        w, h = size
        log(f"Window size: {w}x{h} (expected ~{expected_w}x{expected_h})")

        # Allow some tolerance (sidebar, borders)
        if abs(w - expected_w) > 20 or abs(h - expected_h) > 20:
            log(f"WARNING: Window size doesn't match Fixed Classic layout!")
            log(f"  Set RuneLite to Fixed Classic mode and resize if needed.")
            return False

        return True

    def close(self) -> None:
        """Cleanup mss."""
        if hasattr(self._sct_local, 'instance'):
            self._sct_local.instance.close()

    def get_status(self) -> dict:
        return {"active": True, "game_origin": self._game_origin}
