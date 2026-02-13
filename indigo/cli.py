#!/usr/bin/env python3
"""
Indigo CLI

Usage:
    indigo launch [--skip-vpn]    Connect VPN + launch RuneLite
    indigo stop [--disconnect]    Stop RuneLite (optionally disconnect VPN)
    indigo kill                   Emergency stop (force kill)
    indigo status                 Show VPN and RuneLite status
    indigo test delays            Test delay system (histogram)
    indigo test windmouse         Test WindMouse paths
    indigo test fatigue           Preview fatigue curves
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
import threading
from typing import Callable


def _log(message: str) -> None:
    print(message)


def _wait_for_hotkey(message: str = "Press ` (backtick) to start...") -> None:
    """Block until backtick is pressed."""
    from pynput.keyboard import Listener, KeyCode

    print(message)
    pressed = threading.Event()

    def on_press(key):
        if key == KeyCode.from_char('`'):
            pressed.set()
            return False  # stop listener

    with Listener(on_press=on_press):
        pressed.wait()


def _hotkey_stop_listener(stop_callback: Callable[[], None]) -> threading.Thread:
    """Start a background listener that calls stop_callback on backtick."""
    from pynput.keyboard import Listener, KeyCode

    def on_press(key):
        if key == KeyCode.from_char('`'):
            stop_callback()
            return False

    listener = Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    return listener


def cmd_launch(args):
    """Connect VPN and launch RuneLite."""
    from .session import SessionManager, SessionState

    session = SessionManager(on_log=_log)
    done = threading.Event()
    final_state = [None]

    def on_state_change(state):
        final_state[0] = state
        if state in (SessionState.RUNNING, SessionState.ERROR):
            done.set()

    session._on_state_change = on_state_change
    session.start_session(skip_vpn=args.skip_vpn)

    # Wait for session to reach terminal state
    done.wait(timeout=120)

    if final_state[0] == SessionState.RUNNING:
        print("\nSession running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")
            session.stop_session()
            time.sleep(2)
    elif final_state[0] == SessionState.ERROR:
        print("\nSession failed to start.")
        sys.exit(1)
    else:
        print("\nSession timed out.")
        sys.exit(1)


def cmd_stop(args):
    """Stop RuneLite."""
    from .managers.runelite import RuneLiteManager
    from .managers.vpn import MullvadVPN

    rl = RuneLiteManager(on_log=_log)
    if rl.is_running():
        rl.close()
    else:
        print("[RuneLite] Not running")

    if args.disconnect:
        vpn = MullvadVPN(on_log=_log)
        if vpn.is_connected():
            vpn.disconnect()
        else:
            print("[VPN] Not connected")


def cmd_kill(args):
    """Emergency stop."""
    from .managers.runelite import RuneLiteManager

    rl = RuneLiteManager(on_log=_log)
    if rl.is_running():
        rl.force_kill()
        print("[RuneLite] Force killed")
    else:
        print("[RuneLite] Not running")


def cmd_status(args):
    """Show status of VPN and RuneLite."""
    from .managers.vpn import MullvadVPN
    from .managers.runelite import RuneLiteManager

    vpn = MullvadVPN()
    rl = RuneLiteManager()

    vpn_status = vpn.get_status()
    rl_status = rl.get_status()

    print("=== Indigo Status ===\n")

    print("VPN:")
    print(f"  Available:  {vpn_status['available']}")
    print(f"  Connected:  {vpn_status['connected']}")
    if vpn_status['connected']:
        print(f"  Relay:      {vpn_status['relay']}")
        print(f"  Chicago:    {vpn_status['is_chicago']}")
        print(f"  IP:         {vpn_status['ip']}")

    print("\nRuneLite:")
    print(f"  Available:  {rl_status['available']}")
    print(f"  Running:    {rl_status['running']}")
    print(f"  Credentials:{' ' + rl_status['account'] if rl_status['account'] else ' None'}")
    print(f"  Bot Home:   {rl_status['bot_home']}")


def cmd_test_delays(args):
    """Test delay system."""
    from .core.delay import Delay, NORMAL_ACTION, FAST_ACTION

    print("=== Delay System Test ===\n")
    delay = Delay(seed=42)
    print(delay.test_harness(count=500, profile=NORMAL_ACTION))

    print("\n\n=== Fast Action Profile ===\n")
    print(delay.test_harness(count=500, profile=FAST_ACTION))


def cmd_test_windmouse(args):
    """Test WindMouse paths."""
    from .core.windmouse import WindMouse

    print("=== WindMouse Test ===\n")
    wind = WindMouse(seed=42)
    wind.start_session()
    print(wind.test_harness(count=200))


def cmd_test_vision(args):
    """Test vision system."""
    from .vision import Vision, Color, GameRegions

    print("=== Vision System Test ===\n")

    # Detect game origin
    game_origin = Vision.detect_game_origin(on_log=_log)
    Vision.verify_window_size(on_log=_log)

    vision = Vision(game_origin=game_origin, on_log=_log)

    # Test game view capture
    print("\nCapturing game view...")
    frame = vision.grab(GameRegions.GAME_VIEW)
    print(f"  Frame shape: {frame.shape}")
    print(f"  Game origin: {game_origin}")

    # Find cyan clusters (NPC Indicators)
    cyan = Color(r=0, g=255, b=255)
    clusters = vision.find_color_clusters(GameRegions.GAME_VIEW, cyan, tolerance=15, min_area=40)
    print(f"\nCyan clusters found: {len(clusters)}")
    for i, c in enumerate(clusters[:5]):
        print(f"  #{i}: center={c.center}, area={c.area}, bbox={c.bounding_box}")

    # Test inventory grid
    print("\nInventory grid:")
    occupied = 0
    grid = ""
    for row in range(7):
        row_str = "  "
        for col in range(4):
            idx = row * 4 + col
            has_item = vision.slot_has_item(idx)
            if has_item:
                occupied += 1
            row_str += "[X] " if has_item else "[ ] "
        grid += row_str + "\n"
    print(grid)
    print(f"  Occupied: {occupied}/28")

    # Show slot screen centers for verification
    print("Inventory slot screen centers (first row):")
    for i in range(4):
        print(f"  Slot {i}: {vision.slot_screen_center(i)}")

    vision.close()


def cmd_test_inventory(args):
    """Test inventory detection - shows per-slot stats for calibration."""
    import os
    from .vision import Vision, GameRegions

    print("=== Inventory Calibration Test ===\n")

    game_origin = Vision.detect_game_origin(on_log=_log)
    Vision.verify_window_size(on_log=_log)
    vision = Vision(game_origin=game_origin, on_log=_log)

    # Save full inventory region screenshot
    import cv2
    inv_frame = vision.grab(GameRegions.INVENTORY)
    save_dir = os.path.expanduser("~/.indigo/debug")
    os.makedirs(save_dir, exist_ok=True)
    inv_path = os.path.join(save_dir, "inventory.png")
    cv2.imwrite(inv_path, inv_frame)
    print(f"Inventory screenshot saved: {inv_path}")
    print(f"  Region: ({GameRegions.INVENTORY.x}, {GameRegions.INVENTORY.y}) "
          f"{GameRegions.INVENTORY.width}x{GameRegions.INVENTORY.height}")
    print(f"  Frame shape: {inv_frame.shape}")

    # Per-slot data
    print("\n--- Per-Slot Stats ---\n")
    print(f"  {'Slot':>4}  {'Mean':>6}  {'Std':>6}  {'Item?':>5}  {'Screen Center':>16}  {'Grab Region (game-rel)':>28}")
    print(f"  {'----':>4}  {'----':>6}  {'----':>6}  {'-----':>5}  {'-------------':>16}  {'----------------------':>28}")

    import numpy as np
    slot_stats = []
    for i in range(28):
        slot = GameRegions.get_inventory_slot(i)
        frame = vision.grab(slot)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = float(np.mean(gray))
        std = float(np.std(gray))
        has_item = mean > 40 and std > 15
        screen_center = vision.slot_screen_center(i)
        slot_stats.append((i, mean, std, has_item, screen_center, slot))

        print(f"  {i:4d}  {mean:6.1f}  {std:6.1f}  {'  [X]' if has_item else '  [ ]'}  "
              f"({screen_center[0]:4d},{screen_center[1]:4d})  "
              f"({slot.x:3d},{slot.y:3d}) {slot.width}x{slot.height}")

        # Save each slot image
        slot_path = os.path.join(save_dir, f"slot_{i:02d}.png")
        cv2.imwrite(slot_path, frame)

    print(f"\n  Slot images saved to: {save_dir}/slot_XX.png")

    # Visual grid
    print("\n--- Grid View ---\n")
    occupied = 0
    for row in range(7):
        row_means = "  "
        row_items = "  "
        for col in range(4):
            idx = row * 4 + col
            _, mean, std, has_item, _, _ = slot_stats[idx]
            if has_item:
                occupied += 1
            row_items += "[X] " if has_item else "[ ] "
            row_means += f"{mean:4.0f} "
        print(f"  Row {row}: {row_items}   means: {row_means}")
    print(f"\n  Detected: {occupied}/28 occupied")

    # Summary + hints
    print("\n--- Calibration Hints ---\n")
    empty_means = [s[1] for s in slot_stats if not s[3]]
    full_means = [s[1] for s in slot_stats if s[3]]
    empty_stds = [s[2] for s in slot_stats if not s[3]]
    full_stds = [s[2] for s in slot_stats if s[3]]

    if empty_means:
        print(f"  Empty slots:  mean={min(empty_means):.1f}-{max(empty_means):.1f}, "
              f"std={min(empty_stds):.1f}-{max(empty_stds):.1f}  (n={len(empty_means)})")
    if full_means:
        print(f"  Full slots:   mean={min(full_means):.1f}-{max(full_means):.1f}, "
              f"std={min(full_stds):.1f}-{max(full_stds):.1f}  (n={len(full_means)})")
    if empty_means and full_means:
        gap = min(full_means) - max(empty_means)
        print(f"  Mean gap:     {gap:.1f} (positive = good separation)")
    print(f"\n  Current thresholds: mean > 40, std > 15")
    print(f"  Check slot images in {save_dir}/ to verify grid alignment")

    vision.close()


def cmd_test_drop(args):
    """Test inventory dropping - shift-click drops occupied slots."""
    from .vision import Vision, GameRegions
    from .input import Input
    from .core.delay import Delay
    from .core.windmouse import WindMouse
    from .core.rng import RNG
    from .core.timing import FAST_ACTION

    print("=== Drop Test ===\n")

    game_origin = Vision.detect_game_origin(on_log=_log)
    vision = Vision(game_origin=game_origin, on_log=_log)

    # Parse skip slots
    skip = set()
    if hasattr(args, "skip") and args.skip:
        for part in args.skip.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                skip.update(range(int(lo), int(hi) + 1))
            else:
                skip.add(int(part))

    # Scan inventory before
    print("Scanning inventory...\n")
    occupied = []
    for i in range(28):
        if vision.slot_has_item(i):
            occupied.append(i)

    to_drop = [s for s in occupied if s not in skip]

    # Show grid
    for row in range(7):
        row_str = "  "
        for col in range(4):
            idx = row * 4 + col
            if idx in skip:
                row_str += "[S] "  # skip
            elif idx in occupied:
                row_str += "[X] "  # will drop
            else:
                row_str += "[ ] "
        print(row_str)

    print(f"\n  Occupied: {len(occupied)}  Skipping: {skip or 'none'}  Will drop: {len(to_drop)}")

    if not to_drop:
        print("\n  Nothing to drop.")
        vision.close()
        return

    # Set up input
    rng = RNG()
    delay = Delay(seed=rng.seed, on_log=_log)
    windmouse = WindMouse(seed=rng.seed + 1, on_log=_log)
    inp = Input(delay=delay, windmouse=windmouse, seed=rng.seed + 2, on_log=_log)
    delay.start_session()
    windmouse.start_session()
    inp.start_session()

    # Wait for F12
    _wait_for_hotkey(f"\nPress ` to drop {len(to_drop)} items...")
    print()

    # Drop in column order
    dropped = 0
    for col in range(4):
        for row in range(7):
            idx = row * 4 + col
            if idx not in to_drop:
                continue
            if not vision.slot_has_item(idx):
                continue
            cx, cy = vision.slot_screen_click_point(idx)
            _log(f"Dropping slot {idx} at ({cx}, {cy})")
            inp.shift_click(cx, cy)
            delay.sleep(FAST_ACTION, include_pauses=False)
            dropped += 1

    print(f"\nDropped {dropped} items.")

    # Scan after
    print("\nInventory after:")
    remaining = 0
    for row in range(7):
        row_str = "  "
        for col in range(4):
            idx = row * 4 + col
            has = vision.slot_has_item(idx)
            if has:
                remaining += 1
            row_str += "[X] " if has else "[ ] "
        print(row_str)
    print(f"\n  Remaining: {remaining}/28")

    # Cleanup
    inp.stop_session()
    windmouse.stop_session()
    delay.stop_session()
    vision.close()


def cmd_run_shrimp(args):
    """Run shrimp fishing script."""
    from .vision import Vision
    from .input import Input
    from .core.delay import Delay
    from .core.windmouse import WindMouse
    from .core.rng import RNG
    from .script import ScriptContext
    from scripts.fishing.shrimp import ShrimpScript

    print("=== Shrimp Fishing ===\n")

    # Detect game origin and verify window size
    game_origin = Vision.detect_game_origin(on_log=_log)
    Vision.verify_window_size(on_log=_log)

    # Build context
    rng = RNG()
    delay = Delay(seed=rng.seed, on_log=_log)
    windmouse = WindMouse(seed=rng.seed + 1, on_log=_log)
    vision = Vision(game_origin=game_origin, on_log=_log)
    inp = Input(delay=delay, windmouse=windmouse, seed=rng.seed + 2, on_log=_log)

    # Start sessions
    delay.start_session()
    windmouse.start_session()
    inp.start_session()

    stop_flag = threading.Event()
    ctx = ScriptContext(
        vision=vision,
        input=inp,
        delay=delay,
        rng=rng,
        stop_flag=stop_flag,
    )

    # Set up idle behaviors
    from .idle import IdleBehavior
    idle = IdleBehavior(ctx=ctx, on_log=_log)
    idle.start_session()
    ctx.idle = idle

    max_hours = args.max_hours if hasattr(args, "max_hours") else 6.0
    script = ShrimpScript(ctx=ctx, max_hours=max_hours, on_log=_log)

    # Wait for backtick to start
    _wait_for_hotkey("\nPress ` (backtick) to start...")
    print()

    script.start()

    # Backtick again or Ctrl+C to stop
    f12_listener = _hotkey_stop_listener(lambda: script.stop())
    print(f"Running (max {max_hours}h). Press ` or Ctrl+C to stop.\n")

    try:
        script.wait()
    except KeyboardInterrupt:
        print("\nStopping...")
        script.stop()
        script.wait(timeout=5)

    # Cleanup
    if f12_listener.is_alive():
        f12_listener.stop()
    inp.stop_session()
    windmouse.stop_session()
    delay.stop_session()
    vision.close()
    print("\nDone.")


def cmd_run_trees(args):
    """Run normal trees woodcutting script."""
    from .vision import Vision
    from .input import Input
    from .core.delay import Delay
    from .core.windmouse import WindMouse
    from .core.rng import RNG
    from .script import ScriptContext
    from scripts.woodcutting.trees import TreesScript

    print("=== Normal Trees (Lumbridge) ===\n")

    # Detect game origin and verify window size
    game_origin = Vision.detect_game_origin(on_log=_log)
    Vision.verify_window_size(on_log=_log)

    # Build context
    rng = RNG()
    delay = Delay(seed=rng.seed, on_log=_log)
    windmouse = WindMouse(seed=rng.seed + 1, on_log=_log)
    vision = Vision(game_origin=game_origin, on_log=_log)
    inp = Input(delay=delay, windmouse=windmouse, seed=rng.seed + 2, on_log=_log)

    # Start sessions
    delay.start_session()
    windmouse.start_session()
    inp.start_session()

    stop_flag = threading.Event()
    ctx = ScriptContext(
        vision=vision,
        input=inp,
        delay=delay,
        rng=rng,
        stop_flag=stop_flag,
    )

    # Set up idle behaviors
    from .idle import IdleBehavior
    idle = IdleBehavior(ctx=ctx, on_log=_log)
    idle.start_session()
    ctx.idle = idle

    max_hours = args.max_hours if hasattr(args, "max_hours") else 6.0
    light = args.light if hasattr(args, "light") else False
    script = TreesScript(ctx=ctx, max_hours=max_hours, light=light, on_log=_log)

    # Wait for backtick to start
    mode = "light" if light else "drop"
    _wait_for_hotkey(f"\nPress ` (backtick) to start ({mode} mode)...")
    print()

    script.start()

    # Backtick again or Ctrl+C to stop
    f12_listener = _hotkey_stop_listener(lambda: script.stop())
    print(f"Running (max {max_hours}h). Press ` or Ctrl+C to stop.\n")

    try:
        script.wait()
    except KeyboardInterrupt:
        print("\nStopping...")
        script.stop()
        script.wait(timeout=5)

    # Cleanup
    if f12_listener.is_alive():
        f12_listener.stop()
    inp.stop_session()
    windmouse.stop_session()
    delay.stop_session()
    vision.close()
    print("\nDone.")


def cmd_test_coords(args):
    """Live mouse position readout — screen and game-relative coords."""
    from .vision import Vision

    print("=== Coordinate Inspector ===\n")

    game_origin = Vision.detect_game_origin(on_log=_log)
    gx, gy = game_origin

    print(f"Game origin: ({gx}, {gy})")
    print("Move your mouse. Press Ctrl+C to stop.\n")
    print(f"  {'Screen':>16}  {'Game-Relative':>16}")
    print(f"  {'------':>16}  {'-------------':>16}")

    from pynput.mouse import Controller as MouseController
    mouse = MouseController()

    try:
        while True:
            pos = mouse.position
            sx, sy = int(pos[0]), int(pos[1])
            rx, ry = sx - gx, sy - gy
            print(f"  ({sx:4d}, {sy:4d})    ({rx:4d}, {ry:4d})    ", end="\r")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n\nDone.")


def cmd_run_oaks(args):
    """Run oak trees woodcutting script (banking via deposit box)."""
    from .vision import Vision
    from .input import Input
    from .core.delay import Delay
    from .core.windmouse import WindMouse
    from .core.rng import RNG
    from .script import ScriptContext
    from scripts.woodcutting.oaks import OaksScript

    print("=== Oak Trees (Deposit Box Banking) ===\n")

    # Detect game origin and verify window size
    game_origin = Vision.detect_game_origin(on_log=_log)
    Vision.verify_window_size(on_log=_log)

    # Build context
    rng = RNG()
    delay = Delay(seed=rng.seed, on_log=_log)
    windmouse = WindMouse(seed=rng.seed + 1, on_log=_log)
    vision = Vision(game_origin=game_origin, on_log=_log)
    inp = Input(delay=delay, windmouse=windmouse, seed=rng.seed + 2, on_log=_log)

    # Start sessions
    delay.start_session()
    windmouse.start_session()
    inp.start_session()

    stop_flag = threading.Event()
    ctx = ScriptContext(
        vision=vision,
        input=inp,
        delay=delay,
        rng=rng,
        stop_flag=stop_flag,
    )

    # Set up idle behaviors
    from .idle import IdleBehavior
    idle = IdleBehavior(ctx=ctx, on_log=_log)
    idle.start_session()
    ctx.idle = idle

    max_hours = args.max_hours if hasattr(args, "max_hours") else 6.0
    axe = args.axe if hasattr(args, "axe") else False
    script = OaksScript(ctx=ctx, max_hours=max_hours, axe_in_inventory=axe, on_log=_log)

    # Wait for backtick to start
    _wait_for_hotkey("\nPress ` (backtick) to start...")
    print()

    script.start()

    # Backtick again or Ctrl+C to stop
    f12_listener = _hotkey_stop_listener(lambda: script.stop())
    print(f"Running (max {max_hours}h). Press ` or Ctrl+C to stop.\n")

    try:
        script.wait()
    except KeyboardInterrupt:
        print("\nStopping...")
        script.stop()
        script.wait(timeout=5)

    # Cleanup
    if f12_listener.is_alive():
        f12_listener.stop()
    inp.stop_session()
    windmouse.stop_session()
    delay.stop_session()
    vision.close()
    print("\nDone.")


def cmd_run_willows(args):
    """Run willow trees woodcutting script (banking via deposit box)."""
    from .vision import Vision
    from .input import Input
    from .core.delay import Delay
    from .core.windmouse import WindMouse
    from .core.rng import RNG
    from .script import ScriptContext
    from scripts.woodcutting.willows import WillowsScript

    print("=== Willow Trees (Deposit Box Banking) ===\n")

    # Detect game origin and verify window size
    game_origin = Vision.detect_game_origin(on_log=_log)
    Vision.verify_window_size(on_log=_log)

    # Build context
    rng = RNG()
    delay = Delay(seed=rng.seed, on_log=_log)
    windmouse = WindMouse(seed=rng.seed + 1, on_log=_log)
    vision = Vision(game_origin=game_origin, on_log=_log)
    inp = Input(delay=delay, windmouse=windmouse, seed=rng.seed + 2, on_log=_log)

    # Start sessions
    delay.start_session()
    windmouse.start_session()
    inp.start_session()

    stop_flag = threading.Event()
    ctx = ScriptContext(
        vision=vision,
        input=inp,
        delay=delay,
        rng=rng,
        stop_flag=stop_flag,
    )

    # Set up idle behaviors
    from .idle import IdleBehavior
    idle = IdleBehavior(ctx=ctx, on_log=_log)
    idle.start_session()
    ctx.idle = idle

    max_hours = args.max_hours if hasattr(args, "max_hours") else 6.0
    axe = args.axe if hasattr(args, "axe") else False
    script = WillowsScript(ctx=ctx, max_hours=max_hours, axe_in_inventory=axe, on_log=_log)

    # Wait for backtick to start
    _wait_for_hotkey("\nPress ` (backtick) to start...")
    print()

    script.start()

    # Backtick again or Ctrl+C to stop
    f12_listener = _hotkey_stop_listener(lambda: script.stop())
    print(f"Running (max {max_hours}h). Press ` or Ctrl+C to stop.\n")

    try:
        script.wait()
    except KeyboardInterrupt:
        print("\nStopping...")
        script.stop()
        script.wait(timeout=5)

    # Cleanup
    if f12_listener.is_alive():
        f12_listener.stop()
    inp.stop_session()
    windmouse.stop_session()
    delay.stop_session()
    vision.close()
    print("\nDone.")


def cmd_run_rooftop(args):
    """Run rooftop agility script."""
    from .vision import Vision
    from .input import Input
    from .core.delay import Delay
    from .core.windmouse import WindMouse
    from .core.rng import RNG
    from .script import ScriptContext
    from scripts.agility.rooftop import RooftopScript

    print("=== Rooftop Agility ===\n")

    # Detect game origin and verify window size
    game_origin = Vision.detect_game_origin(on_log=_log)
    Vision.verify_window_size(on_log=_log)

    # Build context
    rng = RNG()
    delay = Delay(seed=rng.seed, on_log=_log)
    windmouse = WindMouse(seed=rng.seed + 1, on_log=_log)
    vision = Vision(game_origin=game_origin, on_log=_log)
    inp = Input(delay=delay, windmouse=windmouse, seed=rng.seed + 2, on_log=_log)

    # Start sessions
    delay.start_session()
    windmouse.start_session()
    inp.start_session()

    stop_flag = threading.Event()
    ctx = ScriptContext(
        vision=vision,
        input=inp,
        delay=delay,
        rng=rng,
        stop_flag=stop_flag,
    )

    # Set up idle behaviors — tuned for agility (more active, shorter gaps)
    from .idle import IdleBehavior
    idle = IdleBehavior(ctx=ctx, on_log=_log)
    idle.start_session()
    idle._burst_chance = rng.truncated_gauss(0.18, 0.04, 0.12, 0.25)
    idle._min_burst_gap = rng.truncated_gauss(8.0, 2.0, 5.0, 12.0)
    ctx.idle = idle

    max_hours = args.max_hours if hasattr(args, "max_hours") else 6.0
    script = RooftopScript(ctx=ctx, max_hours=max_hours, on_log=_log)

    # Wait for backtick to start
    _wait_for_hotkey("\nPress ` (backtick) to start...")
    print()

    script.start()

    # Backtick again or Ctrl+C to stop
    f12_listener = _hotkey_stop_listener(lambda: script.stop())
    print(f"Running (max {max_hours}h). Press ` or Ctrl+C to stop.\n")

    try:
        script.wait()
    except KeyboardInterrupt:
        print("\nStopping...")
        script.stop()
        script.wait(timeout=5)

    # Cleanup
    if f12_listener.is_alive():
        f12_listener.stop()
    inp.stop_session()
    windmouse.stop_session()
    delay.stop_session()
    vision.close()
    print("\nDone.")


def cmd_run_salmon(args):
    """Run salmon/trout fly fishing + cooking script."""
    from .vision import Vision
    from .input import Input
    from .core.delay import Delay
    from .core.windmouse import WindMouse
    from .core.rng import RNG
    from .script import ScriptContext
    from scripts.fishing.salmon import SalmonScript

    print("=== Salmon/Trout Fly Fishing ===\n")

    # Detect game origin and verify window size
    game_origin = Vision.detect_game_origin(on_log=_log)
    Vision.verify_window_size(on_log=_log)

    # Build context
    rng = RNG()
    delay = Delay(seed=rng.seed, on_log=_log)
    windmouse = WindMouse(seed=rng.seed + 1, on_log=_log)
    vision = Vision(game_origin=game_origin, on_log=_log)
    inp = Input(delay=delay, windmouse=windmouse, seed=rng.seed + 2, on_log=_log)

    # Start sessions
    delay.start_session()
    windmouse.start_session()
    inp.start_session()

    stop_flag = threading.Event()
    ctx = ScriptContext(
        vision=vision,
        input=inp,
        delay=delay,
        rng=rng,
        stop_flag=stop_flag,
    )

    # Set up idle behaviors
    from .idle import IdleBehavior
    idle = IdleBehavior(ctx=ctx, on_log=_log)
    idle.start_session()
    ctx.idle = idle

    max_hours = args.max_hours if hasattr(args, "max_hours") else 6.0
    script = SalmonScript(ctx=ctx, max_hours=max_hours, on_log=_log)

    # Wait for backtick to start
    _wait_for_hotkey("\nPress ` (backtick) to start...")
    print()

    script.start()

    # Backtick again or Ctrl+C to stop
    f12_listener = _hotkey_stop_listener(lambda: script.stop())
    print(f"Running (max {max_hours}h). Press ` or Ctrl+C to stop.\n")

    try:
        script.wait()
    except KeyboardInterrupt:
        print("\nStopping...")
        script.stop()
        script.wait(timeout=5)

    # Cleanup
    if f12_listener.is_alive():
        f12_listener.stop()
    inp.stop_session()
    windmouse.stop_session()
    delay.stop_session()
    vision.close()
    print("\nDone.")


def cmd_run_barbarian(args):
    """Run barbarian fishing script (drop all)."""
    from .vision import Vision
    from .input import Input
    from .core.delay import Delay
    from .core.windmouse import WindMouse
    from .core.rng import RNG
    from .script import ScriptContext
    from scripts.fishing.barbarian import BarbarianScript

    print("=== Barbarian Fishing ===\n")

    # Detect game origin and verify window size
    game_origin = Vision.detect_game_origin(on_log=_log)
    Vision.verify_window_size(on_log=_log)

    # Build context
    rng = RNG()
    delay = Delay(seed=rng.seed, on_log=_log)
    windmouse = WindMouse(seed=rng.seed + 1, on_log=_log)
    vision = Vision(game_origin=game_origin, on_log=_log)
    inp = Input(delay=delay, windmouse=windmouse, seed=rng.seed + 2, on_log=_log)

    # Start sessions
    delay.start_session()
    windmouse.start_session()
    inp.start_session()

    stop_flag = threading.Event()
    ctx = ScriptContext(
        vision=vision,
        input=inp,
        delay=delay,
        rng=rng,
        stop_flag=stop_flag,
    )

    # Set up idle behaviors
    from .idle import IdleBehavior
    idle = IdleBehavior(ctx=ctx, on_log=_log)
    idle.start_session()
    ctx.idle = idle

    max_hours = args.max_hours if hasattr(args, "max_hours") else 6.0
    script = BarbarianScript(ctx=ctx, max_hours=max_hours, on_log=_log)

    # Wait for backtick to start
    _wait_for_hotkey("\nPress ` (backtick) to start...")
    print()

    script.start()

    # Backtick again or Ctrl+C to stop
    f12_listener = _hotkey_stop_listener(lambda: script.stop())
    print(f"Running (max {max_hours}h). Press ` or Ctrl+C to stop.\n")

    try:
        script.wait()
    except KeyboardInterrupt:
        print("\nStopping...")
        script.stop()
        script.wait(timeout=5)

    # Cleanup
    if f12_listener.is_alive():
        f12_listener.stop()
    inp.stop_session()
    windmouse.stop_session()
    delay.stop_session()
    vision.close()
    print("\nDone.")


def cmd_run_bonfire(args):
    """Run bonfire firemaking script (bank for logs)."""
    from .vision import Vision
    from .input import Input
    from .core.delay import Delay
    from .core.windmouse import WindMouse
    from .core.rng import RNG
    from .script import ScriptContext
    from scripts.firemaking.bonfire import BonfireScript

    print("=== Bonfire Firemaking ===\n")

    # Detect game origin and verify window size
    game_origin = Vision.detect_game_origin(on_log=_log)
    Vision.verify_window_size(on_log=_log)

    # Build context
    rng = RNG()
    delay = Delay(seed=rng.seed, on_log=_log)
    windmouse = WindMouse(seed=rng.seed + 1, on_log=_log)
    vision = Vision(game_origin=game_origin, on_log=_log)
    inp = Input(delay=delay, windmouse=windmouse, seed=rng.seed + 2, on_log=_log)

    # Start sessions
    delay.start_session()
    windmouse.start_session()
    inp.start_session()

    stop_flag = threading.Event()
    ctx = ScriptContext(
        vision=vision,
        input=inp,
        delay=delay,
        rng=rng,
        stop_flag=stop_flag,
    )

    # Set up idle behaviors
    from .idle import IdleBehavior
    idle = IdleBehavior(ctx=ctx, on_log=_log)
    idle.start_session()
    ctx.idle = idle

    max_hours = args.max_hours if hasattr(args, "max_hours") else 6.0
    script = BonfireScript(ctx=ctx, max_hours=max_hours, on_log=_log)

    # Wait for backtick to start
    _wait_for_hotkey("\nPress ` (backtick) to start...")
    print()

    script.start()

    # Backtick again or Ctrl+C to stop
    f12_listener = _hotkey_stop_listener(lambda: script.stop())
    print(f"Running (max {max_hours}h). Press ` or Ctrl+C to stop.\n")

    try:
        script.wait()
    except KeyboardInterrupt:
        print("\nStopping...")
        script.stop()
        script.wait(timeout=5)

    # Cleanup
    if f12_listener.is_alive():
        f12_listener.stop()
    inp.stop_session()
    windmouse.stop_session()
    delay.stop_session()
    vision.close()
    print("\nDone.")


def cmd_run_stringing(args):
    """Run bow stringing fletching script (bank for supplies)."""
    from .vision import Vision
    from .input import Input
    from .core.delay import Delay
    from .core.windmouse import WindMouse
    from .core.rng import RNG
    from .script import ScriptContext
    from scripts.fletching.stringing import StringingScript

    print("=== Bow Stringing (Fletching) ===\n")

    # Detect game origin and verify window size
    game_origin = Vision.detect_game_origin(on_log=_log)
    Vision.verify_window_size(on_log=_log)

    # Build context
    rng = RNG()
    delay = Delay(seed=rng.seed, on_log=_log)
    windmouse = WindMouse(seed=rng.seed + 1, on_log=_log)
    vision = Vision(game_origin=game_origin, on_log=_log)
    inp = Input(delay=delay, windmouse=windmouse, seed=rng.seed + 2, on_log=_log)

    # Start sessions
    delay.start_session()
    windmouse.start_session()
    inp.start_session()

    stop_flag = threading.Event()
    ctx = ScriptContext(
        vision=vision,
        input=inp,
        delay=delay,
        rng=rng,
        stop_flag=stop_flag,
    )

    # Set up idle behaviors
    from .idle import IdleBehavior
    idle = IdleBehavior(ctx=ctx, on_log=_log)
    idle.start_session()
    ctx.idle = idle

    max_hours = args.max_hours if hasattr(args, "max_hours") else 6.0
    script = StringingScript(ctx=ctx, max_hours=max_hours, on_log=_log)

    # Wait for backtick to start
    _wait_for_hotkey("\nPress ` (backtick) to start...")
    print()

    script.start()

    # Backtick again or Ctrl+C to stop
    f12_listener = _hotkey_stop_listener(lambda: script.stop())
    print(f"Running (max {max_hours}h). Press ` or Ctrl+C to stop.\n")

    try:
        script.wait()
    except KeyboardInterrupt:
        print("\nStopping...")
        script.stop()
        script.wait(timeout=5)

    # Cleanup
    if f12_listener.is_alive():
        f12_listener.stop()
    inp.stop_session()
    windmouse.stop_session()
    delay.stop_session()
    vision.close()
    print("\nDone.")


def cmd_test_grounditem(args):
    """Live ground item detection test — scans for highlighted items."""
    from .vision import Vision, Color, GameRegions

    print("=== Ground Item Detection Test ===\n")

    game_origin = Vision.detect_game_origin(on_log=_log)
    Vision.verify_window_size(on_log=_log)
    vision = Vision(game_origin=game_origin, on_log=_log)

    color = Color.from_hex("FF00A4FF")
    print(f"Scanning for color: r={color.r}, g={color.g}, b={color.b} (#00A4FF)")
    print(f"Game view: {GameRegions.GAME_VIEW.width}x{GameRegions.GAME_VIEW.height}")
    gv = GameRegions.GAME_VIEW
    gv_cx = gv.x + gv.width // 2
    gv_cy = gv.y + gv.height // 2
    print(f"Game view center (game-rel): ({gv_cx}, {gv_cy})")
    print("\nWatching... Press Ctrl+C to stop.\n")

    last_count = -1
    try:
        while True:
            clusters = vision.find_color_clusters(
                GameRegions.GAME_VIEW, color,
                tolerance=15, min_area=10,
            )

            if clusters:
                if len(clusters) != last_count:
                    print()
                last_count = len(clusters)
                lines = [f"  DETECTED — {len(clusters)} cluster(s):"]
                for i, c in enumerate(clusters[:5]):
                    bx, by, bw, bh = c.bounding_box
                    cx, cy = c.center
                    # Distance from game view center (screen coords)
                    dist_x = cx - (gv_cx + game_origin[0])
                    dist_y = cy - (gv_cy + game_origin[1])
                    dist = (dist_x**2 + dist_y**2) ** 0.5
                    lines.append(
                        f"    #{i}: center=({cx},{cy}) area={c.area} "
                        f"bbox={bw}x{bh} dist_from_center={dist:.0f}px"
                    )
                # Overwrite with padding
                for line in lines:
                    print(f"\r{line:<80}")
            else:
                if last_count != 0:
                    print()
                last_count = 0
                print(f"\r  {'Not detected — no clusters found':<80}", end="")

            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\nDone.")
        vision.close()


def cmd_test_xpdrop(args):
    """Live XP drop detection test — uses HSV hue matching for robustness."""
    from .vision import Vision
    from .script import (XP_DROP_REGION, XP_DROP_HUE_LOW, XP_DROP_HUE_HIGH,
                         XP_DROP_SAT_MIN, XP_DROP_VAL_MIN, XP_DROP_PIXEL_THRESHOLD)

    print("=== XP Drop Detection Test (HSV) ===\n")

    game_origin = Vision.detect_game_origin(on_log=_log)
    Vision.verify_window_size(on_log=_log)
    vision = Vision(game_origin=game_origin, on_log=_log)

    region = XP_DROP_REGION
    print(f"Method:    HSV hue matching (magenta FF00FF)")
    print(f"Region:    ({region.x}, {region.y}) {region.width}x{region.height}")
    print(f"Hue:       {XP_DROP_HUE_LOW}-{XP_DROP_HUE_HIGH}  "
          f"Sat>={XP_DROP_SAT_MIN}  Val>={XP_DROP_VAL_MIN}")
    print(f"Threshold: {XP_DROP_PIXEL_THRESHOLD} pixels")
    print("\nWatching for XP drops... Press Ctrl+C to stop.\n")

    detections = 0
    polls = 0
    try:
        while True:
            pixel_count = vision.detect_hsv_pixels(
                region, XP_DROP_HUE_LOW, XP_DROP_HUE_HIGH,
                XP_DROP_SAT_MIN, XP_DROP_VAL_MIN,
            )
            polls += 1

            if pixel_count >= XP_DROP_PIXEL_THRESHOLD:
                detections += 1
                print(f"\r  XP DROP  |  {pixel_count} px  "
                      f"detections={detections}  polls={polls}     ")
            else:
                status = f"{pixel_count} px" if pixel_count > 0 else "0 px"
                print(f"\r  No drop  |  {status}  "
                      f"detections={detections}  polls={polls}     ", end="")

            time.sleep(0.3)
    except KeyboardInterrupt:
        print(f"\n\nSummary: {detections} detections in {polls} polls")
        if polls > 0:
            print(f"Detection rate: {detections/polls*100:.1f}%")
        vision.close()


def cmd_test_bankslot(args):
    """Live mouse position readout for calibrating bank interface slots."""
    from .vision import Vision

    print("=== Bank Slot Calibrator ===\n")

    game_origin = Vision.detect_game_origin(on_log=_log)
    gx, gy = game_origin

    print(f"Game origin: ({gx}, {gy})")
    print("Open the bank interface, hover over the log slot.")
    print("Note the game-relative coords to update GameRegions.BANK_LOG_SLOT.")
    print("Press Ctrl+C to stop.\n")
    print(f"  {'Screen':>16}  {'Game-Relative':>16}")
    print(f"  {'------':>16}  {'-------------':>16}")

    from pynput.mouse import Controller as MouseController
    mouse = MouseController()

    try:
        while True:
            pos = mouse.position
            sx, sy = int(pos[0]), int(pos[1])
            rx, ry = sx - gx, sy - gy
            print(f"  ({sx:4d}, {sy:4d})    ({rx:4d}, {ry:4d})    ", end="\r")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n\nDone.")


def cmd_test_depositbox(args):
    """Live deposit box detection test — uses template matching."""
    import os
    import cv2
    from .vision import Vision, GameRegions, Region

    import indigo
    template_path = os.path.join(
        os.path.dirname(indigo.__file__), "templates", "deposit_box.png"
    )

    print("=== Deposit Box Detection Test ===\n")

    game_origin = Vision.detect_game_origin(on_log=_log)
    Vision.verify_window_size(on_log=_log)
    vision = Vision(game_origin=game_origin, on_log=_log)

    region = GameRegions.DEPOSIT_ALL_BUTTON

    # --capture mode: grab the button region via mss and save as template
    if args.capture:
        print("Open the deposit box, then press Enter to capture the template...")
        input()
        frame = vision.grab(region)
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
        cv2.imwrite(template_path, frame)
        print(f"Saved {frame.shape[1]}x{frame.shape[0]} template to {template_path}")
        vision.close()
        return

    if not os.path.exists(template_path):
        print(f"ERROR: Template not found at {template_path}")
        print("Run with --capture to grab the template via mss:")
        print("  indigo test depositbox --capture")
        return

    threshold = args.threshold
    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    mask = None
    if len(template.shape) == 3 and template.shape[2] == 4:
        mask = template[:, :, 3]
        template = template[:, :, :3]
    th, tw = template.shape[:2]

    padding = 40
    grab_region = Region(
        x=max(0, region.x - padding),
        y=max(0, region.y - padding),
        width=region.width + padding * 2,
        height=region.height + padding * 2,
    )
    print(f"Template:  {tw}x{th}")
    print(f"Region:    ({region.x}, {region.y}) {region.width}x{region.height}")
    print(f"Grab:      ({grab_region.x}, {grab_region.y}) {grab_region.width}x{grab_region.height}")
    print(f"Threshold: {threshold}")
    print("\nWatching... Press Ctrl+C to stop.\n")

    open_count = 0
    polls = 0
    try:
        while True:
            frame = vision.grab(grab_region)
            if mask is not None:
                result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED, mask=mask)
            else:
                result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            polls += 1

            if max_val >= threshold:
                open_count += 1
                print(f"\r  BANK OPEN    |  score={max_val:.3f}  "
                      f"open={open_count}  polls={polls}          ")
            else:
                print(f"\r  Bank closed  |  score={max_val:.3f}  "
                      f"open={open_count}  polls={polls}          ", end="")

            time.sleep(0.3)
    except KeyboardInterrupt:
        print(f"\n\nSummary: detected open {open_count} of {polls} polls")
        vision.close()


def cmd_test_bank(args):
    """Live bank booth detection test — uses same template as deposit box."""
    import os
    import cv2
    from .vision import Vision, GameRegions, Region

    import indigo
    template_path = os.path.join(
        os.path.dirname(indigo.__file__), "templates", "deposit_box.png"
    )

    print("=== Bank Booth Detection Test ===\n")

    if not os.path.exists(template_path):
        print(f"ERROR: Template not found at {template_path}")
        print("Run 'indigo test depositbox --capture' first to grab the template.")
        return

    game_origin = Vision.detect_game_origin(on_log=_log)
    Vision.verify_window_size(on_log=_log)
    vision = Vision(game_origin=game_origin, on_log=_log)

    region = GameRegions.BANK_DEPOSIT_BUTTON
    threshold = args.threshold

    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    mask = None
    if len(template.shape) == 3 and template.shape[2] == 4:
        mask = template[:, :, 3]
        template = template[:, :, :3]
    th, tw = template.shape[:2]

    padding = 40
    grab_region = Region(
        x=max(0, region.x - padding),
        y=max(0, region.y - padding),
        width=region.width + padding * 2,
        height=region.height + padding * 2,
    )
    print(f"Template:  {tw}x{th}")
    print(f"Region:    ({region.x}, {region.y}) {region.width}x{region.height}")
    print(f"Grab:      ({grab_region.x}, {grab_region.y}) {grab_region.width}x{grab_region.height}")
    print(f"Threshold: {threshold}")
    print("\nWatching... Press Ctrl+C to stop.\n")

    open_count = 0
    polls = 0
    try:
        while True:
            frame = vision.grab(grab_region)
            if mask is not None:
                result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED, mask=mask)
            else:
                result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            polls += 1

            if max_val >= threshold:
                open_count += 1
                print(f"\r  BANK OPEN    |  score={max_val:.3f}  "
                      f"open={open_count}  polls={polls}          ")
            else:
                print(f"\r  Bank closed  |  score={max_val:.3f}  "
                      f"open={open_count}  polls={polls}          ", end="")

            time.sleep(0.3)
    except KeyboardInterrupt:
        print(f"\n\nSummary: detected open {open_count} of {polls} polls")
        vision.close()


def cmd_test_fatigue(args):
    """Preview fatigue curves."""
    from .core.fatigue import FatigueManager, FATIGUE_CONFIGS

    print("=== Fatigue Curves ===\n")

    for name, config in FATIGUE_CONFIGS.items():
        fm = FatigueManager(config=config, seed=42)
        results = fm.simulate_duration(hours=2.0, samples=10)
        mults = [f"{r[1]:.3f}" for r in results]
        print(f"  {name:15s}: {' -> '.join(mults)}")

    print("\n--- Default Curve (2h) ---\n")

    fm = FatigueManager(seed=42)
    results = fm.simulate_duration(hours=2.0, samples=20)
    for hours, mult in results:
        bar = "#" * int((mult - 1.0) * 200)
        print(f"  {hours:5.1f}h: {mult:.3f} {bar}")


def main():
    parser = argparse.ArgumentParser(
        prog="indigo",
        description="Indigo - OSRS Color Bot Framework",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # launch
    launch_parser = subparsers.add_parser("launch", help="Connect VPN + launch RuneLite")
    launch_parser.add_argument("--skip-vpn", action="store_true", help="Skip VPN connection")
    launch_parser.set_defaults(func=cmd_launch)

    # stop
    stop_parser = subparsers.add_parser("stop", help="Stop RuneLite")
    stop_parser.add_argument("--disconnect", action="store_true", help="Also disconnect VPN")
    stop_parser.set_defaults(func=cmd_stop)

    # kill
    kill_parser = subparsers.add_parser("kill", help="Emergency stop (force kill)")
    kill_parser.set_defaults(func=cmd_kill)

    # status
    status_parser = subparsers.add_parser("status", help="Show VPN and RuneLite status")
    status_parser.set_defaults(func=cmd_status)

    # test
    test_parser = subparsers.add_parser("test", help="Run test harnesses")
    test_subparsers = test_parser.add_subparsers(dest="test_command")

    delays_parser = test_subparsers.add_parser("delays", help="Test delay distributions")
    delays_parser.set_defaults(func=cmd_test_delays)

    windmouse_parser = test_subparsers.add_parser("windmouse", help="Test WindMouse paths")
    windmouse_parser.set_defaults(func=cmd_test_windmouse)

    fatigue_parser = test_subparsers.add_parser("fatigue", help="Preview fatigue curves")
    fatigue_parser.set_defaults(func=cmd_test_fatigue)

    vision_parser = test_subparsers.add_parser("vision", help="Test vision system")
    vision_parser.set_defaults(func=cmd_test_vision)

    inventory_parser = test_subparsers.add_parser("inventory", help="Test inventory detection")
    inventory_parser.set_defaults(func=cmd_test_inventory)

    drop_parser = test_subparsers.add_parser("drop", help="Test inventory dropping")
    drop_parser.add_argument("--skip", type=str, default="0", help="Slots to skip (e.g. '0', '0,1', '0-3')")
    drop_parser.set_defaults(func=cmd_test_drop)

    coords_parser = test_subparsers.add_parser("coords", help="Live mouse coordinate readout")
    coords_parser.set_defaults(func=cmd_test_coords)

    grounditem_parser = test_subparsers.add_parser("grounditem", help="Live ground item detection test")
    grounditem_parser.set_defaults(func=cmd_test_grounditem)

    xpdrop_parser = test_subparsers.add_parser("xpdrop", help="Live XP drop detection test")
    xpdrop_parser.set_defaults(func=cmd_test_xpdrop)

    bankslot_parser = test_subparsers.add_parser("bankslot", help="Calibrate bank interface slot coords")
    bankslot_parser.set_defaults(func=cmd_test_bankslot)

    depositbox_parser = test_subparsers.add_parser("depositbox", help="Live deposit box open/closed detection")
    depositbox_parser.add_argument("--threshold", type=float, default=0.8, help="Match threshold (0.0-1.0)")
    depositbox_parser.add_argument("--capture", action="store_true", help="Capture template from screen via mss")
    depositbox_parser.set_defaults(func=cmd_test_depositbox)

    bank_parser = test_subparsers.add_parser("bank", help="Live bank booth open/closed detection")
    bank_parser.add_argument("--threshold", type=float, default=0.8, help="Match threshold (0.0-1.0)")
    bank_parser.set_defaults(func=cmd_test_bank)

    # run
    run_parser = subparsers.add_parser("run", help="Run a bot script")
    run_subparsers = run_parser.add_subparsers(dest="run_command")

    shrimp_parser = run_subparsers.add_parser("shrimp", help="Fish shrimp at Lumbridge")
    shrimp_parser.add_argument("--max-hours", type=float, default=6.0, help="Max runtime in hours")
    shrimp_parser.set_defaults(func=cmd_run_shrimp)

    trees_parser = run_subparsers.add_parser("trees", help="Chop normal trees at Lumbridge")
    trees_parser.add_argument("--max-hours", type=float, default=6.0, help="Max runtime in hours")
    trees_parser.add_argument("--light", action="store_true", help="Light logs with tinderbox (slot 0) instead of dropping")
    trees_parser.set_defaults(func=cmd_run_trees)

    oaks_parser = run_subparsers.add_parser("oaks", help="Chop oak trees (bank via deposit box)")
    oaks_parser.add_argument("--max-hours", type=float, default=6.0, help="Max runtime in hours")
    oaks_parser.add_argument("--axe", action="store_true", help="Axe in inventory (lock slot 0, deposits expect 1 item remaining)")
    oaks_parser.set_defaults(func=cmd_run_oaks)

    willows_parser = run_subparsers.add_parser("willows", help="Chop willow trees (bank via deposit box)")
    willows_parser.add_argument("--max-hours", type=float, default=6.0, help="Max runtime in hours")
    willows_parser.add_argument("--axe", action="store_true", help="Axe in inventory (lock slot 0, deposits expect 1 item remaining)")
    willows_parser.set_defaults(func=cmd_run_willows)

    rooftop_parser = run_subparsers.add_parser("rooftop", help="Run rooftop agility course")
    rooftop_parser.add_argument("--max-hours", type=float, default=6.0, help="Max runtime in hours")
    rooftop_parser.set_defaults(func=cmd_run_rooftop)

    salmon_parser = run_subparsers.add_parser("salmon", help="Fly fish salmon/trout, cook on fire, drop")
    salmon_parser.add_argument("--max-hours", type=float, default=6.0, help="Max runtime in hours")
    salmon_parser.set_defaults(func=cmd_run_salmon)

    barbarian_parser = run_subparsers.add_parser("barbarian", help="Barbarian fish and drop (Otto's Grotto)")
    barbarian_parser.add_argument("--max-hours", type=float, default=6.0, help="Max runtime in hours")
    barbarian_parser.set_defaults(func=cmd_run_barbarian)

    bonfire_parser = run_subparsers.add_parser("bonfire", help="Burn logs at a bonfire (bank for more)")
    bonfire_parser.add_argument("--max-hours", type=float, default=6.0, help="Max runtime in hours")
    bonfire_parser.set_defaults(func=cmd_run_bonfire)

    stringing_parser = run_subparsers.add_parser("stringing", help="String bows at bank (fletching)")
    stringing_parser.add_argument("--max-hours", type=float, default=6.0, help="Max runtime in hours")
    stringing_parser.set_defaults(func=cmd_run_stringing)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "test" and not getattr(args, "test_command", None):
        test_parser.print_help()
        sys.exit(0)

    if args.command == "run" and not getattr(args, "run_command", None):
        run_parser.print_help()
        sys.exit(0)

    # Handle Ctrl+C - only install for commands that don't handle it themselves
    if args.command not in ("launch", "run"):
        signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

    args.func(args)


if __name__ == "__main__":
    main()
