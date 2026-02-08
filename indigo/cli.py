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
            cx, cy = vision.slot_screen_center(idx)
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

    # run
    run_parser = subparsers.add_parser("run", help="Run a bot script")
    run_subparsers = run_parser.add_subparsers(dest="run_command")

    shrimp_parser = run_subparsers.add_parser("shrimp", help="Fish shrimp at Lumbridge")
    shrimp_parser.add_argument("--max-hours", type=float, default=6.0, help="Max runtime in hours")
    shrimp_parser.set_defaults(func=cmd_run_shrimp)

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
