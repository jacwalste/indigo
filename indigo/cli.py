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

import argparse
import signal
import sys
import time
import threading


def _log(message: str) -> None:
    print(message)


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

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "test" and not getattr(args, "test_command", None):
        test_parser.print_help()
        sys.exit(0)

    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

    args.func(args)


if __name__ == "__main__":
    main()
