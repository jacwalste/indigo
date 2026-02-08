# Indigo - Project Guide

## Overview

Indigo is a CLI-based color bot framework for Old School RuneScape via RuneLite. It's a clean rewrite of BlueHeeler, focused on:
- **CLI-first**: No GUI. Launch, stop, and test from the terminal.
- **Iterative testing**: Test in-game, not with mocks. Unit test only the math.
- **Undetectability**: Human-like behavior through meta-randomness and imperfection.
- **Simplicity**: Don't over-engineer. Build what you need, when you need it.

## Design Principles

### 1. Meta-Randomness
> "If we're following a random formula, we have to randomize the randomness."

- Use Gaussian distributions, not uniform
- Randomize the Gaussian parameters (mean, stddev) at session start
- Each session gets a unique "personality" that persists throughout
- Per-movement variation prevents reducible patterns

### 2. Human Imperfection
Bots that never make mistakes are detectable:
- WindMouse paths with natural drift and convergence
- Fatigue simulation (delays increase over time)
- Attention cycles (focus/unfocus oscillation)
- Thinking pauses (3% chance of 1.5-5s stops)
- Micro-stutters (15% chance of tiny hesitations)

### 3. Test In-Game
- **DO** unit test core math (RNG distributions, timing profiles, fatigue curves)
- **DON'T** unit test bot scripts with mocked vision/input -- you're just testing your mocks
- **DO** use `indigo test delays|windmouse|fatigue` to validate distributions
- **DO** run scripts in-game for 10 minutes, watch, fix, repeat

### 4. Keep It Simple
- Don't build abstractions for things that happen once
- Don't add features ahead of when you need them
- Three similar lines > a premature helper function
- If you're writing more test code than real code, reconsider

## Architecture

```
indigo/
├── indigo/
│   ├── __init__.py          # Package (v0.1.0)
│   ├── cli.py               # CLI entry point (argparse)
│   ├── session.py           # Session state machine + orchestration
│   ├── core/                # Randomization & timing (pure Python, no deps)
│   │   ├── rng.py           # Gaussian, truncated, skewed, meta-random
│   │   ├── timing.py        # Named profiles, session variation, histograms
│   │   ├── fatigue.py       # Fatigue curves, attention cycles
│   │   ├── delay.py         # Unified: timing + fatigue + pauses + stutters
│   │   └── windmouse.py     # Physics-based mouse paths
│   └── managers/            # External resource control
│       ├── vpn.py           # Mullvad VPN (Chicago enforcement)
│       └── runelite.py      # RuneLite isolation (~/.indigo/runelite)
├── scripts/                 # Bot scripts (future)
├── tests/                   # Unit tests (core math only)
├── pyproject.toml           # pip install -e . → `indigo` CLI
└── requirements.txt
```

### Data Flow

```
CLI Command → Session → Managers (VPN, RuneLite)
                ↓
            Core Systems (Delay, WindMouse, Fatigue)
                ↓
            Scripts (future: state machine + action loop)
```

## CLI Commands

```bash
indigo launch [--skip-vpn]    # Connect VPN → launch RuneLite
indigo stop [--disconnect]    # Stop RuneLite (optionally disconnect VPN)
indigo kill                   # Emergency force kill
indigo status                 # Show VPN/RuneLite/credentials status
indigo test delays            # Delay distribution histograms
indigo test windmouse         # Mouse path statistics
indigo test fatigue           # Fatigue curve preview
```

Requires venv activation: `source .venv/bin/activate`
Then `indigo` works from any directory.

## Patterns & Conventions

### Manager Pattern

All managers follow this structure:

```python
class SomeManager:
    def __init__(self, on_log: Optional[Callable[[str], None]] = None):
        self._log_callback = on_log

    def _log(self, message: str) -> None:
        if self._log_callback:
            self._log_callback(f"[Tag] {message}")
        else:
            print(f"[Tag] {message}")

    def get_status(self) -> dict:
        return {...}
```

### Log Tags

- `[VPN]` - VPN operations
- `[RuneLite]` - RuneLite lifecycle
- `[Session]` - Session state machine
- `[RNG]` - Randomization (debug)
- `[Timing]` - Delay generation (debug)
- `[Fatigue]` - Fatigue system (debug)
- `[Delay]` - Unified delay (debug)
- `[WindMouse]` - Mouse paths (debug)

### Type Hints

Use type hints for all function signatures. Keep imports organized: stdlib → third-party → local.

### Session State Machine

```
IDLE → CONNECTING_VPN → LAUNCHING_RUNELITE → WAITING_WINDOW → RUNNING
                                                                  ↓
                                                              STOPPING → IDLE
Any state → ERROR
```

## Critical Invariants

### RuneLite Isolation

Bot credentials MUST be isolated from main RuneLite:
- Bot home: `~/.indigo/runelite`
- JVM flag: `-Duser.home={bot_home}`
- Credentials: `{bot_home}/.runelite/credentials.properties`
- **NEVER** read from or write to `~/.runelite`

### VPN Enforcement

VPN MUST be connected before RuneLite launches:
1. Check `mullvad status`
2. Set relay to Chicago (`us chi`) if needed
3. Connect and verify
4. If Chicago IP is rate-limited, try a different relay

### Emergency Stop

`indigo kill` force-kills RuneLite immediately (SIGKILL).

## Anti-Detection Principles

### What Gets Detected
1. Pixel-perfect click accuracy every time
2. Mathematically uniform timing patterns
3. No variation over hours of play
4. Instant reactions (faster than humanly possible)
5. Never making mistakes
6. No breaks or AFK periods
7. Reducible mouse curves (pure bezier without noise)

### What Avoids Detection
1. Gaussian timing distributions with session variation
2. WindMouse algorithm (gravity + wind, two-phase behavior)
3. Fatigue simulation (delays increase over time)
4. Attention cycles (focus/unfocus oscillation)
5. Thinking pauses and micro-stutters
6. Per-movement parameter variation

> Detection relies heavily on mouse telemetry sent to servers.
> The more human-like the mouse data, the safer the bot.

## Adding New Systems

### New Core Module

1. Create `indigo/core/new_thing.py`
2. Add exports to `indigo/core/__init__.py`
3. Add CLI test command if it has verifiable properties
4. Test the math, not the integration

### New Manager

1. Create `indigo/managers/new_thing.py` following manager pattern
2. Add to `indigo/managers/__init__.py`
3. Wire into session if it's part of the launch flow

### New Script (future)

1. Create `scripts/skill/name.py`
2. Test iteratively in-game
3. Start with 10-minute runs, watch and fix

## Commit Conventions

Short conventional commits, no description body:

```
feat: add vision system
fix: vpn relay parsing
refactor: extract timing profiles
```

Prefix types: `feat:`, `fix:`, `refactor:`, `chore:`

## Environment

- Python 3.10+
- Venv at `.venv/`
- Install: `pip install -e .` (editable, gives you `indigo` CLI)
- Vision deps (when needed): `pip install -e ".[vision]"`
- macOS only (AppleScript for window positioning)

## What's Built vs. What's Not

### Done
- Core randomization (rng, timing, fatigue, delay, windmouse)
- VPN manager (Mullvad, Chicago enforcement)
- RuneLite manager (isolated launch, credentials, window positioning)
- Session orchestration (state machine, background threading)
- CLI interface (launch, stop, kill, status, test)

### Not Yet Built
- Vision system (screen capture, color detection)
- Input system (mouse movement execution, keyboard)
- Script engine (state machine, action loop)
- Break system
- Any actual scripts
