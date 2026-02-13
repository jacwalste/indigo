# Indigo

CLI-based color bot framework for Old School RuneScape via RuneLite. Focused on undetectability through meta-randomness, human-like imperfection, and session-varied behavior.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[vision]"
```

Requires Python 3.10+, macOS, and a Mullvad VPN subscription.

## Quick Start

```bash
indigo launch              # Connect VPN (Chicago) + launch isolated RuneLite
indigo run shrimp          # Run a script (backtick to start/stop)
indigo stop --disconnect   # Stop RuneLite + disconnect VPN
```

## CLI Commands

### Session Management

| Command | Description |
|---|---|
| `indigo launch [--skip-vpn]` | Connect VPN + launch RuneLite in isolated environment |
| `indigo stop [--disconnect]` | Stop RuneLite (optionally disconnect VPN) |
| `indigo kill` | Emergency force-kill (SIGKILL) |
| `indigo status` | Show VPN, RuneLite, and credential status |

### Scripts

All scripts: `indigo run <script> [--max-hours N]`

| Script | Command | Description |
|---|---|---|
| **Shrimp** | `indigo run shrimp` | Net shrimp at Lumbridge. Drops when inventory fills. |
| **Salmon** | `indigo run salmon` | Fly fish salmon/trout, cook on nearby fire, drop. Two-pass cooking for both fish types. |
| **Barbarian** | `indigo run barbarian` | Barbarian fish at Otto's Grotto. Drops all fish. Camera recovery on obstructed clicks. |
| **Trees** | `indigo run trees [--light]` | Chop normal trees. Drop or light logs with tinderbox. |
| **Oaks** | `indigo run oaks [--axe]` | Chop oaks, bank via deposit box. |
| **Willows** | `indigo run willows [--axe]` | Chop willows, bank via deposit box. |
| **Bonfire** | `indigo run bonfire` | Burn logs at bonfire, bank for more. |
| **Stringing** | `indigo run stringing` | String bows at bank (fletching). Withdraws 14 unstrung bows + 14 bowstrings per cycle. |
| **Rooftop** | `indigo run rooftop` | Run rooftop agility course. Color-coded obstacle sequence with fall detection. |

### Test & Calibration

All tests: `indigo test <command>`

| Command | Description |
|---|---|
| `delays` | Delay distribution histograms |
| `windmouse` | WindMouse path statistics |
| `fatigue` | Fatigue curve preview |
| `vision` | Color cluster detection + inventory grid scan |
| `inventory` | Per-slot inventory stats for calibration |
| `drop [--skip SLOTS]` | Test shift-click dropping |
| `coords` | Live mouse position readout (screen + game-relative) |
| `xpdrop` | Live XP drop detection (HSV hue matching) |
| `grounditem` | Live ground item color detection |
| `bankslot` | Calibrate bank interface slot coordinates |
| `depositbox [--capture]` | Deposit box template matching test |
| `bank` | Bank booth template matching test |

## RuneLite Plugin Requirements

Scripts rely on RuneLite plugins for color-based detection:

| Plugin | Color | Hex | Used By |
|---|---|---|---|
| NPC Indicators | Cyan | `#00FFFF` | Shrimp, Trees, Oaks, Willows |
| Object Markers | Green | `#43FF00` | Salmon (spots), Barbarian (spots) |
| Object Markers | Blue | `#0013FF` | Salmon (fire) |
| Object Markers | Cyan | `#00FFFF` | Bonfire (fire) |
| Object Markers | Red | `#FF0000` | Oaks/Willows (deposit box), Bonfire/Stringing (bank) |
| XP Drop | Magenta | `#FF00FF` | All scripts with XP verification |
| Rooftop obstacles | Per-obstacle | Varies | Rooftop (cyan/green/blue/yellow/purple/orange) |

**Fixed Mode required** -- all coordinate systems are calibrated for OSRS Fixed Mode.

## Anti-Detection Systems

### Meta-Randomness

Every session generates a unique "personality" by varying the parameters of randomization itself:
- Gaussian distributions with session-varied mean/stddev
- Click hold durations, movement speeds, and delay profiles differ per session
- No two sessions produce the same behavioral fingerprint

### Human-Like Input

- **WindMouse** paths with gravity, wind, and two-phase convergence (not bezier curves)
- **Click imperfections**: multi-clicks (18%), position misclicks (4%), wrong click type (3%)
- **Drop styles**: per-item shift-click, held-shift clicking, or chunked groups -- randomly chosen
- **Trackpad lift** simulation on longer mouse moves (6%)
- **Micro-hesitations** mid-movement (12%)

### Fatigue & Attention

- **Fatigue curves**: delays gradually increase over hours of play
- **Attention bursts**: long idle periods punctuated by 2-4 action bursts (checking stats, fidgeting, camera nudges)
- **AFK breaks**: periodic breaks up to ~4 minutes, session-varied frequency
- **Thinking pauses**: 3% chance of 1.5-5s stops
- **Micro-stutters**: 15% chance of tiny hesitations

### Idle Behaviors

Weighted random actions during downtime:
- Mouse fidget / wander
- Camera nudge / spin / circle
- Check stats tab, browse other tabs
- Hover inventory items
- Zoom adjust

### Variable Reaction Times

Not every action is instant. When inventory fills:
- 25% instant reaction
- 40% brief pause (~1s)
- 25% slow notice (~4s)
- 10% AFK delay (~12s) with activity burst on return

## Architecture

```
indigo/
├── indigo/
│   ├── cli.py               # CLI entry point
│   ├── session.py            # Session state machine
│   ├── vision.py             # Screen capture, color detection, inventory grid
│   ├── input.py              # Mouse/keyboard with WindMouse paths
│   ├── script.py             # Base Script class, drop system, XP detection
│   ├── idle.py               # Idle behavior system (attention bursts)
│   ├── core/
│   │   ├── rng.py            # Gaussian, truncated, skewed, meta-random
│   │   ├── timing.py         # Named delay profiles
│   │   ├── fatigue.py        # Fatigue curves, attention cycles
│   │   ├── delay.py          # Unified delays (timing + fatigue + pauses)
│   │   └── windmouse.py      # Physics-based mouse paths
│   └── managers/
│       ├── vpn.py            # Mullvad VPN (Chicago enforcement)
│       └── runelite.py       # Isolated RuneLite launch
├── scripts/
│   ├── fishing/              # shrimp, salmon, barbarian
│   ├── woodcutting/          # trees, oaks, willows
│   ├── firemaking/           # bonfire
│   ├── fletching/            # stringing
│   └── agility/              # rooftop
└── tests/                    # Unit tests (core math only)
```

### Key Design Decisions

- **Test in-game, not with mocks.** Unit test the math (RNG distributions, timing profiles, fatigue curves). Run scripts in-game for 10 minutes, watch, fix, repeat.
- **RuneLite isolation.** Bot credentials run in `~/.indigo/runelite` with `-Duser.home` -- never touches your main `~/.runelite`.
- **VPN enforcement.** Mullvad must be connected to Chicago before RuneLite launches.
- **CLI-first.** No GUI. Launch, run, stop, test from terminal. Backtick hotkey to start/stop scripts.
