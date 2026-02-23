"""
Indigo Core Systems

Randomization, timing, fatigue, delay generation, and mouse path simulation.
"""

from .rng import RNG, DistributionParams
from .timing import (
    TimingProfile,
    DelayGenerator,
    DelayStats,
    TIMING_PROFILES,
    INSTANT,
    FAST_ACTION,
    NORMAL_ACTION,
    CAREFUL_ACTION,
    SLOW_ACTION,
    THINKING_PAUSE,
    MENU_NAVIGATION,
    TYPING_DELAY,
)
from .fatigue import (
    FatigueManager,
    FatigueConfig,
    FatigueCurve,
    AttentionState,
    FATIGUE_CONFIGS,
)
from .delay import (
    Delay,
    DelayConfig,
    DelayBreakdown,
    ThinkingPauseConfig,
    MicroStutterConfig,
    DELAY_CONFIGS,
)
from .windmouse import (
    WindMouse,
    WindMouseConfig,
    Point,
    Path,
    PathStats,
    WINDMOUSE_CONFIGS,
)
from .recorder import (
    Recorder,
    MouseEvent,
    RawRecording,
)
from .replay import (
    ReplayGenerator,
    MovementTemplate,
    PathLibrary,
    build_library,
    save_library,
    load_library,
)

__all__ = [
    # RNG
    "RNG",
    "DistributionParams",
    # Timing
    "TimingProfile",
    "DelayGenerator",
    "DelayStats",
    "TIMING_PROFILES",
    "INSTANT",
    "FAST_ACTION",
    "NORMAL_ACTION",
    "CAREFUL_ACTION",
    "SLOW_ACTION",
    "THINKING_PAUSE",
    "MENU_NAVIGATION",
    "TYPING_DELAY",
    # Fatigue
    "FatigueManager",
    "FatigueConfig",
    "FatigueCurve",
    "AttentionState",
    "FATIGUE_CONFIGS",
    # Delay
    "Delay",
    "DelayConfig",
    "DelayBreakdown",
    "ThinkingPauseConfig",
    "MicroStutterConfig",
    "DELAY_CONFIGS",
    # WindMouse
    "WindMouse",
    "WindMouseConfig",
    "Point",
    "Path",
    "PathStats",
    "WINDMOUSE_CONFIGS",
    # Recorder
    "Recorder",
    "MouseEvent",
    "RawRecording",
    # Replay
    "ReplayGenerator",
    "MovementTemplate",
    "PathLibrary",
    "build_library",
    "save_library",
    "load_library",
]
