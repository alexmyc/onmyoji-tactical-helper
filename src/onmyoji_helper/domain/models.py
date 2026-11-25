"""Core domain models, enumerations, and value objects for Onmyoji Tactical Helper.

This module is strictly pure Python (zero Win32 or Tkinter dependencies), enabling
deterministic offline unit testing across vision, decision, and automation layers.
"""

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

UNKNOWN_SHIKIGAMI: str = "未知"


class OperatingMode(StrEnum):
    """Tri-state operational mode governing AI reasoning, HUD overlays, and auto-clicking."""

    GREEN = "green"    # Draft assistant only (0 VRAM)
    YELLOW = "yellow"  # Draft + HUD target/skill rings + Tactical AI advice (No auto-click)
    RED = "red"        # Full Auto-Battle execution + HUD + AI advice + F12/ESC emergency brake


class DraftPhase(StrEnum):
    """Finite-state machine phases for the Ban/Pick drafting lifecycle."""

    SLEEP = "sleep"
    BAN_WAITING = "ban_waiting"
    ACTIVE_DRAFT = "active_draft"
    IN_BATTLE = "in_battle"


@dataclass(frozen=True)
class CropOffset:
    """Represents black-border crop offsets relative to the raw window client area.

    Attributes:
        x0: Left black-pillar crop offset in client pixels.
        y0: Top black-letterbox crop offset in client pixels.
        client_w: Full uncropped client width in physical pixels.
        client_h: Full uncropped client height in physical pixels.
    """

    x0: int = 0
    y0: int = 0
    client_w: int = 0
    client_h: int = 0


@dataclass
class CaptureFrame:
    """Container for a captured game window frame and its exact physical crop geometry."""

    bgr: np.ndarray
    gray: np.ndarray | None = None
    hwnd: int = 0
    window_title: str = ""
    crop: CropOffset = field(default_factory=CropOffset)

    @property
    def width(self) -> int:
        """Cropped frame width in pixels."""
        return int(self.bgr.shape[1]) if self.bgr is not None and self.bgr.ndim >= 2 else 0

    @property
    def height(self) -> int:
        """Cropped frame height in pixels."""
        return int(self.bgr.shape[0]) if self.bgr is not None and self.bgr.ndim >= 2 else 0


@dataclass
class DraftSnapshot:
    """Immutable-style snapshot of current Ban/Pick draft state emitted by DraftStateMachine."""

    phase: DraftPhase = DraftPhase.SLEEP
    opp_ban: str | None = None
    opp_draft: list[str] = field(default_factory=list)
    my_draft: list[str] = field(default_factory=list)
    just_completed_five_picks: bool = False
    should_reset_battle: bool = False
    status_hint: str = ""


@dataclass(frozen=True)
class TargetDecision:
    """Structured output produced by TargetSelector for a single battle frame."""

    target_slot: str | None
    target_name: str
    target_role: str
    chosen_skill: str
    scope: str
    target_is_enemy: bool


SlotStatusMap = dict[str, dict[str, object]]
SkillAvailabilityMap = dict[str, bool]
SkillSlotMapping = dict[str, str | None]
NormalizedRect = tuple[float, float, float, float]
