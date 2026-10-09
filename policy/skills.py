"""Skill layer: contact-verified grasping.

The BC policy owns the arms. This owns the gripper and the decision to believe
it. A grasp is not "the policy ran"; a grasp is "the jaws stalled at the width
this task's object actually has, verified against the per-task calibration".

That distinction is what lets a miss be retried instead of silently continuing
into the next skill with an empty gripper -- which is what the contact detector
in `contact.py` was built for.

    skill = GraspSkill(calib)
    for _ in range(skill.approach_frames):
        grip_cmd = skill.step(measured_opening)
    ...
    if skill.succeeded:
        ...

`step` returns the gripper command to emit; the caller splices it into the
action chunk. Only the gripper is driven here, because the BC policy has no
verified arm trajectory to fall back on -- an off-policy arm correction needs a
simulator this box does not have.

Uncalibrated tasks (no repeatable held width, or no detected grasps at all) fall
back to accepting any stall above the gap floor, which still detects contact but
cannot tell a good grasp from a bad one.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .contact import GripperContact, classify_width

HERE = Path(__file__).resolve().parent
DEFAULT_CALIBRATION = HERE / "calibration.json"

APPROACH_MARGIN = 0.06      # open this much wider than the object
DEFAULT_APPROACH_FRAMES = 30   # ~1 s at 30 Hz for the arm to arrive
DEFAULT_VERIFY_FRAMES = 20     # ~0.7 s for the jaws to stall and latch
CLOSE_TARGET = 0.0             # command the gripper shut
MIN_APPROACH_WIDTH = 0.10
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_MAX_STALL_WIDTH = 2.0
WIDTH_TOLERANCE = 0.12         # accept a grasp within this of the expected width


class Phase(Enum):
    IDLE = "idle"
    APPROACH = "approach"      # jaws open to straddle the object
    CLOSING = "closing"        # commanded shut; waiting for the jaws to stall
    HELD = "held"              # contact verified
    RETRY = "retry"            # attempt failed, backing off
    FAILED = "failed"          # attempts exhausted
    RELEASED = "released"


@dataclass
class TaskCalibration:
    """What a genuine grasp looks like for one task."""
    name: str
    close_cmd_max: float = 0.5
    gap_on: float = 0.15
    gap_off: float = 0.08
    held_width: float | None = None
    held_width_iqr: float | None = None
    grasp_events: int = 0
    width_repeatable: bool = False

    @property
    def calibrated(self) -> bool:
        """True when the task has a repeatable width we can verify against."""
        return bool(self.width_repeatable and self.held_width)

    @property
    def approach_width(self) -> float:
        """Commanded opening that straddles the object."""
        if not self.calibrated:
            return self.close_cmd_max + APPROACH_MARGIN
        return max(MIN_APPROACH_WIDTH, self.held_width + APPROACH_MARGIN)

    @property
    def max_stall_width(self) -> float:
        """Widest stall that still counts as an object between the jaws.

        A wide measured opening while closing means the jaws have not arrived
        yet, not that they are blocked. Bounding this keeps a late-arriving
        closure from burning retry attempts.
        """
        if self.calibrated:
            return self.held_width * 4.0
        return DEFAULT_MAX_STALL_WIDTH

    def width_ok(self, width: float | None) -> bool:
        """Is this stall the width we expect for this task?

        Returns True for uncalibrated tasks, where any stall is accepted.
        """
        if width is None:
            return False
        if not self.calibrated:
            return True
        return abs(width - self.held_width) <= WIDTH_TOLERANCE

    def classify(self, width: float | None) -> str:
        if not self.calibrated or width is None:
            return "unknown"
        lo = self.held_width - WIDTH_TOLERANCE
        hi = self.held_width + WIDTH_TOLERANCE
        return classify_width(width, handle_max=hi, body_min=float("inf"))


def load_calibration(path: str | Path | None = None) -> dict[str, TaskCalibration]:
    p = Path(path) if path else DEFAULT_CALIBRATION
    if not p.exists():
        return {}
    raw = json.loads(p.read_text()).get("tasks", {})
    out: dict[str, TaskCalibration] = {}
    for name, row in raw.items():
        out[name] = TaskCalibration(
            name=name,
            close_cmd_max=row.get("close_cmd_max", 0.5),
            gap_on=row.get("gap_on", 0.15),
            gap_off=row.get("gap_off", 0.08),
            held_width=row.get("held_width"),
            held_width_iqr=row.get("held_width_iqr"),
            grasp_events=row.get("grasp_events", 0),
            width_repeatable=bool(row.get("width_repeatable", False)),
        )
    return out


@dataclass
class GraspSkill:
    """Contact-verified grasp with bounded retries.

    Feeds the measured gripper opening once per control step and returns the
    gripper command to emit. Retries by reopening and re-straddling; it does not
    re-aim the arm, which remains the BC policy's job.
    """
    calib: TaskCalibration
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    approach_frames: int = DEFAULT_APPROACH_FRAMES
    verify_frames: int = DEFAULT_VERIFY_FRAMES

    phase: Phase = field(default=Phase.IDLE, init=False)
    attempts: int = field(default=0, init=False)
    _contact: GripperContact = field(init=False)
    _tick: int = field(default=0, init=False)
    _last_width: float | None = field(default=None, init=False)
    _rejections: int = field(default=0, init=False)
    history: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self._contact = GripperContact(close_cmd_max=self.calib.close_cmd_max,
                                       gap_on=self.calib.gap_on,
                                       gap_off=self.calib.gap_off,
                                       max_stall_width=self.calib.max_stall_width)

    @property
    def succeeded(self) -> bool:
        return self.phase is Phase.HELD

    @property
    def exhausted(self) -> bool:
        return self.phase is Phase.FAILED

    @property
    def width(self) -> float | None:
        return self._last_width

    def start(self) -> None:
        if self.phase in (Phase.IDLE, Phase.RELEASED):
            self.phase = Phase.APPROACH
            self.attempts = 0
            self._tick = 0
            self._rejections = 0
            self._contact.reset()
            self.history.clear()

    def release(self) -> None:
        self.phase = Phase.RELEASED
        self._last_width = None
        self._contact.reset()

    def step(self, measured: float) -> float:
        """Advance one control step and return the gripper command to emit."""
        if self.phase is Phase.IDLE:
            self.start()

        if self.phase is Phase.APPROACH:
            cmd = self.calib.approach_width
            self._tick += 1
            if self._tick >= self.approach_frames:
                self.phase = Phase.CLOSING
                self._tick = 0
                self._contact.reset()
                self.attempts += 1
            return cmd

        if self.phase is Phase.CLOSING:
            self._tick += 1
            self._contact.update(CLOSE_TARGET, measured)
            st = self._contact.state
            if st.grasped and self.calib.width_ok(st.width):
                self._last_width = st.width
                self.phase = Phase.HELD
                self.history.append(("held", self.attempts, st.width))
                return CLOSE_TARGET
            if st.grasped:
                self._rejections += 1
                self.history.append(("rejected_width", self.attempts, st.width))
            if self._tick >= self.verify_frames:
                self.history.append(("missed", self.attempts, self._last_width))
                self._begin_retry()
            return CLOSE_TARGET

        if self.phase is Phase.RETRY:
            cmd = self.calib.approach_width
            self._tick += 1
            if self._tick >= self.approach_frames:
                if self.attempts >= self.max_attempts:
                    self.phase = Phase.FAILED
                else:
                    self.phase = Phase.APPROACH
                    self._tick = 0
            return cmd

        if self.phase is Phase.HELD:
            # Keep holding shut; the caller releases explicitly.
            return CLOSE_TARGET

        if self.phase is Phase.RELEASED:
            return self.calib.approach_width

        return CLOSE_TARGET

    def _begin_retry(self) -> None:
        self._contact.reset()
        self._tick = 0
        if self.attempts >= self.max_attempts:
            self.phase = Phase.FAILED
        else:
            self.phase = Phase.RETRY

    def summary(self) -> dict:
        return {
            "task": self.calib.name,
            "calibrated": self.calib.calibrated,
            "phase": self.phase.value,
            "attempts": self.attempts,
            "rejected_width": self._rejections,
            "held_width": self._last_width,
            "classified": self.calib.classify(self._last_width),
        }