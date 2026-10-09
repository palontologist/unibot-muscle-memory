"""Proprioceptive grasp-contact detection.

The gripper's measured opening is a contact sensor. Command a close and the
jaws travel to the target opening -- unless something is between them, in which
case they stall at the object's width and the measured opening stays *above* the
commanded target:

    free motion    measured - commanded ~= 0            (jaws reach the target)
    holding       measured - commanded ~= +0.25        (blocked by the object)

The stall magnitude is the object's cross-section, so the same reading answers
both "did I grab anything" and "how thick is it". Verified on the HangCup task
data: free-motion gap has p99 +0.04, held gap +0.247 sustained over ~18 frames,
and the stalled width clusters tightly at 0.313 (the cup handle).

This lets a grasp verify itself instead of replaying a fixed trajectory, which is
what makes it tolerate the object being moved. It does not localise the object;
the camera is still needed to aim the approach.

Calibration is per task. Sweeping the detector over the downloaded task datasets
gives a tight, task-specific held width:

    ZipUp 0.230+-0.031   StackBowls 0.283+-0.077   HangCup 0.304+-0.009
    PlugCharger 0.261+-0.025   Unlock 0.499+-0.053   PickOutBatteries 0.525+-0.018
    PlugInEthernet 0.572+-0.144   DisconnectEthernet 0.982+-0.025

so `gap_on` is a usable floor across tasks but the *expected* width, and any
handle-vs-body thresholds, must come from the task. Three datasets
(ArrangeTestTubes, PaperCupStacking, PlacePhotoFrame) never command a fully
closed gripper and so need a higher `close_cmd_max`.

Note the gap must only be read once the jaws have settled. While the gripper is
still decelerating, measured leads commanded by ~+0.45 -- a *larger* gap than the
~+0.25 stall -- so reading the gap mid-motion reports free-space closing as
contact. `settle_eps`/`settle_frames` exist for that.

Not wired into the BC policy yet -- this is the primitive the skill layer will
consume. Units are whatever the gripper reports (fully open is ~4.5).
"""

from dataclasses import dataclass, field

# Defaults measured on G1_Dex1_HangCup. Free-motion gap p99 is +0.040, held gap
# is +0.247, so 0.15 sits ~4x clear of the noise floor.
DEFAULT_CLOSE_CMD_MAX = 0.5
DEFAULT_GAP_ON = 0.15
DEFAULT_GAP_OFF = 0.08      # hysteresis, so a held grasp cannot chatter
DEFAULT_SETTLE_EPS = 0.02   # measured must be steady before the gap is read
DEFAULT_SETTLE_FRAMES = 3
DEFAULT_HOLD_FRAMES = 3     # stall must persist this long before "grasped"


@dataclass
class ContactState:
    """Snapshot of what the gripper is doing."""
    closing: bool = False       # a close command is being issued
    settled: bool = False       # measured opening has stopped moving
    grasped: bool = False       # jaws stalled on an object, debounced
    gap: float = 0.0            # measured - commanded
    width: float | None = None  # measured opening while held (object width)
    hold_frames: int = 0        # consecutive stalled frames


@dataclass
class GripperContact:
    """Stateful contact detector for one gripper.

    Feed it the commanded and measured opening once per control step:

        c = GripperContact()
        c.update(action_gripper[0], observation_state_gripper[0])
        if c.grasped:
            ...
    """
    close_cmd_max: float = DEFAULT_CLOSE_CMD_MAX
    gap_on: float = DEFAULT_GAP_ON
    gap_off: float = DEFAULT_GAP_OFF
    settle_eps: float = DEFAULT_SETTLE_EPS
    settle_frames: int = DEFAULT_SETTLE_FRAMES
    hold_frames: int = DEFAULT_HOLD_FRAMES
    max_stall_width: float | None = None

    _recent: list = field(default_factory=list, repr=False)
    _stall_run: int = 0
    state: ContactState = field(default_factory=ContactState)

    def update(self, commanded: float, measured: float) -> ContactState:
        commanded = float(commanded)
        measured = float(measured)

        closing = commanded <= self.close_cmd_max
        gap = measured - commanded

        self._recent.append(measured)
        if len(self._recent) > self.settle_frames + 1:
            self._recent.pop(0)
        settled = False
        if len(self._recent) >= self.settle_frames + 1:
            window = self._recent[-(self.settle_frames + 1):]
            settled = (max(window) - min(window)) <= self.settle_eps

        # Only read the gap once the jaws have stopped moving: during the closing
        # transient the jaws lag the command, which is actuator dynamics and not
        # contact.
        stalled = closing and settled and gap > self.gap_on
        # A stall only means contact if the jaws are actually near shut. When the
        # measured opening is still wide the gripper simply has not moved yet,
        # which looks like a huge positive gap and must not be read as an object.
        if stalled and self.max_stall_width is not None and measured > self.max_stall_width:
            stalled = False
        if stalled:
            self._stall_run += 1
        else:
            self._stall_run = 0

        # Hysteresis: require a lower gap to drop out than to enter, and once
        # holding do not re-require the settle window -- a momentary dip in the
        # measured opening must not drop a grasp that is still closed on an
        # object.
        if self.state.grasped:
            grasped = closing and gap > self.gap_off
            if grasped and self.max_stall_width is not None and measured > self.max_stall_width:
                grasped = False
        else:
            grasped = self._stall_run >= self.hold_frames

        self.state = ContactState(
            closing=closing,
            settled=settled,
            grasped=grasped,
            gap=gap,
            width=measured if grasped else None,
            hold_frames=self._stall_run if stalled else 0,
        )
        return self.state

    @property
    def grasped(self) -> bool:
        return self.state.grasped

    @property
    def width(self) -> float | None:
        """Object cross-section while held, None otherwise."""
        return self.state.width

    def reset(self) -> None:
        self._recent.clear()
        self._stall_run = 0
        self.state = ContactState()


def classify_width(width: float, handle_max: float, body_min: float) -> str:
    """Label a held width as 'handle', 'body' or 'unknown'.

    On HangCup the handle stalls at ~0.313 while the cup body is markedly
    thicker, so a width gate rejects a grasp that closed on the wrong part.
    """
    if width is None:
        return "unknown"
    if width <= handle_max:
        return "handle"
    if width >= body_min:
        return "body"
    return "unknown"