"""Checks for the proprioceptive contact detector.

    python -m tests.test_contact
"""

import numpy as np

from policy.contact import GripperContact, classify_width

OPEN = [4.5] * 10
# commanded and measured during the closing sweep, jaws tracking the target
CLOSE_CMD = [4.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.07]
CLOSE_FREE = [4.5, 3.6, 2.4, 1.3, 0.7, 0.35, 0.10]


def replay(cmd, meas, c=None):
    c = c or GripperContact()
    return c, [c.update(a, b) for a, b in zip(cmd, meas)]


def grasped_states(states):
    return [s for s in states if s.grasped]


def test_empty_close_never_reports_contact():
    _, st = replay(OPEN + CLOSE_CMD + [0.07] * 14, OPEN + CLOSE_FREE + [0.07] * 14)
    assert not grasped_states(st), "false positive closing on nothing"
    return "1) empty close -> never grasped:            PASS"


def test_close_on_object_reports_width():
    cmd = OPEN + CLOSE_CMD + [0.07] * 14
    meas = OPEN + CLOSE_FREE + [0.31] * 14
    _, st = replay(cmd, meas)
    g = grasped_states(st)
    assert g, "missed the grasp"
    assert abs(g[-1].width - 0.31) < 0.02, f"width {g[-1].width}"
    return f"2) close on 0.31 object -> width {g[-1].width:.4f}:  PASS"


def test_closing_transient_is_not_contact():
    """Measured leads commanded by ~+0.45 while the jaws decelerate, which is a
    larger gap than the ~+0.25 stall. Without the settle filter this reads as a
    grasp on nothing."""
    cmd = OPEN + CLOSE_CMD
    meas = OPEN + [4.5, 3.6, 2.4, 1.3, 0.7, 0.35]
    _, st = replay(cmd, meas)
    assert not grasped_states(st), "transient false positive"
    return "3) closing transient (gap +0.45) ignored:      PASS"


def test_width_classification():
    assert classify_width(0.31, 0.40, 0.60) == "handle"
    assert classify_width(0.75, 0.40, 0.60) == "body"
    assert classify_width(0.50, 0.40, 0.60) == "unknown"
    assert classify_width(None, 0.40, 0.60) == "unknown"
    return "4) classify_width(handle/body/unknown):       PASS"


def test_reset_clears_state():
    cmd = OPEN + CLOSE_CMD + [0.07] * 14
    meas = OPEN + CLOSE_FREE + [0.31] * 14
    c, st = replay(cmd, meas)
    assert c.grasped
    c.reset()
    assert not c.grasped and c.width is None
    return "5) reset clears state:                        PASS"


def test_hysteresis_holds_grasp_through_dip():
    cmd = OPEN + CLOSE_CMD + [0.07] * 8
    meas = OPEN + CLOSE_FREE + [0.31] * 6 + [0.18, 0.18]
    c, _ = replay(cmd, meas)
    assert c.grasped, "hysteresis released too early"
    return "6) hysteresis holds grasp through dip:        PASS"


def test_release_detected():
    cmd = OPEN + CLOSE_CMD + [0.07] * 10 + [3.0] * 10
    meas = OPEN + CLOSE_FREE + [0.31] * 10 + [3.0] * 10
    c, st = replay(cmd, meas)
    assert c.grasped is False, "still reported holding after release"
    return "7) release drops the grasp:                   PASS"


if __name__ == "__main__":
    tests = [test_empty_close_never_reports_contact,
             test_close_on_object_reports_width,
             test_closing_transient_is_not_contact,
             test_width_classification,
             test_reset_clears_state,
             test_hysteresis_holds_grasp_through_dip,
             test_release_detected]
    for t in tests:
        print(t())
    print(f"\nall {len(tests)} checks passed")