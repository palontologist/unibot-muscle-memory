"""Checks for the contact-verified grasp skill.

    python -m tests.test_skills
"""

import numpy as np

from policy.skills import (APPROACH_MARGIN, CLOSE_TARGET, GraspSkill, Phase,
                           TaskCalibration, load_calibration)

OPEN = [4.5] * 12
CLOSE_CMD = [4.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.07]
FREE = [4.5, 3.6, 2.4, 1.3, 0.7, 0.35, 0.10]

HANG = TaskCalibration(name="G1_Dex1_HangCup", held_width=0.3042,
                       held_width_iqr=0.0089, grasp_events=2, width_repeatable=True)
UNCAL = TaskCalibration(name="G1_Dex1_PaperCupStacking", grasp_events=0,
                        width_repeatable=False)


def run(skill, measured, arm_approach=True):
    """Drive the skill with a measured-opening trace; returns commands emitted."""
    cmds = []
    for m in measured:
        cmds.append(skill.step(m))
    return cmds


def holding_trace(width, n=40):
    return OPEN + FREE + [width] * n


def test_grasp_succeeds_on_matching_width():
    s = GraspSkill(HANG)
    cmds = run(s, holding_trace(0.31))
    assert s.succeeded, s.summary()
    assert abs(s.width - 0.31) < 0.05, s.width
    assert s.calib.classify(s.width) == "handle", s.calib.classify(s.width)
    return f"1) matching width -> HELD, classified 'handle': PASS"


def test_approach_opens_wider_than_object():
    s = GraspSkill(HANG)
    s.start()
    first = s.step(4.5)
    assert abs(first - (HANG.held_width + APPROACH_MARGIN)) < 1e-6, first
    return f"2) approach opens to {first:.4f} (> object {HANG.held_width}): PASS"


def test_miss_then_success_on_retry():
    s = GraspSkill(HANG, max_attempts=6, approach_frames=3, verify_frames=6)
    # exactly one failed attempt: 3 approach frames, then 6 closing frames where
    # the jaws track the command (nothing caught)
    run(s, [4.5] * 3 + [0.07] * 6)
    assert not s.succeeded and not s.exhausted, s.summary()
    assert s.attempts == 1, s.attempts
    # then a successful approach + close that stalls on the handle
    run(s, [4.5] * 6 + [0.31] * 20)
    assert s.succeeded, s.summary()
    assert s.attempts == 2, s.attempts
    return f"miss then success on attempt {s.attempts}: PASS"


def test_wrong_width_is_rejected_and_retried():
    """Gripping the 0.98 body instead of the 0.30 handle must not count."""
    s = GraspSkill(HANG, max_attempts=3, approach_frames=3, verify_frames=8)
    run(s, [4.5] * 3 + [0.98] * 30)
    summary = s.summary()
    assert not s.succeeded, summary
    assert summary["rejected_width"] >= 1, summary
    return f"0.98 body rejected on handle task (rejections={summary['rejected_width']}): PASS"


def test_uncalibrated_task_accepts_any_stall():
    s = GraspSkill(UNCAL, approach_frames=3, verify_frames=6)
    run(s, holding_trace(0.75, 20))
    assert s.succeeded, s.summary()
    return "uncalibrated task accepts any stall:              PASS"


def test_miss_retries_then_fails():
    s = GraspSkill(HANG, max_attempts=2, approach_frames=3, verify_frames=6)
    empty = OPEN + FREE + [0.07] * 200
    run(s, empty)
    assert s.exhausted, s.summary()
    assert s.attempts == 2, s.attempts
    return f"exhausts after {s.attempts} attempts -> FAILED:      PASS"


def test_release_clears_hold():
    s = GraspSkill(HANG)
    run(s, holding_trace(0.31, 20))
    assert s.succeeded
    s.release()
    assert s.phase is Phase.RELEASED and not s.succeeded and s.width is None
    return "7) release clears hold:                           PASS"


def test_calibration_file_loads_and_flags():
    cal = load_calibration()
    assert cal, "calibration.json missing or empty"
    hang = cal.get("G1_Dex1_HangCup")
    assert hang is not None and hang.calibrated, "HangCup should be calibrated"
    assert abs(hang.held_width - 0.3042) < 1e-3, hang.held_width
    cup = cal.get("G1_Dex1_PaperCupStacking")
    assert cup is not None and not cup.calibrated
    n_cal = sum(1 for c in cal.values() if c.calibrated)
    return f"8) calibration.json: {len(cal)} tasks, {n_cal} calibrated:  PASS"


def gripper_sim(skill, object_width, n_steps, open_val=4.5, lag=0.35):
    """Closed-loop replay against a simple gripper model.

    The jaws lag the command (first-order) and stall at `object_width` when
    closing past an object that is between them; `object_width=None` means an
    empty jaw. This is the only valid way to exercise a closed-loop skill: the
    recorded task data is open-loop, so pairing its measured opening with the
    skill's own command is not causally consistent.
    """
    measured = open_val
    trace = []
    for _ in range(n_steps):
        cmd = skill.step(measured)
        target = cmd
        if object_width is not None and measured <= object_width and cmd < measured:
            target = measured  # blocked by the object
        measured += (target - measured) * lag
        trace.append((cmd, measured))
    return trace


def test_closed_loop_latches_on_object():
    s = GraspSkill(HANG, approach_frames=12, verify_frames=20)
    tr = gripper_sim(s, object_width=0.30, n_steps=80)
    assert s.succeeded, s.summary()
    return f"closed loop on 0.30 object -> HELD, width {s.width:.3f}:  PASS"


def test_closed_loop_retries_when_nothing_there():
    s = GraspSkill(HANG, max_attempts=3, approach_frames=12, verify_frames=15)
    tr = gripper_sim(s, object_width=None, n_steps=200)
    assert s.exhausted, s.summary()
    assert s.attempts == 3, s.attempts
    return f"empty jaw -> FAILED after {s.attempts} attempts:    PASS"


def test_closed_loop_ignores_late_closure():
    """A jaw still wide open when commanded shut must not read as an object."""
    s = GraspSkill(HANG, approach_frames=4, verify_frames=25)
    # very slow jaws: they are still wide when the close command lands
    measured = 4.5
    saw_wide = False
    for _ in range(40):
        cmd = s.step(measured)
        target = cmd if cmd >= measured else measured
        measured += (target - measured) * 0.08
        if measured > 2.0:
            saw_wide = True
        assert not s.succeeded, f"latched on a wide-open jaw: {s.summary()}"
    assert saw_wide, "test never exercised a wide jaw"
    return "wide jaw while closing is not a grasp:          PASS"


if __name__ == "__main__":
    tests = [test_approach_opens_wider_than_object,
             test_grasp_succeeds_on_matching_width,
             test_miss_then_success_on_retry,
             test_wrong_width_is_rejected_and_retried,
             test_uncalibrated_task_accepts_any_stall,
             test_miss_retries_then_fails,
             test_release_clears_hold,
             test_closed_loop_latches_on_object,
             test_closed_loop_retries_when_nothing_there,
             test_closed_loop_ignores_late_closure,
             test_calibration_file_loads_and_flags]
    for t in tests:
        print(t())
    print(f"\nall {len(tests)} checks passed")