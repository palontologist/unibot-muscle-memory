"""UniBot-V1 submission adapter for the Muscle Memory policy.

Implements the interface from unitreerobotics/unibot_submission:

    metadata (property) / get_action(obs) / reset()

Proprioception + language only: the policy sees no camera frames, so
`image_resize` is empty and only the state keys the model consumes are declared
as `obs_delta_indices`. Inference is deterministic (eval mode, no sampling),
which keeps per-step latency flat -- the Section 6 credibility metric.
"""

import os
from pathlib import Path

import numpy as np
import torch

from .muscle_memory import MuscleMemoryPolicy, hash_tokens
from .skills import GraspSupervisor, load_calibration, resolve_task

HERE = Path(__file__).resolve().parent
DEFAULT_CKPT = HERE.parent / "checkpoints" / "muscle_memory_challenge.pt"


class MuscleMemorySubmission:
    HORIZON = 4  # stacked proprio frames; must match obs_delta_indices below
    CHUNK = 8    # action steps returned per get_action call

    def __init__(self, checkpoint: str | Path | None = None):
        self._token = os.environ["UNIBOT_SUBMISSION_TOKEN"]
        self._control_space = os.environ.get("UNIBOT_CONTROL_SPACE", "joint")
        path = Path(checkpoint) if checkpoint else DEFAULT_CKPT
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        if not (cfg["action_chunk_size"] == self.CHUNK and cfg["proprio_horizon"] == self.HORIZON):
            raise ValueError(f"checkpoint config {cfg} incompatible with this wrapper")
        self.stats = ckpt["stats"]
        self.model = MuscleMemoryPolicy(
            action_chunk_size=cfg["action_chunk_size"],
            proprio_horizon=cfg["proprio_horizon"],
            num_loops=cfg["num_loops"],
            embed_dim=cfg["embed_dim"],
        )
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.eval()
        self._lang_cache: dict[str, list[int]] = {}
        self.last_confidence = 0.0
        self._calibration = load_calibration()
        self._supervisor: GraspSupervisor | None = None
        self._supervisor_task: str | None = None
        self._instruction: str | None = None
        self.use_contact_supervisor = os.environ.get(
            "UNIBOT_CONTACT_SUPERVISOR", "1") not in ("0", "false", "False")

    @property
    def metadata(self):
        h = [-3, -2, -1, 0]
        return {
            "control_space": self._control_space,
            "obs_delta_indices": {
                "observation.language": [0],
                "observation.state.left_arm": list(h),
                "observation.state.right_arm": list(h),
                "observation.state.left_gripper": list(h),
                "observation.state.right_gripper": list(h),
            },
            "image_resize": {},
            "action_chunk_size": self.CHUNK,
            "token": self._token,
        }

    def _language(self, text: str) -> torch.Tensor:
        ids = self._lang_cache.get(text)
        if ids is None:
            ids = hash_tokens(text, 4096, 12)
            self._lang_cache[text] = ids
        return torch.tensor(ids, dtype=torch.long)[None]

    def _proprio(self, obs: dict) -> torch.Tensor:
        def get(key):
            a = np.asarray(obs[key], dtype=np.float32)
            return a.reshape(-1, a.shape[-1])  # (frames, dim)

        state = np.concatenate([
            get("observation.state.left_arm"),      # (4,7)
            get("observation.state.right_arm"),     # (4,7)
            get("observation.state.left_gripper"),  # (4,1)
            get("observation.state.right_gripper"), # (4,1)
        ], axis=1)                                  # (4,16)
        norm = (state - self.stats["state_mean"][None]) / (self.stats["state_std"][None] + 1e-6)
        return torch.from_numpy(norm.astype(np.float32))[None]

    def _ensure_supervisor(self, instruction: str) -> GraspSupervisor | None:
        """Bind a supervisor to the current task's calibration.

        Inference only ever sees the instruction, so the task is resolved from its
        phrasing. An unresolved task still gets a supervisor -- just one that
        cannot grade the grasp width, which is enough to notice a close that
        caught nothing.
        """
        if not self.use_contact_supervisor:
            return None
        if instruction != self._instruction:
            self._instruction = instruction
            calib = resolve_task(instruction, self._calibration)
            self._supervisor = GraspSupervisor(calib=calib)
            self._supervisor_task = calib.name if calib else None
        return self._supervisor

    @torch.no_grad()
    def get_action(self, obs: dict) -> dict:
        instruction = obs["observation.language"]
        supervisor = self._ensure_supervisor(instruction)

        chunk, confidence = self.model(self._language(instruction),
                                       self._proprio(obs))
        a = chunk[0].numpy()
        a = a * self.stats["action_std"][None] + self.stats["action_mean"][None]
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        T = self.CHUNK
        # action.pivot is the shared lower-body target. The training demos are
        # not zero-mean (dims 4 and 6 sit near 0.15 / 0.74), so hold the pooled
        # dataset mean rather than a zero vector.
        pivot = self.stats.get("pivot_mean")
        pivot = np.zeros(7, np.float32) if pivot is None else np.asarray(pivot, np.float32)

        left_grip = a[:, 14:15]
        right_grip = a[:, 15:16]

        # The supervisor watches the close the policy commanded against the
        # opening actually measured, and only overrides when nothing was caught.
        # When grasps succeed this is a no-op and the chunk is untouched.
        if supervisor is not None:
            measured = float(np.asarray(obs["observation.state.left_gripper"],
                                        dtype=np.float32).reshape(-1)[-1])
            cmd = supervisor.step(float(left_grip[0, 0]), measured)
            if supervisor.intervening or cmd != float(left_grip[0, 0]):
                left_grip = np.tile(np.float32(cmd), (T, 1))

        action = {
            "meta.token": self._token,
            "action.left_gripper": np.ascontiguousarray(left_grip),
            "action.right_gripper": np.ascontiguousarray(right_grip),
            "action.pivot": np.tile(pivot, (T, 1)).astype(np.float32),
        }
        if self._control_space != "joint":
            raise NotImplementedError(
                f"control_space {self._control_space!r} needs a joint->FK stage; "
                "this submission serves 'joint' only"
            )
        action["action.left_arm"] = np.ascontiguousarray(a[:, 0:7])
        action["action.right_arm"] = np.ascontiguousarray(a[:, 7:14])
        self.last_confidence = float(confidence)
        return action

    def reset(self):
        self._lang_cache.clear()
        self.last_confidence = 0.0
        self._supervisor = None
        self._supervisor_task = None
        self._instruction = None
        return {"ok": True}

    def supervisor_summary(self) -> dict:
        """Per-episode grasp supervision stats; None when it is disabled."""
        if not self.use_contact_supervisor:
            return None
        s = self._supervisor.summary() if self._supervisor is not None else {}
        s["intervening"] = bool(self._supervisor and self._supervisor.intervening)
        return s


def main():
    os.environ.setdefault("UNIBOT_SUBMISSION_TOKEN", "dev-token")
    p = MuscleMemorySubmission()
    print("metadata:", {k: v for k, v in p.metadata.items() if k != "token"})
    obs = {
        "observation.language": "hang the cup on the marked spot on the cup rack",
        "observation.state.left_arm": np.zeros((4, 7), np.float32),
        "observation.state.right_arm": np.zeros((4, 7), np.float32),
        "observation.state.left_gripper": np.zeros((4, 1), np.float32),
        "observation.state.right_gripper": np.zeros((4, 1), np.float32),
    }
    for k, v in p.get_action(obs).items():
        print(f"  {k}: {v.shape if hasattr(v, 'shape') else v}")
    print("confidence:", p.last_confidence)


if __name__ == "__main__":
    main()