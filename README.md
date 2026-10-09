# UniBot-V1 Submission — muscle-memory-17t

> **This repository is a submission built on the official UniBot-V1 template.**
> Sections 1–6 below are the template's specification, unmodified, describing the
> policy interface, data formats and workflow. They are reproduced from
> [`unitreerobotics/unibot_submission`](https://github.com/unitreerobotics/unibot_submission).
>
> **Our policy, its training data and results are documented in [MODEL.md](MODEL.md).**
> In brief: a looped-transformer motor policy (800,513 params), trained by
> behavior cloning on 17 of the 32 challenge task datasets, served one process
> per robot port via `run_server.py`. It is **proprioception + language only —
> no vision** — so expect low success rate.
>
> ```bash
> pip install -r requirements.txt
> UNIBOT_SUBMISSION_TOKEN=<token> UNIBOT_CONTROL_SPACE=joint python run_server.py <N>
> ```
>
> ---

# UniBot-V1 Challenge — Submission Template

> 中文版见 [README.zh.md](README.zh.md).

This document specifies the policy interface a submission must implement, the
data format exchanged across that interface, and a bundled reference
implementation.

---

## 1 · Policy interface

A submission consists of one policy class that must expose the three members
below. The class is instantiated on the participant's machine and served by
`PolicyService`; the evaluator connects as a client and invokes it through this
interface.

| Member | Kind | Purpose |
|---|---|---|
| `metadata`        | property → `dict` | Policy config: control space, observation keys and their temporal stacking, image resize, action chunk length, submission token. |
| `get_action(obs)` | method            | Produce one action chunk for the given observation. |
| `reset()`         | method            | Clear per-episode state. Called once at the start of each episode. |

### `metadata` schema

```python
{
    "control_space":     str,             # "joint" or "ee" — the action space (§3)
    "obs_delta_indices": dict[str, list], # selected observation key -> per-frame offsets (§2)
    "image_resize":      dict[str, list], # image key -> [height, width] to resize to (§2)
    "action_chunk_size": int >= 1,        # leading time-axis length of every action (§3)
    "token":             str,             # submission token issued by the organisers (§4)
}
```

The evaluator reads `metadata` according to the per-field conventions described
below. Fill in each field as required; values that do not follow the
requirements may cause interaction errors.

- **`control_space`** — the action space the policy operates in, either
  `"joint"` or `"ee"`; the two are **mutually exclusive**. It determines which
  pair of dual-arm action keys `get_action` must return (§3).
- **`obs_delta_indices`** — does two things at once: **its keys decide which
  observations are sent** (only declared keys are transmitted), and **its value
  is a list of frame offsets declaring that key's temporal stacking** (§2).
- **`image_resize`** — the resolution `[height, width]` of each image in the
  returned observation (§2).
- **`action_chunk_size`** — an `int` ≥ 1: the leading time-axis length of every
  action array returned by `get_action` (§3).
- **`token`** — the submission token issued by the organisers, checked once at
  handshake (§4).

A minimal implementation is in [`example/example_policy.py`](example/example_policy.py).

---

## 2 · Observation format

The observation (`obs`) is sent by the evaluator to the policy over the network
for it to act on. `obs` is a `dict`; the range of available keys, along with
each one's per-frame shape, dtype, and description, is below.

### Observation catalog

| Key | Per-frame shape | dtype | Description |
|---|---|---|---|
| `observation.images.cam_left_high`   | `[480, 640, 3]` | `numpy.uint8`   | Upper-left camera, HWC, RGB |
| `observation.images.cam_right_high`  | `[480, 640, 3]` | `numpy.uint8`   | Upper-right camera, HWC, RGB |
| `observation.images.cam_left_wrist`  | `[480, 640, 3]` | `numpy.uint8`   | Left wrist camera, HWC, RGB |
| `observation.images.cam_right_wrist` | `[480, 640, 3]` | `numpy.uint8`   | Right wrist camera, HWC, RGB |
| `observation.state.left_arm`         | `[7]`           | `numpy.float32` | Left arm joint state |
| `observation.state.right_arm`        | `[7]`           | `numpy.float32` | Right arm joint state |
| `observation.state.left_ee_pose_gripper_base`  | `[6]` | `numpy.float32` | Left gripper end-effector pose in base frame: `xyz(3)` + `rpy(3)` |
| `observation.state.right_ee_pose_gripper_base` | `[6]` | `numpy.float32` | Right gripper end-effector pose in base frame: `xyz(3)` + `rpy(3)` |
| `observation.state.left_gripper`     | `[1]`           | `numpy.float32` | Left gripper opening |
| `observation.state.right_gripper`    | `[1]`           | `numpy.float32` | Right gripper opening |
| `observation.state.lower_body`       | `[15]`          | `numpy.float32` | Lower-body proprioception |
| `observation.language`               | scalar          | `str`           | Natural-language task instruction for the current episode |

**Note:** `observation.language` is a single `str` describing the current
episode's task (e.g. `"move the block to the target position."`), unchanged
throughout and carrying no time axis; the temporal-stacking and shape
conventions below do not apply to `observation.language`.

Metadata fields related to `obs`:

- **`obs_delta_indices`**: decides which keys are sent and how each key is
  temporally stacked;
- **`image_resize`**: decides each image's resolution.

### `obs_delta_indices` — observation selection and temporal stacking

`obs_delta_indices` decides two things at once:

1. **Which observations to select**: only the declared keys are transmitted, so
   declare only the minimal subset the model needs;
2. **How to stack frames**: each value is a list of **frame offsets** relative to
   the current step.

Offset meaning: `0` is the current step and negatives are some number of steps
in the past; the list is ordered oldest-to-newest, and its length is the number
of stacked frames. For example:

```python
"observation.state.left_arm": [-10, -5, -2, 0]  # t-10, t-5, t-2, t — 4 frames
```

Under this config, the array for that key in the returned `obs` has shape
`(4, 7)`, i.e. `(stacked_frames, …per-frame shape…)`; keep the length-`1`
time-axis list even when no stacking is needed. Offsets must be `int`s **≤ 0**
and the index list strictly increasing; when history is insufficient, missing
frames are padded with the first frame. `observation.language` carries no time
axis, so its offset list must be exactly `[0]`.

### `image_resize` — image resolution

The native image resolution is `[480, 640]` (height, width), consistent with the
open-source data. `image_resize` may assign a target `[height, width]` to an
image key that needs resizing; the evaluator resizes it before sending, so the
policy receives it directly at the target resolution. For example:

```python
{"observation.images.cam_left_high": [240, 320]}
```

The value is `[height, width]` (positive ints); the evaluator resizes with
`cv2.resize(frame, (width, height))`, so a resized frame becomes
`[height, width, 3]`. The key must already appear in `obs_delta_indices`. Keys
not listed, or an empty `image_resize` (`{}`), keep the native resolution.

---

## 3 · Action format

Actions are returned by the policy to the evaluator over the network for it to
execute. `get_action` returns a `dict`; the range of all keys, along with each
one's per-frame shape, dtype, and description, is below.

### Action catalog

| Key | Per-frame shape | dtype | Control space | Description |
|---|---|---|---|---|
| `meta.token`           | scalar | `str`           | shared | Token issued by the organisers for this submission (§4) |
| `action.left_gripper`  | `[1]`  | `numpy.float32` | shared | Target left gripper command |
| `action.right_gripper` | `[1]`  | `numpy.float32` | shared | Target right gripper command |
| `action.pivot`         | `[7]`  | `numpy.float32` | shared | Lower-body / pivot target |
| `action.left_arm`  | `[7]` | `numpy.float32` | `joint` | Target left arm joint position |
| `action.right_arm` | `[7]` | `numpy.float32` | `joint` | Target right arm joint position |
| `action.left_ee_pose_gripper_base`  | `[6]` | `numpy.float32` | `ee` | Target left gripper end-effector pose in base frame: `xyz(3)` + `rpy(3)` |
| `action.right_ee_pose_gripper_base` | `[6]` | `numpy.float32` | `ee` | Target right gripper end-effector pose in base frame: `xyz(3)` + `rpy(3)` |

**Note:** `meta.token` is a single `str` (§4) and carries no time axis; the
action-step and shape conventions below do not apply to `meta.token`.

Metadata fields related to actions:

- **`control_space`**: the control mode;
- **`action_chunk_size`**: the number of action steps returned per `get_action`
  call.

### `control_space` — control mode

Either `"joint"` or `"ee"`; the two are **mutually exclusive**. Both modes must
return the "shared" keys in the table, plus the corresponding dual-arm keys for
the chosen mode:

- **`"joint"`**: `action.left_arm`, `action.right_arm`;
- **`"ee"`**: `action.left_ee_pose_gripper_base`, `action.right_ee_pose_gripper_base`.

### `action_chunk_size` — number of action steps

`action_chunk_size` (denoted `T_a`, an `int` ≥ 1) is the number of action steps
returned each time. Each action array has shape `(T_a, …per-frame shape…)`;
index `0` is executed first, then the rest in order, until the evaluator calls
`get_action` again. Keep the length-`1` time axis even when returning a single
step.

---

## 4 · Submission token

Each accepted submission is issued a unique token by the organisers. The policy
must carry this token in **two places**:

1. the `token` field of `metadata`, checked once at handshake;
2. the reserved key `meta.token` in every `get_action` return value, checked at
   each step.

The evaluator rejects any handshake or action whose token is missing or whose
value does not match the issued token.

The reference implementation reads the token from the `UNIBOT_SUBMISSION_TOKEN`
environment variable:

```python
import os
import numpy as np

class MyPolicy:
    def __init__(self):
        self._token = os.environ["UNIBOT_SUBMISSION_TOKEN"]

    @property
    def metadata(self):
        return {
            "control_space":     "joint",   # or "ee"
            "obs_delta_indices": {               # observation key -> per-frame offsets (§2)
                "observation.language":       [0],
                "observation.images.cam_left_high": [0],
                "observation.state.left_arm": [-4, -2, 0],
                # ... remaining observation keys the model consumes
            },
            "image_resize": {                # image key -> [height, width] (§2); may be empty
                "observation.images.cam_left_high": [240, 320],
            },
            "action_chunk_size": 8,
            "token":             self._token,
        }

    def get_action(self, obs):
        T = 8
        return {
            "meta.token":      self._token,
            "action.left_arm": np.zeros((T, 7), dtype=np.float32),
            # ... remaining action.* keys for the declared control_space, shapes per §3
        }
```

---

## 5 · Reference implementation & submission

`example/` provides one runnable reference implementation, offered only as an
example. Participants just follow the `run_server.py` pattern: wrap their own
policy object in `PolicyService` and have it listen on a designated port to
serve it, and the evaluator can connect.

| File | Role |
|---|---|
| [`example/example_policy.py`](example/example_policy.py) | A minimal runnable policy exposing the §1 interface; currently returns an all-zero action chunk. Participants can build on it by swapping in their own model, or implement their own class against the same interface. |
| [`example/example_env.py`](example/example_env.py)       | Local evaluation stub: produces observations per the §2 spec and validates returned actions with the same checks as the production evaluator. |
| [`example/run_server.py`](example/run_server.py)         | **Server-side entry template** — every submission serves its policy this way: hand the policy object to `PolicyService` and listen on a port. `UNIBOT_SUBMISSION_TOKEN` and `UNIBOT_CONTROL_SPACE` (`joint`/`ee`) must both be set. |
| [`example/run_client.py`](example/run_client.py)         | Local end-to-end verification tool that emulates the evaluator connecting to the participant's server and validates the returned actions step by step. In production it is run by the organisers; `UNIBOT_SUBMISSION_TOKEN` must match the server's token. |
| [`policy/`](policy)                                       | `PolicyService` / `RemotePolicy` and the msgpack-NumPy adapter. |
| [`requirements.txt`](requirements.txt)                   | Dependencies for the reference implementation: `numpy`, `msgpack`, `websockets`, `Pillow`. |

### Environment

Install the dependencies into the Python environment used for this submission:

```bash
pip install -r requirements.txt   # numpy, msgpack, websockets, Pillow
```

### Workflow

Local verification is done uniformly through `run_server.py` + `run_client.py`:
one terminal starts the server, another connects as a client emulating the
evaluator. The token must match on both ends.

1. **Run the reference implementation first.** Do a full run with the unmodified
   `ExamplePolicy` and confirm the end-to-end output is normal (the client
   prints `token verified` each step, with no `ACTION REJECTED`), to understand
   the full evaluation flow:
   ```bash
   # Terminal A
   UNIBOT_SUBMISSION_TOKEN=dev-token UNIBOT_CONTROL_SPACE=joint python example/run_server.py 8765

   # Terminal B
   UNIBOT_SUBMISSION_TOKEN=dev-token python example/run_client.py
   ```
2. **Integrate your own model.** Implement the policy class against the §1
   interface and replace the `ExamplePolicy` instantiated in `run_server.py`;
   replace the zero vectors in `get_action` with your model's outputs, and
   adjust `control_space`, `obs_delta_indices`, `image_resize`, and
   `action_chunk_size` as needed.
3. **Run again to confirm the integration.** Repeat the two-terminal commands
   from step 1 and confirm that, with your own model integrated, the client
   still passes validation step by step — this indicates the interface is wired
   up correctly.
4. **Deploy for real.** Write the token issued by the organisers into
   `UNIBOT_SUBMISSION_TOKEN`, start the server as in `run_server.py` and keep it
   online; the evaluator will connect on its own.

   Evaluation runs across **N test robots** in randomized order (the exact N is
   announced separately by the organisers), and the evaluator connects to one
   fixed port per robot for interaction. A submission must therefore expose
   **N consecutive ports** starting at `8765` (`8765`, `8766`, …, `8765 + N − 1`);
   **the number of exposed ports must exactly match the number of robots N**, and
   each port must independently and reliably complete the entire interaction flow
   of its robot. How each port is implemented behind the scenes is left to the
   participant — two common approaches:

   **Approach 1 · One independent server per port (most robust).** Each of the N
   ports runs its own process, each loading its own copy of the model; they are
   isolated and simplest to implement, at the cost of N copies of the model in
   GPU / host memory (expensive for heavy models). For **N = 3**, expose
   `8765`–`8767`, one instance each:

   ```bash
   UNIBOT_SUBMISSION_TOKEN=<token> UNIBOT_CONTROL_SPACE=<joint|ee> python example/run_server.py 8765
   UNIBOT_SUBMISSION_TOKEN=<token> UNIBOT_CONTROL_SPACE=<joint|ee> python example/run_server.py 8766
   UNIBOT_SUBMISSION_TOKEN=<token> UNIBOT_CONTROL_SPACE=<joint|ee> python example/run_server.py 8767
   ```

   The trailing number is the port that instance listens on. These instances may
   be deployed on one machine or spread across several, but all N ports must be
   reachable from outside.

   **Approach 2 · Share one model across ports (resource-saving).** When the
   model is heavy and GPU / host memory can't hold N copies, several ports can
   share one model, but the participant must implement request dispatch on the
   server side (N lightweight endpoints each listen on one port and forward
   requests to a shared inference backend). Two things to watch here:

   - **Concurrency:** different ports may be called at the same time, so the
     shared backend must handle requests concurrently, avoiding mutual blocking
     and inflated latency;
   - **State isolation:** each port corresponds to a separate episode on a
     separate robot, so each port's `reset` and history state must be kept
     mutually isolated and never shared.

---

## 6 · Interaction quality and result credibility

Throughout evaluation the policy is invoked step by step over the network, and
the latency and content of each `get_action` response are recorded. Submissions
should keep interaction latency smooth and stable, and keep returned results as
free of unnecessary anomalies as possible.

When assessing the credibility of a result, the organisers may combine the
various interaction data collected during evaluation, including per-step
response latency and its variability, as well as the statistical regularity of
the returned actions, and these metrics may be displayed alongside the published
result. Participants are accordingly advised to keep per-step inference latency
stable, to ensure returned actions are well-formed and free of anomalous
outliers, and to avoid any behaviour that leads to irregular interaction
patterns.
