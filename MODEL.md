# muscle-memory-17t — UniBot-V1 submission

Looped-transformer motor policy for the UniBot World Challenge, trained by
behavior cloning across **17 of the 32 challenge task datasets**.

The evaluation harness and policy interface in this repository are the official
template from [`unitreerobotics/unibot_submission`](https://github.com/unitreerobotics/unibot_submission);
everything under `policy/muscle_memory*.py`, `run_server.py`, `checkpoints/`
and `MODEL.md` is this submission. The template's spec, examples and validator
are unmodified.

## Honest capability statement

**This policy is proprioception + language only. It has no vision branch.**

It cannot observe where the objects are. The challenge deliberately perturbs
object poses and removes objects (each task's description lists these as
generalization axes), so this policy is expected to:

- **score near zero on success rate**, and
- **bank partial step-score credit** by replaying trajectory fragments for
  object layouts resembling its training data.

It is a genuine multi-task policy — instruction conditioning demonstrably
changes the output — but its generalization ceiling is set by the missing
camera input, not by model capacity.

## Model

| | |
|---|---|
| Architecture | One transformer block looped 4× (`num_loops`), with a learned per-loop embedding |
| Parameters | 800,513 |
| Inputs | 12 hashed language tokens + 4 stacked proprio frames × 16 dims |
| Output | action chunk of 8 steps × 16 dims + scalar confidence gate |
| Control space | `joint` (`ee` is not implemented) |
| Runtime | CPU, ~3–5 ms per action chunk, deterministic (eval mode, no sampling) |

Token sequence: `[ACT] [LANG×12] [PROPRIO×4]`, where the `[ACT]` query token is
read out through the action and confidence heads.

## Training data

Pooled from 17 `unitreerobotics/G1_Dex1_*` task datasets:

`ArrangeTestTubes`, `ConveyorSorting`, `DisconnectEthernet`, `FruitCollection`,
`FruitSorting`, `HangCup`, `OrganizeBookshelf`, `PackBag`, `PaperCupStacking`,
`PickOutBatteries`, `PlacePhotoFrame`, `PlugCharger`, `PlugInEthernet`,
`StackBowls`, `SyringeWaterTransfer`, `Unlock`, `ZipUp`

- 17,320 episodes, 6,716,092 frames
- Each dataset ships ~5 paraphrases of its instruction; the organizers' own task
  descriptions were mixed in, giving 9–11 phrasings per task, sampled randomly
  during training so the language conditioning is not tied to one phrasing.
- Split is **episode-level and stratified per task** (8% val), so no episode
  appears in both halves.

### Normalization

A single **global z-score**, pooled across tasks via the law of total variance
from each dataset's `stats.json` weighted by frame count. Per-dataset
normalization is not usable here: the evaluator supplies a sentence, not a task
id, so the transform must be task-independent.

`action.pivot` is emitted as the pooled dataset mean, **not** zeros — the demos
are not zero-mean (dims 4 and 6 sit near 0.15 / 0.74), and commanding zeros
would place the robot in a posture no demo ever used.

## Result

| Metric | Value |
|---|---|
| Validation Huber (z-scored) | 0.0178 |
| Validation MAE (raw joint units) | 0.0713 rad (~4.1°) |
| Per-task MAE spread | 0.060 – 0.087 across all 17 tasks |

Loss on normalized targets; MAE is in raw joint units. The per-task spread is
flat, i.e. no task collapsed — the shared policy fit all 17 rather than
specializing to a few.

## Running it

Requires the submission token issued by the organizers:

```bash
pip install -r requirements.txt

# one process per robot port, N = number of test robots, ports 8765..8765+N-1
UNIBOT_SUBMISSION_TOKEN=<token> UNIBOT_CONTROL_SPACE=joint python run_server.py 3
```

Local end-to-end check against the bundled evaluator validator:

```bash
# terminal A
UNIBOT_SUBMISSION_TOKEN=dev-token UNIBOT_CONTROL_SPACE=joint python run_server.py 1
# terminal B
UNIBOT_SUBMISSION_TOKEN=dev-token python example/run_client.py
```

Verified: all steps pass validation on ports 8765/8766/8767 served by a single
`run_server.py 3`, mean latency 3.2–4.6 ms per chunk; a mismatched token is
rejected with `invalid meta.token`.

Each port is a separate process holding its own model copy, so episode state
(`reset`) and the language cache are isolated per robot by construction.

## Layout

```
policy/
  web_policy.py            template websocket transport (unmodified)
  muscle_memory.py         model definition (TransformerBlock inlined)
  muscle_memory_policy.py  the metadata / get_action / reset adapter
run_server.py              serves N consecutive ports
checkpoints/
  muscle_memory_challenge.pt   800k-param weights + pooled normalization stats
example/                   template env / validator (unmodified)
```

## What would come next

1. **Vision.** Train on the `videos/` trees (~796 GB for all 32 tasks). The
   patch-token path and `use_vision` branch already exist in the source model;
   they need frame decoding and retraining. This is the single highest-value
   change.
2. The remaining 15 task datasets (~9.6 GB of proprio) for full 32-task coverage.
3. `ee` control space (needs forward kinematics on the G1 URDF).
4. Confidence-gated replanning, executing fewer chunk steps when the gate is low.