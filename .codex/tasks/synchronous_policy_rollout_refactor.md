# Codex Task Contract: Simplify Policy Deployment to Synchronous Chunk Execution

> **Status:** one-off implementation task contract for Codex. This is intentionally placed under `.codex/tasks/` rather than README/permanent architecture documentation. After the implementation is merged and the stable behavior is reflected in source/README, this task file may be removed or archived with the change history.
>
> **Safety:** this repository controls physical robots. This task is source/offline work only. Do **not** run `examples/run_policy.py`, teleoperation, replay, homing, calibration, device discovery, or any code path that opens a live SDK/device unless the user separately and explicitly authorizes supervised hardware validation.

## 1. Objective

Refactor `dexmani_real` learned-policy deployment from the current asynchronous prediction/scheduling architecture to a **simple synchronous action-chunk execution architecture** faithful to the established mechanisms used by ACT and LeRobot.

The target behavior is:

```text
current observation
      ↓
blocking policy inference
      ↓
PolicySpec.n_action_steps actions
      ↓
action queue: a0, a1, ... aN-1
      ↓
execute one action per policy control tick, in order
      ↓
queue empty
      ↓
observe again and run the next blocking inference
```

The implementation must optimize for a **single-researcher PhD experimental runtime**, not a production policy-serving platform. Keep hardware ownership and physical safety strict; keep policy scheduling small, causal, readable, and easy to debug.

Do not replace the current asynchronous machinery with a new DexMani-specific synchronization protocol. The preferred change is deletion and consolidation.

---

## 2. Reference mechanisms and design authority

The implementation must stay close to the following reference behavior. When uncertain, prefer these mechanisms over inventing a new one.

### 2.1 Primary reference: LeRobot synchronous inference

Source:

- `huggingface/lerobot/src/lerobot/rollout/inference/sync.py`
- https://github.com/huggingface/lerobot/blob/main/src/lerobot/rollout/inference/sync.py

Relevant mechanism:

```text
control thread
    → prepare current observation
    → policy.select_action(...)  # blocking
    → postprocess
    → return one action
```

There is no background inference worker, no request/response protocol, and no prediction IPC in the synchronous backend.

**DexMani consequence:** model inference and policy action scheduling should have one process-local owner. Do not keep an independent inference process merely to preserve the old architecture.

### 2.2 Primary reference: LeRobot ACT action queue

Sources:

- `huggingface/lerobot/src/lerobot/policies/act/modeling_act.py`
- `huggingface/lerobot/src/lerobot/policies/act/configuration_act.py`
- https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/modeling_act.py
- https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/configuration_act.py

Relevant mechanism:

```python
if action_queue is empty:
    actions = predict_action_chunk(observation)[:, :n_action_steps]
    action_queue.extend(actions)
return action_queue.popleft()
```

LeRobot distinguishes:

```text
chunk_size      = how much future the model predicts
n_action_steps  = how many actions are actually executed per policy query
```

**DexMani consequence:** synchronous Real execution must use the Policy-owned `n_action_steps`. Real must not maintain a second replanning horizon.

`dexmani_policy.LoadedPolicy.predict(...)` already exposes exactly the desired public contract:

```text
[n_action_steps, control_action_dim]
```

Use it. Do not use `predict_action_chunk()` in synchronous Real deployment.

### 2.3 Primary reference: original ACT non-temporal-aggregation rollout

Sources:

- `tonyzhaozh/act/imitate_episodes.py`
- https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py
- `tonyzhaozh/aloha/aloha_scripts/real_env.py`
- https://github.com/tonyzhaozh/aloha/blob/main/aloha_scripts/real_env.py

Relevant mechanism when temporal aggregation is disabled:

```text
query policy once
→ execute predicted actions sequentially
→ query again only when the current action sequence is exhausted
```

The real robot path sends non-blocking actuator targets and advances by the control period; it does **not** wait for physical convergence of every action before proceeding.

**DexMani consequence:** synchronous policy inference does not imply a physical arm/hand convergence barrier.

### 2.4 Supporting reference: ManiUniCon synchronized execution

Sources:

- `Universal-Control/ManiUniCon/maniunicon/utils/shared_memory/shared_storage.py`
- `Universal-Control/ManiUniCon/maniunicon/core/robot.py`
- `Universal-Control/ManiUniCon/maniunicon/policies/torch_model.py`
- https://github.com/Universal-Control/ManiUniCon

The useful semantic is not the exact `robot_ready` / `policy_ready` Event implementation. The useful semantic is:

```text
complete current synchronized action sequence
→ infer the next sequence
→ re-anchor execution timing to the current/post-inference time
→ execute the full sequence
```

**DexMani consequence:** after a blocking inference, the new action sequence begins on a fresh execution clock. Do not interpret the source observation timestamp as an absolute deadline for action 0.

### 2.5 Supporting reference: LeFranX direct chunk consumption

Source:

- `wengmister/LeFranX/scripts/dual_robot/dual_robot_deploy_dp.py`
- https://github.com/wengmister/LeFranX

Useful behavior:

```text
action chunk empty/exhausted → infer
otherwise → consume next chunk action
```

Use this only as supporting evidence for the simple queue semantics. Do not copy its hardware synchronization implementation.

### 2.6 Explicit counterexample: diffusion_policy async real-world execution

Source:

- `real-stanford/diffusion_policy/eval_real_robot.py`
- `real-stanford/diffusion_policy/diffusion_policy/real_world/real_env.py`

This project intentionally uses asynchronous future timestamps, drops actions that have become stale, and schedules future commands into the robot controller.

**Do not preserve these ideas in the new synchronous DexMani path.** The current `dexmani_real` behavior is conceptually much closer to this asynchronous design; the purpose of this task is to remove that complexity.

---

## 3. Required target architecture

Keep the existing physical worker ownership, but merge model inference and policy scheduling into one process:

```text
                         Main
                  operator / supervisor
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
     PolicyRunner       ArmWorker       HandWorker
     model/CUDA         xArm SDK        XHand SDK
     observation
     blocking infer
     action deque
     IK / SafetyGate
     command publish
          │                ▲                ▲
          └──────── RuntimeChannels ────────┘
                           ▲
                 camera / pointcloud
                           │
                       RecorderIO
```

The process that is currently named `policy` should become the sole Policy/model owner. It may continue to live in `dexmani_real/deployment/executor.py` for this task to minimize file churn; rename `PolicyExecutor` to `PolicyRunner` if doing so is clean, but do not perform unrelated module reshuffling.

The parent/Main process must still avoid importing Torch or constructing the model. Arm and Hand workers must still be the sole live SDK owners.

---

## 4. Synchronous policy contract

The new live policy path must satisfy exactly these four synchronization invariants:

1. **Only one policy action chunk is active at a time.**
2. **A new policy inference occurs only when the current action queue is empty.**
3. **Actions are dispatched in queue order, at most one action per policy control tick.**
4. **There is no stale-prefix skipping, prediction replacement, future-step catch-up, or asynchronous overlapping inference in synchronous mode.**

Do not add extra synchronization invariants unless an existing hardware-safety boundary already requires them.

In particular, do **not** introduce:

```text
robot_ready / policy_ready events
inference request IDs
response IDs
request/response queues
new ACK protocols
consumed_action_id
per-action arm+hand barriers
delivery-grace synchronization
background policy inference threads
future/promise abstractions
```

If the implementation appears to require one of these, reconsider the architecture before adding it.

---

## 5. Policy API and chunk ownership

### Required API

Change the Real-side policy runtime adapter from:

```python
predict_action_chunk(observation)
```

to:

```python
predict(observation)
```

where the required shape is:

```python
(policy_spec.n_action_steps, policy_spec.control_action_dim)
```

Use `dexmani_policy.LoadedPolicy.predict(...)` directly through the existing NumPy adapter.

### Required queue semantics

Use an ordinary process-local `collections.deque` or equivalent minimal container:

```python
actions = deque()

if not actions:
    observation = build_observation(...)
    predicted = runtime.predict(observation)
    validate predicted shape / finite values
    actions.extend(predicted)

action = actions.popleft()
```

Do not create another Real-owned `replan_steps` interpretation.

### Episode boundary

At the start of each formal RUNNING episode:

```text
clear queued actions
reset per-episode policy state / seed through runtime.reset_episode()
reset local chunk/action counters if any are retained only for diagnostics
reset policy control rate anchor
```

At stop, fault, generation change, or episode end:

```text
clear queued actions
never publish a queued action belonging to the old generation
```

A model inference may be blocking. Therefore, after inference returns and **before** accepting the returned actions into the live queue, re-read the current run state/generation. If S/Q/ESC/fault/generation change occurred while the model was running, discard the just-computed chunk and publish nothing from it.

This is not a new synchronization protocol; it is the existing DexMani generation/safety fence applied at the synchronous inference boundary.

---

## 6. Control timing

### 6.1 Reuse the existing `LoopRate`

Do not invent a new scheduler or sleep helper. Use `dexmani_real.utils.rate.LoopRate`.

The policy loop rate is the Policy contract:

```python
policy_hz = 1.0 / float(policy_spec.control_dt_s)
rate = LoopRate(policy_hz, label="policy", busy_wait=False)
```

`runtime.policy.control_hz` currently also serves teleoperation and other runtime code, so do **not** remove it merely because the policy runner can derive its rate from `PolicySpec`. Existing compatibility validation should continue to ensure the two agree.

### 6.2 Re-anchor after blocking inference

A blocking inference is a chunk boundary. After inference successfully returns and the generation is still live, call `rate.reset()` before dispatching the first action from the new chunk.

This gives the synchronized behavior used conceptually by ACT/ManiUniCon:

```text
last action of old chunk
    ↓ one normal policy interval
new observation
    ↓ blocking inference (may be slow)
new chunk ready
    ↓ fresh execution clock
new action 0
    ↓ dt
new action 1
    ↓ dt
...
```

The inference latency is intentionally exposed as a pause at the chunk boundary. Do not compensate for it by dropping actions, advancing an index, or starting inference early.

### 6.3 No catch-up mechanism in policy code

Do not add policy-side logic that:

```text
skips actions because their old logical timestamp passed
publishes multiple actions to catch up
selects a later action index after an overrun
replaces an active chunk with a newer chunk
```

The existing `LoopRate` already handles ordinary rate pacing and re-anchors after sufficiently large missed slots. Do not duplicate that behavior in `PolicyRunner`.

### 6.4 Last action before next inference

The normal loop structure must be:

```text
dispatch final action of chunk
→ rate.wait()
→ next loop iteration sees empty queue
→ build new observation
→ infer next chunk
```

Do not add a `WAIT_CHUNK_END` state. The ordinary control tick already gives the final action its control interval before the next query.

---

## 7. Observation behavior: preserve, do not redesign

This task is about synchronous **policy execution**, not a new sensor alignment algorithm.

Keep the existing causal observation builder in `dexmani_real/deployment/inference/observation.py` unless a minimal signature/import adjustment is required by the process merge.

Preserve the current concepts:

```text
source timestamps
causal state/camera/pointcloud selection
observation history
freshness checks
cross-modal skew checks
current edge-repeat / warm-up behavior
```

It is acceptable for `ObservationBatch` to continue carrying its current logical/provenance timestamp fields. The key change is simply:

> **Do not use observation logical time as the action execution schedule.**

In synchronous mode, observation timestamps describe what the policy saw; execution timing begins after the blocking inference.

Do not use this task to redesign `max_grid_lag_s`, state-history alignment, point-cloud provenance, or camera history selection.

---

## 8. Hardware command semantics: preserve existing safety boundary

The reference implementations do not wait for physical convergence after every policy action. DexMani should not introduce such a barrier.

### Keep

Keep the existing infrastructure that protects physical hardware:

```text
DISARMED / ARMED / RUNNING / FAULT safety state
run_generation invalidation
motion_lock
coupled command publication
final SDK-side ticket/generation/safety fence
E-stop
ArmWorker as sole xArm SDK owner
HandWorker as sole XHand SDK owner
producer SafetyGate
worker-side mechanical/joint limits
worker-side command jump protection
physical HOME lifecycle
```

These are hardware-safety mechanisms, not policy scheduling machinery.

### Remove from the policy path

The new synchronous policy path should not keep the current policy-level `_CommandProgress` synchronization/watchdog machinery merely to preserve behavior that ACT/LeRobot do not require.

Remove policy scheduling dependencies on:

```text
latest_published_action_id progress gating
arm_accepted_action_id progress gating
hand_accepted_action_id progress gating
per-worker progress timestamps
command-progress timeout
first-command timeout
command-silence timeout
```

Do not replace them with a new ACK scheme.

Fresh feedback validation during command preparation and the final worker SDK fence remain sufficient physical safety boundaries for this refactor.

If later hardware measurements demonstrate real command loss caused by the latest-wins coupled ring, handle that as a separate transport task. The existing shared-memory ring already supports exact sequence reads; do not pre-emptively add a new synchronization protocol here.

---

## 9. Startup and lifecycle

Preserve the valuable existing staged-start behavior: the model must load and warm up successfully **before** live hardware workers are started.

Target startup:

```text
spawn PolicyRunner only
    ↓
set policy heartbeat early
lazy import dexmani_policy / Torch inside child
load selected artifact
warmup existing number of samples
mark `policy` READY
    ↓
Main waits for policy READY
    ↓
start arm / hand / camera / pointcloud / recorder workers
    ↓
wait hardware readiness
    ↓
transition ARMED
```

The old independent `inference` process/readiness/heartbeat identity should disappear.

Do not add a background heartbeat thread to survive model inference. Maintain one `policy` heartbeat. If the existing policy heartbeat timeout is demonstrably shorter than legitimate worst-case blocking inference, adjust the existing timeout configuration rather than adding another liveness mechanism.

Preserve B/S/H/Q/ESC behavior. In particular, S/Q/ESC must remain able to revoke motion immediately from Main while PolicyRunner is blocked in inference; generation/final SDK fencing already provides this safety property.

---

## 10. Recording and dry-run behavior

### Recorder

Do not redesign RecorderIO transactions in this task. Keep the existing recorder process, start/stop protocol, raw episode schema, and finalization behavior unless a compile-time dependency on removed async prediction code requires a minimal cleanup.

This task already changes control ownership and timing; mixing a storage protocol redesign into the same patch will make failures hard to attribute.

### Policy trace

The existing `policy_trace.py` is largely an asynchronous scheduler forensic trace (`prediction_sequence`, publish/ingest time, first future index, decision due time, etc.). Do **not** invent a new v2 trace schema as part of this task.

Required approach:

1. remove PolicyRunner's live dependency on async prediction trace semantics;
2. search all offline consumers of `policy_trace.py`;
3. if backward compatibility for old episodes requires the loader/module, keep it as legacy offline code but stop producing new async sidecars;
4. if it has no required consumers, it may be deleted;
5. update README so new synchronous rollout outputs are described truthfully.

Do not replace it with another custom synchronization trace.

### `execute=False`

Preserve validate-only/dry-run behavior. It must use the same synchronous queue and inference semantics, but stop at the existing command publishability/admission boundary rather than writing a physical command.

---

## 11. Required code changes by file/boundary

This section is implementation guidance, not a requirement to maximize file count. Make the smallest coherent vertical change.

### `dexmani_real/deployment/executor.py`

Convert the current asynchronous `PolicyExecutor` into the synchronous policy owner.

Keep useful existing helpers for:

```text
episode lifecycle
action decode
EE IK
joint canonicalization/projection
SafetyGate / prepare_command
publish_command / dry-run admission
recording
operator stop / timeout episode boundaries
```

Move or absorb the model-loading/warmup behavior currently owned by `inference/worker.py` so this process is also the sole model owner.

Replace prediction state with a local action deque.

Delete the asynchronous executor concepts:

```text
read_latest_prediction / prediction_from_record
last_seen_prediction_sequence
active_prediction / active_prediction_sequence
schedule_base_ns
first_future_step_index
stale prediction admission
skipped-prefix logic
prediction replacement
_next_due_action / _advance_prediction semantics based on old logical timestamps
_CommandProgress and policy command-progress watchdogs
first-command / command-silence watchdogs
128 Hz policy executor polling as a scheduling requirement
```

Use one policy-rate `LoopRate` and one action dispatch per control iteration.

Do not rename/split unrelated helpers just for style.

### `dexmani_real/deployment/inference/runtime.py`

Change the protocol to expose:

```python
def predict(self, observation: PolicyObservation) -> np.ndarray: ...
```

Remove the Real deployment dependency on `predict_action_chunk`.

### `dexmani_real/deployment/inference/dexmani_policy.py`

Adapt `predict(...)` directly to `LoadedPolicy.predict(observation.arrays)`.

Expected output shape:

```python
(spec.n_action_steps, spec.control_action_dim)
```

### `dexmani_real/deployment/inference/worker.py`

After model load/warmup ownership has moved into the policy process, this worker should no longer exist. Delete it rather than leaving an unused second implementation.

Keep `inference/observation.py` as the observation module unless a later cleanup justifies a rename.

### `dexmani_real/deployment/prediction.py`

Delete. There is no cross-process `Prediction` object in synchronous mode.

### `dexmani_real/deployment/timing.py`

Delete `first_future_step_index` and any helpers that become unused solely because async prediction scheduling is gone.

Do **not** delete `next_periodic_deadline_ns` blindly: recording currently uses it. Search callers first and keep any timing utility that still has a real owner.

### `dexmani_real/ipc/schema.py`

Remove async prediction wire schema and capacity constants once there are no users:

```text
PREDICTION_DTYPE
MAX_PREDICTION_STEPS
other prediction-only wire constants
```

Do not alter unrelated command/state/record schemas.

### `dexmani_real/ipc/channels.py`

Remove:

```text
prediction_ring
PREDICTION_RING_MAXLEN
prediction shared-memory allocation/cleanup
inference heartbeat slot
inference readiness slot
```

Do not change the coupled command transport in this task.

### `dexmani_real/config/defaults.py`

Remove policy-only asynchronous controls that have no remaining callers, especially:

```text
replan_steps
max_command_silence_s
command_progress_timeout_s
first_command_timeout_s
```

Remove `inference` from heartbeat/readiness subsystem sets when the process no longer exists.

**Do not remove `executor_poll_hz` globally without checking callers.** It is currently also used by teleoperation and therefore is not purely an async policy parameter. The synchronous policy runner should stop using it, but teleop may continue to own it.

**Do not remove `control_hz` globally.** It is shared with teleoperation/runtime behavior; keep compatibility validation against `PolicySpec.control_dt_s`.

Keep hardware/replay parameters such as `action_apply_timeout_s` if they have non-policy users.

### `dexmani_real/deployment/config.py`

Remove validation tied to prediction IPC capacity and Real-owned replanning:

```text
policy_spec.chunk_size <= MAX_PREDICTION_STEPS
runtime.policy.replan_steps <= policy_spec.chunk_size
```

Keep actual Real/Policy compatibility checks:

```text
action representation/dimension
required hand contract
observation modalities/semantics
control_dt_s compatibility
point-cloud shape/semantics
```

`InferenceWorkerConfig` becomes a stale name once there is no inference worker. Rename it to a small neutral model/runtime configuration such as `PolicyRuntimeConfig` if the rename remains localized and coherent. Do not create an abstraction hierarchy around it.

### `dexmani_real/deployment/lifecycle.py`

Build exactly one policy/model process.

The `policy` ProcessSpec should own model startup and should have `ready_name="policy"`.

Preserve staged startup by starting the policy process first, waiting for it to become ready, then starting hardware/recording workers.

Remove all `inference` ProcessSpec handling and provenance fields that refer to a Real-owned replan interval.

Recorder provenance may record Policy-owned `n_action_steps` if useful, but do not create a new scheduling configuration field.

### `dexmani_real/deployment/metrics.py`

Remove metrics whose meaning only exists under the async scheduler:

```text
skipped_prefix_steps
stale_prediction_count
prediction schedule lateness if it no longer has a clear synchronous meaning
command-progress timeout count if the watchdog is removed
```

Keep small research-useful measurements that still correspond to real quantities:

```text
inference latency
observation age/skew
actual publication/control interval
IK rejection count
safety rejection count
```

Do not build a new metrics framework.

### `examples/run_policy.py`

Remove:

```text
--replan-steps
replan_steps from run_config.yaml
replan interval from operator summary
```

Expose/report the Policy-owned contract instead, e.g. `n_action_steps`, without adding a new override.

Update the top-level usage string accordingly.

### `README.md`

After source behavior is correct, update only stable user-facing statements:

```text
policy deployment is synchronous
one Policy-owned n_action_steps chunk is consumed before the next inference
there is no --replan-steps option
new rollout outputs no longer promise async policy-trace sidecars if production is removed
```

Do not put migration history or this implementation plan into README.

---

## 12. Canonical synchronous loop shape

The final active path should be conceptually no more complicated than this:

```python
from collections import deque

class PolicyRunner:
    def __init__(...):
        self.actions = deque()
        self.runtime = ...
        self.rate = LoopRate(
            1.0 / float(policy_spec.control_dt_s),
            label="policy",
            busy_wait=False,
        )

    def start_episode(self, generation):
        self.actions.clear()
        self.runtime.reset_episode()
        self.rate.reset()
        ... existing recording / lifecycle setup ...

    def run_active_tick(self):
        snapshot = read_run_state_snapshot(self.shared)
        if snapshot is not the live RUNNING generation:
            self.actions.clear()
            return

        inference_ms = None
        if not self.actions:
            observation = build_current_causal_observation(...)
            if observation is None:
                self.rate.wait()
                return

            started = time.monotonic_ns()
            predicted = self.runtime.predict(observation)
            inference_ms = (time.monotonic_ns() - started) / 1e6
            validate shape == (spec.n_action_steps, spec.control_action_dim)
            validate finite values

            # S/Q/ESC/fault may have happened while predict() was blocking.
            snapshot_after = read_run_state_snapshot(self.shared)
            if snapshot_after is not the same live RUNNING generation:
                self.actions.clear()
                return

            self.actions.extend(predicted)
            self.rate.reset()  # new chunk executes from post-inference current time

        raw_action = self.actions.popleft()
        decoded = decode / project / prepare with existing safety boundary
        publish or dry-run through existing command boundary
        record existing raw episode row
        self.rate.wait()
```

This pseudocode is a semantic target, not a demand to duplicate logic or ignore current helper ownership.

If the real implementation requires substantially more synchronization state than this, stop and reassess before adding mechanisms.

---

## 13. Explicit non-goals

Do not do the following as part of this task:

```text
implement temporal ensembling in Real
implement real-time chunking / overlap
implement inference prefetch
implement diffusion-policy-style timestamp scheduling
redesign camera/pointcloud history selection
redesign the coupled command ring
introduce FIFO command ACK protocols
redesign ArmWorker or HandWorker SDK ownership
redesign RecorderIO protocol
redesign safety-state transitions
relax joint/mechanical/workspace safety to make rollout easier
change calibration or home behavior
add generalized plugin/backend abstractions
add configuration switches for sync-vs-async behavior unless explicitly requested
retain both old async and new sync paths behind a flag
```

There should be one clear policy deployment mechanism after this change: synchronous.

Do not preserve dead async code “just in case”. Git history is the fallback.

---

## 14. Migration order

Implement in this order so each step has one clear ownership change:

### Phase A — establish synchronous model API

1. Change the Real policy adapter/protocol to `predict()` / `n_action_steps`.
2. Add local finite/shape validation at the model boundary.
3. Do not yet change hardware safety code.

### Phase B — merge process ownership

1. Move model load/warmup/reset/close into the policy process.
2. Give the policy process readiness ownership.
3. Preserve policy-first staged startup.
4. Remove the inference worker from lifecycle.

### Phase C — replace async scheduling with the queue

1. Add process-local action deque.
2. Query only when empty.
3. Re-check generation after blocking inference.
4. `rate.reset()` after a successful inference.
5. Dispatch one action per tick.
6. Remove prediction replacement/stale-prefix/timestamp execution logic.

### Phase D — remove obsolete infrastructure

1. Remove prediction IPC/schema/ring.
2. Remove prediction model/dataclass.
3. Remove policy async watchdog/progress state.
4. Remove `replan_steps` CLI/config/provenance.
5. Remove async-only metrics.
6. Remove inference heartbeat/readiness.

### Phase E — documentation/legacy cleanup

1. Deal with `policy_trace.py` only according to the legacy-consumer rule above; do not invent a replacement schema.
2. Update README to current stable behavior.
3. Run global searches for stale “async inference”, “prediction ring”, “replan”, and removed symbols.

---

## 15. Required correctness cases to reason through

Before considering the task complete, inspect code paths for each case and make them explicit in the final handoff.

### Normal episode

```text
H / valid start pose
B
→ RUNNING
→ observe
→ blocking predict N actions
→ dispatch a0 ... aN-1 at policy rate
→ observe again
→ repeat
S
→ motion fenced
→ episode finalizes
```

### S/Q during blocking inference

Required result:

```text
Main immediately revokes generation/motion
predict() eventually returns
PolicyRunner rechecks run state
computed chunk is discarded
no action from that chunk reaches publish_command
```

Do not solve this with thread cancellation.

### ESC / hardware fault during blocking inference

Same requirement: generation/safety state makes the returned chunk unusable; final SDK fences remain authoritative.

### Observation temporarily unavailable

Required behavior:

```text
no fabricated action
no stale prediction fallback
no async previous-chunk reuse after queue is empty
wait/poll on the normal policy loop and retry observation
```

### Slow inference

Required behavior:

```text
chunk boundary pauses
no prefix is dropped
no future index is selected
rate is re-anchored after inference
full predicted n_action_steps sequence is then consumed in order
```

### Multi-episode session

The model remains loaded, but every formal new episode resets policy episode state and starts with an empty action queue.

### `execute=False`

The same synchronous inference/queue order is used; physical publication remains disabled through the existing validation boundary.

---

## 16. Offline validation only

Follow repository `AGENTS.md`. Before editing a worktree, inspect `git status --short` and preserve unrelated changes.

At minimum run:

```bash
python -m compileall -q dexmani_real examples
git diff --check
git status --short
```

Also run focused import/static checks that already exist and are safe. Do not execute examples as tests.

Useful final repository searches should show no live synchronous-policy references to obsolete async constructs, for example:

```bash
rg "PREDICTION_DTYPE|prediction_ring|first_future_step_index|active_prediction|last_seen_prediction_sequence" dexmani_real examples
rg "replan_steps|--replan-steps" dexmani_real examples README.md
rg "inference_loop|ready_name=\"inference\"|set_heartbeat\(\"inference\"" dexmani_real examples
```

Any remaining match must be either:

```text
an intentional legacy offline reader
an unrelated historical-compatible symbol with a documented reason
```

—not a live policy execution dependency.

Do not claim CUDA/model runtime validation unless an appropriate local device/model was actually exercised. Do not claim hardware validation unless real hardware was explicitly authorized and tested.

---

## 17. Acceptance criteria

The task is complete only when all of the following are true:

### Architecture

- Exactly one child process owns Policy model/CUDA state and learned-policy scheduling.
- There is no independent inference worker in the live deployment lifecycle.
- There is no prediction IPC ring/schema in the live runtime.
- Arm/Hand SDK ownership remains unchanged.

### Policy semantics

- Real calls the Policy public `predict()` API.
- Each query yields exactly `PolicySpec.n_action_steps` control actions.
- Queue exhaustion is the only normal trigger for the next policy query.
- One action is dispatched per policy control tick.
- No active chunk can be replaced by a newer prediction.
- No stale prefix is skipped.
- No action is selected by comparing an old observation timestamp with current wall-clock execution time.

### Timing

- Policy dispatch rate comes from `PolicySpec.control_dt_s` and uses existing `LoopRate`.
- The rate is re-anchored after blocking inference before the new chunk begins.
- Slow inference produces a visible chunk-boundary pause rather than async compensation.

### Safety

- Existing generation fencing remains intact.
- Existing final worker SDK permit checks remain intact.
- S/Q/ESC during inference cannot result in a late command from the completed inference.
- Existing joint/mechanical/workspace safety checks are not weakened.

### Simplicity

- No new Events/Queues/ACK protocol was added for policy synchronization.
- No sync/async feature flag keeps both architectures alive.
- `replan_steps` is removed from learned-policy deployment.
- Async-only prediction scheduling state and watchdogs are removed rather than renamed.

### Documentation

- `examples/run_policy.py` and README describe the synchronous mechanism truthfully.
- Stable docs describe current behavior only; this task plan itself is not copied into README.

---

## 18. Final Codex handoff requirements

In the final implementation handoff, report succinctly:

```text
1. which async components were removed;
2. where model ownership moved;
3. the final synchronous action-queue semantics;
4. how S/Q/ESC during inference is fenced;
5. which safety boundaries were intentionally preserved;
6. which offline checks passed;
7. what was not tested (CUDA / model artifact / real hardware).
```

Do not claim completion based only on compilation. Inspect the final diff and confirm that no unnecessary compatibility layer or speculative synchronization abstraction remains.

The desired result should read like a research control loop, not a policy-serving framework:

```text
observe → infer → queue → dispatch at dt → queue empty → observe → infer
```

with DexMani's existing hardware safety boundary around command publication and SDK execution.