# Codex Task — Simplify DexMani Real into a Reliable PhD Real-Robot Experiment Endpoint

## 0. Read this first

This file is the execution task for the local Codex CLI.

Before editing anything:

1. Read the repository-root `AGENTS.md` in full and obey it.
2. Run:
   ```bash
   git status --short
   git rev-parse HEAD
   ```
3. Preserve unrelated local changes. Do not reset, checkout, stash, or overwrite user work.
4. Inspect the actual current code before applying any instruction below. The design in this task was prepared against:
   - `dexmani_real`: `4bba54d9094291b7b8030464c7983ac93971ce6b`
   - LeRobot: `8352ca62d88ab4464001d427267a6a0758e6cb99`
   - ManiUniCon: `85c6f2e32ecf9f2bed62d202b058c39623444686`
   - LeFranX: `a39906e6629f39490950fe8bd20f4f992ed74fd7`
5. If local HEAD differs, re-trace the affected boundaries from actual entry points instead of blindly applying old line-level assumptions.
6. Do **not** run any hardware-affecting program or SDK path. Do not connect/discover xArm, XHand, RealSense, VR hardware, home, replay, teleoperate, calibrate, or rollout. Offline checks only.
7. Do **not** create a committed `tests/` directory. Use focused one-off offline smoke checks for pure logic.
8. Keep this file at the repository root. Do not delete it as part of the refactor.

The goal is not a framework rewrite. The goal is to make the current research repository smaller, easier to reason about, and more reliable for real experiments.

---

## 1. Repository mission

`dexmani_real` is a personal PhD research repository for exactly two primary workflows:

### A. Real-robot dexterous data collection

```text
Quest/VR human motion
        ↓
teleop mapping + hand retargeting
        ↓
IK / projection / safety
        ↓
xArm7 + XHand
        ↓
timestamped robot/sensor data
        ↓
raw research episode
```

### B. Real-robot evaluation of learned policies trained in other repositories

```text
external trained model
        ↓
model-specific loading / PolicySpec
        ↓
causal multimodal observation
        ↓
synchronous inference + action chunk
        ↓
IK / projection / safety
        ↓
xArm7 + XHand
        ↓
raw evaluation episode + technical metrics
```

This repository is **not**:

- a generic robotics framework;
- a multi-robot plugin platform;
- production serving infrastructure;
- an async inference platform;
- an RTC implementation;
- a policy training framework;
- a generic dataset framework;
- a compatibility layer for arbitrary hypothetical models.

Use this priority order:

> physical safety > experiment correctness > iteration speed > readability > generic extensibility > enterprise robustness

Prefer:

> delete > inline > merge > rewrite > add abstraction

Only preserve complexity that protects a real physical guarantee, a real scientific-data guarantee, or a current experiment requirement.

---

## 2. Reference-project lessons: copy the principles, not the infrastructure

### LeRobot

Adopt:

- clear separation between recording and policy rollout;
- simple conceptual robot boundary: observe / send action;
- explicit action representation at the physical boundary;
- simple operator-facing workflows.

Do not copy:

- generic rollout strategies;
- async inference abstractions;
- plugin registries;
- processor framework complexity;
- broad robot ecosystem abstractions.

Important: current LeRobot recording saves the processed teleop action rather than necessarily the value returned by `robot.send_action()`. Do **not** copy that semantic for DexMani. DexMani action labels must represent an executable high-level target that the real actuator workers actually adopted.

### ManiUniCon

Adopt:

- sensor / policy / robot process separation where it corresponds to real resource ownership;
- shared-memory transport for high-rate data;
- teleop and learned policy as action producers above robot execution.

Do not copy:

- target-timestamp action queues;
- generic future-action buffering;
- Hydra plugin architecture;
- generic control-mode machinery not needed by the current embodiment.

### LeFranX

Adopt:

- simple dexterous collection workflow;
- combined arm + hand operator experience;
- human-friendly home / begin / save / discard loop.

Do not copy:

- serial arm/hand/camera observation as the scientific observation contract;
- sequential sends presented as physical synchronization;
- fixed sleeps as homing verification;
- zero-valued fallback actions after teleop exceptions.

DexMani should keep stronger temporal and tactile semantics than LeFranX.

---

## 3. Target architecture

Keep two workflow-specific upper layers and one shared physical runtime.

```text
                         ┌─────────────────────┐
                         │   Teleop Collection │
                         │  TeleopController   │
                         └──────────┬──────────┘
                                    │
                                    │ high-level target
                                    │
                         ┌──────────▼──────────┐
                         │    RobotCommand     │
                         │ command_id + run_id │
                         └──────────┬──────────┘
                                    │
                    single-inflight coupled command
                                    │
                   ┌────────────────┴────────────────┐
                   ▼                                 ▼
              Arm Worker                        Hand Worker
               xArm SDK                         XHand SDK
                   │                                 │
             command adopt                    command adopt
                                                     │
                                                bounded slew
                                                     │
                                                command reach
                   │                                 │
                   └──────────────┬──────────────────┘
                                  ▼
                       timestamped state rings
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
          Teleop raw recording              Policy observation
                                                   ▲
                                                   │
                                        ┌──────────┴──────────┐
                                        │     Policy Eval      │
                                        │ synchronous predict  │
                                        │   action chunk       │
                                        └─────────────────────┘
```

Do **not** introduce a generic `ProducerInterface`, `RolloutStrategy`, `InferenceBackend`, plugin registry, or service framework.

Teleop and policy may share low-level helpers, but they remain separate workflows.

---

## 4. Hard constraints and invariants

These are non-negotiable.

### 4.1 Keep SDK ownership in workers

- xArm live SDK object stays inside the arm worker process.
- XHand live SDK object stays inside the hand worker process.
- RealSense live object stays in its camera-owning process.
- Model/CUDA ownership stays in the policy process.
- Imports and ordinary constructors must not connect hardware.

### 4.2 Keep separate arm and hand worker processes

Do not collapse arm and hand into one process. XHand blocking/latency must not stall the arm worker.

### 4.3 Preserve real physical checks

The refactor must preserve clear ownership for:

- finite numeric SDK inputs;
- arm and hand mechanical limits;
- arm command jump / speed limits;
- XHand bounded slew;
- required workspace checks;
- required collision checks;
- robot error handling;
- emergency stop;
- safe disconnect;
- stale/revoked command rejection immediately before SDK admission.

Do not duplicate every check in every layer. Keep proposal shaping at the producer side and hard boundary validation at the actuator worker side.

### 4.4 Preserve causal observation semantics

Do not regress to a LeRobot-style sequential `get_observation()` implementation.

For policy observations and teleop recording where relevant, preserve:

- explicit observation anchor;
- latest causal sample with source time `<= anchor`;
- actual source timestamps;
- heterogeneous producer rates;
- same-sample XHand qpos/current/tactile aggregate/tactile dense/validity;
- RGB/depth identity from the same camera sample;
- tactile validity bits;
- policy history semantics;
- actual wall/monotonic timing rather than pretending every sample is perfectly periodic.

### 4.5 Keep synchronous policy inference only

Current scope is synchronous inference. Do not add:

- async predict;
- remote inference;
- RTC;
- action merging;
- interpolation framework;
- background model query;
- target-timestamp future queues.

A blocking `predict()` is acceptable. The runtime must safely fence stale results with a simple run identity.

### 4.6 Teleop C key MUST remain

This task explicitly **must not delete the teleop C key**.

Preserve the existing operator-facing C pause/resume capability for this refactor.

The command transport migration must adapt C safely:

- pausing must revoke/fence pre-pause commands;
- a not-yet-jointly-adopted pending command/sample must not leak across pause;
- resume must continue to require fresh post-pause feedback and a fresh teleop reference/re-anchor;
- C must remain visible in operator help;
- do not silently reinterpret C as discard or episode termination;
- do not remove same-session pause/resume behavior in this task.

A future research decision may revisit recorded-pause semantics, but that is **out of scope here**.

### 4.7 No silent recording downgrade

If teleop collection is configured with recording enabled, camera + recorder availability is a collection requirement.

Do not silently change:

```text
B = teleop + record
```

into:

```text
B = unrecorded teleop
```

because evidence services failed.

Unrecorded/debug teleop must be an explicit configuration/CLI mode.

### 4.8 Raw episode data is source of truth

Do not force raw collection to be a LeRobot dataset.

Keep raw:

- RGB-D;
- camera calibration/identity metadata;
- robot proprioception;
- XHand current;
- tactile aggregate + dense + validity;
- raw VR wrist + landmarks;
- mapped/retargeted intent where currently available;
- executable arm/hand high-level target;
- actual source timestamps;
- control/observation anchor;
- frame status;
- command/adoption facts.

Point clouds should normally be produced offline for collected demonstrations. Online point-cloud generation remains available only when a deployed policy requires it.

---

## 5. Replace the command FIFO with a single-inflight coupled command

This is the core architecture change.

### 5.1 Current problem

The current ordered coupled-command FIFO creates infrastructure that is not needed for the present synchronous research workflows:

- command backlog;
- FIFO capacity;
- FULL retry;
- arm/hand consumed watermarks;
- generation-base sequence;
- per-command expiration used to protect backlog;
- pending exact-candidate retries;
- command-stream corruption/backpressure machinery.

A stale high-level control backlog is undesirable for both policy rollout and human teleoperation.

However, unrestricted latest-wins is also wrong for recorded teleoperation: a human action must not be written as a valid demonstration label if it was overwritten before the actuator workers adopted it.

### 5.2 New command contract

Introduce one small canonical command object:

```python
@dataclass(frozen=True)
class RobotCommand:
    command_id: int
    run_id: int
    issued_monotonic_ns: int
    arm_qpos: np.ndarray | None
    hand_qpos: np.ndarray | None
```

The IPC wire dtype should carry exactly the required fixed-size fields, conceptually:

```text
run_id
command_id
issued_monotonic_ns
arm_present
hand_present
arm_qpos[7]
hand_qpos[12]
```

Do not add a generic command variant/tag system.

Reuse the existing shared-memory ring primitive as a latest command mailbox if practical. The key invariant is semantic, not the container name:

> There may be at most one high-level command that has been published but not yet adopted by every actuator worker present in that command.

The producer must not overwrite that command with the next command.

There is one control producer per session. Do not invent multi-producer arbitration.

### 5.3 Command identity

Use a monotonically increasing `command_id` for a runtime process lifetime.

Use `run_id` for lifecycle cancellation.

Do not use FIFO sequence as command identity.

The same `RobotCommand` must be visible to both actuator workers.

### 5.4 Arm adoption

The arm worker marks a command adopted only after:

1. it reads a new command;
2. command `run_id` still owns motion immediately before the SDK boundary;
3. worker-owned finite/shape/hard-limit checks pass;
4. the xArm SDK accepts the servo target.

Publish in arm state:

```text
last_adopted_run_id
last_adopted_command_id
last_adopted_monotonic_ns
```

These are executor facts, not producer guesses.

### 5.5 XHand adoption versus reach

This distinction is mandatory.

For streaming:

```text
new high-level hand target q*
        ↓
hand worker adopts target
        ↓
worker computes first bounded SDK setpoint
        ↓
SDK accepts first bounded setpoint
        ↓
hand_adopted = command id
        ↓
worker continues bounded slew toward latest target
```

Use:

```text
q_sdk_next =
    q_sdk_current
    + clip(q_target - q_sdk_current, -delta_max, +delta_max)
```

The hand worker must expose two different facts:

```text
last_adopted_run_id
last_adopted_command_id
last_adopted_monotonic_ns

last_reached_run_id
last_reached_command_id
last_reached_monotonic_ns
```

Definitions:

- **adopted**: worker has accepted the new high-level target and at least the first bounded low-level SDK setpoint for that target was accepted;
- **reached**: the exact high-level endpoint has been accepted/reached according to the existing XHand endpoint semantics.

Teleop and policy streaming gate the next high-level command on **adopted**, not **reached**.

Home or any operation that truly requires exact endpoint completion may wait for **reached**.

A streaming target may be superseded before it is reached. Do not fabricate a reached ACK for such a target.

### 5.6 Producer gating

Before publishing command `k+1`, the producer checks the latest arm/hand state and verifies command `k` is adopted by every actuator present in `k`.

If not:

- do not publish another command;
- do not build a backlog;
- do not busy-wait;
- count/report executor lag;
- continue the normal control loop timing;
- do not label a new human/model proposal as successfully executed.

### 5.7 Pending recording row

For a newly published command, keep at most one small producer-local pending recording row.

The row contains:

- observation snapshot/anchor used to generate the command;
- raw/mapped intent needed by the raw schema;
- `RobotCommand`;
- frame status;
- any action representation required by the raw writer.

Only finalize/append that normal command row after the arm and hand adoption facts show that the coupled command was adopted.

This guarantees that a normal command label in the raw dataset corresponds to a command actually adopted by both actuator workers.

If lifecycle revocation occurs before joint adoption:

- drop the pending normal command row;
- log the drop;
- never relabel it as a valid executed action.

Do not block waiting for adoption.

### 5.8 Hold/failure rows

For IK failure, retarget failure, or safety rejection where no new command is published:

- keep the previous executable high-level target in effect;
- do not publish duplicate hold commands solely for dataset bookkeeping unless actual xArm/XHand hardware semantics prove repeated sends are required;
- keep an explicit frame status;
- recording may store the previous valid executable target as the held action.

Do not remove the existing research-useful status distinction.

Recommended stable statuses include at least:

```text
OK
HELD
IK_FAIL
SAFETY_REJECT
RETARGET_FAIL
```

Do not add a large error taxonomy.

---

## 6. Simplify lifecycle cancellation to run_id

The current generation/FIFO transaction should become a much smaller lifecycle fence.

Keep `SafetyState`:

```text
DISARMED
ARMED
RUNNING
FAULT
```

Keep a monotonically increasing `run_id`.

A new RUNNING epoch gets a new `run_id`.

Any stop/pause/quit/fault/new epoch invalidates old work by changing/fencing `run_id`.

### Required stale-inference race protection

This exact race must remain safe:

```text
captured_run_id = current run

model.predict() blocks

operator presses S / Q
or timeout/fault ends the run

run_id changes

predict() returns old action chunk
```

The old chunk must be discarded before any publication.

The actuator worker must also check command `run_id` immediately before SDK admission.

That is why `run_id` must remain even after deleting the FIFO generation system.

### Remove after migration

Once no backlog transport remains and all callers are migrated, remove obsolete concepts such as:

- `run_generation_base_sequence`;
- arm/hand FIFO consumed watermarks;
- FIFO FULL handling;
- command FIFO capacity;
- exact pending FIFO retry state;
- FIFO-generation commit receipts;
- per-command dispatch expiry whose only purpose was protecting queued backlog;
- `RunEndReason.COMMAND_EXPIRED` if it has no independent non-backlog use.

Do not delete an expiry/freshness mechanism before tracing every caller and proving its original hazard is gone.

---

## 7. Target RuntimeChannels shape

Keep the existing shared-memory/process ownership model, but reduce command/lifecycle fields.

The target should be approximately:

```text
state/data
---------
arm_state_ring
hand_state_ring
camera_ring
vr_ring
pointcloud_ring          # only used when requested
record_sample_ring

command
-------
robot_command_ring       # latest mailbox; semantic single-inflight

recording
---------
record_control_q
record_result_q
recorder status needed for safe shutdown

runtime
-------
run_id
run_started_monotonic_ns
run terminal fact(s) only if still useful
safety_state
is_running
is_recording
error_state
estop_request
quit_requested
start/stop request fields actually required by policy parent UX

process health
--------------
heartbeats
ready_flags

home
----
arm_home_q
home completion facts actually required
```

Prefer command adoption facts in the arm/hand state dtypes rather than duplicate standalone shared scalar ACK channels.

Delete fields only after all producers/consumers are migrated.

---

## 8. Teleop workflow

The teleop workflow remains separate from policy evaluation.

Operator-facing normal collection controls must remain easy to understand:

```text
H   home
B   begin/resume active teleop recording workflow as currently defined
C   pause/resume       # MUST remain in this task
S   stop/save
D   stop/discard
Q   quit
ESC emergency stop
```

Do not remove C.

### 8.1 Recording-enabled startup

When `recording_enabled=True`:

- camera and recorder must be ready before B may start recorded collection;
- do not print/use a degraded `B=teleop` mode;
- if collection resources are unavailable, the user may still be allowed to home/quit safely, but a recorded episode must not start.

When `recording_enabled=False`, explicit manual/debug teleop remains valid.

### 8.2 Mid-episode recording failure

Camera/recorder failure is not automatically a physical hardware FAULT.

For teleop collection:

- current demonstration becomes invalid/incomplete;
- stop/revoke current teleop RUNNING episode to ARMED using the normal lifecycle fence;
- do not silently continue recording;
- retain safe operator recovery/home/quit where possible;
- block a new recorded B unless recording resources are available again;
- return a non-zero collection/session result when evidence failed.

Do not build a generic “evidence service priority matrix”.

### 8.3 C pause/resume behavior

For this task, preserve C.

When C pauses:

- invalidate the current `run_id`/motion epoch;
- drop any producer-local command that has not been jointly adopted;
- drop its not-yet-finalized normal recording row;
- do not allow a stale command to cross the SDK boundary;
- clear/rebuild teleop temporal references as required by current behavior;
- continue to require fresh arm/hand/VR feedback newer than the pause boundary before resume/re-anchor.

When C resumes:

- create/fence a fresh active run identity as required by the simplified lifecycle;
- re-anchor from fresh measured state and current VR state;
- do not carry an unpublished pre-pause human target into the resumed period.

Do not redesign the recording policy for C beyond what is needed to preserve correctness during this transport refactor.

### 8.4 Teleop control timing

Keep a fixed teleop control/data grid because it is scientifically useful.

Remove a separate high-frequency executor polling loop if it exists only to service keyboard/control bookkeeping.

Prefer event-or-deadline waiting:

```python
while running:
    controls = wait_for_operator_input_until(next_grid_deadline)
    handle_controls(controls)

    if grid_due:
        run_one_teleop_grid_tick()
```

Do not perform catch-up bursts after missed grid slots. Skip/count missed slots.

Teleop grid frequency belongs to teleop/data-collection config, not policy model config.

---

## 9. Teleop raw-data action semantics

The raw training action must be scientifically explicit.

For a normal command row:

```text
action =
    post-IK / post-projection / safety-admitted
    arm + hand high-level target
    that the real actuator workers actually adopted
```

Do not use as the canonical behavior-cloning action:

- raw VR pose;
- unexecuted pre-IK arm intent;
- XHand intermediate slew setpoints;
- measured qpos;
- a command that was merely published but never adopted.

Still keep human/mapped intent separately in raw data where available.

The raw episode should preserve the chain:

```text
human intent
    ↓
mapped / retargeted robot intent
    ↓
executable high-level RobotCommand
    ↓
measured robot + tactile response
```

---

## 10. Raw schema migration

Current raw schema is v30. Because command/action semantics change, perform an explicit schema bump instead of silently reinterpreting old fields.

Target: bump to v31 (unless local HEAD already bumped for another reason; then use the next unused version).

Update all writer/reader/processing/export code that assumes the current schema.

### New/changed raw facts

A normal or held row should be able to expose the command identity/effect that its stored executable target refers to.

Add or rename fields as needed, with clear semantics, including the equivalent of:

```text
command_id
command_run_id
command_issued_monotonic_ns
arm_command_adopted_monotonic_ns
hand_command_adopted_monotonic_ns
frame_status
```

For rows with no valid prior command, command identity/timestamps may use the schema’s explicit unknown/sentinel representation.

Replace/remove `action_queued` because “queued” is no longer the transport truth.

Do not invent historical adoption timestamps for v30 data.

Do not silently reinterpret v30.

Do not add runtime compatibility branches just to make old data look like v31. Old data conversion, if ever required, must be an explicit offline operation with honest unknown fields.

Keep:

- observation anchor;
- arm source timestamp;
- hand source timestamp;
- VR source timestamp;
- camera source timestamp;
- observation validity;
- camera/tactile validity;
- actual physical state and raw sensor data.

### Recording-row finalization

A normal command row should be appended only once joint adoption is established, and should store the actual adoption timestamps from worker state.

Held/failure rows may reuse the most recent jointly adopted command identity/target when scientifically correct.

---

## 11. Observation and provenance simplification

Keep the scientific temporal contract, simplify IPC-internal proof machinery.

### Keep

- observation anchor;
- latest-before-anchor selection;
- policy history grid;
- past-only selection;
- same-sample XHand state;
- tactile validity;
- RGB/depth identity;
- actual source timestamps;
- derived EEF/fingertip features required by policy;
- inter-modality skew/freshness validation that directly affects observation validity.

### Simplify/remove where not scientifically consumed

Avoid repeatedly proving internal transport chains such as:

```text
source <= receive <= publish <= ring_commit <= anchor
```

at multiple layers.

The producer/ring should own its internal integrity.

The consumer normally needs:

```text
source timestamp
causal relation to anchor
freshness
validity
```

Do not remove a timestamp that a scientific consumer actually uses.

---

## 12. Policy evaluation workflow

Policy eval must become a small synchronous loop.

Conceptual shape:

```python
while session_active:
    handle_parent_requests()

    if not trial_running:
        continue

    if previous_command_is_not_yet_jointly_adopted():
        note_executor_lag()
        wait_for_next_control_deadline()
        continue

    finalize_pending_recording_row_if_adopted()

    if action_chunk_empty():
        obs = build_policy_observation()

        captured_run_id = current_run_id()
        chunk = policy.predict(obs)

        if current_run_id() != captured_run_id:
            discard(chunk)
            continue

        queue_local_chunk(chunk)

    raw_action = pop_next_action()
    command = decode_project_validate(raw_action)

    if recoverable_ik_or_workspace_miss:
        clear_unexecuted_chunk_suffix()
        continue

    publish(command)
    keep_one_pending_recording_row(command, obs, raw_action)

    wait_until_actual_next_control_time()
```

### Timing

- inference remains synchronous;
- first action of a newly returned chunk publishes as soon as inference finishes and the previous command has been adopted;
- subsequent actions follow the policy control period;
- do not catch up missed action times by rapid multi-dispatch;
- anchor cadence to actual physical publication/adoption history as appropriate;
- preserve actual timing in raw timestamps.

### Action chunk

Keep:

- `PolicySpec.n_action_steps`;
- local unexecuted action chunk;
- clear chunk suffix after a recoverable replanning boundary.

Delete:

- pending FIFO publication retry;
- queue-backpressure state;
- async-style scheduling abstractions.

### Policy integration scope

Continue using the public `dexmani_policy.deployment` interface for the currently integrated model source.

Do not build a generic adapter/plugin registry in this refactor merely for hypothetical repositories.

However, keep `PolicyRunner` organized so training-repository-specific loading/model code stays at the deployment boundary and the runtime loop operates on the existing `PolicySpec` plus canonical observation/action contract.

When a second concrete external model repository is actually integrated, add the smallest concrete adapter needed then.

---

## 13. run_policy.py cleanup

Make `examples/run_policy.py` a trustworthy reproducible experiment entry point.

Required:

1. Add `--config` and resolve the selected experiment runtime config through the existing config resolver. Do not create a second config system.
2. Rename internal `num_trials` terminology to `num_episodes` unless a distinction remains scientifically necessary.
3. Keep “technical episode” semantics separate from task success.
4. Write the **full resolved runtime configuration** into the session directory before hardware starts, not only a small hand-written subset.
5. Also save the model/policy facts needed to reproduce the run:
   - experiment selector;
   - checkpoint/artifact;
   - inference steps;
   - seed;
   - device;
   - PolicySpec-relevant control dt/history/action chunk facts;
   - requested episode count and duration.
6. Do not connect hardware during parsing/config/model metadata inspection.
7. Keep the user-facing summary concise.

Prefer one resolved session config file rather than multiple overlapping sources of truth.

---

## 14. Technical episode result semantics

Separate:

```text
technical validity
```

from:

```text
task success
```

A robot that fails to grasp can still be a technically valid eval episode.

A camera failure, policy crash, hardware fault, or required recorder failure is a technically invalid episode.

Store/report enough metadata to distinguish at least:

```text
technical_status: valid | invalid
termination_reason
task_success: true | false | unknown
```

Task success may remain offline/manual if that is current workflow.

Avoid dual counters that mix “trial happened” and “evidence happened” unless both facts are actually needed. Use clear names if both remain.

---

## 15. Recorder scope

Keep the recorder as a dedicated process with shared-memory sample transport.

This is justified by real IO/resource boundaries:

- RGB-D payload size;
- video/image serialization;
- disk IO;
- avoiding large multiprocessing queue copies.

Do not collapse video/HDF5 writing into the control process.

But simplify the *control protocol* where possible.

The logical client API should move toward:

```text
start_episode(metadata)
append(frame)
finish_episode(save=True/False, reason=...)
```

Do not redesign recorder internals until command transport, worker ACK semantics, and raw row semantics are stable.

Policy eval required-recording failure should invalidate/end the current eval workflow cleanly rather than claim a hardware FAULT.

Teleop recording failure handling follows section 8.

---

## 16. Process supervision

Keep only supervision that corresponds to real experiment hazards:

- readiness;
- unexpected critical worker death;
- arm heartbeat;
- hand heartbeat;
- camera heartbeat when required;
- recorder health when required;
- estop;
- episode/session timeout;
- verified process shutdown before unlinking shared memory.

Do not heartbeat-fault the policy child merely because synchronous inference is blocking.

VR freshness should primarily be based on actual VR source freshness where appropriate rather than a generic heartbeat abstraction.

Preserve the verified stop/join/terminate/kill/confirm-before-unlink behavior in `runtime/processes.py` unless a specific part is proven redundant.

Reduce/remove generic “service/evidence” classification once workflow-specific failure handling no longer needs it.

---

## 17. Home / replay / blocking operations

Do not force safe homing into the streaming command abstraction.

Arm home may remain a special physical operation because it owns:

- planned path;
- controller mode transitions;
- workspace/table/collision checks;
- settled feedback;
- mode restoration;
- abort semantics.

Hand home may use exact endpoint semantics.

After command transport migration:

- convert any command ACK caller to explicit `wait_command_adopted` or `wait_hand_reached` semantics;
- use **adopted** for streaming/replay only when that is the existing physical intent;
- use **reached** when exact endpoint completion is truly required;
- trace each call site rather than globally replacing waits.

Do not use fixed sleeps as proof of physical home completion.

---

## 18. Configuration ownership cleanup

Do this after the runtime command migration is stable.

The current config has teleop concepts living under policy-oriented names. Move ownership only when it makes the source of truth clearer.

Target conceptual ownership:

```yaml
hardware:
  arm: ...
  hand: ...
  camera: ...

safety: ...

teleop:
  control_hz: ...
  vr_mapping: ...
  hand_retargeting: ...
  smoothing: ...
  recording behavior: ...

recording:
  data paths / duration / writer settings ...

eval:
  num_episodes / max duration / recording policy ...
```

Policy-model timing and required modalities should come from `PolicySpec`, not be duplicated in runtime YAML.

Do not perform a cosmetic config reorganization that creates broad churn without removing real ambiguity.

Update all call sites atomically for any field moved.

---

## 19. File-level disposition guide

Use actual dependency tracing; this is a target, not permission to blindly delete.

### `examples/run_policy.py`

- add `--config`;
- full resolved config snapshot;
- normalize episode terminology;
- keep no-hardware preflight behavior.

### `examples/collect_teleop.py`

- keep simple research-facing entry point;
- preserve C in help/behavior through teleop loop;
- explicit recorded versus unrecorded mode.

### `dexmani_real/ipc/command_stream.py`

- delete **only after** every consumer has migrated to single-inflight latest-command semantics;
- no compatibility wrapper should remain.

### `dexmani_real/ipc/schema.py`

- replace FIFO command dtype with small `ROBOT_COMMAND_DTYPE`;
- add arm adoption fields;
- replace XHand exact-only ACK identity with adopted + reached identities;
- update record sample dtype for command/adoption facts.

### `dexmani_real/ipc/channels.py`

- replace `coupled_cmd_ring` FIFO semantics with one latest command mailbox/ring;
- remove consumed watermarks and generation-base state after migration;
- keep real sensor/state rings and recorder resources;
- keep process readiness/heartbeat data that is still used.

### `dexmani_real/runtime/safety.py`

- keep `SafetyState`;
- simplify generation into `run_id`;
- preserve atomic begin/revoke lifecycle;
- preserve stale command/inference fencing;
- remove FIFO/backlog/expiry-specific logic after no callers depend on it.

### `dexmani_real/robot/commands.py`

Rewrite around:

- `RobotCommand`;
- command construction/preparation;
- current-feedback read needed for producer safety;
- simple safety result;
- publish to single-inflight mailbox;
- adoption/reach helpers for blocking callers.

Collapse/remove transport-only layers such as:

- `ActionCandidate`;
- `PublishResult`;
- `AcceptanceResult`;
- `PreparedCommand`

when they no longer add a real decision.

Keep real joint/workspace/collision validation.

### `dexmani_real/robot/arm_worker.py`

- latest command observation instead of FIFO consumer;
- ignore already-seen command id;
- reject wrong run id before SDK;
- hard boundary validation;
- mark adopted only after SDK accepts.

### `dexmani_real/robot/hand_worker.py`

- keep same-sample qpos/current/tactile;
- keep XHand bounded slew;
- latest high-level target;
- adopted after first accepted bounded SDK step;
- reached only at exact target;
- no FIFO sequence advancement.

### `dexmani_real/teleop/control_loop/grid.py`

Major simplification:

- keep causal observation;
- keep VR mapping;
- keep hand retargeting;
- keep IK/projection/safety;
- delete FIFO FULL retry/pending publication transaction;
- replace with at-most-one pending command recording row;
- gate next command on joint adoption;
- preserve C pause integration.

### `dexmani_real/teleop/loop.py`

- simplify state/loop bookkeeping where command FIFO complexity disappears;
- keep C pause/resume;
- remove separate executor polling frequency if unnecessary;
- keep operator controls and safe recovery.

### `dexmani_real/teleop/episode_samples.py`

- keep frame status;
- keep source timestamps/observation validity;
- add command/adoption facts;
- remove queue-specific `action_queued` semantics;
- simplify internal provenance validation only after tracing consumers.

### `dexmani_real/teleop/session.py`

- recording-enabled camera/recorder become collection-critical for B;
- remove silent evidence downgrade;
- simplify generic service/evidence classification where possible;
- do not turn recorder failure into hardware FAULT.

### `dexmani_real/deployment/runner.py`

Major rewrite to small synchronous:

```text
observe → predict chunk → decode/project → publish → adoption gate → pace
```

Keep run-id fence around blocking inference.

### `dexmani_real/deployment/observation.py`

Keep scientifically meaningful history/causal logic. Simplify only redundant transport provenance.

### `dexmani_real/deployment/session.py`

Keep process ownership and parent-side timeout/controls. Simplify evidence/service bookkeeping consistent with the new failure model.

### `dexmani_real/replay/replayer.py`

Migrate from FIFO acceptance to explicit adopted/reached semantics based on actual replay intent.

### homing modules

Keep real safety/planning behavior. Only change command acknowledgement plumbing as needed.

### `dexmani_real/recording/*`

- preserve recorder process + SHM;
- bump raw schema;
- update writer/reader/frame construction;
- simplify protocol later, after command/data semantics stabilize.

### `dexmani_real/runtime/supervisor.py`

- preserve critical liveness/safety;
- remove generic evidence service matrix when workflow-specific logic replaces it.

### `dexmani_real/runtime/processes.py`

Mostly preserve.

### `dexmani_real/robot/projection.py`

Keep. It is current algorithm/physical command shaping, not framework overhead.

---

## 20. Mandatory execution phases

Complete phases in order. Do not leave a long-lived half-migration between old and new command semantics.

### Phase 0 — audit and low-risk reproducibility fixes

1. Trace current entry points:
   - `examples/collect_teleop.py`;
   - `examples/run_policy.py`;
   - replay/home callers.
2. Run repository-wide searches for:
   - `coupled_cmd_ring`;
   - `command_stream`;
   - `run_generation`;
   - `run_generation_base_sequence`;
   - `arm_cmd_consumed_sequence`;
   - `hand_cmd_consumed_sequence`;
   - `ActionCandidate`;
   - `PUBLISH_REASON_FIFO_FULL`;
   - `expires_monotonic_ns`;
   - `accepted_target_sequence`;
   - `action_queued`.
3. Document in working notes which producer/consumer/side effect each match belongs to. Do not add a permanent audit document.
4. Implement:
   - `run_policy --config`;
   - full resolved config save;
   - episode naming cleanup;
   - recording-enabled teleop no silent downgrade.
5. Verify C still exists before and after Phase 0.

Run focused offline checks.

### Phase 1 — define new command and state wire contracts

Change `ipc/schema.py`, `ipc/channels.py`, and the smallest pure helpers first.

Add:

- `ROBOT_COMMAND_DTYPE`;
- arm adopted fields;
- hand adopted + reached fields;
- single command mailbox;
- simplified run identity fields.

Do not yet delete old command stream if live callers remain.

Write one-off pure/offline checks that validate dtype shapes and command identity semantics.

### Phase 2 — migrate arm + hand workers and both main producers together

Migrate in one coordinated phase:

- arm worker;
- hand worker;
- teleop grid/loop command path;
- policy runner command path.

Required invariant after this phase:

> No teleop/policy streaming path depends on FIFO ordering, FULL retry, or consumed watermarks.

Implement producer-local single pending recording row.

Keep C working through the new run-id fence.

Run offline fake/shared-memory smoke checks for the scenarios in section 21.

Only after all streaming callers are migrated:

- delete `ipc/command_stream.py`;
- remove FIFO resources and imports;
- remove dead transport status code.

### Phase 3 — migrate blocking command users

Trace replay, home, calibration helpers, or diagnostics that depended on old acceptance receipts.

Replace with explicit:

- adopted wait;
- hand reached wait;
- direct home completion mechanism;

according to real semantics.

Then remove remaining backlog expiry/generation transaction code.

### Phase 4 — raw schema v31 and recorder row semantics

1. Update record sample wire schema.
2. Update `recording/frame.py`.
3. Update storage schema and bump raw version.
4. Update recorder writer/validation.
5. Update reader/processing/export code that requires the current raw version.
6. Ensure normal command rows are finalized only after joint adoption.
7. Keep held/failure rows scientifically explicit.
8. Do not reinterpret old v30.

### Phase 5 — simplify teleop/policy workflow bookkeeping

Now that transport complexity is gone:

- remove obsolete pending-publish/retry state;
- simplify policy evidence bookkeeping;
- simplify teleop service/evidence state;
- remove unnecessary high-frequency polling;
- keep C;
- preserve supervisor/hardware safety.

Do not redesign homing.

### Phase 6 — observation/provenance and config cleanup

- remove duplicated internal provenance checks that no scientific consumer needs;
- preserve source timestamps and causal selection;
- move clearly mis-owned teleop/eval configuration fields;
- ensure one source of truth for teleop control rate and policy control dt.

### Phase 7 — recorder protocol simplification

Only after the above is stable:

- reduce recorder client state/transactions where possible;
- retain recorder process + SHM;
- preserve safe finalization and verified shutdown.

---

## 21. Required offline smoke scenarios

Do not add a committed test framework. Use temporary scripts / inline Python / existing pure constructors.

At minimum verify:

### S1. Single-inflight gate

- publish command 1;
- arm adopted command 1;
- hand has not adopted command 1;
- producer refuses to publish command 2;
- command mailbox is not overwritten.

### S2. Joint adoption unlocks next command

- arm and hand both report adopted command 1 for the active run;
- command 2 may publish.

### S3. Hand adopted != reached

- hand accepts first bounded setpoint for command 1;
- `last_adopted_command_id == 1`;
- `last_reached_command_id` remains previous;
- streaming may advance after arm also adopts;
- no fake reached ACK is produced.

### S4. Exact hand endpoint

- simulated bounded slew eventually reaches exact target;
- reached identity advances only then.

### S5. Stale run command fence

- command belongs to run N;
- run changes to N+1 before SDK admission;
- old command is rejected.

### S6. Blocking inference fence

- capture run N;
- simulate run transition while “predict” is blocked;
- returned chunk for N is discarded.

### S7. Teleop C pause

- active run has an unjointly-adopted pending command;
- C pause invalidates run;
- pending command/sample is dropped;
- no stale publication after resume;
- C remains a recognized key;
- fresh feedback/re-anchor is still required.

### S8. Recording label truth

- normal raw command row is not appended before both executor adoptions;
- after both adoptions, stored command/adoption timestamps correspond to that command id.

### S9. Held/failure row

- IK/safety/retarget failure publishes no new command;
- frame status records failure;
- previous valid executable target remains the held action where appropriate.

### S10. No-record versus recording-required teleop

- explicit no-record mode can start without recorder;
- recording-enabled mode cannot silently start unrecorded when camera/recorder is unavailable.

### S11. No hardware side effects on import

Import modified modules in an offline environment where possible and verify ordinary imports/constructors do not connect hardware.

---

## 22. Repository-wide removal checks

At the end, use `rg` to prove old transport concepts are gone where expected.

Expected absent after full migration, subject to comments/docs being updated:

```text
PUBLISH_REASON_FIFO_FULL
run_generation_base_sequence
arm_cmd_consumed_sequence
hand_cmd_consumed_sequence
command fifo full
coupled command FIFO
```

`ipc/command_stream.py` should no longer exist.

`ActionCandidate`, `PreparedCommand`, `PublishResult`, `AcceptanceResult` should be removed if no real non-transport decision remains.

`expires_monotonic_ns` and `COMMAND_EXPIRED` should be removed if their only remaining purpose was queued-backlog protection.

Do **not** use search-driven deletion without tracing semantics.

Also verify C was **not** removed:

```bash
rg -n "C=.*pause|pause/resume|KeyCode.*c|['\"]c['\"]" dexmani_real examples
```

Adapt the exact search to the implementation.

---

## 23. Standard offline checks after each meaningful phase

Run:

```bash
python -m compileall -q dexmani_real examples
ruff check --select F401,F821,F822,F823 dexmani_real examples
git diff --check
```

If Ruff is unavailable, report that fact. Do not install/upgrade dependencies just for this task.

Also run focused pure checks for transforms, dtype construction, safety helpers, command state, and recording frame construction touched by the phase.

Never run hardware examples as tests.

---

## 24. Final review checklist

Before declaring completion:

### Architecture

- [ ] Teleop and policy remain separate workflows.
- [ ] Arm and hand remain separate SDK-owning processes.
- [ ] Recorder remains a process.
- [ ] Command FIFO/backlog is gone.
- [ ] Single-inflight coupled command is enforced.
- [ ] No unrestricted latest-wins behavior.
- [ ] No generic strategy/backend/plugin framework was added.

### Safety

- [ ] SafetyState still fences physical motion.
- [ ] run_id protects blocked inference and stale commands.
- [ ] worker checks run_id immediately before SDK admission.
- [ ] arm hard validation remains.
- [ ] hand bounded slew remains.
- [ ] estop remains.
- [ ] safe disconnect remains.
- [ ] homing safety remains.

### XHand

- [ ] qpos/current/tactile are still one same-sample state.
- [ ] tactile validity bits remain.
- [ ] adopted and reached are distinct.
- [ ] streaming waits for adopted, not exact reach.

### Teleop

- [ ] B/S/D/H/Q/ESC behavior remains coherent.
- [ ] **C pause/resume remains present.**
- [ ] C safely invalidates stale pending commands.
- [ ] recording-enabled mode never silently downgrades to no-record.
- [ ] VR stale/hand/camera failures cannot create a fake valid command label.

### Policy eval

- [ ] inference is synchronous only.
- [ ] no catch-up dispatch.
- [ ] stale prediction chunk after run change is dropped.
- [ ] action chunk suffix is cleared at recoverable replanning boundaries.
- [ ] PolicySpec remains the model/runtime contract.

### Data

- [ ] raw schema was explicitly version-bumped.
- [ ] normal action row represents an actually adopted executable target.
- [ ] raw intent remains distinct from execution.
- [ ] actual source timestamps remain.
- [ ] observation anchor remains.
- [ ] tactile validity remains.
- [ ] old raw schema is not silently reinterpreted.

### Reproducibility

- [ ] run_policy accepts explicit config.
- [ ] full resolved config is stored before hardware startup.
- [ ] policy artifact/spec/session facts are stored.
- [ ] technical validity is distinct from task success.

### Cleanup

- [ ] dead imports removed.
- [ ] stale docs/README terminology updated.
- [ ] no duplicate old/new command APIs remain.
- [ ] no compatibility shim remains solely to preserve deleted internal architecture.
- [ ] no committed test framework added.
- [ ] no hardware-affecting checks were run.

---

## 25. Expected final handoff from Codex

At completion, report:

1. A concise architecture summary of what changed.
2. The major files modified/deleted.
3. Measured simplification where useful:
   - removed files;
   - removed old command/lifecycle symbols;
   - approximate line-count reduction in the main runtime path if easy to measure.
4. Exact offline checks run and their results.
5. Explicit statement that no hardware validation was performed.
6. Any behavior that still requires real-robot validation.
7. Any remaining complexity intentionally kept because it protects:
   - physical safety;
   - scientific timing/data correctness;
   - a current algorithmic requirement.
8. Confirmation that teleop **C pause/resume was retained**.

Do not claim the refactor is hardware-validated until it has been tested in a supervised physical session.

---

## 26. Complexity budget for every design decision

Before adding or preserving a mechanism, ask:

1. Without it, can the robot become less safe?
2. Without it, can experiment data become scientifically wrong?
3. Without it, will current experiment iteration materially suffer?
4. Has this exact hardware/workflow actually encountered the problem?

If the answer is no to all four, remove or do not add the mechanism.

Justified complexity categories:

### A. Physical hardware

- SDK ownership;
- joint/workspace/collision limits;
- XHand slew;
- safe home;
- robot errors;
- estop;
- safe shutdown.

### B. Scientific data

- actual timestamps;
- coordinate frames;
- tactile validity;
- same-sample XHand modalities;
- causal sensor alignment;
- human intent versus executable action;
- executor adoption timing.

### C. Current algorithms

- observation history;
- action chunks;
- EE IK;
- recoverable replanning;
- point cloud when the deployed model actually requires it;
- model inference steps.

Presumptively removable:

### D. Infrastructure transaction complexity

- lossless streaming command FIFO;
- backlog retry;
- generation-base/watermark transaction;
- generic evidence service orchestration;
- async/RTC scaffolding;
- generic rollout strategy framework;
- hypothetical backend/plugin abstractions.

The final code should feel like a research instrument, not a platform.
