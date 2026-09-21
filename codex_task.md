# Codex Task — Refactor DexMani Real into a Small, Reliable PhD Real-Robot Experiment Endpoint

## 0. Execution contract

This file is the implementation task for the local Codex CLI. Treat it as an engineering specification, not as a brainstorming document.

Before editing:

1. Read repository-root `AGENTS.md` in full and obey it.
2. Run:
   ```bash
   git status --short
   git rev-parse HEAD
   ```
3. Preserve unrelated local changes. Never reset, checkout, stash, clean, or overwrite user work.
4. Trace the actual current code before applying this document. The design was reviewed against:
   - `dexmani_real` runtime code from `4bba54d9094291b7b8030464c7983ac93971ce6b`
   - current task-bearing main after `89beeb690f39c8959cf4be25055c13283e58bc87`
   - LeRobot `8352ca62d88ab4464001d427267a6a0758e6cb99`
   - ManiUniCon `85c6f2e32ecf9f2bed62d202b058c39623444686`
   - LeFranX `a39906e6629f39490950fe8bd20f4f992ed74fd7`
5. If local HEAD differs, source code and resolved configuration are authoritative. Re-trace every affected definition → producer → transformation → consumer → side effect.
6. Do **not** execute hardware-affecting code. No device discovery, SDK connection, homing, replay, teleoperation, calibration writes, policy rollout, camera capture, or physical motion without explicit user authorization.
7. Ordinary imports and constructors used by offline checks must not connect hardware.
8. Do **not** create or restore a committed `tests/` directory. Use temporary/offline smoke checks.
9. Do not install or upgrade the experiment environment just to run checks. If an optional tool is missing, report it.
10. Do not create generic framework abstractions to make the refactor look cleaner. The target is a research instrument.
11. Do not stop after producing a plan. Execute the complete task unless genuinely blocked by unavailable source/dependency information or a required physical validation.
12. Keep this `codex_task.md` at repository root until the user explicitly asks to remove it.
13. Do not claim hardware validation. Offline correctness and physical validation are different.

When a phase touches a cross-process protocol, read and modify **both sides** of the boundary before considering the phase complete.

---

# 1. Repository mission

`dexmani_real` is a personal PhD research repository for two primary workflows.

## 1.1 Dexterous real-robot data collection

```text
Quest / VR
   ↓
wrist mapping + hand retargeting
   ↓
IK / projection / safety
   ↓
xArm7 + XHand
   ↓
timestamped proprioception + tactile + RGB-D + VR
   ↓
raw research episode
```

## 1.2 Real-robot evaluation of models trained in other repositories

```text
external trained model
   ↓
public model metadata / PolicySpec
   ↓
causal multimodal observation
   ↓
synchronous predict + local action chunk
   ↓
decode / IK / projection / safety
   ↓
xArm7 + XHand
   ↓
raw evaluation episode + technical metrics
```

The repository is **not**:

- a generic robotics platform;
- a robot plugin ecosystem;
- production policy serving;
- a remote inference service;
- an async inference framework;
- an RTC implementation;
- a generic rollout strategy framework;
- a policy training framework;
- a universal dataset compatibility layer.

Priority:

> physical safety > experiment correctness > iteration speed > readability > generic extensibility > enterprise robustness

Preferred change order:

> delete > inline > merge > rewrite > add abstraction

A mechanism earns its complexity only if it protects a current physical guarantee, scientific-data guarantee, or current experiment requirement.

---

# 2. What to learn from the reference projects

## 2.1 LeRobot

Adopt:

- explicit separation between teleop recording and learned-policy rollout;
- clear observation/action boundaries;
- simple operator-facing workflows;
- a physical action boundary where clipping/modification is explicit.

Do not copy:

- generic rollout strategies;
- async inference engines;
- processor/plugin infrastructure;
- broad robot ecosystem abstractions.

Important: DexMani must not persist a normal behavior-cloning label merely because a teleop/model action was computed or published. A normal action row must refer to a high-level executable command that the relevant actuator workers actually adopted.

## 2.2 ManiUniCon

Adopt:

- process boundaries that follow actual resource ownership;
- shared memory for high-rate sensor/state transport;
- teleop and learned policy as command producers above robot execution.

Do not copy:

- future target-timestamp action queues;
- generic control-mode machinery;
- Hydra-style extensibility for hypothetical hardware;
- buffered action backlogs.

## 2.3 LeFranX

Adopt:

- simple dexterous collection workflow;
- combined arm/hand operator experience;
- straightforward home / begin / save / discard interaction.

Do not copy:

- serial observation as the scientific timing contract;
- sequential arm/hand sends presented as physical synchronization;
- fixed sleeps as homing proof;
- zero fallback commands after teleop exceptions.

DexMani intentionally keeps stronger timing, tactile, and causal-observation semantics.

---

# 3. Target architecture

Keep two workflow-specific upper layers and one shared physical runtime.

```text
          Teleop Collection                         Policy Eval
          -----------------                         -----------
          TeleopController                          Policy runner
          VR map / retarget                         sync predict
                   │                                action chunk
                   └──────────────┬─────────────────────┘
                                  │
                         executable targets
                                  │
                                  ▼
                           RobotCommand
                    command_id + run_id + targets
                                  │
                     single-inflight per run
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
                Arm worker                  Hand worker
                 xArm SDK                    XHand SDK
                    │                           │
                 adopted                    adopted
                                                │
                                           bounded slew
                                                │
                                             reached
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                        timestamped state rings
                                  │
                ┌─────────────────┴─────────────────┐
                ▼                                   ▼
          raw recording                       policy observation
```

Important:

- “coupled command” means one arm/hand high-level command identity.
- It does **not** mean physically atomic or simultaneous SDK admission.
- Arm and hand are independent workers; partial adoption is possible at lifecycle boundaries and must be represented truthfully.

Do **not** create:

- `ProducerInterface`;
- `RolloutStrategy`;
- `InferenceBackend`;
- a plugin registry;
- a generic runtime service manager;
- a generic failure taxonomy framework.

---

# 4. Hard invariants

## 4.1 SDK ownership

Keep live SDK objects inside their owning processes:

- xArm SDK → arm worker;
- XHand SDK → hand worker;
- RealSense SDK → camera worker;
- CUDA/model runtime → policy worker.

Do not collapse arm and hand into one process. XHand latency must not stall arm execution.

## 4.2 Physical safety ownership

Preserve clear ownership for:

- finite target validation;
- arm/hand hard joint limits;
- arm command step/jump limits;
- XHand low-level bounded slew;
- required workspace checks;
- required collision checks;
- xArm/XHand error handling;
- estop;
- safe disconnect;
- stale/revoked command rejection at the last software fence before SDK admission.

Producer-side projection/soft shaping and worker-side hard boundary validation are different responsibilities. Do not duplicate all checks everywhere.

If a command that already passed the producer boundary later fails a worker-owned hard finite/shape/mechanical validation, treat that as an invariant/physical-safety failure and fail closed; do not leave the producer waiting forever for an ACK that can never arrive.

Likewise, an unexpected SDK send failure for a current valid command is a worker/runtime failure unless the existing driver explicitly classifies that status as a successful/tolerated physical outcome.

## 4.3 Causal observation

Do not replace causal multimodal observation with sequential “read everything now”.

Preserve where scientifically relevant:

- explicit observation anchor;
- latest source sample with source timestamp `<= anchor`;
- actual source timestamps;
- heterogeneous producer rates;
- policy history grids;
- past-only selection;
- same-sample XHand qpos/current/tactile aggregate/tactile dense/validity;
- RGB/depth identity from one camera sample;
- tactile validity;
- actual timing rather than fictitious fixed wall time;
- required derived EEF/fingertip quantities.

## 4.4 Synchronous policy inference only

Current scope is synchronous inference.

Do not add:

- async prediction;
- RTC;
- remote inference;
- background queries;
- action merging;
- interpolation frameworks;
- future target timestamp queues.

A blocking `predict()` is acceptable. Lifecycle fencing must make its returned stale result harmless.

## 4.5 Teleop C key MUST remain

This refactor must **not delete or reinterpret C**.

C remains operator-facing pause/resume.

Requirements:

- C remains in help text and keyboard handling;
- pause fences the active command-admission epoch;
- stale pre-pause commands cannot cross the SDK fence after revocation;
- resume requires fresh post-pause arm/hand/VR data and the current re-anchor behavior;
- unpublished pre-pause intent must not become a post-resume command;
- C must not become discard, stop/save, or episode termination.

Do not redesign the overall “recorded pause” user semantics in this task. Only make it truthful and safe under the new command transport.

## 4.6 Recording-enabled teleop must not silently degrade

When recording is enabled, camera + recorder are collection requirements.

If either is unavailable:

- B must not start a recorded demonstration;
- do not silently relabel B as unrecorded teleop;
- H/Q/safe recovery may remain available if hardware is otherwise healthy.

Unrecorded teleop is an explicit `--no-record` / disabled-recording mode.

## 4.7 Raw episode is source of truth

Do not make LeRobotDataset the raw runtime storage contract.

Preserve raw information needed for current and future research:

- RGB-D;
- camera calibration/identity metadata;
- arm q/qvel/tau as currently recorded;
- XHand q/current;
- tactile aggregate + dense + validity;
- raw VR wrist + landmarks;
- mapped/retargeted intent where currently available;
- executable high-level robot target;
- actual source timestamps;
- observation/control anchor;
- command identity and adoption facts;
- frame status.

Collected demonstrations should normally store RGB-D, not an online point cloud. Generate point clouds offline for dataset processing. Keep online point-cloud production only when a deployed policy explicitly requires it.

---

# 5. New command protocol: single-inflight coupled command

This is the core runtime change.

## 5.1 Why the FIFO goes away

The current ordered command FIFO creates unnecessary streaming infrastructure:

- backlog;
- FULL retry;
- consumed watermarks;
- generation-base sequence;
- queued-command expiry;
- pending exact-candidate retries;
- queue corruption/backpressure logic.

For teleop and synchronous policy deployment, stale high-level commands should not accumulate.

However, unrestricted latest-wins is also incorrect: it can overwrite a human/model action before one actuator adopts it while the dataset still labels it as executed.

Target:

> no backlog, and at most one not-yet-jointly-adopted high-level command in the current run.

## 5.2 Canonical command

Use one small immutable command:

```python
@dataclass(frozen=True)
class RobotCommand:
    command_id: int
    run_id: int
    issued_monotonic_ns: int
    arm_qpos: np.ndarray | None
    hand_qpos: np.ndarray | None
```

Wire fields are conceptually:

```text
command_id
run_id
issued_monotonic_ns
arm_present
hand_present
arm_qpos[7]
hand_qpos[12]
```

No generic action tag or command subclass hierarchy.

Reserve `command_id == 0` for “no command / unknown”.

## 5.3 command_id allocation

`command_id` must be monotonic for the whole shared-runtime lifetime, not local to one producer process.

Reason: command publication can occur from different owning workflows/operations over one runtime lifetime (for example home/warm-up/replay/policy paths). Process-local counters can reuse IDs and make worker ACK identity ambiguous.

Keep a minimal shared `next_command_id` (or equivalent) allocated under the existing short motion/lifecycle lock.

Prefer:

1. producer computes/project/checks target outside the lock;
2. publication acquires the short motion lock;
3. rechecks lifecycle ownership;
4. allocates the next command ID;
5. timestamps `issued_monotonic_ns` at publication;
6. writes one command record;
7. releases the lock.

Do not hold the motion lock across SDK IO.

Do not use shared-memory ring sequence as the public command identity.

## 5.4 Mailbox / ring semantics

A latest-record shared-memory ring may be reused as the physical container. The semantic contract is a mailbox, not a FIFO.

Within one current `run_id`:

- producer may publish command k;
- producer must not publish k+1 until every actuator **present in k** has adopted k;
- no queue/backlog exists.

Across lifecycle invalidation:

- an old-run command in the mailbox is cancelled by `run_id`;
- it must **not** block the new run;
- a new current-run command may overwrite the stale old-run slot even if the old command was never jointly adopted.

Therefore “single-inflight” is a **per-current-run** invariant, not a permanent lock on the last mailbox record.

The session architecture must still guarantee one active command-producing workflow at a time. Do not add multi-producer arbitration infrastructure.

## 5.5 Publication race ordering

Publication and lifecycle transitions must keep a short common ordering fence.

The important race is:

```text
publish command
vs.
pause / stop / quit / timeout / fault
```

The implementation must ensure one of two outcomes:

1. publication wins the lifecycle lock, then a later revocation changes `run_id`; workers reject the stale command unless it had already crossed their SDK admission fence;
2. revocation wins first, then publication sees the changed state/run and fails without writing a current command.

Do not imply that software can retract a vendor SDK call that already crossed its admission fence. It cannot.

## 5.6 Arm adoption

The arm worker marks a command adopted only after:

1. it observes a new `(run_id, command_id)`;
2. arm is present in that command;
3. current command still owns motion at the last software fence immediately before SDK admission;
4. worker-owned hard finite/shape/joint checks pass;
5. xArm SDK accepts the target.

Arm state must expose:

```text
last_adopted_run_id
last_adopted_command_id
last_adopted_monotonic_ns
```

An old-run SDK result may still be published later as a historical adoption fact if the call had already crossed the fence before revocation. Its old `run_id` must never authorize a new run.

For a current-run hard validation or SDK failure, fail closed rather than silently leaving the command unacknowledged.

## 5.7 XHand adoption versus reach

Keep two distinct facts.

### Adopted

A new high-level hand target is adopted when:

1. hand worker observes a new current-run command;
2. hard target checks pass;
3. worker installs it as the active high-level target;
4. worker computes the next bounded low-level setpoint;
5. XHand SDK accepts at least that first bounded setpoint.

Then publish:

```text
last_adopted_run_id
last_adopted_command_id
last_adopted_monotonic_ns
```

### Reached

Continue bounded low-level slew:

```text
q_sdk_next =
    q_sdk_current
    + clip(q_target - q_sdk_current, -delta_max, +delta_max)
```

Only when the exact high-level target reaches the existing endpoint-acceptance semantics publish:

```text
last_reached_run_id
last_reached_command_id
last_reached_monotonic_ns
```

A streaming target can be superseded after adoption but before exact reach. In that case its command ID may never become `last_reached_command_id`. Do not fabricate reach.

Streaming gates on **adopted**.

Blocking hand home or another operation that truly needs endpoint completion may wait on **reached**.

If the current measured/SDK state already equals the new target and the exact target is accepted, adopted and reached may advance together.

## 5.8 Exact adoption comparison

For command k, adoption is true only when the relevant worker reports the exact pair:

```text
worker.last_adopted_run_id == k.run_id
and
worker.last_adopted_command_id == k.command_id
```

Do not rely on FIFO-style `>= sequence` proof.

For a command with no hand target, hand adoption is not required. Likewise for arm-absent operations.

## 5.9 Worker repeated-read behavior

Workers may observe the mailbox repeatedly.

- Arm must not repeatedly re-send the same high-level `(run_id, command_id)` as a new command.
- Hand must not re-adopt the same command, but it must continue bounded slew toward its currently active target on later worker ticks.
- Stale old-run mailbox records are ignored, not faulted.
- A new run with a new command ID must remain distinguishable from every previous command.

---

# 6. Partial adoption is real and must not be hidden

Arm and hand SDK admission is not atomic.

This race is physically possible:

```text
publish coupled command k

arm crosses SDK fence and accepts k

C / S / timeout / fault revokes run

hand has not accepted k
```

The old task incorrectly allowed “drop the pending row” for every non-jointly-adopted command. That would erase a real physical arm action.

The new implementation must distinguish:

```text
NO_ADOPTION
JOINT_ADOPTION
PARTIAL_ADOPTION
ADOPTION_UNKNOWN
```

Do not build a general state-machine framework; this is a narrow recording truth requirement.

## 6.1 Producer-local pending descriptor

After successful publication, keep at most one small pending command/sample descriptor containing:

- selected observation/anchor;
- raw/mapped intent required by raw recording;
- returned `RobotCommand`;
- action target representations;
- current frame status/context.

Do not append its normal action row immediately.

## 6.2 Normal completion

If every present actuator reports exact adoption for that command:

- finalize one normal command row;
- store actual arm/hand adoption timestamps;
- clear the pending descriptor;
- allow the next current-run command.

## 6.3 Lifecycle boundary before joint adoption

On C/S/Q/timeout/fault/new-run boundary:

- revoke/fence the old run immediately;
- do **not** immediately relabel the command as executed;
- do **not** blindly discard the pending descriptor if any physical adoption may have occurred.

Resolve it from post-boundary worker state when possible.

A fresh post-boundary state sample from each still-live relevant worker must be newer than the revocation/pause boundary before using it to classify the old command. This allows an SDK call admitted just before revocation to return and publish its historical old-run adoption fact.

Then:

### Neither relevant worker adopted

- no physical high-level command was observed as adopted;
- dropping the pending normal row is acceptable.

### Every relevant worker adopted

- finalize the command row with the actual adoption facts;
- it was a real jointly adopted command even if a lifecycle boundary followed immediately.

### Only a subset adopted

- persist a diagnostic row if recording is still available;
- store target + per-actuator adoption bits/timestamps;
- use a dedicated small frame status such as `PARTIAL_ADOPTION`;
- never treat this row as a normal training label;
- mark the episode technically/integrity-invalid for default training export because the coupled physical action was split.

### Cannot determine because worker died / state never became fresh / shutdown removed evidence

- persist `ADOPTION_UNKNOWN` if the recorder is still available;
- mark the episode technically/integrity-invalid;
- never fabricate adoption timestamps.

This logic is especially important around C pause and stop/fault boundaries.

Before any later operation publishes a new command that could overwrite worker `last_adopted_*` evidence, resolve the old pending command from fresh post-boundary state or explicitly classify it `ADOPTION_UNKNOWN` and invalidate the episode. This is a short **adoption-accounting barrier**, not a motion queue: it exists only to preserve truthful evidence across lifecycle boundaries.

Do not block normal control indefinitely to resolve historical adoption. Use existing bounded lifecycle/shutdown timing; if the bounded accounting barrier cannot resolve the old command, record/mark `ADOPTION_UNKNOWN`, invalidate the episode, and only then allow a later command-producing operation to proceed.

---

# 7. Lifecycle: simplify generation into run_id

Keep:

```text
SafetyState:
    DISARMED
    ARMED
    RUNNING
    FAULT
```

Use a monotonic `run_id` as the command-admission epoch identity.

Important semantic clarification:

- every RUNNING epoch has a unique current `run_id`;
- command revocation/invalidation advances/fences the ID;
- ARMED maintenance operations that use the shared RobotCommand mailbox may use the current run ID with an explicit required safety state;
- aborting such an operation must invalidate its command epoch before another operation reuses the mailbox.

Do not treat `run_id` as “policy trial ID”; it is a runtime command-admission epoch.

## 7.1 Blocking inference fence

Required race:

```text
captured_run_id = current run

predict() blocks

S / Q / timeout / fault / C boundary occurs
run_id changes

predict() returns old chunk
```

The returned chunk must be discarded before publication.

## 7.2 Worker SDK fence

Immediately before a current command is admitted to an SDK, recheck:

- runtime still running;
- no estop;
- no sticky error;
- safety state permits that operation;
- command `run_id` equals current `run_id`.

The short lock is released before SDK IO. A vendor call already admitted before revocation may finish afterward; that is a historical old-run physical fact, not permission for more old-run commands.

## 7.3 Remove old lifecycle/FIFO concepts after full cutover

Remove when no caller depends on them:

- `run_generation_base_sequence`;
- arm/hand consumed sequence watermarks;
- FIFO capacity/FULL;
- FIFO commit receipts;
- FIFO exact retry state;
- command queue corruption handling;
- per-command dispatch expiry whose only hazard was queued backlog;
- `RunEndReason.COMMAND_EXPIRED` if no independent use remains.

Do not delete a freshness/deadline mechanism if it protects a different real hazard. Trace first.

---

# 8. RuntimeChannels target

The target is approximately:

```text
state/data
----------
arm_state_ring
hand_state_ring
camera_ring
vr_ring
pointcloud_ring          # only active when requested
record_sample_ring

command
-------
robot_command_ring       # latest mailbox semantics
next_command_id

runtime/lifecycle
-----------------
run_id
run_started_monotonic_ns
minimal run terminal fact(s) still required
safety_state
is_running
is_recording
error_state
estop_request
quit_requested
policy start/stop/home authorization fields still genuinely needed
motion_lock

recording
---------
record_control_q
record_result_q
minimal recorder completion/transport state required for safe shutdown

process health
--------------
heartbeats
ready_flags

home
----
arm_home_q
minimal verified completion facts
```

Prefer executor adoption facts in arm/hand state records, not duplicate shared ACK scalars.

Delete fields only after all call sites are migrated.

---

# 9. Teleop workflow

Keep teleop and policy eval as separate workflows.

Normal operator controls:

```text
H   home
B   begin
C   pause/resume
S   stop/save
D   stop/discard
Q   quit
ESC emergency stop
```

B is not “resume”; C owns pause/resume.

## 9.1 Recording-enabled startup

When recording is enabled:

- arm/hand/VR readiness remains required as today;
- camera and recorder must also be ready before B can start a recorded episode;
- collection may stay ARMED for safe home/quit if recorder/camera is unavailable;
- do not allow a degraded unrecorded B.

When recording is explicitly disabled, no-record debug teleop remains valid.

## 9.2 Mid-episode recorder/camera failure

Do not convert a pure evidence failure into hardware FAULT unless there is also a physical runtime fault.

For recorded teleop:

- current episode becomes invalid/incomplete;
- revoke active RUNNING teleop to ARMED;
- do not continue pretending collection is valid;
- retain H/Q/safe recovery where possible;
- block another recorded B unless recording resources are healthy again;
- final session result is non-zero.

Do not retain a generic “evidence service priority matrix” if simple workflow-specific handling replaces it.

## 9.3 C pause/resume under the new command protocol

On C pause:

1. establish the pause/revocation monotonic boundary;
2. fence the old `run_id`;
3. stop publication of new commands;
4. keep any potentially partially adopted pending descriptor until section 6 classification is possible;
5. clear teleop proposal/reference state as current behavior requires;
6. require fresh arm/hand/VR feedback newer than the pause boundary.

On resume:

1. resolve any old pending adoption truth sufficiently for safe/raw-data accounting;
2. establish the fresh current command epoch;
3. re-anchor from fresh measured robot state and current VR state;
4. do not reuse an unpublished pre-pause target.

Keep the existing same-session C behavior otherwise. Do not redesign pause UX in this task.

## 9.4 Teleop timing

Keep a fixed teleop control opportunity grid:

```text
t_k = t_0 + k * dt_teleop
```

At each due grid:

- process operator controls first;
- if a current-run previous command is still waiting for joint adoption, do not publish a new command;
- count executor-lag/missed-command opportunity;
- do not catch up with burst commands;
- advance to the next grid deadline.

It is acceptable for persisted raw rows to be fewer than nominal grid opportunities during executor lag; preserve actual timestamps rather than inventing rows that would need to be reordered around a pending command.

Do not build a second high-frequency executor scheduler merely for keyboard bookkeeping. Prefer event-or-deadline waiting where practical.

Teleop control rate belongs to teleop/data-collection config, not to a learned model spec.

---

# 10. Teleop and raw action semantics

For a normal command row, the canonical executed high-level target is:

```text
post-IK / post-projection / producer safety-admitted
arm + hand RobotCommand target
with all present actuator adoption facts confirmed
```

Do not use as the canonical normal BC action:

- raw VR pose;
- pre-IK Cartesian intent;
- XHand intermediate slew setpoint;
- measured qpos;
- a merely published command;
- a partially adopted command.

Keep human/mapped intent separately where the current raw schema supports it.

The raw chain should remain recoverable:

```text
human intent
   ↓
mapped/retargeted robot intent
   ↓
high-level executable target
   ↓
per-actuator adoption facts
   ↓
measured proprioception/tactile/vision response
```

---

# 11. Raw schema migration: v30 → v31

Current raw schema is v30. The command/action truth changes, so bump explicitly to v31 unless local HEAD already consumed v31; then use the next unused version.

Do not silently reinterpret v30.

Do not add runtime compatibility branches that pretend v30 has new adoption facts.

## 11.1 Required command fields

The v31 row contract should contain the equivalent of:

```text
command_id                         uint64
command_run_id                     uint64
command_issued_monotonic_ns        uint64

command_arm_present                bool
command_hand_present               bool

arm_command_adopted                bool
hand_command_adopted               bool

arm_command_adopted_monotonic_ns   uint64
hand_command_adopted_monotonic_ns  uint64
```

Use explicit zero/false sentinel semantics for “not present / not adopted / unknown timestamp” and document the distinction through presence + adoption bits.

Keep the existing action target arrays unless a rename is clearly worth the migration cost. A schema-version bump makes changed semantics explicit; do not perform gratuitous dataset renames.

## 11.2 Frame statuses

Keep the existing small statuses and add only the two needed interruption statuses.

At minimum:

```text
OK
HELD
IK_FAIL
SAFETY_REJECT
RETARGET_FAIL
PARTIAL_ADOPTION
ADOPTION_UNKNOWN
```

Do not create a broad error code framework.

## 11.3 Remove queue truth

Remove `flag_action_queued` / `action_queued` as persisted truth. “Queued” is no longer meaningful.

Do not replace it with another redundant transport flag if command identity + presence + adoption already provide the truth.

## 11.4 Replay/export send mask

Current replay/processing may derive a send mask from `flag_action_queued`. Replace that logic deliberately.

For v31, a row is a candidate “new executable command row” only when:

- `command_id > 0`;
- every actuator present in the command has its adoption bit true;
- the command ID is different from the previously represented command;
- frame status is not partial/unknown;
- any additional current replay validity requirements still pass.

Held/failure rows that reference the previous jointly adopted command must not cause duplicate command publication merely because they contain action target arrays.

Partial/unknown adoption rows are never normal replay/training command rows.

Update raw processing/export/replay readers consistently.

## 11.5 Normal, held, partial rows

Normal command row:

- new command ID;
- all present adoption bits true;
- actual adoption timestamps.

Held / IK / safety / retarget row:

- no new command;
- may reference the last jointly adopted command ID/target where scientifically correct;
- must not claim a new publication/adoption.

Partial adoption row:

- records the interrupted new command ID and intended high-level targets;
- per-actuator adoption bits/timestamps show what physically happened;
- episode marked invalid/non-exportable by default for training.

Unknown adoption row:

- records what is actually known;
- unknown timestamps stay zero/false;
- episode invalid/non-exportable by default.

## 11.6 Episode-level integrity

Do not hide command-level corruption by filtering one row from an otherwise sequential demonstration.

If a recorded episode contains `PARTIAL_ADOPTION` or `ADOPTION_UNKNOWN`:

- raw data may still be saved for diagnosis;
- episode technical/integrity status must be invalid;
- default training export must reject/skip it unless an explicit future recovery tool is written.

This is especially important because later robot state may depend on the partial physical action.

---

# 12. Observation/provenance simplification

Preserve scientific timing, remove duplicate internal transport proofs.

Keep:

- observation anchor;
- arm source timestamp;
- hand source timestamp;
- VR source timestamp;
- camera source timestamp;
- observation validity;
- camera/tactile validity;
- causal latest-before selection;
- policy history semantics;
- inter-modality skew/freshness rules that directly affect scientific validity.

Simplify only when no scientific consumer needs it:

```text
source <= receive <= publish <= ring_commit <= anchor
```

should not be re-proven independently in every layer.

Producer/ring owns internal record integrity. Consumer owns source causality/freshness relative to its anchor.

Do not remove a timestamp because it looks implementation-specific without tracing actual dataset/observation consumers.

---

# 13. Policy evaluation workflow

Target runner shape:

```python
while session_active:
    handle_parent_requests()

    if not episode_running:
        continue

    resolve_pending_command_if_needed()

    if current_run_has_pending_unjointly_adopted_command():
        note_executor_lag()
        wait_without_catchup()
        continue

    if action_chunk_empty():
        obs = build_policy_observation()
        captured_run_id = current_run_id()

        chunk = model_runtime.predict(obs)

        if current_run_id() != captured_run_id:
            discard(chunk)
            continue

        validate_chunk(chunk)
        store_local_unexecuted_chunk(chunk)

    raw_action = pop_next_action()
    decoded = decode_or_ik(raw_action)

    if recoverable_ik_or_workspace_miss:
        clear_unexecuted_chunk_suffix()
        continue

    target = project_and_safety_check(decoded)
    command = publish_robot_command(target, required_state=RUNNING)

    if command_published:
        create_one_pending_record_descriptor(command, obs, raw_action)

    pace_without_catchup()
```

## 13.1 Exact policy timing rule

Do not leave timing ambiguous.

For action commands in one continuous RUNNING episode:

```text
earliest_next_publish =
    last_successful_publish_monotonic + PolicySpec.control_dt_s
```

Rules:

1. Never publish before that deadline.
2. Also never publish while the previous current-run command is not jointly adopted.
3. If adoption or synchronous inference finishes after the deadline, publish the next eligible command as soon as both conditions are satisfied.
4. After a late publish, anchor the following deadline to that **actual successful publish time**.
5. Never emit catch-up bursts for missed periods.
6. First action of a newly inferred chunk may publish immediately after inference only if the previous command adoption gate is satisfied and the earliest-next-publish rule is satisfied.
7. When the local chunk becomes empty, the next inference may begin at the existing query boundary semantics, but no returned command bypasses the publication cadence.
8. Preserve actual timestamps in raw output.

This intentionally converts model/control delay into slower real wall-time execution rather than stale-command catch-up.

## 13.2 Recoverable replanning

Keep current useful behavior:

- ordinary EE IK/no-solution or recoverable workspace miss can clear the unexecuted local chunk suffix;
- re-observe and re-infer in the same episode;
- do not convert a normal recoverable miss into hardware FAULT.

Hard contract violations/internal exceptions remain failures.

## 13.3 External model boundary

The repository goal includes models trained elsewhere, but do **not** build a plugin framework now.

Keep training-repository-specific code confined to the narrow deployment boundary:

- parent metadata inspection;
- one child-side model loader;
- public `PolicySpec`;
- loaded runtime’s minimal current methods such as `predict/reset/close`, according to the actual public API.

The core real-robot runner should operate on the existing validated spec + canonical observation/action arrays, not on model architecture internals.

Continue using the public `dexmani_policy.deployment` API for the currently integrated model.

When a second concrete training repository is actually integrated, add the smallest concrete adapter needed then. Do not pre-build a registry.

---

# 14. run_policy.py reproducibility

Make `examples/run_policy.py` a trustworthy experiment entry point.

Required:

1. Add `--config`.
2. Resolve it with the existing `resolve_experiment_config(yaml_path=...)` path; do not create a second configuration system.
3. Normalize internal `num_trials` naming to `num_episodes` unless code semantics genuinely require two separately named concepts.
4. Keep task success separate from technical validity.
5. Before any hardware worker starts, write one complete resolved session configuration containing:
   - full resolved `ExperimentConfig`;
   - policy experiment selector;
   - resolved checkpoint/artifact;
   - inference steps;
   - seed;
   - device;
   - relevant `PolicySpec` facts including control dt, observation history/modalities, action representation and action chunk length;
   - requested episode count;
   - per-episode duration;
   - repository HEAD / dirty-state fact when cheaply available without mutating the repo.
6. Do not connect hardware during parsing, config resolution, metadata inspection, or writing this file.
7. Prefer one canonical resolved session YAML/JSON over overlapping partial config files.
8. Keep operator summary concise.

Do not require the external policy package to be a Git checkout merely to record a version. Record what can be resolved reliably; do not invent version metadata.

---

# 15. Episode result semantics

For policy eval, separate:

```text
technical_status: valid | invalid
termination_reason
task_success: true | false | unknown
```

A failed grasp can be technically valid.

Examples of technical invalidity:

- required camera failure;
- recorder failure when eval evidence is required;
- policy crash;
- hardware fault;
- partial/unknown command adoption;
- invalid observation that terminates the episode.

Task success can remain offline/manual.

Store the result in the existing episode/session metadata path or a simple sidecar already consistent with the repository; do not build a results database.

For teleop collection, use the same integrity concept where useful, but do not force task-success labeling.

---

# 16. Recording process and failure policy

Keep RecorderIO as a separate process with SHM sample transport.

Justification:

- RGB-D payload size;
- video/image encoding;
- disk IO;
- avoiding large queue copies.

Do not move serialization into the control loop.

Eventually simplify the logical client API toward:

```text
start_episode(metadata)
append(frame)
finish_episode(save=..., reason=...)
```

but do not rewrite recorder transaction internals until command + raw-row semantics are stable.

## 16.1 Policy eval recorder failure

When policy eval recording is required:

- stop/fence the current episode;
- mark it technically invalid;
- request clean session shutdown/non-zero result;
- do not claim hardware FAULT unless hardware also failed;
- do not continue running additional unrecorded eval episodes.

This is simpler and scientifically safer than “control continues but evidence is missing”.

## 16.2 Teleop recorder failure

Use section 9.2: current demonstration invalid, motion returns to ARMED when safe, manual recovery/home/quit remain possible, no new recorded B until recorder/camera health is restored.

---

# 17. Process supervision

Keep supervision that maps to real hazards:

- readiness;
- unexpected critical worker death;
- arm heartbeat;
- hand heartbeat;
- camera heartbeat when camera is required;
- recorder health when recording is required;
- estop;
- parent-side episode/session timeout;
- verified shutdown before SHM unlink.

Do not heartbeat-fault the policy worker merely because synchronous `predict()` blocks.

VR health should primarily use actual VR source freshness where appropriate.

Preserve `runtime/processes.py` stop → join → terminate → kill → confirm behavior unless a concrete part is proven unnecessary.

Remove generic service/evidence matrices when direct workflow-specific handling replaces them.

---

# 18. Home, replay, calibration, keyboard teleop

Do not treat these as “later callers” after workers already switched protocols. They are command publishers/waiters and must be part of the same command-transport cutover.

Search all actual callers of:

- `publish_command`;
- `wait_command_accepted`;
- `ActionCandidate`;
- `CommandStreamConsumer`;
- `coupled_cmd_ring`;
- old acceptance generation/sequence fields.

Known current paths include at least:

- teleop control grid;
- keyboard teleop/session;
- policy runner;
- hand homing;
- replay;
- camera calibration motion;
- arm/hand workers.

Trace repository-wide; do not assume this list is exhaustive.

## 18.1 Arm home

Keep arm home as its special physical operation if that remains the simplest safe design. It owns:

- planned waypoints;
- control-mode behavior;
- workspace/table/collision constraints;
- abort;
- settled feedback;
- mode restoration.

Do not force it into RobotCommand merely for API uniformity.

Its old generation cancellation should become the new `run_id`/operation-epoch fence without changing physical semantics.

## 18.2 Hand home

If hand home continues to use the shared RobotCommand mailbox:

- required safety state is explicitly ARMED;
- use current operation `run_id`;
- wait for **hand reached**, not merely hand adopted;
- abort invalidates the operation epoch;
- no FIFO retry exists.

A dedicated hand-home request is acceptable only if it is clearly simpler than sharing RobotCommand; do not create a generic “operation command” framework.

## 18.3 Replay

Trace the intended replay semantics per step.

- For streaming steps, gate next command on adoption, not exact XHand reach, unless the existing replay intentionally requires endpoint blocking.
- For startup hand home, require reach.
- Replace `flag_action_queued`-derived send logic using the v31 command identity/adoption contract from section 11.
- No backlog/catch-up.

## 18.4 Calibration motion

Preserve its exact required SafetyState and blocking/adoption semantics.

Do not globally replace every old acceptance wait with the same new wait. Decide per call site whether the physical requirement is:

- arm adopted;
- hand adopted;
- hand reached;
- settled measured state;
- dedicated home completion.

---

# 19. Configuration ownership

Do this only after command/data semantics are stable.

Current config mixes teleop concepts into policy-oriented fields.

Desired conceptual ownership:

```yaml
arm: ...
hand: ...
camera: ...
pointcloud: ...
safety: ...

teleop:
  control_hz: ...
  vr_mapping: ...
  retargeting/smoothing: ...

recording:
  episode paths / durations / writer settings: ...

eval:
  episode-count / duration / recording behavior: ...
```

Important:

- learned-model control dt belongs to `PolicySpec`;
- teleop grid dt belongs to teleop config;
- do not maintain two authoritative policy control rates.

However, do not perform broad cosmetic YAML/dataclass churn merely to match the conceptual tree. Move a field only when it removes a real source-of-truth ambiguity or substantially simplifies code.

The minimum required result is that teleop timing is not semantically owned by a learned-policy setting, and policy deployment validates/uses `PolicySpec.control_dt_s` without an unnecessary duplicate runtime truth.

Update every call site atomically for a moved field.

---

# 20. File-level target

Use dependency tracing; this is not permission for blind deletion.

## `examples/run_policy.py`

- `--config`;
- full resolved session config;
- episode naming cleanup;
- no hardware during preflight.

## `examples/collect_teleop.py`

- keep explicit `--no-record`;
- preserve C through the teleop loop;
- keep a simple research-facing CLI.

## `dexmani_real/ipc/schema.py`

- replace FIFO command dtype with `ROBOT_COMMAND_DTYPE`;
- arm adopted identity/timestamp;
- hand adopted + reached identity/timestamps;
- record-sample v31 command/adoption fields;
- remove FIFO sequence identity from worker state when no longer used.

## `dexmani_real/ipc/channels.py`

- mailbox-style robot command ring;
- shared monotonic `next_command_id`;
- simplified `run_id`;
- remove FIFO consumed/base fields after full cutover;
- keep sensor rings, recorder resources, readiness/health.

## `dexmani_real/runtime/safety.py`

- keep SafetyState and short motion lock;
- simplify generation to command-admission `run_id`;
- preserve begin/revoke/invalidate race ordering;
- remove FIFO/backlog/expiry transaction code after all callers migrate.

## `dexmani_real/robot/commands.py`

Rewrite around the smallest useful concepts:

- `RobotCommand`;
- current feedback snapshot needed for producer safety;
- safety gate/result if it still carries a real decision;
- command publication;
- exact adoption status helpers;
- blocking helpers for adopted/reached callers.

Remove transport-only layers when no longer useful:

- `ActionCandidate`;
- `PublishResult`;
- `AcceptanceResult`;
- `PreparedCommand`;

Do not force removal if one tiny result type still materially clarifies a real recovery decision. Avoid wrappers that only rename booleans.

## `dexmani_real/robot/arm_worker.py`

- mailbox/latest new command;
- exact run/id handling;
- one arm SDK send per new high-level command;
- hard worker validation;
- historical old-run adoption may publish but never authorize current motion.

## `dexmani_real/robot/hand_worker.py`

- preserve same-sample qpos/current/tactile;
- active high-level target;
- adopted after first accepted bounded setpoint;
- continue slew on later ticks;
- reached only at exact endpoint;
- no FIFO consumer/watermark.

## `dexmani_real/teleop/control_loop/grid.py`

Keep:

- causal observation;
- VR mapping;
- retargeting;
- IK;
- projection;
- safety.

Delete:

- FIFO FULL retry;
- exact pending FIFO candidate;
- queue terminal-acceptance hacks.

Add:

- one pending command/sample descriptor;
- adoption gating;
- partial-adoption accounting;
- C integration.

## `dexmani_real/teleop/loop.py`

- simplify state made obsolete by FIFO;
- keep C;
- keep safe re-anchor;
- simplify polling/scheduling only if behavior remains responsive.

## `dexmani_real/teleop/episode_samples.py`

- v31 command/adoption fields;
- frame statuses including partial/unknown;
- actual source timestamps;
- no `action_queued`;
- do not over-prove internal publish/ring timestamps.

## `dexmani_real/teleop/session.py`

- recording resources required for recorded B;
- no silent evidence downgrade;
- recorder/camera failure is collection failure, not automatically hardware FAULT;
- simplify generic service/evidence bookkeeping.

## `dexmani_real/deployment/runner.py`

Target:

```text
observe → synchronous infer → local chunk → decode/project/safety
       → publish → adoption gate → pace
```

Keep run-id fence around blocking inference.

## `dexmani_real/deployment/observation.py`

Keep causal/history logic. Simplify only duplicate transport provenance.

## `dexmani_real/deployment/session.py`

Keep process ownership, parent controls/timeouts, policy worker isolation, required recording supervision. Simplify evidence bookkeeping.

## `dexmani_real/teleop/keyboard_session.py`

Migrate old FIFO publication/ACK semantics in the same atomic command cutover.

## `dexmani_real/replay/replayer.py`

Migrate publication/waits/send-mask semantics in the same atomic command cutover.

## `dexmani_real/calibration/camera/motion.py`

Migrate old publication/acceptance in the same atomic command cutover.

## `dexmani_real/robot/hand_homing.py`

Migrate old FIFO publication to new adopted/reached semantics in the same atomic cutover.

## `dexmani_real/robot/arm_homing.py`

Preserve physical planning/safety; migrate generation naming/fencing only as required.

## `dexmani_real/recording/*`

- v31 schema;
- writer/reader/frame/sample transport update;
- preserve RecorderIO process;
- protocol simplification later.

## `dexmani_real/runtime/supervisor.py`

Keep real liveness/safety supervision; remove generic evidence service matrix if obsolete.

## `dexmani_real/runtime/processes.py`

Mostly preserve.

## `dexmani_real/robot/projection.py`

Preserve. It is real command shaping, not infrastructure overhead.

---

# 21. Mandatory execution phases

Do the phases in order. “Phase complete” means code compiles and no intentionally broken old/new boundary remains.

## Phase 0 — audit + low-risk reproducibility fixes

Before changing transport, repository-wide search at least:

```text
coupled_cmd_ring
command_stream
CommandStreamConsumer
run_generation
run_generation_base_sequence
arm_cmd_consumed_sequence
hand_cmd_consumed_sequence
ActionCandidate
publish_command(
wait_command_accepted(
PUBLISH_REASON_FIFO_FULL
expires_monotonic_ns
accepted_target_sequence
action_queued
flag_action_queued
```

Make working notes only; do not add a permanent audit document.

Trace every command publisher/waiter and both workers.

Implement low-risk independent fixes:

- `run_policy --config`;
- full resolved session config;
- episode naming cleanup where safe;
- recorded teleop no silent downgrade.

Verify C exists before and after.

Run offline checks.

## Phase 1 — design the cutover from actual code

Do not introduce a second live transport yet.

Using the audit, define the exact local implementation plan for:

- RobotCommand wire dtype;
- arm state adoption fields;
- hand adopted/reached fields;
- shared next_command_id;
- run_id lifecycle;
- v31 raw row fields;
- every old command caller’s new adopted/reached requirement.

If a helper/type can be added without creating a parallel runtime path, it may be added. Do not leave unused “future” abstractions.

The output of Phase 1 is a coherent edit plan in Codex working context, not a permanent design file.

## Phase 2 — one atomic command + raw-data cutover

This is intentionally broad. Do **not** migrate workers first and leave old publishers for a later phase.

In one coordinated implementation phase migrate:

- `ipc/schema.py`;
- `ipc/channels.py`;
- `runtime/safety.py`;
- `robot/commands.py`;
- arm worker;
- hand worker;
- teleop grid/loop publisher;
- keyboard teleop publisher;
- policy runner publisher;
- hand homing;
- replay;
- camera calibration motion;
- every other repository-wide old command publisher/waiter found in Phase 0;
- record sample wire dtype;
- raw storage schema v31;
- frame builder;
- recorder writer/reader validation;
- raw processing/export;
- replay/send-mask consumers of old `action_queued`.

The phase is not complete until:

- every command publisher speaks the new protocol;
- both workers speak the new protocol;
- every blocking caller has explicit adopted/reached/settled semantics;
- normal recording row truth matches executor adoption;
- partial/unknown adoption is representable;
- v31 readers/writers/processors agree.

Only then:

- delete `ipc/command_stream.py`;
- remove old FIFO resources;
- remove old FIFO imports/status/retry code;
- remove generation-base/watermark fields.

Do not keep a compatibility shim for the deleted internal FIFO.

Run all smoke scenarios below before proceeding.

## Phase 3 — simplify teleop and policy state

Now remove state that existed only for FIFO/evidence infrastructure:

- pending FIFO publication retries;
- FIFO depth/backpressure logging;
- obsolete policy dispatch state;
- generic evidence-service matrices;
- unnecessary high-frequency polling if event/deadline waiting is simpler.

Keep:

- C pause/resume;
- fresh re-anchor;
- safety state;
- run-id fence;
- one pending command recording descriptor;
- current research-useful frame statuses;
- useful session metrics.

Policy required-recording failure should end invalid eval rather than continue evidence-less.

## Phase 4 — observation/provenance + config ownership

- remove duplicate transport proof checks only after tracing consumers;
- preserve causal/source-time semantics;
- eliminate duplicate control-rate truth;
- move only clearly mis-owned config fields;
- update README stable workflow/terminology if it became stale.

Do not turn this into a wholesale configuration rewrite.

## Phase 5 — recorder protocol simplification

Only after command/data semantics are stable:

- reduce recorder client/transaction state where possible;
- preserve RecorderIO process + SHM;
- preserve safe finalization and verified shutdown;
- avoid a new generic recorder framework.

---

# 22. Required offline smoke scenarios

Do not commit a test suite. Use temporary scripts, inline Python, fake/shared-memory pure logic, or existing non-hardware constructors.

## S1 — process-lifetime command IDs

Simulate publication from two different logical producers/operations sharing one RuntimeChannels.

Verify command IDs are unique/monotonic and never reset per producer.

## S2 — single-inflight within one run

- publish command 1;
- arm adopts 1;
- hand has not adopted 1;
- command 2 cannot replace command 1 in the same run.

## S3 — joint adoption unlocks next command

- exact arm + hand adoption pair matches command 1;
- command 2 may publish.

## S4 — stale old run does not block new run

- command 1 belongs to run N and remains unadopted;
- lifecycle advances to run N+1;
- old run-N mailbox record cannot cross SDK;
- a valid run-N+1 command can publish and replace/advance past the stale slot.

## S5 — hand adopted != reached

- first bounded hand setpoint accepted;
- adopted ID advances;
- reached ID stays previous;
- streaming can advance after required arm adoption;
- no fake reach.

## S6 — hand exact reach

Simulated slew reaches exact high-level target; reached identity advances only then.

## S7 — stale SDK fence

- worker observes run-N command;
- run becomes N+1 before SDK admission;
- worker does not send old command.

Also check the allowed historical case:

- SDK admission already happened in N;
- run becomes N+1 before SDK return;
- returned acceptance may be published as old-run historical adoption but cannot authorize N+1.

## S8 — worker hard rejection does not deadlock

A current-run published target that fails worker-owned hard validation must cause a fail-closed runtime/worker outcome, not an endless pending command with no ACK.

## S9 — blocking inference fence

- capture run N;
- simulate blocking predict;
- advance to N+1;
- returned N chunk is discarded.

## S10 — policy cadence, no catch-up

Simulate:

- successful publish at t0;
- adoption/inference finishes after t0 + dt;
- next publish occurs late;
- following deadline is late_publish + dt;
- no burst sends.

## S11 — C pause with no adoption

- command published;
- neither worker adopted;
- C fences run;
- post-boundary fresh worker states confirm no adoption;
- pending normal row may be dropped;
- resume requires fresh re-anchor;
- C remains recognized.

## S12 — C/stop partial adoption

- arm adopted command;
- hand did not;
- lifecycle fences run;
- post-boundary state resolves partial adoption;
- raw diagnostic row stores exact known facts;
- episode integrity invalid;
- row is not normal training/replay action.

## S13 — adoption unknown

Simulate worker loss before final post-boundary evidence.

Verify:

- no adoption fact is invented;
- episode becomes invalid;
- unknown diagnostic state is representable when recorder is available;
- the bounded adoption-accounting barrier resolves to UNKNOWN before any later command is allowed to overwrite the worker adoption evidence.

## S14 — normal recording label truth

Normal row is not finalized before all present actuator adoptions.

After joint adoption, row’s command ID/run ID/issued/adoption timestamps match the command and worker facts.

## S15 — held/failure row

IK/safety/retarget failure:

- publishes no new command;
- preserves prior physical target;
- records explicit status;
- does not create a new command ID;
- replay send mask will not resend it as a new command.

## S16 — recording-required teleop

- explicit no-record mode works without recorder;
- recording-enabled mode cannot start B unrecorded when camera/recorder unavailable.

## S17 — v31 replay/export mask

Given rows containing normal command, held row, partial row, and another normal command:

- only jointly adopted new command IDs become replay/send actions;
- held row causes no duplicate send;
- partial/unknown episode is rejected/skipped by default training export.

## S18 — imports have no hardware side effects

Import modified pure/runtime modules offline where possible. Ordinary imports/constructors must not connect devices.

---

# 23. Required repository-wide cleanup checks

At the end, verify old transport concepts are absent where expected.

Search for:

```text
PUBLISH_REASON_FIFO_FULL
run_generation_base_sequence
arm_cmd_consumed_sequence
hand_cmd_consumed_sequence
command fifo full
coupled command FIFO
CommandStreamConsumer
flag_action_queued
```

`dexmani_real/ipc/command_stream.py` should be deleted after the atomic cutover.

`ActionCandidate`, `PreparedCommand`, `PublishResult`, `AcceptanceResult`, `expires_monotonic_ns`, and `COMMAND_EXPIRED` should be absent if no traced non-FIFO responsibility remains.

Do not delete by grep alone; trace semantics first.

Verify C explicitly remains, using searches appropriate to the actual keyboard implementation.

Also inspect for stale README/comments that still describe:

- command FIFO;
- FULL retry;
- generation watermarks;
- evidence continuing after required recorder loss;
- v30 as current schema.

---

# 24. Standard offline checks

After each meaningful phase:

```bash
python -m compileall -q dexmani_real examples
ruff check --select F401,F821,F822,F823 dexmani_real examples
git diff --check
```

If Ruff is unavailable, report it; do not install it.

Run focused offline checks for changed pure logic, including:

- dtype shape/field consistency;
- command/run identity;
- adoption classification;
- policy cadence;
- frame builder/storage schema agreement;
- v31 processing/export/replay selection;
- transforms/FK/IK/projection only when affected.

Never use a hardware example as a “test”.

---

# 25. Final acceptance checklist

## Architecture

- [ ] Teleop and policy remain separate workflows.
- [ ] Arm and hand remain separate SDK-owning workers.
- [ ] Recorder remains a process.
- [ ] Command FIFO/backlog is gone.
- [ ] Single-inflight is enforced within the current run.
- [ ] Stale old-run slot does not block a new run.
- [ ] command_id is unique/monotonic across producer-process changes.
- [ ] No generic strategy/backend/plugin framework was added.

## Safety

- [ ] SafetyState remains.
- [ ] run_id fences stale predict/results/commands.
- [ ] publication and revocation share a short ordering fence.
- [ ] worker rechecks run/state immediately before SDK admission.
- [ ] already-admitted SDK calls are treated honestly as non-retractable.
- [ ] worker hard validation failure fails closed.
- [ ] arm hard limits remain.
- [ ] XHand slew remains.
- [ ] estop remains.
- [ ] safe disconnect remains.
- [ ] homing physical guarantees remain.

## XHand

- [ ] qpos/current/tactile remain same-sample.
- [ ] tactile validity remains.
- [ ] adopted != reached.
- [ ] streaming waits on adopted.
- [ ] hand home waits on reached when endpoint completion is required.

## Teleop

- [ ] B/H/S/D/Q/ESC remain coherent.
- [ ] **C pause/resume remains.**
- [ ] C cannot leak stale pre-pause intent.
- [ ] C still requires fresh re-anchor.
- [ ] partial adoption around C is represented, not hidden.
- [ ] recording-enabled B never silently becomes unrecorded.

## Policy

- [ ] inference remains synchronous.
- [ ] stale inference result after run change is discarded.
- [ ] publication cadence is explicit and no-catch-up.
- [ ] action chunk suffix is cleared at recoverable replanning boundaries.
- [ ] model-specific code remains confined to the deployment boundary.
- [ ] `PolicySpec` remains the model/runtime contract.

## Data

- [ ] raw schema explicitly bumped from v30.
- [ ] v30 is not silently reinterpreted.
- [ ] normal action row means all present actuators adopted the command.
- [ ] partial/unknown adoption has explicit diagnostics.
- [ ] partial/unknown episode is non-exportable by default.
- [ ] raw intent remains distinct from executable target.
- [ ] actual source timestamps remain.
- [ ] observation anchor remains.
- [ ] tactile validity remains.
- [ ] replay/export no longer depends on queue truth.

## Reproducibility

- [ ] run_policy accepts explicit config.
- [ ] full resolved runtime + policy/session facts are saved before hardware startup.
- [ ] episode technical validity is separate from task success.
- [ ] no duplicate authoritative policy control rate remains.

## Cleanup

- [ ] old command stream deleted only after all callers migrated.
- [ ] no old/new command compatibility shim remains.
- [ ] dead imports/config/comments removed.
- [ ] README stable workflow terms are current.
- [ ] no committed test framework added.
- [ ] no hardware-affecting validation was run.

---

# 26. Final Codex handoff

Report:

1. concise final architecture;
2. major files changed/deleted;
3. exact old command mechanisms removed;
4. raw schema version and semantic changes;
5. how partial adoption is handled;
6. exact teleop C behavior after refactor;
7. policy cadence semantics after refactor;
8. offline checks run with pass/fail/skip results;
9. explicit statement that no hardware validation was performed;
10. real-robot validation steps still required;
11. measured simplification when easy to obtain, such as deleted file count / line reduction / removed state fields;
12. any complexity intentionally retained because it protects physical safety, scientific timing/data correctness, or a current algorithm.

Do not claim success merely because code compiles. Confirm the protocol invariants and smoke scenarios.

---

# 27. Complexity budget

Before adding or preserving a mechanism, ask:

1. Without it, can the robot become less safe?
2. Without it, can experiment data become scientifically wrong?
3. Without it, will current experiment iteration materially suffer?
4. Has the current hardware/workflow actually created this problem?

If all are no, remove it or do not add it.

Justified complexity:

### Physical

- SDK process ownership;
- hard limits;
- XHand slew;
- safe home;
- workspace/collision checks actually used by experiments;
- robot errors;
- estop;
- safe shutdown;
- stale command fence.

### Scientific

- actual timestamps;
- coordinate frames;
- causal sensor selection;
- same-sample hand modalities;
- tactile validity;
- human intent vs executable target;
- per-actuator adoption timing;
- partial-adoption truth.

### Current algorithm

- observation history;
- action chunk;
- EE IK;
- recoverable replanning;
- point cloud when a deployed policy requires it;
- model inference steps.

Presumptively removable:

- lossless streaming command FIFO;
- stale backlog;
- FIFO FULL retries;
- sequence watermarks/base generations;
- per-command backlog expiry;
- generic evidence-service orchestration;
- async/RTC scaffolding;
- generic rollout strategies;
- hypothetical backend/plugin abstractions.

The final code should feel like a precise real-robot research endpoint, not a platform.