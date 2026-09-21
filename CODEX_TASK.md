# CODEX TASK — Finish `dexmani_real` Research-Simplicity Refactor

> This file is the execution task for Codex.  
> Repository-wide safety and engineering rules in `AGENTS.md` are authoritative and must be read first.  
> This task is specific to the remaining architecture cleanup. If this task conflicts with `AGENTS.md`, follow `AGENTS.md`.

## 0. Mission

Finish the current simplification work so that `dexmani_real` is a **personal PhD real-robot research repository**, not a generic robotics framework.

Primary workflows:

1. VR / keyboard teleoperation and real-robot data collection.
2. Learned-policy deployment / rollout / evaluation through `dexmani_policy`.
3. Calibration, home, replay, conversion, and visualization utilities.

Optimization order:

```text
physical safety
> experiment correctness
> research iteration speed
> code readability
> generic extensibility
> enterprise robustness
```

Refactoring order:

```text
delete
> inline
> merge
> rewrite
> introduce abstraction
```

Do not solve remaining complexity by creating another manager, protocol, service layer, schema family, lifecycle framework, or compatibility adapter.

---

## 1. Repository authority and known starting state

Read, in this order:

1. `AGENTS.md`
2. this `CODEX_TASK.md`
3. `README.md`
4. the actual source and resolved configuration

At the time this task was written:

- repository: `haoyangzhanglab/dexmani_real`
- integration PR: **#20**
- integration branch: `refactor/research-simplicity-20260921`
- #20 was Draft and mergeable
- #20 HEAD before adding this task file was `6298a94ebedb63052f4b6cd5eececa8f7cdbf069`
- base `main` was `eaeb7a14e2be400488e62bc7897254270ce313d2`
- older overlapping PRs #18 and #19 were still open
- offline verification for the pre-task-file source tree was:
  - 176 passed
  - 78 subtests passed
  - compileall passed
  - targeted Ruff undefined-name/dead-import checks passed
  - `git diff --check` passed

**Do not trust these SHAs blindly. Re-resolve the repository state before editing.**

Reference implementations to inspect when a design choice is unclear:

- LeRobot: `huggingface/lerobot@d20a4538016d9fe0374a34011aebc1c8cddf3e0d`
- ManiUniCon: `Universal-Control/ManiUniCon@85c6f2e32ecf9f2bed62d202b058c39623444686`

Use them for design principles, not for mechanical code copying.

Useful principles from them:

- LeRobot normal paths are direct: observe → infer/teleop → send → record.
- LeRobot async RTC uses a **local reset epoch** only where an in-flight inference race exists.
- LeRobot recording has one writer owner for save/discard/finalize.
- ManiUniCon keeps multiprocessing primitives simple: shared rings/queue, a few events/flags, one main orchestrator.
- ManiUniCon raw recording → one meaningful offline conversion is acceptable.
- Do **not** copy LeRobot's generic interactive rollout framework or ManiUniCon's weaker synchronization semantics when they reduce required safety/correctness.

---

## 2. Absolute constraints

### 2.1 Never run hardware without explicit user authorization

Do not execute:

- xArm SDK discovery/connect/motion
- XHand SDK discovery/connect/motion
- homing
- physical replay
- teleoperation
- policy rollout
- calibration writes
- examples that can connect hardware

Imports/constructors must be inspected before execution when side effects are uncertain.

### 2.2 Preserve real safety invariants

The final design must still enforce, with one clear owner for each invariant:

- mechanical arm/hand limits
- finite values before SDK calls
- real velocity / command-step limits
- emergency stop
- robot error detection
- safe stop/disconnect
- no old/revoked command reaching the SDK after stop/reset
- workspace/collision constraints where current experiments rely on them
- worker shutdown before shared-memory release
- correct distinction between SDK acceptance and physical convergence

Do not remove a check solely because its name sounds defensive. Trace the race/hazard it protects first.

### 2.3 Preserve experimental semantics

Do not silently change:

- coordinate frames
- units
- RGB/depth/point-cloud frame pairing
- tactile validity semantics
- observation history timing
- actual action timing
- committed target vs raw policy/teleop intent
- incomplete-recording preservation
- non-uniform `policy_eval` timing semantics

A policy rollout with non-uniform real timing must not become a fixed-dt demonstration by convenience.

### 2.4 No internal backward-compatibility architecture

Allowed:

- break/move internal APIs
- remove old config keys
- remove wrappers
- remove tests for deleted implementation detail
- remove obsolete CLI
- remove old processed schema support

Not allowed:

- `LegacyFoo`
- `DeprecatedFoo`
- `FooCompat`
- runtime old/new adapters solely to preserve historical internals

Git history is the compatibility layer for old internal code. Historical research data may use an explicit offline conversion when genuinely needed.

---

## 3. Explicit non-goals

Do **not**:

- turn xArm7 + XHand into one OS process merely to reduce process count
- replace the two-consumer ordered command broadcast with a normal work-sharing `Queue`
- remove `SafetyState` just to replace it with several booleans
- create a distributed-system lease protocol
- create a new recording storage interface
- create a new policy interface
- create a new generic dataset schema ecosystem
- create remote-serving, multi-robot, multi-user, multi-backend abstractions
- introduce a second configuration framework
- preserve an abstraction because a test currently imports it

The desired result is a direct research system, not a minimal line count at the expense of physical behavior.

---

## 4. Working protocol

Before modifying anything:

```bash
git status --short
git branch --all --verbose --no-abbrev
git log --graph --decorate --oneline --all -n 80
```

If `gh` is authenticated, also inspect PR #18, #19, and #20. If it is unavailable, use local refs/remotes and do not claim remote actions were completed.

Rules during implementation:

1. Preserve unrelated user changes.
2. Trace actual entry point → producer → transform → consumer → side effect before deleting a mechanism.
3. Read both sides of every changed process/IPC/file boundary.
4. Make one coherent architectural change at a time.
5. Run targeted offline tests after each change.
6. Never delete a test just to make a failure disappear.
7. Delete tests that only preserve removed bookkeeping/internal structure.
8. Keep one-off metrics/review scripts outside the committed source tree.
9. Do not perform broad keyword replacement in safety/comments/docs.
10. Prefer ordinary exceptions unless there is a real recovery decision.

Use small, meaningful commits. Suggested boundaries are listed below.

---

# PHASE 0 — Integrate the existing simplification work and clean branches

## Goal

Make #20 the single integrated baseline, recover any useful behavior that exists only in older overlapping branches, merge the baseline, and remove obsolete duplicate branches/PRs only after verification.

## 0.1 Audit #18/#19/#20 by behavior, not PR number

Do not assume #20 contains every useful change from #18/#19.

At minimum check the older branches for unique:

- hardware-boundary regression tests
- policy artifact/spec boundary tests
- data conversion behavior
- safety behavior
- documentation corrections

Known example that must be checked:

- #19 contained a no-hardware fixed xArm7-axis identity test that accepted 7 axes and rejected 6/8. If equivalent coverage is absent on #20, port the behavior to the **current** implementation path. Do not restore removed `control/publication` or deployment wrappers to make the old test run.
- #19 also contained tests for loading the exact requested deployment artifact and closing a loaded policy when its spec does not match the inspected spec. Preserve those boundary semantics if still relevant, adapted to current `deployment/runner.py` APIs.

For every older difference classify it as:

```text
PORT BEHAVIOR
ALREADY EQUIVALENT
DELETE AS HISTORICAL IMPLEMENTATION DETAIL
UNRELATED — PRESERVE SEPARATELY
```

## 0.2 Validate the integrated baseline

Required before merge:

```bash
python -m compileall -q dexmani_real examples
python -m pytest -q
git diff --check
```

Run existing lint/static checks if configured. Do not add a new lint framework.

## 0.3 Merge and clean safely

After the final baseline HEAD passes:

1. Merge #20 into `main` using the repository's normal merge policy.
2. Verify the resulting `main` tree contains the intended source.
3. Only then close #18/#19 as superseded.
4. Delete duplicate remote/local work branches only after confirming they contain no unrelated unique work.
5. Do not delete a branch merely because its name starts with `work/`.
6. If remote permissions are unavailable, prepare the repository locally and report the exact remote actions still required. Never claim they were done.

Known duplicate/review branches to inspect include:

- `refactor/research-simplicity-20260920`
- `refactor/research-simplicity-20260921`
- `refactor/research-simplicity-final-20260920`
- `research-simplify/remove-legacy-policy-trace`
- `work/research-simplicity-resume-20260921`
- `work/research-simplicity-validated-20260921`
- `docs/claude-runtime-refactor-v2`

After baseline integration, create a fresh branch from updated `main` for the remaining work, for example:

```text
refactor/research-simplicity-final
```

Do not mix branch-cleanup history with the subsequent architectural changes if avoidable.

### Phase 0 done when

- [ ] useful unique behavior from #18/#19 is accounted for
- [ ] baseline offline suite passes
- [ ] #20 is integrated into main, or exact unavailable remote step is documented
- [ ] older duplicate PRs/branches are cleaned only after verification
- [ ] remaining work starts from the integrated main baseline

---

# PHASE 1 — Record a disposable baseline and retrace the four critical paths

Before the next edits, record **uncommitted/disposable** baseline metrics:

- Python LOC
- `dexmani_real` LOC
- test LOC
- Python file count
- core module count
- class count
- dataclass count
- enum count
- config-field count
- validate/check-named function count

Keep the counting script/result in `/tmp` or another untracked location.

Retrace and write short working notes for:

```text
teleop / policy → robot
robot / sensors → policy
control loop → recorder → files
raw episode → policy dataset
```

Do not commit an architecture-audit document just to record these notes.

### Phase 1 done when

- [ ] current flow is understood from actual code, not old docs
- [ ] baseline metrics are captured outside the permanent source tree
- [ ] every planned deletion has a known current producer and consumer

---

# PHASE 2 — Simplify deployment/runtime ownership

## Goal

Keep the real process/resource boundaries, but remove trial/runtime bookkeeping that does not protect motion or experimental correctness.

Target mental model:

```text
Main/session
  ├─ operator input thread
  ├─ policy process
  ├─ robot workers
  ├─ required sensor workers
  └─ optional recorder/camera service

policy runner
  observe → predict → prepare command → publish

robot boundary
  permission/generation/expiry → ordered stream → worker SDK fence
```

## 2.1 Keep these semantics

Keep:

- `SafetyState` if its four states still have distinct behavior
- one motion-cancellation generation/epoch
- ordered arm+hand command sequence
- actuator-specific final acceptance semantics
- independent operator stop/estop responsiveness
- main-process ownership of process start/stop/join
- model/CUDA ownership inside policy process
- SDK ownership inside hardware workers

Do not merge xArm and XHand processes in this phase.

## 2.2 Remove duplicated trial identity/result state

Trace every consumer of:

- `RunEpoch`
- `RunStateSnapshot`
- `RunEndReason`
- shared ended-generation fields
- shared run-start/run-end timestamp mirrors
- session/trial failure mirrors
- any shared field used only to print/report a result already known locally

Desired end state:

- shared cross-process state contains only values genuinely required across processes
- runner owns trial-local counters/result/reason
- session owns session-local result/cleanup reporting
- motion subsystem owns motion permission/cancellation
- recording subsystem owns recording success/failure

Prefer local variables / simple return values over new dataclasses.

If an end timestamp is scientifically persisted in an episode, keep that data. Remove only duplicated runtime bookkeeping.

## 2.3 Merge deployment operator code only if ownership becomes clearer

If `deployment/operator.py` has only one real consumer, merge its orchestration into `deployment/session.py` or move true robot-home helpers to the existing robot/home module.

Important:

- **preserve the independent keyboard/listener thread**
- S/Q/ESC must remain responsive while policy inference, home, or recording finalization blocks elsewhere
- do not turn “merge module” into “run everything on one blocking thread”

Delete the old module after all imports/tests/docs are migrated. Do not leave an import compatibility shim.

## 2.4 Simplify supervisor/process utilities

Reduce supervisor/process code to concrete responsibilities:

- wait for required startup readiness
- monitor required process liveness / sticky hardware errors
- classify optional recording-service failure separately from motion failure
- request stop
- join/terminate if needed
- release shared memory last

Remove generic reports/specs/wrappers that have one caller and no independent behavior.

Do not weaken shutdown ordering.

### Phase 2 tests

Protect behavior, not implementation names:

- stop while policy inference is blocked
- stop then later inference return cannot publish old motion
- quit/estop stays responsive during home
- worker fault terminates motion path
- recording-service failure does not become a hardware fault
- shared memory is released only after workers stop

### Phase 2 done when

- [ ] one motion cancellation identity remains
- [ ] trial bookkeeping is local instead of protocol-wide
- [ ] operator stop responsiveness is unchanged
- [ ] no compatibility wrappers remain
- [ ] targeted offline tests pass

Suggested commit:

```text
refactor: localize rollout lifecycle state
```

---

# PHASE 3 — Bound same-generation command staleness without building a lease system

## Problem

A current-generation command can remain queued/pending for too long and still be considered valid. Cross-generation stale work is fenced; same-generation backlog still needs a simple execution-age bound.

## Design

Use **one deadline timestamp**, not a lease framework.

Preferred representation:

```text
expires_monotonic_ns
```

The deadline belongs to the command/candidate that may eventually reach an SDK.

Requirements:

1. Deadline is created once for the command intent.
2. FIFO FULL retry keeps the **same numeric target and same deadline**.
3. Publication checks expiry while holding the same motion-permission critical section.
4. Arm and hand workers re-check expiry immediately before the final SDK send/admission.
5. Do not create:
   - lease objects
   - renewals
   - another generation ID
   - another command ID
   - a timeout manager process
6. Do not skip an expired FIFO head and execute newer commands behind it.
7. Expiry is a safe cancellation/stop condition, not an estop and not automatically a hardware fault.
8. A new run requires an explicit new operator/workflow start.

The producer's intended execution time matters. Do not “freshen” an old action merely because it was inserted into the FIFO later.

For periodic policy/teleop commands, derive/resolve the allowed lag from the existing configured cadence plus one clearly-owned bound. Avoid exposing several duplicate timeout knobs.

**Do not invent an aggressive production value without evidence.** Inspect existing control periods, worker timing, acceptance timeout assumptions, and current tests. Implement the mechanism and a single clear configuration/constant owner. If a final live threshold cannot be justified offline, choose a conservative fail-safe default consistent with current timing assumptions and mark the exact value as requiring hardware timing validation in the handoff. Do not disable the protection silently.

Home/calibration/replay commands that already serialize publish→accept may use an appropriate explicit deadline; do not exempt them by accident.

## Required tests

- FIFO FULL retry does not extend deadline
- expired command is never sent to arm SDK boundary
- expired command is never sent to hand SDK boundary
- expiry at one consumer prevents later commands from overtaking the expired head
- STOP/generation revoke still wins over expiry classification
- new run does not resurrect expired/old work
- ordinary in-budget commands preserve current order
- XHand intermediate slew acceptance still cannot acknowledge the final target early

### Phase 3 done when

- [ ] same-generation backlog has a bounded execution age
- [ ] only one new temporal concept was added: the deadline
- [ ] no lease subsystem exists
- [ ] both worker SDK boundaries enforce it
- [ ] targeted offline tests pass

Suggested commit:

```text
safety: expire delayed robot commands at the SDK boundary
```

---

# PHASE 4 — Give recording one episode-finalization owner

## Goal

Keep asynchronous recording so the control loop never blocks on file/video IO, but eliminate the second layer of finalization machinery inside the recorder process.

Target path:

```text
control/policy loop
    ↓
RecorderClient.add_frame(EpisodeFrame)
    ↓
sample ring
    ↓
recorder process
    ↓
EpisodeRecorder.add_frame()
    ↓
finish episode synchronously inside recorder process
    ↓
one result back to client
```

## 4.1 Keep the real process boundary

Keep:

- recorder process
- sample shared-memory ring
- control queue for episode boundaries
- one result channel
- video/image worker threads only if they are necessary for continuous IO
- `through_sequence` or equivalent boundary required to ensure the last committed sample is drained before finalization

## 4.2 Delete duplicate episode lifecycle ownership

Inspect and remove, if still present:

- `_PendingFinalization`
- finalizer thread dedicated only to closing one episode
- nested finalizer result queue
- polling state whose only purpose is to supervise that internal thread
- duplicate save/discard ownership split across client, IO session, and serializer

The recorder process itself is already asynchronous relative to robot control. Let it block while closing **its own** episode.

The client should remain a thin cross-process handle, not a second recorder state machine.

Desired public operations can remain conceptually:

```text
start_episode(...)
add_frame(frame)
finish_episode(save: bool, reason: str)
poll/wait for one finish result
close()
```

Do not create a storage backend abstraction.

## 4.3 Preserve failure semantics

Must preserve:

- copy-before-ACK / slot-release safety
- last-frame drain before save
- explicit discard really discards
- unexpected recording failure preserves a non-empty safely-closed prefix as incomplete
- if safe close cannot be confirmed, keep staging rather than publishing a “complete” episode
- recording failure must not abruptly stop manual teleoperation
- rollout/session result must still report recording failure
- no next episode starts until previous finalization result is known
- shutdown stops motion first, then allows bounded recording finalization, then releases shared memory
- finalization timeout is handled by the outer process/session owner, not another finalizer thread

LeRobot's useful principle here is one writer owner with an idempotent finalization path. Copy the ownership principle, not the full dataset framework.

## Required tests

- exact final sample is included
- finish waits for `through_sequence`
- save success
- explicit discard
- camera/recording degradation preserves allowed incomplete prefix
- disk/video failure does not publish a false complete episode
- repeated close/finish is harmless or clearly rejected once
- no next start while prior finish is unresolved
- recorder process death is reported without reclassifying it as arm/hand hardware failure

### Phase 4 done when

- [ ] recorder process is sole file-lifecycle owner
- [ ] no nested episode-finalizer thread/state machine remains
- [ ] control loop remains non-blocking on file IO
- [ ] prefix/discard/failure semantics remain correct
- [ ] targeted recording tests pass

Suggested commit:

```text
recording: make recorder process own episode finalization
```

---

# PHASE 5 — Remove mandatory processed-HDF5 architecture; export raw directly to policy Zarr

## Goal

End with one canonical research source plus one explicit offline conversion:

```text
raw episode (source of truth)
    ↓
offline geometric / sensor transforms
    ↓
policy Zarr consumed by dexmani_policy
```

The current mathematical processing remains valuable. The **mandatory persisted processed HDF5 contract** does not need to remain.

## 5.1 Preserve real transforms

Keep the numerical behavior currently required for:

- temporal admission/alignment
- camera geometry
- RGB-D handling
- point cloud generation/sampling
- coordinate transforms
- arm FK / EE representation
- fingertip positions
- contact/tactile validity
- action representation
- task identity
- dtype/shape expected by `dexmani_policy`

Prefer pure functions that operate on raw episode data and feed the Zarr writer.

## 5.2 Collapse the two-step CLI

Target one user-facing offline command, preferably keeping the existing discoverable name:

```bash
python examples/export_policy_zarr.py episodes/<task> --dry-run
python examples/export_policy_zarr.py episodes/<task>
```

The real export and dry-run must share the same validation path.

Delete the mandatory sequence:

```text
process_episodes.py
→ processed HDF5
→ export_policy_zarr.py
```

Do not replace it with another intermediate schema.

## 5.3 Remove processed-only architecture

After the direct exporter is proven equivalent, delete/merge as appropriate:

- processed schema/version constants used only for the intermediate file
- processed-HDF5 validator ecosystem
- processed discovery/preflight that duplicates raw admission + final Zarr checks
- `examples/process_episodes.py`
- `examples/visualize_episode_processed.py`
- processed-only replay mode / `--processed`, if it exists only to consume this intermediate format
- config fields/directories that exist only for `episodes_processed`
- tests that only freeze the deleted intermediate representation

Do not keep a compatibility loader.

Historical processed-only data can be handled with Git history or a one-off explicit conversion if there is a concrete dataset that needs it. Do not build runtime compatibility preemptively.

## 5.4 Keep output safety

The direct exporter must still:

- never overwrite raw episodes
- refuse occupied output by default
- write atomically/staged where current code already guarantees this
- validate final Zarr arrays at the external dataset boundary
- keep dry-run read-only
- never write inside protected source roots
- preserve task identity
- provide actionable error messages without an error taxonomy framework

## 5.5 Prove equivalence before deleting the old path

Before removing the old processed writer, use existing fixtures/sample data to compare:

```text
raw → old processed → old Zarr
vs
raw → new direct Zarr
```

Compare every policy-consumed array:

- keys
- episode boundaries
- shapes
- dtypes
- timestamps / timing semantics
- numerical values with justified tolerance
- point clouds
- fingertip/contact/tactile validity
- actions
- task labels

Do not require byte-identical container metadata if it is irrelevant to policy consumption.

Important:

- do not use “truncate all streams to min length” as a replacement for actual time alignment
- do not silently admit non-uniform `policy_eval` as fixed-dt training data
- preserve current refusal until a deliberate time-aware training conversion exists

### Phase 5 done when

- [ ] one raw→policy conversion command remains
- [ ] mandatory processed HDF5 is gone
- [ ] policy Zarr semantics match the old valid path
- [ ] raw remains the source of truth
- [ ] no runtime compatibility layer remains
- [ ] dataset tests pass

Suggested commit:

```text
dataset: export raw episodes directly to policy zarr
```

---

# PHASE 6 — Remove dead config/tests/comments and update stable documentation

Do this only after the architecture above is working.

## 6.1 Configuration

Remove:

- fields used only by deleted processed format
- fields used only by deleted lifecycle/evidence bookkeeping
- duplicate derived values
- hypothetical backend options
- values that are actually immutable xArm7/XHand hardware constants

Keep configuration for things researchers actually change:

- IP / serial
- frequencies
- camera resolution
- paths
- checkpoint/artifact
- task
- control gains
- experiment parameters
- real safety thresholds

One source of truth per setting.

## 6.2 Tests

Keep tests for:

- FK/IK
- transforms
- retargeting
- geometry
- point clouds
- dataset conversion
- action clipping
- command expiry/revocation
- concrete past regressions
- recording boundary correctness

Delete/rewrite tests for:

- exact exception wording
- deleted internal state enums
- old evidence/provenance bookkeeping
- old processed schema identity
- mocks whose sole purpose is to preserve removed multiprocessing choreography
- imports of deleted wrappers

Do not reduce test coverage of a real safety invariant.

## 6.3 Comments/docstrings

For every changed area:

Delete comments that:

- narrate removed architecture
- refer to old module paths/classes
- restate obvious code
- describe historical migration stages
- claim guarantees no longer provided

Keep concise comments that explain:

- non-obvious race conditions
- units / coordinate frames
- copy-before-ACK
- why a worker performs the final SDK fence
- why XHand intermediate slew points do not acknowledge the final target
- why policy_eval timing cannot be treated as fixed-dt
- why a particular shutdown order is safety-critical

Do not perform global terminology replacement. For example, scientific provenance such as task/checkpoint/seed may still be useful even if internal “evidence protocol” terminology is removed.

## 6.4 README and stable docs

Update `README.md` to describe the **final current system**, not the refactor history.

README should contain:

- current installation/config precedence
- current research entry points
- teleop/data collection
- policy rollout
- raw→policy dataset conversion
- calibration/home/replay/visualization
- concise safety ownership
- offline verification commands
- explicit note that offline tests are not hardware validation

Update/remove every stale command and path, especially:

- processed-HDF5 workflow
- `episodes_processed`
- processed visualizer
- deleted module names
- obsolete evidence/lifecycle terminology

Keep `AGENTS.md` as the repository-wide authority. Keep `CLAUDE.md` small and referential.

Do not create a large docs hierarchy unless README genuinely becomes unreadable. If needed, at most extract stable `docs/hardware.md` and/or `docs/data.md`.

Keep `CODEX_TASK.md` during this execution. Do not delete it automatically unless the user explicitly asks.

Suggested commit:

```text
docs: align repository guidance with simplified architecture
```

---

# PHASE 7 — Final offline verification, metrics, and handoff

## 7.1 Static/offline gates

At minimum:

```bash
python -m compileall -q dexmani_real examples
python -m pytest -q
git diff --check
```

Run repository-configured Ruff/static checks if available.

Also use `git grep` / AST inspection to verify there are no stale references to deleted:

- modules
- config keys
- CLI flags
- schema names
- comments/docs terminology

Do not run hardware to make tests pass.

## 7.2 Recompute the same metrics as Phase 1

Report before/after using the exact same counting method:

- Python LOC
- `dexmani_real` LOC
- test LOC
- Python file count
- core modules
- classes
- dataclasses
- enums
- config fields
- validate/check-named functions

Metrics are evidence, not a quota. Do not delete useful safety code to hit a number.

## 7.3 Final diff review

Manually inspect:

- action path
- stop/revoke path
- both worker SDK boundaries
- recorder finish path
- raw→Zarr path
- README commands

Check for:

- dead imports
- dead config
- half-migrated terminology
- compatibility wrappers
- duplicate owners
- comments describing old behavior

## 7.4 Hardware validation remains separate

Do not claim hardware validation.

Provide a manual, operator-authorized checklist for later real-hardware testing, including at least:

- low-speed arm+hand connection/start/stop
- S/Q/ESC response while inference is slow
- command-expiry behavior under induced backlog
- xArm/XHand error handling
- home behavior
- long recording/finalization
- RealSense/tactile data validity
- real checkpoint rollout
- emergency stop
- safe disconnect

Do not execute these steps without explicit authorization.

---

# 8. Definition of Done

This task is complete only when all of the following are true:

### Git / branches

- [ ] #20 baseline is integrated or the exact unavailable remote operation is explicitly reported
- [ ] #18/#19 unique useful behavior is accounted for
- [ ] superseded branches/PRs are cleaned only after verification
- [ ] subsequent refactor work is isolated from obsolete branch history

### Runtime / motion

- [ ] policy/teleop → robot path has one clear command preparation/publication owner
- [ ] only one motion cancellation generation/epoch remains
- [ ] same-generation stale commands have one deadline mechanism
- [ ] no lease framework or extra command identity was introduced
- [ ] arm and hand SDK boundaries reject revoked/expired work
- [ ] operator stop/estop remains responsive during blocking work

### Recording

- [ ] recorder process is the sole episode file-lifecycle owner
- [ ] nested finalizer thread/state protocol is removed
- [ ] last-frame/prefix/discard/failure semantics remain correct
- [ ] recording failure is not mislabeled as robot hardware failure

### Dataset

- [ ] raw episode is the canonical source of truth
- [ ] one direct raw→policy Zarr conversion exists
- [ ] mandatory processed HDF5 architecture is removed
- [ ] old and new policy-consumed values were compared before removing the old path
- [ ] policy_eval timing semantics remain honest

### Cleanup/docs

- [ ] dead config/imports/tests/comments are removed
- [ ] README commands match the final implementation
- [ ] no obsolete compatibility adapters remain
- [ ] no new generic framework was added

### Verification

- [ ] full offline tests pass, or every failure is explicitly reported with no false success claim
- [ ] compileall passes
- [ ] diff check passes
- [ ] configured static checks pass
- [ ] before/after complexity metrics use one consistent method
- [ ] hardware validation is explicitly marked pending unless separately authorized and actually performed

---

# 9. Required final Codex report

At completion, report concisely:

1. **What was deleted / merged / simplified / kept**
2. **Why every remaining non-trivial mechanism is load-bearing**
3. **Final action path**
4. **Final observation path**
5. **Final recording path**
6. **Final raw→policy dataset path**
7. **Branch/PR actions actually completed**
8. **Before/after metrics**
9. **Exact tests/checks run and results**
10. **Hardware validation still pending**
11. **Any deliberate deviations from this task and the concrete reason**

Do not claim success for work that was not executed.

The architectural success criterion is:

> A researcher should be able to trace VR → robot, policy → robot, robot/sensors → policy, control loop → recorder, and raw episode → policy dataset without learning an internal distributed protocol.
