# CODEX TASK — Finish the research-simplicity refactor

Revision 2 · reviewed 2026-09-21 · repository `haoyangzhanglab/dexmani_real`

Read applicable `AGENTS.md` / `AGENTS.override.md`, this entire file, then README and relevant source. Platform/user and scoped instructions still apply. Read truncated files through EOF; do not assume Codex automatically loads this task.

This revision supersedes the previous task/long chat prompt. The launch prompt references this file, not a second specification. Keep this user-requested task; never weaken its criteria to declare success.

## 0. Mission, scope and deliverable

Personal PhD research: xArm7 + XHand + RealSense + VR/HTS data collection and `dexmani_policy` rollout/evaluation. Preserve teleop, home, calibration, raw replay, conversion and useful visualization, not a general robotics/serving framework.

```
physical safety > experiment correctness > iteration speed > readability > extensibility
preferred change: delete > inline > merge > rewrite > introduce abstraction
```

Execute three outcomes: consolidate branches; simplify motion/recording/dataset ownership; clean comments/config/tests/docs. Implementation, not another proposal.

Target: one motion-authority owner, one episode-file owner, one raw-to-policy path. Delete rather than add managers/protocols/backends/schema families. Internal APIs may break without shims; preserve research data and public `dexmani_policy` semantics.

### Authority and exclusions

- Execution of this task authorizes relevant source/docs/tests edits and task-branch commits. Subject to credentials, user approvals and repository rules, it also covers integrating existing PR #20 and closing/deleting verified superseded PRs/branches listed in Phase 0. Authentication is not permission to bypass approvals.
- Do not merge the NEW motion/recording/dataset implementation PR automatically. Deliver it for review, with live-hardware checks outstanding where applicable. Software integration and experiment-machine deployment are separate actions.
- Do not connect/discover devices, send motion, run live home/replay/teleop/rollout, alter calibration, or access laboratory devices as a test. Inspect tests, imports, constructors and fixtures before executing Python where side effects are possible. A `--help` or `--dry-run` name alone proves nothing.
- Do not modify recorded episodes, existing processed data, checkpoints, calibration files or `assets/**`. Do not change the sibling `dexmani_policy` repository without separate authorization; read its actual public interface as needed. Synthetic fixtures belong in temporary directories.
- Do not force-push, reset user work, run destructive `git clean`, delete arbitrary branches, change branch protections, auto-stash user changes, or upgrade the laboratory/global Python environment. Do not upload research data or credentials into commits/PRs.
- Keep separate xArm and XHand SDK processes, the two-consumer ordered broadcast, necessary final ACKs, and the four meaningful SafetyState values in this round. A logical Robot owner does not require one OS process.

## 1. Reference baseline and selective reading

These are historical anchors, not assumptions about current remote state:

| Item | Anchor |
|---|---|
| Existing integration | PR #20, `refactor/research-simplicity-20260921` |
| Pre-task source | `6298a94ebedb63052f4b6cd5eececa8f7cdbf069` |
| Original task commit | `d380b53ba62d25dac613cfd6907ce76cd2bb567f` |
| Historical main | `eaeb7a14e2be400488e62bc7897254270ce313d2` |
| Superseded candidates | PR #18 and PR #19 |
| Historical offline result | 176 tests + 78 subtests passed; not verification of any new HEAD |

Resolve actual refs/commit/tree IDs. GitHub default-branch search is not #20 content: fetch the right ref. If this file is absent, read it from verified #20 before switching; do not overwrite user work.

Read relevant implementations, not entire reference repositories:

| Reference | Fixed revision | Files to inspect |
|---|---|---|
| LeRobot | `d20a4538016d9fe0374a34011aebc1c8cddf3e0d` | `src/lerobot/robots/robot.py`, `scripts/lerobot_record.py`, `datasets/dataset_writer.py`, `rollout/inference/{sync,rtc}.py` (all under `src/lerobot/`) |
| ManiUniCon | `85c6f2e32ecf9f2bed62d202b058c39623444686` | `main.py`, `maniunicon/core/robot.py`, `maniunicon/utils/shared_memory/shared_storage.py`, `maniunicon/policies/torch_model.py`, `tools/process_demo_data.py` |

Borrow direct loops, resource ownership, local cancellation and meaningful conversion. Do not copy a generic controller hierarchy or treat queue draining/re-anchoring/min-length trimming as equivalent safety/alignment. Class names do not prove hardware behavior. Reference checkouts stay outside tracked code; unavailable references are reported, not invented.

## 2. Invariants that constrain every phase

### Motion and concurrency

- Mechanical limits, finite SDK inputs, real velocity/step bounds, error detection, emergency stop and safe disconnect remain. Preserve current experiment-dependent workspace/collision checks; do not claim home collision checking covers every policy trajectory.
- One authoritative motion-cancellation counter is sufficient. Copies of its value on commands/ACKs are not extra counters: retain them while needed to reject late ACKs/results. Camera frame identity is a different requirement and is not removed under the slogan “one generation”.
- Read/write related permission, generation and deadline/start-time fields atomically. Deleting `RunStateSnapshot` must not replace a coherent read with separately sampled shared values.
- Preserve an independent parent-side run-duration limit while `predict()` blocks. Retain one shared run start or absolute run deadline, paired with its generation, if that is what the parent needs. Do not make the blocked runner the only timeout owner.
- Parent-side S/Q/ESC and critical-device health monitoring remain responsive during inference, home and file finalization. Keep the effective callbacks/listener boundary; moving `operator.py` does not mean serializing all operations onto one blocking loop.
- Keep per-consumer FIFO order, FULL retries of the same numerical candidate, no catch-up burst, and XHand endpoint acceptance distinct from intermediate slew or CRC-unconfirmed sends. SDK acceptance is not physical arrival.

### Define the software guarantee honestly

The current `coupled_command_may_cross_sdk()` releases `motion_lock` BEFORE SDK IO. A last check is not atomic with a later vendor call. Identify this boundary explicitly during implementation and tests:

1. Under the short motion lock, admission/revocation decisions have an ordering point.
2. A revoked/expired generation cannot receive a NEW software admission after that point.
3. A call admitted earlier can be in flight or complete later. A late reply must not authorize another step, satisfy a newer wait, or overwrite a newer terminal cause.
4. The two SDKs are not a physical transaction. A fast consumer may already be ahead before the other reports expiry. Stop further admissions after revocation; do not promise rollback of earlier sends or add a per-command two-phase barrier.

Do not hold the global motion lock across potentially blocking SDK calls to manufacture a stronger guarantee at the cost of STOP responsiveness. Preserve immediate driver checks and existing stop mechanisms; document residual check-to-call scheduling/in-flight behavior. Stronger bounded physical stopping requires measured SDK/firmware behavior and authorized hardware testing, not wording in a test. Do not weaken existing safeguards.

### Experimental data and resources

Preserve units, frames, common history anchors, run-start history exclusion, RGB/depth/point-cloud pairing, real timestamps, tactile validity, intent vs committed target vs observed state, task/checkpoint/seed metadata, and incomplete-prefix handling. Recording-only camera failure and required-policy-camera failure have different consequences.

Do not unlink SHM with live users. Process termination is neither physical ESTOP nor durable recording. Raw stays untouched; non-uniform `policy_eval` remains rejected for fixed-dt training/replay.

## 3. Execution method and stop conditions

Start with read-only inspection:

```bash
git status --short
git remote -v
git branch --all --verbose --no-abbrev
git worktree list
git log --graph --decorate --oneline --all -n 80
```

Inspect applicable instructions, test collection/import side effects, `pyproject.toml`, existing test commands and Python dependencies. Fetch required refs when allowed. Use a clean worktree for unrelated dirty state. Missing refs/credentials block affected Git operations, not permission to guess.

Execute Phase 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 with one coherent, testable commit per logical change. Update callers, tests, imports, comments and README in the SAME change; Phase 6 is a final sweep, not permission for intermediate broken imports. No concurrent writers to shared motion/IPC/schema files. Read-only review may be parallel; integrate and test serially.

Before deletion trace definition -> producer -> transform -> consumer -> side effect. Note the real capability/race outside tracked code. Prefer direct functions/variables; retained mechanisms need concrete consumers.

Use one isolated offline environment for baseline/final comparison. Historical tests used Python 3.11, NumPy 1.26.4, Pinocchio 2.7.0 and NLopt 2.7.1; resolve the complete dependencies from current project/test requirements, not that incomplete list. Do not install unconstrained newest numerical packages and call the results comparable.

If interrupted, leave a coherent commit plus an external checkpoint: HEAD, phase, results, next action, blockers. Resume from inspected Git/code, not replayed mutations. Baselines/metrics/comparison fixtures stay untracked.

Make ordinary code choices autonomously. A failed gate stops the affected deletion/merge, not independent safe work. Preserve the known-good path and report the blocker. Mocking an unavailable dependency does not verify integration.

## Phase 0 — Consolidate existing work without losing changes

Inspect current #18/#19/#20 metadata, files, review/check status and exact base/head refs. Already merged/closed/deleted items are handled idempotently; do not reopen or recreate them automatically. Inspect repository rules and required checks; unknown mergeability is not a green light.

Compare ancestry AND two-tip content AND behavior. GitHub three-dot compare describes changes from the merge base; it does not by itself prove the newer branch subsumes the older one. Use explicit tip diffs/patch review. Classify unique work as PORT BEHAVIOR / ALREADY EQUIVALENT / OBSOLETE INTERNAL DETAIL / UNRELATED-PRESERVE.

Check the #19 no-hardware xArm7 identity test (7 accepted, 6/8 rejected), exact artifact loading, and close-on-spec-mismatch behavior against current `deployment/runner.py`. Port useful coverage without restoring obsolete import paths or enforcing obsolete wrapper details.

Run a pre-change offline baseline, then tests on the integrated #20 candidate. Record the exact tested HEAD. If baseline checks cannot run, report why; do not merge on the strength of historical counts.

With authorization and permissions:

1. Refresh main and #20. Include intended ported fixes on #20; do not merge #18, #19 and #20 sequentially.
2. Pass required checks/review. Mark the Draft ready only after the gate. Prefer a normal merge commit if repository policy permits; never bypass protections/admin review. Match the tested head SHA at merge time, and revalidate if head/base changed.
3. Fetch the merge result and verify the intended tree/behavior on main. Do not require main to equal an old tree if legitimate base changes exist.
4. Close superseded #18/#19 with the replacement reference only after independent changes are accounted for.
5. Delete only verified superseded branch refs at the inspected tips. Recheck for newly pushed commits/other PRs/worktrees. Use exact names, never wildcard cleanup. The docs branch below is a candidate to inspect, not presumed disposable.

Candidates (not unconditional deletion commands):

```
refactor/research-simplicity-20260920
refactor/research-simplicity-20260921
refactor/research-simplicity-final-20260920
research-simplify/remove-legacy-policy-trace
work/research-simplicity-resume-20260921
work/research-simplicity-validated-20260921
docs/claude-runtime-refactor-v2
```

For ported/squashed non-ancestor branches, preserve the tip in a durable PR/tag or external Git bundle before deleting its sole reference. No force-push to fix ancestry; no deletion of checked-out/unexplained unique work.

Continue on `refactor/research-simplicity-final` from verified main, or resume existing work. Without remote permission, branch from the locally verified #20 candidate and label it stacked/unmerged; cleanup stays pending. Never restart from old main or claim a local merge is remote integration.

**Gate:** useful older behavior accounted for; exact integration candidate tested; actual Git actions recorded. Remote-unavailable is a BLOCKED Git gate, not completed cleanup.

## Phase 1 — Fix the comparison baseline and map ownership

Keep the integrated baseline reachable in a detached comparison worktree or recorded retained commit. Pin the environment/config and create synthetic raw fixtures BEFORE removing the old dataset path. Read both sides of the real `dexmani_policy` load/spec/array boundary.

Starting points after #20 (resolve renamed paths from source):

| Work | Definitions AND callers to read |
|---|---|
| Motion | `robot/commands.py`, `{arm,hand}_worker.py`, drivers, `{arm,hand}_homing.py`, `runtime/safety.py`, `ipc/{channels,command_stream,schema}.py` |
| Control | `deployment/{runner,session,operator,observation}.py`, `teleop/{loop,session,keyboard_session}.py`, `teleop/control_loop/grid.py`, replay and calibration motion callers |
| Recording | `recording/{client,io_worker,recorder,frame}.py`, storage/video writers, `runtime/{supervisor,processes}.py`, every recording caller |
| Dataset | `dataset/{processing,processed,export,contracts}.py`, conversion CLIs, raw/processed viewers, replay loaders, annotation and export tests |

Record disposable metrics using one script: tracked Python physical LOC (including blank/comments), core/test LOC and file counts, AST classes/dataclasses/enums, configuration declarations, validate/check-named functions. Scope core/examples/tests separately; exclude vendored/assets/reference/generated files and this task. Record exact config-field/name-matching definitions. Also count actual IPC primitive families, runtime states and trace the four critical paths; do not invent universal call-depth numbers.

**Gate:** correct integrated comparison ref/environment, baseline results, source/consumer map and raw fixtures are available. No new permanent audit harness.

## Phase 2 — Localize deployment/runtime ownership

MERGE the one-consumer `deployment/operator.py` orchestration into session or existing home helpers, preserving independent immediate S/Q/ESC callbacks even while home blocks. Preserve fresh post-home BEGIN confirmation and current startup/model-warmup/hardware order. Do not casually combine CUDA, SDK IO and keyboard monitoring.

DELETE redundant trial containers/mirrors (`RunEpoch`, `RunStateSnapshot`, excess `run_ended_*` or result wrappers) AFTER replacing their real behavior. Trial counts/results belong to runner; session cleanup reporting to session; motion permission to robot; file outcomes to recorder. Keep coherent snapshots and the parent-side run-timeout input described in Section 2. Retain a minimal generation-scoped first-terminal fact if another process needs it; do not reconstruct an earlier cause from later global state or add a new event-log protocol.

MERGE actual motion-authority helpers into the robot boundary without recreating `control/` under a new name. A small concrete safety module is acceptable; blindly pasting everything into a giant `commands.py` is not a win. Keep camera frame identity and necessary ACK generation copies. Remove static/shared trial fields with no remaining consumer, not all shared state.

SIMPLIFY supervisor/process utilities to startup readiness, required-device health, optional recording failure, bounded stop/join and final resource release. Required camera/pointcloud/tactile observation loss must still stop policy use; recording-only service loss must not abruptly stop manual teleop. Preserve ownership of every started process handle even after failed startup. Do not replace health checks by process liveness alone for devices that can hang while alive.

**Gate tests:** blocked inference + parent timeout; late timeout/result from old run cannot cancel a new run; stop/quit/estop during home; same-batch STOP beats BEGIN; required-vs-recording-only camera loss; partial startup failure; atomic state/generation reads; no SHM release with an unreaped user.

Commit suggestion: `refactor: localize rollout and motion ownership`.

## Phase 3 — Bound delayed commands with one immutable deadline

### Semantics before code

Use one new absolute monotonic deadline on each endpoint candidate/transport record, e.g. `expires_monotonic_ns`. It is not an ID, lease object or new epoch. Use the same host clock domain and unit through all processes; wall time is only for human metadata. Do not replace scientific timestamps with this field or bump raw-data schema solely for IPC changes.

For each producer identify the time the endpoint becomes eligible under its EXISTING scheduler. Compute its deadline once from that intended execution time and one clearly-owned finite lag budget, e.g. `max_dispatch_delay_s`. Use integer monotonic nanoseconds in transport and define `now >= deadline` as expired.

- Do not start every future chunk endpoint's deadline at inference/chunk creation; preserve policy chunk pacing and the no-catch-up rule.
- Do not reset an old candidate's origin when FULL clears, it is copied, revalidated or reinserted. FULL retries retain the same target, generation and deadline; do not rerun IK/projection to disguise stale intent.
- Dispatch freshness is not observation freshness. Keep current observation/source-age and run-budget checks; a recent enqueue cannot make a stale prediction valid.
- Audit ALL endpoints: policy, VR, keyboard, replay, hand home and calibration. `arm_home_q` is a separate planned-motion path: bound a queued HOME request's admission with its existing timing budget or an explicit start deadline, then retain the plan's own execution/abort timeout. Do not apply one short servo dispatch deadline to the duration of an already-started multi-second home trajectory.
- A retained XHand endpoint keeps its deadline through intermediate slew and CRC-unconfirmed retries; every attempted SDK setpoint admission checks it. The budget must accommodate intended slew behavior. Never acknowledge an intermediate point as the endpoint.

### No invented live default

First determine whether an existing documented budget has the same semantics and can be reused. Otherwise add ONE explicit finite positive lag setting at startup. If a justified live value is unavailable, leave it unresolved (`None` or an equivalent configuration requirement) and fail closed BEFORE hardware startup/arming on motion-producing entry points. Do not guess a “conservative safe” value, silently disable expiry, or use infinity/zero to bypass it. Offline tests use explicit synthetic budgets; data-only/offline commands do not require a live budget. Report the live configuration requirement as an intentional commissioning change. An explicit number is not proof it is physically safe.

### Admission and cancellation

Publication rejects early under the short motion lock; worker admission immediately before IO owns the final permission/generation/deadline decision. Reuse a small check: different time windows are not repeated array validation.

On expiry, recheck the candidate generation under the same cancellation lock before revoking it. Old expiry cannot revoke a newer run; repeated consumers cannot repeatedly increment the current generation or overwrite a higher-priority terminal cause. Revoke/stop the current operation; do not skip its head and continue, auto-home, auto-rearm, or relabel software expiry as a physical ESTOP/hardware fault.

Section 2 defines admission ordering, not simultaneous SDK acceptance. Late replies retain their original generation; keep consumer ACK semantics without a lockstep barrier. Pre-admitted in-flight actions do not justify NEW admissions after STOP.

**Gate tests:** before/at/after deadline; unchanged FULL target/deadline; delayed hand slew/CRC retries; each worker admission; one consumer ahead then other expires; expiry concurrent with STOP/ESTOP; stale expiry after restart; late ACK cannot satisfy new wait; healthy ordering/pacing; HOME queue wait vs execution timeout; missing/nonfinite live budget rejected before hardware constructors. Use fake clocks and Events/barriers, not lucky sleeps or actual SDKs.

Commit suggestion: `safety: bound delayed command admission`.

## Phase 4 — Make the recorder process the sole file-finalization owner

Target: control tick -> owned `EpisodeFrame` -> sample ring -> recorder process -> HDF5/video -> one result. Keep a thin client, control/result channels and necessary continuous video workers; no storage abstraction.

Remove `_PendingFinalization`, the episode-finalizer thread and its nested result queue/polling. The recorder process is already asynchronous to control and may finalize its episode synchronously. Do NOT delete this thread without changing the outer supervision in the SAME commit.

### Boundary and outcome rules

- `add_frame` and finish-request submission must not wait for file/video close. Ring FULL must end/report recording rather than block manual control indefinitely. A save/discard decision is fixed for that finish operation.
- Freeze `through_sequence` at the final published sample, stop further publication for that episode, and drain through that cutoff before saving. A control Queue and sample ring have no implicit total order. Copy/own arrays before releasing a ring slot. The cutoff is not increased during retry/shutdown.
- Exactly one process owns file save/discard/incomplete publication. One client consumes its results. Keep the existing episode identity or strict single-in-flight discipline to match late replies; a timed-out transport cannot start another episode and consume an old result as new.
- Successful close publishes once; explicit discard really discards. A safely closed nonempty interrupted prefix remains `incomplete`, not “complete” or automatically trainable. If writers cannot be confirmed stopped/closed, retain staging for inspection and report failure. No exactly-once/durability claim across arbitrary process death.
- Handle producer disappearance and recorder-internal auto-finalization/error, not just an ordinary client STOP. They must enter the same finish supervision before blocking. Preserve recording failure in the final session result without inventing a robot hardware fault.

### Mandatory outer supervision update

Use the existing main/session monitor, not a new service. It must know when draining/finalizing starts and its ONE fixed outer deadline before the recorder blocks (including automatic error/capacity paths). Reuse existing status/control fields where sufficient; one minimal finishing/deadline signal is acceptable if required. This resource signal is not another lifecycle framework.

During that bounded finish, the recorder's tick heartbeat may stop: monitor process exit plus finish deadline rather than treating absent tick heartbeat as immediate failure. Never suspend arm/hand/required-sensor monitoring. The result channel still has one consumer; the supervisor must not race the client for the completion message. Polling, repeated STOP, unrelated heartbeats and shutdown must not renew the finish deadline.

If the recorder is killed or dies, disable its transport for the session; do not reuse potentially damaged queues/locks or auto-restart it. Do not take over its HDF5 writer from the parent. Termination/join can fail; do not unlink SHM or claim completion while a process or child writer is unreaped. Check ownership of video subprocesses as well as Python threads.

Shutdown order: latch/revoke motion promptly -> stop episode sample production at a known cutoff -> retain recorder/result-owner and necessary source buffers long enough to drain/finish under the SAME deadline -> stop/join remaining users -> unlink SHM last. Do not shut the result owner down before its result can be collected. Do not make emergency stopping wait for recording.

**Gate tests:** exact last frame/cutoff, copied arrays, save/discard/incomplete, disk/video errors, blocked finalization with bounded outer timeout, recorder-internal auto-finish, producer death, late result, no next START while pending, repeated finish without renewal, retained critical-device monitoring, process/child cleanup and no false complete publication. Include a small real no-hardware subprocess regression for the process-boundary change; no new test framework.

Commit suggestion: `recording: centralize episode finalization in recorder process`.

## Phase 5 — Convert raw directly to policy Zarr, with bounded memory

Target:

```
raw episode (source of truth) -> current numerical transforms -> policy Zarr
python examples/export_policy_zarr.py episodes/<task> --dry-run
python examples/export_policy_zarr.py episodes/<task>
```

KEEP FK/IK-related transforms, fingertip/contact/tactile semantics, RGB-D/point-cloud math, task/annotation decisions, actual time alignment, gap handling and exact policy-consumed keys/dtypes/layout. MERGE orchestration in `dataset/processing.py` and `dataset/export.py`; DELETE mandatory processed-HDF5 persistence/schema, not the mathematics. Stream one episode or bounded frame chunk at a time; do not replace intermediate files with the entire dataset resident in RAM or GPU.

### Establish the comparison oracle first

Run old raw -> processed -> Zarr in the retained BASE worktree/process, and new raw -> Zarr in the new worktree/process. Ensure the imports resolve to the intended tree; do not run “old” code against the new editable install. Use the same raw fixtures, config, calibration and dependencies. Fix sampling randomness or compare using a justified deterministic method; do not widen tolerances until mismatches disappear.

Compare every actual policy-consumed field, episode boundaries/order, shapes/dtypes, values, frame conventions, validity, task metadata, and timing/selection decisions. Also compare rejection behavior: explicit include/exclude annotations, task conflicts, missing/invalid modalities, gaps, empty inputs and non-uniform policy_eval. Do not require fabricated fields that the real policy reader does not consume. Exercise its public reader when available; if unavailable, state that external integration remains unverified rather than inventing a substitute.

Use synthetic raw fixtures from current test infrastructure. Real research samples, when available and permitted, are read-only and must not be uploaded. Keep a compact semantic regression without retaining the old implementation as a permanent oracle/compatibility branch. If numerical equivalence is blocked, keep the coherent old path on the working branch and mark this phase BLOCKED; do not delete first and lose the comparison basis.

### One conversion, not two hidden passes

Normal export validates raw input at its owner, transforms each payload once, writes staged Zarr, then checks the persisted output in bounded chunks before atomic publication. Dry-run shares the same admission/transformation/finite-value checks but writes no output/staging/report files and cannot claim to verify a persisted store it did not write. Do not call a full dry-run transformation internally and then repeat it for real export. A human explicitly running dry-run followed by export is different.

Retain no-overwrite and resolved-path/symlink protection for raw, rollouts, existing processed data and existing stores; protect existing scientific data even after removing its loader. Do not place output inside inputs. Preserve atomic publication and cleanup of task-created staging; cleanup must never delete user sources.

After the comparison gate passes, remove `dataset/processed.py` and processed-only validators/discovery; remove `examples/process_episodes.py`, `visualize_episode_processed.py`, processed-only replay flag/loader and their dead tests/config/imports/docs in the SAME change. Keep raw replay and useful raw visualization. Retain actually used annotation/export options in the remaining CLI; intentionally removed options get a documentation update, not a compatibility parser.

Never delete existing `episodes_processed/` data. Keep raw v29-to-v30 conversion; processed-only history uses pinned historical code or a concrete one-off converter. No default cache/second source of truth. Do not change `dexmani_policy` or its public format to evade comparison failures.

**Gate:** differential values AND admission behavior pass; bounded-memory/no-write dry-run/output protection tests pass; policy-reader test passes or is honestly recorded as an external integration blocker; all deleted-format consumers are removed together. Non-uniform policy_eval remains refused for fixed-dt training/replay.

Commit suggestion: `dataset: export raw episodes directly to policy zarr`.

## Phase 6 — Final cleanup and usable documentation

Remove dead imports/wrappers/names/CLI/config/tests by consumer tracing. Keep numerical and safety behavior, not exception wording/audit spans/deleted schemas. Never remove meaningful failing regressions to obtain green checks.

Keep one setting owner (CLI override > experiment YAML > Python default where currently used). Keep researcher-controlled IP/serial/frequency/resolution/path/checkpoint/task/gains and real thresholds. Immutable hardware identity is not an experiment setting; configured soft limits/calibration may still be experiment-specific. No new configuration framework or broad dependency upgrade.

README describes the ACTUAL final installation, current workflows/keys, output paths, remaining CLI, raw -> Zarr command, physical safety boundary, live lag-budget requirement, policy timing and incomplete data. Describe software admission vs physical stop honestly. Remove refactor history and old module inventories from everyday guidance. Keep units, coordinate conventions, copy-before-ACK and non-obvious race comments near their implementation; remove obvious code narration and obsolete guarantees.

Read existing docs before deleting old plans: retain still-current experiment/calibration/safety instructions. Prefer README sections; only extract `docs/hardware.md` or `docs/data.md` if necessary. Keep `CLAUDE.md` referential and `AGENTS.md` concise. Do not edit agent instructions to bypass a constraint. Keep this root task as explicitly requested; do not add another permanent audit/metrics/report framework.

Check stale references in live code/docs while allowing historical names in THIS task, negative tests and intentional historical conversion notes. A blanket zero-occurrence grep would wrongly flag them. Verify help/examples against actual parsers only after import safety review; never run motion examples as smoke tests.

**Gate:** callers, docs and configuration describe one implemented path; no dead compatibility shims; the researcher can find and use each retained workflow.

## Phase 7 — Verification, reviewable delivery and honest status

Run targeted tests after each logical change. At final handoff run the full audited OFFLINE suite in the same environment as the baseline, plus:

```bash
python -m compileall -q dexmani_real examples tests
python -m pytest -q
git diff --check
git diff --cached --check
git diff --check "$BASE" HEAD
```

`BASE` must be the recorded integrated comparison commit, not an unset variable. Working-tree diff alone checks nothing after all changes are committed. Run project-configured static checks and relevant existing undefined-name/dead-import checks; do not introduce a lint platform. Capture real exit codes (`pipefail` if piping through tee). Collection failures, timeouts, skips and unavailable dependencies must be reported separately from test passes. Never use `-k`, skip markers, stubs or reduced collection to present targeted success as a full-suite pass.

Review the final action/observation/recording/data paths against Section 2; inspect both sides of every changed boundary. For concurrency tests force the interleaving with controlled clocks/events and bounded joins; do not rely on sleeps or assume ordinary dict mocks test cross-process behavior.

Recompute Phase 1 metrics identically, including IPC/state/config/path changes. Separate runtime, tests/comments and moves; count task/docs separately. Deletion is a result, not a quota.

Commit/push only task work when permitted. Open/update ONE implementation PR with tested head/base, scope and remaining gates; do not create a new PR on every resumption or silently merge this new implementation. If remote writes are blocked, supply exact local commits/patch and pending actions. Do not leave uncommitted half-migrated APIs; a blocked phase should retain a coherent safe path and be labelled incomplete.

### Completion gates (separate outcomes)

| Gate | PASS means | BLOCKED / NOT RUN means |
|---|---|---|
| GIT | #20 integration and verified superseded cleanup actually completed | permissions/refs/review prevent an operation; list it, do not mark cleanup complete |
| SOFTWARE | intended simplifications implemented; required offline checks and numerical comparisons pass on exact final code | meaningful test/comparison/implementation missing; report partial delivery, not “all complete” |
| EXTERNAL INTEGRATION | real public policy reader/artifact boundary exercised with available dependencies/fixtures | not tested or dependency/checkpoint unavailable; do not claim it passed |
| HARDWARE | separately authorized, actually performed device validation | pending is expected here and is NOT supplied by offline tests |

Historical counts or reporting failures do not pass gates. Blocked Git/hardware steps do not stop independent safe work, but do stop the affected merge/deletion.

Final report must contain actual DELETE/MERGE/SIMPLIFY/KEEP results, remaining load-bearing mechanisms, the four final paths, before/after metrics with method, exact test commands/results, actual Git actions and final SHA/PR, deliberate deviations and all gate statuses. Ordinary documentation edits need not rerun hardware tests, but do not recycle old results as new ones.

Provide an UNEXECUTED commissioning checklist: explicit finite dispatch budget based on measured producer/SDK/slew delay; low-speed startup/stop; blocked inference with parent timeout; S/Q/ESC during home; backlog expiry and two-consumer partial progress; required camera/tactile loss; recording finalize timeout; real checkpoint/reader/data comparison; safe disconnect and physical emergency-stop response. Do not initiate those experiments. Never describe process kill, software revocation or a passing unit test as physical safety certification.

Success: a researcher can trace VR -> robot, policy -> robot, sensors -> policy, control tick -> episode, and raw -> policy dataset without learning a runtime audit protocol.
