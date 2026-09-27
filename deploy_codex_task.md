# DexMani deployment simplification — Codex task

## 0. Mission

Refactor dexmani_policy and dexmani_real into a research-oriented real-robot evaluation stack with:

1. one saved experiment configuration,
2. one normal training checkpoint path,
3. one shared inference loader,
4. the existing Real physical-safety runtime.

This is a deliberate breaking change. Do not preserve compatibility with current simple.v3 checkpoints, deployment.v3 artifacts, deployment_latest.pt, or old deployment APIs.

Do not build migration code, compatibility adapters, deprecated aliases, version registries, ABI layers, hashes, or dual old/new execution paths.

These repositories are personal PhD research repositories, not production serving infrastructure. Optimize in this order:

1. physical safety,
2. experiment correctness,
3. research iteration speed,
4. direct readable code,
5. minimal long-lived abstractions.

Prefer delete, inline, or merge over adding another framework.

## 1. Reference-project principles

### Official Diffusion Policy

Use the important pattern, not its exact implementation:

    saved experiment config
    + normal workspace/training checkpoint
    -> reconstruct policy
    -> load model/EMA state
    -> run inference

There is no second mandatory deployment artifact grammar before real-robot evaluation.

### ManiUniCon

Use the important pattern:

    Hydra/config-driven model construction
    + thin observation wrapper
    + thin action wrapper
    + separate policy process / shared-memory robot runtime

Do not import ManiUniCon timestamp/stale-prefix action filtering into DexMani's current empty-queue synchronous chunk executor. That filtering assumes a different execution model.

### DexMani-specific addition

Keep DexMani Real's stronger physical architecture:

- policy/CUDA in the policy worker;
- SDK ownership in arm/hand/camera workers;
- SafetyState;
- run_id;
- final stale-command fence before SDK admission;
- sensor freshness;
- HOME;
- recorder sequence integrity;
- verified shutdown before SHM release.

The goal is simpler model/config plumbing without weakening physical safety.

## 2. Local repository and environment facts

Run from dexmani_real.

Sibling Policy repository:

    ../dexmani_policy

Managed Conda environments:

    dexmani_policy -> policy
    dexmani_real   -> real_robot

Prefer:

    conda run --no-capture-output -n policy <command>
    conda run --no-capture-output -n real_robot <command>

GPU and CUDA are correctly configured on the real development machine. If a sandbox reports CUDA unavailable, treat that as a sandbox limitation. Do not disable CUDA paths, alter GPU defaults, or make the implementation CPU-only to satisfy a sandbox probe.

Before editing each repository:

    git status --short

Preserve unrelated changes.

Read both repositories' AGENTS.md files before editing.

Do not create a tests directory.

Do not run hardware-affecting code during validation.

## 3. Final architecture

The final research artifact is the experiment directory, not a self-contained deployment checkpoint.

Required shape:

    experiments/<policy>/<task>/<run>/
        config.yaml
        checkpoints/
            <normal training checkpoints>
            latest.pt
            best_ckpt.json

The lifecycle must be:

    source Hydra config
        |
        v
    build actual dataset
        |
        +--> snapshot actual Real runtime input recipe into in-memory cfg
        |
        v
    save resolved experiment config.yaml
        |
        v
    normal training checkpoints
        |
        +--> exact training resume
        +--> simulation/offline inference
        +--> real-robot inference

Real evaluation:

    experiment/config.yaml
        -> derive small PolicyInfo
        -> Real compatibility preflight
        -> size RuntimeChannels
        -> spawn policy process
        -> policy process loads normal checkpoint
        -> warmup / policy_ready
        -> only then start hardware/sensor workers

This startup order is important: a bad checkpoint/model must fail before hardware workers connect.

There must no longer be:

    training checkpoint
      -> deployment export
      -> deployment artifact
      -> deployment schema
      -> PolicySpec / DeploymentSpec
      -> deployment restore validation
      -> Real schema re-validation

## 4. config.yaml is the single inference configuration source of truth

The saved resolved experiment config.yaml must own all model construction and inference recipe information.

Inference must read architecture and model-facing behavior from this saved config, not from current source YAML and not from a second checkpoint contract.

This includes existing resolved fields such as:

- agent;
- action_key;
- n_obs_steps;
- n_action_steps;
- horizon;
- dataset sensor_modalities;
- deterministic RGB validation preprocessing;
- normal inference defaults where already defined.

Do not mirror the same information into PolicySpec, DeploymentSpec, AgentContract, RgbPreprocessingSpec, or checkpoint inference metadata.

Changing the repository's current source config after training must not redefine an existing experiment.

## 5. Add only one small saved Real-runtime snapshot to config.yaml

Some Real-produced numerical facts are not reconstructable from the Agent config but do affect the trained input distribution.

After the actual training dataset has been instantiated, derive these facts from the actual canonical Real Policy Zarr and attach them to the in-memory resolved config before workspace.save_hydra_config(cfg).

Use one small top-level config section, for example:

    real_runtime:
      control_dt_s: ...
      pointcloud: {...}

For current scope, this section should contain only:

- control_dt_s;
- exact numerical PointCloudConfig used to produce the training point cloud, when point_cloud is consumed.

For sim-only or non-canonical-Real datasets:

    real_runtime: null

Real deployment must reject real_runtime: null rather than guess.

Do not put in this section:

- Git SHA;
- URDF hash;
- point-cloud algorithm ID;
- function/class names;
- camera serial;
- camera intrinsics/extrinsics;
- current table plane;
- current hand mount;
- ABI/version registry;
- provenance text.

Current physical calibration remains Real-owned at runtime.

Replace the current broad deployment_data_semantics snapshot with this minimal extraction.

## 6. Keep the training checkpoint focused

Do not redesign the training checkpoint format merely for deployment simplification.

The existing checkpoint container may remain close to its current structure if that is the smallest reliable change. Old checkpoint compatibility is explicitly out of scope, so a breaking cleanup is allowed, but a new format/version is not a goal by itself.

The normal training checkpoint should own training state and model states, for example:

    state:
      epoch
      global_step
      next_micro_step
      EMA updater state
      RNG / resume-only facts

    weights:
      model
      ema_model
      optimizer
      scheduler

Delete fields only when they are genuinely obsolete after the deployment artifact path is removed.

In particular:

- do not copy the saved experiment config into every checkpoint;
- do not copy Real runtime facts into every checkpoint;
- do not add checkpoint inference metadata;
- do not require parent-side weights_only/meta checkpoint inspection;
- do not rename/restructure checkpoint fields unless the resulting implementation is actually simpler.

The parent Real process should inspect saved config.yaml; only the policy child loads the checkpoint.

Because old checkpoints are explicitly unsupported, old-format rejection may stay simple. Do not implement migration.

## 7. Do not broaden this task into a training-resume rewrite

Exact training resume may remain strict and may keep a dedicated training-only resume contract/facts if that is the safest and smallest implementation.

The deployment refactor only requires that inference no longer depends on training-resume validation.

Required cleanup:

- remove deployment_data_semantics from the training resume contract once real_runtime in saved config replaces it;
- stop using resume-contract agent/data/normalization semantics to construct or validate inference;
- stop calling validate_resume_contract from offline or Real inference;
- remove build_agent_contract or other resume helpers only if they become genuinely unused or if a small local simplification is obvious.

Do not rewrite DDP resume, loader cursor restoration, optimizer/scheduler resume, or RNG restoration merely to eliminate training-only duplication.

Training resume strictness should remain training-owned. Inference should use saved config.yaml plus strict model state restoration.

Do not build a schema registry or migration graph.

## 8. One inference loader for offline and Real evaluation

Create one shared Policy-side inference path used by:

- offline/simulation checkpoint eval;
- best-checkpoint eval;
- demo/eval tools where applicable;
- dexmani_real.

Conceptually:

    load saved experiment config.yaml
    -> instantiate saved Agent
    -> set action_key / necessary runtime fields
    -> load selected raw or EMA state with strict=True
    -> move to device
    -> eval()
    -> return thin LoadedPolicy

Do not call validate_resume_contract from inference.

Do not rebuild and compare another AgentContract.

Keep load_state_dict(..., strict=True).

Do not use strict=False as a generic convenience.

## 9. Agent constructors must not require training-only external initialization assets at inference

The generic inference loader must not know concrete Policy types.

Fix the two current root causes directly.

### 9.1 Uni3D

A complete trained checkpoint must not require the original pretrained Uni3D file merely to instantiate for inference.

Move pretrained initialization out of the inference constructor path.

Preferred semantics:

    construct architecture
    -> training only: optionally initialize pretrained weights
    -> training
    -> checkpoint state owns final weights

### 9.2 DQ-RISE

A complete checkpoint must not require the original external codebook_path NPZ at inference.

Persistent codebook state belongs in state_dict.

Move external codebook import into the training initialization path owned by DQ-RISE/codebook code.

Keep this targeted. Do not redesign unrelated Agents.

After this change, delete algorithm-specific deployment/export config sanitization.

## 10. RGB preprocessing stays Policy-owned

Do not create RgbPreprocessingSpec.

Use the saved experiment config's dataset fields directly:

- rgb_preprocess_size;
- rgb_random_crop_size as deterministic center crop for validation/inference;
- rgb_keep_uint8 where relevant.

Real provides raw uint8 HWC RGB.

Policy performs the same deterministic validation preprocessing before the Agent.

The Agent's own ImageProcessor remains Agent-owned.

Do not persist warmup-only RGB metadata.

## 11. PointCloudConfig should remain numerical, not become an ABI

The saved experiment config freezes the numerical point-cloud recipe used for training.

Keep PointCloudConfig, but make from_dict additive-friendly:

- supplied known values override dataclass defaults;
- a newly added optional field may be absent and use an old-behavior default;
- unknown keys should fail clearly;
- new optional defaults must preserve previous behavior.

Do not require exact saved-key equality with every current dataclass field.

Do not add point-cloud ABI version, implementation ID compatibility gates, or Git hash gates.

Freeze numerical behavior, not implementation identity.

## 12. Runtime-only PolicyInfo

Replace persisted PolicySpec, ObservationFieldSpec, and DeploymentSpec with one small ordinary runtime object derived from saved config.yaml.

A suitable shape:

    PolicyInfo:
        experiment_dir
        checkpoint_path
        policy_name
        task_name
        observation_fields
        n_obs_steps
        n_action_steps
        action_mode
        control_dt_s
        pointcloud_config
        default inference_steps if useful

This object is not persisted.

Derive:

    observation_fields <- saved dataset.sensor_modalities
    action_mode        <- action_key
    horizons           <- saved Agent/config
    Real recipe        <- saved real_runtime

Do not duplicate tensor dtype/order schemas.

## 13. CLI and checkpoint selection

Keep the experiment directory as the unit of evaluation.

Recommended CLI:

    python examples/run_policy.py EXPERIMENT         --checkpoint best|latest|<filename>         [--weights ema|raw]         [--inference-steps N]

Remove --artifact.

Do not make standalone checkpoint paths a primary workflow unless they can unambiguously resolve their parent experiment config with no extra machinery.

best_ckpt.json may remain.

Make its reader additive-friendly:

- require only keys needed by selection;
- ignore unrelated extra bookkeeping keys;
- do not require exact top-level key equality.

## 14. Delete the deployment artifact subsystem

Once direct experiment-config + training-checkpoint inference works, remove obsolete code rather than leaving compatibility aliases.

Delete when unused:

- dexmani_policy/deployment/export.py;
- dexmani_policy/deployment/contract.py;
- dexmani_policy/deployment/restore.py;
- ExportReceipt;
- deployment artifact selector/publication logic;
- deployment_latest.pt assumptions;
- DEPLOYMENT_FORMAT;
- PolicySpec;
- ObservationFieldSpec;
- DeploymentSpec;
- RgbPreprocessingSpec;
- artifact producer/source-commit metadata;
- deployment export scripts and stale README commands.

dexmani_policy/deployment/runtime.py may remain only as a thin public bridge around the shared inference loader.

Do not recreate the same concepts under different names.

## 15. Real compatibility boundary must become small

Rewrite Real compatibility validation around actual capability.

Preflight should approximately check only:

1. runtime.policy.hand_enabled is true;
2. requested observation field names are supported and include joint_state;
3. policy control frequency does not exceed arm/hand worker service rate;
4. if point cloud is requested, saved real_runtime.pointcloud is present and parses as PointCloudConfig.

Remove cross-repository equality checks for:

- exact dtype declarations;
- exact shape declarations already canonically constructed by Real;
- joint-name arrays;
- RGB channel ordering;
- finger ordering;
- tactile sensor IDs;
- tactile axis names;
- tactile point indices.

Keep final predicted action shape and finite-value validation at the actual Policy -> Real execution boundary.

## 16. Remove duplicate runtime config wrappers

Delete FingertipAssemblerConfig if the policy child can construct FK from the runtime object it already receives.

Use current Real-owned:

- XHand URDF;
- runtime.hand.fingertip_link_names;
- runtime.hand.T_eef_handbase_pos_xyz;
- runtime.hand.T_eef_handbase_quat_wxyz.

Shrink or remove PolicyRuntimeConfig.

A tiny process argument grouping object is acceptable only if it improves readability. It must not become another contract.

## 17. Preserve current safe startup order

The current RuntimeSupervisor.start([policy]) waits for policy_ready before sensor/hardware processes are started.

Preserve this property.

Required order:

    read saved config / pure preflight
    -> allocate RuntimeChannels
    -> spawn policy process
    -> policy loads checkpoint + model + warmup
    -> policy_ready
    -> only then start arm/hand/camera/pointcloud/recorder workers
    -> ARMED

Do not reorder hardware before model restore.

## 18. Preserve physical safety architecture

Do not simplify away:

- SafetyState;
- run_id;
- motion_lock;
- arm/hand SDK worker ownership;
- camera/point-cloud worker ownership;
- policy/CUDA worker ownership;
- final worker-side authority fence;
- finite SDK targets;
- physical/mechanical limits;
- xArm speed/acceleration controls;
- emergency stop;
- sensor freshness;
- latest-target semantics;
- verified process shutdown;
- Raw v34 recording semantics;
- recorder exact START/STOP behavior;
- episode_valid / frame_valid.

Do not add:

- command ACK/adoption/reached ledgers;
- per-command transactions;
- generic software delta/slew filters;
- hard CUDA watchdog/heartbeat infrastructure;
- checkpoint hashes/signatures;
- Git/URDF/agent/geometry ABI gates;
- async RTC execution.

Keep synchronous chunk execution:

    queue empty -> inference -> execute chunk

Do not copy ManiUniCon stale-prefix filtering into this executor.

## 19. Fix actual START lifecycle bugs

### 19.1 Require hand-enabled dexterous deployment

Fail in pure preflight when runtime.policy.hand_enabled is false.

Do not imply arm-only support.

### 19.2 B is one-shot

Consume start_request before attempting admission.

If admission fails for any reason, another physical B press is required.

Do not leave B level-latched until sensors recover.

### 19.3 Recheck both arm and hand home at B

Require:

- physical_home_completed;
- arm qpos within runtime.arm.homing.convergence_rad of runtime.arm.home_qpos;
- hand qpos within an appropriate current Real home tolerance of deg2rad(runtime.hand.home_qpos_deg).

Prefer one small shared helper if HOME code already exposes equivalent logic.

Start pose is an evaluation protocol property of current Real runtime, not checkpoint metadata.

### 19.4 Build the exact first Policy observation before RUNNING

Use episode-start repeated-row padding:

    [row] * n_obs_steps

Run the same normal build_policy_observation path.

This must catch unavailable tactile, cloud, RGB, FK, and other required modalities before recorder START/RUNNING.

If it fails:

- remain ARMED;
- do not begin an episode;
- require another B.

### 19.5 Recheck after recorder START

Recorder START may block.

After it succeeds:

    read fresh row
    -> recheck arm + hand start pose
    -> rebuild full first Policy observation
    -> commit RUNNING

On failure:

- discard the just-started recording with save=false;
- stay ARMED;
- require another B.

Keep direct flow:

    request -> admission -> recorder prepare -> fresh recheck -> RUNNING

Do not build a generic transaction abstraction.

## 20. Reset EEF IK episode state

Add a small planner/IK reset_episode path.

At each successfully begun episode:

    model.reset_episode()
    planner.reset_episode()

Reset:

- IK fallback RNG to the configured fixed seed;
- episode-local warning/failure state where applicable.

Do not change model seed semantics to base_seed + episode.

## 21. Keep timing and metrics simple

Do not add a metrics database or formal benchmark schema.

Make RolloutStats episode-local rather than silently accumulating across episodes.

At episode end, log one clear per-episode summary containing existing useful values:

- publications;
- inference mean/p95/max;
- action interval mean/p95/max and effective Hz;
- arm/hand/workspace clipping;
- IK failures;
- publication rejection count if already available.

A dedicated rollout_metrics.jsonl is not required for this refactor. Add it only if it is materially simpler than using the existing logger and introduces no new schema machinery.

Task success remains offline.

## 22. Keep timeout semantics simple

Do not add a Main CUDA watchdog merely to make max_running_s a hard wall-clock deadline.

Treat it as the cooperative runner episode budget.

Safety authority remains:

- S/Q/ESC;
- SafetyState;
- run_id;
- worker-side command fence.

A blocked inference cannot publish a newly valid command after authority is revoked.

## 23. run_config.yaml should stay small

Keep one research provenance file.

Record useful execution facts only:

- experiment selector;
- resolved checkpoint path/name;
- raw/EMA choice;
- actual inference steps;
- seed/device;
- episode count/max duration;
- resolved Real runtime config;
- compact PolicyInfo;
- best-effort dexmani_real HEAD;
- best-effort ../dexmani_policy HEAD.

Do not hash the checkpoint.

Do not gate on Git cleanliness.

Git status is optional; omit it if removing it makes the code simpler.

## 24. Session setup order

Before creating the rollout session directory or starting any process:

    resolve experiment
    -> load saved experiment config.yaml
    -> resolve checkpoint selection
    -> derive PolicyInfo
    -> resolve Real config
    -> pure compatibility preflight
    -> validate episode/recording limits

Then:

    create session directory
    -> write run_config.yaml
    -> run deployment

Do not introduce DeploymentPlan.

## 25. Thin Policy runtime API

Keep the cross-repository public surface small.

A suitable API:

    resolve_experiment(...)
    resolve_checkpoint(...)
    load_experiment_config(...)
    inspect_policy(...)
    load_policy(...)
    PolicyInfo
    LoadedPolicy

Names may be adjusted to minimize churn.

LoadedPolicy should approximately expose:

    info
    warmup(...)
    predict(obs) -> np.ndarray
    reset_episode()
    close()

Prediction validation should only protect the actual execution boundary:

- required observation fields present;
- deterministic preprocessing succeeds;
- predict_action succeeds;
- control_action exists;
- exact expected physical action shape;
- all finite.

Delete validation that control_action must equal a hard-coded slice of pred_action.

Policy-owned action selection is research behavior.

## 26. Warmup

Keep warmup for CUDA lazy initialization.

Prefer:

    reset_episode()
    warmup
    reset_episode()

Do not preserve the current large global Python/NumPy/Torch/CUDA RNG save/restore wrapper unless a concrete Policy still needs it after the refactor.

Do not add warmup-only persisted metadata.

## 27. Files expected to change

Trace callers before editing.

### dexmani_policy

Likely:

- dexmani_policy/common/checkpoint_io.py
- shared inference helper/module
- dexmani_policy/training/resume.py
- dexmani_policy/training/trainer.py
- dexmani_policy/training/workspace.py
- dexmani_policy/training/eval_utils.py
- dexmani_policy/training/build_utils.py
- dexmani_policy/datasets/real_policy_contract.py
- Uni3D initialization owner
- dexmani_policy/agents/core/dqrise.py and codebook owner
- dexmani_policy/deployment/runtime.py
- dexmani_policy/deployment/__init__.py
- dexmani_policy/smoke_test.py
- README.md
- AGENTS.md

Delete when unused:

- dexmani_policy/deployment/export.py
- dexmani_policy/deployment/contract.py
- dexmani_policy/deployment/restore.py
- obsolete deployment export scripts

### dexmani_real

Likely:

- examples/run_policy.py
- dexmani_real/deployment/config.py
- dexmani_real/deployment/session.py
- dexmani_real/deployment/runner.py
- dexmani_real/deployment/observation.py
- planner/IK reset ownership
- dexmani_real/planning/kinematics/ik.py
- dexmani_real/config/pointcloud.py
- README.md
- AGENTS.md if stale

Do not change Raw v34 or Policy Zarr v15 unless strictly necessary.

## 28. Explicit non-goals

Do not:

- support old checkpoints;
- redesign Raw v34;
- redesign Policy Zarr v15;
- add runtime timing arrays to Zarr;
- alter latest-causal observation semantics;
- introduce cross-modal interpolation;
- add async/RTC inference;
- alter action representation;
- add SAT-style action spaces;
- add ACK ledgers;
- add generic slew filtering;
- add hard watchdog infrastructure;
- redesign HOME planning;
- add normal-stream path/environment collision planning;
- move task-success judgment online;
- introduce plugin/factory/version registries;
- create a tests directory.

Do not silently alter trained numerical behavior while deleting infrastructure.

## 29. Old 25 findings after this refactor

Expected closure:

1. DEP-001 — FIXED: hand-enabled preflight.
2. DEP-002 — FIXED: one-shot B.
3. DEP-003 — FIXED: arm+hand start pose.
4. DEP-004 — FIXED: full initial observation before RUNNING.
5. DEP-005 — ACCEPTED DESIGN: cooperative timeout, not safety watchdog.
6. DEP-006 — ELIMINATED: no deployment artifact identity.
7. DEP-007 — ELIMINATED: no duplicate persisted PolicySpec.
8. DEP-008 — FIXED: IK/planner episode reset.
9. DEP-009 — ACCEPTED DESIGN: synchronous chunk baseline; timing remains visible.
10. DEP-010 — FIXED: post-recorder fresh recheck.
11. DEP-011 — ACCEPTED DESIGN: start pose belongs to current Real eval protocol.
12. DEP-012 — LIGHTWEIGHT: two repo HEADs in run provenance.
13. DEP-013 — FIXED where useful: pure preflight before session directory; no session framework.
14. DEP-014 — ACCEPTED DESIGN: task success offline.
15. DEP-015 — OPTIMIZED: episode-local stats and per-episode summary log.
16. DEP-016 — ELIMINATED: no second semantic PolicySpec projection.
17. DEP-017 — OPTIMIZED: per-episode clipping/IK summary.
18. DEP-018 — FIXED: compact direct experiment-config + checkpoint load/predict smoke.
19. DEP-019 — ACCEPTED BY DESIGN: experiment directory is the artifact unit; standalone checkpoint is intentionally unnecessary.
20. DEP-020 — ACCEPTED DESIGN: latest-causal semantics; no new skew gate.
21. DEP-021 — ELIMINATED AS REQUIREMENT: numerical pointcloud config, no ABI.
22. DEP-022 — ELIMINATED AS REQUIREMENT: saved config + strict state_dict, no agent ABI.
23. DEP-023 — ELIMINATED: no deployment artifact producer metadata.
24. DEP-024 — ELIMINATED: no deployment artifact publication path.
25. DEP-025 — ACCEPTED LOW-VALUE ISSUE: no extra architecture for fail-earlier HOME planner creation.

Do not finish with DEP-001, DEP-002, DEP-003, DEP-004, DEP-008, DEP-010, or DEP-018 open.

## 30. Validation

No hardware execution.

### Policy

At minimum:

    cd ../dexmani_policy
    conda run --no-capture-output -n policy python -m compileall -q dexmani_policy
    conda run --no-capture-output -n policy ruff format --check dexmani_policy
    conda run --no-capture-output -n policy ruff check --select F401,F821,F822,F823,I dexmani_policy
    git diff --check

Use existing config-only/full smoke entry points for representative affected policies.

Cover at least:

- one RGB policy;
- R3D/Uni3D;
- DQ-RISE;
- one joint-action policy;
- one EEF-action policy if an existing config supports it.

Compact inference smoke must prove:

    build/save resolved experiment config
    -> save normal training checkpoint
    -> load saved config
    -> select raw/EMA
    -> strict model restore
    -> synthetic predict
    -> exact physical action shape + finite output

Also prove one intentionally corrupted state_dict fails strict restore.

Do not add a tests directory.

CUDA diagnostic is allowed:

    conda run --no-capture-output -n policy python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

If a sandbox reports unavailable, report NOT VERIFIED. Do not change code.

### Real

At minimum:

    conda run --no-capture-output -n real_robot python -m compileall -q dexmani_real examples
    conda run --no-capture-output -n real_robot ruff format --check dexmani_real examples
    conda run --no-capture-output -n real_robot ruff check --select F401,F821,F822,F823,I dexmani_real examples
    git diff --check

Focused pure-logic checks should cover:

- PolicyInfo derivation from saved config;
- compatibility preflight;
- one-shot B;
- first observation admission;
- hand-home tolerance;
- PointCloudConfig partial/default parsing;
- IK reset reproducibility.

Do not connect hardware.

## 31. End-to-end acceptance trace

Before finishing, trace:

    examples/run_policy.py
      -> experiment/config.yaml
      -> checkpoint selection
      -> PolicyInfo
      -> Real preflight
      -> RuntimeChannels sizing
      -> policy process
      -> normal checkpoint load
      -> warmup / policy_ready
      -> hardware workers
      -> PolicyRunner
      -> observation builder
      -> LoadedPolicy.predict
      -> decode/projection/IK
      -> publish_command
      -> worker run_id fence

Confirm:

- no deployment artifact is produced/read;
- no PolicySpec/DeploymentSpec persists;
- current repository source YAML cannot redefine a saved experiment;
- parent does not load model weights merely to inspect policy requirements;
- hardware starts only after policy restore/warmup succeeds;
- Real still sizes pointcloud SHM correctly;
- model/CUDA remains in policy worker;
- SDKs remain in hardware workers;
- blocking inference still cannot bypass run_id revocation;
- recorder semantics remain intact;
- Raw semantics remain intact.

## 32. Code-quality requirement

The final diff should delete substantially more deployment machinery than it adds.

Prefer:

- saved resolved config;
- plain functions;
- direct dict access;
- one small runtime dataclass;
- Agent-owned algorithm-specific logic;
- direct lifecycle control flow.

Avoid:

- new manager/service/factory frameworks;
- schema registries;
- compatibility matrices;
- generic transaction objects;
- duplicate validation of internal canonical values.

Comments should explain robotics, safety, timing, numerical, or ownership reasons, not refactor history.

## 33. Documentation cleanup

Because this changes a public workflow:

- update dexmani_policy/AGENTS.md Deployment Boundary;
- update stale dexmani_real standing instructions if needed;
- update README deployment/eval commands;
- remove deployment export / --artifact instructions and obsolete scripts.

Do not turn AGENTS into a file inventory.

## 34. Final report

Report:

1. final diff summary for both repositories;
2. deleted files/mechanisms;
3. final experiment-config + checkpoint inference path;
4. START/IK fixes;
5. actual validation commands and PASS/FAIL/NOT VERIFIED;
6. environment limitations;
7. compact DEP-001..DEP-025 closure table.

Do not claim hardware validation unless explicitly authorized and performed.

## 35. Definition of done

Done only when:

- experiment config.yaml is the single saved inference configuration source;
- it snapshots only minimal real_runtime facts from the actual training Zarr;
- normal training checkpoint is used directly for offline and Real inference;
- old checkpoint compatibility is intentionally absent;
- no deployment artifact/export/contract/restore subsystem remains;
- offline and Real use one shared strict inference restore path;
- inference never validates training resume contracts;
- Uni3D/DQ-RISE external initialization assets are not required for complete-checkpoint inference;
- Real derives only a tiny runtime PolicyInfo from saved config;
- compatibility validation is limited to actual runtime capability;
- B/start lifecycle bugs are fixed;
- IK episode RNG/state resets;
- episode stats no longer silently accumulate across episodes;
- existing safety/process/recording architecture remains intact;
- no production-style replacement framework is introduced;
- focused offline validation passes or is clearly reported NOT VERIFIED.
