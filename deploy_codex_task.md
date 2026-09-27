# DexMani deployment simplification — Codex task

## 0. Mission

Refactor dexmani_policy and dexmani_real into a research-oriented real-robot evaluation stack with one checkpoint format, one inference restore path, and the existing Real physical-safety runtime.

This is a deliberate breaking change. Do not preserve compatibility with existing simple.v3 training checkpoints, deployment.v3 artifacts, deployment_latest.pt, or the current deployment artifact APIs. Do not build migration code, compatibility adapters, deprecated aliases, or dual old/new execution paths.

The repositories are personal PhD research repositories, not production software or a public model-serving platform. Optimize for:

1. physical safety,
2. experiment correctness,
3. research iteration speed,
4. direct readable code,
5. minimal long-lived abstractions.

Prefer deletion over adding validation frameworks, schema registries, ABI layers, hashes, transactions, or compatibility machinery.

The intended design is informed by:

- official Diffusion Policy: a training checkpoint is directly restorable for inference rather than requiring a second deployment artifact format;
- LeRobot: model persistence is fundamentally configuration plus weights, while rollout execution is a separate concern;
- ManiUniCon: keep policy/runtime wrappers thin and add asynchronous execution machinery only when experiments require it;
- existing DexMani Real: retain its stronger physical process ownership, run_id fencing, safety state, sensor freshness, homing, and recorder integrity.

## 1. Local repository and environment facts

Run Codex from the dexmani_real repository.

The sibling Policy repository is:

    ../dexmani_policy

Use the existing managed Conda environments:

    dexmani_policy: conda env policy
    dexmani_real:   conda env real_robot

For non-interactive commands prefer:

    conda run --no-capture-output -n policy <command>
    conda run --no-capture-output -n real_robot <command>

GPU and CUDA are available and correctly configured in the real development machine. If a sandboxed tool reports that CUDA/GPU is unavailable, treat that as a sandbox limitation. Do not change the implementation to CPU-only behavior, disable CUDA paths, remove GPU validation, or alter policy defaults merely to satisfy a sandbox probe.

Do not install or upgrade dependencies unless the repository already requires that for this task.

Before editing either repository:

    git status --short

Preserve unrelated user changes.

Read and obey both repositories' AGENTS.md files. Where this task intentionally changes an existing Deployment Boundary rule, update the relevant AGENTS.md at the end so standing instructions describe the new stable design.

Do not create a tests/ directory.

Do not run hardware-affecting examples or connect xArm, XHand, RealSense, VR, calibration, homing, teleoperation, replay, or rollout during validation.

## 2. Non-negotiable architectural target

The final model lifecycle must be:

    training
        |
        v
    unified training checkpoint
        |
        +--> exact training resume
        |
        +--> simulation/offline inference
        |
        +--> real-robot inference
                  |
                  v
              PolicyInfo
                  |
                  v
          dexmani_real runtime
                  |
                  v
           physical safety boundary

There must no longer be:

    training checkpoint
        -> deployment export
        -> deployment artifact
        -> deployment schema
        -> DeploymentSpec / PolicySpec
        -> restore contract
        -> Real contract re-validation

The only persistent model file used by inference must be the normal training checkpoint.

The final strictness distribution is:

- exact training resume: strict;
- model parameter restoration: strict state_dict restoration;
- physical robot boundary: strict;
- inference metadata and ordinary repository refactors: minimal validation only.

## 3. New unified checkpoint format

Replace simple.v3 with one clean major format. Existing checkpoints do not need to load.

Use one stable top-level identifier, for example:

    _format: dexmani.checkpoint.v1

Do not introduce simple.v4 plus a separate deployment format.

### 3.1 Recommended payload ownership

The checkpoint should contain only a small number of top-level sections with clear ownership:

    {
        "_format": "dexmani.checkpoint.v1",
        "_saved_at": ...,

        "config": <resolved plain training config>,

        "real_runtime": {
            "control_dt_s": ...,
            "pointcloud": {...} or null
        } or null,

        "weights": {
            "model": ...,
            "ema": ... or null
        },

        "train": {
            "epoch": ...,
            "global_step": ...,
            "next_micro_step": ...,
            "optimizer": ...,
            "scheduler": ...,
            "ema_updater_step": ...,
            "ema_decay": ...,
            "rng_states": ...,
            "resume_contract": ...
        }
    }

The exact implementation may differ slightly if the existing trainer makes another arrangement substantially simpler, but preserve the ownership principles below.

### 3.2 config is the inference architecture/configuration source of truth

Store one resolved plain configuration snapshot.

Inference must obtain architecture and model-facing behavior from this saved config, especially:

- agent constructor config;
- action_key;
- n_obs_steps / n_action_steps / horizon through the saved Agent config;
- dataset sensor_modalities;
- RGB validation preprocessing settings;
- normal inference defaults when needed.

Do not mirror the same information into PolicySpec, DeploymentSpec, AgentContract, RgbPreprocessingSpec, or another persisted inference grammar.

The experiment's current config.yaml may still be useful to humans, but changing config.yaml after training must not alter the behavior of an already saved checkpoint.

### 3.3 real_runtime stores only Real-produced numerical facts that model config cannot reconstruct

For Real Policy Zarr training, snapshot only facts that the Real runtime genuinely needs but the Agent/config cannot reconstruct on its own.

Initially this should be limited to:

- control_dt_s;
- the numerical PointCloudConfig used to create the training input, when point_cloud is consumed.

Do not store or gate on:

- Git commit;
- URDF hash;
- point-cloud algorithm ID string;
- implementation description;
- source function names;
- camera serial;
- camera calibration;
- current table plane;
- hand mount calibration;
- semantic ABI version;
- agent ABI version;
- geometry ABI version.

Current physical calibration and hardware configuration remain owned by dexmani_real at runtime.

If a checkpoint is not trained from one canonical Real Policy Zarr, real_runtime may be null. Real deployment must reject such a checkpoint with a short clear error instead of guessing.

### 3.4 weights own learned numerical state

The selected model state owns:

- learned parameters;
- fitted normalizer parameters;
- persistent codebook buffers;
- any other persistent inference state that is already part of the module state_dict.

Do not add a separate inference normalization contract. Training may still validate its normalization recipe, but inference must not require current code to reproduce the old normalization mode metadata when the fitted scale/offset tensors are already in the state_dict.

### 3.5 train is training-only

Keep exact training resume strict.

It is acceptable for the current training resume contract to remain strict and detailed if that is useful for exact continuation. It must live entirely on the training path and must not be parsed or validated by simulation inference, checkpoint inspection for Real, or real-robot inference.

Do not spend this task building a generic version migration framework for training resume.

## 4. Checkpoint serialization must be safe to inspect

dexmani_real must inspect the training checkpoint before allocating RuntimeChannels because the point-cloud shared-memory dtype requires num_points.

Therefore the new checkpoint must be loadable with a safe lightweight inspection call such as:

    torch.load(
        checkpoint_path,
        map_location="meta",
        weights_only=True,
    )

Design the saved payload so weights_only=True works for the entire checkpoint.

Persist plain safe structures:

- dict / list / tuple;
- str / int / float / bool / None;
- torch.Tensor;
- other types known to be accepted by weights_only loading only when unavoidable.

Do not persist OmegaConf objects, custom dataclasses/classes, or arbitrary pickle-only objects.

Pay special attention to RNG state. Current NumPy RNG capture contains ndarray state. Convert saved RNG state to a weights_only-safe plain/tensor representation and restore it exactly on training resume. Do not weaken exact training resume merely to satisfy inspection.

The inspector must not instantiate the model and must not allocate real model tensors on CPU/GPU.

## 5. Checkpoint reader philosophy

Do not repeat the current exact-key schema style.

Avoid code such as:

    if set(payload) != EXPECTED_KEYS:
        raise ...

Require only fields actually needed by the current operation.

Unknown additive fields must be ignored.

The format major identifier should change only for a genuinely incompatible core meaning, not when a metadata field is added.

Because old checkpoints are explicitly out of scope, there is no need for simple.v3 fallback, migration, legacy parsing, or warnings for old formats. Reject them clearly and briefly.

## 6. One inference loader for all evaluation

Create one Policy-side inference restore path used by:

- select_best_ckpt / checkpoint evaluation;
- eval_best_ckpt;
- record_demo where applicable;
- dexmani_real real-robot rollout.

There must not be one restore implementation in training/eval_utils.py and another in deployment/restore.py.

The shared loader should conceptually do only:

    read checkpoint
    -> read saved config
    -> instantiate saved Agent
    -> set action_key / necessary runtime properties
    -> select raw or EMA state
    -> strict load_state_dict
    -> move to device
    -> eval mode
    -> return thin LoadedPolicy

Keep model parameter restoration strict=True.

Do not use strict=False as a general compatibility strategy.

Do not call validate_resume_contract from the inference path.

Do not re-build and compare a second AgentContract after construction.

## 7. Agent construction and training-only external assets

Remove the need for deployment/export._sanitize_agent_config.

The generic inference loader must not understand R3D, DQ-RISE, SAT, ManiFlow, or any other concrete algorithm.

Trace the two current cases that make inference construction depend on training-only external assets:

1. Uni3D pretrained initialization:
   - current pc_encoder_config can request use_pretrained_weights and a path;
   - once a checkpoint contains the trained encoder state, inference must not re-read the pretrained initialization file.

2. DQ-RISE codebook initialization:
   - current Agent constructor can receive codebook_path;
   - complete persistent codebook state already belongs in the model state_dict;
   - inference must not require the original external NPZ.

Preferred outcome:

- constructors used by checkpoint inference construct architecture and empty/persistent module state only;
- training initialization of external pretrained/codebook assets happens explicitly in the training build path before training;
- the checkpoint state_dict fully restores inference state.

Keep this refactor targeted. Do not rewrite unrelated Agents.

If an Agent needs an explicit training-only initialize_from_* method, keep that method local to the Agent/encoder that owns the data rather than adding algorithm-specific logic to the generic checkpoint loader.

Preserve existing training behavior numerically.

## 8. RGB preprocessing ownership

Do not create RgbPreprocessingSpec.

Use the saved resolved dataset config directly for deterministic validation/real inference preprocessing:

- rgb_preprocess_size;
- rgb_random_crop_size used as center crop for validation/inference;
- rgb_keep_uint8 where relevant.

The Agent's own ImageProcessor remains Agent-owned.

Real continues to provide raw uint8 HWC RGB.

The thin Policy runtime performs the same deterministic preprocessing used by validation before the Agent sees RGB.

Do not persist warmup_rgb_hw. Synthetic warmup can choose an input H/W compatible with the saved resize/crop recipe.

## 9. Point-cloud configuration must be additive-friendly

Keep PointCloudConfig as a numerical configuration object in dexmani_real, but remove exact-key coupling.

Current PointCloudConfig.from_dict requires the persisted dict to contain exactly every dataclass field. Change the parser so:

- known persisted values override dataclass defaults;
- a newly added config field can use its dataclass default when absent;
- adding a new optional parameter with an old-behavior default does not invalidate an earlier new-format checkpoint;
- clearly unknown input keys should still fail rather than be silently ignored, unless there is a strong project-specific reason not to.

New PointCloudConfig defaults must preserve old behavior when introducing an optional parameter.

Do not introduce point-cloud ABI IDs or version registries.

The checkpoint freezes the numerical recipe, not implementation identity.

## 10. Thin runtime PolicyInfo

Replace persisted PolicySpec / ObservationFieldSpec / DeploymentSpec with one ordinary runtime-only inspection result.

A suitable shape is:

    PolicyInfo:
        checkpoint_path
        policy_name / task_name if useful for display only
        observation_fields: tuple[str, ...]
        n_obs_steps
        n_action_steps
        action_mode
        control_dt_s
        pointcloud_config: dict | None
        default inference settings if useful

Do not serialize PolicyInfo into the checkpoint.

Derive values directly from the saved config and real_runtime.

Derive action_mode from the canonical action_key:

- action -> joint
- action_ee -> eef

Do not add an action ABI.

The parent Real process needs PolicyInfo only to:

- decide which sensor workers are required;
- size point-cloud shared memory;
- know history/chunk lengths and cadence;
- select joint versus EEF execution;
- print the run summary.

## 11. Policy checkpoint selection and CLI

Remove deployment artifacts and deployment_latest.pt.

Preserve useful research ergonomics for selecting experiments/checkpoints, but use the training checkpoint directly.

Recommended CLI direction:

    python examples/run_policy.py EXPERIMENT \
        --checkpoint best|latest|<checkpoint filename/path> \
        [--inference-steps N] \
        [--weights ema|raw] \
        ...

Remove --artifact.

best_ckpt.json may remain a selection record.

Make best_ckpt.json parsing additive-friendly:

- require only fields needed for selection;
- ignore unknown extra bookkeeping fields;
- use saved best inference settings when present;
- do not require an exact top-level key set.

For an explicit/latest checkpoint, use checkpoint/config inference defaults unless the CLI overrides them.

Do not create a second persisted inference-default contract solely for deployment.

## 12. Delete the deployment artifact subsystem

After the new direct-checkpoint path is working, delete obsolete mechanisms rather than leaving them deprecated.

In dexmani_policy remove, if no longer needed:

- dexmani_policy/deployment/export.py;
- dexmani_policy/deployment/contract.py;
- dexmani_policy/deployment/restore.py;
- ExportReceipt;
- deployment artifact publication/selector logic;
- deployment_latest.pt assumptions;
- DEPLOYMENT_FORMAT / deployment.v3 grammar;
- PolicySpec / ObservationFieldSpec / DeploymentSpec;
- RgbPreprocessingSpec;
- producer/source_commit artifact metadata;
- artifact verification and atomic publication code;
- deployment export scripts and stale README instructions.

Keep dexmani_policy/deployment/runtime.py only if it remains a useful thin public inference bridge. It may import/re-export shared inference helpers; do not recreate the old contract there under new names.

Do not retain a hidden legacy path for old artifacts.

## 13. dexmani_real compatibility boundary must become small

Rewrite dexmani_real/deployment/config.py around actual Real capability, not mirrored tensor contracts.

Real compatibility preflight should approximately check only:

1. runtime.policy.hand_enabled must be true for the current DexMani dexterous deployment;
2. all requested observation field names are supported and include joint_state;
3. policy control frequency does not exceed the limiting arm/hand worker service rate;
4. if point_cloud is requested, the checkpoint has a valid PointCloudConfig.

Do not continue cross-repository equality validation of:

- exact field dtype declarations;
- exact tensor shape declarations that Real itself constructs canonically;
- joint-name arrays;
- RGB channel ordering;
- finger ordering;
- tactile sensor IDs;
- tactile axis labels;
- tactile point index arrays.

These are canonical DexMani implementation invariants, not public user-supplied schemas.

Keep output action shape and finite-value validation at the actual Policy -> Real execution boundary.

## 14. Remove duplicate runtime configuration objects

Delete FingertipAssemblerConfig if the policy child can construct fingertip FK directly from the runtime object it already receives.

The policy child should use the current Real-owned:

- XHand URDF;
- runtime.hand.fingertip_link_names;
- runtime.hand.T_eef_handbase_pos_xyz;
- runtime.hand.T_eef_handbase_quat_wxyz.

Do not duplicate these values solely to make a second configuration contract.

Shrink or remove PolicyRuntimeConfig after PolicySpec/artifact disappear. A small runtime grouping dataclass is acceptable if it materially improves process spawning readability, but it must contain only actual execution inputs such as:

- checkpoint path;
- device;
- seed;
- inference_steps / EMA selection;
- PolicyInfo if needed.

It must not become another model contract.

## 15. Preserve Real safety/process architecture

Do not simplify away mechanisms that correspond to actual hardware/resource boundaries.

Preserve:

- SafetyState DISARMED / ARMED / RUNNING / FAULT;
- run_id lifecycle epoch;
- motion_lock;
- SDK-owner arm and hand workers;
- camera worker and point-cloud worker ownership;
- model/CUDA ownership in the policy process;
- final worker-side run_id authority fence before SDK admission;
- finite SDK target validation;
- arm/hand physical/mechanical limits;
- xArm speed/acceleration controls;
- emergency stop and safe disconnect;
- sensor freshness;
- latest-target command semantics;
- bounded verified process shutdown before shared-memory release;
- Raw v34 recording semantics;
- recorder exact START/STOP sequence behavior;
- episode_valid and frame_valid semantics.

Do not add:

- command ACK/adoption/reached ledgers;
- per-command transactions;
- generic software delta/slew filters;
- a hard CUDA heartbeat/watchdog framework;
- artifact hashes/signatures;
- repository-cleanliness gates;
- URDF/git/algorithm ABI gates;
- async RTC execution as part of this task.

Synchronous chunk execution remains the baseline:

    queue empty -> inference -> execute chunk

Do not copy ManiUniCon stale-prefix dropping into this empty-queue synchronous executor.

## 16. Fix the real START lifecycle bugs

These are real runtime bugs and must be fixed independently of contract deletion.

### 16.1 policy deployment requires the hand

Fail before hardware startup when:

    runtime.policy.hand_enabled == false

The current Policy/Real stack is dexterous and still requires fresh hand observation and hand action execution. Do not silently interpret this as arm-only policy support.

### 16.2 B is a one-shot fresh authorization edge

Current start_request can remain true when _begin_episode returns because observation is unavailable, causing an automatic later start.

Change semantics so a B request is consumed once before attempting admission.

Conceptually:

    if idle and start_request:
        consume start_request
        try _begin_episode once

Any admission failure requires a new physical B press.

Do not create a StartTransaction abstraction.

### 16.3 revalidate both arm and hand start pose at B

At B admission, require:

- physical_home_completed is true;
- current arm qpos is within runtime.arm.homing.convergence_rad of runtime.arm.home_qpos;
- current hand qpos is within a clearly resolved tolerance of the canonical current Real hand home.

Use:

    home hand = deg2rad(runtime.hand.home_qpos_deg)

For the initial implementation, runtime.hand.home_tolerance_deg is acceptable as the B-time hand tolerance unless the existing HOME code already exposes a better shared helper. Prefer reusing one small helper over duplicating conversion/tolerance math.

This is an Evaluation Protocol check using the current Real runtime. Do not put the start pose in the model checkpoint.

### 16.4 validate the exact first Policy observation before RUNNING

Before recorder START / RUNNING commit, prove that the first model input can actually be built.

Use the current episode-start padding semantics:

    initial_rows = [row] * n_obs_steps

Run the same build_policy_observation path that normal execution uses.

This must catch at least:

- required dense/aggregate tactile invalidity;
- missing point cloud;
- RGB requirements;
- EEF/fingertip FK construction failure;
- other required initial modality failure.

If initial input cannot be built:

- remain ARMED;
- do not start an episode;
- do not create/save a recorder episode;
- require another B.

### 16.5 recheck after recorder prepare

Recorder START is a resource preparation step and may block.

After recorder.start_episode succeeds, read one fresh observation and repeat the start-pose/full-input checks before entering RUNNING.

If the recheck fails:

- stop/discard that just-started recording with save=false;
- stay ARMED;
- require another B.

Keep the implementation direct:

    request -> admission -> recorder prepare -> recheck -> RUNNING

Do not build a generic transaction framework.

## 17. Reset EEF IK episode state

Current EEF planner/OnlineIKSolver RNG is session-lived while model inference seed resets per episode.

Add a small planner/IK reset_episode path.

At each successfully begun episode:

    model.reset_episode()
    planner.reset_episode()

Reset:

- IK fallback RNG to the configured base random seed;
- episode-local IK warning/failure state where applicable.

Do not change the current model inference seed policy to base_seed + episode.

## 18. Keep timeout semantics simple

Do not add a separate Main heartbeat/watchdog solely to make max_running_s a hard CUDA wall-clock deadline.

Document/retain:

- max_running_s is a cooperative runner episode budget;
- S/Q/ESC, SafetyState and run_id revocation are the actual safety authority;
- a blocked inference cannot publish a fresh command after authority was revoked because runner and worker fences remain.

This is an accepted research-runtime limitation, not an open P1 issue.

## 19. Per-episode research metrics

Current RolloutStats aggregates most values across a multi-episode session.

Make metrics episode-local.

At the beginning of a successfully committed episode, reset the episode statistics.

At the end of each begun episode append one compact JSON object to:

    <session_dir>/rollout_metrics.jsonl

Use one writer only; the policy process is acceptable.

Include useful research/executor facts such as:

- episode index;
- termination reason;
- duration;
- publications/steps;
- inference count;
- inference mean / p95 / max ms;
- action-step interval mean / p95 / max ms;
- effective action Hz;
- arm clip count and max magnitude;
- hand clip count and max magnitude;
- workspace clip count and max magnitude;
- IK failure counts;
- publication rejections if available.

Do not add task_success to the control/runtime state. Task success remains an offline human/evaluation judgment.

Do not build a metrics database, schema registry, server, or annotation system.

## 20. run_config.yaml is provenance, not a compatibility gate

Keep one simple session run_config.yaml.

Record at least:

- resolved checkpoint path and checkpoint selector;
- actual raw/EMA selection;
- actual inference_steps;
- seed/device;
- num episodes/max duration;
- resolved Real runtime config;
- compact PolicyInfo;
- dexmani_real HEAD and status, best effort;
- ../dexmani_policy HEAD and status, best effort.

Do not hash the checkpoint.

Do not reject dirty repositories.

Do not require commit equality.

Provenance helps interpret experiments; it does not decide whether a checkpoint is allowed to run.

## 21. Session setup order

Pure checkpoint inspection and Real compatibility preflight must happen before creating the rollout session directory and before spawning model or hardware workers.

Recommended CLI order:

    resolve checkpoint / selection
    -> inspect checkpoint
    -> resolve Real config
    -> validate PolicyInfo vs Real capability
    -> validate episode/recording limits
    -> create session directory
    -> write run_config.yaml
    -> start deployment

Do not build a DeploymentPlan class unless the final code is materially simpler than direct local variables.

## 22. Public Policy API after refactor

Keep the public cross-repository surface very small.

A reasonable API is:

    resolve_checkpoint(...)
    inspect_checkpoint(...)
    load_policy(...)
    PolicyInfo
    LoadedPolicy

Exact names may reuse current runtime naming if that reduces churn, but do not retain experiment/artifact terminology that no longer exists.

LoadedPolicy should expose approximately:

    info
    warmup(...)
    predict(observation) -> np.ndarray [n_action_steps, physical_action_dim]
    reset_episode()
    close()

Prediction validation should focus on the actual execution boundary:

- required observation fields are present;
- deterministic preprocessing succeeds;
- Agent predict_action runs;
- control_action exists;
- returned physical action shape is exactly expected;
- returned action is finite.

Delete validation that control_action must equal a hard-coded slice of pred_action. Action selection is Policy-owned research behavior.

Do not require full pred_action/tail outputs if the Real runtime never consumes them.

## 23. Warmup

Keep warmup because CUDA lazy initialization before the first physical action is useful.

Simplify RNG handling.

Preferred semantics:

    reset_episode()
    warmup synthetic samples
    reset_episode()

Do not preserve a large save/restore wrapper for global Python/NumPy/Torch/CUDA RNG states unless an actual Policy requires it after this refactor.

Synthetic observations should be generated from PolicyInfo/config and the normal preprocessing path. Do not add warmup-only persisted metadata.

## 24. best_ckpt and evaluation utilities

Refactor training/eval_utils.py so inference uses the same loader as Real.

Keep selection/statistics helpers that are genuinely evaluation-specific.

Relax best_ckpt.json from exact-set validation. Required selection keys must be checked; additive unknown bookkeeping fields should not invalidate the record.

Do not let best-checkpoint bookkeeping become a model compatibility layer.

## 25. Files expected to change

Trace actual callers before editing, but expect at least these areas.

### dexmani_policy

Primary:

- dexmani_policy/common/checkpoint_io.py
- dexmani_policy/common/inference.py or the chosen shared inference module
- dexmani_policy/training/resume.py
- dexmani_policy/training/trainer.py
- dexmani_policy/training/workspace.py
- dexmani_policy/training/eval_utils.py
- dexmani_policy/training/build_utils.py
- dexmani_policy/datasets/real_policy_contract.py
- relevant Agent/encoder code for Uni3D initialization
- dexmani_policy/agents/core/dqrise.py and owned codebook implementation as needed
- dexmani_policy/deployment/runtime.py
- dexmani_policy/deployment/__init__.py
- dexmani_policy/smoke_test.py
- README.md if current deployment commands are public there
- AGENTS.md Deployment Boundary

Delete when unused:

- dexmani_policy/deployment/export.py
- dexmani_policy/deployment/contract.py
- dexmani_policy/deployment/restore.py
- obsolete deployment export scripts

Do not edit frozen docs/ merely to narrate implementation history unless the repository rules clearly require it.

### dexmani_real

Primary:

- examples/run_policy.py
- dexmani_real/deployment/config.py
- dexmani_real/deployment/session.py
- dexmani_real/deployment/runner.py
- dexmani_real/deployment/observation.py
- dexmani_real/deployment/action.py or planner reset ownership if needed
- dexmani_real/planning/kinematics/ik.py
- dexmani_real/config/pointcloud.py
- README.md if run_policy public usage changes
- AGENTS.md if standing deployment instructions become stale

Keep Raw/Zarr schemas unchanged unless a change is strictly necessary for this task. Do not put software provenance into Raw.

## 26. Do not broaden scope

Do not:

- redesign Raw v34;
- redesign Policy Zarr v15;
- add runtime timestamps to training Zarr;
- change latest-causal observation semantics;
- introduce cross-modal interpolation/synchronization;
- add async/RTC inference;
- alter action representation;
- add SAT-style action spaces;
- add ACK ledgers;
- add generic delta/slew filtering;
- add hard watchdog infrastructure;
- redesign HOME path planning;
- add environment collision checking to normal streaming;
- alter task-success judging;
- introduce plugin/framework registries;
- introduce a tests/ directory;
- preserve old checkpoint compatibility.

Do not silently alter trained numerical behavior while simplifying infrastructure.

## 27. Old 25-review-findings closure target

After implementation, the old review findings should have the following status.

1. DEP-001 hand_enabled=false deployment — fixed by Real preflight.
2. DEP-002 latched B start request — fixed by one-shot B consumption.
3. DEP-003 B does not check hand pose — fixed by arm+hand admission.
4. DEP-004 first full Policy input checked only after RUNNING — fixed before recorder/RUNNING.
5. DEP-005 max_running_s not a hard watchdog — accepted/documented design, not a defect.
6. DEP-006 artifact content identity/hash — eliminated because deployment artifacts are removed.
7. DEP-007 duplicate PolicySpec authority — eliminated because persisted PolicySpec is removed.
8. DEP-008 EEF IK RNG not reset — fixed with planner/IK reset_episode.
9. DEP-009 synchronous chunk latency enters physical timeline — accepted baseline; metrics make it visible.
10. DEP-010 START check/recorder/RUNNING TOCTOU — fixed by fresh post-recorder recheck.
11. DEP-011 training start pose not checkpoint-owned — accepted; current Real home is Evaluation Protocol, not model ABI.
12. DEP-012 Policy runtime revision provenance missing — solved by best-effort two-repo run_config provenance.
13. DEP-013 session created before pure preflight / final status — preflight ordering fixed; do not add a heavy session-result framework.
14. DEP-014 no runtime task-success layer — accepted; success is offline.
15. DEP-015 stats are session-wide — fixed by episode-local stats + rollout_metrics.jsonl.
16. DEP-016 training semantics richer than deployment PolicySpec — eliminated with no second semantic projection.
17. DEP-017 executor clips/IK intervention not attributable — solved by episode metrics.
18. DEP-018 no stable deployment regression path — solved by a small direct checkpoint load/predict smoke.
19. DEP-019 artifact is experiment-addressed rather than standalone — eliminated; checkpoint is the artifact.
20. DEP-020 no cross-modal skew gate — accepted under control_step_latest_causal semantics; no new gate in this task.
21. DEP-021 point-cloud implementation ABI absent — eliminated as a requirement; numerical recipe is frozen, implementation ID is not a gate.
22. DEP-022 agent behavior ABI absent — eliminated as a requirement; saved Agent config + strict state_dict are the boundary.
23. DEP-023 producer.source_commit ambiguous — eliminated with deployment artifact producer metadata.
24. DEP-024 artifact publish lacks fsync crash durability — eliminated with artifact publication subsystem.
25. DEP-025 HOME planner constructed after worker readiness — accepted low-value fail-earlier issue; do not add architecture for it.

Do not finish the task with any of 1, 2, 3, 4, 8, 10, 15, or 18 still open.

## 28. Smoke and offline validation

Do not run real hardware.

### 28.1 Policy static/offline checks

At minimum:

    cd ../dexmani_policy
    conda run --no-capture-output -n policy python -m compileall -q dexmani_policy
    conda run --no-capture-output -n policy ruff format --check dexmani_policy
    conda run --no-capture-output -n policy ruff check --select F401,F821,F822,F823,I dexmani_policy
    git diff --check

Run config-only smoke for affected configs.

Run targeted full smoke for representative affected policies, especially:

- one RGB policy;
- R3D/Uni3D path;
- DQ-RISE path;
- at least one joint-action policy;
- at least one EEF-action policy if an existing config supports it.

Use actual existing config names discovered from the repository; do not invent a config merely for this validation.

The root smoke should cover the new direct checkpoint path:

    train/build minimal model
    -> save new checkpoint
    -> safe inspect with weights_only=True/meta
    -> direct load_policy
    -> strict restore
    -> synthetic predict
    -> expected physical action shape and finite values

Also verify raw and EMA loading when EMA is present.

Verify that an intentionally corrupted state_dict key fails strict restore.

Do not add a tests/ tree.

### 28.2 GPU/CUDA validation

On the real development environment, the Policy environment is expected to have working CUDA.

A basic diagnostic is acceptable:

    conda run --no-capture-output -n policy python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

If a sandbox reports false/unavailable, report the limitation. Do not modify core behavior to make the sandbox pass.

Do not start full training or long evaluation merely to prove CUDA.

### 28.3 Real static/offline checks

At minimum:

    cd <dexmani_real>
    conda run --no-capture-output -n real_robot python -m compileall -q dexmani_real examples
    conda run --no-capture-output -n real_robot ruff format --check dexmani_real examples
    conda run --no-capture-output -n real_robot ruff check --select F401,F821,F822,F823,I dexmani_real examples
    git diff --check

Add focused temporary/offline checks for pure logic when helpful:

- PolicyInfo compatibility preflight;
- start-request one-shot semantics;
- initial-observation admission helper;
- hand home tolerance helper;
- PointCloudConfig partial/additive dict parsing;
- IK reset reproducibility.

Do not import/construct code paths that connect hardware as part of these checks.

## 29. Cross-repository integration acceptance

Before finishing, trace the actual end-to-end code path from:

    examples/run_policy.py
        -> checkpoint selection
        -> inspect_checkpoint
        -> Real compatibility preflight
        -> RuntimeChannels sizing
        -> policy process
        -> load_policy
        -> warmup
        -> PolicyRunner
        -> build_policy_observation
        -> LoadedPolicy.predict
        -> decode/projection/IK
        -> publish_command
        -> arm/hand worker run_id fence

Confirm that:

- no deployment artifact is read or produced;
- no current experiment config is used to redefine a checkpoint's model behavior;
- Real still sizes point-cloud SHM before spawning workers;
- model/CUDA stays in the policy worker;
- hardware SDKs remain in their existing workers;
- S/Q/ESC/run_id fences still invalidate stale proposals after blocking inference;
- recorder transaction semantics are unchanged;
- Raw recording semantics are unchanged.

## 30. Efficiency and code-quality requirements

Do not replace deleted complexity with differently named complexity.

A good final diff should visibly delete more deployment concepts than it adds.

Prefer:

- plain functions;
- ordinary dicts;
- one small runtime dataclass where it has real value;
- Agent-owned logic for Agent-specific behavior;
- direct local control flow.

Avoid:

- manager/service/factory registries;
- version registries;
- compatibility matrices;
- generic transaction objects;
- broad exception wrapping;
- duplicate validation of internal values already produced canonically by the same codebase.

Validate external inputs at the boundary that owns them.

Comments should explain robotics/safety/numerical reasons, not narrate refactor history.

## 31. Documentation cleanup

Because this task changes a repository-level workflow and public cross-repository interface:

- update dexmani_policy/AGENTS.md Deployment Boundary to the new direct-checkpoint design;
- update dexmani_real standing instructions only where they become factually stale;
- update README command examples that reference deployment export/artifacts or --artifact;
- remove obsolete deployment export instructions/scripts.

Do not turn AGENTS.md into an implementation inventory. Keep only stable rules.

Do not edit frozen background docs unless necessary to prevent an actively referenced public workflow from being wrong.

## 32. Final delivery report

When implementation is complete, report:

1. final HEAD/diff summary for both repositories;
2. key deleted mechanisms/files;
3. new checkpoint structure and inference path;
4. Real START/IK/metrics fixes;
5. commands actually executed and PASS/FAIL/NOT VERIFIED;
6. any environment limitation without changing code to hide it;
7. a compact table for DEP-001 through DEP-025 showing:
   - FIXED,
   - ELIMINATED BY DESIGN,
   - ACCEPTED DESIGN / NOT A DEFECT,
   - or any remaining issue.

Do not claim hardware validation unless real hardware was explicitly authorized and used. Offline validation is not hardware validation.

## 33. Definition of done

The task is done only when all of the following are true:

- new training creates one dexmani.checkpoint.v1 checkpoint usable for resume and inference;
- old simple.v3/deployment.v3 compatibility is intentionally absent;
- checkpoint safe inspection works without constructing the model;
- simulation/offline and Real inference share one restore implementation;
- inference uses strict state_dict restoration but not training resume-contract equality;
- external Uni3D/DQ-RISE initialization assets are not required to restore a complete checkpoint for inference;
- deployment export/artifact/contract/restore subsystem is removed;
- no deployment_latest.pt workflow remains;
- Real receives only a tiny runtime PolicyInfo, not a persisted semantic schema;
- Real compatibility validation is reduced to actual runtime capability checks;
- point-cloud numerical config remains frozen without algorithm/version ABI gates;
- B is one-shot and start admission validates arm, hand, and full first Policy observation;
- recorder prepare is followed by fresh revalidation before RUNNING;
- EEF planner/IK episode RNG resets;
- per-episode rollout metrics are persisted simply;
- run_config records both repo revisions only as provenance;
- existing run_id/safety/worker/recording architecture remains intact;
- no tests/ directory or production-style replacement framework is introduced;
- focused smoke/static checks pass, or any environment-limited check is explicitly reported as NOT VERIFIED.
