# Online IK final refactor task

## Baseline

Implement this task against the current `main` baseline reviewed at:

- repository: `haoyangzhanglab/dexmani_real`
- reviewed HEAD: `1d3a96d1f64a8114941b1f9ac83020db7d171183`
- the IK/deployment implementation is effectively the code introduced through `f681fdc6a6d9ff2e1cc6e8dee5cd203013a8da41`; the final `1d3a96d` commit only simplifies standing documentation.

Read and follow `AGENTS.md` first. Before editing, run `git status --short` and preserve unrelated work.

This task is an **offline code refactor**. Do not connect to xArm/XHand/RealSense, do not run teleoperation, homing, replay, rollout, calibration, diagnostics, or any hardware-affecting example.

---

## 1. Goal

Refactor the current xArm7 online Cartesian IK into one small, deterministic mechanism shared by:

- VR teleoperation;
- keyboard/calibration Cartesian jog;
- learned-policy `action_mode="eef"`.

The mechanism must optimize the quantity that actually matters:

```text
high-quality executable IK success
= Cartesian accuracy
+ branch continuity
+ operational-limit validity
+ executable angle representation
+ arm/hand endpoint self-collision validity
+ bounded online latency
```

The intended solver architecture is:

```text
continuation-first exact IK
+ bounded null-space redundancy search
+ weak manipulability-aware candidate selection
```

Keep MPlib/Pinocchio `compute_IK_CLIK` as the primary geometric solver.

Do **not** replace it with a generic planner IK call, TRAC-IK, a new optimizer, full online path planning, or many random restarts.

---

## 2. Current code findings that this task must address

The following findings were verified on the baseline above.

### 2.1 Good behavior to preserve

The repository already has several correct decisions:

- online Cartesian IK uses low-level MPlib/Pinocchio CLIK;
- `current_qpos` and `previous_qpos_cmd` are distinct inputs;
- previous-command seeding and fast accept already exist;
- equivalent-angle canonicalization, J4 elbow protection, global branch protection, hardware-band checks, and endpoint self-collision checks already exist;
- collision uses the final prepared XHand target;
- `make_online_ik_config()` already centralizes the teleop/policy IK profile;
- policy EEF failure already clears the unexecuted action-chunk suffix and re-observes/re-infers;
- teleop now treats isolated control failures as recoverable and only pauses after repeated failures; do not undo this;
- dataset admission remains whole-episode all-or-nothing for non-OK control rows.

Preserve these semantics unless this task explicitly changes them.

### 2.2 P0 correctness gap: validated q can differ from published q

Online IK currently validates candidates using the MPlib/URDF limits, while Cartesian callers subsequently run:

```python
project_arm_command(..., runtime.arm.joint_limit_lower, runtime.arm.joint_limit_upper)
```

Runtime arm limits may legally be narrower than the URDF/mechanical limits.

Therefore the current pipeline can produce:

```text
q_validated_by_IK != q_published
```

after clipping/canonicalization.

For Cartesian IK the required invariant is:

```text
q_validated == q_published
```

Operational runtime limits must therefore be part of online IK itself.

### 2.3 P0 false-failure source: post-IK null-space refinement

The current solver first finds and validates a candidate, then `_command_from_target_qpos()` applies:

```text
q <- q + N * joint_limit_gradient
```

and rechecks pose/collision.

This secondary first-order modification can turn a valid primary IK solution into a failed IK result.

Remove this command-level refinement entirely.

Null-space must be retained only as a **seed-generation mechanism** followed by CLIK correction.

### 2.4 Hot-path work is unnecessarily expensive

The current solver:

- computes measured-posture manipulability before it knows fallback is needed;
- computes FK + Jacobian + Yoshikawa manipulability for every candidate;
- checks collision before fallback candidates are ranked;
- can compute another Jacobian/FK/collision during post-IK null-space refinement.

A normal continuation frame should not pay these costs.

### 2.5 Random fallback does not exploit xArm7 redundancy

Current fallback is:

```text
prev_cmd
current_qpos
random ±5 deg x 3
```

For a regular rank-6 xArm7 Cartesian Jacobian, the local null space is one-dimensional. Search that redundancy structure instead of primarily perturbing all seven joints randomly.

### 2.6 Manipulability exists, but its current use is not ideal

Current scoring uses raw Yoshikawa:

```text
sqrt(det(J J^T))
```

for all candidates and normalizes it against the measured posture.

This both adds hot-path Jacobian cost and mixes translational/angular Jacobian rows without an explicit characteristic length.

Keep a manipulability mechanism, but use it only in fallback candidate selection and make the spatial scaling explicit.

### 2.7 Failure diagnostics are string-parsed and one label is too strong

Attempts are currently strings such as:

```text
prev_cmd:mplib_failed(2.3ms)
```

and later reparsed.

Also, all-seed CLIK non-convergence is classified as `unreachable`, which is not justified by bounded local numerical search.

Use structured attempt records and call this `solver_no_convergence` / no solution found, not unreachable.

---

## 3. Final design

The final online IK flow must be:

```text
TARGET EEF
   |
   v
Stage A: continuation
   prev_cmd -> CLIK
   -> cheap joint-space checks
   -> FK/pose check
   -> if local, accurate, not near operational limit:
          collision check
          if free: FAST RETURN
   |
   v
Stage B: physical-state recovery
   current_qpos -> CLIK
   |
   v
collect current-target, continuity-valid kinematic candidates
   |
   v
choose best current-target redundancy anchor
   |
   v
Stage C: structured null-space search
   scaled Jacobian(anchor)
   -> regular rank-6? obtain 1-D null direction n
   -> operationally bounded +n seed
   -> operationally bounded -n seed
   -> each seed goes through CLIK correction to the SAME Cartesian target
   |
   v
fallback exact-candidate pool
   -> base quality + weak scaled singularity-margin score
   -> sort
   -> collision check in score order
   -> first collision-free candidate wins
   |
   v
Stage D: optional final basin escape
   policy EEF only by default:
   one deterministic bounded random seed
   -> CLIK -> full validation -> collision
   |
   v
SUCCESS or structured FAILURE
```

The final command must always be an actual CLIK result that passed the complete execution contract.

Never directly publish `q + alpha*n` or `q + N*grad`.

---

## 4. Solver invariants

### 4.1 Dual anchors

Keep:

```text
current_qpos         = fresh physical encoder anchor
previous_qpos_cmd    = last successfully published arm target
```

Use `current_qpos` for:

- equivalent-angle representation;
- physical distance;
- hardware representation.

Use `previous_qpos_cmd` for:

- first warm start;
- branch/jump continuity.

Never update previous-command state after a failed solve or failed publication.

### 4.2 Hard constraints vs soft quality

Hard candidate validity remains:

1. finite numeric output;
2. canonicalizable into operational limits;
3. operational joint limits;
4. per-joint jump;
5. J4 elbow branch;
6. global branch distance;
7. hardware-band / hardware-distance validity;
8. Cartesian pose tolerance;
9. endpoint self-collision before publication.

Soft candidate quality may use:

- distance to measured q;
- distance to previous command;
- operational joint-limit penalty;
- pose accuracy inside tolerance;
- weak singularity/manipulability margin.

Never trade a hard constraint for a better manipulability score.

### 4.3 No hidden Cartesian assistance

For policy EEF actions do not add:

- EMA;
- orientation relaxation;
- midpoint targets;
- partial Cartesian execution;
- hidden interpolation.

The policy target remains the policy target, except for the already-explicit workspace position clip and hand projection.

---

## 5. Operational limits become part of OnlineIK

### Required change

Extend the online IK profile/solver so it resolves one `operational_limits` array with shape `(7, 2)`.

Runtime-backed callers must pass:

```python
runtime.arm.joint_limit_lower
runtime.arm.joint_limit_upper
```

These values are already validated to lie inside mechanical limits.

If an `OnlineIKConfig` is created without runtime operational limits, it may fall back to the model/URDF limits for internal planner-only use, but all real runtime Cartesian callers must use the resolved runtime limits.

Use operational limits for:

- canonicalizing equivalent revolutions;
- candidate hard-limit rejection;
- joint-limit penalty;
- null-seed bounds;
- bounded random fallback.

Do not change `IKGeometry.joint_limits` semantics globally; it remains the model/URDF kinematic authority.

### Caller consequence

After successful Cartesian IK, do **not** call `project_arm_command()` in order to change the solution.

For Cartesian callers:

```text
IK qpos -> RobotCommand
```

must preserve the exact validated q.

`project_arm_command()` remains appropriate for direct joint targets such as joint-action policy and replay.

In `PolicyRunner`, branch explicitly:

- joint mode: keep arm projection;
- EEF mode: use the validated IK q directly.

Do the same removal of post-IK arm projection in VR teleop, keyboard Cartesian jog, and camera Cartesian motion after they use the runtime operational limits inside IK.

---

## 6. Simplify OnlineIKConfig

Replace the command-level null-space/random configuration with search-level configuration.

Target shape:

```python
@dataclass(kw_only=True)
class OnlineIKConfig:
    max_ik_jump_deg: tuple[float, ...] = (...)
    max_pose_error_pos_m: float = ...
    max_pose_error_rot_rad: float = ...

    fast_accept_max_delta_deg: float = 8.0

    operational_joint_lower_rad: tuple[float, ...] | None = None
    operational_joint_upper_rad: tuple[float, ...] | None = None

    redundancy_seed_step_deg: float = 5.0
    joint_limit_search_margin_deg: float = 15.0

    enable_random_fallback: bool = False
    random_fallback_step_deg: float = 5.0
    random_seed: int | None = 42

    previous_command_distance_weight: float = 0.25
    joint_limit_penalty_weight: float = 0.01
    pose_accuracy_weight: float = 0.1
    pose_rotation_weight: float = 0.5

    singularity_margin_weight: float = 0.02
    jacobian_characteristic_length_m: float = 0.4

    joint_weights: tuple[float, ...] = (...)
    previous_command_joint_weights: tuple[float, ...] | None = (...)
```

Names can be adjusted for consistency, but semantics must stay clear.

Remove obsolete fields/concepts:

- `position_ik_num_random_seeds`;
- command-oriented `position_ik_seed_offset_deg` if replaced by the two explicit search radii;
- raw Yoshikawa hard threshold;
- `enable_nullspace_optimization`;
- `nullspace_step_size_deg`;
- `runtime.policy.ik_nullspace_step_rate_deg_s`;
- command-level `nullspace_joint_limit_margin_deg` naming.

Do not add a large set of new tunables. Keep defaults local to `OnlineIKConfig` unless a real experiment needs YAML ownership.

---

## 7. Simplify make_online_ik_config()

The existing helper is good and must remain the central runtime constructor, but its current `control_dt_s` input only exists to convert post-IK null-space command rate.

After removing command refinement, eliminate that time-rate coupling.

Prefer:

```python
def make_online_ik_config(
    runtime,
    *,
    max_pose_error_pos_m: float | None = None,
    max_pose_error_rot_rad: float | None = None,
    enable_random_fallback: bool = False,
) -> OnlineIKConfig:
    ...
```

It must always populate runtime operational arm limits.

Use it consistently:

- VR teleop: policy pose tolerances, random fallback off;
- policy EEF action planner: policy pose tolerances, random fallback on;
- keyboard: keyboard precision tolerances, random fallback off;
- camera Cartesian motion: keyboard/calibration precision tolerances, random fallback off;
- other runtime-backed online-IK construction should not hand-build a divergent limit profile.

Update `deployment/smoke_test.py` accordingly.

Do not preserve the old `control_dt_s` argument as an unused compatibility branch.

---

## 8. Stage A: real fast path

A normal continuation frame should do:

```text
1 CLIK
+ cheap q checks
+ 1 FK
+ 1 collision query
```

and **zero Jacobian/manipulability calculations**.

### Candidate preparation order

After CLIK returns:

1. finite raw q;
2. canonicalize against `current_qpos` using operational limits;
3. operational-limit check;
4. duplicate check;
5. per-joint jump against previous command;
6. J4 elbow-flip check;
7. global branch L2 check;
8. hardware-band mismatch;
9. hardware-distance limit;
10. FK pose;
11. Cartesian error tolerance.

Only after these pass should collision/manipulability work be considered.

Do not run FK/Jacobian/collision for candidates already rejected by cheap joint-space conditions.

### Fast return

For the `prev_cmd` seed, preserve the current local/accurate preference:

- max wrapped physical joint distance <= approximately 8 deg;
- position and rotation error <= half of configured hard tolerance.

Add one exception:

- if the candidate lies inside the operational joint-limit search margin, retain it as a valid base candidate but enter fallback redundancy search instead of immediately returning.

For a far-from-limit fast candidate:

- check endpoint self-collision once;
- if collision-free, return immediately;
- do not compute Jacobian or manipulability.

If it self-collides, keep it eligible as a redundancy anchor if all non-collision constraints are valid.

---

## 9. Stage B: current-q recovery

If fast return did not happen, try a second CLIK solve from fresh `current_qpos`.

Do not replace continuation priority with measured-state priority.

The order remains:

```text
prev_cmd -> current_qpos
```

Collect non-duplicate candidates that satisfy all non-collision hard constraints.

A candidate may remain in this kinematic pool even if it self-collides; collision can be escaped by redundancy and must not destroy a useful current-target anchor.

A branch-invalid candidate must never be a redundancy anchor.

---

## 10. Stage C: null-space as a solution generator

### 10.1 Choose an anchor on the current target manifold

Choose the best candidate from the current target's non-collision-valid kinematic pool using base quality/continuity.

The anchor must already satisfy:

- operational limits;
- branch continuity;
- hardware representation;
- Cartesian pose tolerance.

It may be:

- near an operational limit;
- self-colliding with the prepared hand;
- otherwise suboptimal.

Do not use an elbow-flipped/branch-jumped solution as the anchor.

### 10.2 Scaled spatial Jacobian

At the anchor, compute the world Jacobian:

```text
J = [Jv; Jw]
```

Construct:

```text
J_scaled = [Jv / L; Jw]
```

with `L = jacobian_characteristic_length_m` (default 0.4 m).

Use this same scaled Jacobian SVD for:

- rank diagnosis;
- 1-D null direction;
- anchor singularity margin.

Row scaling is nonsingular and does not change the mathematical null space, while making translational/angular scaling explicit.

### 10.3 Rank handling

Use a documented relative SVD tolerance.

For regular xArm7 Cartesian IK:

```text
rank(J_scaled) == 6
nullity == 1
```

Only in that regular case perform the structured 1-D null search.

If rank < 6:

- record the rank/singular values in diagnostics;
- do not pretend there is a unique one-dimensional redundancy direction;
- skip the structured `null +/-` search for that anchor;
- allow the bounded final fallback if enabled.

Do not add an adaptive DLS implementation in this task.

### 10.4 Predictor -> corrector

With SVD `J_scaled = U S Vh`, use:

```python
n = Vh[-1]
```

(sign is irrelevant because both directions are considered).

Construct two predictor seeds:

```text
q_seed_plus  = q_anchor + alpha_plus  * n
q_seed_minus = q_anchor - alpha_minus * n
```

where each alpha is bounded so every seed is already inside operational limits.

The desired scale is approximately:

```text
max(abs(delta_q)) ~= redundancy_seed_step_deg
```

(default 5 deg).

If a direction has effectively no room or creates a duplicate seed, skip it; do not clip an out-of-range seed after construction.

Order the two seeds by the cheap operational joint-limit quality of the predictor point, better margin first.

For each seed:

```text
predict along null space
-> run CLIK against the original target
-> canonicalize/validate the corrected exact candidate
```

The predictor itself is never executable output.

---

## 11. Manipulability / singularity-margin mechanism

Keep manipulability as a weak fallback quality mechanism, but do not use the current raw-Yoshikawa hot-path behavior.

### 11.1 Metric

For each fallback candidate that participates in an actual comparison, compute:

```text
sigma_min(J_scaled)
```

where:

```text
J_scaled = [Jv / L; Jw]
```

Use the smallest singular value as the local Cartesian singularity margin.

Do not use it as a hard rejection threshold.

### 11.2 Lazy evaluation

Do not compute singularity margin on the normal fast path.

Only compute it when:

- fallback has produced multiple exact candidates that need ranking; or
- the Jacobian was already required for null-space search and its value can be reused.

If there is only one viable fallback candidate, manipulability cannot change selection and should not force extra Jacobian work.

### 11.3 Candidate-pool normalization

Avoid normalizing against measured-posture Yoshikawa.

For a fallback comparison pool, normalize:

```text
manip_quality_i = sigma_min_i / max_j(sigma_min_j, eps)
```

so the term lies approximately in `[0, 1]`.

Then use it only as a weak tie-breaker.

### 11.4 Score

Preserve the current basic scoring intent:

```text
score =
    distance_to_current
  + previous_command_weight * distance_to_previous_command
  + joint_limit_weight * operational_joint_limit_penalty
  + pose_weight * normalized_pose_cost
  - singularity_margin_weight * manip_quality
```

Start from the current relative weights where possible:

- previous-command distance: 0.25;
- joint-limit penalty: 0.01;
- pose accuracy: 0.1;
- singularity margin: 0.02.

Do not simultaneously retune all weights during this refactor.

Continuity must dominate modest manipulability improvements.

Do not implement `N * grad(manipulability)` or any post-IK manipulability gradient ascent.

---

## 12. Collision ordering

Collision remains a hard execution constraint.

### Fast path

A candidate that would be returned immediately must be collision checked before return.

### Fallback

Do not collision-check every fallback candidate before scoring.

Instead:

1. collect exact candidates that pass all non-collision hard constraints;
2. compute fallback quality and sort them;
3. check endpoint self-collision in score order;
4. return the first collision-free candidate.

A self-colliding exact candidate can still serve as a null-space anchor before final selection.

This specifically allows xArm7 redundancy to escape an arm/hand endpoint collision while preserving the same EEF target.

Do not add swept/current-to-target collision checking or environment/table collision to normal online IK.

---

## 13. Stage D: one final numerical basin fallback

Delete the current `random x 3` primary fallback.

Keep at most one deterministic bounded random seed as a last numerical-basin escape.

Default:

- VR teleop: off;
- policy EEF: on;
- keyboard/calibration: off.

Try it only after structured continuation/current/null-space search has failed to produce an executable candidate.

Sample directly inside:

```text
operational_limits intersect [reference - delta, reference + delta]
```

Do not sample then clip.

Use a fixed RNG seed for reproducibility.

If a current-target redundancy anchor exists, use it as the random reference; otherwise use the previous command/current state as appropriate.

---

## 14. Remove command-level null-space code

After the structured-search implementation is complete, remove obsolete command-refinement code:

- `_command_from_target_qpos()` behavior that applies null-space repulsion;
- `apply_nullspace_optimization()` if no other caller remains;
- `nullspace_projector()` if no other caller remains;
- time-based null-space warning state;
- runtime `ik_nullspace_step_rate_deg_s`;
- any smoke checks tied to per-control-step null-space command rate.

The final selected candidate has already been validated. Build the success `IKResult` directly from that exact selected candidate and its cached pose/quality data.

Do not re-run FK/collision after selection unless some code actually changes q after validation; this task should ensure nothing does.

---

## 15. Rename the API to match its real scope

Rename:

```python
solve_teleop_ik(...)
```

to:

```python
solve_online_ik(...)
```

Update all current callers:

- VR teleop;
- deployment EEF action decoder;
- keyboard Cartesian jog;
- camera Cartesian motion;
- any smoke code.

Do not keep a stale compatibility alias unless a real external public dependency is demonstrated. Git history preserves the old name.

Update class/docstrings that still say this is teleop-only.

---

## 16. Structured failure reporting

### 16.1 Failure kinds

Use top-level semantics equivalent to:

```python
NO_SOLUTION_FOUND
NO_VALID_CANDIDATE
INVALID_OUTPUT
```

Meaning:

- `NO_SOLUTION_FOUND`: bounded numerical CLIK/search did not produce a usable kinematic solution;
- `NO_VALID_CANDIDATE`: one or more numerical solutions existed, but all were rejected by the online execution contract;
- `INVALID_OUTPUT`: NaN/malformed/model/numerical-contract error.

Do not use `unreachable` for all-seed CLIK non-convergence.

Only `INVALID_OUTPUT` is a technical/system failure category. The other two remain recoverable command rejections.

### 16.2 Structured attempts

Replace string attempt parsing with small dictionaries, for example:

```python
{
    "seed": "prev",
    "result": "mplib_failed",
    "solve_ms": 2.3,
}
```

or:

```python
{
    "seed": "null_preferred",
    "result": "self_collision",
    "solve_ms": 3.1,
    "pos_err_m": ...,
    "rot_err_rad": ...,
    "score": ...,
    "singularity_margin": ...,
}
```

No `seed:tag(...)` string -> parser loop.

### 16.3 Candidate funnel

Include bounded counters in `IKResult.report`, for example:

```text
attempted
clik_converged
operational_limits_valid
continuity_valid
hardware_valid
pose_valid
collision_free
```

Also keep:

- selected seed/mode;
- total IK timing;
- number of CLIK calls;
- number of collision checks;
- selected pose error;
- selected singularity margin when it was computed.

Keep diagnostics small; do not build a telemetry framework.

---

## 17. Caller/workflow semantics

### 17.1 Teleop

Preserve the current latest-main behavior:

```text
IK failure
-> no RobotCommand
-> previous published command stays unchanged
-> accepted EMA target stays unchanged
-> next VR sample can recover
```

Do not send measured q as a synthetic hold command.

Do not reintroduce “single IK failure stops teleop”.

The current repeated-failure debug pause/re-anchor behavior can remain.

Do not redesign recorder lifecycle in this IK task. Failed control rows remain raw diagnostics and whole-episode Zarr admission remains strict.

### 17.2 Policy EEF

Preserve:

```text
IK failure
-> publish nothing
-> clear remaining action chunk
-> wait one policy control period
-> fresh observation
-> fresh inference
```

Do not add policy-action EMA or target relaxation.

Policy EEF gets the one final deterministic random fallback because a decoder false-negative costs an entire chunk/re-inference.

Do not add a new rollout state machine or persistent-failure abort in this task; existing rollout timeout/lifecycle remains the authority. Add only bounded IK diagnostics if useful.

### 17.3 Direct joint policy

Do not route joint actions through Cartesian IK.

Keep `project_arm_command()` for joint-mode arm targets.

---

## 18. Raw frame-status scope

Current VR mapping failure is still emitted as `FRAME_IK_FAIL`.

That is diagnostically imperfect, but changing persisted frame-status enum semantics is a separate data-schema decision.

For this IK refactor:

- do not silently reinterpret existing raw status values;
- do not bump the raw schema merely to clean this label;
- do not use raw `FRAME_IK_FAIL` as the authoritative detailed IK taxonomy.

Use the structured `IKResult` report for actual solver diagnosis.

A separate schema task can add a mapping-specific persisted status if desired.

---

## 19. XHand model-order safety invariant

The online collision filter depends on the fixed SDK -> URDF hand-joint remap.

Make the model order assumption explicit at model construction with minimal startup checks.

Expose the expected URDF hand-joint sequence from `robot/model.py` instead of keeping it only as an unverified private tuple, then verify when constructing the full Pinocchio collision model:

```text
7 arm active joints + expected 12 hand URDF joints
```

and the expected nq.

Keep the check direct; do not build a general model-schema framework.

If the standalone hand kinematics path already has an equivalent exact-name check, reuse the same constants; otherwise add the same small invariant there.

This is an offline/model correctness check, not a hardware safety protocol.

---

## 20. Intentional real-hand mount offset

Do **not** change the URDF mount or runtime transform as part of this task.

The real custom adapter is intentionally 10 mm thinner than the nominal URDF/CAD mount:

```text
nominal URDF mount:  x = -0.005 m
real observation/FK: x = -0.015 m
```

Add a concise comment above `HandParams.T_eef_handbase_pos_xyz` explaining this physical modification so future reviews do not misclassify it as configuration drift.

No other geometry change is requested.

---

## 21. Files expected to change

Keep the diff focused. Expected paths include:

- `dexmani_real/planning/kinematics/ik.py`
- `dexmani_real/planning/kinematics/ik_geometry.py` only if a small helper is genuinely needed
- `dexmani_real/planning/planner.py`
- `dexmani_real/planning/collision.py`
- `dexmani_real/robot/model.py`
- `dexmani_real/config/defaults.py`
- `dexmani_real/deployment/action.py`
- `dexmani_real/deployment/runner.py`
- `dexmani_real/teleop/control_loop/grid.py`
- `dexmani_real/teleop/loop.py` only if constructor/API updates require it
- `dexmani_real/teleop/keyboard_session.py`
- `dexmani_real/calibration/camera/session.py`
- `dexmani_real/calibration/camera/motion.py`
- `dexmani_real/robot/arm_homing.py` only if its online-IK profile construction requires the central factory
- `dexmani_real/deployment/smoke_test.py`

Do not touch unrelated recording, point-cloud, dataset, or runtime lifecycle code.

---

## 22. Implementation shape

Keep `ik.py` direct. A small private candidate dataclass is appropriate, for example:

```python
@dataclass
class _Candidate:
    qpos: np.ndarray
    seed_name: str
    pos_err_m: float
    rot_err_rad: float
    distance_current: float
    distance_previous: float
    limit_penalty: float
    base_score: float
    singularity_margin: float | None = None
    score: float | None = None
```

A small set of private helpers is enough:

```text
solve()
_try_clik()
_prepare_candidate()
_check_joint_contract()
_choose_redundancy_anchor()
_scaled_jacobian_svd()
_make_null_seeds()
_score_fallback_candidates()
_select_collision_free()
_build_failure_report()
```

Do not introduce `IKStrategy`, `CandidateManager`, `ValidationRegistry`, generic pipeline classes, plugin-style solvers, or other abstraction layers.

---

## 23. Offline verification

The repository intentionally does not use a committed tests directory.

Use focused offline checks only.

### 23.1 Existing low-cost checks

Run:

```bash
python -m compileall -q dexmani_real examples
ruff format --check dexmani_real examples
ruff check --select F401,F821,F822,F823,I dexmani_real examples
git diff --check
```

If Ruff is unavailable, report that; do not install/upgrade the environment.

Run the existing deployment smoke module if its imports are safe in the available environment:

```bash
python -m dexmani_real.deployment.smoke_test
```

Inspect it first; do not let an offline check connect hardware.

### 23.2 Focused IK smoke checks

Use one-off offline Python checks/mocks to verify:

1. **Fast path cost contract**
   - prev seed success;
   - no Jacobian/manipulability call;
   - one collision query before return;
   - no random/null fallback.

2. **Operational-limit invariant**
   - construct a runtime profile narrower than model limits;
   - verify an IK result outside the operational limits is rejected or represented by an equivalent in-range solution;
   - verify Cartesian caller does not modify a successful q afterward.

3. **Null predictor-corrector**
   - create a valid current-target anchor;
   - confirm generated +/- seeds stay inside operational limits;
   - confirm final returned null candidate is a fresh CLIK solution, not the predictor q.

4. **Rank handling**
   - regular 6x7 Jacobian produces one null direction;
   - mocked rank-deficient Jacobian skips the 1-D structured null search without numerical failure.

5. **Manipulability ranking**
   - use mocked candidate Jacobians;
   - confirm better singularity margin can break a close soft-quality tie;
   - confirm it cannot override a hard branch/limit/collision failure.

6. **Collision ordering**
   - fallback candidates are scored before collision queries;
   - collision queries stop at the first collision-free candidate.

7. **Failure semantics**
   - all CLIK non-convergence reports no-solution/solver-no-convergence, never unreachable;
   - malformed/non-finite solver output reports `INVALID_OUTPUT`.

8. **Caller behavior**
   - EEF mode publishes exact validated IK q;
   - joint mode still uses arm projection;
   - policy EEF failure still clears chunk and publishes nothing;
   - teleop failure still publishes nothing and does not update `prev_qpos_cmd`.

### 23.3 Known-feasible EEF benchmark

If representative clean raw/Zarr demonstration data is locally available, run a one-off oracle benchmark; do not fabricate data or commit dataset-specific test infrastructure.

For each clean timestep:

```text
known q_gt = recorded arm joint target
known T    = FK(q_gt) = action_ee arm target
known hand = recorded hand target
```

Run online IK from the recorded measured/previous states and report:

- known-feasible recovery rate;
- prev fast-success fraction;
- current recovery fraction;
- null-space recovery fraction;
- final-random recovery fraction;
- remaining failure tags;
- p50/p95/max IK latency;
- mean CLIK calls;
- mean collision calls;
- selected seed distribution;
- selected singularity-margin distribution when computed.

Compare against the current implementation if practical.

Do not set an invented success-rate threshold without data; report actual before/after numbers and investigate any regression.

---

## 24. Acceptance criteria

The task is complete only when all of the following are true.

### Correctness

- Cartesian IK uses runtime operational arm limits.
- Successful Cartesian IK q is published unchanged.
- No post-IK null-space command modification remains.
- Null-space predictor points are never published.
- Every published EEF-mode q passed pose, continuity, hardware representation, operational-limit, and endpoint self-collision checks.
- Branch/elbow/hardware-band protections remain hard constraints.
- Self-collision candidates may seed redundancy search but can never be published.

### Search quality

- Search order is deterministic:
  `prev -> current -> null preferred/opposite -> optional one random`.
- Null search uses the current target's valid kinematic anchor, not blindly the previous target posture.
- Structured null search is skipped when the Jacobian does not have the expected regular rank-6 structure.
- Manipulability is a weak fallback ranking term, not a hard threshold or direct gradient controller.

### Performance

- Normal fast path performs no Jacobian/manipulability calculation.
- Random x3 is gone.
- Fallback collision checks occur in score order and stop on the first free candidate.
- No duplicate final FK/collision pass exists after an unchanged/selected solution.
- The implementation remains bounded and deterministic.

### Workflow

- Existing recoverable teleop behavior remains intact.
- Existing policy EEF `clear chunk -> reobserve -> reinfer` behavior remains intact.
- No hidden EEF smoothing/relaxation is introduced.
- Direct joint policy behavior remains intact.
- Recorder/dataset lifecycle is not redesigned by this task.

### Repository discipline

- No hardware code was executed for validation.
- No committed `tests/` directory was introduced.
- No unrelated refactor is mixed in.
- Dead old config/functions/imports are removed.
- Comments describe current robotics/math intent, not implementation history.
- Final diff passes available offline checks.

---

## 25. Handoff report

At completion, report concisely:

1. files changed;
2. old IK path vs new IK path;
3. config fields removed/added;
4. operational-limit invariant implementation;
5. null-space search implementation and rank handling;
6. manipulability metric/scaling and where it is evaluated;
7. fast-path and fallback call counts from offline checks;
8. any oracle benchmark before/after results available;
9. commands/checks run and their results;
10. anything not hardware-validated.

Do not claim hardware validation unless separately and explicitly authorized.

---

## 26. Final design summary

The intended end state is:

```text
normal frame:
    prev warm start
    -> CLIK
    -> cheap checks
    -> FK pose
    -> endpoint collision
    -> publish exact validated q

hard frame:
    prev/current exact candidates
    -> choose current-target redundancy anchor
    -> scaled-Jacobian SVD
    -> bounded null +/- predictor seeds
    -> CLIK correction
    -> weak singularity-margin-aware scoring
    -> collision in score order
    -> optional one deterministic basin fallback
    -> publish exact validated q

failure:
    reject this command
    preserve successful trajectory state
    let the owning workflow recover
```

The key rules are:

```text
CLIK guarantees Cartesian correctness.
Null space generates alternative exact-solution basins.
Manipulability helps rank legal redundancy postures.
Hard execution constraints always dominate secondary quality.
Validated q must equal published q.
```
