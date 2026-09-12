# dexmani_real Clean-Code / Repository Hygiene 最终执行指南

> Status: **CURRENT EXECUTION GUIDE**  
> Repository: `haoyangzhanglab/dexmani_real`  
> Reviewed branch: `main`  
> Reviewed SHA: `f4d82d05bcb5ec4ce46cd01c3e50c656d9d4d78e`  
> Review date: 2026-09-12  
> Scope: **post-refactor cleanup only; no architecture redesign, no hardware-policy change, no schema migration**

本文档固化 2026-09-12 对当前 `main` 的多轮 GitHub review 结论，用于本地 Claude Code / 人工执行一次低风险、可验证、可回滚的 clean-code 与 repository hygiene cleanup。

它不是新的架构重构计划。它的目标是：

1. 清除已经确认的 stale current-facing documentation / tooling residue；
2. 删除极少量已经经过 caller graph 反证的 dead production symbol；
3. 清理明显重复、过时或只复述代码的 comments/docstrings；
4. **不改变** real-robot safety、timing、IPC、dataset semantics、hardware backend、planner semantics 或 persisted schema；
5. 防止“为了干净而多删”，同时避免遗漏已经影响后续维护/agent 判断的 stale contract。

如果执行时 `main` 已经前移，**源码、schema、配置定义永远优先于本文**。实施者必须重新搜索对应 callsites/tests/contracts，不允许机械套用旧 patch。

---

## 1. Authority 与执行原则

首先遵循仓库已有 authority：

```text
runtime behavior / source code
→ schema / configuration definitions
→ README / focused implementation docs
→ repo_map.md
→ agent guidance
```

同时必须阅读并遵循：

```text
AGENTS.md
CLAUDE.md
code_style.md
```

本 cleanup 的优先级固定为：

```text
Physical Safety
→ Experimental Correctness
→ Temporal / Causal Correctness
→ Data / Schema Integrity
→ Hardware Diagnostic Capability
→ Clear Ownership / Data Flow
→ Repository Cleanliness
→ LOC Reduction
```

因此：

- “没有 repo caller”是删除必要条件之一，但不是充分条件；
- safety path、hardware diagnostic、historical migration、duck-typed test seam 需要额外判断；
- 不以“删掉多少行”为目标；
- 不做 unrelated reformat / rename / relocation；
- 每个删除都必须能明确回答：**谁不再需要它，为什么不会影响 runtime / test / local config / historical data？**

---

## 2. 当前 canonical facts

本 review 基线下，当前 schema 为：

```text
raw episode      = v29
processed HDF5   = v20
Policy Zarr      = v13
```

当前 XHand observation/data path 已经是：

```text
one XHand SDK read
    ↓
one hand sample
    ├── qpos / current
    ├── tactile_aggregate [5,3]
    ├── tactile_dense     [5,120,3]
    ├── tactile_aggregate_valid
    └── tactile_dense_valid
    ↓
one hand_state_ring publication
    ↓
causal alignment selects one hand sample identity per policy slot
    ↓
raw v29
    ↓
processed v20
    ↓
Policy Zarr v13
```

不存在新的独立 tactile provenance ring，也不应重新引入 backward tactile repair。

---

# 3. HARD PROTECTED SCOPE

以下内容在本轮 cleanup 中视为 **protected**。除非用户另外开启专项任务，否则不得修改其行为。

## 3.1 `assets/**` 完全冻结

用户明确要求：

```text
DO NOT DELETE
DO NOT MOVE
DO NOT RENAME
DO NOT REGENERATE
DO NOT REFORMAT
```

包括但不限于：

- visual `.glb/.obj/.mtl`；
- collision meshes；
- xArm/XHand URDF；
- SRDF；
- audio/resource assets。

即使 repo 内没有显式 caller，也不能在本轮处理。

## 3.2 `examples/*.py` 文件全部保留

任何 `examples/` 下代码文件都：

```text
不得删除
不得 rename
不得移动
```

允许：

- 修 stale version/help/docstring；
- 删除 confirmed-unused local import；
- 删除 confirmed-never-called private helper；
- 删除 unreachable local branch；
- 清理只复述代码的 comments。

不允许因为与 production driver 重复就删除 diagnostic capability。尤其：

```text
examples/xhand_control_example.py
examples/realsense_record_example.py
examples/pointcloud_process_example.py
```

独立 hardware diagnostic path 本身不是 dead code。

## 3.3 Persisted schema 不变

本轮禁止：

```text
EPISODE_SCHEMA_VERSION bump
PROCESSED_SCHEMA_VERSION bump
POLICY_ZARR_SCHEMA_VERSION bump
删除/改名 raw dataset field
改变 raw field semantics
改变 processed/Zarr key semantics
```

因此以下字段即使新数据中接近常量，也 **KEEP**：

```text
fill_reason
flag_sample_valid
observation_valid
meta.success
```

只允许修正对它们的文档解释。

## 3.4 Safety / timing / IPC 不变

禁止削弱或重写：

- `SafetyState` / sticky fault；
- `run_generation`；
- command ticket / expiry；
- arm/hand mechanical limits；
- workspace / collision fail-closed；
- worker final SDK fence；
- heartbeat / supervisor；
- observation source/publish/anchor causality；
- freshness / skew / grid alignment；
- action scheduling semantics；
- RuntimeChannels ownership；
- shared-memory dtype/shape；
- recorder transaction / atomic publication。

## 3.5 Hardware backend 不删

本轮保留：

```text
DexPilot
TAG
RealSense D400 / L515 branches
XHand serial / EtherCAT
wrist / eye-in-hand related future-capable path
calibrated table + fallback safety geometry
```

是否退役 backend 必须是独立的 research/hardware decision。

## 3.6 Planner API / collision path 暂不收窄

**不要删除**：

```text
XArm7MotionPlanner.__getattr__
IKCandidateSearch.check_path_collisions()
paths.py collision compatibility fallback
paths.py getattr(... has_table ...)
```

理由：

1. `XArm7MotionPlanner` 当前自身仍通过 `self.compute_eef_pose_world()`、`self.canonicalize_qpos()` 等 delegated API 工作，直接删除 `__getattr__` 会破坏 planner；
2. `check_path_collisions()` 是 self-collision-only API，而 `check_path_combined_collisions()` 语义更宽，并非严格重复；
3. home/path safety 代码收益很低而 regression 风险高；
4. test fake / duck-typed seam 也可能依赖兼容面。

若未来要移除 planner proxy，必须另开专项：

```text
先枚举全部 delegated symbol
→ 将 planner 内部 self.* 显式改为 self.kin / self.ik_mgr / self.mplib_planner
→ 更新所有 callers/tests
→ targeted planner tests
→ 再删除 __getattr__
```

本轮不做。

## 3.7 Historical migration 保留

保留：

```text
tools/convert_raw_v24_to_v25.py
tools/convert_raw_v25_to_v26_tactile.py
对应 migration tests/docs
```

它们是 frozen historical conversion utilities，不是 current runtime，但仍是 data lineage / recovery capability。

## 3.8 Historical test filenames 保留

例如：

```text
tests/test_raw_v26_recording.py
tests/test_v26_producers.py
tests/test_camera_v25_telemetry.py
```

不要为了 current version 数字整洁而 rename。`repo_map.md` / historical review 已明确：测试名中的历史数字不等于 runtime 仍支持该版本。

---

# 4. Fact-checked cleanup inventory

## 4.1 P0 — current-facing documentation / tooling correctness

这部分优先级最高，因为它基本不改变 runtime，却会直接影响用户、未来维护者和 coding agent 的判断。

### P0.1 `README.md`

必须修正 current architecture / schema 描述。

当前 stale 内容包括但不限于：

- raw v28；
- processed v19；
- Policy Zarr v12；
- 独立 `hand_contact_source_monotonic_ns` current-path 描述；
- aggregate contact 独立选择 / provenance 的旧语义。

修订原则：

```text
current workflow → 描述 raw29 / processed20 / Zarr13 和 one-hand-sample semantics
historical salvage → 明确 Historical，不改写其当时真实版本号
```

不要全局机械替换所有 `v28/v19/v12`，因为历史 incident / migration evidence 里的旧版本号可能是正确事实。

### P0.2 `docs/pointcloud_pipeline.md`

current pipeline 部分更新到：

```text
raw v29 → processed HDF5 v20 → Policy Zarr v13
```

历史 sections 若在描述旧 artifact，不机械改数字。

### P0.3 `examples/process_episodes.py`

文件保留，只修 CLI description 中写死的 processed v18。

推荐避免继续在一般 CLI help 中硬编码版本：

```text
"Process complete control-step episodes into the current processed HDF5 schema."
```

如果确实需要显示版本，直接引用 canonical constant，而不是复制数字。

### P0.4 `examples/visualize_episode_processed.py`

文件保留。

修：

- module docstring 的 v18；
- error message 的 v18；
- current-facing version wording。

不要改变 viewer payload admission / rendering logic。

### P0.5 `examples/replay_episode.py`

只修 CLI help 中 `raw schema-v28`。

不要改变 replay safety / command / trajectory behavior。

### P0.6 `examples/visualize_episode.py`

只修 module docstring 的 `raw-v28` current description。

### P0.7 `dexmani_real/dataset/pointcloud.py`

module docstring 从：

```text
Raw-v28 camera metadata ...
```

改成不随 schema bump 漂移的表述，例如：

```text
Current raw-episode camera metadata to canonical xArm-base point-cloud inputs.
```

不改 numeric processing。

### P0.8 `docs/tactile_unit_si_verification_pending.md`

该文档的 scientific question 与 historical evidence 要保留。

只更新 “当前 HEAD” 描述：

- current schema；
- current function names `_parse_tactile_aggregate` / `_parse_tactile_dense`；
- 已删除的 old provenance/freshness fields 不再描述为 current。

历史 v25/v26/v28 evidence 保留原样并明确时间背景。

### P0.9 `docs/policy_eval_finalization_plan.md`

**不要直接 archive。** 当前文件头仍把它定义为 current execution basis。

执行：

1. 保留文件；
2. current implementation snapshot 更新到 raw29 / processed20 / Zarr13；
3. 已完成事项改成 implemented/completed；
4. 不改变其 research priority / safety decisions；
5. 若其中存在真正历史 transition（例如 v18→v19），保留并标为 historical transition。

### P0.10 `docs/policy_eval_refactor_plan.md`

同样不要直接 archive。

建议将定位收敛为：

```text
architecture / rationale document
current execution status follows policy_eval_finalization_plan.md and source
```

避免两个文档同时都宣称“唯一当前执行依据”。

不需要重写整份设计；只解决 governance ambiguity 和 current snapshot drift。

### P0.11 `docs/xhand_tactile_research_simplification_plan.md`

当前文件仍写：

```text
PROPOSED — NOT IMPLEMENTED
```

但它定义的主要目标已经落地：

```text
raw28→29
processed19→20
Zarr12→13
single hand_state sample
aggregate/dense validity split
remove separate tactile ring/backward repair
```

因此：

```text
Status → IMPLEMENTED
```

并注明 source/current schemas authoritative。

不要删除历史设计推导；它仍可作为 rationale/evidence。

### P0.12 `user_design.md`

XHand SDK code `1501035` 的 accepted grasp-contact 语义保留。

删除或修正当前不存在的陈述：

```text
hand worker 累计该事件用于 episode quality metric
```

禁止因此重新引入 counter。

### P0.13 `dexmani_real/robot/hand_worker.py`

保留以下 invariant：

```text
software-bias readiness
× per-read aggregate/dense validity
→ published validity bits
```

删除 production comment 中对历史 implementation plan `§5.4` 的依赖；注释应自洽地解释本地 invariant。

### P0.14 `dexmani_real/planning/kinematics/arm_fk.py`

修正 `ArmFK` class docstring 中 stale `for arm_loop`。

当前事实是 EEF 从 aligned qpos 通过 canonical FK 被 dataset/deployment 等多个 consumer 使用。

不修改 FK 数学、URDF、frame 或 rot6d semantics。

### P0.15 `.claude/settings.local.json`

这是 machine-local file，含：

- `/home/...` absolute path；
- local env command；
- one-off debugging permission；
- 已不存在的 module/script path。

执行：

```text
DELETE .claude/settings.local.json from Git
ADD exact .claude/settings.local.json ignore rule
KEEP .claude/skills/**
```

不要 ignore 整个 `.claude/`。

### P0.16 `.codex/agents/*.toml`

当前多个 agent 配置要求读取不存在的：

```text
docs/policy_eval_workflow.md
```

修正：

- 指向真实 current docs；
- 优先 `AGENTS.md` / `code_style.md` / source；
- 对 policy eval 可指向 `docs/policy_eval_finalization_plan.md` 与相关 focused guide；
- 删除对不存在文件的强依赖。

不要改变 model / reasoning effort，除非该项本身错误。

### P0.17 `.codex/config.toml`

当前共享 project config 含 machine-specific writable root：

```text
/home/zhanghaoyang/Desktop/dexmani_real
```

不要删除整个 `.codex/config.toml`。

目标是让 tracked project config 不依赖个人绝对路径。根据本地 Codex 支持的配置语义，优先：

1. 删除不必要的显式 `writable_roots`；或
2. 使用 project-relative / portable 配置；
3. 如果 Codex 必须使用绝对 root，则把该项移入不跟踪的 local override，而不是猜一个新路径。

**不要在不知道 Codex 配置语义时编造字段。** 如果无法确认 portable 写法，只删除 machine-specific tracked value，并在 handoff 中说明 local configuration requirement。

### P0.18 `.gitignore`

只做最小整理：

- 添加 `.claude/settings.local.json`；
- 删除 exact duplicate（如重复 `.vscode/`、`episodes/`）；
- 保留项目实际使用的 ignored directories，包括 local experiments/data/output；
- 不做大规模“Python template 精简”，避免误伤个人工作流。

### P0.19 `deployment/inference/observation.py` comments/docstrings

这是 comment cleanup 的最高价值文件之一，但 **不得设 LOC reduction 百分比目标**。

必须保留解释以下内容的 comments：

- source ≤ publish ≤ anchor causality；
- run-start boundary；
- warm-up edge repeat；
- newest source ≤ reference selection；
- skew bound；
- same selected hand index 投影到 qpos/aggregate/dense/validity/provenance；
- tactile validity 在 alignment 后 gating，而不是用来改变 sample identity；
- oldest-first model history；
- shared-memory copy ownership；
- pointcloud/RGB causal alignment。

可以删除/压缩：

- dataclass field comment 与 type annotation 完全重复；
- Docstring 中逐字段复制 signature；
- obvious NumPy conversion 说明；
- 已由类型/名称直接表达的 WHAT comments。

---

## 4.2 P1 — confirmed dead production surface

只有下列项经过当前 GitHub caller graph 反证，允许本轮直接删除。

### P1.1 `sensor/camera/transforms.py::resize_depth()`

当前 repo 搜索只有定义，无 caller。

删除整个 function，不删除 module，不修改 `resize_rgb()`。

### P1.2 `sensor/camera/transforms.py::resize_camera_intrinsic()`

同样只有定义，无 caller。

删除整个 function。

### P1.3 `FillReason.CAUSAL_HOLD_LAST`

当前 recorder 只写：

```python
FillReason.SOURCE
```

删除 enum member，不改变 `fill_reason` dataset field，不改变 `SOURCE = 0` value。

### P1.4 `FillReason.LEADING_PLACEHOLDER`

同上，仅删除 dead enum member。

### P1 删除后必须保持

```text
EPISODE_SCHEMA_VERSION == 29
fill_reason field remains present
new recorder rows still store fill_reason == SOURCE == 0
flag_sample_valid remains present and True for current rows
```

---

## 4.3 P2 — conditional cleanup

### P2.1 `ArmParams.tracking_error_warn_rad`

GitHub repo 内当前只有：

- dataclass field；
- `ArmParams.validate()`。

没有 runtime consumer。

但 `config/experiment.py` 对 unknown YAML fields 是 strict error，而本地 `experiments/` 等目录被 `.gitignore` 排除，GitHub 无法证明本机配置没有使用它。

因此删除前必须在 **本地同步后的真实工作区** 执行：

```bash
rg -n -uu --glob '!.git/**' 'tracking_error_warn_rad' .
```

决策：

```text
只有 defaults.py + docs/history 命中
    → 可删除 field + validation + current docs mention

任何 local YAML / experiment config / launcher / shell script 命中
    → STOP；本轮保留该 field，并在 handoff 报告 consumer
```

不要自动迁移 local config；用户明确决定后再做。

### P2.2 `HAND_TACTILE_SUM_SHAPE` terminology

它不是 dead code。当前语义已经是 aggregate tactile，但常量名仍是历史 `SUM`。

可选 rename：

```text
HAND_TACTILE_SUM_SHAPE
→ HAND_TACTILE_AGGREGATE_SHAPE
```

只有在满足以下条件时做：

1. 全 repo callers 可一次性原子更新；
2. 不涉及 persisted key / external policy key；
3. targeted tests 全通过；
4. 不把 historical migration/docs 中当时的 `tactile_sum` 名称机械改掉。

这是低优先级 readability cleanup。若 scope 已经较大，**本轮跳过更好**。

---

# 5. Comment / docstring cleanup policy

所有 Python 文件统一使用下面的判定，不做机械批量删除。

## KEEP

必须保留或只做轻量压缩：

### 5.1 Safety rationale

例如：

- 为什么 fail closed；
- 为什么某个 collision check 不能省；
- 为什么 SDK error/status 被当作 accepted/recoverable；
- 为什么 worker 必须做 final guard；
- 为什么不 clip action。

### 5.2 Concurrency / ownership

例如：

- 哪个 process owns SDK；
- shared-memory copy lifetime；
- generation/ticket ownership；
- recorder transaction ownership；
- async finalizer 的必要 blocking reason。

### 5.3 Temporal / causal semantics

例如：

```text
source / receive / publish / anchor
freshness
causal selection
history order
edge-repeat warmup
grid skew
latest-wins
```

### 5.4 Non-obvious hardware/SDK behavior

例如：

- XHand partial tactile validity；
- error code semantics；
- RealSense clock/format behavior；
- xArm firmware/URDF frame difference。

### 5.5 Non-obvious math

例如：

- Jacobian frame transform；
- equivalent-joint wrapping；
- manipulability；
- collision/path interpolation rationale；
- rot6d canonicalization。

## DELETE / SHORTEN

优先清理：

- 注释只复述下一行代码；
- docstring 机械重复函数名和完整 signature；
- 已由 dataclass field/type/shape constant 明确表达的重复 prose；
- “以前某 PR/计划如何做”的 production archaeology；
- production comment 链接已完成 implementation plan；
- stale schema version；
- stale caller/module name；
- commented-out code；
- obvious standard-library/NumPy operation narration。

## 不做的事情

```text
不设“每文件减少 X% comments”KPI
不为短小而拆 helper
不把 WHY comment 改成 cryptic code
不删除 hardware/safety caveat
不改变 exception/failure behavior
```

---

# 6. 本地执行流程

下面流程是 Claude Code 必须遵循的顺序。

## Phase 0 — Sync / preflight / establish baseline

### 0.1 同步

```bash
git status --short
git branch --show-current
git fetch origin
git checkout main
git pull --ff-only origin main
```

如果用户要求在已有 feature branch 执行，不切 main，但必须先确认 branch 与 origin/main 的关系。

### 0.2 不覆盖用户修改

如果 `git status --short` 非空：

- 不 stash/delete/reset 用户改动；
- 先判断是否与 cleanup 路径冲突；
- 有冲突则跳过相应文件并在 handoff 报告；
- 无冲突可以继续 narrow changes。

### 0.3 记录 baseline

```bash
git rev-parse HEAD
git log -1 --oneline
```

如果 HEAD 已不是本文 reviewed SHA，不直接假设结论仍成立。至少重新跑：

```bash
rg -n 'resize_depth|resize_camera_intrinsic' dexmani_real tests examples
rg -n 'CAUSAL_HOLD_LAST|LEADING_PLACEHOLDER|FillReason' dexmani_real tests examples
rg -n -uu --glob '!.git/**' 'tracking_error_warn_rad' .
rg -n 'policy_eval_workflow\.md' .codex .claude AGENTS.md CLAUDE.md README.md docs || true
```

### 0.4 Protected-path snapshot

```bash
git ls-files assets > /tmp/dexmani_assets_before.txt
git ls-files examples > /tmp/dexmani_examples_before.txt
```

最终必须满足：

```bash
git ls-files assets > /tmp/dexmani_assets_after.txt
git ls-files examples > /tmp/dexmani_examples_after.txt
diff -u /tmp/dexmani_assets_before.txt /tmp/dexmani_assets_after.txt
diff -u /tmp/dexmani_examples_before.txt /tmp/dexmani_examples_after.txt
```

两个 diff 都必须为空。

此外：

```bash
git diff --name-only -- assets
```

必须为空。

`examples` 可以内容修改，但不能删除/rename 文件。

---

## Phase 1 — Documentation + tooling only

先做 P0，不碰 production behavior。

建议顺序：

```text
1. README current contract
2. pointcloud/current-facing docs
3. examples help/docstrings
4. policy eval docs governance/current snapshot
5. xhand simplification plan status
6. user_design stale metric sentence
7. .claude local settings cleanup
8. .codex stale reference / machine path
9. minimal .gitignore cleanup
```

### Phase 1 验证

```bash
git diff --check
git diff --stat
git diff -- README.md docs examples .claude .codex .gitignore user_design.md
```

再做 stale search：

```bash
rg -n 'raw[- ]?v28|processed( HDF5)? v1[89]|Policy Zarr v1[12]' \
  README.md repo_map.md examples dexmani_real docs
```

**不要把所有命中当 bug。** 对每个命中分类：

```text
CURRENT-FACING → 更新
HISTORICAL EVIDENCE → 保留
MIGRATION CONTRACT → 保留
SUPERSEDED PLAN BASELINE → 可保留，但状态必须清楚
```

再检查 tooling dead links：

```bash
rg -n 'policy_eval_workflow\.md|teleop/core/pipeline\.py|robot/inner_loop\.py|keyboard_teleop_real\.py|dexmani_real\.policy' \
  .claude .codex CLAUDE.md AGENTS.md README.md docs || true
```

任何 current agent/tool configuration 中的不存在 path 都应修正；historical prose 可按语境保留。

---

## Phase 2 — Conservative comments/docstrings

只对已经审查过的高价值文件做 narrow pass：

```text
dexmani_real/deployment/inference/observation.py
dexmani_real/robot/hand_worker.py
dexmani_real/planning/kinematics/arm_fk.py
必要时 examples current-facing docstrings
```

不要全仓库 regex 删除 `#` / triple-quoted strings。

每次修改后阅读完整函数上下文，确认注释删除没有让以下语义不可见：

```text
safety
causality
ownership
hardware quirk
math rationale
failure behavior
```

### Phase 2 验证

```bash
python -m compileall -q dexmani_real examples
git diff --check
git diff -- dexmani_real/deployment/inference/observation.py \
             dexmani_real/robot/hand_worker.py \
             dexmani_real/planning/kinematics/arm_fk.py
```

如果 diff 中出现 executable statement 变化，必须逐行解释；comment-only phase 不应顺手重构代码。

---

## Phase 3 — Confirmed dead symbols

按顺序处理，单项完成后立即搜索：

### 3.1 camera transform helpers

删除：

```text
resize_depth()
resize_camera_intrinsic()
```

删除后：

```bash
rg -n 'resize_depth|resize_camera_intrinsic' dexmani_real tests examples || true
```

应无 active-code caller/definition。

保留：

```text
resize_rgb()
dexmani_real/sensor/camera/transforms.py
```

### 3.2 dead `FillReason` enum members

删除：

```text
CAUSAL_HOLD_LAST
LEADING_PLACEHOLDER
```

保留：

```text
FillReason
SOURCE = 0
fill_reason raw dataset field
```

删除后：

```bash
rg -n 'CAUSAL_HOLD_LAST|LEADING_PLACEHOLDER' dexmani_real tests examples docs || true
rg -n 'FillReason' dexmani_real tests examples
```

确认 recorder 仍写 `FillReason.SOURCE`。

### Phase 3 targeted validation

至少运行：

```bash
python -m compileall -q dexmani_real examples
python -m unittest discover -s tests -p 'test_raw_v26_recording.py'
python -m unittest discover -s tests -p 'test_recorder_io_boundary.py'
python -m unittest discover -s tests -p 'test_control_step_dataset.py'
```

如果当前环境具备完整 offline dependencies，再运行：

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

不得运行任何会连接 xArm/XHand/RealSense/Quest 或产生 robot command 的 example。

---

## Phase 4 — Conditional `tracking_error_warn_rad`

先执行：

```bash
rg -n -uu --glob '!.git/**' 'tracking_error_warn_rad' .
```

### Case A — 只有 source definition / validation / stale docs

允许：

- 删除 `ArmParams.tracking_error_warn_rad`；
- 删除对应 validate block；
- 删除 current-facing docs mention。

然后验证 config loading：

```bash
python - <<'PY'
from dexmani_real.config.experiment import resolve_experiment_config
cfg = resolve_experiment_config()
print(type(cfg.arm).__name__)
PY
```

这只是 pure config load；若本地 config resolution 会读取 table calibration JSON，它仍不得连接硬件。

### Case B — local YAML / launcher / experiment config 命中

**STOP，不删该 field。**

在 handoff 报告：

```text
tracking_error_warn_rad repo runtime unused, but local config consumer exists; preserved for compatibility.
```

不要偷偷改用户本地实验配置。

---

## Phase 5 — Optional nomenclature cleanup

默认可以跳过。

只有在主 cleanup 已通过且 diff 仍很小的情况下，才考虑：

```text
HAND_TACTILE_SUM_SHAPE → HAND_TACTILE_AGGREGATE_SHAPE
```

禁止改 historical migration schema/key 名称。

如果执行，必须：

```bash
rg -n 'HAND_TACTILE_SUM_SHAPE' dexmani_real tests examples
python -m compileall -q dexmani_real examples
python -m unittest discover -s tests -p 'test_xhand_driver_tactile.py'
python -m unittest discover -s tests -p 'test_hand_finger_order.py'
```

如果 rename 开始扩散到 historical docs/tools 或 serialized key，立即放弃本项。

---

# 7. Full validation gate

完成所有实际执行的 phase 后：

## 7.1 Static validation

```bash
python -m compileall -q dexmani_real examples
git diff --check
git diff --stat
git status --short
```

## 7.2 Protected path validation

```bash
git diff --name-only -- assets
```

必须为空。

确认 examples 文件集合没变化：

```bash
git ls-files examples > /tmp/dexmani_examples_after.txt
diff -u /tmp/dexmani_examples_before.txt /tmp/dexmani_examples_after.txt
```

必须为空。

## 7.3 No unintended schema change

```bash
rg -n 'EPISODE_SCHEMA_VERSION|PROCESSED_SCHEMA_VERSION|POLICY_ZARR_SCHEMA_VERSION' \
  dexmani_real/recording/storage/schema.py \
  dexmani_real/dataset/processed.py \
  dexmani_real/dataset/export.py
```

期望仍为：

```text
raw 29
processed 20
Zarr 13
```

并检查：

```bash
git diff -- dexmani_real/recording/storage/schema.py \
             dexmani_real/dataset/processed.py \
             dexmani_real/dataset/export.py \
             dexmani_real/ipc/schema.py
```

`recording/storage/schema.py` 本轮只允许 dead enum member cleanup；不允许 dataset spec/version 变化。

## 7.4 No planner/safety accidental change

```bash
git diff -- dexmani_real/planning/planner.py \
             dexmani_real/planning/paths.py \
             dexmani_real/planning/kinematics/ik_candidates.py \
             dexmani_real/control \
             dexmani_real/robot/arm_worker.py \
             dexmani_real/robot/hand_worker.py
```

预期：

- planner/paths/ik_candidates executable code **无变化**；
- control executable code **无变化**；
- hand_worker 最多 comment-only；
- safety threshold / timeout / limit 无变化。

## 7.5 Offline tests

优先 targeted tests；环境允许时再 full discover：

```bash
python -m unittest discover -s tests -p 'test_raw_v26_recording.py'
python -m unittest discover -s tests -p 'test_recorder_io_boundary.py'
python -m unittest discover -s tests -p 'test_control_step_dataset.py'
python -m unittest discover -s tests -p 'test_deployment_eef_tactile.py'
python -m unittest discover -s tests -p 'test_xhand_driver_tactile.py'
```

若完整 offline test suite 可运行：

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

如因本地缺少可选 dependency（例如 MPlib / Pinocchio / vendor package）导致 import failure，要区分：

```text
code regression
vs
missing local dependency
```

不可通过跳过 safety assertion、修改 test semantics 或 import-time fallback 来“让测试绿”。

---

# 8. STOP / rollback conditions

出现任一情况立即停止该项，不继续扩大 scope：

1. dead symbol 搜索出现新的 runtime caller；
2. 删除 config field 会破坏 local experiment YAML；
3. 修改需要 schema bump；
4. 修改需要改变 IPC dtype/shape；
5. 修改开始触碰 `assets/**`；
6. 需要删除/rename `examples/*.py`；
7. 需要弱化 collision/freshness/generation/ticket/safety gate；
8. comment cleanup 导致必须重构 executable logic 才能“更好看”；
9. docs archive/move 会产生大量 inbound-link churn；
10. optional terminology rename 开始影响 serialized/historical contract；
11. 为通过测试需要改变真实 runtime semantics；
12. 无法解释 diff 中任何 safety/timing/data-path executable change。

原则：

```text
保留一个可能多余但安全的符号
优于
误删一个隐藏 consumer / diagnostic / safety seam
```

---

# 9. Commit strategy

不要做一个巨大 cleanup commit。推荐：

### Commit 1 — current docs / tooling

```text
docs: sync current contracts and repository tooling
```

内容：

- README/docs/examples text；
- `.claude/settings.local.json` removal + ignore；
- `.codex` stale references / machine path；
- `.gitignore` minimal dedupe。

### Commit 2 — comments/docstrings

```text
refactor: trim stale comments without changing runtime behavior
```

只 comment/docstring changes。

### Commit 3 — confirmed dead symbols

```text
refactor: remove confirmed unused camera and recording symbols
```

仅：

- two camera transform helpers；
- two `FillReason` members。

### Commit 4 — optional config cleanup

仅当 local preflight 允许：

```text
refactor: remove unused arm tracking warning config
```

不要和其他逻辑混合。

---

# 10. Handoff report format

Claude Code 完成后必须报告：

```text
Baseline
- starting SHA
- ending SHA / branch

Changed
- files grouped by docs/tooling/comments/dead-symbol/config

Deleted symbols
- exact symbol list
- caller-search evidence

Protected
- assets untouched
- examples file set unchanged
- schema versions unchanged
- planner/safety executable logic unchanged

Validation
- compileall
- targeted tests
- full tests if run
- diff --check

Not validated
- hardware NOT RUN
- any optional dependency unavailable

Conditional decisions
- tracking_error_warn_rad deleted or preserved, with local search result
- optional tactile constant rename performed or skipped
```

禁止写“hardware validated”除非真实硬件确实被用户明确要求并实际运行。

---

# 11. Expected end state

本轮完成后，仓库应满足：

```text
Current-facing docs agree with raw29 / processed20 / Zarr13.
Current tactile docs agree with one-hand-sample atomicity.
Claude/Codex tooling no longer points at nonexistent files or personal absolute paths.
Production comments emphasize safety/causality/ownership/math rather than archaeology.
resize_depth / resize_camera_intrinsic are gone if still unreferenced.
FillReason only exposes SOURCE if historical enum members remain unused.
tracking_error_warn_rad is removed only if local ignored configs do not use it.
assets/** is byte-for-byte untouched by this cleanup.
examples/*.py file set is unchanged.
No schema bump.
No planner/safety/timing/hardware behavior change.
```

Cleanliness is accepted only when the diff is easier to audit **and** the runtime contracts remain unchanged.

---

# Appendix A — 本地 Claude Code 完整执行指令

将下面整段原样交给同步后的本地 Claude Code。Claude Code 必须先阅读本文，再执行修改。

```text
你正在 haoyangzhanglab/dexmani_real 仓库执行一次 post-refactor clean-code / repository-hygiene cleanup。

第一步必须阅读并遵循：
1. AGENTS.md
2. CLAUDE.md
3. code_style.md
4. docs/repo_cleanup_execution_guide.md

其中 source code / schema / config 是最高事实来源。如果当前 main 已经前移，不得机械套用 guide 中基于 f4d82d05 的结论；先重新 rg callsites/tests/contracts。

本任务目标：
- 修 current-facing stale docs/tooling；
- 保守清理 comments/docstrings；
- 删除 guide 中经过 fact-check 的极少量 dead production symbols；
- 不改变 runtime behavior、safety、timing、IPC、hardware backend、planner semantics、persisted schema。

HARD CONSTRAINTS：
- assets/** 完全不动：不删除、不移动、不 rename、不修改。
- examples/*.py 文件全部保留：不删除、不 rename、不移动；只允许 stale text/comments 和 confirmed local dead code cleanup。
- 不 bump raw/processed/Zarr schema。
- 不删除 fill_reason / flag_sample_valid / observation_valid / meta.success persisted fields。
- 不删除 XArm7MotionPlanner.__getattr__。
- 不删除 IKCandidateSearch.check_path_collisions()。
- 不删除 paths.py collision/table fallback。
- 不删除/改写 DexPilot/TAG、D400/L515、serial/EtherCAT、table fallback 等 hardware/backend path。
- 不 rename historical test files。
- 不删除 frozen migration tools/tests/docs。
- 不运行任何会连接 xArm/XHand/RealSense/Quest、command motion、home、replay、teleop 或写 calibration 的代码。
- 不覆盖现有用户未提交改动。

执行顺序：

PHASE 0 — PRECHECK
- git status --short
- git fetch origin
- 确认当前 HEAD/branch；如果 main 已前移，重新搜索相关 callsites。
- 保存 git ls-files assets/examples baseline。
- 运行：
  rg -n 'resize_depth|resize_camera_intrinsic' dexmani_real tests examples
  rg -n 'CAUSAL_HOLD_LAST|LEADING_PLACEHOLDER|FillReason' dexmani_real tests examples
  rg -n -uu --glob '!.git/**' 'tracking_error_warn_rad' .
  rg -n 'policy_eval_workflow\.md' .codex .claude AGENTS.md CLAUDE.md README.md docs || true

PHASE 1 — DOCS/TOOLING
按 docs/repo_cleanup_execution_guide.md P0 清单修：
- README current raw29 / processed20 / Zarr13 + current one-hand-sample tactile semantics；历史 artifact 版本不能机械替换。
- docs/pointcloud_pipeline.md current section。
- examples/process_episodes.py、visualize_episode_processed.py、replay_episode.py、visualize_episode.py 的 stale current-facing help/docstrings；保留文件和逻辑。
- dataset/pointcloud.py stale Raw-v28 module docstring。
- tactile_unit_si_verification_pending.md 的 current HEAD 部分，保留 historical evidence。
- policy_eval_finalization_plan.md：保留为 current finalization guide，同步 implementation snapshot/status。
- policy_eval_refactor_plan.md：保留 architecture/rationale，但消除和 finalization plan 的“双 current authority”。
- xhand_tactile_research_simplification_plan.md：目标已实现，状态改 IMPLEMENTED。
- user_design.md：保留 1501035 accepted grasp-contact，删除不存在的 episode metric counter claim。
- .claude/settings.local.json 从 Git 删除，并 exact-ignore；保留 .claude/skills。
- .codex/agents/*.toml 去掉不存在 docs/policy_eval_workflow.md 的引用，改指真实文档。
- .codex/config.toml 去掉 personal absolute writable root；不要编造未知 Codex 字段。
- .gitignore 只 minimal dedupe + local Claude settings ignore，不做大规模模板重写。

PHASE 2 — COMMENTS
只做保守 pass：
- deployment/inference/observation.py
- robot/hand_worker.py
- planning/kinematics/arm_fk.py
- 必要 examples docstrings

KEEP 所有 safety / causality / ownership / hardware quirk / non-obvious math comments。
删除只复述代码、stale version、stale caller、历史 implementation-plan 链接、commented-out code。
不要以 LOC reduction 为目标。

PHASE 3 — CONFIRMED DEAD SYMBOLS
在再次确认无 caller 后，仅删除：
- sensor/camera/transforms.py::resize_depth
- sensor/camera/transforms.py::resize_camera_intrinsic
- FillReason.CAUSAL_HOLD_LAST
- FillReason.LEADING_PLACEHOLDER

保留 resize_rgb；保留 FillReason.SOURCE=0；保留 fill_reason raw field；schema version 不变。

PHASE 4 — CONDITIONAL CONFIG
tracking_error_warn_rad 只有在：
  rg -n -uu --glob '!.git/**' 'tracking_error_warn_rad' .
确认没有 local ignored YAML/experiment/launcher consumer 时才删除 field + validation。
只要发现 local consumer，立即保留，并在最终报告说明。
不要自动修改用户 local experiment config。

PHASE 5 — OPTIONAL
HAND_TACTILE_SUM_SHAPE → HAND_TACTILE_AGGREGATE_SHAPE 默认跳过。
除非前面全部稳定、diff 很小、可原子更新且不影响 serialized/historical name，才考虑。

每个 phase 后检查 focused diff，不要顺手做邻近重构。

VALIDATION：
- python -m compileall -q dexmani_real examples
- git diff --check
- git diff --stat
- python -m unittest discover -s tests -p 'test_raw_v26_recording.py'
- python -m unittest discover -s tests -p 'test_recorder_io_boundary.py'
- python -m unittest discover -s tests -p 'test_control_step_dataset.py'
- python -m unittest discover -s tests -p 'test_deployment_eef_tactile.py'
- python -m unittest discover -s tests -p 'test_xhand_driver_tactile.py'
- 环境具备全部 offline dependency 时，再 python -m unittest discover -s tests -p 'test_*.py'

FINAL PROTECTION CHECK：
- git diff --name-only -- assets 必须为空。
- examples tracked file list 与 baseline 必须完全相同。
- raw/processed/Zarr version 必须仍为 29/20/13。
- planner.py / paths.py / ik_candidates.py 不应出现本任务导致的 executable logic change。
- safety threshold / timeout / collision / generation / freshness semantics 不得变化。

若出现任何需要 schema bump、hardware behavior change、planner/safety semantic change、删除 example 文件或修改 assets 的需求，STOP，不自行扩大 scope。

提交建议分开：
1. docs: sync current contracts and repository tooling
2. refactor: trim stale comments without changing runtime behavior
3. refactor: remove confirmed unused camera and recording symbols
4. （仅本地 config preflight 通过）refactor: remove unused arm tracking warning config

完成后给我一份结构化报告：baseline SHA、changed files、deleted symbols + caller evidence、protected-path checks、tests、未执行验证、tracking_error_warn_rad 决策。明确写 Hardware validation: NOT RUN，除非我另外要求并真实执行了硬件测试。
```
