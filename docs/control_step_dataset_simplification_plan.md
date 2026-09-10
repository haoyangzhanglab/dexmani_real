# Control-Step Dataset 简化、旧数据抢救与 Codex 自动执行计划

> 面向 Codex 的 canonical 执行文档。目标是把 DexMani Real 的数据链路从过度严格的
> camera-master / row-level cleaning 方案收敛为适合个人 PhD 研究项目的简单实现，同时
> 最大化保留昂贵真实机器人 demonstration。
>
> 当前基线（写本文时）：raw v27 / processed v16 / Policy Zarr v9。
> 源码和当前配置永远是最终 authority；若执行时版本已前移，schema 各自从当时当前版本
> 只递增一次，不要机械复用本文数字。

---

## 0. Executive decision

本轮不是继续“调宽阈值”，而是删除不必要的机制。

最终设计原则：

```text
16 Hz control step = canonical training timeline

obs[t]:
    latest causal camera available before control step T[t]
    latest causal arm state available before T[t]
    latest causal hand state available before T[t]
    latest causal contact/tactile available before T[t]

action[t]:
    exact action actually sent after obs[t]
```

对每个 modality 只要求：

```text
0 < source_monotonic_ns <= control_step_anchor_monotonic_ns
```

不要求：

```text
camera_source == arm_source == hand_source == tactile_source
```

也不要求：

```text
tactile_source <= camera_source
```

正常 real-world sensor timing variation 是 audit/debug 信息，不是自动删训练帧的理由。

### Accepted episode 的核心 invariant

```text
one raw episode
    -> one processed episode
    -> one Policy Zarr episode

accepted episode:
    processed_steps == raw_frames
    every raw row is preserved

rejected episode:
    reject the whole episode
    never delete a middle range and compact the remainder
```

当前 `pick_place_toy` 61 条旧数据的目标结果：

```text
60 / 61 episodes accepted with every source row preserved
1 / 61 rejected: episode_20260827_224527 (persistent IK failure)
```

按已有 incident baseline，若 raw 集合未变化：

```text
total raw frames:          14309
rejected IK episode:         197
expected retained frames:  14112  (98.62%)
expected retained episodes: 60/61 (98.36%)
```

这些数字是 acceptance target，不允许在代码里硬编码。若实际结果不同，必须先查明原因。

---

# 1. Project principles

开始工作前必须读取并遵守：

```text
AGENTS.md
CLAUDE.md
code_style.md
repo_map.md
```

本任务尤其遵守以下原则：

1. **这是个人博士研究代码，不是通用机器人平台。**
2. correctness / hardware safety 不能降低，但不要为未知 future consumer 建 framework。
3. 主路径应能顺读：record -> process -> train/export -> deploy。
4. 一种语义只保留一套 canonical implementation。
5. 决定废弃的机制直接删除，不做长期 deprecated wrapper。
6. Raw 保存昂贵、不可重建的真实传感器事实；processed 只保存当前实验真正需要的表示。
7. Runtime safety/freshness 与 offline dataset admission 分离。
8. 不为“更多数据”修改 robot command safety、collision、generation、shutdown 或 worker fault policy。

禁止新增：

```text
generic SensorManager
modality registry
plugin framework
data quality service
compatibility framework
abstract dataset backend hierarchy
```

如果一个 helper 只被一个简单调用点使用，优先内联或删除。

---

# 2. LeRobot / ManiUniCon reference conclusions

本方案参考的是它们的数据工程思想，不照搬具体实现。

## 2.1 LeRobot

核对版本：`huggingface/lerobot` commit `71a11efe77f55e61f3ab2ce45b40da8cf626afa9`。

重点源码：

```text
src/lerobot/scripts/lerobot_record.py
src/lerobot/datasets/dataset_writer.py
src/lerobot/datasets/feature_utils.py
```

其 recording 主路径是：

```text
robot.get_observation()
-> teleop/policy action
-> robot.send_action()
-> dataset.add_frame({observation, action})
```

Writer 主要验证：

```text
feature presence
shape
dtype
episode non-empty
feature schema consistency
```

它没有在 dataset writer 中建立“camera 是 master clock，所有 robot/tactile state 必须回退到
camera exposure timestamp”的通用合同。

**借鉴点**：一个 policy/control iteration 是 dataset frame 的自然单位。

**不照搬点**：DexMani Real 继续保存 source timestamps、exact sent action、robot safety provenance，
并保持实时硬件侧 fail-closed。

## 2.2 ManiUniCon

核对版本：`Universal-Control/ManiUniCon` commit `85c6f2e32ecf9f2bed62d202b058c39623444686`。

重点源码：

```text
tools/process_demo_data.py
maniunicon/utils/replay_buffer.py
```

`process_demo_data.py` 对 state/action/camera buffer 取共同 `n_steps=min(...)`，构造 episode，
显式检查 action NaN 后整条 `ReplayBuffer.add_episode()`。

**借鉴点**：episode-first、直接、少层级；不要把 offline preprocessing 变成传感器同步系统。

**不照搬点**：DexMani Real 不使用简单 `min(length)` 掩盖真实结构错误；recording 已有更明确
control-grid 和 exact sent action，因此继续保留这些优势。

## 2.3 Final interpretation

本项目采取中间位置：

```text
比 ManiUniCon 更严格：
    保留 causal timestamps / exact sent action / structural validation

比当前 DexMani v16 更简单：
    不做 camera-master synchronization
    不做 normal per-row deletion
    不维护 row repair / source segment machinery
```

---

# 3. What is a valid training step?

定义第 `t` 个 control step anchor 为 `T[t]`。

一个正常 observation 可以是：

```text
camera source   = 1034 ms
hand source     = 1051 ms
tactile source  = 1051 ms
arm source      = 1057 ms
control anchor  = 1062 ms
action          = after this observation
```

这是合法的 asynchronous multimodal observation：

```text
all source <= 1062 ms
```

`tactile_source > camera_source` 不等于 future information；只要 tactile 在 action 之前已经可用，
它对 Behavior Cloning 仍是 causal observation。

### Dataset 不负责证明

```text
all sensors physically sampled at the same instant
```

Dataset 只需保证：

```text
row identity is correct
payload is structurally readable
observation is from the same control step
action is the action associated with that step
no episode-internal compaction
```

---

# 4. Target architecture

## 4.1 Recording

```text
control-grid anchor T[t]
       |
       +-- camera frame available at the grid cut
       +-- arm feedback available at the grid cut
       +-- hand feedback available at the grid cut
       +-- tactile/contact feedback available at the grid cut
       |
       v
construct action
       |
       v
publish exact action
       |
       v
record one source row
```

Recording 不再额外构造第二套：

```text
camera source C
-> re-read arm/hand/tactile high-rate rings
-> policy_observation_*
```

## 4.2 Processing

```text
raw episode
    |
    +-- technical structural validation
    +-- episode-level persistent IK check
    +-- optional user include/exclude episode annotation
    |
    +-- rejected -> no processed file
    |
    `-- accepted -> transform ALL rows, same order, same length
```

不再有正常意义上的“clean rows”。

## 4.3 Export

```text
one processed file == one accepted complete episode

validate schema / payload / task consistency
-> append arrays
-> append episode_end
```

Exporter 不再重新判断某个 processed episode 是否曾经删过帧，因为 processed vNext 根本不允许
发布删过帧的 artifact。

## 4.4 Deployment

Deployment 仍保留实时 causal/freshness/safety 检查，但 observation alignment 使用同一 logical
control grid reference：

```text
reference times = policy logical control-grid times

camera[t]  = newest valid camera <= reference[t]
arm[t]     = newest valid arm <= reference[t]
hand[t]    = newest valid hand <= reference[t]
contact[t] = newest valid contact <= reference[t]
```

不要为了 dataset 简化而删除 runtime `not_before_ns`、max age、camera health、state validity、
unit/calibration 或 robot safety gate。

---

# 5. Delete vs keep

## 5.1 Delete: camera-aligned recording sub-system

如果 source search 确认没有其他 independent consumer，删除：

```text
teleop/control_loop/grid.py:
    _empty_policy_observation_signals
    _recording_policy_observation_signals
    TeleopGridObservation.policy_observation_signals

ipc/causal.py:
    read_structured_frame_aligned_to_source
    read_valid_structured_frame_aligned_to_source
```

并删除 raw / sample-ring 中所有：

```text
policy_observation_arm_qpos
policy_observation_hand_qpos
policy_observation_valid
policy_observation_contact_force
policy_observation_contact_force_valid
policy_observation_tactile_force
policy_observation_tactile_force_valid
policy_observation_tactile_source_monotonic_ns
policy_observation_tactile_calibrated
policy_observation_tactile_unit_code
```

这应该成为 raw v28（若执行时 raw 版本已前移，则从当前版本 +1）。

对应 tests 如果只证明被删除机制，直接删除，不改写成新 wrapper tests。

## 5.2 Keep: cheap raw facts

Raw 继续保存当前真实采集事实，包括：

```text
arm_qpos / qvel / tau
hand_qpos / current
hand_contact
hand_tactile_force
action_arm_joint_sent
action_hand_joint
action_arm_ee
observation_anchor_monotonic_ns
arm/hand/tactile/camera source timestamps
tactile_sum_fresh
tactile_fresh
tactile_calibrated
tactile_unit_code
flag_camera_fresh
camera frame numbers
RGB/depth media and calibration
```

`camera_health` 是一个廉价且有明确 runtime 语义的 scalar，可继续保留用于 debug；但 processing
不得依赖它删除普通 training row。若最终 source audit 证明它只服务已删除 offline admission 且
保留价值不足，可删除；不要为“字段少一个”强行删除有实际 debug 价值的 telemetry。

## 5.3 Delete: cross-row tactile repair

删除：

```text
select_tactile_rows_to_references
tactile_forward_fill
previous-row tactile repair
tactile_source_row selection
相关 selector tests
```

Canonical processed：

```text
contact_force[t] = raw hand_contact[t]
```

Raw dense `hand_tactile_force` 保留；但 dense tactile 不作为当前 canonical processed / Zarr
admission requirement。

## 5.4 Delete: row-level cleaning / segment machinery

Accepted episode 不删 row，因此删除 canonical path 中：

```text
selected_indices
keep_mask
drop_reason_bits
drop_reason_names
hard_invalid_reason_names
source_gap_findings
source_segment_ends
repair_masks
row compaction
row-range deletion
```

如果 `dataset/clean.py` 删除这些后没有独立职责，删除整个文件，把极少数 episode checks 放到
`dataset/processing.py` 的窄 helper 中。不要为了文件名继续保留一个空壳 cleaner。

## 5.5 Delete: temporal-quality framework

当前 `dataset/quality.py` 的 abrupt step / impulse / tracking / stall detectors 不影响 admission，
且当前论文任务没有直接依赖这些 detector。

因此删除：

```text
dexmani_real/dataset/quality.py
QualityPolicy
TemporalQualityConfig
TemporalQualityAssessment
assess_temporal_quality
对应 CLI options
对应 tests
```

若以后某个研究问题需要 trajectory quality 分析，再写一个 task-specific offline analysis script，
不要维护通用 quality framework。

## 5.6 Delete: processed row provenance group

对于 row-preserving processed：

```text
processed row i == raw row i
```

因此 `/provenance` group 没有信息增益，删除：

```text
source_row_index
source_sample_index
source_timestamp_s
source_segment_ends
source_keep_mask
source_drop_reason_bits
tactile_source_row_index
observation_reference_monotonic_ns
tactile_source_monotonic_ns
```

Processed 只需保留简单 attrs：

```text
source_path
source_episode
source_schema_version
source_frames
episode_steps
```

并验证：

```text
source_frames == episode_steps
```

## 5.7 Delete: processed datasets not used by canonical training artifact

Raw 保留昂贵信息，processed 是当前实验 cache。

建议 processed v17 删除：

```text
eef_pose
full tactile_force
```

理由：

- `eef_pose` 可由 `joint_state[:7]` deterministic FK 重建；
- full dense tactile 当前不进入 Policy Zarr core；
- 两者的 semantic attrs / validators 带来大量维护成本。

保留：

```text
joint_state
action
action_ee
contact_force
fingertip_points
+ rgb/depth/camera_intrinsic/camera_extrinsic when requested
+ point_cloud when requested
```

`action_ee` 必须保留，因为它表达 controller/operator intent，不能从 IK 后 joint target 无损重建。

`fingertip_points` 当前 Zarr 使用，先保留；若 source audit 证明所有当前 Policy 都不使用，再单独删除。

## 5.8 Delete: processed decision JSON machinery

删除：

```text
source_decision_json
quality_summary_json
invalid-frame report machinery
```

如果希望保留廉价统计，可以保留一个很小的 `audit_json`，但它不是 schema requirement，也不能
参与 admission。优先不加，除非当前实际 workflow 需要。

## 5.9 Simplify processed replay

不再读取 retained rows。

目标：

```text
processed.source_path
-> load full raw trajectory
-> assert processed_steps == raw_frames
-> replay raw exact sent action
```

删除：

```text
processed replay row-selection
retained-row mapping
segment consistency logic
```

---

# 6. Episode admission: only small, explicit rules

必须区分“数据质量 rejection”和“文件无法读取”。

## 6.1 Dataset quality rejection

Canonical automatic behavior-quality rejection 只保留：

```text
persistent IK failure episode
```

沿用当前已经验证的 transient/persistent IK boundary，除非源码事实证明该阈值已变化。
不要因为本任务随意重新调 threshold。

Short IK hold：

```text
KEEP
```

其他 sensor timing/runtime flags：

```text
KEEP
```

## 6.2 User rejection

Annotation 只保留 episode-level：

```text
include: bool
task_name: optional string
```

删除：

```text
include_ranges
exclude_ranges
```

禁止手工剪掉 episode 中间一段再把前后拼起来。

## 6.3 Technical processing failure

以下不是“quality filter”，而是 artifact 根本无法可靠构造，应直接 processing error / skip source：

```text
required file missing/unreadable
required dataset shape/dtype corrupt
RGB/depth cannot be decoded
required action payload NaN/Inf or wrong shape
source frame count internally inconsistent
severely non-monotonic episode control timestamp / sample identity
```

对于当前 61 条 historical dataset，已有 evidence 预期这些都不会触发；若触发必须调查，不能为了
达到 60/61 目标而 bypass。

---

# 7. Processed v17 target contract

如果执行时 processed 当前仍为 v16，则 bump 到 v17。

建议最小 attrs：

```text
schema_name = dexmani-real-processed-hdf5
schema_version = 17
domain = real
profile
task_name
source_path
source_episode
source_schema_version
source_frames
episode_steps
dt
obs_alignment = obs[t]_before_action[t]
observation_alignment = control_step_latest_causal
state_alignment = control_step
contact_force_source = raw_hand_contact_control_step
action_semantics = teleop_published_joint_target
processing_config_json   # only if required to reconstruct pointcloud processing config
```

要求：

```text
episode_steps == source_frames
```

Data keys：

```text
core:
    joint_state          float32 [N,19]
    action               float32 [N,19]
    action_ee            float32 [N,21]
    contact_force        float32 [N,5,3]
    fingertip_points     float32 [N,5,3]

RGB profiles:
    rgb
    depth
    camera_intrinsic
    camera_extrinsic

Point-cloud profiles:
    point_cloud
```

Mapping：

```text
joint_state[t]
    = concat(raw arm_qpos[t], raw hand_qpos[t])

action[t]
    = concat(raw action_arm_joint_sent[t], raw action_hand_joint[t])

action_ee[t]
    = concat(raw action_arm_ee[t], raw action_hand_joint[t])

contact_force[t]
    = raw hand_contact[t]

fingertip_points[t]
    = FK(raw arm_qpos[t], raw hand_qpos[t])

RGB/depth/pointcloud[t]
    = derived from raw camera row t
```

Do not re-select another raw row for state/tactile.

---

# 8. Policy Zarr v10 target contract

如果当前仍为 v9，bump 到 v10，因为 observation semantics 改变。

Zarr 不再需要 processed provenance admission。

Exporter 做：

```text
validate each processed artifact
validate one task/profile/dt/shape/dtype contract
append one processed artifact as one episode
append cumulative episode_end
transactionally publish destination
```

删除：

```text
_ArtifactRejection
_whole_episode_rejection
_indices_to_ranges
invalid-frame range report
gap tolerance / source segment logic
processed keep-mask admission
```

新的 complete-episode proof 很简单：

```text
processed v17 itself is only publishable when episode_steps == source_frames
```

Exporter 不重复证明同一件事。

---

# 9. Deployment alignment simplification

这是 safety-sensitive change，必须由 `astra-high` 设计和最终 review。

当前 deployment 已有：

```text
_select_control_grid_reference_ns(...)
_align_state_history_to_reference_ns(...)
_select_camera_control_grid(...)
```

因此不要新增新的 alignment abstraction。

目标改法：

1. 对所有 Policy（包括 RGB/pointcloud），先构造同一个 logical control-grid `reference_ns`。
2. Camera/pointcloud 仍按这些 desired logical grid times 选择 latest valid causal visual frame。
3. Arm/hand/contact/dense-tactile history 直接对齐到 `reference_ns`。
4. 删除 state/tactile 再对齐到实际 camera source timestamp 的分支。
5. 如果无其他 caller，删除 `_align_state_history_to_camera_frames()`。

必须保留：

```text
run_started_ns / not_before_ns
source <= publish <= anchor causality
max input age
max grid lag
state_valid / qpos_stale gates
camera live health gates
contact calibration/unit/source-match gates
dense tactile fresh gate when policy explicitly requests tactile_force
```

不要把 dataset 的“timing 不删训练行”错误传播成 runtime 的“任何旧 sensor 都能喂给 policy”。

---

# 10. Legacy `pick_place_toy` salvage

这是本任务的 hard gate：旧数据抢救成功后再继续删除历史兼容机制。

## 10.1 Never mutate legacy raw

原始目录只读：

```text
episodes/pick_place_toy/episode_*
```

禁止：

```text
modify data.h5
rewrite timestamp
fabricate camera_health
fabricate policy_observation_*
copy future tactile backward
insert/delete source rows
rename schema version in place
```

## 10.2 Scale lineage audit first

旧数据经历过 XHand tactile scale 变化。

必须逐 episode 检查：

```text
schema_version
converted_from_schema
```

直接录制 v25 的 frozen v25->v26 converter 使用 `*10` 恢复 SDK-native scale；v24-derived
v25 必须使用已有 explicit lineage option，绝不能根据 payload magnitude 猜 scale。

如果 61 条已经是正确 v26：

```text
DO NOT scale again
```

## 10.3 Legacy processing semantics

对旧 v26/v25 数据使用它真实具备的 control-step fields：

```text
joint_state   <- arm_qpos + hand_qpos
contact_force <- hand_contact
action        <- exact action_arm_joint_sent + action_hand_joint
RGB/depth     <- same raw row
point_cloud   <- same raw row camera
```

`tactile_source > camera_source` 不构成 rejection。

Historical：

```text
frame0 camera/tactile deficit -> KEEP
flag_camera_fresh=false rows   -> KEEP
short IK                       -> KEEP
```

只有 persistent IK episode 整条 reject。

## 10.4 Legacy reader implementation rule

不要把 v25/v26 compatibility 扩散进 runtime/raw canonical reader。

优先选择最小实现：

```text
a narrow read-only adapter/local path inside the offline salvage tool or dataset processing boundary
```

只支持实际存在的 historical schema 和新 processing 所需字段。

禁止构建：

```text
version registry
migration graph
adapter hierarchy
schema compatibility framework
```

如果一个 50~100 行的明确 legacy read path 能完成任务，就不要抽象。

## 10.5 Expected output

Dry-run 必须首先得到：

```text
source episodes: 61
accepted:        60
rejected:         1
```

Rejected 必须只包含：

```text
episode_20260827_224527
reason: persistent IK failure
```

然后正式生成新的 processed v17 + Zarr v10。

对每条 accepted episode：

```text
processed_steps == source raw frames
```

全局若 baseline 未变化：

```text
retained frames == 14112
```

如果不是，停止后续 destructive cleanup，交给 `astra-high` root-cause review。

## 10.6 Salvage evidence

保存一个小的、machine-readable manifest（不要复制大 raw data）：

```text
artifacts/pick_place_toy_salvage_manifest.json
```

建议只记录：

```text
source episode name
source schema/version lineage
source frame count
accepted/rejected
rejection reason
processed frame count
output schema versions
```

不要把本机绝对路径写入 tracked artifact。

---

# 11. Historical tools cleanup

完成 salvage 并把关键结果写入 incident/documentation 后，评估删除：

```text
tools/analyze_camera_tactile_alignment.py
artifacts/camera_tactile_alignment_baseline.json
```

当前 forensic tool 已服务完 incident，而且其旧/current exporter 命名容易随代码演化变 stale。

如果删除，关键历史数值必须先保存在：

```text
docs/invalid_frames_export_incident.md
```

不要为了保存一次 incident 的脚本继续维护旧 cleaner semantics。

已有 historical raw migration tools 在 61 条数据完成 scale lineage/salvage 前不要删除。

---

# 12. CLI simplification

`examples/process_episodes.py` 建议最终只保留当前实际需要的参数：

```text
input_root
--output-root
--profile
--pointcloud-num-points
--task-name
--annotations
--dry-run
--verify-output  # 若现有 workflow 仍使用；否则可删
```

删除：

```text
--horizon
--min-full-windows
--max-camera-age-s
--max-observation-skew-s
--quality-policy
--abrupt-arm-step-rad
--abrupt-hand-step-rad
--compare-profiles
```

不要保留 compatibility aliases。

---

# 13. Tests: test the new contract, not deleted machinery

不要试图把旧 tests 全部“修到绿”；先判断它测试的是当前 contract 还是已删除行为。

## 13.1 Keep/add essential tests

### D1. Row preservation

Synthetic raw N=40：

```text
processed steps == 40
row order unchanged
```

### D2. Camera timing is not offline rejection

构造：

```text
flag_camera_fresh=false on isolated row
camera source still causal to control anchor
```

期待：

```text
episode accepted
all rows preserved
```

### D3. Tactile can be newer than camera

构造：

```text
camera_source  = 1000 ms
tactile_source = 1016 ms
anchor         = 1025 ms
```

期待：

```text
KEEP
direct raw hand_contact used
no previous-row selection
```

### D4. Dense tactile does not gate canonical dataset

构造 one row：

```text
tactile_sum valid
dense tactile fresh=false
```

期待：

```text
processed/Zarr canonical episode still accepted
contact_force present
```

### D5. Short IK hold

`FRAME_IK_FAIL` run within current transient threshold：

```text
KEEP all rows
```

### D6. Persistent IK

run exceeds current persistent threshold：

```text
reject WHOLE episode
no processed artifact for it
```

### D7. Exact sent action

Processed action must equal raw `action_arm_joint_sent + action_hand_joint` exactly before float32 cast.

### D8. Export

Two accepted processed files：

```text
2 input HDF5 -> 2 Zarr episodes
correct episode_ends
```

No gap/rejection machinery exists.

### D9. Deployment common control-grid reference

Visual Policy fixture：

```text
camera source < logical grid reference
state/tactile source may be > camera source but <= logical grid reference
```

期待 state/tactile selected against logical control-grid reference, not camera source.

Runtime age/not_before/health failures must still fail as before.

## 13.2 Delete obsolete tests

Delete tests whose only purpose is:

```text
camera-source tactile selection
frame0 camera-aligned tactile
validity-gated aligned-source recording helper
cross-row tactile selector
forward-fill
row-gap segmentation
keep-mask provenance
gap-tolerance exporter
temporal-quality framework
```

Do not keep dead behavior alive just because a test exists.

---

# 14. Codex automated agent workflow

This is an orchestration policy, not application runtime code. Do not add model names to DexMani Python modules.

Requested model routing:

```text
astra-high   = extremely hard / architecture-critical / safety-critical / final adversarial review
astra-medium = important implementation / focused cross-boundary changes
luna-max     = simple but tedious search / mechanical edits / test runs / doc cleanup
```

If the Codex agent runtime supports explicit model aliases, use these exact aliases. If an alias is unavailable,
use the closest available reasoning tier and record the substitution in the final execution evidence; do not
silently pretend the requested model ran.

## 14.1 Agents

### Agent A — Lead Architect / Adversarial Reviewer

```text
model: astra-high
mode: read/write only when integrating; otherwise review
```

Responsibilities:

```text
confirm control-step contract
review schema boundaries
review deployment alignment safety
approve salvage semantics
review final diff for accidental complexity
```

### Agent B — Repository Inventory

```text
model: luna-max
mode: read-only
```

Responsibilities:

```text
rg all symbols/fields scheduled for deletion
produce caller/consumer list
identify obsolete tests/docs
find current schema constants and version labels
```

### Agent C — Dataset Simplification

```text
model: astra-high
```

Reason: this changes training semantics and historical data interpretation.

Responsibilities:

```text
processed vNext contract
row-preserving processing
episode-level persistent IK rejection
legacy salvage path
```

### Agent D — Export Simplification

```text
model: astra-medium
```

Responsibilities:

```text
remove provenance/gap admission
simplify Zarr writer
focused export tests
```

### Agent E — Recording / Raw Cleanup

```text
model: astra-medium
```

Responsibilities:

```text
remove camera-aligned policy_observation recording path
raw schema cleanup
sample/frame/schema propagation cleanup
```

### Agent F — Deployment Alignment

```text
model: astra-high
```

Reason: live policy observation semantics and robot safety are critical.

Responsibilities:

```text
use common logical control-grid reference
remove camera-source state/tactile realignment
preserve all live health/freshness/safety gates
focused deployment regressions
```

### Agent G — Mechanical Cleanup / Test Runner

```text
model: luna-max
```

Responsibilities:

```text
remove stale imports/comments/tests
CLI option deletion
version-label grep cleanup
run compile/test/rg checks
document exact outputs
```

### Agent H — Final Independent Review

```text
model: astra-high
mode: read-only adversarial review
```

Must try to falsify:

```text
"accepted episodes preserve every row"
"only persistent IK rejects the historical dataset"
"training and deployment use the same logical reference"
"runtime safety was not weakened"
"deleted mechanisms have no live caller"
```

If H finds a material issue, route the fix back to the owning implementation agent and rerun review.

## 14.2 Concurrency policy

Read-only inventory/reference agents may run in parallel.

Editing agents must NOT concurrently modify the same boundary/files.

Preferred ordering:

```text
A architecture decision
    |
    +--> B read-only inventory
    |
    v
C dataset vNext
    |
D export vNext
    |
C + G historical salvage dry-run / rebuild
    |
    | HARD GATE: 60/61 expected
    v
E recording/raw cleanup
    |
F deployment alignment
    |
G mechanical deletion/docs/tests
    |
H final adversarial review
```

Do not spawn agents merely to increase count. One owner per phase is preferred.

---

# 15. Codex phase-by-phase execution

## Phase 0 — Preserve and inspect

Owner: `luna-max`, reviewed by `astra-high`.

Run:

```bash
git status --short
git log -8 --oneline
```

Read project contracts and this document.

Create a deletion inventory with `rg` for at least:

```text
policy_observation_
read_structured_frame_aligned_to_source
read_valid_structured_frame_aligned_to_source
select_tactile_rows_to_references
source_segment_ends
source_keep_mask
source_drop_reason_bits
source_decision_json
quality_summary_json
TemporalQualityConfig
QualityPolicy
assess_temporal_quality
```

No semantic edits in this phase.

## Phase 1 — Simplify processed dataset first

Owner: `astra-high`.

Why first: secure the desired training artifact semantics before touching live recording/deployment.

Implement processed v17 row-preserving contract.

At this stage current raw v27 may still exist unchanged; processing should already use direct control-step raw fields,
not `policy_observation_*`.

Expected:

```text
normal accepted raw N -> processed N
```

Remove row deletion/repair/quality machinery as far as possible without blocking legacy salvage.

## Phase 2 — Simplify Zarr exporter

Owner: `astra-medium`.

Implement Policy Zarr v10 direct one-file -> one-episode projection.

No keep-mask/gap admission.

Run synthetic D1-D8 relevant tests.

## Phase 3 — Salvage historical 61 episodes

Owner: implementation `astra-high`; repetitive execution/report `luna-max`.

Do not continue to raw/runtime deletion until this phase passes.

Steps:

```text
1. audit schema/scale lineage
2. dry-run all 61 with new control-step semantics
3. investigate every rejection
4. require expected 60 accepted / 1 persistent-IK rejected
5. formally rebuild processed v17
6. export Policy Zarr v10
7. verify lengths and episode_ends
8. write salvage manifest
```

If any episode other than `episode_20260827_224527` is rejected:

```text
STOP destructive cleanup
astra-high reviews root cause
```

Do not add a special case solely to force acceptance.

## Phase 4 — Recording/raw deletion

Owner: `astra-medium`, reviewed by `astra-high`.

Now delete the camera-aligned recording sub-system and raw `policy_observation_*` fields.

Bump raw v27 -> v28 if still current.

Do not create v26/v27 -> v28 migration solely to make old data look current. Historical data has already been rebuilt
from read-only raw into the new training artifact.

## Phase 5 — Deployment common control-grid alignment

Owner: `astra-high`.

Use existing logical control-grid references for all modalities.

Delete camera-source state/tactile re-alignment branch.

Do NOT weaken live freshness/safety gates.

Run D9 and existing deployment safety-focused tests.

## Phase 6 — Delete obsolete frameworks

Owner: `luna-max`, reviewed by `astra-medium`.

Delete remaining dead:

```text
quality.py
row segment/provenance code
legacy gap tests
camera-alignment recording tests
stale CLI options
stale docs/comments
```

Run `rg` to prove scheduled-for-deletion symbols no longer appear outside frozen historical documentation where explicitly retained.

## Phase 7 — Documentation and incident closure

Owner: `luna-max`; technical review `astra-high`.

Update:

```text
docs/data_schema.md
README.md
repo_map.md
docs/invalid_frames_export_incident.md
```

After new architecture is implemented and evidence recorded, this document remains the implementation rationale / execution record.
The older `docs/camera_tactile_episode_integrity_fix_plan.md` describes a superseded camera-master architecture and should be
removed once all unique historical evidence has been transferred to the incident document.

Do not keep two canonical design docs with contradictory architecture.

## Phase 8 — Final adversarial review

Owner: independent `astra-high`.

Review source, not commit messages.

Trace:

```text
recording control step
-> raw row
-> processed row
-> Zarr row
-> deployment observation
```

Verify all five use the same control-step interpretation.

---

# 16. Validation gates

At minimum run:

```bash
python -m compileall -q dexmani_real examples tools
git diff --check
git diff --stat
git status --short
```

Run all focused unit/offline tests touched by this change.

If repository full pytest is affordable, run it after focused tests.

Never run hardware-affecting programs automatically.

### Required static cleanup checks

Final source should have no live canonical references to removed mechanisms. Use `rg` and inspect every result, e.g.:

```bash
rg -n "policy_observation_|select_tactile_rows_to_references|tactile_forward_fill" dexmani_real examples tests
rg -n "source_keep_mask|source_drop_reason_bits|source_segment_ends" dexmani_real examples tests
rg -n "TemporalQualityConfig|QualityPolicy|assess_temporal_quality" dexmani_real examples tests
rg -n "_align_state_history_to_camera_frames" dexmani_real tests
```

Historical incident docs may mention deleted names as historical facts; runtime/current docs must not describe them as active.

---

# 17. Acceptance criteria

## Architecture

- [ ] Canonical training timeline is control-step-centric.
- [ ] Camera is not master clock for arm/hand/tactile training alignment.
- [ ] Accepted processed episode preserves every source row.
- [ ] No canonical row compaction / source segmentation / cross-row tactile repair exists.
- [ ] Dense tactile remains in raw but does not gate current processed/Zarr dataset.
- [ ] Processed no longer stores redundant `eef_pose` / dense tactile unless a verified current consumer requires one.
- [ ] Processed provenance group and row-selection JSON machinery are removed.
- [ ] Zarr exporter is one processed file -> one episode with no gap logic.
- [ ] Deployment uses logical control-grid reference for state/tactile even with visual policies.
- [ ] Runtime causal/freshness/safety gates remain intact.

## Historical salvage

- [ ] Original `pick_place_toy` raw is unmodified.
- [ ] Scale lineage is checked before reuse.
- [ ] 61 source episodes audited.
- [ ] Expected 60 accepted, one rejected.
- [ ] Only `episode_20260827_224527` is rejected for persistent IK failure, unless new objective corruption evidence is found.
- [ ] Every accepted processed length equals its source raw frame count.
- [ ] Expected retained total is 14112 frames if source baseline is unchanged.
- [ ] New Zarr contains exactly 60 episodes if source baseline is unchanged.

## Code simplicity

- [ ] `dataset/quality.py` removed unless a current research consumer is demonstrated.
- [ ] camera-aligned recording sub-system removed.
- [ ] cross-row tactile selector removed.
- [ ] row-level range annotations removed.
- [ ] stale tests are deleted rather than preserving dead behavior.
- [ ] no new generic compatibility/manager/plugin abstraction was introduced.

---

# 18. Failure policy

If a simplification conflicts with hardware safety:

```text
keep safety
simplify dataset code instead
```

If old data cannot be proven to satisfy a newly invented field:

```text
do not fabricate the field
use the old data's real control-step semantics
```

If one old episode has a true persistent behavioral failure:

```text
reject whole episode
```

If a sensor timing flag is merely imperfect telemetry but the recorded control-step payload exists:

```text
keep the row
```

If an implementation needs a large abstraction to support only the 61 historical episodes:

```text
reject the abstraction
write a narrow offline salvage path
```

If a deleted mechanism later becomes scientifically relevant:

```text
reintroduce it as an explicit research hypothesis + ablation
not as default infrastructure
```

---

# 19. Commit strategy

Prefer a small number of coherent commits, not one commit per tiny cleanup.

Suggested sequence:

```text
1. refactor: simplify processed data to row-preserving control-step semantics
2. refactor: simplify policy zarr episode export
3. data: rebuild and verify legacy pick-place-toy dataset
4. refactor: remove camera-aligned recording path
5. refactor: align deployment observations to the control grid
6. cleanup: remove obsolete dataset quality/provenance machinery
7. docs: record simplified dataset contract and salvage evidence
```

Do not commit generated large datasets to Git. Only commit small manifests/reports intentionally tracked by the project.

---

# 20. Final Codex handoff report

Codex 最终必须报告：

```text
Baseline HEAD
Final HEAD
Raw / processed / Zarr final schema versions
Deleted modules/functions/fields
Remaining deliberate timing/safety mechanisms
Focused tests run and result
Full tests result if run
Hardware validation: NOT RUN
```

Historical salvage 必须给 actual numbers：

```text
source episodes
accepted episodes
rejected episodes + exact reason
source frames
retained frames
Zarr episode_count
```

并回答：

```text
Did any accepted episode lose a row?        must be NO
Did any raw historical episode get edited? must be NO
Did any camera/tactile timing-only event reject an old episode? expected NO
Was runtime safety/freshness weakened?      must be NO
```

最后运行：

```bash
git status --short
git log -8 --oneline
```

附 concise summary。

---

# 21. One-paragraph context recovery

如果后续聊天只读一段：DexMani Real 是个人 PhD 真实机器人研究项目，dataset 不应模拟生产级
multi-sensor synchronization system。Canonical timeline 改为 16 Hz control step；camera、robot state、
contact/tactile 允许异步，只需是 control action 前可用的 causal observation。Accepted episode 不做
normal per-row cleaning，所有 raw rows 原序保留；只有 persistent IK 等真正 episode-level behavior
failure 整条拒绝，技术性文件损坏则作为 processing error。删除 camera-aligned `policy_observation_*`
recording 子系统、cross-row tactile selector、row compaction/segment/provenance machinery、通用 temporal
quality framework；processed 只保留当前训练需要的字段，Zarr 一 processed file 对应一 episode。
Deployment 使用同一个 logical control-grid reference，但继续严格执行 live freshness/health/safety。
历史 `pick_place_toy` 61 条 raw 永不修改，按真实 control-step semantics 重建，目标 60 条全帧保留、
仅 `episode_20260827_224527` persistent IK 整条拒绝。