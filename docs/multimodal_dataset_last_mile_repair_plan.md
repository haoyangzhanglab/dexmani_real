# DexMani 多模态数据集最后收尾修复方案（Fact-Checked）

> 状态：**implementation plan / last-mile only**  
> 主工作仓库：`dexmani_real`  
> 邻接 consumer：`../dexmani_policy`  
> Fact-check 基线：
> - `dexmani_real@ad9b9af5808ec0db2226242a80610ae06a8cac8d`
> - `dexmani_policy@512f8b82e39f50d1b5a999669b5911646d42072f`
> - `huggingface/lerobot@71a11efe77f55e61f3ab2ce45b40da8cf626afa9`
> - `Universal-Control/ManiUniCon@85c6f2e32ecf9f2bed62d202b058c39623444686`
>
> 本文只处理当前主架构之后仍真实存在的缺口。**不要再做 dataset architecture redesign。**

---

## 0. 最终判断

当前主架构已经正确：

```text
control step = row identity
raw = source facts
processed v18 = row-preserving multimodal research dataset
Policy Zarr v11 = reusable multimodal superset
Policy = requested-key consumer
```

以下机制已经正确，不再修改：

```text
no camera-master alignment
no row compaction / stitching
no cross-row tactile repair
no modality-specific OutputProfile
native aligned RGB-D in processed
pad_before=n_obs_steps-1
run-start observation edge-repeat
healthy camera reuse across adjacent policy ticks
recording BEGIN does not require dense tactile
short-IK held action_ee fallback via FK(held qpos)
schema_version is not a Policy exact-version compatibility gate
```

本轮只修：

```text
1. point-cloud numeric train/deploy contract
2. fingertip numeric geometry train/deploy contract
3. dense tactile dataset semantics + low-cost validity telemetry
4. legacy v26 contact/dense validity over-claim
5. processing exception swallowing
6. v18/v11 historical salvage evidence
7. stale names/docs + episode discovery
```

完成后停止继续重构 dataset infrastructure。

---

# 1. 参考 ManiUniCon / LeRobot 后采用的原则

## 1.1 LeRobot：data feature 与 model preprocessing 分离

LeRobot 当前 recording 主链仍是直接：

```text
robot.get_observation()
→ observation processor
→ teleop action
→ robot.send_action()
→ dataset.add_frame()
```

Dataset 由 feature metadata 描述数据，temporal query 在 episode 边界 clamp 到 edge frame，并额外提供 padding mask。它不会要求 episode 存储虚构的负时间历史。

参考：

- `src/lerobot/scripts/lerobot_record.py`
- `src/lerobot/datasets/dataset_reader.py::_get_query_indices`

借鉴：

```text
data保存事实/feature
model-specific preprocessing留给Policy
edge padding属于sampling
只在真正的consumer boundary验证需要的语义
```

不照搬 Hub/backend/version framework。

## 1.2 ManiUniCon：简单 episode-first pipeline

ManiUniCon 的 `tools/process_demo_data.py`：

```text
episode_*
→ load state/action/camera
→ construct multimodal obs/action
→ replay_buffer.add_episode()
```

并保留 RGB/depth/intrinsics/transforms 等观测，而不是按当前 policy 删除未来可能使用的模态。

其 realtime `SharedMemoryRingBuffer.get_last_k(k)` 在历史不足时重复 oldest available row，说明 edge-repeat 本身是正常 runtime temporal semantics。

借鉴：

```text
explicit episode_* discovery
multimodal reusable data
small direct functions
no generic synchronization framework
```

不照搬其 `min(lengths)` 截断行为；DexMani 已有更强的 row identity/control-step invariant，应继续保持 exact row preservation。

---

# 2. Fact-check matrix

| 原 review 项 | 核实结果 | 最终处理 |
|---|---|---|
| Point-cloud numeric config 未闭环 | **CONFIRMED** | 立即修 |
| Fingertip mount/link geometry 未闭环 | **CONFIRMED** | 立即修，使用小 JSON，不做 hash framework |
| Dense tactile dataset semantics 缺失 | **CONFIRMED** | 立即补 dataset/Zarr semantics |
| 现在就给 Policy 增 `tactile_force` deployment bridge | **NOT JUSTIFIED NOW** | **不做**；当前 Policy 无 tactile model consumer，其现有 deployment guide 也明确将 EEF/tactile support 留作独立研究任务 |
| Legacy v26 `contact_force_valid = finite` | **CONFIRMED BUG** | 使用历史 freshness proxy 保守标注；dense legacy 同样处理 |
| v18/v11 salvage 已有机器证据 | **FALSE** | README 已写 v18/v11，但 tracked manifest 仍是 v17/v10；必须真实重跑后更新 |
| processing technical exception 只影响单 episode | **CONFIRMED DESIGN BUG** | 删除 broad swallowing，technical failure 直接 fail batch |
| stale v17/v10/v27 名称 | **CONFIRMED** | 最后机械清理 |
| episode discovery 接受所有非隐藏目录 | **CONFIRMED WEAKNESS** | 改为 `episode_*` |
| warm-start first action 被跳过是 bug | **NOT A BUG** | 不改 scheduler；这是 full-future-chunk + `first_future_step_index()` 的预期 stale-prefix handling |
| 应新增 `point_cloud_valid` | **DEFER** | 当前无 mask-aware PC consumer；真实数据证明 empty-PC 高频后再研究 |

---

# 3. P0 — Point-cloud numeric train/deploy contract

## 3.1 已确认事实

Offline processed 已保存：

```text
processing_config_json = {
    pointcloud: PointCloudConfig.to_dict(),
    table_plane_abcd: ...
}
```

其中真实影响 point-cloud distribution 的参数包括：

```text
depth_min_m / depth_max_m
edge_jump_m / edge_surface_band_m
support thresholds
table thresholds
workspace
voxel_size_m
outlier_radius_m / neighbor/component thresholds
candidate multiplier
sampling_coarse_voxel_stride
num_points
```

Realtime `pointcloud_worker.py` 使用当前 `runtime.pointcloud` 调同一 `build_point_cloud()`。

但是当前 Policy deployment field semantics 只传播：

```text
frame
position_units
color_order/color_source
policy_id
table_plane_abcd_json
sampling
transform
```

因此 numeric config 变化仍可能通过 compatibility check。

## 3.2 最小修法

**直接传播已经存在的 canonical JSON；不要 hash，不要新 registry。**

### dexmani_policy

在：

```text
../dexmani_policy/dexmani_policy/deployment/export.py::_validate_point_cloud
```

读取 Zarr attr：

```text
processing_config_json
```

要求：

```text
non-empty valid JSON object
contains pointcloud + table_plane_abcd
```

不在 Policy 重写一遍 `PointCloudConfig` validator。

将原 canonical JSON string 原样放进 point-cloud field semantics：

```text
semantics["processing_config_json"]
```

### dexmani_real

在：

```text
dexmani_real/deployment/config.py::_expected_pointcloud_semantics
```

增加：

```python
"processing_config_json": canonical_json({
    "pointcloud": runtime.pointcloud.to_dict(),
    "table_plane_abcd": table_plane,
})
```

canonical JSON：

```python
json.dumps(..., sort_keys=True, separators=(",", ":"), allow_nan=False)
```

然后复用现有 `_validate_field_semantics()` 做 exact compare。

### 不要做

```text
pointcloud_config_sha256
new PointCloudContract class
schema registry
version gate
```

## 3.3 Regression

至少测试一个非 shape 参数变化：

```text
training voxel_size_m = 0.005
deployment voxel_size_m = 0.006
→ compatibility MUST fail
```

再测：

```text
workspace mismatch -> fail
outlier_radius mismatch -> fail
identical config -> pass
```

---

# 4. P1 — Fingertip numeric geometry contract

## 4.1 已确认事实

Offline fingertip points 由：

```text
joint_state
hand URDF / FK implementation
fingertip_link_names
T_eef_handbase position
T_eef_handbase quaternion
```

共同决定。

当前 artifact 只保存：

```text
frame
unit
derivation
policy_id
```

而 Real deployment 用当前 runtime mount/link config 重新 FK。

虽然当前 Policy 没有实际 fingertip model consumer，但 exporter 已经把 `fingertip_points` 作为支持字段暴露，所以这个 public surface 应该闭环。

## 4.2 最小修法

在 processed/Zarr 新增一个 string attr：

```text
fingertip_config_json
```

内容只保存 portable numeric/semantic inputs：

```json
{
  "fingertip_link_names": ["..."],
  "handbase_position_eef_m": [x,y,z],
  "handbase_quat_eef_wxyz": [w,x,y,z]
}
```

**不要保存 absolute URDF path。**

已有：

```text
fingertip_points_policy_id
```

继续承担 FK implementation / canonical hand-model identity；若未来 URDF/算法语义真的改变，应显式 bump policy id，而不是引入文件 hash framework。

### Propagation

```text
processing.py
→ processed attrs
→ dataset/export.py semantic attr copy
→ Policy _validate_fingertip_points()
→ ObservationFieldSpec semantics
→ Real _expected_fingertip_semantics(runtime)
→ exact compare
```

### Regression

```text
same link/mount config -> pass
mount translation mismatch -> fail
mount quaternion mismatch -> fail
link order mismatch -> fail
```

---

# 5. P1 — Dense tactile data semantics and low-cost telemetry

## 5.1 Fact-check correction

Dense tactile 已经正确进入 processed/Zarr：

```text
tactile_force [N,5,120,3]
tactile_force_valid [N]
```

但 artifact 没有完整说明这个 tensor 的物理/排列语义。

Real runtime 已经拥有相应语义常量，因此应把它们放入数据 artifact；这是**数据自描述**，不是新增 Policy model capability。

## 5.2 新增 tactile attrs

Processed/Zarr root attrs 增加：

```text
tactile_force_representation = xhand_sdk_raw_force_fx_fy_fz_bias_corrected
tactile_force_finger_order = thumb_index_mid_ring_pinky
tactile_force_sensor_order = xhand_sdk_sensor_data_order
tactile_force_point_order = xhand_sdk_sensor_data_raw_force_order
tactile_force_axis_labels = fx_fy_fz
tactile_force_unit = xhand_sdk_native_unknown_si
tactile_force_si_verified = false
tactile_force_spatial_geometry_verified = false
```

如果已有 canonical constants，复用；不要再复制 magic string 到更多 Real modules。

## 5.3 保留低成本 row telemetry

当前 raw 已经有：

```text
tactile_sum_fresh
tactile_fresh
tactile_calibrated
tactile_unit_code
```

对 reusable tactile dataset，这些信息非常便宜且能解释 invalid 原因。Processed/Zarr 建议增加：

```text
contact_force_fresh   bool [N]  <- raw tactile_sum_fresh
tactile_force_fresh   bool [N]  <- raw tactile_fresh
tactile_calibrated    bool [N]
tactile_unit_code     uint8 [N]
```

不要增加 reason enum / quality bits。

这 4 个 scalar rows 足以区分：

```text
payload missing
stale
uncalibrated
wrong/unknown unit lineage
```

## 5.4 不做 Policy tactile bridge

不要修改：

```text
../dexmani_policy/_SUPPORTED_OBSERVATION_FIELDS
Policy encoders
Policy modality configs
```

当前 Policy source search 没有 `tactile_force` model consumer；现有 `policy_deployment_simplification_guide.md` 也明确说明 EEF/tactile end-to-end Policy support 是独立 hypothesis-driven task。

因此本轮只保证：

```text
raw -> processed -> Zarr
```

完整保存和自描述。

Real 现有未被 Policy 使用的 tactile runtime support 不需要扩展，也不需要为本任务删除。

---

# 6. P0 — Legacy v26 contact/dense validity conservative fix

## 6.1 已确认事实

Frozen v25→v26 converter 明确说明：

```text
legacy v25 collapsed aggregate/dense freshness
v26 tactile_sum_fresh = old tactile_fresh
```

这是保守 proxy，不是 current v28 独立 aggregate provenance。

当前 v18 processing 对 v26：

```python
contact_force_valid = finite(contact)
```

会把历史 finite-zero placeholder 误认为真实 no-contact。

Dense tactile 当前 validity 同样没有使用 legacy `tactile_fresh`，也存在同类 over-claim 风险。

## 6.2 最小规则

### Current raw

保持当前 independent-source 逻辑：

```python
contact_force_valid = finite(contact) & (contact_source_ns > 0)

tactile_force_valid = (
    finite(tactile)
    & calibrated
    & unit_is_native
    & (tactile_source_ns > 0)
)
```

`*_fresh` 单独作为 timing telemetry，不重新与 current validity 强耦合。

### Legacy v26

保守使用历史仅有证据：

```python
contact_force_valid = (
    finite(contact)
    & tactile_sum_fresh
)

tactile_force_valid = (
    finite(tactile)
    & tactile_fresh
    & calibrated
    & unit_is_native
    & (tactile_source_ns > 0)
)
```

这里允许 false-negative，不允许把 unknown/placeholder 宣称为 valid measurement。

### Important

不要：

```text
用 payload magnitude 推断 validity
把 zero 当 invalid（zero是真实可测值）
生成新的历史 source timestamp
跨行补 tactile/contact
修改原 v25 raw
```

## 6.3 文档语义

明确写：

```text
current v28 validity = direct current producer evidence
legacy v26 validity = conservative best-available proxy
```

两者都可用统一 `*_valid` mask 消费，但历史 mask 不能宣传为与 current provenance 等价。

---

# 7. P0 — Processing technical failures must fail loudly

## 7.1 已确认问题

当前 `process_episode_root()` 捕获：

```text
FileNotFoundError
OSError
ValueError
KeyError
RuntimeError
IndexError
```

并转成 episode rejection。

但注释又声明 programming errors 应 fatal。

这会让：

```text
forgotten key
index bug
unexpected runtime algorithm error
```

被误包装成：

```text
bad episode skipped
```

对于个人 research dataset，这是比 batch 中断更危险的失败模式。

## 7.2 最简单修法：删除 broad catch

`analyze_episode()` 本身已经能返回 intentional `EpisodeDecision`：

```text
excluded by annotation
persistent IK failure
```

因此 `process_episode_root()` 不需要 broad exception-to-decision conversion。

建议删除：

```text
_ANALYSIS_REJECTION_EXCEPTIONS
try/except _ANALYSIS_REJECTION_EXCEPTIONS around episode analysis
```

结果：

```text
behavior-quality decision -> EpisodeDecision
technical/source/programming failure -> raise, fail the batch
```

这是最清晰的数据流。

已知坏 episode 若用户确实想忽略：

```text
annotation include:false
```

显式处理，而不是 silent exception swallowing。

### 不要新增

```text
EpisodeSourceError hierarchy
RecoverableDataError framework
error policy config
```

当前项目不需要。

## 7.3 Tests

把现有：

```text
technical corruption -> no output published
```

进一步 pin 成：

```text
missing key -> raises
IndexError injected -> raises
RuntimeError injected -> raises
persistent IK -> still EpisodeDecision reject
include:false -> still skip without source read
```

---

# 8. P0 — Rebuild and close `pick_place_toy` v18/v11 evidence

## 8.1 Fact-check

当前 README 已宣称：

```text
episodes_processed/salvage_v18/pick_place_toy
datasets/salvage_v11/pick_place_toy.zarr
```

但 tracked：

```text
artifacts/pick_place_toy_salvage_manifest.json
```

仍然记录：

```text
processed v17
Policy Zarr v10
salvage_v17
salvage_v10
```

因此当前 GitHub **不能证明 v18/v11 rebuild 已完成**。

## 8.2 执行顺序

必须在完成 §§3–7 后，用本地真实历史数据：

```text
original v25 raw (READ ONLY)
→ existing frozen v25→v26 normalization
→ final v18 multimodal processed
→ final v11 Zarr
```

不要通过修改 manifest 来“宣称完成”。

## 8.3 Hard gate

若 historical source 未变化，必须得到：

```text
source episodes = 61
accepted = 60
rejected = 1
rejected = episode_20260827_224527 / persistent IK
source frames = 14309
retained frames = 14112
Zarr episodes = 60
accepted_episode_lost_rows = false
original_raw_modified = false
```

并额外统计（仅 aggregate counts，避免 manifest 变成 quality framework）：

```text
contact_force_invalid_rows
tactile_force_invalid_rows
```

这些 count 只证明 validity propagation，不参与 episode rejection。

如果出现额外 rejected episode：

```text
STOP
root-cause
```

不要 hardcode filename exception。

## 8.4 Manifest

真实 rebuild 通过后更新：

```text
artifacts/pick_place_toy_salvage_manifest.json
```

至少修正：

```text
processed_root -> salvage_v18
zarr_path -> salvage_v11
processed_schema_version -> 18
zarr_schema_version -> 11
new validity counts / verification
```

仍不记录 machine-specific absolute path。

---

# 9. P1 — Mechanical cleanup

只在历史 hard gate 通过后做。

## 9.1 Episode discovery

当前 task root 会接受所有 non-hidden subdirectory。

改成 ManiUniCon 同样的显式 convention：

```python
children = sorted(
    child for child in root.iterdir()
    if child.is_dir() and child.name.startswith("episode_")
)
```

**不要要求 `(child / "data.h5").is_file()` 才 discover**，否则一个损坏的 `episode_*` 会被静默跳过；应让后续 open fail loudly。

单 episode root `(root/data.h5 exists)` 的现有 direct path 保留。

## 9.2 Stale current-version names

修当前行为相关文本：

```text
dexmani_real/dataset/pointcloud.py: Raw-v27 -> current raw / raw v28
examples/process_episodes.py: processed v17 -> v18
examples/visualize_episode_processed.py: v17/v27 -> current v18/v28
xhand_tactile_correctness_upgrade_guide.md 的 current-schema note -> v18/v11
```

历史 incident 正文中的旧版本号可以保留，只需明确 Historical/Superseded。

## 9.3 Policy test cleanup

将：

```text
../dexmani_policy/tests/test_deployment_zarr_v10.py
TestPolicyZarrV10Boundary
```

改为不绑定废弃整数的名字，例如：

```text
tests/test_deployment_real_zarr.py
TestRealZarrBoundary
```

删除 fixture 中 production 已不存在的 `profile` vocabulary；直接按每个 test 需要创建字段。

测试 `schema_version is informational` 可以保留，但 docstring 不再写“schema-version gate”。

## 9.4 Plan docs

`control_step_dataset_simplification_plan.md` 和 `multimodal_research_dataset_finalization_plan.md` 是历史 execution plans，不要逐段改写。

在顶部加清楚 banner：

```text
Implementation status: superseded by current v18/v11 implementation.
For remaining last-mile fixes see multimodal_dataset_last_mile_repair_plan.md.
```

避免未来 Claude/Codex 把历史计划当 current contract。

---

# 10. Warm-start fact-check：不改代码

当前：

```text
training first sample:
obs = [o0, o0]
canonical predicted control slice starts at action row0
```

Real inference first observation可发生在 `run_started_ns` 之后；prediction 仍以 logical control grid 为基准，executor 的：

```text
first_future_step_index()
```

会跳过已经来不及执行的 action prefix。

这与现有 full-future-chunk deployment design 一致：

```text
Policy predicts canonical future chunk
Real owns target timestamps / stale-prefix filtering
```

因此：

```text
DO NOT shift logical_step_ns merely to force action[0] execution
DO NOT remove first_future_step_index()
DO NOT special-case first inference chunk
```

只需把文档中的 causality 表述写精确：

```text
real logical slots:
    selected sensor source <= logical reference

leading warm-start padding slots:
    repeat the first complete post-run observation;
    preserve its real source timestamp;
    padding slots are sampling slots, not physical recorded rows

all observations:
    real source <= actual inference causal anchor
```

---

# 11. `point_cloud_valid`：本轮不加

当前 offline canonical pointcloud derivation若返回 `None` 会 fail processing。

这确实让 derived point cloud 成为 multimodal dataset 的 technical requirement，但现在：

```text
DP3/R3D/ManiFlow等现有 consumer没有 point_cloud_valid mask semantics
```

直接加 mask 会继续扩散到 sampler/model/deployment。

因此本轮：

```text
KEEP current hard requirement
```

只有真实新数据证明：

```text
RGB-D structurally good but isolated point-cloud empty频繁发生
```

再单独立项研究 `point_cloud_valid`。

不要为了理论 future-proof 提前增加机制。

---

# 12. Schema/version policy

本轮所有改动都是：

```text
semantic metadata completion
small scalar telemetry addition
legacy validity fix
consumer compatibility fix
```

考虑项目是个人研究仓库，且 tracked manifest 尚未证明 final v18/v11 artifact，**不要再 bump 到 processed v19 / Zarr v12**。

继续使用：

```text
raw v28
processed v18
Policy Zarr v11
```

将 v18/v11 视为当前最终化中的 canonical version，并重新生成最终 artifact。

Policy 继续不做 exact-version gate。

---

# 13. Implementation order

严格按以下顺序：

```text
Phase 0  refresh HEAD + source inventory
Phase 1  pointcloud processing_config_json contract
Phase 2  fingertip_config_json contract
Phase 3  tactile attrs + 4 cheap telemetry arrays + legacy validity rules
Phase 4  remove broad processing exception swallowing
Phase 5  focused Real + Policy tests
Phase 6  rebuild pick_place_toy v18/v11 HARD GATE
Phase 7  mechanical cleanup/docs/test rename/episode_*
Phase 8  full offline regression + adversarial source review
```

不要先清 legacy path，再做 historical rebuild。

---

# 14. Expected files

## dexmani_real

主要：

```text
dexmani_real/dataset/processed.py
dexmani_real/dataset/processing.py
dexmani_real/dataset/export.py
dexmani_real/deployment/config.py
```

测试：

```text
tests/test_control_step_dataset.py
tests/test_control_step_export.py
tests/test_observation_builder.py / deployment compatibility tests
legacy processing focused tests
```

机械：

```text
dexmani_real/dataset/pointcloud.py
examples/process_episodes.py
examples/visualize_episode_processed.py
docs/* current-status banners
artifacts/pick_place_toy_salvage_manifest.json  # ONLY after real rebuild
```

## ../dexmani_policy

功能修改应非常小：

```text
dexmani_policy/deployment/export.py
```

只为现有 supported fields：

```text
point_cloud processing_config_json
fingertip config_json
```

传播 semantics。

不要修改：

```text
agents/
datasets/
normalizers/
training/
Policy tactile support
```

除非 compile/test 证明直接依赖。

---

# 15. Required tests

## A. Point cloud

```text
same full numeric config -> pass
voxel mismatch -> fail
workspace mismatch -> fail
outlier mismatch -> fail
```

## B. Fingertip

```text
same mount/link config -> pass
position mismatch -> fail
quaternion mismatch -> fail
link order mismatch -> fail
```

## C. Tactile/current

```text
current valid contact + dense invalid remain independent
current dense telemetry copied exactly
invalid dense payload remains NaN, never zero-filled valid
```

## D. Tactile/legacy

```text
v26 finite contact + tactile_sum_fresh=false -> contact_force_valid=false
v26 finite contact + tactile_sum_fresh=true -> contact_force_valid=true
v26 finite dense + tactile_fresh=false -> tactile_force_valid=false
zero payload is not intrinsically invalid; mask depends on provenance flags
```

## E. Processing failure

```text
persistent IK -> intentional EpisodeDecision reject
include:false -> skip
missing/corrupt source field -> raises / batch fails
injected RuntimeError -> propagates
injected IndexError -> propagates
```

## F. Historical hard gate

```text
61 source
60 accepted
1 persistent IK
14112 retained
60 Zarr episodes
0 accepted row loss
0 original raw modification
```

---

# 16. Static cleanup proof

Real：

```bash
rg -n "processed HDF5 v17|processed v17|Policy Zarr v10" \
  README.md repo_map.md examples dexmani_real docs

rg -n "Raw-v27|raw v27" dexmani_real examples docs

rg -n "_ANALYSIS_REJECTION_EXCEPTIONS" dexmani_real tests

rg -n "OutputProfile|needs_rgb|needs_pointcloud|--profile" \
  dexmani_real examples tests
```

Policy：

```bash
rg -n "TestPolicyZarrV10Boundary|test_deployment_zarr_v10|schema-version gate" \
  ../dexmani_policy

rg -n "tactile_force" ../dexmani_policy/dexmani_policy
```

最后一条预计仍为空；这正是本轮**不扩 tactile Policy capability**的设计结果。

历史文档命中旧版本允许存在，但必须有明确 Historical/Superseded context。

---

# 17. Verification

禁止自动硬件执行。

Real：

```bash
python -m compileall -q dexmani_real examples tools
pytest <focused tests>
pytest
git diff --check
git status --short
```

Policy：

```bash
cd ../dexmani_policy
python -m compileall -q dexmani_policy
pytest <focused deployment tests>
pytest <applicable offline suite>
git diff --check
git status --short
cd ../dexmani_real
```

如果 full suite 受 CUDA/pretrained/sim external dependency 阻塞，明确报告，不能写成 passed。

Hardware validation：

```text
NOT RUN
```

---

# 18. Final acceptance

必须全部成立：

```text
[ ] point-cloud training numeric config == deployment runtime numeric config
[ ] fingertip stored derivation geometry == deployment runtime geometry when field is requested
[ ] dense tactile artifact is self-describing
[ ] tactile/contact freshness/calibration/unit telemetry survives raw -> processed -> Zarr
[ ] current contact/dense validity remain independent
[ ] legacy invalid/unknown finite-zero payload is never promoted to valid measurement
[ ] technical processing/programming errors fail loudly
[ ] persistent IK remains the intentional automatic behavior rejection
[ ] final pick_place_toy evidence is real v18/v11, not copied v17/v10 metadata
[ ] 60/61 and 14112 historical baseline remains unchanged if source is unchanged
[ ] episode discovery only treats episode_* directories as episode candidates
[ ] stale current-version names/tests are cleaned
[ ] warm-start/full-future scheduler is unchanged
[ ] no point_cloud_valid framework added
[ ] no Policy tactile model/deployment capability added
[ ] no schema registry/hash/version framework added
[ ] no runtime safety gate weakened
```

---

# 19. Stop condition

完成以上内容后，停止 dataset infrastructure 重构。

后续新研究需求按具体 hypothesis 单独进入 Policy，例如：

```text
tactile encoder / visuotactile fusion
point-cloud invalid-mask learning
new 3D preprocessing
multi-camera policy
```

不要预先在数据基础设施中为未知研究方向建立抽象。

最终 mental model 保持：

```text
Record facts.
Preserve useful modalities.
Represent uncertainty explicitly.
Derive once where useful.
Validate only physics-changing contracts.
Let Policy choose what it consumes.
Fail loudly on technical corruption.
```
