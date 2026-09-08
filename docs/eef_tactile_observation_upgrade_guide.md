# EEF + Full Tactile Observation Upgrade Guide

> 历史实施记录：本文涉及的 raw v24 保持约束已由后续 [raw v25 contract](raw_v25_runtime_simplification_guide.md) 取代；当前 schema 与工作流见 [data_schema.md](data_schema.md)。

> Status: IMPLEMENTED in `4056e13631478a37a4477a018f4f94f220a527dd`.
> Original design baseline: `064a6fb1d5ab263e4d82e01197c8253fe258cf0f`.
>
> 本文是已实施升级的 historical implementation plan，保留原设计步骤供审计，
> 不是待执行任务或 current schema reference。当前合同以源码和
> [`data_schema.md`](data_schema.md) 为准；本升级未修改 `dexmani_policy`。
>
> 后续 targeted repair 收敛了 canonical anatomical order、部署 tactile semantic 校验、
> 真正的 metadata-only SHM projection、单次 full tactile snapshot，以及从 policy-visible
> float32 joint_state 派生 EEF/fingertips 和 canonical rot6d 验证。raw v24、processed v14、
> Policy Zarr v7 与既有 v14 required attrs 保持不变。

## 0. 开始前必须遵守

先阅读并遵守：

- `AGENTS.md`
- `CLAUDE.md`
- `code_style.md`
- `repo_map.md`

开始编辑前：

```bash
git status --short
```

保留所有无关用户修改。不要运行任何会连接 xArm7、XHand、RealSense、Quest/HTS 或发送机器人命令的代码。本任务应全部通过 offline/pure-function/schema tests 完成。

本文的目标不是做通用框架，而是在现有架构上完成一个最小、清晰、fail-closed 的 vertical change。

---

# 1. 目标与非目标

## 1.1 用户目标

在 Real 数据链中增加两个可用于后续 policy training/deployment 的 observation：

```text
eef_pose       [T, 9]          = position_m(3) + rot6d(6)
tactile_force  [T, 5, 120, 3]  = XHand SDK raw_force fx/fy/fz
```

同时满足：

1. 历史合法 raw v24 尽可能直接重新处理，不要求全部重录。
2. processed HDF5 中保存 train-time EEF/full tactile observation。
3. 真机 deployment 侧能以相同语义实时生成 `eef_pose` 与 `tactile_force`。
4. raw / processed visualizer 可显示 EEF position，视觉风格参考 `fingertip_points`，EEF 球更大。
5. 当前 `dexmani_policy` 与 Policy Zarr v7 契约不变。

## 1.2 明确非目标

本次不要：

- 修改 raw v24 recording schema。
- 修改 `dexmani_policy`。
- 升级 Policy Zarr v7。
- 改变 robot control / safety / learned-policy rollout semantics。
- 从 xArm firmware API 直接读取 Cartesian EEF pose。
- 假设 `contact_force == tactile_force.sum(axis=taxel)`。
- 假设 120 个 tactile point 已有 verified 2D/3D taxel geometry。
- 为 tactile 做未经证明的 offline calibration rescue。
- 为本任务增加新 multiprocessing / SHM channel。

---

# 2. Pre-upgrade 源码事实（原设计基线）

本节描述原设计基线，不代表当前尚未实现的功能。

## 2.1 Raw v24 已经包含全部源信息

`dexmani_real/recording/storage/schema.py` 当前为：

```text
EPISODE_SCHEMA_VERSION = 24
```

raw 已包含：

```text
arm_qpos              (N, 7)
arm_ee                (N, 9)

hand_contact          (N, 5, 3)
hand_tactile_force    (N, 5, 120, 3)

hand_source_monotonic_ns
tactile_source_monotonic_ns
tactile_fresh
tactile_calibrated
tactile_unit_code

policy_observation_arm_qpos
policy_observation_hand_qpos
policy_observation_reference_monotonic_ns
...
```

raw semantic contract 已定义：

```text
robot_world_frame = xarm_base
arm_ee_frame = xarm_base
tactile_unit = sdk_scaled_unknown_si
tactile_si_unit_verified = False
```

因此，本任务不存在“采集端缺少 EEF/full tactile”的问题。

## 2.2 EEF 是 qpos 的 canonical derived geometry

`dexmani_real/planning/kinematics/arm_fk.py`：

```text
arm_qpos [7]
  -> make_arm_fk()
  -> custom_eef_link
  -> eef_pos [3] + eef_rot6d [6]
```

`make_arm_fk()` 是 cached, URDF-consistent 的唯一 canonical factory。不要使用 xArm SDK 的 firmware Cartesian pose 作为 policy observation source。

## 2.3 XHand full tactile 的真实来源

`dexmani_real/robot/drivers/xhand.py`：

```text
sensor.calc_force               -> tactile_sum        [5,3]
sensor.raw_force[0:120].fx/fy/fz -> tactile_force     [5,120,3]
```

二者都乘 `_TACTILE_SCALE = 0.1`，并分别使用 software bias：

```text
_tactile_bias_sum
_tactile_bias_raw
```

因此：

```text
contact_force 与 tactile_force 来自同一次 sensor sample
```

但不能推导：

```text
contact_force == sum(tactile_force)
```

## 2.4 Pre-upgrade baseline was processed HDF5 v13

Current implementation is processed HDF5 v14。升级前 core：

```text
joint_state       [19]
action            [19]
action_ee         [21]
contact_force     [5,3]
fingertip_points  [5,3]
```

升级前缺失（现已实现）：

```text
eef_pose
tactile_force
```

## 2.5 Pre-upgrade deployment 尚不支持这两个 field

升级前 `deployment/config.py` 与 `deployment/inference/observation.py` 只接受：

```text
joint_state
point_cloud
rgb
contact_force
fingertip_points
```

但 deployment 已经具备所需 source：

- `arm_state_ring` 有实时 arm qpos history。
- `hand_tactile_ring` 已经有完整 tactile tensor。
- observation assembler 已有 causal camera/control-grid alignment。
- `fingertip_points` 已在 inference process 内通过 FK 实时构造。

所以不需要新增 IPC。

---

# 3. 核心设计不变量

实现过程中必须始终保持以下 invariants。

## 3.1 Raw v24 完全冻结

不要修改：

```text
dexmani_real/recording/storage/schema.py
dexmani_real/recording/frame.py
dexmani_real/recording/sample.py
dexmani_real/robot/hand_worker.py
```

除非在实现过程中发现独立 correctness bug；若发现，应停止本任务范围内的顺手修复，单独报告。

## 3.2 Processed EEF 必须来自 processed/aligned arm qpos

正确：

```text
processed joint_state[:, :7]
  -> canonical ArmFK
  -> eef_pose
```

错误：

```text
eef_pose = raw arm_ee[selected]
```

原因：visual profile 的 `joint_state` 使用 camera-aligned `policy_observation_arm_qpos`，而 raw `arm_ee[selected]` 是 control-grid raw state。直接复制会制造 silent timing mismatch。

## 3.3 Train / deploy 使用相同 EEF 定义

统一定义：

```text
eef_pose(t) = canonical_fk(causally_aligned_measured_arm_qpos(t))
```

不是：

```text
latest arm qpos regardless of observation reference
```

也不是：

```text
xArm SDK reported Cartesian pose
```

## 3.4 EEF 与 fingertip 共用同一次 Arm FK

每个 aligned timestep 只做一次 Arm FK：

```text
arm_qpos[t]
    |
    | ONE Arm FK
    v
 eef_pose[t]
    | \
    |  +------------------------> persist / policy eef_pose
    |
    + hand mount + HandFK(hand_qpos[t])
                         |
                         v
                  fingertip_points[t]
```

不要分别为 `eef_pose` 和 `fingertip_points` 重复计算 Arm FK。

## 3.5 Contact summary 与 full tactile 必须来自同一 raw tactile row

统一 source selector：

```text
raw tactile provenance
  -> selected_tactile_row[t]
      |- hand_contact        -> contact_force[t]
      `- hand_tactile_force  -> tactile_force[t]
```

必须满足：

```text
source_row(contact_force[t]) == source_row(tactile_force[t])
```

不要用两个独立 alignment 函数。

## 3.6 Tactile selector 只负责 provenance

selector 只判断：

```text
fresh
calibrated
unit_code == 0
source_ns > 0
hand_source_ns == tactile_source_ns
source_ns <= reference_ns
reference_ns - source_ns <= max_observation_skew
candidate row 不晚于当前 persisted target row
```

raw `EpisodeReader.require_valid()` 已负责 payload schema/finite contract；processed validator 再负责输出 finite contract。selector 不要再绑定 `(5,3)` payload shape。

## 3.7 不伪造 tactile spatial semantics

只声明源码可证明的语义：

```text
sensor order = XHand SDK sensor_data order
point order  = XHand SDK sensor_data[i].raw_force order
axis labels  = fx, fy, fz
```

已核实 anatomical finger order 为 thumb、index、middle、ring、pinky，见
`robot/model.py` 的 canonical sensor ID 映射。taxel adjacency 与 physical XYZ 仍未验证。

## 3.8 Policy Zarr v7 必须保持完全兼容

不仅 data keys 不变，root semantic attrs 也不能因为 processed v14 新字段而变化。

exporter 必须：

```text
完整验证 processed v14
  -> 显式投影既有 v7 keys + v7 attrs
  -> 写 Policy Zarr v7
```

不能继续让 `profile.dataset_keys` / 全量 processed semantics 自动决定 v7 输出。

---

# 4. Processed HDF5 v14 contract

将：

```text
PROCESSED_SCHEMA_VERSION = 13
```

升级为：

```text
PROCESSED_SCHEMA_VERSION = 14
```

## 4.1 Core datasets

所有 `OutputProfile` 的 robot/tactile core：

| dataset | tail shape | dtype | semantics |
|---|---:|---|---|
| `joint_state` | `(19,)` | `float32` | arm7 + hand12 |
| `eef_pose` | `(9,)` | `float32` | position_m3 + rot6d6, xarm_base |
| `action` | `(19,)` | `float32` | unchanged |
| `action_ee` | `(21,)` | `float32` | unchanged |
| `contact_force` | `(5,3)` | `float32` | XHand SDK `calc_force`, causally selected |
| `tactile_force` | `(5,120,3)` | `float32` | XHand SDK `raw_force`, same causal source row |
| `fingertip_points` | `(5,3)` | `float32` | xarm_base FK |

RGB / depth / camera K/T / point cloud 保持现状。

`tactile_force` 不加入 `_FRAME_CHUNKED_DATASETS`。沿用 numeric chunking 即可，避免大量 one-row HDF5 chunks。

## 4.2 新 provenance

在 `/provenance` 增加：

```text
tactile_source_row_index             (N,) int64
observation_reference_monotonic_ns   (N,) int64
tactile_source_monotonic_ns          (N,) int64
```

其中：

```text
source_row_index[t]
    = processed row 对应的 raw source/action row

tactile_source_row_index[t]
    = contact_force / tactile_force 实际采用的 raw tactile row
```

`source_row_index` 与 `tactile_source_row_index` 可以不同；合法 forward-fill 时后者可以更早。

### Required provenance checks

```text
0 <= tactile_source_row_index < source_frames
0 <= tactile_source_row_index <= source_row_index

observation_reference_monotonic_ns > 0
tactile_source_monotonic_ns > 0

tactile_source_monotonic_ns <= observation_reference_monotonic_ns
observation_reference_monotonic_ns - tactile_source_monotonic_ns
    <= max_observation_skew_s
```

不要要求 tactile source row 自身也出现在 processed kept rows 中。

## 4.3 EEF semantic attrs

在 owning module 定义稳定 constants，不要散落 magic strings。建议：

```text
eef_pose_frame          = xarm_base
eef_pose_components     = position_m(3)+rot6d(6)
eef_pose_derivation     = canonical_arm_fk_from_aligned_qpos
eef_pose_algorithm_id   = xarm7_custom_eef_pinocchio_fk_v1
```

## 4.4 Full tactile semantic attrs

只记录源码可证明内容：

```text
tactile_force_representation
    = xhand_sdk_raw_force_fx_fy_fz

tactile_force_sensor_order
    = xhand_sdk_sensor_data_order

tactile_force_point_order
    = xhand_sdk_sensor_data_raw_force_order

tactile_force_axis_labels
    = fx_fy_fz

tactile_force_unit
    = sdk_scaled_unknown_si

tactile_force_si_verified
    = False

tactile_force_spatial_geometry_verified
    = False

tactile_force_fresh_required
    = True

tactile_force_calibrated_required
    = True

tactile_force_unit_code
    = 0

tactile_force_causal_to_reference
    = True

tactile_force_hand_source_match_required
    = True
```

`xhand_sdk_sensor_data_order` 已唯一解析为 thumb、index、middle、ring、pinky。
无需给已有 processed v14 增加 required finger-order attr；部署 PolicySpec 则显式校验
`finger_order=thumb_index_mid_ring_pinky`，保留已序列化的 `mid` 拼写。

---

# 5. Geometry implementation

## 5.1 `planning/kinematics/arm_fk.py`

新增共享 history helper，供 processed 与 deployment 同时使用：

```python
def compute_eef_pose_history_xarm_base(
    arm_qpos: np.ndarray,
    *,
    arm_fk: ArmFK | None = None,
) -> np.ndarray:
    """Return finite [T,9] position+rot6d from aligned arm qpos."""
```

要求：

```text
input  [T,7]
output [T,9]
internal compute float64
caller 可在 storage/model boundary cast float32
```

复用 `ArmFK.compute()`，不要实现第二套 rotation conversion。

同时在该 module 定义 EEF semantic identity constants。

## 5.2 `planning/kinematics/fingertip.py`

把 history-level fingertip 计算放到 geometry owner，而不是继续只留在 `dataset/processing.py`。

建议增加：

```python
def compute_fingertip_history_xarm_base(
    arm_qpos: np.ndarray,
    hand_qpos: np.ndarray,
    *,
    hand_fk: HandKinematics,
    handbase_position_eef_m: np.ndarray,
    handbase_quat_eef_wxyz: np.ndarray,
    arm_fk: ArmFK | None = None,
    eef_pose_history: np.ndarray | None = None,
) -> np.ndarray:
    ...
```

规则：

- 若 `eef_pose_history` 已提供，则不要做 Arm FK。
- 若未提供，则要求 `arm_fk`，保持 standalone correctness。
- per-frame 继续调用已有 `compute_fingertip_points_xarm_base()`。

Processed 与 deployment 都使用这两个共享 history helpers。

---

# 6. Offline tactile alignment implementation

## 6.1 `dataset/clean.py`

将：

```python
align_tactile_sum_rows_to_references(...)
```

改名并收窄职责为：

```python
select_tactile_rows_to_references(
    hand_source_monotonic_ns,
    tactile_source_monotonic_ns,
    tactile_fresh,
    tactile_calibrated,
    tactile_unit_code,
    reference_monotonic_ns,
    *,
    max_observation_skew_s,
) -> np.ndarray
```

保留当前 Fenwick/prefix restriction 算法与行为：

- 不假设 source timestamps 单调。
- 后写入的 raw row 不能修复更早的 observation。
- 每个 reference 选 `latest valid source <= reference`。
- 不 interpolation。

删除 `contact_force` payload 参数以及 `(N,5,3)` shape/finite 检查。

### `analyze_episode()`

仍然通过 selector 得到 `tactile_source_rows`，并维持：

```text
tactile_valid
tactile_forward_fill
contact_force finite hard gate
```

不要为了 quality audit 把整段 `(N,5,120,3)` full tactile 加载到内存。raw v24 validator 已保证 fresh tactile rows 的 `hand_contact` 与 `hand_tactile_force` 都 finite。

---

# 7. `dataset/processing.py`

## 7.1 Robot geometry flow

在 `joint_state = _processed_joint_state(...)` 后：

```text
joint_state[:, :7]
  -> compute_eef_pose_history_xarm_base
  -> output[eef_pose]

joint_state[:, :7]
joint_state[:, 7:19]
precomputed eef_pose
  -> compute_fingertip_history_xarm_base
  -> output[fingertip_points]
```

这样每 row exactly one Arm FK。

## 7.2 Tactile gather

不要：

```python
all_tactile_force = reader.h5f["hand_tactile_force"][:]
```

先算 `selected_tactile_rows`，再只读取需要行。

由于 selected indices 可能重复（causal forward fill），不要直接依赖 h5py 对重复 fancy indices 的行为。增加一个很小的 file-local helper：

```python
def _gather_dataset_rows(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(indices, return_inverse=True)
    values = np.asarray(dataset[unique])
    return values[inverse]
```

然后：

```python
contact_force = _gather_dataset_rows(
    reader.h5f["hand_contact"],
    selected_tactile_rows,
).astype(np.float32, copy=False)

tactile_force = _gather_dataset_rows(
    reader.h5f["hand_tactile_force"],
    selected_tactile_rows,
).astype(np.float32, copy=False)
```

两个 output 必须使用同一个 `selected_tactile_rows`。

## 7.3 Write provenance

直接从 raw 写：

```text
tactile_source_row_index
observation_reference_monotonic_ns
    = reader.h5f[reference_key][selected]

tactile_source_monotonic_ns
    = reader.h5f[tactile_source_monotonic_ns][selected_tactile_rows]
```

不要重复保存可以由这些字段确定的 `tactile_forward_fill` bool。

---

# 8. `dataset/processed.py`

## 8.1 Schema

- `PROCESSED_SCHEMA_VERSION = 14`
- `_CORE_DATASET_SPECS` 加 `eef_pose` 与 `tactile_force`
- `_PROVENANCE_DATASETS` 加 3 个 tactile/reference proof fields
- `ProcessedProvenance` 同步增加 typed arrays

## 8.2 Payload validation

继续沿用 chunked scan。

对 `eef_pose`：

```text
shape [N,9]
dtype float32
all finite
validate_canonical_rot6d(block[:,3:9])
```

对 `tactile_force`：

```text
shape [N,5,120,3]
dtype float32
all finite
```

不要检查：

```text
tactile_force != 0
contact_force == tactile_force.sum(...)
```

## 8.3 Semantic validation

增加 EEF/full tactile semantic validator，和现有 fingertip semantic validator 同一层级。

Semantic validator 只检查 persisted identity；不要在每个 consumer 中重新运行 Pinocchio FK。

`eef_pose == FK(joint_state)` 属于 producer correctness，应由 focused unit test 固定，而不是每次读取 HDF5 都重复 FK。

## 8.4 Provenance validation

按第 4.2 节验证 causal relation。

从 root attrs 读取：

```text
source_frames
max_observation_skew_s
```

并检查每个 processed row 的 tactile source/reference proof。

---

# 9. Historical raw v24 reuse

不要新增独立 `raw24_rescue.py`。

现有：

```bash
python examples/process_episodes.py ... --dry-run
```

已经承担：

```text
raw schema validation
arm_sent_stream requirement
tactile causal validity
camera / policy observation validity
source contiguity
horizon admission
```

因此历史数据流程是：

```text
raw v24
  -> process_episodes.py --dry-run
  -> accepted episodes
  -> 用新 v14 processor 重新生成 processed HDF5
```

不要做：

```text
processed v13 -> in-place migrate -> v14
```

因为 v13 根本没有 full tactile。

## 9.1 预期结果

- current `EpisodeReader` 判 VALID、存在 `action_arm_joint_sent`、且有足够 causal calibrated tactile rows：可直接重新处理。
- `arm_sent_stream=False`：仍必须 fail closed，不要降级成 logical arm action。
- tactile invalid rows：继续通过 hard mask 丢弃；只要剩余 contiguous segment 满足 horizon，episode 仍可用。
- 若 raw full tactile 缺失或 raw 不再满足 current v24 contract，则不能凭空恢复 full tactile。

不要为了“救数据”弱化 existing action/tactile provenance gates。

---

# 10. Policy Zarr v7 compatibility boundary

本次 processed 升 v14，但 Policy Zarr 保持：

```text
POLICY_ZARR_SCHEMA_VERSION = 7
```

## 10.1 不能继续自动继承 processed core keys

当前 exporter 会用 `_Artifact.dataset_shapes` 决定：

```text
Zarr data keys
Zarr array shapes
data copy loop
validation expected keys
```

还会把 `_Artifact.semantic_attrs` 全量写入 Zarr root attrs。

所以必须显式定义 legacy projection。

建议：

```python
_POLICY_ZARR_V7_CORE_KEYS = (
    "joint_state",
    "action",
    "action_ee",
    "contact_force",
    "fingertip_points",
)
```

再按 profile 添加现有 RGB/point-cloud fields。

## 10.2 `_inspect_artifact()` 正确职责

顺序必须是：

```text
1. 验证完整 processed v14 structure/payload/semantics/provenance
2. 检查 task/profile uniformity
3. 只把 Zarr v7 projected shapes/dtypes/attrs 写入 _Artifact
```

也就是说，不能为了保持 v7 而跳过对 `eef_pose`/`tactile_force` 的 v14 validation。

## 10.3 Zarr v7 attrs 也必须冻结

不要把新：

```text
eef_pose_*
tactile_force_*
```

semantic attrs 放入 v7 root attrs。

`_Artifact.semantic_attrs` 应继续只代表 Zarr v7 semantics，或明确增加单独的 `zarr_semantic_attrs` 字段。不要复用一个 dict 同时代表 processed-v14 全语义与 zarr-v7 投影语义。

最终验收必须证明：

```text
同一 profile 下，修改前后的 Zarr v7 data key set 完全一致
修改前后的 Zarr v7 root semantic attr key set/values 完全一致
```

（除非 source task/data 本身不同。）

---

# 11. Real deployment: `eef_pose`

本次只增加 Real capability；当前 dexmani_policy 不请求该 field，因此 existing policy behavior 不变。

## 11.1 `deployment/config.py`

加入 supported field：

```text
eef_pose: ((9,), "float32")
```

增加 expected semantics，例如：

```text
representation = position_m_rot6d
frame = xarm_base
position_units = m
rotation_representation = rot6d
derivation = canonical_arm_fk_from_aligned_qpos
algorithm_id = xarm7_custom_eef_pinocchio_fk_v1
```

只在 PolicySpec 实际包含 `eef_pose` 时校验。

## 11.2 `deployment/inference/observation.py`

`eef_pose` 的 `(9,)` shape 由 builder/FK 投影构造；内部 `PolicyObservation` 不重复验证容器内容。

不要给 `ObservationBatch` 新增 EEF SHM history；它是 arm qpos 的 deterministic derived modality。

在 `_to_policy_observation()`：

```text
policy-visible joint_state(float32)[:, :7]
  -> shared compute_eef_pose_history_xarm_base
  -> float32 C-contiguous
  -> arrays["eef_pose"]
```

如果同一 policy 还请求 `fingertip_points`，必须把同一个预计算 `eef_pose_history` 传给 fingertip history helper，避免第二次 Arm FK。

## 11.3 Runtime ownership

不要把 EEF 写进 `ARM_STATE_DTYPE`。

正确 ownership：

```text
arm worker owns measured qpos
inference process owns causal alignment
inference process derives policy EEF from aligned qpos
```

这样不会出现两个独立 realtime state sources。

---

# 12. Real deployment: `tactile_force`

## 12.1 No new IPC

完整 tensor 已经在：

```text
shared.hand_tactile_ring
```

不要新增 SHM dtype/channel。

## 12.2 Preserve contact-only performance

原方案误判了 contact-only 的复制成本：旧 `get_last_k()` 仍复制完整 structured record，
包括 full tactile tensor。这是 pre-existing performance bug；后续 repair 使用
`get_last_k_fields()` 直接从 live record 只复制 source/fresh/calibrated/unit metadata，
并在复制前后验证同一 seqlock marker。

增加独立：

```python
_read_tactile_force_history(...)
```

仅当 PolicySpec 请求 `tactile_force` 时读取 `(5,120,3)` payload。

它使用与现有 tactile provenance reader 相同 gates：

```text
fresh
calibrated
unit_code == 0
not_before_ns <= source <= publish <= anchor
age <= max_age
```

返回 `FrameWindow(values=[T,5,120,3], ...)`。

不要为了减少几行重复代码而让 contact-only policy 每次复制 full tactile tensor。

## 12.3 Causal alignment

新 `hand_tactile_force_history` 必须和其他 history 一样：

- visual policy：对齐到 selected camera source timeline。
- non-visual policy：对齐到 control-grid references。
- 使用 `_align_state_history_to_reference_ns()` / camera wrapper。
- 不 interpolation。

若同时请求 `contact_force` 与 `tactile_force`，要求：

```text
hand_tactile_sum_history.source_monotonic_ns
== hand_tactile_force_history.source_monotonic_ns
```

否则 observation fail closed (`return None`)。

请求 full tactile 时，一次 full history snapshot 同时提供 payload 和 provenance proof，
不额外读取 metadata，`hand_tactile_provenance_history=None`。仅 contact-only 使用 metadata
projection，并对齐后比较 sum/provenance source identity。

## 12.4 `ObservationBatch`

增加：

```text
hand_tactile_force_history: FrameWindow | None
```

并纳入：

```text
causal-cut validation
history length validation
observation_timing_ms latest-source/skew calculation
```

## 12.5 `_to_policy_observation()`

若请求：

```text
tactile_force
```

要求存在 calibrated full tactile history，然后：

```python
arrays["tactile_force"] = np.ascontiguousarray(
    observation.hand_tactile_force_history.values,
    dtype=np.float32,
)
```

`PolicyObservation` expected tail：

```text
(5,120,3)
```

## 12.6 Lifecycle

`deployment/lifecycle.py::_requires_hand_sensor()` 增加 `tactile_force`。

不要因为 `eef_pose` 启动额外 hand sensor；EEF 只依赖 arm qpos。

---

# 13. Raw / processed visualization

目标是实现用户要求的 EEF sphere visualization，不为 tactile 伪造 3D geometry。

## 13.1 统一视觉规范

```text
point cloud radius  = 0.003 m
fingertip radius    = 0.012 m
EEF radius          = 0.020 m
```

EEF 使用固定易区分颜色；不要复用五指颜色。

## 13.2 Raw visualizer

`examples/visualize_episode.py`：

- 数据源：raw `/arm_ee[:3]`。
- 新增 `_log_eef(step_idx)`。
- 3D view admission 改为：

```text
pointcloud enabled
OR hand_fingertip present
OR arm_ee present
```

raw `arm_ee` 允许 all-NaN sentinel；若当前 row invalid，必须：

```python
rr.log("eef", rr.Clear(recursive=False))
```

不要只 `return`，否则 Rerun 可能继续显示上一帧 stale sphere。

raw viewer 的职责是显示 raw 实际记录值，所以不要在这里重新 FK 替代 `/arm_ee`。

## 13.3 Processed visualizer

`examples/visualize_episode_processed.py`：

- preload 小型 `eef_pose [T,9]`。
- 3D 位置显示 `eef_pose[:, :3]`，radius `0.020`。
- 增加 `eef_pose` time-series labels：

```text
ee_x ee_y ee_z ee_r0 ... ee_r5
```

不要 preload full `tactile_force` 到 RAM；若 `--info` 需要统计，使用 chunked scan。

## 13.4 不做 tactile 3D overlay

当前没有 verified：

```text
raw_force point index -> physical taxel XYZ/UV
```

所以不要把 120 个 point 画到手指表面。

---

# 14. Documentation updates after code

完成实现后同步：

- `docs/data_schema.md`
  - raw v24 保持不变
  - processed v14 新字段/provenance/semantics
  - raw v24 -> processed v14 -> Zarr v7 projection
- `README.md`
  - processed schema version / supported offline fields
- `repo_map.md`
  - 若新增测试/文件或 module primary responsibility 变化
- `examples/process_episodes.py`
  - v13 文案 -> v14
- `examples/visualize_episode_processed.py`
  - v13 文案 -> v14

不要修改 `AGENTS.md` / `CLAUDE.md`，本任务不改变 repository-wide agent contract。

---

# 15. Focused tests

新增或扩展 focused offline tests。不要用 hardware scripts 当测试。

至少覆盖以下行为。

## 15.1 Geometry

```text
compute_eef_pose_history_xarm_base:
  input shape validation
  finite validation
  output [T,9]

processed eef_pose == FK(processed joint_state[:,:7])

new fingertip history == old per-frame implementation numerically

当 eef_pose + fingertip 同时计算时：
  每个 timestep 只调用一次 ArmFK.compute
```

可以用 counting fake ArmFK 证明调用次数，不需要 Pinocchio benchmark。

## 15.2 Tactile selector

覆盖：

```text
latest source <= reference
skew bound
fresh false reject
calibrated false reject
unit_code != 0 reject
hand_source != tactile_source reject
non-monotonic source timestamps
later persisted row cannot repair earlier observation
forward-fill duplicate source row
```

## 15.3 HDF5 gather

验证：

```text
indices 有重复
indices 非连续
unique+inverse gather 恢复正确原顺序
contact/full tactile 使用同一 source row
```

## 15.4 Processed v14

验证：

```text
required keys
shape/dtype
EEF canonical rot6d
tactile finite
new provenance shape/dtype/causality
invalid tactile reference/skew fails closed
```

## 15.5 Historical raw24

用现有 fixture/fake episode（若 repo 无 fixture，则构造最小 HDF5 fixture）：

```text
valid v24 -> v14 accepted
missing action_arm_joint_sent -> still rejected
invalid tactile provenance -> rows rejected
```

不要降低现有 `unsafe fallback is disabled` 行为。

## 15.6 Zarr v7 compatibility

这是必须测试的 boundary：

```text
processed v14 可通过 exporter validation
Zarr schema_version 仍为 7
Zarr data keys 不包含 eef_pose/tactile_force
Zarr root attrs 不包含 eef_pose_*/tactile_force_*
现有 v7 keys/semantics 保持不变
```

## 15.7 Deployment

纯 fake ring / FrameWindow tests：

```text
eef_pose 来自 aligned arm_history，而非 latest uncontrolled arm state
visual camera alignment parity
nonvisual control-grid alignment parity

tactile_force only when requested
contact-only path 不读取/copy full tactile
tactile_force causal/calibrated/unit gates
contact + tactile_force source timestamps must match
builder 输出的 PolicyObservation shape/dtype/C-contiguous
```

## 15.8 Visualization helper

至少测试可提取的 pure helper：

```text
finite EEF -> position accepted
NaN raw EEF -> clear path
processed EEF shape validation
```

无需启动 Rerun GUI 做自动测试。

---

# 16. 推荐实施顺序

严格按以下顺序做，避免同时打开过多边界：

```text
Phase 1  shared geometry helpers
  arm_fk.py
  fingertip.py

Phase 2  tactile selector
  clean.py

Phase 3  processed v14
  contracts.py
  processing.py
  processed.py

Phase 4  Zarr v7 compatibility projection
  export.py

Phase 5  Real deployment capability
  deployment/config.py
  deployment/inference/observation.py
  deployment/inference/worker.py (only if imports/runtime wiring require)
  deployment/lifecycle.py

Phase 6  visualization
  visualize_episode.py
  visualize_episode_processed.py

Phase 7  docs + focused tests
```

每个 phase 后先跑最小相关测试，再继续下一 phase。

---

# 17. Acceptance criteria

完成任务必须同时满足：

## Raw

```text
raw schema_version == 24
raw dataset keys/semantics 未因本任务改变
recording/control behavior 未改变
```

## Processed geometry

```text
processed schema_version == 14

eef_pose.shape == [N,9]
eef_pose.dtype == float32
eef_pose == canonical FK(processed arm qpos)
eef_pose rot6d canonical

fingertip_points 使用相同 precomputed EEF
每 timestep exactly one Arm FK
```

## Processed tactile

```text
tactile_force.shape == [N,5,120,3]
tactile_force.dtype == float32
all finite

contact_force[t] 与 tactile_force[t]
来自同一个 tactile_source_row_index

source <= observation reference
skew <= max_observation_skew
fresh && calibrated && unit_code == 0
hand source == tactile source
```

## Historical data

```text
current-valid raw v24 可直接重新生成 v14
processed v13 不做伪 migration
missing arm sent stream 仍 fail closed
```

## Zarr

```text
Policy Zarr schema_version == 7
existing v7 data keys unchanged
existing v7 root semantic attrs unchanged
new processed-only fields 不泄漏到 v7
```

## Deployment

```text
现有 policy 无行为变化
Real 新增 latent support:
  eef_pose [T,9]
  tactile_force [T,5,120,3]

EEF 由 causally aligned measured arm qpos 通过同一 canonical FK 得到
full tactile 从现有 hand_tactile_ring causal 对齐得到
无新 IPC/SHM
```

## Visualization

```text
fingertip radius = 0.012 m
EEF radius = 0.020 m
raw arm_ee invalid row 不显示 stale EEF
processed eef_pose 可在 3D view 中显示
```

---

# 18. 禁止的“捷径”

Claude Code 不要为了让测试通过而做以下事情：

```text
- 修改 raw v24 schema
- 直接复制 raw arm_ee 到 visual processed eef_pose
- 用两个独立 tactile aligner
- 对 tactile 做 linear interpolation
- 用零填充替代 invalid tactile admission
- 使用 contact_force 反推 full tactile
- 假设 tactile_force.sum == contact_force
- 声称 120 taxel 有已验证 spatial geometry
- 将 eef_pose 持久化到 ARM_STATE_DTYPE 作为第二 realtime state source
- 为 contact-only policy 无条件读取 full tactile tensor
- 自动把 processed v14 新 fields/attrs 泄漏到 Zarr v7
- 为“救旧数据”回退到非 sent arm action
- 运行真实硬件验证但没有用户明确授权
```

---

# 19. Handoff checklist

提交前：

```bash
python -m compileall -q dexmani_real examples
git diff --check
git diff --stat
git status --short
```

然后运行新增的 focused offline tests，以及现有受影响 tests。

若本地有历史 raw v24 数据，可在确认 `examples/process_episodes.py` 仍是 offline-only 后执行一次：

```bash
python examples/process_episodes.py <episode-or-task-root> --dry-run
```

不要在没有用户明确授权的情况下运行：

```text
examples/run_policy.py
teleop
replay to hardware
calibration
任何会 connect/start robot/camera 的入口
```

最终 handoff 必须报告：

1. 修改了哪些 boundary。
2. 哪些 invariants 被 tests 覆盖。
3. 历史 raw24 dry-run 是否实际执行。
4. 是否执行过任何 hardware test（默认应为没有）。
5. 当前仍未修改 `dexmani_policy`，因此新 Real observation capability 只有在后续 PolicySpec 暴露对应 fields 后才会被实际 policy 请求。
