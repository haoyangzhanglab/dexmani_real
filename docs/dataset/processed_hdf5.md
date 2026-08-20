# DexMani Real processed HDF5 v3 与 Policy Zarr

本文定义 Real raw episode（schema v18，兼容 v17）的离线清洗、Real-native processed HDF5 和
dexmani_policy Zarr 合同。Sim 格式只用于参考键名；Real 与 Sim 数据不会混用，数值、
frame 和单位不要求相同。

## 1. 目录和一对一规则

```text
episodes/<task_name>/episode_xxx/
    data.h5 + depth.h5 + rgb.mp4

episodes_processed/<task_name>/
    episode_xxx.h5
    processing_index.json

dataset/<task_name>.zarr/
    data/*
    meta/episode_ends
```

- 一个 accepted raw episode 只产生一个同名 processed HDF5。
- 不产生 `__segNNN`，不把一条 demo 变成多个训练 episode。
- 所有模态共用一个 keep mask；删除行后保持原顺序并压紧。
- 失败或不可用 demo 由操作者在处理前删除；代码不判断任务成功，不维护
  `task_outcome` 或 `task_success`。
- 不使用 `inputs/` staging 目录。

## 2. 命令

推荐先 dry-run，检查每条轨迹的 retained rows、reason counts 和 bridge：

```bash
conda run -n real_robot python examples/process_episodes.py \
  --input-root episodes/pick_apple_messy \
  --profile rgb_pc \
  --dry-run

# 报告无 rejection 后再去掉 --dry-run 发布 HDF5。
conda run -n real_robot python examples/process_episodes.py \
  --input-root episodes/pick_apple_messy \
  --profile rgb_pc

conda run -n real_robot python examples/export_policy_zarr.py \
  --input-root episodes_processed/pick_apple_messy \
  --output dataset/pick_apple_messy.zarr \
  --task-name pick_apple_messy
```

处理器和导出器都拒绝覆盖已存在的目标。它们在目标同一父目录中写 staging，完成
重新打开、shape/dtype/finite/语义校验后再原子发布。

批处理中任何未被 annotation 显式排除的 raw episode 若校验或 bridge 策略拒绝，
整个批次都不发布；错误会列出 episode 与拒绝原因，不能静默漏掉坏轨迹。

## 3. Processed HDF5 v3

root attrs：

```text
schema_name = dexmani-real-processed-hdf5
schema_version = 3
domain = real
profile = joint | rgb | pointcloud | rgb_pc
episode_steps = M
source_frames = N
dt = 1 / control_hz
time_semantics = logical_control_grid_after_row_compaction
task_name = <task>
```

所有 profile 的核心逐帧 dataset：

| key | shape | dtype | Real 语义 |
|---|---:|---|---|
| `joint_state` | `(M,19)` | `float32` | `arm_qpos[7] + hand_qpos[12]` |
| `action` | `(M,19)` | `float32` | arm sent-to-worker target + queued XHand target；不是硬件 ACK |
| `action_ee` | `(M,21)` | `float32` | xArm-base EEF position/rot6d `[9]` + queued XHand target `[12]` |
| `contact_force` | `(M,5,3)` | `float32` | XHand per-finger SDK-scaled tactile summary；SI 未验证 |
| `fingertip_points` | `(M,5,3)` | `float32` | xArm-base FK fingertip，m |

`rgb` / `rgb_pc` 增加：

| key | shape | dtype | Real 语义 |
|---|---:|---|---|
| `rgb` | `(M,240,320,3)` | `uint8` | MP4 RGB，无 crop resize |
| `depth` | `(M,240,320)` | `uint16` | aligned Z16，nearest resize，0 invalid |
| `camera_intrinsic` | `(M,9)` | `float32` | resize 后 row-major K |
| `camera_extrinsic` | `(M,4,4)` | `float32` | `T_xarm_base_camera`，camera optical → xArm base |

`pointcloud` / `rgb_pc` 增加：

| key | shape | dtype | Real 语义 |
|---|---:|---|---|
| `point_cloud` | `(M,1024,6)` | `float32` | xArm-base XYZ[m] + RGB `[0,1]` |

eye-to-hand 相机逐行重复静态 `T_xarm_base_camera`；eye-in-hand 相机使用
`T_xarm_base_eef(obs[t]) @ T_eef_camera`。齐次矩阵、SO(3) 和 finite 都必须通过
validator。depth 的真实米制比例由 HDF5/Zarr 的 `depth_scale_m_per_unit` 声明，不能
默认套用 Sim 的 `1/1000`。

## 4. HDF5 provenance

provenance 只在 processed HDF5 中存在：

```text
provenance/source_row_index          int64[M]
provenance/source_sample_index       int64[M]
provenance/source_timestamp_s        float64[M]
provenance/source_keep_mask          bool[N]
provenance/source_drop_reason_bits   uint64[N]
```

`drop_reason_bit_names_json` 定义位到原因名的映射。`source_row_index` 必须严格递增并
等于 `flatnonzero(source_keep_mask)`；kept row 的 reason bits 必须为 0。

`processing_index.json` 是批处理审计报告，保存阈值、每条轨迹的 reason counts、
selected source ranges、bridge findings 和写后验证结果。它不会进入 Zarr。

## 5. 行清洗

硬无效条件包括：

- 非 source/invalid control-grid row；
- action 未 queued、held、frame status 非 OK 或 safety rejection；
- arm/hand source、history、timing、finite 或 joint limits 无效；
- `action_ee`、tactile summary 或 fingertip 非 finite；
- tactile 不 fresh、未校准或没有有效 source timestamp；
- camera profile 下 freshness、causal history、health、clock reset 或 age 无效；
- RGB profile 下 depth 全 invalid；
- point-cloud profile 下点云派生失败或 depth ratio 无效；
- annotation 显式排除的行。

时序检测分三种策略：

- `hard_only`：不运行停滞/抖动 detector；
- `audit`：只报告 suspect/high-confidence；
- `strict`：额外删除高置信度可逆 impulse、arm feedback stall 和 command-apply stall。

abrupt step 和 persistent tracking error 本身只作为 suspect，防止误删有意快速动作或
任务所需静止。

压紧后只按 `M >= horizon + min_full_windows - 1` 判断窗口数量，不按原连续段分别准入。

## 6. Bridge 风险

删除区间会把两端变成相邻训练帧。每个 bridge 记录：

```text
source rows、removed count/reasons、source sample/time delta、
max arm action delta、max hand action delta、risky
```

默认 `--bridge-policy reject`：若 source continuity 损坏，或压紧后 arm/hand action
超过配置的 abrupt threshold，整条 raw 被拒绝/要求人工复核。仍然不会切段。

人工确认逻辑时间压缩可接受后，可以显式使用：

```bash
--bridge-policy audit
```

该选择会写入 `processing_config_json` 和 index，不能静默放宽。

## 7. Annotation 边界

正常工作流中，任务失败或不可用 demo 应由操作者删除整个 raw episode。Annotation 仅用于
显式审计过的 row override、整条排除或兼容旧平铺数据：

```yaml
episodes:
  episode_20260818_192648:
    include: true
    task_name: pick_apple_messy
    include_ranges: [[0, 221]]
    exclude_ranges: [[40, 42]]
```

只允许 `include`、`task_name`、`include_ranges`、`exclude_ranges`；未知字段直接拒绝。
不存在 `task_outcome` 或 `task_success`。Range 使用 source-row 半开区间，所有模态共同
应用；annotation 删除行造成的 bridge 仍受默认 reject 策略约束。

## 8. 最小 Policy Zarr

Zarr v2 只包含：

```text
<task>.zarr/
├── data/<profile datasets>
└── meta/episode_ends
```

明确不包含：

```text
meta/task_success
dataset_manifest.json
source episode/file/path/hash
source groups/segments
processing_index
HDF5 provenance
```

root attrs 只保存 dataset-level schema 语义：domain、profile、task name、dt、depth
scale、camera extrinsic 方向和 tactile unit/frame。一个 processed HDF5 对应一个
`episode_ends` 条目；所有 data arrays 的首维必须等于最后一个 episode end。

当前 `dexmani_policy` 的 ReplayBuffer/BaseDataset 能直接读取这些 arrays；
`action_ee` 是 21D；`camera_extrinsic` 的 4×4 shape 可被其 RGB-D 数据路径读取，但
数值语义始终是 camera-to-xArm-base，不能忽略 root attr 后泛化成任意 Sim world。
若启用 depth 几何反投影，必须从 Real 数据配置实际 depth scale，而不是使用 Sim
常见默认值。

`RGBPCDataset` 默认读取 joint/RGB/depth/point-cloud/K/extrinsic；`contact_force`、
`fingertip_points` 和以 `action_ee` 为 action 需要在训练配置中显式选择。直接可读不代表
训练配置会自动使用所有导出的键。

## 9. 已核对目标 episode

`episode_20260818_192648` 有 221 个 raw rows。`rgb_pc` 清洗删除 3 个 camera-invalid
rows 和 6 个 held/IK-fail rows，压紧候选为一个 212-row HDF5。五个新增模态没有增加
丢帧：tactile 全程 fresh/calibrated，action EE、fingertip 和 depth 完整。

但 row 120→127 跨过 6 个 held rows 后，hand action 最大变化为约 `0.547 rad`，超过
默认 `0.2 rad` bridge threshold。因此默认 reject；只有人工接受该逻辑时间压缩后才
应以 `--bridge-policy audit` 发布。
