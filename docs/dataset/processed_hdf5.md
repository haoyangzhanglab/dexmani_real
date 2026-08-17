# DexMani Real 清洗与 Sim-label HDF5 格式

> 本文描述离线处理格式 `dexmani-real-simlabel-hdf5/v1`。源 Real v17
> episode 始终只读；输出是 real-domain 的 Sim-label 数据视图，不是可在
> SAPIEN 中回放的标准 Sim episode。

## 1. 使用方式

先比较不同模态对数据保留率和切段的影响：

```bash
conda run -n real_robot python examples/process_episodes.py \
  --input-root episodes \
  --output-root episode_processed \
  --compare-profiles
```

实际写入前可对选定 profile 做 dry-run：

```bash
conda run -n real_robot python examples/process_episodes.py \
  --input-root episodes \
  --output-root episode_processed \
  --profile rgb_pc \
  --dry-run
```

确认报告后执行：

```bash
conda run -n real_robot python examples/process_episodes.py \
  --input-root episodes \
  --output-root episode_processed \
  --profile rgb_pc
```

默认拒绝覆盖已存在的 `episode_processed`。实现会先在同一父目录写临时
目录，重新打开并验证全部 HDF5，再通过 fsync + rename 原子发布整批结果。
输入根目录下每个非隐藏的直接子目录都会被审计；缺少三件套或 schema 损坏的
目录会作为 rejected source 写入报告，不会被静默跳过。

## 2. 输出布局

只有一个连续有效段时保留源 episode stem：

```text
episode_processed/
├── episode_20260816_234045.h5
└── processing_index.json
```

中间存在硬无效帧时不能删行后拼紧，而是写成独立 episode：

```text
episode_processed/
├── episode_20260816_234045__seg000.h5
├── episode_20260816_234045__seg001.h5
└── processing_index.json
```

`processing_index.json` 保存源 episode 决策、丢弃原因、segment 范围、质量指标、
输出文件和写后验证结果。Sim 的 HDF5→Zarr 转换器只枚举顶层 `.h5/.hdf5`，
因此会自然忽略该 JSON。

## 3. Profile 与字段合同

Profile 参与 hard-valid mask；它不是单纯的“写哪些字段”选项。点云无效不会
破坏 joint-only 轨迹，但会切断 `pointcloud`/`rgb_pc` episode。

| Profile | 顶层 datasets |
|---|---|
| `joint` | `joint_state`, `action` |
| `rgb` | 上述字段 + `rgb`, `camera_intrinsic` |
| `pointcloud` | joint 字段 + `point_cloud` |
| `rgb_pc` | `joint_state`, `action`, `rgb`, `camera_intrinsic`, `point_cloud` |

同一个待聚合批次必须使用一个 profile，保证所有 HDF5 的 key、tail shape 和 dtype
一致。

### 3.1 `joint_state`

```text
concat(arm_qpos[7], hand_qpos[12]) → (N,19) float32 rad
```

它是控制网格锚点因果选择的实测状态。源数组是 `float64`；输出显式转换为 Sim
数据管线惯用的 `float32`。

### 3.2 `action`

```text
concat(action_arm_joint_sent[7], action_hand_joint[12])
→ (N,19) float32 rad
```

必须同时满足 `meta.arm_sent_stream=True` 和 sent dataset 存在。禁止回退到
`action_arm_joint`。arm 字段表示已转发给 worker 的命令，不是硬件 ACK；hand 字段
表示 queued target，没有 sent/ACK stream。

### 3.3 `rgb` 与 `camera_intrinsic`

- RGB 顺序解码 MP4，一次遍历直接写入被选择的 segment，不把整段视频缓存到内存。
- `(H,W,3)` 使用无 crop resize 输出 `(240,320,3) uint8`。
- 缩小时使用 OpenCV `INTER_AREA`，放大时使用 `INTER_LINEAR`。
- `camera_K` 按 `sx=target_w/source_w`、`sy=target_h/source_h` 调整
  `fx/cx` 与 `fy/cy`，输出 `(N,9) float32`。

K 缺失、非有限、非 pinhole 形式或与源 viewport 不一致时 fail closed，不生成单位阵。

### 3.4 `point_cloud`

点云在写段时从源 depth 逐帧确定性派生（`PointCloudProcessor`，无 RNG）：

```text
depth + K + 外参 + desk_plane + pc config → (2048,6) XYZRGB → (1024,6) float32
```

- 使用派生时的 `last_source_point_count` 识别唯一点前缀；
- 唯一点不少于目标数时做确定性 FPS；
- 唯一点不足时按固定顺序循环补点；
- RGB 保持 `[0,1]`；
- 不做额外 crop、voxel、坐标旋转或颜色缩放；
- XYZ 始终标为 `xarm_base`，`frame_compatibility_with_sim_world=False`。

全零、非有限、颜色越界或无真实源点的行不可输出（派生时顺带做数值校验）。

## 4. 明确省略的 Sim 字段

第一版不生成：

```text
depth
segmentation
camera_extrinsic
contact_force
fingertip_points
imagine_point_cloud
action_ee
done
```

省略比全零/末帧常量占位更安全：Sim 的全零 segmentation 表示 background，全零
contact force 表示无接触，`done=False` 表示 transition 未终止；它们都不表示 unknown。

## 5. 清洗和切段

### 5.1 Episode 级准入

- `EpisodeReader.require_valid()`；
- schema v17；
- 三个源成员和帧数一致；
- sent action stream 存在；
- 请求的模态完整；
- 至少一个 segment 满足训练窗口要求。

### 5.2 Core 行级硬无效

- 非 source sample；
- action 未 queued；
- held、安全拒绝或非 OK frame status；
- arm/hand 来源链、连接或手状态无效；
- action timing 非法；
- joint/action 非有限；
- arm 状态/动作超硬限位；
- hand action 超命令限位；
- hand 实测状态超额定机械包络加 3° 反馈容差。

手部反馈容差只适用于 measured qpos。命令仍使用严格 `1e-6 rad` 容差；不能以
伺服落点、接触回弹的正常小幅超调为由放宽 action。

### 5.3 模态硬无效

RGB 要求 camera fresh、causal source chain、无 clock reset，且
`camera_age_s <= 0.25`。Point cloud 还要求合法 `pointcloud_valid_depth_ratio`
（depth 派生质量）、正 source point count，以及派生数组 finite/nonzero/RGB-range 检查。

有效当前帧上的 `camera_frame_gap` 是软诊断，不单独删除。

### 5.4 连续性和短段

相邻 source index 不是 `+1`，或 timestamp delta 偏离 `grid_dt_s` 超过配置容差时，
只在两行之间建立边界，不删除两侧有效数据。任意硬无效行也会结束当前 segment。

默认：

```text
horizon = 16
min_full_windows = 1
min_segment_frames = horizon + min_full_windows - 1 = 16
```

短段不会写出。报告同时保留每段长度和完整训练窗口数。

## 6. 质量解释

技术有效性和任务结果分开：

```text
technical validity = schema + row gates + continuity + segment admission
task outcome        = success / failure / unknown
```

Recorder 的保存成功不等于任务成功。没有人工 annotation 时，输出
`task_outcome=unknown`，也不生成 `done`。

以下是软指标，不逐行删除数据：

- recorder `tracking_error` p50/p95/p99/max；
- action step、velocity、acceleration、jerk；
- idle step ratio；
- camera age/gap/duplicate；
- point count 和 valid-depth ratio。

`action[t]-joint_state[t]` 是 command delta，不是 post-action tracking error；idle 也
可能是稳定抓取阶段。两者均不能作为无任务上下文的自动删帧依据。

## 7. Annotation YAML

可选文件采用源 episode 名作为 key，range 为 source-grid 半开区间：

```yaml
episodes:
  episode_20260816_234045:
    include: true
    task_name: pick_bottle
    task_outcome: success
    include_ranges:
      - [20, 900]
    exclude_ranges:
      - [400, 420]
```

支持字段只有 `include/task_name/task_outcome/include_ranges/exclude_ranges`；未知字段
直接报错；布尔值不能写成字符串，range 边界必须是整数且不能超过源帧数。若一个
标注为成功的源 episode 被切成多段，每个 segment 的
`task_outcome` 仍为 `unknown`，仅 `source_task_outcome` 保留源级标注，避免把所有
片段错误标成成功。

## 8. HDF5 attributes

每个文件至少包含：

```text
schema_name = dexmani-real-simlabel-hdf5
schema_version = 1
domain = real
source_schema_version = 17
profile
episode_steps
dt
action_dim = 19
action_space = joint
obs_alignment = obs[t]_before_action[t]
source_episode
source_frame_start
source_frame_end_exclusive
task_name
task_outcome
processing_config_json
quality_summary_json
source_member_sha256_json
```

所有 datasets 使用 gzip level 4。写后 validator 检查精确 key 集合、shape、dtype、
首维、压缩、finite、点云颜色/非零帧、K、source range/segment、处理配置、三件套
SHA-256、schema 版本以及 real-domain/frame 标记。

## 9. 当前样例基线

`episode_20260816_234045` 在默认 horizon 16 下：

| Profile | 源帧 | 输出帧 | Segments | 完整窗口 |
|---|---:|---:|---:|---:|
| `joint` | 960 | 960 | 1 | 945 |
| `rgb_pc` | 960 | 939 | 13 | 744 |

RGB/点云版本删除 9 个开头空点云帧和 12 个中间无效点云帧；没有按全局
`observation_valid` 将数据误缩到 344 帧，也没有按历史损坏的 IK/retarget flags
删除 frame-status 为 OK 的数据。
