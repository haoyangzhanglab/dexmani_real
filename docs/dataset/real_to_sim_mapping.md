# Real v17 → Sim/Policy label 边界

本文定义“标签关联”和“数值转换”的边界。它不是一个自动转换脚本，也不证明 Real 与 Sim 的物理语义等价。

Real episode 合同见 [`hdf5_episode.md`](hdf5_episode.md)，Sim 容器见 [`sim_hdf5_zarr.md`](sim_hdf5_zarr.md)，实际数值处理见 [`processed_hdf5.md`](processed_hdf5.md)。

## 1. 两种操作必须分开

### label-only view

只登记来源关系，不改变数组值、shape、dtype、单位、frame、行序或时间轴。允许：

- `alias`：记录来源路径；
- `copy`：逐元素原样复制；
- `concat_view`：保留有序组件，且可以无损拆回；
- `reshape_view`：仅 C-order reshape，不转置、不换轴；
- `repeat_metadata`：逐行关联静态 metadata；
- `omit`：没有等价来源时明确省略。

### numeric processing

resize、dtype cast、单位缩放、坐标变换、FK/IK、点云派生、插值、切段和采样都属于数值处理，必须有独立版本和 provenance。它们不能被写成“label mapping”。

## 2. 13 个 Sim label 的 Real 来源

| Sim label | Real 候选 | 结论 |
|---|---|---|
| `rgb` | `rgb.mp4` 解码 | 可作为 Real-native RGB；分辨率和 dtype 以 Real episode 为准，不自动满足 Sim `(240,320,3)`。 |
| `depth` | `depth.h5` | raw unit、aligned Z 和 frame 语义不同；不能直接改名为 Sim depth。 |
| `segmentation` | 无 | 不用全零伪造；Sim 的 0 表示 background，不表示 unknown。 |
| `point_cloud` | 从 depth 派生 | 可生成 Real-native view；点数、滤波和坐标 frame 不等于 Sim world。 |
| `camera_intrinsic` | `/meta.camera_K` | 可重复 metadata；Real viewport、dtype 和 frame 必须单独声明。 |
| `camera_extrinsic` | `/meta.camera_T_world_camera` / `camera_T_eef_camera` | 方向、frame 和 shape 不同；不能直接 rename。 |
| `joint_state` | `arm_qpos + hand_qpos` | 唯一稳定的结构关系之一；保持组件顺序，dtype conversion 另记。 |
| `contact_force` | `hand_contact` | 不等价：Real 是 SDK-scaled tactile 汇总，Sim 是 world-frame physics force。 |
| `fingertip_points` | `hand_fingertip` | frame、模型和 shape 语义不同；不能直接 rename。 |
| `imagine_point_cloud` | 无 | 需要 Sim mesh/FK 派生，不能用 Real cloud 冒充。 |
| `action` | `action_arm_joint_sent + action_hand_joint` | 仅在 sent stream 存在时可作有序 Real-native joint action；不是硬件 ACK。 |
| `action_ee` | `action_arm_ee + action_hand_joint` | shape 相似但 frame 和 action 语义不同，不能直接 concat 后改名。 |
| `done` | 无 | Real `/meta.success`、`truncated`、recorder status 都不是 Sim transition outcome。 |

最小可信结构关系只有：

```text
joint_state ↔ [arm_qpos, hand_qpos]
action      ↔ [action_arm_joint_sent, action_hand_joint]
```

第二项必须满足 `/meta.arm_sent_stream=True` 且 `action_arm_joint_sent` 实际存在；缺失时禁止静默回退到 `action_arm_joint` 并继续称为 sent action。

## 3. Frame 和 action 边界

- Real 当前空间值使用 `xarm_base`；受支持 runtime 令 `world == xarm_base`。
- Sim 空间值使用 SAPIEN world；两者不能仅凭数值看起来相近就互换。
- Real `arm_ee`/`hand_fingertip` 是当前 robot frame 的状态；Sim `action_ee` 是由 Sim joint action 定义的 world-frame 表达，二者不是同一 observation/action 相位。
- `action_arm_joint_sent` 表示转发到 arm worker；`action_hand_joint` 是 queued target。二者都不能写成“hardware applied action”。
- Real raw action 的有效性由 reader mask 定义；held/failure 行中的兼容值不能被当作 solver raw。

## 4. Real-only 质量字段

这些字段用于 downstream admission/annotation，不属于 Sim label mapping：

```text
timestamp, source_sample_index,
flag_sample_valid, flag_observation_valid, flag_frame_status,
flag_action_queued, flag_held, flag_ik_ok, flag_retarget_ok,
arm_connected, arm_state_valid, hand_connected, hand_error_state,
flag_camera_fresh, camera_age_s, pointcloud_valid_depth_ratio,
tactile_fresh, tactile_calibrated,
action_arm_joint_raw_valid_mask, action_hand_joint_raw_valid_mask
```

映射阶段不删行、不压紧 episode、不猜 `done`。下游如果不能消费质量字段，应在 admission 层拒绝 episode 或创建有版本的 processed view。

## 5. 建议的 manifest

任何 label-only 或数值视图都应保存独立 manifest，至少包含：

```yaml
source:
  episode: episode_...
  schema_version: 17
  data_path: data.h5
target:
  domain: real
  representation: label_only  # or numeric_view
  profile: joint
mapping:
  joint_state: [arm_qpos, hand_qpos]
  action: [action_arm_joint_sent, action_hand_joint]
compatibility:
  frame_compatible_with_sim_world: false
  task_outcome_source: unknown
provenance:
  rule_version: 1
  source_config_sha256: ...
```

缺失的 task、scene、object、seed、done 或 contact semantics 写 `unknown`/`omit`，不要用零、NaN、末帧常量或 recorder status 补造。

## 6. 验收清单

- source 容器和 schema 已识别；
- 每个 target label 有明确 source 或明确 `omit`；
- 没有隐式 cast、缩放、frame 变换或插值；
- joint/action 的组件顺序和 action phase 已声明；
- Real frame 没有被标记成 Sim world；
- provenance、质量字段和未知值语义可追踪。
