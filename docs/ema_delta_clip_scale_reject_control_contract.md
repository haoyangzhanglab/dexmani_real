# EMA、Delta Clip、Scale 与 Reject 控制合同

本文整理 DexMani Real 中会改变控制目标、限制目标或拒绝目标的
EMA、delta clip、scale 和 reject 操作。内容以当前实现为准，目标是让每个操作的
位置、默认状态、输入输出和责任边界可以直接追溯。

## 1. 范围与术语

本文覆盖：

- VR 遥操作的机械臂和手部命令链；
- keyboard、camera calibration、replay 和 learned-policy deployment 的动作边界；
- 与控制语义相关的录制字段；
- 会改变物理量含义的传感器 scale。

本文不把以下操作视为控制策略：

- `arccos` 前将余弦裁到 `[-1, 1]` 等数值域保护；
- 可视化色值裁剪；
- 数组 shape、finite、dtype 等普通输入校验。

术语定义：

| 术语 | 本文含义 |
|---|---|
| EMA / filter | 使用历史状态改变当前目标；alpha 表示新值权重 |
| delta clip | 将本次命令相对参考命令的变化量截断到上限 |
| limit clip | 将目标投影到固定上下界，改变原目标 |
| reject | 不改变非法目标，而是拒绝整条候选并 hold、丢弃或进入 fault |
| scale | 映射增益、几何比例、单位换算或归一化；不同类别不共享一个抽象 |

## 2. 默认行为快照

| 操作 | 默认值 | 默认状态 | 来源 |
|---|---:|---|---|
| EEF 位置 EMA | `alpha_pos=0.6` | 启用 | `config/defaults.py::EMAParams` |
| EEF 旋转 EMA | `alpha_rot=0.25` | 启用 | `config/defaults.py::EMAParams` |
| VR 位置/旋转 scale | `1.0 / 1.0` | 数值直通 | `config/defaults.py::VRMappingParams` |
| VR 单帧旋转 clip | `0.52 rad/frame` | 启用 | `teleop/arm_mapper.py::ArmWristMapper` |
| VR reset-relative 总旋转 clip | `3.0 rad` | 启用 | `config/defaults.py::VRMappingParams` |
| arm 关节 delta clip | `120 deg/s ÷ 16 Hz = 7.5 deg/tick` | 启用 | `teleop/loop.py` |
| hand retargeter | `tag` | 启用 | `config/defaults.py::PolicyParams` |
| TAG pinch EMA | `0.4` | 启用 | `config/defaults.py::TAGRetargetingParams` |
| hand 启动 ramp | `0.5 s` | 启用 | `config/defaults.py::PolicyParams` |
| hand delta clip/reject | `None` | 关闭 | `config/defaults.py::HandParams` |
| DexPilot 内部低通 | `0.6` | 选择 DexPilot 时启用 | `assets/retargeting/xhand_right_dexpilot.yml` |
| DexPilot 外层输出 EMA | `1.0` | 直通、关闭 | 同上 |
| pointcloud temporal EMA | 不存在 | 已删除 | 使用 median、speckle 等单帧空间滤波 |

## 3. VR 机械臂命令链

当前顺序为：

```text
VR wrist pose
  -> 输入 shape/finite/quaternion reject
  -> 单帧旋转 delta clip
  -> position scale / rotation scale
  -> reset-relative 总旋转 clip
  -> Cartesian pose EMA
  -> workspace position clip
  -> contact-stall re-anchor/hold
  -> IK candidate reject/rank/canonicalize
  -> final IK joint-limit/pose/collision reject
  -> arm joint delta clip
  -> application joint-limit clip
  -> coordinated preflight
  -> SafetyGate
  -> arm worker generation/expiry/command validation
  -> xArm Mode 6 firmware
```

### 3.1 映射、scale 和旋转 clip

`ArmWristMapper` 先以原始 wrist orientation 为相邻帧基准限制旋转突刺，再对
reset-relative 位移和旋转应用 `pos_scale`、`rot_scale`。总旋转 clip 限制的是相对
reset 姿态的累计旋转，不是速度限制。

单帧门限 `max_per_frame_rot_rad=0.52` 当前是 mapper 构造默认值，没有进入 runtime
配置。它属于输入异常门，而不是 xArm 关节速度限制。

### 3.2 Cartesian EMA 与 workspace clip

EEF 目标在 IK 前使用位置和旋转两个独立 alpha 做 EMA。位置随后投影到配置的
workspace box；旋转没有 workspace clip。`target_eef_pos_raw`、
`target_pos_before_clamp` 和 `action_arm_ee` 分别记录映射后、workspace clip 前和
IK 实际追踪的阶段。

### 3.3 IK reject 与关节端点整形

IK 会拒绝非法候选、过大跳变、超限、位姿误差和碰撞结果。成功结果之后仍存在两步
应用层整形：

1. 以最后已发布命令为参考执行逐 tick joint delta clip；
2. 将结果投影到 arm joint limits。

第二步通常是 no-op：IK 的最终 limit reject 已允许 `1e-5 rad` 数值容差，因此该
clip 主要消除容差内的微小越界。第一步是真正会持续改变 IK 端点的 stateful slew
limit。

“SafetyGate velocity envelope 已删除”只能证明 gate 不再按速度拒绝候选，不能单独
证明上游 delta clip 必须存在或必须删除。是否继续保留该 clip 是控制所有权决策，
不应被表述为已经确认的合同违规。

### 3.4 两次 workspace 检查

teleop coordinated preflight 和 SafetyGate 都调用 planner workspace segment check。
两次检查使用的反馈读取时点可能不同，并且前者负责 arm/hand 协调 hold，后者负责
发布前的最终 fail-closed 校验。因此它们使用了相同 predicate，但不能在不保留 hold
语义的情况下直接删除前者。

## 4. VR 手部命令链

TAG 和 DexPilot 必须分开描述；把二者合并为一条“手部 EMA”链会产生错误结论。

### 4.1 默认 TAG 路径

```text
VR landmarks
  -> landmark shape/finite/geometry reject
  -> operator-to-MANO transform
  -> adaptive pinky geometry scale
  -> fingertip/robot geometry scale
  -> Stage 1 temporal regularization
  -> pinch-factor EMA
  -> Stage 2 regularization and bound projection
  -> SDK joint order output
  -> startup smoothstep ramp
  -> optional command delta clip (default off)
  -> operational command-box clip
  -> controller reject-whole preflight
  -> SafetyGate
  -> hand worker reject
  -> XHand driver reject
```

TAG 的 `smooth_weight` 是优化目标中的时序正则，不是 EMA。`pinch_ema_alpha` 只平滑
pinch activation factor，也不是整个 joint command 的输出 EMA。

### 4.2 可选 DexPilot 路径

DexPilot 的 `SeqRetargeting` 内部低通 alpha 为 `0.6`。代码还保留一层资源控制的
teleoperator output EMA，但 bundled config 的 alpha 为 `1.0`，因此当前直通。选择
DexPilot 时应把内部低通和外层 EMA 分别记录，不能与 TAG 的优化正则混为一谈。

### 4.3 Clip 与 reject 的分工

VR teleop 是唯一主动把 hand command 投影进 operational box 的主要路径；可选
`max_delta_rad` 启用时，它也先 clip delta。随后 controller preflight、SafetyGate、
worker 和 driver 都采用 reject-whole，不把非法端点转换成另一个端点。

keyboard、replay、calibration、home 和 learned-policy 等耦合路径依赖共享 hand
preflight 来拒绝 operational、rated mechanical 和可选 delta 违规，避免 arm 已发布而
hand 被后级拒绝。

## 5. 其他动作路径

### 5.1 Keyboard 与 camera calibration

两条路径都将按键输入形成固定 Cartesian delta，再做 workspace clip、IK reject 和
SafetyGate。keyboard 额外限制目标相对 measured pose 的位置/旋转 lead。

camera calibration 原有 `target_lead_max_m` 从未被读取，现已删除；标定路径当前没有
keyboard 的 measured-lead cap，不能在文档中声称二者完全等价。

### 5.2 Replay

Replay 的离线 dense preflight 会对相邻关节端点插值并检查 limits、workspace 和
collision。它是执行前的路径 reject，不是运行时 command clip，也不会改变录制端点。

### 5.3 Learned-policy deployment

Coordinator 对 arm/hand proposal 执行 shape、finite、joint limits、hand mechanical/
delta 和 SafetyGate 检查。当前构造的 SafetyGate 没有安装 planner workspace callback，
所以 deployment 不具备与 VR teleop 相同的应用层 workspace segment check。

是否给 deployment 增加 planner/workspace 依赖属于待决设计项，不是当前实现事实。

## 6. Scale 分类

不同 scale 的单位、输入和失败语义不同，不应设计一个通用 `scale()` 配置：

| 类别 | 示例 | 语义 |
|---|---|---|
| 控制映射增益 | VR `pos_scale`、`rot_scale` | 人体运动到 EEF 目标的增益 |
| 手部几何比例 | adaptive pinky、DexPilot `scaling_factor`、TAG finger ratio | 人手 landmark 到机器人几何的匹配 |
| 传感器单位换算 | RealSense `depth_scale`、XHand tactile `0.1` | SDK 原始量到项目内部数值；触觉不得据此宣称为 SI force |
| 规划归一化 | joint range normalization、candidate weights | 仅影响评分或数值条件，不直接代表物理增益 |

每个 scale 应在名称或相邻文档中说明输入单位、输出单位、默认值和是否改变控制目标。

## 7. Reject 边界与失败结果

| 边界 | 典型 reject | 失败结果 |
|---|---|---|
| Mapper/retargeter | malformed、non-finite、退化 landmarks、跟踪异常 | 保持上一安全目标或不产生候选 |
| IK/planner | 无解、jump、limit、pose error、collision、workspace path | arm hold；允许合法 hand-only motion 的分支除外 |
| Coupled preflight | hand operational/mechanical/delta、协调 workspace | arm/hand 协调 hold 或 abort |
| SafetyGate | candidate 良构、joint limit、可选 workspace | 拒绝整个候选，不裁剪 |
| Worker/driver | generation、expiry、连接、limit、可选 hand delta | 丢弃或拒绝，不修改端点；硬件/IPC 故障按 supervisor 合同处理 |
| Firmware | xArm velocity、acceleration、collision 等硬件限制 | 最终硬件 backstop，不替代应用层输入和数据质量检查 |

## 8. 录制可观测性

HDF5 v16 已提供以下阶段：

- `target_eef_pos_raw` / `target_eef_rot6d_raw`：VR mapping 后、Cartesian EMA 前；
- `target_pos_before_clamp`：Cartesian EMA 后、workspace clip 前；
- `action_arm_joint_raw`：IK 原始解；
- `action_arm_joint`：应用层 delta/joint-limit 处理并通过 gate 的候选；
- `action_arm_joint_sent`：实际转发给 arm worker 的命令流；
- `action_hand_joint_raw`：ramp、delta/operational clip 和后续校验前的 retarget 输出；
- `action_hand_joint`：最终接受的 hand command。

这些字段足以支持不改变行为的内部整理，但不能分别还原 arm delta clip 与 joint-limit
clip，也不能逐层还原 TAG/DexPilot filter、ramp、clip 或每个 reject 原因。如果目标是
逐操作审计，需要协调 writer、reader、visualization、replay 和 schema marker，而不是
向 v16 静默增加含义。

## 9. 已完成清理与兼容性

本次整理同步完成：

1. 删除 `PointCloudProcessorConfig.depth_ema_alpha`、EMA 状态、处理分支和
   `pc_depth_ema_alpha` metadata；点云只保留适合运动场景的单帧空间滤波；
2. 删除未被读取的 `CalibrationConfig.target_lead_max_m`；
3. 将 HDF5 文档中的 “firmware joint-limit clip” 更正为应用层 clip；
4. 更正 `action_arm_joint`、`action_hand_joint_raw` 和 TAG/DexPilot EMA 的过时表述。

这些变化不修改共享内存 dtype 或 HDF5 v16 固定 dataset。历史 episode 的自由格式
camera metadata 中可能仍包含 `pc_depth_ema_alpha`；读取方应把它视为历史配置快照，
新 episode 不再写入该键。

## 10. 后续修改规则

新增或调整 EMA、clip、scale、reject 时，至少同时回答：

1. 操作位于 raw、candidate、accepted、sent 中的哪一阶段；
2. 默认启用还是关闭，配置来源是否唯一；
3. 它改变端点还是 reject-whole，失败后 hold、drop、abort 还是 fault；
4. temporal state 在 begin、pause、home、generation 变化和反馈故障时如何 reset；
5. producer、SafetyGate、worker、driver 是否存在同义检查及其各自责任；
6. v16 现有字段能否表达变化，是否需要显式 schema 升级。

当前仍需单独决策的两项是 arm joint delta clip 的长期所有权，以及 learned-policy
deployment 是否必须接入与 teleop 相同的 workspace planner callback。
