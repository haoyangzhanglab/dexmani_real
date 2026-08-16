# EMA、Delta Clip、Scale 与 Reject 控制合同

> **文档状态：CURRENT + DECISION REGISTER**
>
> 当前事实基线：`88becfd`（2026-08-16）。
>
> 未提交的工作树实现改动不计入 `CURRENT`；合入后必须重新核对本合同。
>
> 本文只描述、评估控制语义，不授权修改运行时行为、配置、共享内存 dtype 或 HDF5 v16。

本文统一描述 DexMani Real 中会改变控制目标、限制目标或拒绝目标的
EMA、filter、ramp、delta/limit clip、scale 和 reject。目标是让读者能够快速回答：

1. 一个操作位于哪一控制阶段；
2. 它是否改变端点，还是拒绝整个候选；
3. 它依赖什么历史状态，状态何时推进和 reset；
4. 哪个边界负责验证，失败后由哪个控制源决定 hold、drop、abort 或 fault；
5. 现有录制字段能否观察该操作。

文中状态标签含义如下：

| 标签 | 含义 |
|---|---|
| `CURRENT` | 当前基线中已经存在的实现事实 |
| `GAP` | 当前实现或可观测性尚未闭环，但本文不改变行为 |
| `DECISION` | 后续必须显式决定的控制所有权问题 |
| `FUTURE` | 只有获得后续实现授权后才能执行的建议 |

## 1. 范围和统一阶段

本文覆盖 VR teleop、keyboard、camera calibration、replay、learned-policy
deployment、home 的 arm/hand 动作边界，以及与这些语义相关的 HDF5 v16 字段。
传感器 scale 仅在它改变物理量含义时纳入。

本文不把 `arccos` 前裁剪余弦、可视化色值裁剪，以及普通 shape、finite、dtype
校验称为控制策略；但这些校验仍可能是候选发布边界的一部分。

### 1.1 控制阶段

全文使用以下阶段，不单独使用含义不明的 “raw”：

```text
observation
  -> mapped
  -> shaped
  -> candidate
  -> validated
  -> published
  -> SDK-accepted
  -> measured
```

| 阶段 | 定义 |
|---|---|
| observation | VR、相机或机器人反馈等输入快照 |
| mapped | 完成坐标变换、retargeting 和控制 scale 后的目标 |
| shaped | 完成 EMA、ramp、delta/limit clip 后的目标 |
| candidate | 带 generation、时效、单位和表示契约的 `ActionCandidate` |
| validated | 通过当前控制源要求的 gate/preflight，端点不再被修改 |
| published | 已写入有序 arm queue 或 latest-wins hand ring |
| SDK-accepted | worker/driver 已接受并转交设备 SDK |
| measured | 设备反馈，允许因接触、限流或跟踪滞后而不同于命令 |

“raw” 只能相对相邻阶段解释。例如 `action_arm_joint_raw` 是 final IK output、
pre-joint-limit-clip，并不是未经筛选的 MPlib 解。

### 1.2 操作语义

| 操作 | 语义 |
|---|---|
| EMA / filter | 使用历史状态改变当前目标；本文 alpha 均表示新值权重 |
| ramp | 从锚点按显式进度生成一系列中间目标 |
| delta clip | 相对参考命令截断本次变化量，生成不同端点 |
| limit clip | 将目标投影到固定上下界，生成不同端点 |
| reject | 不修复候选，拒绝整条候选；调用方决定失败结果 |
| scale | 映射增益、几何比例、单位换算或规划归一化；不同类别不共享通用抽象 |

Scale 必须保留类别和单位：VR `pos_scale/rot_scale` 是控制映射增益；adaptive pinky、
TAG finger ratio 和 DexPilot `scaling_factor` 是人手到机器人几何的匹配；RealSense
`depth_scale` 与 XHand tactile `0.1` 是 SDK 数值换算，后者不能据此宣称为 SI force；
joint-range normalization 和 candidate weights 只影响规划评分。不得把这些语义收敛为
一个通用 `scale()` 配置。

## 2. 默认行为与时钟

| 操作 | 当前默认 | 状态 | 时钟/参考 | 来源 |
|---|---:|---|---|---|
| EEF position EMA | `alpha_pos=0.6` | 启用 | control grid；上一成功发布目标 | `config/defaults.py::EMAParams` |
| EEF rotation EMA | `alpha_rot=0.25` | 启用 | control grid；上一成功发布目标 | 同上 |
| VR position/rotation scale | `1.0 / 1.0` | 数值直通 | reset-relative wrist motion | `VRMappingParams` |
| VR rotation spike clip | `0.52 rad/map` | 启用 | 相邻有效 mapper input | `ArmWristMapper` 构造默认值 |
| VR total rotation clip | `3.0 rad` | 启用 | reset-relative 总旋转 | `VRMappingParams` |
| hand retargeter | `tag` | 启用 | 每个有效 landmarks snapshot | `PolicyParams` |
| hand startup ramp | `0.5 s` | 启用 | control-grid step；重锚时的 hand pose | `PolicyParams` |
| TAG pinch-factor EMA | `0.4` | 启用 | 成功 Stage 1 的 retarget call | `TAGRetargetingParams` |
| DexPilot internal low-pass | `0.6` | 选择 DexPilot 时启用 | retarget call | bundled YAML |
| DexPilot outer output EMA | `1.0` | 直通、关闭 | retarget call | bundled YAML |

固定 alpha 和 `rad/map` 都与调用频率耦合。修改 `control_hz` 会改变 EEF EMA、TAG
pinch EMA 和 DexPilot low-pass 的实际时间常数；hand ramp 通过
`round(duration_s * control_hz)` 保持近似时长。

Pointcloud temporal EMA 当前不存在；相机点云只保留 median、speckle 等单帧空间
滤波，不属于本控制链。

## 3. 共享候选发布边界

所有普通 arm/hand 候选共享以下发布尾部：

```text
producer mapping/shaping
  -> build ActionCandidate
  -> runtime gate
  -> arm feedback health snapshot
  -> SafetyGate
  -> optional hand feedback health snapshot
  -> optional coupled-hand preflight
  -> arm queue / hand ring publish
  -> worker validation
  -> driver / SDK
```

`validate_and_send_candidate()` 当前先运行 `SafetyGate`，再运行 coupled-hand
preflight。VR teleop 在进入共享尾部前还有一次本地 hand sanitizer，用于把可预期的
hand 问题转换成协调 hold；它不是第二个通用发布边界。

### 3.1 责任分层

| 边界 | 当前责任 | 是否修改端点 | 失败语义 |
|---|---|---|---|
| Producer | mapping、EMA/ramp、必要的主动 clip、IK/retargeting | 可以 | 路径自有 hold/drop |
| `SafetyGate` | representation/units/frame、generation、shape/finite、arm/hand command limits、可选 workspace segment | 不修改 | 返回 typed gate reject |
| Publication boundary | runtime flags、SafetyState、arm feedback freshness/health、hand feedback health、hand operational/mechanical preflight、传输 | 不修改 | 返回 `CommandPublishResult` |
| Arm worker | dtype、finite、generation、expiry，SDK 前 safety-state gate | 不修改 | 丢弃或停止发送 |
| Hand worker | arm-worker 同类检查，加 operational/mechanical | 不修改 | 丢弃命令 |
| XHand driver | shape/finite、operational/mechanical、SDK send | 不修改 | reject whole / send failure |
| xArm firmware | velocity、acceleration、collision/current 等设备限制 | 设备 backstop | controller error 进入 supervisor 合同 |

`SafetyGate` 已删除 velocity、collision 和 transition geometry 检查。固件是最终物理
backstop，但不替代应用层输入、路径质量和数据质量检查。

## 4. VR arm 控制链

```text
VR wrist observation
  -> shape/finite/quaternion reject
  -> per-map wrist rotation delta clip
  -> position/rotation scale
  -> reset-relative total rotation clip
  -> mapped EEF target
  -> Cartesian pose EMA
  -> Cartesian workspace position clip
  -> contact-stall re-anchor/hold
  -> IK candidate reject/rank/canonicalize
  -> final IK limit/pose/collision reject
  -> application joint-limit clip
  -> local candidate checks
  -> shared publication boundary
  -> arm worker
  -> xArm Mode 6
```

### 4.1 Mapper、scale 与旋转 clip

`ArmWristMapper` 以相邻有效 **raw wrist orientation** 为 spike-clip 参考。完整输入和
mapped output 验证成功后，才推进 raw baseline 和 quaternion-continuity state；非法帧
不会污染下一帧。

顺序是先在 wrist input 空间执行 `0.52 rad/map` clip，再应用 `rot_scale`，最后限制
reset-relative 总旋转。因此当 `rot_scale > 1` 时，clip 后的 EEF 旋转变化仍会被放大；
`0.52` 不是 EEF 或关节速度限制，也没有随 source timestamp 归一化。

### 4.2 Cartesian EMA 与 workspace

Cartesian EMA 使用上一条**成功发布候选对应的 EEF 目标**作为历史状态；IK failure、
workspace reject 或发布失败不会提交本帧 EMA state。首次重锚后的目标直通。

当前有两个不同的 workspace 操作，而不是两次相同检查：

1. IK 前把 Cartesian position target 投影到配置 box；这是 limit clip；
2. `SafetyGate` 通过唯一的 `planner.is_workspace_segment_safe(measured_arm, candidate_arm)`
   检查实际关节命令段；这是 reject-whole。

当前 teleop 不再在 gate 前第二次调用相同 workspace segment predicate。

### 4.3 IK 后 arm 端点整形

Final IK output 已完成 canonicalization、joint-limit、pose-error 和 collision reject。随后：

1. 投影到 application arm joint limits（通常只处理 IK limit tolerance 内的微小越界）；
2. `SafetyGate` 从最新 measured arm qpos 到 shaped command 做 workspace segment reject。

`RESOLVED`（2026-08-16，工作树未合入）：应用层逐 tick arm delta clip 已删除（见 D1），
速度/加速度平滑由 Mode 6 固件独占（同一 `max_joint_velocity_deg_per_s` 经
`joint_max_speed_rad_per_s` 下发，另 `set_joint_maxacc`）。v16 的
`action_arm_joint_raw` vs `action_arm_joint` 现在只反映 joint-limit 投影的差异；
collision 已按 2026-08-12 从 SafetyGate 移除，归 Mode 6 兜底。

`RESOLVED`（2026-08-16，工作树未合入）：手命令（retarget → ramp → delta/floor clip →
sanitize）已前移到 arm IK 之前，`planner.set_hand_qpos` 现在同步本帧 post-shaping 的
hand command，不再滞后一帧。final IK collision reject 现基于当前帧手构型；但最终
arm+hand 联合候选仍不通过 19-DoF endpoint/segment collision 验证（collision/transition
已按 2026-08-12 从 SafetyGate 移除，归 Mode 6 兜底）。

## 5. VR hand 控制链

TAG 与 DexPilot 必须分开描述；二者没有统一的“hand output EMA”。

### 5.1 TAG

```text
VR landmarks
  -> landmark shape/finite/geometry reject
  -> operator-to-MANO transform
  -> adaptive pinky geometry scale
  -> fingertip/robot geometry scale
  -> Stage 1 temporal regularization
  -> pinch-factor EMA
  -> Stage 2 regularization + optimizer bound projection
  -> SDK joint order output
```

TAG `smooth_weight` 是优化目标中的时序正则；`pinch_ema_alpha` 只平滑 pinch activation
factor；当前 TAG 没有额外的整关节输出 EMA。

### 5.2 DexPilot

DexPilot `SeqRetargeting` 内部 low-pass alpha 为 `0.6`。包装层仍支持第二层 output EMA，
但 bundled `smoothing_alpha=1.0`，当前直通。`action_hand_joint_raw` 在 DexPilot 路径
已经包含内部 low-pass 和当前直通的包装层结果，不能与 TAG optimizer output 等同。

### 5.3 共享 hand shaping 与 reject

两种 retargeter 输出随后进入：

```text
retargeter output
  -> startup smoothstep ramp
  -> operational command-box clip (VR only)
  -> local reject-whole sanitizer
  -> ActionCandidate / SafetyGate
  -> centralized hand feedback + operational/mechanical preflight
  -> latest-wins hand ring
  -> hand worker reject
  -> XHand driver reject
```

VR 主动把 hand command 投影进 operational box。其他耦合发布路径依赖共享 preflight
拒绝 operational、rated mechanical 违规，避免 arm 已入队而 hand 被后级拒绝。Home
发布精确 home 端点（单条命令），不 clip 隐式改写最终 home 端点。

## 6. 其他控制源差异

| 控制源 | 主动改变端点 | 应用层几何检查 | hand 边界 | 主要失败结果 |
|---|---|---|---|---|
| VR teleop | mapper scale/clip、EEF EMA、workspace clip、hand ramp/box clip | IK final endpoint collision；SafetyGate workspace segment | local sanitizer + shared preflight | mapper/IK/local reject hold；workspace gate reject hold；意外发布失败可 fault |
| Keyboard | fixed Cartesian delta、workspace clip、measured position/rotation lead cap | IK final endpoint collision + SafetyGate workspace | 携带 hand 时用 shared preflight | block 当前按键目标或结束流程 |
| Camera calibration | fixed Cartesian delta、workspace clip | IK final endpoint collision + SafetyGate workspace | 正常路径为 arm-only | reject 当前目标；不可恢复失败终止标定 |
| Replay | 不修改录制端点 | 执行前 dense limits/workspace/collision preflight；运行时 SafetyGate workspace | 携带 hand 时用 shared preflight | preflight/publish failure abort |
| Learned-policy deployment | 不插值、不主动 clip model endpoint；逾期 step 可 coalesce | `SafetyGate` workspace segment（process-local planner，见 D2）；无应用层 collision preflight | shared preflight | gate/hand semantic reject abort policy；feedback/transport failure drop tick，silence watchdog 最终 abort |
| Home | 显式 joint milestones | 独立 dense planned path checks | 发布精确 home 端点（bounds preflight） | unsafe path/acceptance failure hold 或 abort |

Keyboard 具有 measured-lead cap；camera calibration 已删除从未生效的
`target_lead_max_m`，两条路径不能表述为完全等价。

Replay dense preflight 是执行前 path reject，不是运行时 command clip，不改变 episode
端点。

## 7. Reject status 与 disposition

Reject boundary 只判定候选是否合法；最终 disposition 属于控制源，不应从
`SafetyGate` 本身推导。

| Reject/失败 | VR teleop | Deployment | Keyboard/calibration/replay/home |
|---|---|---|---|
| Mapper/retargeter invalid | 保持上一目标或不产生候选 | 不适用 | 路径自有 reject |
| IK failure | arm hold；合法 hand-only motion 可继续 | 不适用 | block/abort 当前目标 |
| Local hand sanitizer | arm/hand 协调 hold | 不适用 | 不适用 |
| `GATE_REJECTED: workspace` | 发布 arm hold，hand 不前进 | 当前 deployment 无 workspace callback | 调用方 block/abort |
| 其他 semantic gate reject | 非预期时可进入 sticky fault | 立即 policy abort，回到非运行态 | 调用方 abort |
| `HAND_PREFLIGHT_REJECTED` | 本地 sanitizer 后理论上少见；非预期时 fault | 立即 policy abort | 调用方 abort |
| Feedback/runtime/transport failure | runtime gate 可保持静默；不可恢复失败 fault | drop 当前 tick；silence watchdog 最终 abort | 同步流程返回失败/abort |
| Worker generation/expiry/bounds reject | 丢弃，不修改端点 | 同左 | 同左 |
| Driver/SDK failure | worker health + supervisor 合同 | 同左 | 同左 |

`error_state` 仍是 sticky。Firmware backstop 不授权上游删除 shape/finite、generation、
workspace、mechanical envelope 或数据质量检查。

## 8. Temporal state、commit 与 reset

| 状态操作 | Reference | 何时推进 | 典型 reset | 下游 reject 是否回滚 |
|---|---|---|---|---|
| Mapper rotation spike baseline | 上一有效 raw wrist orientation | 完整 map 成功 | clear/re-anchor、quiescence | 不适用；非法 map 不推进 |
| Cartesian EEF EMA | 上一成功发布对应的 EEF target | candidate 成功发布后 | begin/re-anchor、pause/home quiescence、contact-stall、feedback/camera re-warm 边界 | 是；发布前失败不提交 |
| TAG optimizer regularization | 上一成功 optimizer output | TAG solve 成功 | retargeter reset，优先以 hand pose warm-start | 下游 reject 不回滚 |
| TAG pinch EMA | 上一 pinch factor | Stage 1 成功后、Stage 2 前 | retargeter reset 为 0 | 下游 reject 不回滚 |
| DexPilot filters | retargeter 内部上一输出 | 成功 retarget output | retargeter reset | 下游 reject 不回滚 |
| Hand startup ramp | 重锚时 hand anchor + 当前 retarget target | 每次执行 ramp shaping 时 step 前进 | re-anchor/quiescence | 下游 reject 不回滚 step |

`run_generation` 使旧 queue/ring command 失效，但不会自动 reset 所有进程内滤波器；拥有
temporal state 的 producer 必须在 begin、pause、home、feedback fault、camera re-warm
等 generation 边界显式 clear/reset/reseed。

## 9. HDF5 v16 可观测性

| 字段 | 精确阶段语义 |
|---|---|
| `target_eef_pos_raw` / `target_eef_rot6d_raw` | mapper output，已包含 mapper scale/rotation clip，位于 Cartesian EMA 前 |
| `target_pos_before_clamp` | Cartesian EMA 后、workspace position clip 前 |
| `action_arm_ee` | 提供给 IK 的 desired EEF target；不是 joint-limit-clipped `action_arm_joint` 的 FK 保证 |
| `action_arm_joint_raw` | final IK validated output，位于 teleop joint-limit clip 前 |
| `action_arm_joint` | 应用层 shaping 后、通过发布边界的 arm candidate；hold 槽保存 hold target |
| `action_arm_joint_sent` | 实际转发给 arm worker 的命令流；部分非标准 episode 可缺省 |
| `action_hand_joint_raw` | ramp/operational clip 前；TAG 成功时为 optimizer SDK-order output，其他路径回退为 retargeter/held output |
| `action_hand_joint` | 通过 hand shaping 和发布边界的最终 hand candidate |
| `arm_qpos` / `hand_qpos` | 网格对齐的 measured feedback，不等同于 accepted command |

这些字段适合统计 arm clip 是否触发、shaped/accepted/measured 差异，但不能：

- 还原 joint-limit clip；
- 逐层还原 TAG/DexPilot filter、ramp 和内部 optimizer state；
- 证明每次 temporal-state refactor 行为等价；
- 区分每个 reject/status 原因。

需要逐操作审计时，应先定义是否使用现有 metrics/日志；只有确需 episode 级持久化时，
才协调 writer、reader、analysis、visualization、replay 和 schema marker。不得向 v16
静默增加字段或改变既有含义。

## 10. 待决设计登记

### D1 — Arm delta clip 的动态所有权

- `RESOLVED`（2026-08-16，工作树未合入）：应用层逐 tick arm delta clip 已删除；速度/
  加速度平滑由 Mode 6 固件独占（`joint_max_speed_rad_per_s` + `set_joint_maxacc`）。
  joint-limit 投影保留，是硬限位安全网，与速度所有权无关。
- `PENDING`：真机验证 Mode 6 独占轨迹生成下的遥操作跟踪响应（快速反转过冲/滞后）。
  验证通过后回填 commit、前移基线指针。

### D2 — Deployment geometry contract

- `DECISION → RESOLVED`（2026-08-16）：workspace 是所有控制源的全局物理 envelope。
- `CURRENT`：coordinator 启动时构建 process-local `XArm7MotionPlanner`，经
  `planner_action_safety_gate` 把 `workspace_check = planner.is_workspace_segment_safe`
  接入 deployment gate；`workspace_bounds` 取自 `runtime.policy.workspace`，与 VR
  teleop 使用同一 workspace 检查。endpoint/segment collision preflight 仍按
  2026-08-12 决策移除，未恢复。
- 本改动为工作树未合入改动；合入后需回填 commit 并把基线指针前移。

### D3 — Hand delta clip

- `RESOLVED`（2026-08-16，工作树未合入）：`hand.max_delta_rad` 命令间速率限位机制已
  整体删除（config → 发布边界 → worker/driver → teleop → coordinator → hand-home
  里程碑 → 4 个 example）。默认本就 `None`，运行时行为不变；手部速度保护现仅由固件
  PID + per-joint `tor_max` 电流限位承担。原「统一 reference / resync / 重新 ramp」
  的 DECISION 随功能一并关闭；若后续路径需要限速，需按原链路重新引入并先定义
  reference 合同。

### D4 — Rate-dependent filter 配置

- `DECISION → RESOLVED`（2026-08-16，文档化，不改代码）：保持 rate-tuned 固定 alpha。
  系数按 `control_hz=16` 的每 tick/每帧调用速率定义，只有 hand startup ramp 通过
  `hand_ramp_duration_s × control_hz` 显式参数化；其余 EMA/scale/clip 均隐式耦合
  `control_hz`。
- 耦合面（改 `control_hz` 必须同步重推导）：VR pos/rot scale（默认 1.0/1.0）、
  per-frame 旋转尖峰 clip 与 total-from-reset 旋转 cap（`max_delta_rot_rad`）、
  EEF 位置/姿态 EMA、TAG `pinch_ema_alpha`/`smooth_weight`/`reg_last_weight`
  （0.4/0.02/0.8）、DexPilot low-pass/smoothing EMA、hand 启动 ramp。
- `FUTURE`：任何调整必须同时更新默认值、派生率、metadata 和离线行为检查。

### D5 — Reject observability

- `DECISION → RESOLVED`（2026-08-16，metrics/log 归因）：工程诊断由 metrics/log 完成，
  不升级 v16。coordinator 对 reject-whole 分支逐操作计数——gate 拒绝按 `GateRejectCode`
  拆分为 `safety_reject_<code>` 计数器（叠加 aggregate `SAFETY_REJECTIONS`），hand
  preflight 拒绝计入 `HAND_PREFLIGHT_REJECTIONS`（此前未计数）。
- `FUTURE`：暂不规划 episode 级 schema 升级；若后续需要把逐操作归因写入训练数据，
  再显式规划 v16 升级。

## 11. 后续修改检查表

新增或调整 EMA、filter、ramp、clip、scale、reject 时，至少同时回答：

1. 输入和输出分别属于 observation、mapped、shaped、candidate、validated、published、
   SDK-accepted、measured 中的哪一阶段；
2. 默认启用还是关闭，配置来源是否唯一，单位和调用时钟是什么；
3. 操作改变端点还是 reject-whole；
4. temporal reference 是 input、last candidate、last published 还是 last accepted，状态何时提交；
5. begin、pause、home、generation、feedback fault、camera re-warm 如何 reset/reseed；
6. producer、SafetyGate、publication boundary、worker、driver 是否有同义检查，各自责任是什么；
7. 每个控制源在失败后 hold、drop、abort 还是 fault；
8. shaping 后是否生成了需要重新做 pose/workspace/collision 验证的新端点；
9. v16 能否表达变化；若不能，是否真的需要 schema 升级。

## 附录 A：历史兼容性与交叉文档

- Pointcloud temporal EMA、`pc_depth_ema_alpha` 新 metadata 和未使用的
  `CalibrationConfig.target_lead_max_m` 已从当前实现删除。历史 episode 的自由格式
  camera metadata 仍可能包含旧键，应视为历史配置快照。
- HDF5 v16 固定 dataset 未因上述清理改变。
- 截至本文基线，`docs/hand_retargeting.md` 仍有把 TAG 描述为存在 outer EMA 的过时
  段落；当前 TAG 实现和本文第 5/9 节是本控制合同的事实基线。交叉文档应在独立文档
  一致性修改中同步，不能据此推断运行时存在 TAG output EMA。
