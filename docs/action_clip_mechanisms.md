# 动作 Clip、限幅与拒绝机制

本文基于当前源码、配置默认值和持久化 schema 梳理动作（arm/hand/EEF）在
teleop、learned-policy、replay 和数据清洗路径中的限幅机制。这里的“fact-check”
指以运行时代码为准；README、注释或文件名与代码不一致时，以代码为准。

## 1. 先看结论

项目没有一个统一的 `clip(action)` 入口，而是分成两类：

| 类型 | 行为 | 对动作的影响 |
|---|---|---|
| producer-side clip / shaping | 在 mapper、动作 proposal 或 worker 中将候选值压回边界，或限制一步变化 | 候选动作会被修改后继续流转 |
| reject-only safety check | 在 `SafetyGate`、worker preflight、IK 验证或离线清洗中检查；失败即拒绝、保持或标记无效 | 候选动作不会被改写；部分路径会终止当前 policy run |

最容易混淆的三点：

1. VR teleop 的 arm/hand proposal 确实会做 joint bound 和每 tick delta clip。
2. learned-policy 的 `PolicyExecutor` 对 arm/hand 超限 endpoint 都是 reject，不是 clip；首条动作
   相对 measured feedback，后续动作相对上一条成功发布 target 检查。
3. EMA、smoothstep ramp、IK nullspace 优化、xArm 速度/加速度是动作整形或轨迹参数，不应统称为 clip。

## 2. 运行时会真正修改候选动作的机制

| 路径/阶段 | 对象 | 机制 | 具体参数 | 基准、时机与结果 | 代码依据 |
|---|---|---|---|---|---|
| VR `ArmWristMapper` | wrist rotation | 限制相邻接受姿态的单帧旋转增量 | 构造器默认 `max_per_frame_rot_rad=0.52 rad`，约 `29.8°`；当前调用链没有单独 runtime override | 以 mapper 上一帧接受的 wrist rotation 为基准；超限时压回最大允许旋转 | [`arm_mapper.py:138`](../dexmani_real/teleop/arm_mapper.py#L138)、[`loop.py:153`](../dexmani_real/teleop/loop.py#L153) |
| VR `ArmWristMapper` | 从 reset anchor 累积的 wrist rotation | 限制相对 reset 姿态的累计旋转 | `runtime.policy.vr_mapping.max_delta_rot_rad=3.0 rad`，约 `171.9°` | 以 reset 时的参考姿态为基准；达到边界后压回。若需要继续转动，必须重新建立 reset/re-anchor | [`arm_mapper.py:222`](../dexmani_real/teleop/arm_mapper.py#L222)、[`defaults.py:119`](../dexmani_real/config/defaults.py#L119) |
| VR EEF target proposal | EEF position | 工作空间 `np.clip` | 默认 `x=[0.25,0.72] m`、`y=[-0.50,0.50] m`、`z=[0.05,0.50] m` | 在 EMA 之后对 world/arm-base 目标位置限幅；限幅前位置另存为 `target_pos_before_clamp` | [`defaults.py:62`](../dexmani_real/config/defaults.py#L62)、[`action_proposal.py:77`](../dexmani_real/teleop/action_proposal.py#L77)、[`action_proposal.py:109`](../dexmani_real/teleop/action_proposal.py#L109)、[`control_grid.py:774`](../dexmani_real/teleop/control_grid.py#L774) |
| VR arm action proposal | 7 个 arm joint | 先做 joint bound clip，再做相对上一条 arm command 的 wrap-aware delta clip，最后再做 joint bound clip | `arm_max_delta_rad_per_tick=8°=0.139626 rad` | 相对上一条已发布/接受的 arm command；角度按最近等价表示比较；这是 producer-side 修改 | [`action_proposal.py:239`](../dexmani_real/teleop/action_proposal.py#L239)、[`defaults.py:642`](../dexmani_real/config/defaults.py#L642) |
| VR hand action proposal | 12 个 hand joint | 先压到 operational command bounds，再限制相对上一条 published hand endpoint 的 delta | `hand_max_delta_rad_per_tick=0.3 rad` | 以上一条 published endpoint 为参考；每个 joint 独立限幅 | [`action_proposal.py:123`](../dexmani_real/teleop/action_proposal.py#L123)、[`action_proposal.py:236`](../dexmani_real/teleop/action_proposal.py#L236)、[`defaults.py:499`](../dexmani_real/config/defaults.py#L499) |
| TAG retarget optimizer | 内部 `target_factors` | 对优化变量做 box constraint | `target_factors ∈ [0,1]` | 这是 retarget 内部变量边界，不是最终 arm/hand SDK command 的 clip | [`tag_optimizer.py:223`](../dexmani_real/teleop/retarget/tag_optimizer.py#L223) |
| TAG retarget optimizer | optimizer joint result / warm start | 按 URDF joint bounds 压回可行范围 | 每个 joint 使用对应 URDF lower/upper bound | 作用于优化过程的候选 q；之后仍需经过上层 IK/安全路径 | [`tag_optimizer.py:270`](../dexmani_real/teleop/retarget/tag_optimizer.py#L270) |
| Keyboard / camera calibration EEF command | EEF position | 工作空间 clip，并留出 command margin | `workspace_command_margin_m=0.005 m`；默认有效范围为 `x=[0.255,0.715]`、`y=[-0.495,0.495]`、`z=[0.055,0.495] m` | 这是相对 nominal workspace 向内收缩 `5 mm` 后的 command workspace | [`keyboard_session.py:539`](../dexmani_real/teleop/keyboard_session.py#L539)、[`control.py:286`](../dexmani_real/calibration/camera/control.py#L286) |
| Keyboard EEF command | position lookahead | 限制目标相对当前 EEF 的 lead；超限时按向量比例缩放 | `command_lookahead_frames=5`、`delta_pos_m=0.008 m/frame`，所以最大 position lead 为 `0.040 m` | 这是 lead cap/scaling，不是 arm joint-space clip；用于限制键盘命令的前视目标 | [`keyboard_session.py:561`](../dexmani_real/teleop/keyboard_session.py#L561) |
| Keyboard EEF command | rotation lookahead | 限制目标相对当前 EEF 的旋转 lead；超限时按向量比例缩放 | `delta_rpy_rad=0.03 rad/frame`、`command_lookahead_frames=5`，所以最大 rotation lead 为 `0.15 rad` | 这是旋转目标的 lead cap；不等于 arm joint delta limit | [`keyboard_session.py:561`](../dexmani_real/teleop/keyboard_session.py#L561) |
| hand worker → SDK | hand joint target | 对 SDK setpoint 做每 servo tick slew limiting | `hand.hand_max_delta_rad_per_tick=0.3 rad/tick` | RUNNING 以上一条 SDK accepted setpoint 为参考；ARMED/homing 以 measured qpos 为参考；可用中间 setpoint 接近原 endpoint | [`hand_worker.py`](../dexmani_real/robot/hand_worker.py) |

### 2.1 VR arm proposal 的顺序

当前实现的有效顺序是：

```text
IK q candidate
  → arm joint bounds clip
  → relative to previous arm command: wrap-aware delta clip (8°/tick)
  → arm joint bounds clip again
  → SafetyGate / worker validation
```

因此，`8°/tick` 不是从 measured qpos 计算的，而是以 proposal 保存的上一条 arm command 为基准。
这与 arm worker 的 `20°` command-jump guard（见第 4 节）属于不同边界。

### 2.2 VR hand proposal 与 hand worker 的区别

两层都可能看到 `0.3 rad`，但参考状态不同：

| 层 | 参考状态 | 用途 |
|---|---|---|
| VR hand proposal | 上一条 published hand endpoint | 控制网格内的连续动作 proposal |
| hand worker | RUNNING：上一条 SDK accepted setpoint；ARMED：measured qpos | SDK-level actuator slew protection |

## 3. 参数总表

以下为当前默认值；如果通过 YAML/CLI 覆盖，runtime resolved config 才是实际值。

| 参数 | 默认值 | 单位/shape | 作用 |
|---|---:|---|---|
| `policy.control_hz` | `16` | Hz | learned-policy/control grid 的目标频率；`hand_ramp_duration_s=0.5` 对应约 8 个 policy grid frame |
| `policy.executor_poll_hz` | `128` | Hz | PolicyExecutor 的发布/检查轮询；高于 16 Hz control grid，但每轮最多处理一个到期 endpoint，不制造 burst |
| `policy.arm_max_delta_rad_per_tick` | `0.139626` | rad，等于 `8°` | VR arm proposal 的单 tick delta clip |
| `policy.hand_max_action_jump_rad` | `1.0` | rad | learned-policy hand 每 joint endpoint jump 的独立 reject 阈值 |
| `hand.hand_max_delta_rad_per_tick` | `0.3` | rad | VR hand proposal shaping 与 hand worker SDK setpoint slew 的界限；不用于 PolicyExecutor shaping |
| `max_per_frame_rot_rad` | `0.52` | rad，约 `29.8°` | VR wrist mapper 相邻接受姿态的单帧旋转 clip；构造器默认，不是独立 runtime field |
| `vr_mapping.max_delta_rot_rad` | `3.0` | rad，约 `171.9°` | VR wrist 相对 reset anchor 的累计旋转 clip |
| `workspace` | `x=[0.25,0.72]`、`y=[-0.50,0.50]`、`z=[0.05,0.50]` | m | VR nominal EEF workspace |
| `workspace_command_margin_m` | `0.005` | m | keyboard/calibration workspace 向内 margin；每一侧收缩 5 mm |
| `command_lookahead_frames` | `5` | frame | keyboard EEF command 的 lookahead horizon |
| `delta_pos_m` | `0.008` | m/frame | keyboard 单 frame position lead 上限；5 frame 合计 0.040 m |
| `delta_rpy_rad` | `0.03` | rad/frame | keyboard 单 frame rotation lead 上限；5 frame 合计 0.15 rad |
| `endpoint_delta_tolerance_rad` | `1e-12` | rad | endpoint delta 检查的数值容差；不是放宽正常动作限幅的幅度 |
| `arm.max_servo_command_jump_rad` | `0.349066` | rad，等于 `20°` | PolicyExecutor SafetyGate 的 arm endpoint reject 阈值，也是 arm worker 对相邻 accepted target 的 fail-closed jump guard |
| `max_ik_jump_deg` | `(30,30,30,35,40,40,40)` | °，7 joints | IK candidate jump reject threshold |
| `hand_ramp_duration_s` | `0.5` | s | hand startup smoothstep ramp；是时间整形，不是 clip |
| `ik_nullspace_step_rate_deg_s`（VR） | `50` | °/s | VR IK nullspace step rate；按 16 Hz grid 折算为约 `3.125°/grid tick` |
| `TeleopProfile.nullspace_step_deg`（keyboard/calibration 默认） | `1` | °/step | teleop IK nullspace 单步调整量 |
| xArm speed | `135` | °/s | xArm trajectory 参数，不是 qpos clip |
| xArm acceleration | `810` | °/s² | xArm trajectory 参数，不是 qpos clip |

### 3.1 Arm joint bounds

默认 7-DoF arm joint bounds（rad）为：

| joint index | lower | upper |
|---:|---:|---:|
| 0 | `-6.28318530718` | `6.28318530718` |
| 1 | `-2.059` | `2.0944` |
| 2 | `-6.28318530718` | `6.28318530718` |
| 3 | `-0.19198` | `3.927` |
| 4 | `-6.28318530718` | `6.28318530718` |
| 5 | `-1.69297` | `3.14159265359` |
| 6 | `-6.28318530718` | `6.28318530718` |

对应向量：

```text
lower = (-6.28318530718, -2.059, -6.28318530718, -0.19198,
          -6.28318530718, -1.69297, -6.28318530718)
upper = ( 6.28318530718,  2.0944,  6.28318530718,  3.927,
           6.28318530718,  3.14159265359,  6.28318530718)
```

来源：[`defaults.py:270`](../dexmani_real/config/defaults.py#L270)。运行时若使用实际机器人/URDF覆盖，应以 resolved config 和机器人模型为准。

### 3.2 Hand bounds

以下均为 12-DoF hand joint bounds（rad，按项目 hand joint order）。

| bound set | min | max | 用途 |
|---|---|---|---|
| rated/mechanical | `(0,-0.698,0,-0.174,0,0,0,0,0,0,0,0)` | `(1.832,1.745,1.745,0.174,1.919,1.919,1.919,1.919,1.919,1.919,1.919,1.919)` | 机械/额定范围；worker/preflight 的硬边界依据 |
| operational command | `(0,-0.698,0.1745329252,-0.174,0,0.0872664626,0,0.0872664626,0,0.0872664626,0,0.0872664626)` | `(1.832,1.745,1.745,0.174,1.919,1.919,1.919,1.919,1.919,1.919,1.919,1.919)` | teleop hand command 的运行范围；proposal clip 使用 |

来源：[`defaults.py:444`](../dexmani_real/config/defaults.py#L444)、[`defaults.py:447`](../dexmani_real/config/defaults.py#L447)。

## 4. 只拒绝、不修改候选动作的机制

| 阶段/路径 | 检查对象 | 具体参数/条件 | 失败行为 | 代码依据 |
|---|---|---|---|---|
| Teleop IK candidate | 相邻 IK q candidate 的 joint jump | `max_ik_jump_deg=(30,30,30,35,40,40,40)°` | reject candidate；不会把 q 改成边界值 | [`types.py:190`](../dexmani_real/planning/types.py#L190)、[`ik.py:265`](../dexmani_real/planning/ik.py#L265) |
| Teleop IK pose validation | EEF position/orientation error | VR 默认 `0.02 m`、`5°`；keyboard/calibration 默认 `0.002 m`、`2°` | reject candidate | [`defaults.py:637`](../dexmani_real/config/defaults.py#L637)、[`defaults.py:749`](../dexmani_real/config/defaults.py#L749)、[`ik.py:269`](../dexmani_real/planning/ik.py#L269) |
| `SafetyGate` | representation、generation、shape、finite、joint limits、workspace、可选 endpoint delta、collision | endpoint delta 使用 `endpoint_delta_tolerance_rad=1e-12`；其余由 safety config、robot bounds 和 collision model 决定 | reject candidate；不做 silent clip | [`safety_gate.py:219`](../dexmani_real/control/safety_gate.py#L219) |
| learned-policy `PolicyExecutor` | arm/hand endpoint delta | arm 使用 `20°=0.349066 rad`；hand 使用独立 `policy.hand_max_action_jump_rad`；首条相对 measured、后续相对上一条成功发布 target | 任一超限则拒绝整个 coupled step；target 不 clip、不 shaping，reference 不推进 | [`executor.py`](../dexmani_real/deployment/executor.py)、[`safety_gate.py`](../dexmani_real/control/safety_gate.py) |
| hand publication preflight | hand operational/mechanical bounds | 必须同时满足 operational command bounds 和机械/额定硬边界 | reject publication | [`publication.py:489`](../dexmani_real/control/publication.py#L489)、[`limits.py:64`](../dexmani_real/utils/limits.py#L64) |
| arm worker | joint bounds、相邻 accepted command target jump | `max_servo_command_jump_rad=20°=0.349066 rad` | fail closed；保持/故障退出取决于 worker lifecycle，不会按 20°自动重写目标 | [`arm_worker.py`](../dexmani_real/robot/arm_worker.py) |
| offline data cleaning | deployment action endpoint delta | 使用部署时的 arm/hand endpoint limit contract；首段以 measured state 为基准，后续使用保留的上一条 action | 删除/标记 `deployment_action_limit` 无效行，不把行内动作 clip 后继续使用 | [`clean.py:210`](../dexmani_real/data/clean.py#L210)、[`clean.py:722`](../dexmani_real/data/clean.py#L722) |
| physical replay preflight | replay action 的 bounds、workspace、collision 等 | replay 读取 `/action_arm_joint_sent`；`wrap_nearest_equivalent` 只选择角度等价表示 | preflight reject；不进行一般意义的 action clip | [`trajectory.py:140`](../dexmani_real/replay/trajectory.py#L140)、[`trajectory.py:506`](../dexmani_real/replay/trajectory.py#L506) |

`SafetyGate` 与 worker 的 reject 是故意 fail-closed 的安全边界：不能为了让测试或 replay 通过而将 reject 改成 clip。

## 5. 不同动作源的实际覆盖范围

| 动作源 | producer-side 修改 | reject-only 检查 | worker 最后边界 |
|---|---|---|---|
| VR teleop | wrist rotation clip、EEF workspace clip、arm joint/delta clip、hand operational/delta clip | IK jump/pose、SafetyGate、worker guards | arm worker jump/bounds；hand worker SDK setpoint `0.3 rad/tick` slew |
| Keyboard teleop | EEF workspace clip、position/rotation lookahead scaling | IK jump/pose、SafetyGate、worker guards | 同上；键盘自身没有独立的 arm joint `8°/tick` producer clip |
| Camera calibration control | EEF workspace clip（5 mm margin） | IK/控制 preflight、SafetyGate 及 worker guards | 同上 |
| Learned policy | 无 producer-side smoothing；仅允许 tiny float32 hand endpoint roundoff canonicalization | arm/hand SafetyGate reject current coupled step、action-step/silence/progress watchdog | arm worker jump guard；hand worker SDK setpoint slew limiting |
| Physical replay | 不做一般 action clip；只做角度等价表示转换 | replay preflight、SafetyGate、worker guards | 同上 |

因此，“项目有动作 clip”不能简化为“所有路径都会 clip”：同一个超限动作在 VR producer、PolicyExecutor、offline cleaning 和 replay 中的处理语义不同。

## 6. 录制与数据审计字段

动作 clip 是否发生，不能只看最终发送值。当前录制/IPC schema 保留了可审计的原始值和限幅前位置：

| 字段 | 含义 |
|---|---|
| `action_arm_joint_raw` | producer 产生的原始 arm action；在 proposal clip 前后用于诊断来源差异 |
| `action_hand_joint_raw` | producer 产生的原始 hand action |
| `target_pos_before_clamp` | EEF workspace clamp 前的位置；可与最终位置对照 |
| `action_arm_joint` / `action_hand_joint` | episode sample 中的动作字段；具体 sample 语义需结合 source/path 使用 |
| `action_arm_joint_sent` | 发送/部署语义下的 arm endpoint；physical replay 要求该字段存在 |
| `action_semantics` | 当前部署动作语义为 `deployment_grid_rate_limited_target` |
| `deployment_action_limit` | offline cleaning 对 endpoint limit violation 的 reason mask；表示该行无效，不表示动作已被修复 |

字段定义和写入路径：[`recording/schema.py:47`](../dexmani_real/recording/schema.py#L47)、[`recording/schema.py:96`](../dexmani_real/recording/schema.py#L96)、[`recording/schema.py:103`](../dexmani_real/recording/schema.py#L103)、[`recording/client.py:175`](../dexmani_real/recording/client.py#L175)、[`control_grid.py:711`](../dexmani_real/teleop/control_grid.py#L711)。

建议审计一条动作时至少同时比较：

```text
raw producer action
  → clipped proposal / EEF target
  → action_arm_joint / action_hand_joint
  → action_*_joint_sent
  → worker measured state and SDK target
```

## 7. 与 clip 相关、但不是 clip 的机制

| 机制 | 当前参数 | 实际作用 |
|---|---:|---|
| EEF EMA | position `alpha=0.5`、rotation `alpha=0.5` | 对输入做指数平滑；会改变响应，但不是把值压到上下界 |
| TAG pinch EMA | `pinch_ema_alpha=0.75` | 当前 pinch factor 使用 `25%` 历史值 + `75%` 当前值；只影响 pinch 激活权重，不直接对 12 维 hand qpos 做 EMA |
| hand startup ramp | `hand_ramp_duration_s=0.5 s` | smoothstep 插值让 hand command 平滑启动；不是安全边界 clip |
| IK nullspace optimization | VR `50°/s`（16 Hz 约 `3.125°/grid tick`）；keyboard/calibration 默认 `1°/step` | 在 joint-limit 等约束下优化 posture；不等于把最终动作直接 `np.clip` 到边界 |
| xArm trajectory shaping | speed `135°/s`、acceleration `810°/s²` | SDK/轨迹执行速度参数；不等于 qpos endpoint clip |
| latest-wins / temporal freshness | 由 IPC/control grid 的 freshness 与 sequence 规则控制 | 丢弃过期或被覆盖的动作；不是数值限幅 |

## 8. 一页式动作流

```text
VR:
  VR input → wrist/EEF clip → IK reject checks → arm/hand proposal clip
           → SafetyGate reject checks → worker bounds/jump checks → SDK

Keyboard / calibration:
  EEF workspace/lead shaping → IK reject checks → SafetyGate
                              → worker bounds/jump checks → SDK

Learned policy:
  model Prediction → PolicyExecutor decode
                   → arm/hand reject-only jump + joint/workspace checks
                   → exact coupled non-blocking publication
                   → arm worker reject + hand worker 0.3 rad/tick SDK slew → SDK

Replay:
  recorded sent action → angle-equivalent representation (if needed)
                       → replay/SafetyGate/worker preflight → SDK
```

本文描述源码与配置语义，不能替代真机回归。learned-policy 的真机验证状态见
[`policy_deployment.md`](policy_deployment.md)；physical replay 仍须按其 preflight 单独验证。
