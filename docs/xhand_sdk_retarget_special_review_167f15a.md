# XHand SDK 使用与 Retarget 专项深度代码审查

> 审查对象：DexMani Real `167f15a5f76b798ea5a90e44fe3e478eecc266d2`  
> 审查日期：2026-08-09  
> 审查方式：源码调用链、固定版本对照、mock 故障注入、数值不变量、Pinocchio/NLopt 离线验证、已有 HDF5 录制数据复核  
> 安全边界：未连接、探测或控制任何真实硬件；未运行遥操作、回放、归位、校准、RealSense 或厂商 SDK 真机命令

## 1. 结论

本次专项审查确认 **10 个问题**：

- P0：0 项；
- P1：4 项；
- P2：5 项；
- P3：1 项。

其中最需要优先处理的是：

1. XHand 成功打开后，初始化异常和 device-id 不匹配路径没有执行统一的 EtherCAT INIT/close/watchdog 清理；
2. 持续 `send_command()` 失败不会升级为全局 sticky fault，健康的状态读取还会覆盖发送错误状态；
3. `qpos_stale` 把“手保持静止”当成“反馈失效”，一旦 policy 因此 hold，可能形成自锁；
4. 新鲜但几何退化的 VR landmarks 会被 TAG 和 DexPilot 接受，并产生可发送的显著闭合动作，而不是 hold。

专项结论不是“当前实现整体不可用”。相反，下列关键链路已通过离线复核：TAG 的 SDK↔Pinocchio 双向映射、Pinocchio Jacobian、严格关节边界、Quest→MANO→URDF 当前旋转、policy/driver 双层有限值与步长限制均基本正确。风险主要集中在 **故障语义、初始化清理、数据新鲜度定义和 retarget 输入前置条件**。

## 2. 严重度排序总表

| ID | 严重度 | 状态 | 问题 | 主要后果 |
|---|---:|---|---|---|
| XH-SDK-01 | P1 | confirmed | post-open 失败路径绕过统一 EtherCAT 清理 | 下次连接可能需要 XHand 断电重启；异常时设备可能停留在非预期总线状态 |
| XH-SDK-02 | P1 | confirmed | 持续发送失败不会锁存全局故障，健康 read 会清除 driver error | 手不执行动作但系统仍可能保持 RUNNING，臂继续运动，抓取物可能失稳 |
| XH-SDK-03 | P1 | confirmed | 静止 qpos 被误判 stale，policy hold 后可自锁 | 正常保持姿态约 0.5 s 后手控制被永久冻结，直到反馈抖动或外力改变 qpos |
| XH-RT-01 | P1 | confirmed | 有限但退化的 21 点输入被当作有效手姿 | 全零输入可使 TAG 产生约 81.6°、DexPilot 产生约 110° 的关节目标；步长限制只减速，不会拒绝错误动作 |
| XH-TACT-01 | P2 | confirmed | contact-only raw tactile ring 在 release 后永久 forward-fill | HDF5 同一帧可能同时记录 `contact=False` 和上一次接触的 raw taxel 数据 |
| XH-TACT-02 | P2 | confirmed | 启动 tactile software bias 没有执行“无接触”前置条件 | 启动时真实接触载荷会被吸收到 bias，本次会话后续接触被低估或完全归零 |
| XH-RT-02 | P2 | confirmed | pinky progressive scaling 使用已修改父节点计算下一骨段 | PIP→DIP 可翻向；真实录制 427 帧中有 297 帧发生方向反转 |
| XH-RT-03 | P2 | confirmed | DexPilot reset 对同一 joint mapping 应用了两次正向映射 | home pose warm-start 最大错位 75.66°，首帧收敛、时延和输出稳定性劣化 |
| XH-ARCH-01 | P2 | confirmed | policy/TAG 为取限位导入 `robot.xhand`，连带加载厂商 SDK | 违反进程边界；policy 被原生 SDK 的导入兼容性和副作用耦合 |
| XH-SDK-04 | P3 | confirmed | SDK 缺失时隐式 stub 的连接、反馈和 homing 语义互相矛盾 | `connect=True`、`is_connected=False`，反馈固定零，canonical worker 最终仍启动失败 |

### 2.1 严重度定义

- **P0**：可直接造成不可接受的人身/设备危险、数据不可恢复损坏，且在常规路径可触发，需要立即停止使用；
- **P1**：控制安全、故障隔离或 fail-closed 语义存在实质缺口，可能导致错误动作、跨子系统失协或设备不可恢复，需要优先修复；
- **P2**：会稳定造成数据错误、性能退化、恢复异常或架构边界破坏，但已有保护显著限制了直接物理风险；
- **P3**：主要影响开发、诊断或非默认模式，真实硬件主路径影响较低。

严重度同时考虑可达性、最坏影响、检测能力、现有保护和恢复成本；不是仅按代码风格或与参考项目的差异定级。

### 2.2 Confirmed 判定标准

问题只有在至少满足下列一种证据链时才列为 confirmed：

1. 源码可达调用链和分支条件足以确定错误状态；
2. mock/纯函数输入能够稳定复现，且输出违反明确 API、排列、几何或生命周期不变量；
3. 只读 HDF5 数据与源码时序共同支持同一结论；
4. 有限差分、双向 round-trip 或边界测试能够证实数值不变量被破坏。

仅有“参考仓库实现不同”、缺少厂商单位文档、依赖真机接触力或任务成功率才能判断的项目，统一放入第 6 节，不计入 confirmed 数量。

## 3. 审查范围与版本

### 3.1 主项目调用链

SDK 链路：

```text
main supervisor
  -> hand_loop(shared, config)
    -> XHand.connect()
      -> enumerate/open EtherCAT or RS485
      -> list_hands_id
      -> make_command
      -> forced read_state
      -> reset_sensor + tactile bias
    -> home convergence
    -> latest-wins hand_cmd_ring
      -> XHand.send_action()
    -> XHand.get_state(force_update=True)
      -> hand_state_ring (dense)
      -> hand_tactile_ring (contact-only sparse)
```

Retarget 链路：

```text
HTS HandFrame
  -> vr_receiver_process: Unity-left -> FLU
  -> vr_ring
  -> policy._compute_hand_command()
    -> TAGHandRetargeter or XHandRetargeter(DexPilot)
      -> palm frame + operator->MANO
      -> adaptive pinky scaling
      -> optimizer
      -> model/internal order -> SDK order
      -> EMA
    -> finite/limit/delta sanitize
    -> arm-hand transition collision check
    -> hand_cmd_ring
```

### 3.2 固定对照版本

| 项目 | Commit | 用途 |
|---|---|---|
| DexMani Real | `167f15a` | 被审查实现 |
| LeFranX | `a39906e` | Quest landmarks、operator→MANO、DexPilot、pinky scaling、stub 语义 |
| DexUMI | `acddb8f` | XHand SDK 状态读取、命令参数、触觉采集、30 Hz 独立 reader |
| Dexora | `6f2869d` | 发送节流、RS485 独立硬件代理和错误重试 |
| TAG | `2f5b1ab` | 两阶段 NLopt、Pinocchio Jacobian、pinch refinement、model→SDK mapping |

参考实现只用于验证 SDK 契约、坐标约定和设计意图。实现差异本身不构成缺陷。

## 4. Confirmed findings

## 4.1 XH-SDK-01 — post-open 失败路径没有统一清理

**严重度：P1**

### 位置与可达调用链

- `dexmani_real/robot/xhand.py:195-250`：`connect()` 在 `_retry_open_device()` 成功后设置 `connected_flag=True`，随后无 `try/finally` 地执行 `_verify_device()`、`make_command()` 和 `_init_hand_state()`；
- `dexmani_real/robot/xhand.py:225-236`：device id 不存在时只调用一次 `close_device()`，没有请求 EtherCAT INIT，也没有 watchdog wait；
- `dexmani_real/robot/xhand.py:372-403`、`459-468`：初始化会进入 tactile verify 的原生 `read_state()`，该调用抛异常时会直接穿透 `connect()`；
- `dexmani_real/robot/hand_process.py:74-105`：worker 的外层初始化异常分支只标记失败并 `return`，没有调用 `hand.disconnect()`；
- `dexmani_real/robot/xhand.py:670-693`：完整清理只在 `connected_flag=True` 时执行 `_request_slave_init()`、`close_device()` 和等待。

调用链：

```text
hand_loop
  -> XHand.connect
    -> open succeeds
    -> connected_flag = True
    -> tactile verify read_state raises
  -> outer except
  -> mark startup failure + return
  -> no stop/disconnect
```

device-id 不匹配是另一条路径：

```text
open succeeds
  -> list_hands_id excludes configured id
  -> direct close_device
  -> connect returns False
  -> hand_loop calls disconnect
  -> disconnect skips because connected_flag=False
  -> no INIT request and no watchdog wait
```

### 确定性复现

用 mock control 让 open 和 `list_hands_id()` 成功，再让 `_init_hand_state()` 抛异常：

```text
post_open_exception_connected_flag True close_calls 0
```

让设备列表不包含 configured id，并统计 canonical INIT 清理调用：

```text
id_mismatch_ok False close_calls 1 init_calls 0
```

### 影响

驱动自身在 `xhand.py:711-720` 已明确记录：不干净的上一会话可能导致后续 SDO 错误，并需要 XHand 24 V 断电重启。异常发生在 open 之后、ready 之前时，主进程会进入 FAULT，但无法保证 EtherCAT slave 已回到可重连状态；若固件仍保持上次目标，还存在“进程已退出、设备状态未完成显式 stop”的额外风险。

### 现有保护为何不足

- `hand_loop` 仅在 `connect()` 正常返回 `False` 的分支做 disconnect；异常分支没有资源所有权清理；
- `disconnect()` 又以 `connected_flag` 作为是否清理原生 handle 的条件；
- device-id mismatch 在 `connected_flag=True` 之前发生，因此后续 disconnect 无法补偿；
- OS 回收 socket 不等同于 SDK 级 INIT 状态转换和 SM-watchdog 等待。

### 最小修复设计

1. 在 `XHand.connect()` 内部建立单一 `_abort_open_device()`，只要 native open 已成功，无论 `connected_flag` 是否已发布，都执行 best-effort：stop/INIT、close、watchdog wait；
2. 用 `try/except` 包住 open 后的全部初始化；异常时先 `_abort_open_device()`、清空 `control/hand_command/connected_flag`，再重新抛出或返回 `False`；
3. `hand_loop` 的初始化异常分支增加 best-effort `hand.disconnect()`，作为第二层兜底；
4. 清理函数必须幂等，避免 failed-open handle 被重复 close。

### 回归测试

- open 成功后分别在 `list_hands_id`、`make_command`、initial read、tactile reset、bias read 注入异常；
- 断言 INIT/close 被调用至多一次，`connected_flag=False`，control 不可继续使用；
- device-id mismatch 必须走同一清理路径；
- cleanup 自身异常不能遮蔽原始启动错误；
- 真机验收见第 9 节：连续 20 次启动失败/重启不得要求断电恢复。

## 4.2 XH-SDK-02 — 持续发送失败不会升级为全局 sticky fault

**严重度：P1**

### 位置与可达调用链

- `dexmani_real/robot/xhand.py:824-872`：`send_action()` 失败时设置 `error_state=True` 并增加 `_consecutive_send_errors`；
- `dexmani_real/robot/xhand.py:807-821`：下一次健康 `read_state()` 会把 `error_state` 重写为“当前 board registers 是否非零”，从而清除 transport/send error；
- `dexmani_real/robot/xhand.py:746-750`：`clear_error()` 只清 Python 字段，没有向 SDK/固件发恢复命令；
- `dexmani_real/robot/hand_process.py:220-248`：send watchdog 达阈值后只周期调用上述本地 `clear_error()`；
- `dexmani_real/robot/hand_process.py:334-355`：只要本 tick `error_state=False`，board/error counter 就 reset；
- `dexmani_real/robot/hand_process.py:1-5` 的模块说明称三类 counter 都会升级全局错误，但 send counter 实际没有该行为。

调用链：

```text
policy writes new command
  -> send_command returns error
  -> driver.error_state=True
  -> same hand-loop tick read_state succeeds
  -> driver.error_state = board_registers_nonzero = False
  -> error_state watchdog resets
  -> send counter reaches threshold
  -> only local clear_error(); no shared.error_state latch
  -> repeat forever
```

### 确定性复现

mock `send_command()` 返回 code 7，再让 `read_state()` 返回 code 0、board registers 全零：

```text
send_ok False error_after_send True
error_after_healthy_read False consecutive_send_errors 1
```

### 影响

当总线只允许读、不接受写，或 `send_command()` 持续失败但 `read_state()` 正常时：

- XHand 不执行新动作；
- `hand_connected` 仍为真；
- 全局 `error_state` 不锁存，main 不会可靠进入 FAULT；
- arm 仍可继续运动；若手正在抓物体，可能发生滑落、碰撞或数据污染。

### 现有保护为何不足

- qpos stale 不是可靠替代：它本身有 XH-SDK-03 的静止误报，而且只让 policy hold 手，不会协调停止 arm；
- `clear_error()` 没有执行厂商恢复动作；
- board error、read error、send error 共用一个布尔 `driver.error_state`，健康 read 覆盖了另一故障域的状态；
- driver 的 `_consecutive_send_errors` 和 worker 的 `_send_error_counter` 重复计数，但两者都没有形成 sticky supervisor 信号。

### 最小修复设计

1. 分离 `transport_read_error`、`transport_send_error` 和 `board_error`，健康 read 只能清 read/board 的瞬时状态，不能清 send fault；
2. worker send counter 达到 `send_err_watchdog_frames` 后直接设置 `shared.error_state=True`；
3. 若厂商 SDK 没有真正的 clear API，不要把本地字段清零命名为恢复；可在有限次数重发后 fail closed；
4. 在 state dtype/HDF5 中兼容新增 `last_send_ok`、`consecutive_send_errors` 或 health bit，旧 episode reader 默认缺失字段为 unknown；
5. 日志记录错误码、首次失败时间、最后成功 send seq。

### 回归测试

- send 连续失败、read 连续成功：必须在阈值处锁存全局 fault；
- send/read 交替失败：任何一个持续故障域不能被另一个成功域清除；
- 单次失败后恢复：counter 清零但不误触发；
- board error 自动清除仍保持原有瞬时语义；
- shared sticky fault 只能由 supervisor/正式复位流程清除。

## 4.3 XH-SDK-03 — 静止姿态误判 stale 并形成 hold 自锁

**严重度：P1**

### 位置与调用链

- `dexmani_real/config/defaults.py:74-80`：15 帧、`1e-4 rad` 的 qpos delta 阈值；
- `dexmani_real/robot/hand_process.py:313-327`：只比较相邻/基准 qpos 是否变化，没有考虑命令是否要求运动、反馈读取是否成功或 SDK 帧是否新鲜；
- `dexmani_real/policy/vr_teleop_policy.py:1218-1224`：一旦 stale，强制把手命令回退为 `prev_hand_qpos` 并标记 retarget 失败。

### 触发条件

XHand 正常连接、状态读取正常，且物理手在约 0.5 s 内稳定保持，所有关节变化小于 `1e-4 rad`。这既可能发生在操作员静止，也可能发生在开手、稳定抓持或 policy 暂时输出恒定目标时。

### 为什么会自锁

1. 静止反馈累计到 15 帧，`qpos_stale=True`；
2. policy 因 stale 改发 hold；
3. hold 使物理 qpos 更不可能变化；
4. worker 的 stale counter 只有 qpos 变化才清零；
5. 因而 stale 可永久保持，除非编码器噪声超过阈值或外力移动手指。

### 影响

正常静止被解释成 driver board lockout，retarget 停止、数据标为失败，控制在本应继续可用时冻结。当前可用录制 `episode_20260808_230538` 的 427 帧均未触发 stale，说明实际 encoder/PID 抖动在该段数据中避免了误报；这不改变静止输入下的确定性逻辑错误。

### 现有保护为何不足

- `force_update=True` 只保证发起硬件刷新，不代表“qpos 必须变化”；
- heartbeat 证明进程活着，不证明返回的是新 SDK 帧；
- policy hold 是安全降级，但没有恢复条件，反而强化误报状态。

### 最小修复设计

优先把“反馈新鲜度”和“执行跟踪”拆开：

- `state_stale`：由 SDK read 成功、SDK/host receive timestamp、序号或实际帧 age 判定；
- `tracking_stalled`：只有在最近成功接受的命令与测量 qpos 存在显著误差、命令确实要求运动且误差长期不收敛时才置位；
- 稳态条件 `max(abs(last_accepted_cmd - qpos)) <= tolerance` 必须清空 stall counter；
- 若 SDK 没有序号，至少用“成功 force-read + 命令误差”替代“qpos 必须变化”。

### 回归测试

- 连续 100 帧相同 qpos、相同已收敛命令：永不 stale；
- 连续变化命令、反馈固定且 tracking error 大：达到阈值后 stalled；
- 命令改变后反馈逐步收敛：counter 持续复位/下降，不误报；
- read 返回缓存帧或 timestamp 不变：`state_stale` 触发；
- hold 后恢复一帧有效跟踪：状态可自动恢复，不自锁。

## 4.4 XH-RT-01 — 几何退化 landmarks 被接受并产生显著动作

**严重度：P1**

### 位置与调用链

- `dexmani_real/sensor/vr_receiver_process.py:107-135`：右手 HandFrame 只做类型检查、坐标转换和 `(21,3)` reshape，没有 tracking validity、confidence 或几何检查；
- `dexmani_real/teleop/hand_retarget.py:53-109`：掌坐标估计在 x 轴长度退化、SVD/叉积退化时返回 identity；
- `dexmani_real/teleop/hand_retarget.py:250-304`：DexPilot 只检查 shape 和 finite；
- `dexmani_real/teleop/hand_retarget.py:508-565`：TAG 同样只检查 shape 和 finite；
- 安装的 `hand_tracking_sdk/parser.py:127-140,180-208` 只验证 float 可解析和 63 个数值，`float("nan")`/全零/共点几何没有被协议层拒绝；
- `dexmani_real/policy/vr_teleop_policy.py:133-152` 之后只做关节范围和每帧 `0.20 rad` delta clamp；
- `dexmani_real/planning/collision_model.py:9-11,130-132` 明确没有任何 hand-hand collision pair。

### 确定性复现

对新鲜的 `np.zeros((21,3))`：

```text
_estimate_palm_frame -> identity

TAG output (rad):
[ 1.423371, 0.533880, 1.061928, -0.044816,
  1.153435, 1.130161, 1.163222, 1.133484,
  1.141751, 1.127639, 1.120298, 1.121335 ]

DexPilot output (rad):
[ 1.430996, 1.354878, 0.173500, -0.175000,
  0.253952, 1.870850, 0.316061, 1.920000,
  0.277730, 1.901994, 0.249874, 1.866787 ]
```

两条 backend 都返回非 `None` 且有限，因此 `_compute_hand_command()` 把它们标为 `retarget_ok=True`。最高目标约为 81.6°（TAG）和 110°（DexPilot）。

### 影响

新鲜但损坏/跟踪退化的一帧可驱动手向显著闭合状态运动。policy 的 0.20 rad/frame 限制会把大跳变摊到若干帧，但不会识别目标是错误的；在 16 Hz 下，约 7–10 帧即可接近上述大角度目标。hand-hand collision 被设计为关闭，无法拒绝指间错误闭合。

### 现有保护为何不足

- VR freshness 只排除旧帧，不排除新鲜坏帧；
- finite 只排除 NaN/Inf；
- identity fallback 把“无法建立掌坐标系”转换成“看似有效的坐标系”；
- 关节 clip、delta clamp、EMA 和 startup ramp 只限制速度/幅度，不验证意图；
- arm-hand collision 只防手撞臂，不防错误手形。

### 最小修复设计

1. 增加共享纯函数 `validate_hand_landmarks()`，在 VR producer 和两个 retargeter 边界复用；
2. 至少验证：wrist→middle MCP 长度、index↔pinky palm width、MCP 三角形面积/条件数、逐骨段长度范围、左右手 chirality、相邻帧尺度和速度；
3. `_estimate_palm_frame()` 退化时抛出明确异常或返回 `None`，不得回退 identity；
4. 若上游将来提供 confidence/tracked bit，将其加入 VR dtype，并保留旧 schema 兼容默认 unknown；
5. 任何几何失败都保持上一条有效命令，记录 `retarget_ok=False` 和具体原因。

可用录制的稳定几何可作为初始阈值依据：wrist→middle MCP 约 95.65 mm，index MCP↔pinky MCP 约 61.76 mm；阈值仍应从多用户数据分布确定，不能只拟合一个 episode。

### 回归测试

- 全零、全部共点、三点共线、掌宽接近零、单骨段零长度：必须返回 hold；
- 整体平移/旋转：验证结果不变；
- 合理手大小范围内整体尺度变化：允许；
- 单帧不合理 0.5 m 跳变：拒绝；
- TAG/DexPilot 对失败输入不得改变 optimizer/filter 的 temporal state。

## 4.5 XH-TACT-01 — release 后 raw tactile 与 contact 状态不一致

**严重度：P2**

### 位置与调用链

- `dexmani_real/robot/hand_process.py:372-379`：raw `(5,120,3)` ring 仅在任一 finger contact 时写入；
- `dexmani_real/policy/vr_teleop_policy.py:908-918`：没有新 raw frame 时无限 forward-fill 上一帧；
- `dexmani_real/policy/vr_teleop_policy.py:1728-1769`：dense `tactile_contact` 来自当前 hand state，raw force 来自独立 sparse ring；
- `dexmani_real/recording/episode_recorder.py:396-404`：两者未经 age/sequence 一致性校验写入同一 HDF5 sample。

时序：

```text
t0 contact=True  -> sparse raw A written
t1 contact=False -> no sparse write; policy reuses A
t2 contact=False -> policy still reuses A
t3 contact=True  -> new sparse raw B written
```

### 录制数据证据

在 `episodes/episode_20260808_230538/data.h5`（schema v12，427 帧）中：

- 114 帧存在 contact；
- 6 次 contact→release 转换；
- 313 个 no-contact 帧中，71 帧记录了非零 raw tactile；
- 其中一次 release 后 raw frame 与最后接触帧完全相同并持续 132 个 policy sample，另一次持续 59 个 sample。

这不是单纯的“no-contact 也可能有传感器噪声”：逐字节/逐元素完全相同的长时间平台与代码的 forward-fill 语义一致。

### 影响

- HDF5 同一时刻的 `hand_tactile_contact=False`、dense force sum 和 raw taxel data 可能不属于同一采样时刻；
- contact→release→contact 的学习数据会把旧接触纹理错误标到 release 段；
- 质量工具无法从现有 raw dataset 判断数据 age；
- 如果以后将 raw tactile 接入 TAG pinch gate，陈旧接触会直接影响控制。

### 最小修复设计

最小无 schema 方案：hand worker 在 contact falling-edge 额外写一次当前 raw frame，使 release 有明确终止样本。更完整方案：

- `HAND_TACTILE_DTYPE` 增加 `timestamp/sequence/valid/contact_mask`；
- policy 只在 raw age 小于阈值且 sequence 与 dense hand state 可对齐时使用；
- recorder 增加 `hand_tactile_age_s` 或 `hand_tactile_fresh`；
- schema 升级并让旧 reader 在字段缺失时返回 unknown，而不是伪造 fresh。

### 回归测试

- contact A→release→持续 release：release 第一帧后不得出现 A 的无限 forward-fill；
- 五指分别 release；
- read exception 与真实 release 必须可区分；
- HDF5 round-trip 保留 raw timestamp/freshness；
- 老 schema 无 freshness 字段仍可读取。

## 4.6 XH-TACT-02 — 启动 bias 可吸收真实接触

**严重度：P2**

### 位置与调用链

- `dexmani_real/robot/xhand.py:397-403`：每次 connect 都自动 reset 五个 tactile sensor；
- `dexmani_real/robot/xhand.py:440-506`：reset/verify 后无条件进入 software bias；
- `dexmani_real/robot/xhand.py:515-550`：docstring 明确要求“hand must NOT be in contact”，实现只平均 5 个 fresh sample，没有验证该条件；
- `dexmani_real/robot/xhand.py:1032-1076`：后续每帧都减去该 bias。

### 触发条件与确定性

如果启动时某 finger 承受稳定真实载荷 `F_contact`，采集均值近似：

```text
bias = sensor_offset + F_contact
reported = sensor_reading - bias
```

同一接触持续时，reported 约为 0。代码不可能仅凭“5 帧稳定”区分稳定 offset 和稳定真实接触；多次 `reset_sensor()` 还可能在硬件层先把接触归零。

### 影响

- 当前会话该 finger 的触觉绝对力系统性偏低；
- contact threshold 可能永不触发；
- raw/dense HDF5 记录错误；
- tactile-based quality 或未来闭环 pinch gate 获得错误零点。

### 现有保护为何不足

verify threshold 检查发生在 reset 后，而且最后仍会用软件 bias 补偿残余；它不能证明传感器处于无接触状态。docstring/日志提示不是运行时前置条件，canonical worker 也没有等待操作员确认或发布 calibration-valid 状态。

### 最小修复设计

1. 在 reset 前读取并保存 pre-reset force；明显高载荷时拒绝自动校零并发布 `tactile_calibrated=False`；
2. 把 tactile zeroing 变成显式、可确认的启动阶段，而不是 `connect()` 的隐式副作用；
3. 保存 bias、校零时间和校零有效标志到 state/episode metadata；
4. 若无法确认无接触，使用最近一次持久化、经验证的 calibration，或禁用绝对接触判定；
5. 真机流程明确要求手指悬空、无物体接触。

### 回归测试

- offset-only 数据可被正确归零；
- 稳定 5 N 接触不得被当作 bias 接受；
- 单个 sensor read 失败不产生 NaN bias；
- calibration-invalid 状态必须进入 HDF5/quality report；
- 重新校零前后 contact threshold 单位一致。

## 4.7 XH-RT-02 — pinky scaling 会翻转中间骨段

**严重度：P2**

### 位置与调用链

- `dexmani_real/teleop/hand_retarget.py:112-157`；
- 问题集中在 `148-155`：先覆盖 PIP，再用“原 DIP - 已修改 PIP”计算 PIP→DIP；随后同样混用原 TIP 和已修改 DIP；
- DexPilot `_build_ref_value()` 在 `230-248` 使用该结果；
- TAG `retarget()` 在 `531-537` 使用同一结果。

当前实现：

```python
PIP_new = MCP + (PIP_old - MCP) * s
DIP_new = PIP_new + (DIP_old - PIP_new) * s
TIP_new = DIP_new + (TIP_old - DIP_new) * s
```

第二、三项不是原始骨段向量。若 `PIP_new` 越过 `DIP_old`，下一段立即反向。

### 确定性复现

对共线 pinky：MCP=0、PIP=0.03、DIP=0.06、TIP=0.10 m，scale=2.2：

```text
scaled joint x = [0.00000, 0.06600, 0.05280, 0.15664]
scaled segment = [0.06600, -0.01320, 0.10384]
```

原本正向的 PIP→DIP 变为 `-13.2 mm`。

真实录制 `episode_20260808_230538` 的 427 个 VR 帧中，缩放后的 PIP→DIP 与原骨段点积为负的帧数为 **297/427**。

### 参考项目证据

LeFranX `src/lerobot/teleoperators/xhand_vr/vr_hand_detector_adapter.py:27-84` 含相同实现，因此能说明本项目的来源，但不能证明几何是正确的；这里以方向不变量和真实数据复现确认缺陷。

### 影响

- DexPilot 的差向量目标和 TAG 的 pinky tip target 被非刚性、非单调几何污染；
- pinky 可能出现过度屈曲、方向抖动或触达误差；
- TAG 对当前 episode 的 action FK→target pinky 中位误差约 32.7 mm、P95 约 62.1 mm。该误差还包含可达性、滤波和优化设计，不能全部归因于此 bug，但反向骨段是明确输入污染。

### 最小修复设计

在写任何新坐标前保存原始链向量：

```python
v_mcp_pip = old[PIP] - old[MCP]
v_pip_dip = old[DIP] - old[PIP]
v_dip_tip = old[TIP] - old[DIP]

new[PIP] = old[MCP] + s * v_mcp_pip
new[DIP] = new[PIP] + s * v_pip_dip
new[TIP] = new[DIP] + s * v_dip_tip
```

如果真实意图只是扩大 MCP→各 joint 的相对位置，也可统一使用 `MCP + s*(joint_old-MCP)`；两种方案必须通过离线任务指标选择，不能混用新旧父节点。

### 回归测试

- 每个新骨段与对应旧骨段点积必须非负；
- 每段长度应等于 `s * old_length`（选择逐段方案时）；
- 输入数组不被原地修改；
- scale=1 时严格 identity；
- 从 curled 到 extended 扫描时 target 连续、单调；
- 在 427 帧 episode 上方向反转计数必须从 297 降为 0，并重新评估 pinky FK 误差。

## 4.8 XH-RT-03 — DexPilot warm-start 使用错误的逆映射

**严重度：P2**

### 位置与调用链

- `dexmani_real/teleop/hand_retarget.py:213-216`：`retargeted_joint_order` 的语义是“对每个 SDK joint，找到 internal robot index”；
- `dexmani_real/teleop/hand_retarget.py:294-295`：输出 `q_internal[retargeted_joint_order]`，正确得到 SDK order；
- `dexmani_real/teleop/hand_retarget.py:333-340`：reset 时从 SDK 返回 internal 却再次使用同一正向 mapping，之后又按 `idx_pin2target` subset；
- installed `dex_retargeting/seq_retarget.py:128-130` 已提供 `set_qpos(robot_qpos)`，其输入明确是 full robot/internal order。

当前 mapping：

```text
SDK-output mapping = [9,10,11,0,1,2,3,4,7,8,5,6]
inverse mapping    = [3,4,5,6,7,10,11,8,9,0,1,2]
```

reset 实际形成近似 `q_sdk[mapping][idx_pin2target]`，对排列应用了两次正向变换。

### 确定性复现

用中央 home pose 调用 `XHandRetargeter.reset(home_qpos)`：

```text
actual seed deg:
[ 6.76, 5.00, 6.53, 5.00, 10.13, 5.00,
  0.00, 80.66, 5.11, 5.00, 33.20, 0.00 ]

expected target-order seed deg:
[ 0.00, 80.66, 33.20, 0.00, 5.11, 5.00,
  6.53, 5.00, 6.76, 5.00, 10.13, 5.00 ]

max seed error = 75.66 deg
```

### 影响

该问题不直接改变稳态 joint order；常规 `retarget()` 输出映射是正确的。它只破坏 B-press、resume、audio gate 结束时的 optimizer warm-start，导致：

- 首帧 SLSQP 从错误关节姿态开始；
- 收敛迭代和时延增加；
- temporal regularization 锚定错误姿态；
- 某些输入下首帧输出偏差更大。

policy startup ramp 和 0.20 rad delta clamp 限制了物理跳变，所以定为 P2 而不是 P1。

### 最小修复设计

首选使用依赖库的公开语义：

```python
q_internal = q_sdk[np.argsort(self.retargeted_joint_order)]
self.retargeter.set_qpos(q_internal)
```

然后 reset LPFilter、projected flags 和 teleoperator EMA。若直接写 `last_qpos`，应明确它是 target-joint order；当前 YAML 的 target joint names 恰为 SDK order，但依赖这一偶然关系不如 `set_qpos()` 稳健。

### 回归测试

- 任意 12 维唯一值的 SDK→internal→SDK round-trip；
- home、joint midpoint、随机 bounds 内姿态 reset 后，`retargeter.get_qpos()` 必须回到同一 SDK pose；
- reset 前后第一帧 LPFilter 不带旧 episode 状态；
- TAG 和 DexPilot 共用同一 mapping invariant 测试。

## 4.9 XH-ARCH-01 — TAG 在 policy 进程连带导入厂商 SDK

**严重度：P2**

### 位置与调用链

- `dexmani_real/policy/vr_teleop_policy.py:424-441`：policy 初始化 `TAGHandRetargeter`；
- `dexmani_real/teleop/hand_retarget.py:443-458`：TAG 为合并 driver limits 导入 `dexmani_real.robot.xhand.XHandConfig`；
- `dexmani_real/robot/xhand.py:13-19`：模块顶层立即尝试导入 `xhand_controller.xhand_control` 原生扩展。

```text
policy process
  -> TAGHandRetargeter.__init__
    -> import robot.xhand.XHandConfig
      -> import vendor xhand_controller native module
```

这违反仓库不变量：“只有 hand worker 导入/使用 XHand SDK”。TAG 只需要中央 `hand.qpos_min_rad/qpos_max_rad`，并不需要 driver class。

### 影响

- policy 的启动被厂商 `.so` 的 ABI、依赖库和导入副作用耦合；
- native module import 异常会让 retargeter 初始化降级，即使问题与优化器本身无关；
- 扩大 SDK crash/全局符号冲突的进程范围；
- 使 offline TAG 测试无谓触碰硬件 SDK 包边界。

当前路径不会实例化 `XHandControl`，本次离线运行也未连接设备，因此直接风险低于前述 P1，但架构边界是确定违反的。

### 最小修复设计

`TAGHandRetargeter` 已导入 `defaults.hand as hand_d`，直接使用：

```python
driver_lo_sdk = np.asarray(hand_d.qpos_min_rad, dtype=np.float64)
driver_hi_sdk = np.asarray(hand_d.qpos_max_rad, dtype=np.float64)
```

限位的 source of truth 本来就在 `defaults.py`；driver 和 policy 都从此读取即可。

### 回归测试

- 在 `xhand_controller` 明确不可导入的环境初始化 TAG，必须成功；
- TAG 初始化后断言 `xhand_controller` 不在 policy process 的 `sys.modules`；
- driver limits 与 TAG optimizer bounds 的 intersection 测试保持不变。

## 4.10 XH-SDK-04 — 隐式 stub 语义互相矛盾

**严重度：P3**

### 位置与调用链

- `dexmani_real/robot/xhand.py:13-19,195-212`：SDK ImportError 自动进入 stub，`connect()` 返回 True、`connected_flag=True`；
- `dexmani_real/robot/xhand.py:725-741`：stub state 永远返回零 qpos，但 `is_connected()` 因 `control is None` 返回 False；
- `dexmani_real/robot/hand_process.py:107-140`：canonical worker 随后用非零 home pose 做收敛判断；
- `dexmani_real/config/defaults.py:204-217`：home 最大关节为 80.66°，并非全零。

### 确定性复现

```text
stub_connect True
is_connected False
max(abs(stub_qpos - home_qpos)) = 1.4078 rad
```

worker 会在 3 s homing timeout 后把本次启动标为失败。因此该 stub 既不是“可运行的仿真设备”，也不是明确的 fail-closed 缺 SDK 状态。

### 影响

- API 使用者看到 `connect=True` 却无法通过 `is_connected()`；
- 日志先称进入 stub，随后又以 home failure 退出，诊断混乱；
- 若其他调用者绕过 homing，command 会更新 `last_qpos_cmd`，但反馈仍永远为零，数据语义虚假。

### 最小修复设计

- canonical hardware worker 默认 `allow_stub=False`，SDK 缺失直接启动失败；
- 只有显式 CLI/config 才允许 stub；
- 真正 stub 的反馈应跟随 joint-limited `last_qpos_cmd`，`connect/is_connected/get_state/disconnect` 语义一致；
- 录制 metadata 明确标记 `hardware_mode=stub`，防止混入真机数据。

### 回归测试

- SDK 缺失且未显式允许：fail closed；
- 显式 stub：connect/is_connected 一致，home 可收敛，send→read round-trip；
- stub episode 必须有不可混淆的 metadata 标志。

## 5. 已复核为正确或基本合理的部分

以下候选经过复核，没有作为 finding：

### 5.1 SDK 命令与状态基本契约

- `read_state(device_id, True)` 使用 fresh read，和 DexUMI `hand_api_cls.py:171-175` 的做法一致；
- `parse_state()` 按 `finger_state.item.id` 放入 SDK joint slot，能抵抗返回列表重排；
- position 单位全链路使用 rad；`torque` 字段按参考实现解释为电流/mA；
- mode 3、`kp=100`、`ki=kd=0`、`tor_max=300/380` 与厂商例程/TAG/DexUMI 的使用区间一致；具体增益仍需真机调参，但未发现单位或 joint id 级别的代码错误；
- command 在 policy 和 driver 两层做 finite、joint limit、per-step delta 限制。

### 5.2 TAG joint order 与 bounds

- 当前 Pinocchio model→SDK mapping：`[9,10,11,0,1,2,3,4,7,8,5,6]`，与 TAG `Realtime_Retargeting_xhand.py:169-181` 一致；
- `_mapping_sdk_to_model=np.argsort(...)` round-trip 精确成立；
- bounds 使用 URDF 与中央 driver strict bounds 的交集，避免 optimizer 生成 driver 随后会 clip 的不可达角；
- TAG reset 在 bounds 内测试中 SDK→model warm-start round-trip 误差为 0。

### 5.3 Pinocchio Jacobian

对 12 个 DOF 做中心有限差分，结果：

```text
loss = 2.7e-05
max absolute gradient error = 4.37e-14
relative L2 gradient error = 1.13e-10
```

因此 `LOCAL_WORLD_ALIGNED` Jacobian、FreeFlyer 6 列裁剪和 joint order 没有发现错误。

### 5.4 Quest→MANO→URDF 当前旋转

在已有 427 帧 Quest episode 上，用 recorded action FK 对比当前 target：

| target rotation | 全指中位误差 |
|---|---:|
| 当前 identity | 24.7 mm |
| 直接套 TAG glove reference rotation | 278.7 mm |
| 套其转置 | 282.1 mm |

TAG 原项目 `config_xhand.py:46` 的 `[0, π/2, π/2]` 是手套 FK/root 的坐标约定，不能直接迁移到已经经过 Quest operator→MANO 的 target。当前 identity 明显更符合现有 Quest 数据，因此不判定为坐标系 bug。

### 5.5 优化器失败策略

- TAG Stage 1 失败返回 `None`，caller hold；Stage 2 失败回退 Stage 1；
- invalid/non-finite optimizer result会被 bounds helper 或 policy sanitizer 拒绝；
- policy 在 IK 失败时仍对 hand-only motion 做 arm-hand transition collision check；
- FAULT gate、startup ramp、EMA 和 0.20 rad/frame 限制均存在。

这些保护降低了错误动作速度和跨子系统碰撞风险，但不能替代 XH-RT-01 的输入有效性检查。

## 6. 尚不能仅凭源码定性的风险

以下项目需要厂商文档或真机数据，不列入 confirmed finding 数量：

### 6.1 Tactile `0.1` 缩放单位

`xhand.py:1032-1076` 假设 `raw_force/calc_force` 为 10 LSB/N 并乘 0.1。DexUMI、pi-r2-flow 和 TAG 直接记录 SDK 数值，没有统一物理单位注释；当前项目的 1 N threshold 又引用 DexUMI raw cutoff 10，间接支持 0.1，但缺少本地权威 SDK 规格。必须用标定砝码或厂商手册确认，不能仅凭参考仓库是否乘 0.1 判错。

### 6.2 EtherCAT slave position 固定为 1

`xhand.py:645-649` 的 `set_firmware_state(device_id, 1, INIT, ...)` 固定 slave position 1。单手、单 slave 总线可能正确；多 slave 或非首位置场景需厂商确认 device-id 与 slave-position 的关系。

### 6.3 TAG Stage 2 零距离 pinch 与无 tactile gate

`optimizer.py:343-360` 用 `p_finger-p_thumb -> 0`，权重 2000；当前 hand-hand collision 关闭，tactile 不参与控制。已有 episode 中 thumb-index 输出 frame 最小距离约 0.9 mm，多个 joints 经常接近 bounds。这可能是 fingertip frame 定义下的预期接触，也可能导致真机猛夹/过载。需要物体接触、触觉峰值和掉落率实验后定级。

### 6.4 TAG 绝对 fingertip error 与 bounds 饱和

当前 427 帧重跑无 optimizer failure，median/p95 solve time 约 0.19/0.85 ms；但 raw solve 的五指中位 FK target error 约 `[10.8,19.1,19.1,26.7,31.7] mm`，pinky P95 约 60.7 mm，若干 joints 大量贴近 bounds。这既可能来自人机长度/基座 offset 的固定标定不足，也可能是不可达目标和 position-only objective 的正常残差。建议以任务成功率和多用户标定数据决定是否调整 finger lengths、lateral scaling、方向项或 bounds，而不是直接修改旋转。

### 6.5 Finger temperature 低字节

厂商例程读取 `temperature & 0xFF`，当前 `parse_state()` 直接转 float。该字段目前没有进入 shared state/HDF5，也不参与安全控制，所以不构成当前运行缺陷；若未来发布温度，需要先验证符号位/packed register 契约。

## 7. 验证记录

### 7.1 已运行

1. 固定版本和工作树检查；
2. 参考仓库 commit 校验；
3. SDK mock 故障注入：
   - post-open init exception；
   - device-id mismatch；
   - send error + healthy read；
   - SDK unavailable stub；
4. retarget 数值复现：
   - 全零/共点 landmarks 的 TAG 与 DexPilot 输出；
   - pinky 共线链方向反转；
   - DexPilot warm-start mapping；
   - TAG SDK↔model mapping round-trip；
   - Pinocchio Jacobian 中心有限差分；
5. HDF5 只读复核：
   - `episodes/episode_20260808_230538/data.h5`，427 帧，schema v12；
   - pinky reversal、tactile release、坐标候选、FK error、bounds、pinch distance；
6. 非硬件测试：

```text
conda run -n real_robot pytest -q \
  tests/test_hand_optimizer_bounds.py \
  tests/test_return_home.py \
  tests/test_collision_safety.py \
  tests/test_teleop_responsiveness.py

44 passed in 1.33s
```

原计划中的两个基线文件包含在上述命令内，且通过；扩展加入 collision 与 teleop responsiveness 后总计 44 passed。

### 7.2 未运行

- 任何 `examples/real/` 硬件脚本；
- XHand EtherCAT/RS485 连接、enumerate、read、send、reset_sensor；
- VR 实时流、遥操作、回放、归位、校准；
- RealSense；
- 厂商 SDK 真机错误码和断连恢复；
- 实物 pinch、抓取、力标定和温升测试。

## 8. 推荐修复顺序

### 第一批：先修 fail-closed 和错误动作入口

1. XH-RT-01：退化 landmarks fail closed；
2. XH-SDK-02：send watchdog 锁存全局 fault，拆分错误域；
3. XH-SDK-01：统一 post-open cleanup；
4. XH-SDK-03：重定义 freshness 与 tracking stall。

这四项应作为一个安全回归批次，覆盖 worker death、heartbeat、sticky fault 和 policy hold/recovery。

### 第二批：修数据与几何确定性

5. XH-RT-02：pinky 原始骨段向量；
6. XH-RT-03：DexPilot inverse mapping/`set_qpos()`；
7. XH-TACT-01：release frame + tactile timestamp/freshness；
8. XH-TACT-02：显式 tactile calibration state。

若增加 dtype/HDF5 字段，必须按 schema v14 或后续版本兼容扩展，旧 episode reader 对可选字段给出 unknown/default，不能改变现有 dataset 含义。

### 第三批：清理边界和开发语义

9. XH-ARCH-01：TAG 直接读取 defaults，policy 不加载 vendor SDK；
10. XH-SDK-04：stub 必须显式启用且行为自洽。

## 9. 最小真机验收清单

在完成代码修复和 mock 测试后，真机验收需单独授权，并确保工作区清空：

1. **连接生命周期**：连续 20 次 connect→graceful disconnect→reconnect；在 tactile init 和 list-hands 阶段分别模拟失败，均无需 24 V power-cycle；
2. **send fault**：安全悬空状态下阻断写通道但保持 read，确认阈值内进入全局 FAULT，arm/hand 协调 hold；
3. **静止状态**：open hand、稳定抓持各静止 10 s，不得 stale；再模拟 command 变化但 feedback 固定，必须 stall；
4. **VR tracking loss**：遮挡、脱手、全零/冻结/突跳 packet，手必须保持上一有效姿态；
5. **pinky**：从完全弯曲到伸直缓慢扫描，记录每骨段方向、target、SDK command 和 measured qpos，不得出现非意图反向；
6. **tactile zeroing**：无接触标零，再以已知砝码检查 5 指 gain；带接触启动必须拒绝 calibration；
7. **release 时序**：contact→release→contact，HDF5 raw、sum、contact、timestamp 一致；
8. **pinch**：thumb 对四指逐一 pinch，监测峰值电流、触觉、最小距离和 joint bounds；先降低速度/torque，确认零距离 Stage 2 是否需要目标距离或 tactile gate；
9. **多次 episode/reset**：TAG 与 DexPilot 首帧 command、optimizer latency 和 warm-start 都应稳定，无 joint-order 跳变；
10. **数据验收**：quality tool 明确报告 tactile fresh/calibrated、send/read health、retarget input validity。

## 10. 最终判断

当前 XHand 主链已经具备较完整的双层限位、worker 隔离、状态发布、错误重试和 TAG 优化框架，但仍存在四个会破坏 fail-closed 语义的 P1 缺口。最关键的设计原则应调整为：

- “SDK read 成功”与“SDK send 成功”是两个独立健康域；
- “反馈新鲜”不等于“关节必须运动”；
- “landmarks 是有限数”不等于“landmarks 表示一只有效的手”；
- sparse tactile 必须携带 freshness/release 语义；
- 任何 native SDK 资源一旦 open，所有退出路径都必须经过同一个幂等清理事务。

在上述 P1 修复完成、离线回归通过之前，不建议把系统状态从“实验室受控遥操作”提升为“可无人看护的数据采集”。本报告没有修改运行源码、公共 API、shared dtype、HDF5 schema 或配置。
