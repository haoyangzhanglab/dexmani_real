# Arm / Hand 控制进程化实施计划 (v2)

日期: 2026-07-19（v2：四镜头反思修订）
状态: 设计定稿（未动代码）；灵巧手在修，手侧按最坏假设设计、维修后独立验收
范围: `ArmInnerLoop`（XArm7, Mode 6）与 `XHand` 从主进程线程/直连改造为独立 `mp.Process`，经 SHM 通信。

### v1 → v2 变更
- **F1（D1 翻转）**: hand E2/E3 clip/EMA 状态机由"子进程+轮询写回"改为**主进程持有**；子进程退化为无状态执行器+关节限位兜底+echo 校验（§3.5）。
- **F2**: façade 增加 seqlock 复检 + 关节范围合理性检查 + last-good 缓存，堵 `SharedMemoryRingBuffer` 撕裂读漏洞（§4.7）。
- **F3**: 新增录制"内环实发流" `/action_arm_joint_sent`（schema v9，§4.9），replay 一致性量化（§8 F3a/F3b）。
- **U1**: 内环 50Hz 从可选 P4 升级为 **P1 验收项**（25/50Hz 同批 A/B）。
- **M1**: 新增 §10 策略部署就绪（环即动作 API、chunk 节拍分发、SPMC 只读、validate 咽喉不变式、延迟预算）。
- 手侧分期：P2 对 stub 开发，硬件验收延后为独立批次（§8 G）；P3 拆 P3a/P3b；R1 按最坏假设设计（§5.2）。

---

## 1. 目标与非目标

### 目标
1. **GIL 解耦**：内环时序不再受主进程计算（`solve_teleop_ik` ~27ms、手部 NLP retarget、HDF5 写）影响；验收内环恢复 50Hz（当前 25Hz 降频即 GIL 妥协，见 `inner_loop.py:73`）。
2. **故障隔离**：SDK 原生崩溃（xArm C++ bindings / xhand_controller / pinocchio）不再拖垮主进程与录制线程。
3. **架构对称**：与 `CameraProcess` / `VRReceiverProcess` 的「独立进程 + SHM 环形缓冲」范式统一。
4. **录制-执行一致性**：新增内环实发流，使 replay 一致性可量化、为策略学习提供"实际驱动硬件的动作"。
5. **策略部署就绪**：SHM 环成为 teleop / replay / policy 共用的唯一动作 API（接口本次落地，Policy 进程本身为后续工作）。
6. **零行为回归**：所有已验收的安全语义逐条保留（ramp 重武装、C22/C24 自愈、手部钳位语义、"所发即所录"）。

### 非目标
- 不改控制律（Mode 6 直发、E2/E3 算法本身、validate 七级门逻辑）。
- 不实现 Policy 进程（只落地其 SHM 接口与消费 stub）。
- 不改仿真入口（`examples/sim/`）。

### 前提假设
- A1: 外环维持 16Hz；内环以 50Hz 为设计目标验收，跟踪误差回退则回落 25Hz。
- A2: 沿用 `fork` 启动；所有硬件 SDK 在子进程 `run()` 内惰性导入（`inner_loop.py:249` 既有模式）。
- A3: 双通路仅过渡：arm 验收后删 arm 双通路（P3a），手硬件验收后删手双通路（P3b）。
- A4: XHand 位置伺服（mode 3, kp=80）断连后固件自持末位——设计按此假设，维修后实测确认（R1→验证项）。

---

## 2. 现状契约清点（必须保留的接口与语义）

### 2.1 ArmInnerLoop 触点

| 调用方 | 接口 | 语义要点 |
|---|---|---|
| `TeleopController.__init__` | `ArmInnerLoop(cfg, sync).start()` | sync 为 `SharedSyncPrimitives`（**已是 `mp.Event`**，跨进程免改） |
| `_tick` / `_record_held_tick` | `get_state() → (qpos[7], error_state, target_ts)` | qpos 为 copy；error_state 锁存 |
| `_tick` / `_record_held_tick` | `get_dynamics() → (qvel, tau, temps)` | **NaN 直至首次有效回读**；供扭矩/温度门 + 录制 |
| `_tick` / PAUSED / VR stale | `set_target(cmd \| None)` | `None` = hold 哨兵；0.2s 无目标自动 hold |
| `_print_status` | `tracking_error` property | 被动监测，只读 |
| `_do_home` / `_escalate_to_emergency` | `set_target(None)` + `stop()` | home 前停环避免**双连接冲突**（进程化后消失） |
| `_ensure_inner_running` | `is_alive` / `wait_ready(10s)` / 重建重启 | B 键早按保护：未 ready 拒绝进 TELEOP |
| `_build_record_config` / `_ensure_inner_running` | `getattr(_arm_inner, "_cfg")` | 读 `max_joint_delta` 等 → 门面暴露 `.config` |
| `RobotInterface.return_to_home` | **经 XArm7 第二条连接**：`get_state` / `send_action`(Mode 1) / `reset`(Mode 0 wait=True) / `is_error` / `last_error_message` | 当前必须先停内环 |
| `RobotInterface.get_state(arm_qpos=None)` | `arm.get_state()` 兜底 | 无内环路径（test 入口） |
| `validate_action` | `arm.is_error()` / `is_connected()` / `qpos_min_soft` / `qpos_max_soft` | 前两者动态，后两者静态配置 |
| 入口脚本 | `vr_teleop_arm_only*.py`(×3)、`keyboard_teleop_real.py`、`replay_traj.py` | 各自直驱 ArmInnerLoop + 重启 helper |

### 2.2 XHand 触点

| 调用方 | 接口 | 语义要点 |
|---|---|---|
| `RobotInterface.send_action` | `hand.send_action(qpos_cmd) → bool` + `hand.last_qpos_cmd` | 限位→E3 delta→E2 EMA **在 send_action 内部**；v2 将该状态机移至主进程门面（F1） |
| `RobotInterface.get_state` | `hand.get_state()` → qpos[12] / tactile_force_sum(5,3) / tactile_force(5,120,3) | `force_update=True` 每帧硬件读 |
| `validate_action` | `hand.connected_flag` / `hand.error_state` / `hand.config.qpos_min/max` | 连接手才 gate；退化放行 |
| `RobotInterface` | `reset()` / `stop()` / `clear_error()` / `_sync_hand_collision_model()` | stop = 扭矩清零（松手）——仅主进程显式急停调用 |
| `_build_record_config` | `hand_cfg.max_delta_rad/mode/ema_alpha` | 静态配置 |
| 入口脚本 | `test_quest_hand_teleop.py`（`send_trajectory`） | 轨迹宏命令 |

### 2.3 关键隐含语义（回归雷区）
- **所发即所录**：`/action_hand_joint` 记录 clip/EMA 后实发值；hold 基线跟随实发值（controller.py:391-401）。v2 由主进程持有钳位状态机后**零竞态**满足（§3.5）。
- **ramp 重武装**：任意 hold 后 ramp 归零软启动；速度下限 `1.25×qvel_inf` 防 C24（[[c24-ramp-reset-midmotion]]）。
- **可恢复错误**：C22/C24 → 清锁存 + `_recover_mode`（3 次重试）+ hold，**不**置 error_state。
- **dynamics 新鲜度**：扭矩/温度门在 NaN/陈旧上静默降级——façade 新鲜度闸必须堵此洞（§4.7，[[l515-midrun-stream-stall]] 教训）。
- **录制≠执行缺口**：`/action_arm_joint` 是外层命令，物理执行含内环 delta 钳位 + 固件规划饱和——v2 以实发流补齐（§4.9）。

---

## 3. 目标架构

```
┌─────────────────── 主进程 (16Hz, 指挥者) ───────────────────┐
│ 动作生产（三源互斥）:                                        │
│   VR teleop 管线 | replay 流 | 策略 chunk（未来, §10）       │
│ 主进程职责: 节拍器 + 手部 E2/E3 钳位状态机(F1) +             │
│   validate_action 七级门(所有动作源咽喉) + 录制              │
│ RobotInterface（纯门面，不持硬件连接）                        │
│   .arm → ArmSHMFaçade     .hand → HandSHMFaçade              │
└──┬──────────┬──────────────┬───────────┬────────────────────┘
   ▼target    ▲state         ▼cmd        ▲state(+echo 校验)
┌──────────────────────┐   ┌──────────────────────────┐
│ ArmControlProcess    │   │ HandControlProcess       │
│ 25-50Hz, 唯一        │   │ 30Hz, 唯一 xhand 连接    │
│ XArmAPI 连接         │   │ 无状态执行: 关节限位兜底  │
│ 环内核原样搬入       │   │ + 发硬件 + 状态/触觉发布  │
│ + RPC 宏命令执行器   │   │ + echo(seq,实发值)       │
│ + last_sent 发布     │   │ + RPC 宏命令(含轨迹插值)  │
└──────────────────────┘   └──────────────────────────┘
   ▲ RPC (arm_cmd/result)     ▲ RPC (hand_cmd_macro/result)
   ▲ policy_chunk 环（未来策略进程写入，主进程消费, §10）
```

**原则**：
1. 环内核代码一行不动（`_run`/`_send_target`/`_recover_mode`/`_hold_position`/`_monitor` 整体搬入子进程）——[[c24-ramp-reset-midmotion]]、[[mode6-tracking-error-root-cause]] 修复原样保留。
2. 主进程是所有动作的**唯一生产者与安全咽喉**；子进程是执行器。
3. 平滑职责在固件（Mode 6 在线规划）——主进程不做客户端插值（对 ManiUniCon 的结构性简化，§10）。

---

## 4. SHM 接口定义

复用 `shm/ring_buffer.py::SharedMemoryRingBuffer`（单生产者/单消费者写、FILO、slot 带 monotonic `timestamp_ns`+`sequence`、`frame_age_ns()` 现成；**读取侧允许多消费者 latest 读，见 D9**）。新布局放 `shm/robot_layouts.py`。

### 4.1 arm_state（子→主/策略只读，maxlen=3）

```python
ARM_STATE_DTYPE = np.dtype([
    ("qpos", "<f8", (7,)),  ("qvel", "<f8", (7,)),  ("tau", "<f8", (7,)),
    ("temps", "<f8", (7,)),                     # qvel/tau/temps NaN 直至首次回读
    ("error_state", "u1"), ("connected", "u1"), ("mode", "i4"),
    ("tracking_err", "<f8"),
    ("last_sent", "<f8", (7,)),                 # 内环实发（delta 钳位后）→ §4.9 录制流
    ("ramp_step", "i4"),
])
```
写入时机：每个内环 tick 读完 `get_joint_states` 后。

### 4.2 arm_target（主→子，maxlen=2）

```python
ARM_TARGET_DTYPE = np.dtype([
    ("target", "<f8", (7,)),
    ("is_hold", "u1"),          # 1 = hold 哨兵（set_target(None)）
    ("producer_id", "u4"),      # 1=teleop 2=replay 3=policy；不匹配拒收告警（D9）
])
```
`is_hold=1` 或 `frame_age_ns > target_timeout_s(0.2s)` → hold + ramp 重武装；slot monotonic ts 替代现 `perf_counter` 超时判据。

### 4.3 arm_cmd / arm_cmd_result（RPC，各 maxlen=2）

```python
ARM_CMD_DTYPE = np.dtype([
    ("cmd", "u4"),                  # 1=EXEC_WAYPOINTS 2=RESET_BLOCKING 3=CLEAR_ERROR
                                    # 4=EMERGENCY_STOP 5=REINIT_MODE6
    ("n_waypoints", "u4"), ("waypoints", "<f8", (2048, 7)),  # 114KB/slot；稠密 home 路径典型 <360 点
    ("dt", "<f8"), ("target", "<f8", (7,)), ("speed", "<f8"), ("acc", "<f8"),
])
ARM_CMD_RESULT_DTYPE = np.dtype([
    ("cmd_seq", "u8"), ("ok", "u1"), ("arm_err", "i4"), ("sdk_ret", "i4"),
    ("final_qpos", "<f8", (7,)),
])
```
- 宏命令期间暂停遥测分发，state 环持续更新（主进程监测收敛）。
- `EXEC_WAYPOINTS` 子进程内 Mode 1 `set_servo_angle_j` 逐点（同现 `XArm7.send_action`），结束自动重建 Mode 6；`RESET_BLOCKING` 走 Mode 0 `set_servo_angle(wait=True)`（同现 `XArm7.reset`）——home 语义零变更。超 2048 点分段下发。
- 已否决备选：主进程以 target 环流式喂 waypoint + 轮询收敛——改变末段收敛语义，验收风险高。
- replay 起点对齐亦走 `RESET_BLOCKING`（首帧 qpos 亚度收敛，优于现状）。

### 4.4 hand_state（子→主/策略只读，maxlen=3）

```python
HAND_STATE_DTYPE = np.dtype([
    ("qpos", "<f8", (12,)),
    ("last_qpos_cmd", "<f8", (12,)),    # 子进程实际发往硬件的值（关节限位兜底后）
    ("last_cmd_seq", "u8"),             # 回显已处理 hand_cmd 的 sequence
    ("tactile_sum", "<f8", (5, 3)),
    ("tactile_force", "<f8", (5, 120, 3)),   # 14.4KB/帧，录制全带宽
    ("connected", "u1"), ("error_state", "u1"), ("consecutive_errs", "u4"),
    ("last_error_code", "i8"), ("limit_clipped", "u1"),
])
```

### 4.5 hand_cmd（主→子，maxlen=2）— F1 翻转后的语义

```python
HAND_CMD_DTYPE = np.dtype([
    ("qpos_cmd", "<f8", (12,)),     # 已经主进程 E3 delta + E2 EMA 处理
    ("producer_id", "u4"),
])
```

**职责划分（F1）**：
- **主进程（HandSHMFaçade.send_action）**：持有 `last_qpos_cmd` / `_ema_qpos` 状态机，执行限位→E3 delta→E2 EMA（与现 `XHand.send_action` 代码逐行同源搬迁），产出 `expected_cmd`：写环、返回给调用方录制（所发即所录，零等待零竞态）、更新 hold 基线。
- **子进程**：无状态——关节限位 `np.clip`（与 validate 第 8 级重复，纯兜底）→ 发硬件 → 回显 `(last_cmd_seq, 实发值)`。
- **echo 校验**：主进程每 tick 非阻塞读 echo；`seq` 出现缺口（FILO 丢包）或值不符 → 告警 + 以 echo 值重同步基线。异常收敛为单一可检测事件。
- 子进程 30Hz 环仅在出现新 seq 时发送（位置伺服无需刷新保位）；预留调谐旋钮：命令环 60Hz 轮询 + 状态 30Hz 发布的双速率环（传输抖动从 ≤33ms 降至 ≤16ms）。
- **宏命令**（`SEND_TRAJECTORY`/`RESET`）在子进程内执行，宏期间内部自建临时钳位状态机（与主进程流互斥，状态机交接以 macro 开始/结束为界）。

### 4.6 hand_cmd_macro / result（RPC）
`RESET(qpos[12])` / `STOP` / `CLEAR_ERROR` / `SEND_TRAJECTORY(waypoints[256,12], duration_s, max_speed)`；`MotorTrajectoryInterpolator` 随宏命令搬入子进程。result 回 `{cmd_seq, ok, err_code}`。

### 4.7 新鲜度闸 + 撕裂读防护（façade 层，F2）

**撕裂读防护**（`SharedMemoryRingBuffer` 无 seqlock，`CameraRingBuffer:461-467` 有——照抄后者）：
1. 读 `write_idx` 指向 slot 的 `seq1` → copy data → 复检同 slot `seq2`；
2. `seq1 != seq2`（写者 wrap 覆盖中）→ 重试一次；
3. 再败 → 返回 **last-good 缓存** + 节流告警（不向外传播半写数据）。

> 实现注记（2026-07-20 落地）：`SeqlockRingBuffer`（`shm/robot_ring.py`，子类化 `SharedMemoryRingBuffer`）采用**真 odd/even 协议**——写前存 `2·seq−1`（奇）、写后存 `2·seq`（偶），读侧要求 `seq1==seq2` 且为偶才接受。审查发现"单次 sequence 存储 + 前后复检"的相机模式（`CameraRingBuffer:461-467`）存在 sequence 先于 data 落盘的中间态漏洞，odd/even 严格强于上述规格；对外 API 不变（逻辑 seq = 标记/2，RPC cmd_seq 关联与手部 echo 语义不受影响）。
叠加**范围合理性**：arm qpos 有限且落在硬限位 ±0.05rad 内，否则视同撕裂。
（maxlen=3 @25Hz 覆盖窗口 120ms、@50Hz 60ms——主循环一次 IK 尖峰即可触发，50Hz 下为必需而非保险。）

**新鲜度闸**：
- `ArmSHMFaçade.get_state()`：`frame_age_ns > 3 × loop_period`（50Hz→60ms）→ 返回 error 语义触发急停；杜绝陈旧 tau/temps 让扭矩/温度门静默失效（§2.3，[[l515-midrun-stream-stall]]）。
- `HandSHMFaçade` 阈值 3 × hand_dt = 100ms；**手陈旧不升级急停**（与现状降级语义一致）。
- 所有 façade 读返回 `(data, age_ns)`——策略延迟补偿的现成输入（§10）。

### 4.8 急停与同步
- `estop_event = mp.Event()`：ESC/VR 超时 → 主进程 set；两子进程每 tick **最先**检查（arm→`set_state(4)`、hand→`stop()`），≤1 tick 延迟。
- `SharedSyncPrimitives`（robot_ready/policy_ready）已是 `mp.Event`，零改动；其设计意图即策略分块推理握手（§10）。

### 4.9 录制：内环实发流（schema v9，F3）

- 新增 `/action_arm_joint_sent(T,7)`：每 tick 取 arm_state 环的 `last_sent`（内环 delta 钳位后实发值），与 `/arm_qpos` 同属状态网格——时间对齐反而优于外层命令流（外层命令 t 时刻算、稍后执行）。
- `/action_hand_joint` 语义不变：F1 后主进程录制的就是钳位后实发值。
- meta 增 `arm_sent_stream: True` + `schema_version: 9`；`visualize_episode.py` / `check_episode_health.py` 增量兼容（老 episode 无此数据集 → 工具 fallback）。
- hold 期间 `last_sent` = hold 位置，与 measured qpos 趋势一致——sent 流自带"执行意图连续性"。

### 4.10 策略 chunk 环（接口本次落地，消费者 stub）

```python
POLICY_CHUNK_DTYPE = np.dtype([
    ("n_steps", "u4"),
    ("arm_qpos",  "<f8", (16, 7)),      # K ≤ 16 步
    ("hand_qpos", "<f8", (16, 12)),
    ("target_dt", "<f8"),               # 步长（默认 1/16Hz 网格）
])
```
未来 Policy 进程写入（producer_id=3），主进程消费后逐步入 validate → 节拍分发到 arm_target/hand_cmd（§10）。

---

## 5. 进程生命周期

### 5.1 启动顺序
1. 主进程创建全部 SHM 环（`create=True`）；`FileExistsError` → 清陈旧重建（`camera_process.py:204` 模式，**必须**：残留 arm_target 会追旧目标，比相机陈旧帧危险得多）。
2. ArmControlProcess → `ready_event.wait(30s)`（XArmAPI init、固件版本检查、Mode 6 验证、初始 qpos 回读；失败 → error_state，拒绝进 TELEOP，等价 `_ensure_inner_running` 保护）。
3. HandControlProcess → `ready_event.wait(15s)`；失败 → 退化模式（connected=False，臂-only 继续）。
4. 主循环进入。

### 5.2 信号、崩溃与手侧最坏假设设计（R1→设计定案）
- **arm 子进程 daemon=True**（同 CameraProcess）：主进程被 SIGKILL 时随之死；arm 断连后 Mode 6 固件自持末位（inner_loop.py:445 已确认）。
- **hand 子进程非 daemon + watchdog 自重启**（A4 最坏假设）：
  - 结构依据：mode 3 + kp=80 位置伺服，固件持位不依赖指令流刷新；子进程死/串口断 → 固件自持；
  - 指令环陈旧 >0.5s（主进程死）→ 保持末位、停发新指令；
  - 串口错误累积 → watchdog 触发 `_retry_open_device` 三段重连；
  - A4 维修后实测确认（清单 G1）。
- **Ctrl-C**（SIGINT 达整个进程组）：子进程 handler → arm 发 hold 后退出、hand 保持末位（绝不 tor_max=0 松手）→ 主进程 `join(3s)` 兜底 terminate。
- **崩溃检测**：主进程每 tick 轮询 `is_alive`（CameraProcess.crashed 模式）：arm 死→急停；hand 死→connected=False 降级。
- **重启**：façade 封装 `_ensure_inner_running`：`is_alive` 假 → 重建环 + 重启 + `wait_ready(10s)`；`_sync.policy_ready.clear()` 清陈旧握手。

### 5.3 关机序列
录制收尾 → arm: hold + stop_event + join(3s)（finally 中 disconnect，固件保持）→ hand: hold + stop_event + join → unlink 全部 SHM。

### 5.4 加固（可选）
子进程 `os.nice(-10)`（无需特权）降低被相机进程抢占概率；SCHED_FIFO 需权限，列为备选项。子进程内删除 `time.sleep(0)` GIL yield（单线程无意义）。

---

## 6. 分阶段实施

| 阶段 | 内容 | 工期 | 验收 |
|---|---|---|---|
| **P0 地基** | `shm/robot_layouts.py`（§4 全部 dtype，含 producer_id / policy_chunk）+ `shm/robot_rpc.py` 薄 RPC + 单测（layout 往返、超时→hold 假钟、seqlock 撕裂注入、陈旧清理、RPC 超时） | 1d | pytest 全绿，零行为变更 |
| **P1 Arm 进程化** | `ArmControlProcess`（环内核原样搬入 + RPC 执行器 + last_sent 发布）、`ArmSHMFaçade`（含 §4.7 全部防护）、RobotInterface 换 façade（过渡开关）、return_to_home 走 RPC、录制 sent 流（schema v9）、`replay_traj.py --source sent\|cmd`、4 类入口适配、policy_chunk 消费 stub | 3d + 占机 | P0 测试 + §8 A/B/C/E + **25/50Hz A/B**（跟踪误差回退则留 25Hz）+ F3a replay 一致性基线 |
| **P2 Hand 进程化（stub 驱动）** | `HandControlProcess`（30Hz 无状态执行 + echo + 触觉 + RPC + watchdog）、`HandSHMFaçade`（E2/E3 状态机 + echo 校验）、validate/碰撞同步改读 façade；对 `_stub_mode` 开发 + 单测（monkeypatch SDK / 注入通信失败），含 D1 逐位一致性测试 | 2d | §8 D（stub 部分）+ 手部 E3 逐位一致（stub 向量比对）；**硬件验收延后为 §8 G 独立批次** |
| **P3a Arm 清理** | 删 arm 线程版与过渡开关；XArm7 阻塞方法标注"仅子进程用"；CLAUDE.md 架构图/anti-pattern 更新（"SDK 连接只在子进程建立"、"validate 是所有动作源咽喉"） | 0.5d | 全量回归 |
| **P3b Hand 清理** | 手硬件验收（§8 G）通过后，删手双通路 | 0.5d | §8 G 全绿 |

合计约 **7-9 人日** + 手维修后验收批次；关键路径 P1 占机。

---

## 7. 行为决策点（v2 定案）

| # | 决策 | 结论 | 理由 |
|---|---|---|---|
| D1 | hand clip/EMA 位置 | **主进程**（F1 翻转） | 轮询写回破坏 16Hz 节律且引入跨 tick 竞态；生产者持有钳位状态机 → 所发即所录零等待；子进程无状态+关节限位兜底+echo 校验；异常收敛为单一可检测事件 |
| D2 | return_to_home 执行 | RPC 宏命令（§4.3） | 保 Mode 1/0 现有 home 语义 |
| D3 | 触觉全带宽过 SHM | 是（14.4KB/帧） | schema v8 录制流，memcpy 成本可忽略 |
| D4 | hand 生命周期 | **非 daemon + watchdog + 持位语义** | A4 最坏假设设计，维修后实测确认 |
| D5 | 陈旧 SHM | 启动即 unlink 重建 | 残留目标 = 立即运动风险 |
| D6 | 双通路 | P3a/P3b 分批硬删除 | arm 不被手维修进度拖住 |
| D7 | 录制 sent 流 | **schema v9 增量**（§4.9） | replay 一致性与策略学习的价值核心；不可逆性由 schema 版本号兜底 |
| D8 | 策略 chunk 平滑 | **主进程节拍分发，不引入插值器** | Mode 6 固件即在线规划器——对 ManiUniCon 的结构性简化 |
| D9 | 环消费者模型 | 写 SPSC 互斥（状态机 + producer_id），读 SPMC latest 安全 | 策略进程只读附着 state/camera 环，零额外管道 |

---

## 8. 边界重测清单

### A. 时序/安全（最高优先）
- [ ] A1 VR 拔线：0.5s 内 arm 静止；恢复后 ramp 从 0.2 rad/s 重武装（log 速度曲线）
- [ ] A2 注入 C24/C22：自愈 ≤3 次重建 Mode 6、不升级急停、delta 钳位基线不污染（[[c24-ramp-reset-midmotion]] 回归）
- [ ] A3 主进程注入 200ms 停顿：arm 走 timeout hold 而非追陈旧目标；恢复无速度尖峰（C24 原根因场景）
- [ ] A4 `kill -STOP` arm 子进程：新鲜度闸 60ms(50Hz)/120ms(25Hz) 内升级急停；扭矩门不在陈旧 tau 上静默放行
- [ ] A5 ESC → `set_state(4)` 实测 <60ms @50Hz
- [ ] A6 tracking_error 与进程化前同量级（p95 20.2° 基线，[[arm-only-record-session-2026-07-18]]）
- [ ] A7 **撕裂读注入**（P0 单测 + 真机 IK 尖峰场景）：seqlock 复检命中率、last-good 回退路径、无垃圾 qpos 进 IK seed

### B. 录制对齐
- [ ] B1 网格填充 ≤0.2% unfilled（状态源换 SHM 不得回退）
- [ ] B2 所发即所录：`/action_hand_joint` == 主进程钳位后值 == echo 实发值（echo 缺口注入测试：告警+基线重同步路径）
- [ ] B3 held 帧不被成功帧回填
- [ ] B4 `/arm_tau`/`/arm_qvel` 无全 NaN 段；tau 全 NaN 告警 5s 节流
- [ ] B5 `/hand_tactile_force` 帧数==T，抓放力曲线可比（手硬件验收批次）
- [ ] B6 ENOSPC 假报回归（PR5，三入口一致）
- [ ] B7 **`/action_arm_joint_sent` 正确性**：人工注入触发内环钳位（大跳变目标）→ sent 流显示钳位后值、cmd 流显示原值；hold 期间 sent == hold 位置

### C. 生命周期/故障注入
- [ ] C1 Ctrl-C 于 TELEOP/录制中/return_to_home 中（三入口）：arm hold、hand 持位、episode 正确处置
- [ ] C2 `kill -9` 主进程：arm 固件保持；hand 持位（A4 确认前按设计预期记录行为）
- [ ] C3 子进程崩溃注入：检测 + 急停/降级，无挂起
- [ ] C4 陈旧 SHM 残留启动：清理后不追旧目标
- [ ] C5 B 键早按（未 ready）：拒绝进 TELEOP
- [ ] C6 return_to_home 中 planner 异常：RPC 超时 → hold + 报错，arm 不悬停于半空宏命令
- [ ] C7 producer_id 不匹配注入（伪造 SHM 写入）：拒收 + 告警

### D. 手专属（stub 可测部分）
- [ ] D1 **E3 逐位一致**（F1 核心验收）：同命令序列经"原 XHand.send_action"与"façade+子进程"两条路径，输出差异 <1e-9（stub 向量比对）
- [ ] D2 退化模式：手未连接 → validate 放行、hand_connected=False、臂遥操正常
- [ ] D3 CRC/BOOT_CMD 注入：50ms/500ms 恢复延迟保留，连接态 error 门生效
- [ ] D4 send_trajectory 经 RPC：末位误差同基线

### E. 入口/工具
- [ ] E1 `replay_traj.py`：起点对齐（examples 审查 critical）经 RESET_BLOCKING 不回归；`--source sent` 路径可用
- [ ] E2 keyboard_teleop + preflight 全流程
- [ ] E3 arm_only ×3 真机冒烟
- [ ] E4 check_episode_health 对 v9 episode 全绿（含老 v8 episode 兼容）
- [ ] E5 **策略接口冒烟**：脚本向 policy_chunk 环注入一段录制动作 → 主进程消费 → 硬件执行（producer_id=3 路径端到端）

### F. 量化验收（真机对照）
- [ ] F1 **25/50Hz A/B**：各 5 条同任务 episode，tracking error 分布（p95 基线 20.2°）、填充率、IK 成功率统计对比；回退则留 25Hz
- [ ] F2 home 末位误差对比（现 log "final error: X.XX deg"）
- [ ] F3a **replay 一致性**：`replay(--source sent)` vs 原 episode `/arm_qpos`，逐关节 RMSE/L∞ 分布（固件规划残差基线）
- [ ] F3b **replay→record 往返**：重放 X 录成 Y，Y.measured vs X.measured 对比（数据飞轮闭环）
- [ ] F4 独立核查确认无回退（参考 20-agent 核查先例）

### G. 手硬件验收批次（维修后，P3b 前置）
- [ ] G1 A4 确认：手持物拔 USB / kill -9 hand 子进程 → 固件持位不松劲（若松劲 → watchdog 参数收紧 + 复盘 D4）
- [ ] G2 D1-D4 真机复测（stub 已覆盖逻辑，此处验硬件闭环）
- [ ] G3 抓放 episode：触觉流 + 夹持力曲线 + 手部传输抖动统计（≤33ms 设计值；超标则启用 60Hz 命令环旋钮）
- [ ] G4 手部 delta 钳位饱和场景（快速张合）：E3 行为与进程化前录像逐帧对比

---

## 9. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| R1 XHand 断连行为（A4） | 未知 | 高 | 最坏假设设计（非 daemon+watchdog+持位），G1 实测收口 |
| R2 fork + SDK 线程继承 | 低 | 中 | SDK 全部 run() 内惰性导入（既有模式）；子进程早于主进程重线程启动 |
| R3 撕裂读（F2） | 中 | 高 | seqlock 复检 + 范围检查 + last-good（P0 单测注入验证） |
| R4 return_to_home RPC 分段边界 | 中 | 中 | 宏命令执行器完整复用 XArm7 现有路径；C6/E1 专项 |
| R5 手部传输抖动（≤33ms） | 低 | 低 | G3 统计验收；60Hz 命令环旋钮预留 |
| R6 录制隐性回归 | 中 | 高 | F2-F4 量化 A/B + health 门槛 + B7 sent 流专项 |
| R7 schema v9 工具链同步 | 低 | 低 | 纯增量数据集；visualize/health fallback 老格式（E4） |

---

## 10. 策略部署就绪

本次只落地接口与不变式，Policy 进程为后续工作——但接口现在定对，未来零返工。

### 10.1 SHM 环即唯一动作 API
teleop 管线 / replay 工具 / 未来策略进程，全部经 `arm_target`/`hand_cmd`（策略经 `policy_chunk` → 主进程分发）驱动硬件。三入口一路径，安全语义统一。写入互斥由状态机 + `producer_id` 保证（D9）。

### 10.2 chunk 分发 = 节拍器，无插值器
VLA/扩散策略一次推理产出 K≤16 步 chunk（带 target_dt）。主进程 chunk 缓冲 + 16Hz 网格节拍逐步入 validate → target 环。**不引入 ManiUniCon 式客户端插值器**——Mode 6 固件本身是在线轨迹规划器（D8）。这是本架构相对 ManiUniCon（200Hz 循环 + PoseTrajectoryInterpolator + 每 tick pink IK）的结构性简化。

### 10.3 synchronized = 分块推理模式
`robot_ready`（chunk 耗尽）/ `policy_ready`（新 chunk 就绪）——`mp.Event` 跨进程现成，语义与 VLA 分块推理逐条对应。

### 10.4 观测：SPMC 只读附着
策略进程直接附着 arm_state / hand_state / CameraRingBuffer 读 latest（读路径不修改共享字，多读者安全；各读者独立 seqlock 复检）。观测与主循环同源同时基。

### 10.5 安全咽喉不变式
**所有动作源在进入 target 环前必须经主进程 validate_action 七级门**——进程化后安全模型的锚点，入 CLAUDE.md anti-pattern。

### 10.6 延迟预算（设计值）

| 环节 | 25Hz 内环 | 50Hz 内环 |
|---|---|---|
| 状态陈旧度（arm_state 环） | ≤40ms | ≤20ms |
| 相机陈旧度（30Hz） | ≤33ms | ≤33ms |
| 网格对齐（16Hz 外环） | ≤62.5ms | ≤62.5ms |
| 内环转发 | ≤40ms | ≤20ms |
| IPC + 固件重规划 | ~10-30ms | ~10-30ms |
| **观测→动作地板（不含推理）** | **~140-200ms** | **~120-160ms** |

façade 读返回 `(data, age_ns)`——策略做观测时间戳对齐/延迟补偿的现成输入。
