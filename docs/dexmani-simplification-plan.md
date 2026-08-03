# DexMani 架构简化方案

**目标：向 ManiUniCon 设计哲学对齐，以简洁高效为原则，大幅削减工程复杂度。**

> 复核：2026-08-02，通过 Accuracy / Completeness / Consistency 三维审查，已修正所有 HIGH 级别问题。

---

## 1. 当前问题诊断

### 1.1 臃肿量化

| 文件 | 当前行数 | 问题 |
|------|---------|------|
| `examples/real/vr_teleop_arm_only_record_plus.py` | 1228 | Main 进程包含 VR→IK→validate→send→record 全部逻辑 |
| `robot/inner_loop.py` | 1043 | Macro RPC、进程内重连看门狗、三级 tracking error、双路径急停 |
| `robot/hand_process.py` | 1252 | Macro RPC 轨迹子系统（无调用者）、分级 watchdog、orphan exit |
| `robot/arm_process.py` | 1006 | 与 interface.py 重复的归位逻辑、producer ID 门控、adapter 层 |
| `robot/interface.py` | 633 | 与 arm_process.py 重复的归位实现、重复的 `return_to_home()`（无外部调用者） |
| `shm/robot_ring.py` | 282 | SeqlockRingBuffer 独立子类 |
| `shm/ring_buffer.py` | 586 | 包含 CameraRingBuffer、基类无 seqlock |
| `shm/robot_rpc.py` | 151 | RPC 层仅 hand 使用，且轨迹 RPC 无调用者 |
| `shm/robot_layouts.py` | 102 | 5 个 dtype，tactile_force 占 95.5% 体积 |
| `shm/frame_manager.py` | 129 | VR ring 管理，未与 SharedStorage 统一 |
| `shm/layouts.py` | 183 | VR/Camera 帧 dtype，与 robot_layouts.py 分立 |
| `recording/episode_recorder.py` | 884 | 集中式 accumulate-then-dump，与多进程数据产生模式不匹配 |
| `robot/validate.py` | 158 | Protocol 类、单次用 helper、lambda chain |
| **合计** | **~7,636** | |

### 1.2 根因

DexMani 的架构不是设计出来的，是从"hacked-together teleop script"逐步硬化的。每个安全机制对应一个踩过的坑，但层层叠加而不清理旧方案：

- **Main 是全知全能的上帝进程**——和 ManiUniCon 的薄 Main 完全相反
- **SHM 创建分散在 6+ 个文件**——要理解数据平面必须追踪 `facade.start()` 的调用链
- **过度防御**——进程内重连、双路径急停、三级 tracking error、分级 watchdog——在"操作者拿着急停按钮坐在旁边"的实验室场景下多余
- **死代码**——Macro RPC 轨迹子系统（`HAND_MACRO_SEND_TRAJECTORY` 无任何调用者）、Producer ID 门控（`PRODUCER_REPLAY`/`PRODUCER_POLICY` 仅在定义处出现）

---

## 2. 目标架构

### 2.1 核心原则（抄自 ManiUniCon）

1. **Main 不参与数据面**——只创建 SharedStorage、spawn 子进程、持有 `is_running`
2. **SharedStorage 是唯一的数据平面**——一个类拥有所有 ring/queue/event，纯读写无业务逻辑
3. **进程间只传结构化数据**——状态、动作、相机帧，不传 Python 对象或函数引用
4. **各进程独立生命周期**——以自己的频率运行，通过 `is_running` 标志统一退出
5. **操作者在场是最好的安全系统**——软件不需要替代物理急停按钮

### 2.2 进程拓扑（5 个子进程 + Main）

```
┌─ Main Process (~80 行) ──────────────────────────────────────────────────┐
│                                                                           │
│  args = parse_args()                                                      │
│  shared = SharedStorage.create(prefix="dexmani")                          │
│                                                                           │
│  procs = [                                                                │
│      mp.Process(target=camera_loop, args=(shared,), name="cam"),          │
│      mp.Process(target=vr_loop,     args=(shared,), name="vr"),           │
│      mp.Process(target=policy_loop, args=(shared,), name="pol"),          │
│      mp.Process(target=arm_loop,    args=(shared,), name="arm"),          │
│      mp.Process(target=hand_loop,   args=(shared,), name="hand"),         │
│  ]                                                                        │
│  for p in procs: p.start()                                                │
│                                                                           │
│  # 等待各进程就绪（含自检结果）                                              │
│  for name, ev in [("arm", shared.arm_ready),                              │
│                    ("hand", shared.hand_ready),                            │
│                    ("cam", shared.camera_ready)]:                          │
│      if not ev.wait(timeout=15):                                          │
│          logger.error(f"{name} startup timed out: {shared.startup_error}") │
│          shutdown()                                                        │
│                                                                           │
│  try:                                                                     │
│      while all(p.is_alive() for p in procs):                              │
│          if shared.error_state.value:                                     │
│              logger.error("Error state set, shutting down")               │
│              break                                                        │
│          time.sleep(0.5)                                                  │
│  except KeyboardInterrupt:                                                │
│      pass                                                                 │
│  finally:                                                                 │
│      shared.is_running.value = False                                      │
│      for p in procs: p.join(timeout=5)                                    │
│      for p in procs:                                                      │
│          if p.is_alive(): p.terminate()                                   │
│      shared.close()                                                       │
└───────────────────────────────────────────────────────────────────────────┘
        │                │                │                │                │
        ▼                ▼                ▼                ▼                ▼
┌─ CameraProcess ┐ ┌─ VRProcess ───┐ ┌─ PolicyProcess ┐ ┌─ ArmProcess ┐ ┌─ HandProcess ┐
│ (基本不变)      │ │ (独立保留)     │ │ (从 Main 拆出) │ │ (精简版)    │ │ (精简版)     │
│                 │ │               │ │                │ │             │ │              │
│ while running:  │ │ while running:│ │ while running: │ │ while run:  │ │ while run:   │
│  frame=wait()   │ │  tcp.poll()   │ │  arm_state=    │ │  act=q.get()│ │  cmd=ring    │
│  write_cam_ring │ │  parse_hts()  │ │    read_ring() │ │  if HOME:   │ │    .read()    │
│  if record:     │ │  write_vr_ring│ │  vr=read_ring()│ │    homing() │ │  set_target()│
│    passthrough  │ │               │ │  camera=read() │ │  else:      │ │  read_qpos() │
│                 │ │ ~250 行       │ │  ik()→action  │ │  NaN guard  │ │  write_ring()│
│ ~300 行         │ │ (精简)        │ │  write_act_q()│ │  limit clip │ │              │
│                 │ │               │ │  state machine│ │  servo()    │ │ ~250 行      │
│                 │ │               │ │  audio+keybd  │ │  write_ring │ │              │
│                 │ │               │ │  owns record  │ │             │ │              │
│                 │ │               │ │ ~500 行       │ │ ~300 行     │ │              │
└─────────────────┘ └───────────────┘ └────────────────┘ └─────────────┘ └──────────────┘
```

**关键变化：**

- **VRReceiverProcess 保持独立**——它运行 HTS TCP 服务器，有独立的故障模式（TCP 断连、Quest 休眠）。Policy 通过 `vr_ring` 读 VR 帧，不直接持有 TCP 连接。
- **Policy 拥有录制**——这是解决多流时间对齐最简洁的方案（见 ISSUE 4）。Policy 读所有状态 ring，写 action，同时将 `(state, action, camera)` 一起写入 `TimestampAlignedBuffer`。Arm/Hand/Camera 不在自己的循环中录制——只发布状态到 ring。

### 2.3 SharedStorage：数据平面

```python
# shm/shared_storage.py (~120 行)

@dataclass
class SharedStorage:
    """所有进程通过它交换数据。不包含任何业务逻辑。"""

    # ---- Rings: 连续流，只读最新 ----
    camera_ring:      CameraRingBuffer       # CameraProcess  -> PolicyProcess
    vr_ring:          SeqlockRingBuffer      # VRProcess      -> PolicyProcess
    arm_state_ring:   SeqlockRingBuffer      # ArmProcess     -> PolicyProcess
    hand_state_ring:  SeqlockRingBuffer      # HandProcess    -> PolicyProcess
    hand_tactile_ring: SeqlockRingBuffer     # HandProcess    -> PolicyProcess (稀疏写入)
    hand_cmd_ring:    SeqlockRingBuffer      # PolicyProcess  -> HandProcess (latest-wins)

    # ---- Queue: 有序动作（仅 arm，Mode 6 需要时序） ----
    arm_action_q:     mp.Queue               # PolicyProcess  -> ArmProcess

    # ---- Flags ----
    is_running:       mp.Value               # Main -> 所有进程（唯一 writer）
    is_recording:     mp.Value               # PolicyProcess -> Arm/Hand/Camera
    error_state:      mp.Value               # Arm/Hand -> 所有进程（sticky latch，只升不降）
    estop_request:    mp.Value               # PolicyProcess -> Arm/Hand（急停请求）

    # ---- Events ----
    arm_ready:        mp.Event               # ArmProcess     -> Main
    hand_ready:       mp.Event               # HandProcess    -> Main
    camera_ready:     mp.Event               # CameraProcess  -> Main
    vr_ready:         mp.Event               # VRProcess      -> Main

    # ---- 诊断 ----
    startup_error:    mp.Array('c', 256)     # 任一子进程 -> Main（启动失败原因）

    # ---- 录制元数据（Policy 写入） ----
    record_dir:       Optional[str]
    record_dt:        float = 1/16

    @classmethod
    def create(cls, prefix: str) -> "SharedStorage":
        ...

    def close(self):
        ...
```

**关键设计决策：**

| 决策 | 理由 |
|------|------|
| **Arm 动作用 Queue**（`maxsize=2`） | Mode 6 需要有序执行；bounded queue 提供 backpressure。若 Policy 快于 Arm，Policy 阻塞等 Arm 消费——这是正确的行为（VR 帧被跳过，不积累 stale 命令） |
| **Hand 动作用 Ring**（`hand_cmd_ring`） | 手部是直接位置伺服，无固件轨迹规划。最新目标胜于历史目标。若 Policy 快于 Hand，旧目标被覆盖（正确）；若 Policy 慢，Hand 重复读最后目标（hold，正确） |
| **状态全用 Ring** | 只读最新值，自动覆盖历史 |
| **无 RPC 层** | 所有跨进程操作用 flag/ring/queue 完成 |
| **`error_state` 是 sticky latch** | ArmProcess 或 HandProcess 设置后永不自动清除。只有 Main（重启时）清零 |

### 2.4 数据类型：ArmState / HandState 替代单体 RobotState

当前 `RobotState` 是包含 arm + hand + tactile + fingertip 的 22 字段单体结构。在多进程架构中，没有单个进程同时拥有所有字段。**拆分为独立类型：**

```python
# 各进程只写自己拥有的字段

@dataclass
class ArmState:
    """ArmProcess 写入 arm_state_ring"""
    qpos: np.ndarray      # (7,)  joint positions
    qvel: np.ndarray      # (7,)  joint velocities
    tau: np.ndarray       # (7,)  joint torques
    eef_pos: np.ndarray   # (3,)  FK computed by ArmProcess
    eef_rot6d: np.ndarray # (6,)  FK computed by ArmProcess
    error_code: int
    connected: bool
    mode: int
    tracking_err: float
    timestamp: float

@dataclass
class HandState:
    """HandProcess 写入 hand_state_ring"""
    qpos: np.ndarray      # (12,) joint positions
    current: np.ndarray   # (12,) joint currents
    tactile_sum: np.ndarray # (5, 3) 触觉合力（控制用）
    tactile_contact: np.ndarray  # (5,) 每指是否有接触
    error_state: bool
    connected: bool
    timestamp: float

# hand_tactile_ring 单独承载 14,400 字节/帧的 tactile_force
# maxlen=8，只在 tactile_contact[finger] 非零时写入（稀疏）
```

**Policy 进程负责 FK 补充**：Policy 持有 `XArm7Kinematics` 实例，从 `ArmState.qpos` 计算 `eef_rot6d`（IK 需要），以及从 ArmState.eef_pos + HandState.qpos 计算 `fingertip_pos`（录制需要）。录制时组装完整 `RobotState`。

### 2.5 完整数据流

```
VRProcess                      PolicyProcess                     ArmProcess
────────                       ─────────────                     ──────────
while running:                 while running:                    while running:
  tcp.poll()                     arm_state = arm_state_ring       action = arm_action_q
  frame = parse_hts()              .read_latest()                    .get(timeout=0.1)
  vr_ring.write(frame)           vr = vr_ring.read_latest()      if is HOME sentinel:
                                 camera = camera_ring                do_homing_sequence()
                                     .read_latest()               else if action is None:
CameraProcess                                                        continue  # hold
─────────────                    # IK 管道                        # NaN guard
while running:                   target_eef = mapper.map(vr)      target = clip(action.qpos)
  frame = cam.read()             target_eef = ema(target_eef)     arm.set_servo_angle(target)
  camera_ring.write(frame)       target_eef = clamp(target_eef)   qpos,qvel,tau = read()
                                 arm_qpos = ik(target_eef,        error = arm.error_code
                                               arm_state.qpos)
HandProcess                      arm_qpos = delta_clamp(arm_qpos) arm_state_ring.write(
──────────                                                          ArmState(qpos,qvel,tau,
while running:                   arm_action_q.put(                    eef_pos,eef_rot6d,...))
  cmd = hand_cmd_ring              ArmAction(qpos=arm_qpos,
    .read_latest()                   timestamp=now))              if error in (22,24,31):
  if cmd is not None:                                               arm.clean_error()
    hand.set_target(cmd.qpos)     hand_cmd_ring.write(               arm.set_mode(6)
  qpos = hand.get_qpos()            HandCmd(qpos=hand_qpos))      elif error != 0:
  hand_state_ring.write(                                             shared.error_state = True
    HandState(qpos,...))          # 状态机                            break
  if contact:                     if key == 'B':
    hand_tactile_ring.write(        shared.is_recording = True
      tactile_force)                recorder.start_episode()
                                  if key == 'C':
                                    shared.is_recording = False
                                    recorder.stop_episode()
                                  if key == 'H':
                                    arm_action_q.put(HOME_SENTINEL)
                                  if key == 'ESC':
                                    shared.estop_request = True
                                    shared.is_running = False

                                  # 录制（Policy 拥有）
                                  if shared.is_recording:
                                    recorder.add_frame(
                                      state=merge(arm_state, hand_state),
                                      action=action,
                                      camera=camera)
```

### 2.6 急停机制

**策略**：Policy（持有键盘）检测 ESC → 设 `estop_request=True` + `is_running=False`。Arm/Hand 在下一次循环迭代中检测到 `is_running=False`，在退出前调用 SDK 的 stop/set_state(4)。

**不去做**：不需要 Policy 直接发 `set_state(4)`（跨进程 SDK 调用不可能）、不需要双路径 fallback（物理急停按钮在手边）、不需要 `register_arm_servo` 模式（急停路径简化为 flag → loop exit → cleanup）。

### 2.7 归位（Homing）机制

**问题**：XArm7 固件不支持两个 SDK 连接同时存在。不能从 Policy 或 Main 开第二个连接调 `set_servo_angle`。

**方案**：Policy 向 `arm_action_q` 放入特殊的 `HOME_SENTINEL`。ArmProcess 检测到后执行归位序列（碰撞感知的 joint-space path planning → waypoint 执行 → 收敛到 home），完成后放回正常 action。这与当前 `do_return_home` 逻辑相同，但执行位置从 `arm_process.py` 的 facade 层移到 `arm_loop` 内部。

**归位期间的 hand**：ArmProcess 向 `hand_cmd_ring` 写 home 位置。

**取消**：Policy 再次按 H 或 ESC → 向 queue 放入 `HOME_CANCEL` sentinel → ArmProcess 中断归位。

### 2.8 预检查（Preflight）

当前 `preflight_check(robot)` 通过 `RobotInterface` 检查 arm/hand 连接和状态。在新架构中 Main 没有 `RobotInterface`。

**方案**：预检查分散到各进程的启动阶段。每个子进程在 `ready_event.set()` 之前完成自检：
- ArmProcess：`arm.is_connected()` + `arm.error_code == 0` + `qpos` 非 NaN → `arm_ready.set()`
- HandProcess：同上 → `hand_ready.set()`
- 自检失败 → 向 `shared.startup_error` 写入原因 → `ready_event` **不设置** → Main 超时退出并打印错误

---

## 3. 各组件精简方案

### 3.1 ArmInnerLoop: 1043 → ~300 行

**删除：**

| 删除项 | 行数 | 原因 |
|--------|------|------|
| Macro RPC (`exec_macro`, RESET_BLOCKING 等 5 指令) | ~170 | 归位改为 HOME sentinel 走 action queue |
| 进程内重连看门狗 (5 error → disconnect+reconnect ×3) | ~100 | 操作者在场，报错后手动重启 |
| 双路径急停 fallback (线程死后开新连接发 set_state(4)) | ~20 | 急停改为 flag → loop exit → cleanup |
| `tracking_error_adaptive_max_rad` / `tracking_error_anomaly_cap_rad` | 字段 | 保留 `tracking_error_warn_rad`；`assess_trajectory_quality.py` 改为用固定阈值 |
| sent-command stream (`_last_sent_cmd` vs `_last_sent_target` 双轨) | ~30 | 离线可从 recording 重建 |
| 三级 tracking error (warn/adaptive/anomaly) | ~20 | 留一个 warning |

**保留（非协商）：**

- Mode 6 初始化与维护
- NaN guard（发送前检查）
- Joint limit clip（硬件安全）
- Target timeout → hold（防追 stale target）
- 基础 error code 检查 + recoverable error (22/24/31) 自动 clear
- 状态读回 (qpos/qvel/tau)
- FK 计算 (eef_pos/eef_rot6d) —— ArmProcess 有 URDF 模型，FK 成本极低

**保留的工具类：**

- `ThrottledWarner`——实际用于 6 个位置（arm_process、inner_loop、hand_process），是高频控制循环中防止日志刷屏的实用工具。保留，从 validate.py 移到 `shm/` 或 `utils/`。

```python
# arm_loop (~300 行) — 精简后
def arm_loop(shared: SharedStorage, config: ArmConfig):
    arm = XArm7(ip=config.ip)
    arm.set_mode(6)
    arm.set_state(0)
    kin = XArm7Kinematics(urdf_path=config.urdf)

    last_target = None
    shared.arm_ready.set()  # 自检已通过

    while shared.is_running.value:
        # ---- 急停 ----
        if shared.estop_request.value:
            arm.set_state(4)
            break

        # ---- 读动作 ----
        try:
            action = shared.arm_action_q.get(timeout=0.1)
        except Empty:
            action = None

        # ---- HOME sentinel → 归位 ----
        if _is_home_sentinel(action):
            _do_homing_sequence(arm, kin, shared)
            continue

        # ---- 正常伺服 ----
        if action is not None:
            target = action.qpos
            if not np.isfinite(target).all():
                target = last_target
            target = np.clip(target, JOINT_LO, JOINT_HI)
            last_target = target

        if last_target is not None:
            arm.set_servo_angle(angle=last_target, wait=False)

        # ---- 读回状态 + FK ----
        qpos, qvel, tau = arm.get_joint_states(num=3)
        eef_pos, eef_rot6d = kin.compute_eef(qpos)
        error_code = arm.error_code

        # ---- 错误处理 ----
        if error_code in (22, 24, 31):
            arm.clean_error()
            arm.set_mode(6)
        elif error_code != 0:
            shared.error_state.value = True
            break

        # ---- 发布状态 ----
        shared.arm_state_ring.write(ArmState(
            qpos=qpos, qvel=qvel, tau=tau,
            eef_pos=eef_pos, eef_rot6d=eef_rot6d,
            error_code=error_code,
            connected=arm.connected,
            mode=arm.mode,
            timestamp=time.monotonic(),
        ))

    arm.set_state(4)
```

### 3.2 Hand 进程: 1252 → ~250 行

**删除：**

| 删除项 | 行数 | 原因 |
|--------|------|------|
| Macro RPC 轨迹子系统 | ~200 | **无任何调用者** |
| RPC 层整体 (`RpcClient`/`RpcServer`, 两个 macro ring) | ~150 | reset/stop/clear_error 用 Event flag 或在 HandProcess 内部自愈 |
| 分级 send-error watchdog | ~50 | 出错设 error_state，操作者处理 |
| 分级 qpos-stale watchdog | ~30 | 同上 |
| Orphan exit (daemon=False + `os.getppid()`) | ~30 | 统一 daemon=True，Main 死全部死 |
| `inspect.signature` abort_event 检测 | ~10 | 随轨迹子系统删除 |
| `force_update` 参数 | 0 | 文档注明 "Ignored in SHM mode" |

**tactile_force 拆分：**

| Ring | Dtype | 体积 | maxlen | 写入策略 |
|------|-------|------|--------|---------|
| `hand_state_ring` | qpos(12) + current(12) + tactile_sum(5,3) + contact(5) + error/connected | ~300B | 3 | 每 tick |
| `hand_tactile_ring` | tactile_force(5,120,3) | 14,400B | 8 | 仅当 `contact[finger]` 非零 |

hand_state_ring 从 15KB 降到 ~300B。录制时 Policy 从两个 ring 合并完整 HandState。

### 3.3 Policy 进程（新建）: ~500 行

从 Main 拆出的完整遥操作循环。职责清单：

| 职责 | 说明 |
|------|------|
| **VR→IK 管道** | `ArmWristMapper.map()` → EMA → workspace clamp → `solve_teleop_ik()` → delta clamp |
| **手部 retargeting** | `XHandRetargeter.retarget()`（当前 arm-only 未接线，预留） |
| **状态机** | IDLE / TELEOP / PAUSED，键盘+VR 按钮驱动 |
| **录制管理** | `is_recording` flag 的唯一 writer；持有 `EpisodeRecorder`；每 tick 调 `add_frame(state, action, camera)` |
| **急停触发** | ESC → `estop_request=True` + `is_running=False` |
| **归位触发** | H → `arm_action_q.put(HOME_SENTINEL)` |
| **音频反馈** | 状态转换时播放 TTS（`AudioFeedback` 搬入 Policy） |
| **键盘监听** | `KeyboardHandler`（pynput hook）搬入 Policy |

**初始化**：Policy 进程构造 `ArmWristMapper`、`XArm7MotionPlanner`（MPlib/collision）、`XArm7Kinematics`（FK 用于 recording）、`XHandRetargeter`（可选）。这些对象重（~2 秒构造），在 Policy 启动时做，不阻塞 Main。

### 3.4 VRReceiverProcess: 380 → ~250 行

VR 进程保持独立——运行 HTS TCP 服务器，接收 Quest 手部跟踪帧，写入 `shared.vr_ring`。精简点：
- 删除 `get_stats()` / `VRReceiverConfig` 的冗余字段
- 改用 SharedStorage 的 vr_ring（不再自己创建 `SharedMemoryFrameManager`）
- `shm/frame_manager.py`（129 行）删除，VR ring 管理合并到 SharedStorage

### 3.5 Main 进程: 1228 → ~80 行

```python
def main():
    args = parse_args()
    shared = SharedStorage.create(prefix="dexmani")

    procs = [
        mp.Process(target=camera_loop, args=(shared,), name="cam"),
        mp.Process(target=vr_loop,     args=(shared,), name="vr"),
        mp.Process(target=policy_loop, args=(shared,), name="pol"),
        mp.Process(target=arm_loop,    args=(shared,), name="arm"),
        mp.Process(target=hand_loop,   args=(shared,), name="hand"),
    ]
    for p in procs:
        p.daemon = True
        p.start()

    # 等待就绪（含自检）
    for name, ev, timeout in [
        ("arm", shared.arm_ready, 15),
        ("hand", shared.hand_ready, 15),
        ("camera", shared.camera_ready, 15),
        ("vr", shared.vr_ready, 15),
    ]:
        if not ev.wait(timeout=timeout):
            err = shared.startup_error.value.decode().strip('\x00') or "unknown"
            logger.error(f"{name} startup failed: {err}")
            shared.is_running.value = False
            shutdown(procs, shared)
            return

    try:
        while all(p.is_alive() for p in procs):
            if shared.error_state.value:
                logger.error("Error state set, shutting down")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown(procs, shared)


def shutdown(procs, shared):
    shared.is_running.value = False
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    shared.close()
```

### 3.6 CameraProcess: 基本不变 (~300 行)

当前 `camera_process.py` 已是独立 `mp.Process`。调整：改用 SharedStorage 的 `camera_ring`。`CameraRingBuffer` 从 `shm/ring_buffer.py` 移到 `sensor/camera_process.py`（它是相机的专属传输层）。CameraProcess 本身不再管理录制——只在 `is_recording` 时确保帧被发布到 ring，由 Policy 统一录制。

### 3.7 录制：集中式，Policy 拥有

**不采用"各进程独立写 .npz + 离线 merge"**——这会引入跨进程时间戳漂移和 action-state 配对不一致（ISSUE 4）。

**采用"Policy 拥有录制"**：

```
Policy 每 tick:
  arm_state = shared.arm_state_ring.read_latest()
  hand_state = shared.hand_state_ring.read_latest()
  camera = shared.camera_ring.read_latest()
  action = compute_action(arm_state, vr)

  # 所有数据来自同一 tick → 天然对齐
  recorder.add_frame(state=merge(arm_state, hand_state),
                     action=action, camera=camera)
```

Arm/Hand/Camera 不参与录制——只发布状态/帧到 ring。Policy 是唯一数据消费者，天然拥有正确的 `(state, action, camera)` 配对和统一时间戳。

`episode_recorder.py` 保持 accumulate-then-dump 模式（避免 fsync 阻塞），但从 884 → ~400 行（去掉 forward-fill 逻辑——不再需要——和 daemon thread 复杂度）。

### 3.8 SHM 层: 1,484(lines across 6 files) → ~350 行

| 文件 | 当前 | 改后 | 变化 |
|------|------|------|------|
| `shm/shared_storage.py` | — | +120 | **新建**，所有 ring/queue/event 的创建和访问 |
| `shm/ring_buffer.py` | 586 | ~200 | 删 CameraRingBuffer（移到 camera），seqlock 合并进基类 |
| `shm/robot_ring.py` | 282 | 0 | **删除**，合并到 ring_buffer.py |
| `shm/robot_rpc.py` | 151 | 0 | **删除** |
| `shm/robot_layouts.py` | 102 | ~30 | 仅保留 ArmState/HandState/HandCmd dtype（tactile_force 移到 hand_tactile dtype） |
| `shm/frame_manager.py` | 129 | 0 | **删除**，VR ring 管理合并到 SharedStorage |
| `shm/layouts.py` | 183 | 0 | **删除**，VR/Camera dtype 合并到对应的 `sensor/` 模块 |
| `shm/process_helpers.py` | 52 | 52 | **保留**——`install_sigint_handler()` 的 `threading.Event`（非 `mp.Event`）细节对信号安全关键 |

### 3.9 RobotInterface: 633 → ~350 行

| 删除项 | 行数 |
|--------|------|
| `return_to_home()` 重复实现（无任何外部调用者——6 个入口都直接 import `do_return_home`） | ~150 |
| `_check_joint_path_safe` 薄包装 | ~15 |
| `PROXIMAL_MASK` 等归位常量 | ~10 |

RobotInterface 退化为 arm/hand SDK 的薄封装，仅在以下场景使用：
- **`replay_traj.py`**（回放路径不需要 SharedStorage，直接持有 ArmInnerLoop）
- **`calibrate_camera.py`**（校准场景不需要多进程）
- 过渡期（Phase 1 期间，新架构尚未完全接入）

### 3.10 Validate: 158 → ~40 行

```python
# validate.py (~40 行)
def validate_action(arm_qpos: np.ndarray, hand_qpos: np.ndarray,
                    arm_error: bool, arm_connected: bool) -> tuple[bool, str]:
    """预发送安全门。fail-fast 顺序。"""
    if arm_error:
        return False, "arm_error"
    if not arm_connected:
        return False, "arm_disconnected"
    if not np.isfinite(arm_qpos).all():
        return False, "arm_qpos_nan"
    if not np.isfinite(hand_qpos).all():
        return False, "hand_qpos_nan"
    return True, "ok"
```

删除：`SupportsValidation` Protocol（仅 mypy 用）、`_validate_sensor_array` 单次 helper、lambda chain、velocity NaN 检查（SDK 自身保证）、`ThrottledWarner`（移到 `utils/`）。

### 3.11 `sim` 路径和 `replay_traj.py`

**`replay_traj.py`（1447 行）**：回放不需要多进程架构（从 HDF5 文件读取，无 VR，无 live recording）。保留当前单进程模式，使用精简后的 `RobotInterface` + `ArmInnerLoop`。**不在本次简化范围内，单独处理。**

**`examples/sim/`**：已删除（2026-08-03 架构迁移）。`TeleopPipeline`（原 `teleop/core/pipeline.py`，231 行）也已在 2026-08-04 teleop 扁平化中删除——其管线逻辑由 `vr_teleop_policy.py` 内联实现（arm_mapper → workspace clamp → EMA → IK → assemble）。若未来恢复仿真路径，直接复用 `teleop/arm_mapper.py` + `planning/ik.py`，无需中间 Pipeline 抽象。

---

## 4. 与 ManiUniCon 的对齐清单

| ManiUniCon 设计决策 | DexMani 对齐方式 |
|---------------------|-----------------|
| Main 不参与数据面 | Main 从 1228 行减到 80 行，只 orchestrate |
| SharedStorage 中心数据平面 | 新增 `shared_storage.py`，所有 ring/queue/event 集中创建 |
| Sensor / Policy / Robot 进程分离 | Camera + VR / Policy / (Arm + Hand) 五进程 |
| 状态用 Ring Buffer | arm_state/hand_state/hand_tactile/vr/camera 全部用 ring |
| Arm 动作用 Queue（有序执行） | `arm_action_q: mp.Queue(maxsize=2)`，bounded backpressure |
| Hand 动作用 Ring（latest-wins） | `hand_cmd_ring`，对应位置伺服语义 |
| 无 RPC 层 | 删除 `robot_rpc.py`，cross-proc 操作用 flag/ring/queue |
| Policy 统一录制 | Policy 读所有 state ring → 写 `TimestampAlignedBuffer` → 生成 HDF5 |
| 无 seqlock（接受统计级风险） | **保留 seqlock**（arm 控制不能接受 torn read） |
| 操作者在场是最好的安全系统 | 删除进程内重连、双路径急停、分级 watchdog |

**刻意保留的 DexMani 差异化：**

- Seqlock 协议（控制回路不能接受 torn read——ManiUniCon 明确说这是弱点）
- Arm+Hand 独立进程（12-DOF 灵巧手 ≠ 简单夹爪）
- Mode 6 轨迹规划 + NaN guard + joint limit clip（硬件安全底线）
- 集中式 HDF5 录制（Policy 拥有，但格式不变——训练代码无需修改）
- `ThrottledWarner`（6 个调用点，高频控制循环中实用）
- `process_helpers.py` 的信号安全细节

---

## 5. 实施路线

### Phase 1: 零风险清理（不改变行为）

| 步骤 | 内容 | 影响文件 |
|------|------|---------|
| 1.1 | 删 `interface.py` 的 `return_to_home()` 重复实现（确认无外部调用者） | `interface.py` |
| 1.2 | 删 `PRODUCER_REPLAY`/`PRODUCER_POLICY` 常量 + `producer_id` 门控 | `arm_process.py`, `robot_layouts.py` |
| 1.3 | 删 `SupportsValidation` Protocol（仅 mypy 用） | `validate.py` |
| 1.4 | 移 `ThrottledWarner` 到 `utils/`（保留，6 个调用点） | `validate.py`, `utils/throttle.py` |
| 1.5 | 删 `_validate_sensor_array` helper，inline 唯一调用 | `validate.py` |
| 1.6 | validate_action lambda chain → flat if | `validate.py` |
| 1.7 | 删 `HandSHMAdapter.get_state(force_update=...)` 参数 | `hand_process.py` |
| 1.8 | 删 `tracking_error_adaptive_max_rad`/`anomaly_cap_rad` config 字段；更新 `assess_trajectory_quality.py` 用固定阈值 | `inner_loop.py`, `tools/assess_trajectory_quality.py` |

**收益：~150 行删除，零行为变化。**

### Phase 2: 结构重构（核心变化）

| 步骤 | 内容 | 影响文件 |
|------|------|---------|
| 2.1 | **新建 `SharedStorage`** | `shm/shared_storage.py` (+120) |
| 2.2 | **新建 Policy 进程**（VR→IK 管道 + 状态机 + 录制 + 音频 + 键盘） | `policy/vr_teleop_policy.py` (+500) |
| 2.3 | **重写 Main**（1228→80 行） | `examples/real/vr_teleop_arm_only_record_plus.py` |
| 2.4 | **精简 ArmInnerLoop**（删 RPC/reconnect/双路径 estop；归位改为 HOME sentinel） | `robot/inner_loop.py` (1043→300) |
| 2.5 | **精简 Hand 进程**（删 RPC/trajectory/watchdog；改用 hand_cmd_ring） | `robot/hand_process.py` (1252→250) |
| 2.6 | **精简 VRReceiverProcess**（删 stats/config 冗余；改用 SharedStorage） | `sensor/vr_receiver_process.py` (380→250) |
| 2.7 | **合并 SHM**（ring_buffer 合并 seqlock+robot_ring；删 robot_rpc/frame_manager/layouts；CameraRingBuffer 移到 sensor） | `shm/*` |
| 2.8 | **拆分 tactile_force** 到独立 ring | `hand_process.py`, `shm/robot_layouts.py` |
| 2.9 | **定义 ArmState/HandState** 替代单体 RobotState | `robot/types.py` |
| 2.10 | **精简 RobotInterface**（删重复归位） | `robot/interface.py` (633→350) |

**收益：~4,000 行删除/重组，架构与 ManiUniCon 对齐。**

### Phase 3: 验证与清理

| 步骤 | 内容 |
|------|------|
| 3.1 | 真机冒烟测试（VR 遥操作一个 episode，验证录制格式兼容） |
| 3.2 | `check_episode_health.py` 跑新 episode，确认 HDF5 格式兼容 |
| 3.3 | 更新 CLAUDE.md 和内部文档 |
| 3.4 | `replay_traj.py` 适配新 RobotInterface（可能需单独简化任务） |

---

## 6. 最终代码行数目标

| 文件 | 当前 | 目标 | 变化 |
|------|------|------|------|
| `examples/real/vr_teleop_arm_only_record_plus.py` | 1228 | 80 | -94% |
| `policy/vr_teleop_policy.py` | — | 500 | 新建 |
| `robot/inner_loop.py` | 1043 | 300 | -71% |
| `robot/hand_process.py` | 1252 | 250 | -80% |
| `robot/arm_process.py` | 1006 | 重组到 inner_loop | — |
| `robot/interface.py` | 633 | 350 | -45% |
| `robot/validate.py` | 158 | 40 | -75% |
| `robot/types.py` | — | +50 | ArmState/HandState 类型 |
| `sensor/vr_receiver_process.py` | 380 | 250 | -34% |
| `shm/shared_storage.py` | — | 120 | 新建 |
| `shm/ring_buffer.py` | 586 | 200 | -66% |
| `shm/robot_ring.py` | 282 | 0 | 删除 |
| `shm/robot_rpc.py` | 151 | 0 | 删除 |
| `shm/robot_layouts.py` | 102 | 30 | -71% |
| `shm/frame_manager.py` | 129 | 0 | 删除 |
| `shm/layouts.py` | 183 | 0 | 删除 |
| `shm/process_helpers.py` | 52 | 52 | 保留 |
| `recording/episode_recorder.py` | 884 | 400 | -55% |
| **合计** | **~8,069** | **~2,622** | **-68%** |

---

## 7. 刻意不做的事

| 事项 | 原因 |
|------|------|
| 不做 Supervisor 进程 | Main 每 0.5s 检查子进程 `is_alive()` 足够 |
| 不做 capability negotiation | 单机器人单配置场景，不需要运行时协商 |
| 不做 temporal ensemble 动作平滑 | Mode 6 固件已做轨迹规划 |
| 不做无 seqlock 的 ring buffer | 控制数据 torn read 不可接受 |
| 不合并 arm+hand 为一个进程 | 灵巧手 ≠ 简单夹爪，故障隔离必要 |
| 不做硬实时保证 | Python multiprocessing 本来就做不到 |
| 不做跨模态严格时间同步 | 当前软同步对模仿学习足够；Policy-owns-recording 天然保证单一时钟域 |
| 不简化 `replay_traj.py` | 回放是独立的单进程模式，需要但不在此次范围内 |
| 不删除 `ThrottledWarner` | 6 个调用点，高频控制循环中的实用工具 |
| 不删除 `process_helpers.py` | 信号安全的 `threading.Event` 细节不可丢失 |

---

## 8. 风险与回退

### 已知风险

| 风险 | 缓解 |
|------|------|
| Policy 作为独立进程，VR→action 延迟增加一次 Queue 通信 | `mp.Queue` 内部用 Pipe + pickle，~0.1ms，可忽略 |
| ARM_ACTION_Q 的 `maxsize=2` 意味着 Policy 可能阻塞等 Arm | 这是正确行为——Arm 跟不上时不应积累 stale 命令。Mode 6 在 hold 期间轨迹平滑 |
| `hand_cmd_ring` 的 latest-wins 语义意味着快速手部动作可能掉帧 | 手部 retargeting 目前未接线（arm-only teleop）；接上后手部动作相对慢，16Hz 足够 |
| `replay_traj.py` 与新 RobotInterface 的兼容性 | Phase 3 单独验证；如问题大，replay 保留使用旧 interface |
| 多进程启动时自检失败没有诊断信息 | `startup_error` mp.Array 传递失败原因 |

### 回退策略

- **Phase 1 完全零风险**——纯删除死代码，可独立进行，每步骤 git commit
- **Phase 2 每步独立**——每个步骤可独立 merge 和真机验证
- **旧代码保留在 git history**——任何步骤可 `git revert` 回退
