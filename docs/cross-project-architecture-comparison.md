# DexMani vs ManiUniCon vs T-Rex/hardware vs LeFranX：进程、数据录制、同步架构对比审查

**审查日期**: 2026-07-21
**审查范围**: 4 个项目的进程架构、数据录制流水线、跨进程/跨线程同步机制
**方法**: 逐文件源码阅读 + 事实核验（所有断言附代码引用）

---

## 一、进程架构对比

### 1.1 进程拓扑

| 维度 | **DexMani** | **ManiUniCon** | **T-Rex/hardware** | **LeFranX** |
|------|------------|----------------|---------------------|-------------|
| **进程数** | 3+1（Main + Arm子进程 + Hand子进程 + Camera子进程） | 4+（Main + Robot + Policy + 每相机1进程） | **1**（纯多线程，无 multiprocessing） | 1（LeRobot `record_loop`） |
| **Arm控制** | 独立 fork 子进程 50Hz，`daemon=True`，独占 XArmAPI | Robot 进程内双线程：state_receiver 50Hz + control 100Hz | 主线程 30Hz 写 action buffer，独立线程 **300Hz** 执行 | 主线程 `FrankaFER.send_action()` |
| **Hand控制** | 独立 fork 子进程 30Hz，`daemon=False`（孤儿检测），独占 XHand | 无 Mentok 手（仅 Robotiq 夹爪） | 300Hz 线程发送 SharpaWave SDK | 主线程 `XHand.send_action()` |
| **相机** | 独立 CameraProcess（fork），`CameraRingBuffer` SHM | 独立 `mp.Process` 每相机，通过 `SharedStorage` | ZMQ receiver 线程（`HeadCameraReceiver` / `WristCameraReceiver`） | OpenCV 相机，主线程内 |
| **VR追踪** | `VRReceiverProcess`（fork），`SharedMemoryRingBuffer` SHM | 内嵌于 Policy 进程 | 内嵌于 `TeleopTargetSource`（主线程） | 内嵌于 `VRTeleoperator` |

**代码引用**:
- DexMani arm 子进程: `arm_process.py:183-386`（`_arm_child_main`）
- DexMani hand 子进程: `hand_process.py`（`_hand_child_main`）
- DexMani camera 子进程: `camera_process.py:1-7`（module docstring）
- ManiUniCon Robot 进程: `core/robot.py:16-36`（`class Robot(mp.Process)`）
- ManiUniCon Policy 进程: `core/policy.py`（同模式）
- T-Rex 单进程多线程: `main_teleop.py` — 所有组件为 `threading.Thread`，无 `mp.Process`
- LeFranX: `dual_vr_teleoperator.py:28-70`（`FrankaFERXHand` 组合机器人）

### 1.2 进程间通信机制

#### DexMani: SeqlockRingBuffer SHM + RPC（最复杂、最安全）

```
Main (16Hz)  ──arm_target──►  Arm Child (50Hz, fork, daemon=True)
              ◄──arm_state──   SeqlockRingBuffer (odd/even 协议)
              ──arm_cmd──►     RPC (macros: RESET/ESTOP/WAYPOINTS/REINIT_MODE6)
              ◄──result──

Main (16Hz)  ──hand_cmd──►   Hand Child (30Hz, fork, daemon=False)
              ◄──hand_state── SeqlockRingBuffer
              ──macro──►      RPC (RESET/STOP/CLEAR_ERROR/SEND_TRAJECTORY)
              ◄──result──     孤儿检测: 无新 cmd seq → hold + 安全退出
```

**代码引用**:
- `arm_process.py:87-91`: Ring 命名规范（`state` maxlen=3, `target`/`cmd`/`cmd_result` maxlen=2）
- `arm_process.py:235-243`: 子进程 attach rings（`create=False`）
- `arm_process.py:727-735`: 父进程 create rings（`create=True, stale_cleanup=True`）
- `robot_ring.py:1-98`: SeqlockRingBuffer odd/even 协议文档
- `robot_rpc.py`: RPC 实现（`RpcClient` / `RpcServer`）
- `hand_process.py:38-48`: daemon=False + 孤儿检测理由

#### ManiUniCon: SharedMemoryManager + SharedStorage

```
Main (supervisor) ──► Robot (mp.Process) ◄──policy_ready/robot_ready──► Policy (mp.Process)
                         │ 2 线程: state_receiver + control
                         │ PoseTrajectoryInterpolator
                    Sensors (N × mp.Process, 每相机)
```

**代码引用**:
- `main.py:29-31`: `SharedMemoryManager()` + `SharedStorage(shm_manager=...)`
- `shared_storage.py:140-259`: `SharedStorage` 定义 — ring buffers + queues + Events
- `core/robot.py:302-309`: 2-phase 握手（`policy_ready.wait()` → clear → `robot_ready.set()`）
- `shared_storage.py:185-196`: `robot_ready` / `policy_ready` Event 定义

#### T-Rex: threading.Lock + 共享 dict（单进程，最简单）

```
Main Thread (30Hz state machine)
  ├── Full Robot Action Thread (300Hz): action_buf_lock + hardware_lock
  ├── Tactile Fetch Thread (30Hz): tactile_buf_lock
  ├── Viz Render Thread (15Hz): viz_lock
  └── EpisodeKeyListener Thread: cbreak 模式热键
```

**代码引用**:
- `main_teleop.py:704-731`: action buffer 共享 + `action_buf_lock`
- `main_teleop.py:712`: `hardware_lock = threading.Lock()`
- `arm_hand_control.py:975-1101`: `full_robot_action_loop` — 300Hz 高频执行线程
- `config.py:105-108`: 速率定义 — `command_hz=30.0`, `arm_action_hz=300.0`

#### LeFranX: LeRobot 框架（无自定义 SHM）

**代码引用**:
- `dual_vr_record.py:308-317`: `record_loop(robot=robot, teleop=teleop, dataset=dataset, ...)`

### 1.3 错误隔离

| 故障场景 | **DexMani** | **ManiUniCon** | **T-Rex** | **LeFranX** |
|---------|------------|----------------|-----------|-------------|
| Arm 进程崩溃 | 主进程 `ensure_running()` 自动重建（`arm_process.py:549-577`）；重建前清除 stale rings | `error_state.value=True` → 全部停 | 单进程，任何线程崩溃 → 全部停 | `record_loop` 异常 → 全部停 |
| Hand 进程崩溃 | 非 daemon，孤儿检测 → 固件 hold（`hand_process.py:156`）；主进程可检测 | N/A（仅 Robotiq） | 同上 | 同上 |
| 相机进程崩溃 | 独立进程，`flag_camera_fresh` 标记冻结帧 → DataValidator 事后检测 | 传感器进程独立，主循环可检测 | ZMQ recv 线程崩溃 → 相机数据丢失 | 相机异常 → record_loop 中断 |
| 主进程死亡 | daemon arm 子进程跟随退出，固件 hold；非 daemon hand 孤儿检测退出 | 全部停（`mp.Process` 随父进程） | 全部停（同进程） | 全部停 |

**事实核验**: DexMani arm 的 `daemon=True` 和 hand 的 `daemon=False` 是**有意为之的区别**：
- `arm_process.py:442`: `daemon=True` — xArm7 Mode 6 固件在连接断开后自动 hold 位置
- `hand_process.py:154-165`: `daemon=False` — XHand 固件在失去命令刷新后行为不确定（假设 A4），需要非 daemon 子进程存活以继续 hold 或检测孤儿状态

### 1.4 Feature Flag 进程隔离切换

DexMani **独有**：可在不修改代码的情况下在进程内/进程间模式切换：

**代码引用**:
- `arm_process.py:896-949`: `make_arm_servo()` — 根据 `use_arm_isolation` 或环境变量 `DEXMANI_PROCESS_ISOLATION=1` / `DEXMANI_ARM_PROCESS_ISOLATION=1` 选择 `ArmInnerLoop`（进程内）或 `ArmInnerLoopSHMAdapter`（跨进程）
- `isolation.py`: `arm_isolation_enabled()` / `hand_isolation_enabled()` 环境变量检查
- `arm_process.py:783-813`: `ArmServo` Protocol — 两种模式都满足相同接口，调用方零改动

---

## 二、数据录制架构对比

### 2.1 存储格式与 Schema

| 维度 | **DexMani** | **ManiUniCon** | **T-Rex/hardware** | **LeFranX** |
|------|------------|----------------|---------------------|-------------|
| **主格式** | HDF5（单文件，所有流） | npz（每流独立文件：`state.npz` + `action.npz`） | HDF5（数值） + MP4/MKV（图像/触觉） | Parquet + 视频文件（LeRobot 数据集） |
| **Schema 版本** | **v8/v9** 显式版本号（`episode_recorder.py:32-33`） | 无版本号 | 无版本号（`data_writer.py` 无 schema 字段） | LeRobot 框架 schema |
| **压缩** | rgb/depth: **LZF**（`episode_recorder.py:593,604`）; pointcloud: gzip-1（`episode_recorder.py:621-622`）; 数值: gzip（`episode_recorder.py:683,698`） | 无压缩（npz 的 deflate 默认） | 相机: **libx264rgb CRF 18**（接近视觉无损）; 触觉: **libx264 -qp 0**（数学无损灰阶） | LeRobot 内置 |
| **Chunking** | 全部 HDF5 dataset 启用 chunking（`chunks=True`） | N/A（预分配 np array） | HDF5 resize per frame（无预分配 chunk 优化） | LeRobot 内置 |

**关键纠正**: DexMani 的相机数据**并非**原始未压缩数组。从 v8 开始，rgb/depth 使用 LZF 压缩（`episode_recorder.py:584-604`）。LZF 是 h5py 内置的快速压缩算法，图像数据通常实现 2-3× 压缩。但相比 H.264 视频编码（通常 30-50×），仍有 **~10-20× 的差距**。

### 2.2 录制流水线

#### DexMani: 三级流水线（最成熟的数据完整性保障）

```
TeleopController (16Hz)                 RecordingSession 线程              EpisodeRecorder
─────────────────────                   ────────────────────              ───────────────
record_frame(bundle)                     queue.Queue(maxsize=2000)
  → session.record(bundle)  ──FRAME──►  _run() 循环                        add_frame()
                                           → loop.record_frame(**bundle)     → TimestampAlignedBuffer
                                             → recorder.add_frame()            (record-time grid 对齐)
                                                                             → 相机帧入队
                                          ──STOP──►                        _stop_episode_impl()
                                                       _handle_stop()        → buffer flush → HDF5
                                                         → join_stop()       → 相机帧 forward-fill
                                                         → DataValidator    → meta attr 写入
```

**代码引用**:
- `recording_session.py:36-135`: `RecordingSession` 完整实现
- `recording_session.py:39-52`: `max_queue=2000`, `stop_timeout_s=35.0`, daemon 写线程
- `recording_session.py:66-71`: `record()` — `put_nowait` + `queue.Full` → drop + warning（背压策略）
- `recording_session.py:73-85`: `stop()` — STOP 排在所有 FRAME 之后入队，消除拖尾帧丢失
- `collection_loop.py:91-123`: `record_frame()` — 门控 `is_recording` + auto-stop on max_frames
- `episode_recorder.py:237-251`: `start_episode()` — `eps=0.5` 的 round-to-nearest 网格对齐
- `episode_recorder.py:584-604`: LZF 压缩的 rgb/depth dataset 创建
- `episode_recorder.py:42-55`: `_LIVE_RECORDERS` WeakSet + `atexit.register(_flush_all_recorders)`
- `episode_recorder.py:337-349`: `add_frame()` — 门控 + 对齐 + 相机入队

#### ManiUniCon: 双流 Falling-edge Dump

```
Robot._state_receiver_thread (50Hz)              Robot.run() (100Hz)
──────────────────────────────────              ─────────────────────
if is_recording:                                   read_all_action()
  buffer.add(state)                                if is_recording:
else:                                                buffer.add(action)
  if buffer:                                       else:
    dump("state.npz")                                if buffer:
    buffer = None                                      dump("action.npz")
```

**代码引用**:
- `core/robot.py:186-204`: state buffer 的 falling-edge dump — `is_recording` 从 True→False 时触发
- `core/robot.py:369-380`: action buffer 的相同模式
- `utils/timestamp_accumulator.py:87-212`: `TimestampAlignedBuffer` — 预分配 `max_record_steps`，`overwrite=False` 模式按时间戳分配全局索引

**关键特征**: 录制期间**零磁盘 I/O**。所有数据在内存中累积到 `max_record_steps`（默认 500，≈5s @100Hz）。崩溃 = 丢失整个 episode。无增量 flush，无 atexit 安全网。

#### T-Rex: 异步多格式分流写入

```
Main Thread (30Hz)                  DataWriter Thread
──────────────────                  ─────────────────
data_writer.queue_frame(frame_data)   _writer_loop():
  → queue.Queue.put()                    queue.get(timeout=0.1)
                                          _process_frame(data)
                                            → HDF5 resize(idx) + write
                                            → MP4 writeFrame(rgb)
                                            → MKV writeFrame(tactile_gray)
                                            → per 100 frames: hdf5.flush()
```

**代码引用**:
- `data_writer.py:181-664`: `DataWriter` 完整实现
- `data_writer.py:522-531`: `_writer_loop()` — `queue.get(timeout=0.1)` + `_process_frame()`
- `data_writer.py:536-581`: `_process_frame()` — per-frame HDF5 resize + video write + periodic flush
- `data_writer.py:456-483`: 相机视频设置: `libx264rgb`, `-crf 18`, `-preset ultrafast`, `-pix_fmt rgb24`
- `data_writer.py:164-178`: 触觉视频设置: `libx264 -qp 0`（数学无损）或 `ffv1`（归档级无损）, `-pix_fmt gray`
- `data_writer.py:144-146`: 相机分辨率: `IMAGE_WIDTH=640, IMAGE_HEIGHT=360`
- `data_writer.py:583-612`: `_write_tactile_data()` — 5 指水平拼贴 `(H, 5*W)` 单通道灰阶帧
- `data_writer.py:636-664`: `stop()` — drain queue + join thread + 写 metadata attrs + 关闭所有 writer

### 2.3 磁盘占用估算（60s episode）

| 流 | DexMani (HDF5+LZF, 16Hz, 640×480) | T-Rex (HDF5+MP4/MKV, 30Hz, 640×360) |
|----|-----------------------------------|-------------------------------------|
| RGB 相机 | ~280-420 MB（LZF 2-3×） | ~15-40 MB（H.264 CRF 18） |
| Depth 相机 | ~190-280 MB（LZF 2-3×） | N/A（T-Rex 无深度） |
| 数值数据（arm/hand/action） | ~5-15 MB（gzip） | ~5-15 MB（HDF5） |
| 触觉 DEFORM+RAW | ~6 MB（当前 force 数据） | ~20-60 MB（无损灰阶 MKV, 5指×2手） |
| **合计** | **~480-720 MB** | **~40-115 MB** |

**结论**: 即使 DexMani 已使用 LZF 压缩，相机数据仍是主要存储瓶颈。T-Rex 的视频编码策略可节省 **~5-15×** 的相机存储空间。如果 DexMani 引入同策略，60s episode 从 ~500-700 MB 降至 ~30-100 MB。

### 2.4 数据验证

**DexMani 独有**：8 项自动验证检查（`data_validator.py:1-16`）：

1. `no_nan_obs` — arm_qpos, arm_ee, hand_qpos 无 NaN
2. `no_nan_action` — action_arm_joint, action_arm_ee, action_hand_joint 无 NaN
3. `non_zero_variance` — 每维度 variance > epsilon
4. `camera_fresh` — 前 10 帧不全为零
5. `min_frames` — episode 达最小帧数（默认 50）
6. `no_duplicate_frames` — 无连续相同帧（传感器卡住）
7. `timestamp_monotonic` — 时间戳无倒退
8. `camera_stall` — `flag_camera_fresh` 冻结帧占比 ≤10%

**代码引用**: `data_validator.py:65-131`（`validate()` 方法）, `recording_session.py:122-126`（仅在 `validate=True` 时调用）

**其他项目**: ManiUniCon、T-Rex、LeFranX 均无对等的自动数据验证层。

### 2.5 元数据

**DexMani**: 最丰富的元数据（`episode_recorder.py:253-307`）:
- 基础：`task_label`, `operator`, `tags`, `control_hz`, `fps`, `schema_version`
- 相机：`camera_serial`, `camera_type`, `camera_T_world_camera`, `camera_T_eef_camera`, `camera_K`, `depth_scale`
- 点云：`pc_num_points`, `pc_depth_min`, `pc_depth_max`, `pc_voxel_size`, `has_pointcloud`
- 手部：`hand_delta_clip`, `hand_ema_alpha`, `hand_low_pass_alpha`
- 控制：`ema_alpha_pos`, `ema_alpha_rot`
- 录制：`skip_initial_frames`, `truncated`, `stop_reason`, `cam_frames_dropped`, `cam_items_written`
- record_config（动态，如 `arm_joint_max_delta` 等）
- 额外：sidecar JSON（`collection_loop.py:178-199`）

**T-Rex**: 基础元数据（`data_writer.py:637-650`）: `total_steps`, `command_hz`, `episode_duration`, `tactile_maps_storage`, `tactile_video_codec`

---

## 三、同步机制对比

### 3.1 核心同步方案

| 维度 | **DexMani** | **ManiUniCon** | **T-Rex** |
|------|------------|----------------|-----------|
| **热路径锁策略** | **odd/even Seqlock**（无锁，`robot_ring.py:67-98`） | Python `mp.Lock`（有锁，RingBuffer 内部） | `threading.Lock`（进程内有锁） |
| **跨进程信号** | `mp.Event` × 4（estop/stop/ready/crashed） | `mp.Event` × 5（running/recording/error/robot_ready/policy_ready） | N/A（单进程 `threading.Event`） |
| **RPC** | cmd/result ring 对 + seq 相关 + timeout（`robot_rpc.py`） | 无（`SharedStorage` 直写） | 无（共享 dict + lock） |
| **读写模式** | **SPSC**（每 ring 单生产者单消费者，plan §4.1-4.6） | 多读者单写者（`SharedStorage`） | 多读者多写者（多线程，lock 保护） |
| **内存模型** | 显式 layout 定义 + 64B cache-line 对齐（`robot_layouts.py`） | Python `multiprocessing.shared_memory` 自动管理 | Python 堆对象（同进程） |
| **死锁风险** | **零**（无锁设计 + timeout 保护） | 存在（无显式死锁检测） | 存在（无显式死锁检测） |

### 3.2 DexMani Seqlock 协议的独特性

DexMani 的 `SeqlockRingBuffer` 实现了**真正的 odd/even seqlock 协议**，这是本项目相对于其他三个项目的最显著技术优势。

**协议定义**（`robot_ring.py:1-38`）:

```
写入:
  slot.sequence = 2*seq - 1    ← ODD（= 写入进行中）
  slot.timestamp_ns = now_ns
  slot.data = payload
  slot.sequence = 2*seq        ← EVEN（= 写入完成）

读取:
  seq1 = slot.sequence
  if seq1 == 0 or (seq1 & 1):  → TORN（奇数 = 写入中，或 0 = 未写入）
  copy timestamp + data
  seq2 = slot.sequence
  if seq1 != seq2:             → TORN（写入者已覆盖此 slot）
  return data                  ← EVEN 且前后一致 = 有效
```

**为什么不能用简单的 seq1==seq2 重检**（`robot_ring.py:23-31`）:

> TSO 下存在合法交叠：读者的 seq1 load 落在写者的 sequence store 之后、data stores 之前 → seq1==seq2==新 seq → torn data 通过验证。odd/even 消除了这个窗口：进行中的 slot 永远是奇数，读者第一采样是奇数 → 直接拒绝。

**对比**:

| 方案 | 使用位置 | TSO 安全性 |
|------|---------|-----------|
| `CameraRingBuffer.read_latest`（简单重检） | `ring_buffer.py:467-475` | **有窗口**：maxlen=5 @30Hz=~167ms，风险较低 |
| `SeqlockRingBuffer.read_latest`（odd/even） | `robot_ring.py`（继承 `SharedMemoryRingBuffer` 的 layout） | **无窗口**：针对 maxlen=3 @50Hz=~60ms（单次 IK spike 即可 wrap） |

**代码引用**:
- `robot_ring.py:67-98`: SeqlockRingBuffer 类文档 + torn-read 降级策略（重试 1 次 → last-good 缓存 → None → throttled warning ≤1/5s）
- `ring_buffer.py:33-52`: `SharedMemoryRingBuffer` 基类 — 纯 FILO（依赖 x86_64 aligned uint64 store 的指令级原子性）
- `ring_buffer.py:260-515`: `CameraRingBuffer` — 简单 re-read 重检 + torn-read size/shape validation（line 416-426, 438-446）

### 3.3 2-Phase Policy-Robot 同步握手

ManiUniCon 引入，DexMani **已复制基础设施但未接线**。

**ManiUniCon 实现**（`shared_storage.py:185-196`）:
```python
self.robot_ready = Event()    # 初始 True：robot 准备好接收第一个 action
self.robot_ready.set()
self.policy_ready = Event()   # 初始 False：policy 尚未生成 action
```

**DexMani 对应实现**（`sync_primitives.py:1-48`）:
```python
class SharedSyncPrimitives:
    """Two-phase handshake events for synchronized policy-robot execution.
    Ref: ManiUniCon SharedStorage."""
    def __init__(self):
        self.robot_ready = mp.Event(); self.robot_ready.set()
        self.policy_ready = mp.Event()
```

**当前接线状态**: `sync=None`。`arm_process.py:941` 的 `make_arm_servo()` 在 isolation 路径传 `sync=None`。`arm_process.py:230-231` 中 `ArmInnerLoop` 需要 `inner_cfg.synchronized=True` 才会使用 sync。**基础设施已就位，等待 Policy 部署时接线。**

### 3.4 Freshness Gate + Range Sanity 双重防护

DexMani 独有（`arm_process.py:579-617`）:

```
get_state() 流程:
  1. read_latest() → None?                  → fabricate error record（connected=0, error_state=1）
  2. age_ns > state_stale_mult/loop_hz?     → fabricate error record（从 donor/last-good 复制 qpos）
  3. qpos 非有限 或 超出 [soft_min-0.05, soft_max+0.05]? → 视为 torn read → 返回 last-good cache
  4. 通过所有检查 → 更新 last-good cache → 返回有效记录
```

**设计理由**（`arm_process.py:584-588`）:
> age > state_stale_mult/loop_hz → fabricated error record（error_state=1, connected=0），所以 validate_action 会拦截，而不是用陈旧的 tau/temps 做门控（[[l515-midrun-stream-stall]] 的教训）。

**默认参数**:
- `target_timeout_s=0.2`（arm_target 的 staleness gate）
- `state_stale_mult=3.0`（state 的 staleness gate：3/50Hz=60ms）
- `target_timeout_s` 从 `ArmInnerLoopConfig` 传播到 `ArmProcessConfig`（`arm_process.py:939`）

### 3.5 优先级通道

DexMani 在 arm 子进程 tick 循环中实现了显式的优先级顺序（`arm_process.py:303-363`）:

```python
while not stop_event.is_set():
    limiter.wait()
    # 1. estop FIRST（优先级最高，绕过 macro_lock）
    if estop_event.is_set():
        inner.set_target(None)
        if not estop_done:
            fast_estop()  # set_state(4) on own connection, ≤1 tick
        continue
    # 2. SIGINT → hold + exit
    # 3. Target ring → inner loop
    # 4. Publish state every tick
    # 5. RPC macros (dedicated thread, macro_lock)
```

**其他项目**: 均无对等的优先级通道设计。

---

## 四、Gap 分析：DexMani 最有收益的改动

### 事实核验后的修正

以下每一项建议都经过了跨项目代码的事实核验。

### Top 1: 相机 LZF → H.264 视频编码（借鉴 T-Rex）

**核验状态**: ✅ 已确认

**当前状态事实**:
- DexMani 相机: 640×480 RGB + 640×480 Depth，LZF 压缩（`episode_recorder.py:584-604`）
- LZF 压缩比: ~2-3×（LZF 是速度优先的轻量压缩器，非图像专用）
- 每个 60s episode 相机数据: ~470-700 MB（经 LZF）
- LZF 压缩在后台线程执行（`episode_recorder.py:121-128`: `_cam_queue` + `_cam_writer` 线程），防止阻塞热路径

**目标状态**:
- 借鉴 T-Rex `data_writer.py:456-483` 的 `skvideo.io.FFmpegWriter` + `libx264rgb CRF 18`
- 同等质量下相机数据 → ~20-40 MB/episode（**节省 ~10-20×**）
- 保持 frame k ≡ HDF5 row k 的索引对齐（T-Rex 模式: `video frame k == hdf5 row k`）

**实现风险**: 低
- 仅改 `EpisodeRecorder._cam_writer` 线程（`episode_recorder.py:120-128`），不碰控制链路
- 需添加 `skvideo` 依赖（`pip install scikit-video`）
- 深度图需要特殊处理（灰度 16-bit → 可考虑 2×8-bit 通道或 HEVC 10-bit）

### Top 2: 2-Phase Policy 同步握手接线（借鉴 ManiUniCon）

**核验状态**: ✅ 已确认基础设施存在，未接线

**当前状态事实**:
- `SharedSyncPrimitives` 已定义（`sync_primitives.py:1-48`），含 `mp.Event` 的 `robot_ready` / `policy_ready`
- `arm_process.py:230-231`: `ArmInnerLoop._sync = sync` 仅当 `inner_cfg.synchronized=True` 时注入
- `arm_process.py:941`: `make_arm_servo()` 在 isolation 路径传 `sync=None`
- ManiUniCon 的握手在 `core/robot.py:302-309, 442-454`

**目标状态**: 为 Policy 进程部署预留正确架构，避免日后的 workaround

**实现风险**: 中等
- 需要修改 `InnerLoop` 的 tick 逻辑（当前 `synchronized` 参数是预埋的，但实际同步逻辑可能不完整）
- 当前 VR teleop 不需要此功能，收益在**未来** policy deployment 场景

### Top 3: 触觉 Map 视频压缩（借鉴 T-Rex）

**核验状态**: ✅ 已确认 T-Rex 方案可行

**当前状态事实**:
- DexMani 触觉 force 数据: `/hand_tactile_force(5,120,3)` float32（~7.2KB/帧，数据量不大）
- 如果未来加 DEFORM/RAW map（像 T-Rex 的 `(5,240,240)` + `(5,240,320)`），原始存储将爆炸

**目标状态**: 借鉴 T-Rex `data_writer.py:486-500` 的无损灰阶 MKV

**实现风险**: 低（当前数据量小，按需实施即可）

### Top 4: LeRobot 格式导出工具（借鉴 LeFranX）

**核验状态**: ✅ 已确认可行

**当前状态事实**:
- DexMani 已有 `export_hdf5_to_zarr.py`（`tools/` 目录）
- LeFranX 使用 `LeRobotDataset.create()` + `record_loop()`（`dual_vr_record.py:201-208`）

**目标状态**: 加一个 `export_hdf5_to_lerobot.py`（同现有 Zarr 导出模式），让 DexMani 数据进入 LeRobot 训练生态

**实现风险**: 低（纯工具链，不影响采集）

---

## 五、不建议借鉴的反模式

| 项目 | 不借鉴什么 | 为什么 | 代码证据 |
|------|----------|--------|---------|
| T-Rex | 单进程多线程模型 | DexMani 的多进程隔离是**明确的架构优势**：arm/hand/camera 各自崩溃不影响主循环。T-Rex 的 300Hz action 线程异常会带走整个进程 | `main_teleop.py`: 所有线程 daemon，无进程隔离 |
| ManiUniCon | Falling-edge dump（录制期间零 I/O） | DexMani 的 `TimestampAlignedBuffer` + 周期性 flush（~10s 间隔）比纯内存 dump 更抗崩溃：中间崩溃只丢 ~10s 而非整个 episode | `core/robot.py:186-204` vs `episode_recorder.py:96-97` |
| ManiUniCon | `mp.Process` per camera | DexMani 的 `CameraProcess` 一次 fork 管所有相机更轻量；ManiUniCon 多相机进程增加 SHM 碎片和启动复杂度 | `main.py:82-84`（per-sensor start） |
| LeFranX | 全面采用 LeRobot 框架 | 框架锁定太重（`record_loop` 必须传入 `LeRobotDataset`）；DexMani 的轻量 HDF5 直写更灵活，导出工具足以桥接 | `dual_vr_record.py:308-317` |

---

## 六、推荐实施顺序

```
Phase 1 (低风险、高收益):  相机 LZF → H.264 视频编码   磁盘 -10~20×, 热路径零改动
Phase 2 (架构预留):       2-Phase 同步握手接线        Policy 部署架构就绪
Phase 3 (按需):           触觉视频压缩                 为高分辨率触觉扩展准备
Phase 4 (工具链):         LeRobot 格式导出             训练工具链复用
```

---

## 附录：项目关键文件索引

### DexMani
| 文件 | 职责 |
|------|------|
| `dexmani_real/robot/arm_process.py` | Arm 子进程 + ArmSHMFaçade + ArmServo Protocol |
| `dexmani_real/robot/hand_process.py` | Hand 子进程 + HandSHMFaçade |
| `dexmani_real/shm/robot_ring.py` | SeqlockRingBuffer（odd/even seqlock 协议） |
| `dexmani_real/shm/ring_buffer.py` | SharedMemoryRingBuffer + CameraRingBuffer |
| `dexmani_real/shm/robot_layouts.py` | 所有 SHM dtype 定义 |
| `dexmani_real/shm/robot_rpc.py` | RpcClient / RpcServer（cmd/result ring 对） |
| `dexmani_real/shm/sync_primitives.py` | 2-Phase 握手 Events |
| `dexmani_real/recording/recording_session.py` | 单写线程 + 队列驱动的录制会话 |
| `dexmani_real/recording/episode_recorder.py` | HDF5 录制器（TimestampAlignedBuffer + 相机队列） |
| `dexmani_real/recording/collection_loop.py` | 录制生命周期编排 |
| `dexmani_real/recording/data_validator.py` | 8 项自动数据验证 |
| `dexmani_real/recording/timestamp_buffer.py` | TimestampAlignedBuffer（网格对齐） |
| `dexmani_real/sensor/camera_process.py` | 相机子进程（640×480 @30Hz） |

### ManiUniCon
| 文件 | 职责 |
|------|------|
| `main.py` | 主入口（SharedMemoryManager + 进程启动/停止） |
| `core/robot.py` | Robot 进程（双线程 + falling-edge dump + 2-phase 同步） |
| `utils/shared_memory/shared_storage.py` | SharedStorage（SHM rings + Events） |
| `utils/timestamp_accumulator.py` | TimestampAlignedBuffer + 网格分配算法 |

### T-Rex/hardware
| 文件 | 职责 |
|------|------|
| `hardware_code/teleop/main_teleop.py` | 主入口（状态机 + 单进程多线程） |
| `hardware_code/teleop/data_writer.py` | DataWriter（HDF5 + MP4/MKV 异步写入） |
| `hardware_code/teleop/arm_hand_control.py` | ArmIKManager + SmoothingAndSafetyManager + 300Hz action loop |
| `hardware_code/teleop/config.py` | TeleopConfig（command_hz=30, arm_action_hz=300） |

### LeFranX
| 文件 | 职责 |
|------|------|
| `scripts/dual_robot/dual_vr_teleoperator.py` | VR 遥操作测试脚本 |
| `scripts/dual_robot/dual_vr_record.py` | 录制脚本（LeRobotDataset + record_loop） |
| `src/lerobot/robots/franka_fer_xhand/franka_fer_xhand.py` | FrankaFER + XHand 组合机器人 |
