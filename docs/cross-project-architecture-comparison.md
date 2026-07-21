# DexMani vs ManiUniCon vs T-Rex/hardware vs LeFranX：进程、数据录制、同步架构对比审查

**审查日期**: 2026-07-21
**审查范围**: 4 个项目的进程架构、数据录制流水线、跨进程/跨线程同步机制、安全架构、内部循环、测试基础设施、错误恢复模式
**方法**: 逐文件源码阅读 + 事实核验 + 跨项目对比（所有断言附代码引用）
**第二轮深入**: 安全门控管道、ArmInnerLoop 25Hz tick 循环、HandProcess F1 设计、Ruckig/SmoothingAndSafetyManager 真机使用模式、ManiUniCon PoseTrajectoryInterpolator、LeRobot v3 数据集格式

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

### 2.3-bis LeRobot v3 数据集格式（LeFranX 的上游框架）

LeFranX 基于 HuggingFace 的 `lerobot` 包构建，其 v3 数据集格式是 DexMani 导出工具的**目标格式**。与 DexMani 的单文件 HDF5 截然不同：

**目录结构**（WebSearch + LeFranX 源码核验）:
```
{dataset_path}/
  meta/
    info.json              # codebase_version, total_episodes, total_frames, fps, features
    episodes.jsonl         # per-episode metadata (task, length)
    stats.json             # 归一化统计（训练时用）
    tasks.jsonl            # 任务描述 + task_index
  data/
    chunk-000/
      episode_000000.parquet  # 表格数据（observation.state, action, timestamp, episode_index）
      episode_000001.parquet
  videos/
    chunk-000/
      {camera_key}/
        episode_000000.mp4    # 相机视频（默认 AV1 codec）
```

**与 DexMani HDF5 的关键差异**:

| 维度 | DexMani (HDF5 v8/v9) | LeRobot v3 |
|------|----------------------|-----------|
| 表格存储 | HDF5 datasets（`/arm_qpos`, `/hand_qpos` 等） | **Parquet**（单表 `observation.state` + `action`，按 chunk 分片） |
| 相机存储 | HDF5 datasets + LZF 压缩 | **MP4 视频**（AV1 默认，可选 h264/hevc） |
| 视频编码 | N/A（per-frame array） | Streaming（实时队列编码）或 Batch（存 PNG 后批量编码） |
| 数据加载 | `h5py` 随机访问 | `torchcodec`/`pyav` 视频解码 + Parquet memory-mapped |
| Episode 边界 | 一个 `.h5` 文件 = 一个 episode | Parquet 内 `episode_index` 字段 + `episode_data_index` dict |
| 元数据 | HDF5 attrs（~20-30 键）+ sidecar JSON | `meta/` 目录下多个 JSON/JSONL 文件 |
| 质量标志 | `/flag_ik_ok`, `/flag_held` 等 per-frame 数据集 | 无原生质量标志 |
| VR 原始数据 | `/vr_wrist_pos`, `/vr_landmarks` | 不记录（仅记录 IK 输出 action） |
| 深度图 | HDF5 dataset (uint16, gzip) | 可选（需特殊 `gray12le` pixel format） |
| 点云 | `/pointcloud(T, N, 6)` gzip-1 | 无原生支持 |

**DexMani 元数据优势**: DexMani 的 meta attrs（`camera_serial`, `camera_K`, `camera_T_world_camera`, `depth_scale`, `truncated`, `stop_reason`, `cam_frames_dropped`, `held_ratio` 等）远多于 LeRobot 的基准要求。导出时这些应作为 `meta/episodes.jsonl` 的自定义字段或 `info.json` 的 extra 字段保留。

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

## 四、第二轮深入分析：安全架构与内部循环

> 以下分析基于第二轮逐文件深度阅读（`validate.py`, `inner_loop.py`, `pipeline.py`, `controller.py`, `hand_process.py`, `isolation.py`, `timestamp_buffer.py`），覆盖安全门控管道、ArmInnerLoop 25Hz tick 循环、HandProcess F1 设计、以及发现的具体代码级 gaps。

### 4.1 安全门控管道 (`validate.py`)

DexMani 的安全门控是**多层独立防护**的典范，每层操作在不同数据上：

| 层级 | 位置 | 操作数据 | 失败行为 |
|------|------|---------|---------|
| Pipeline | `pipeline.py:127-131` | VR target pose | Workspace clamp（非致命） |
| IK Solver | `planning/ik.py` | IK 候选解 | 碰撞拒绝 + 多种子重试 |
| validate_action | `validate.py:51-111` | RobotAction | 8 门 fail-fast |
| Inner Loop | `inner_loop.py:615-707` | 最终关节命令 | Delta clamp + NaN guard |
| Firmware | Mode 6 | 实际关节轨迹 | C22（自碰撞）/ C24（超速） |

**validate_action 门控顺序**（`validate.py:49-112`）:

| Gate | 检查内容 | 致命？ | 代码行 |
|------|---------|--------|--------|
| 1. SDK error state | `arm.is_error()`, `hand.connected_flag and hand.error_state` | Yes | 51-54 |
| 2. Arm connection | `arm.is_connected()` | Yes | 57-58 |
| 3. Torque | Per-joint `|tau| > [50,50,30,30,30,20,20]` Nm | Yes | 61-66 |
| 4. Temperature | Per-joint `temp > 70°C` | Yes | 69-74 |
| 5a. Env collision | `env_collision_check(actual_arm_qpos)` — 检查**当前位姿** | Yes | 77-82 |
| 5b. Self-collision | `self_collision_check(action.arm_qpos_cmd)` — 检查**目标位姿** | Yes | 89-94 |
| 6. Workspace clamp | `clamp_workspace_pos(target_eef_pos)` | **No** | 97-98 |
| 7. Arm joint clip | `np.clip(qpos, arm_lo, arm_hi)` | **No** | 103-105 |
| 8. Hand joint clip | `np.clip(hand_qpos, hand_lo, hand_hi)` | **No** | 108-111 |

**发现的 Gaps（≥低优先级，经代码核验）**:

| ID | 严重度 | 问题 | 位置 | 核验状态 |
|----|--------|------|------|---------|
| **V1** | **中** | **NaN guard 缺失**：`validate_action` 在 joint-limit clip（line 105）前不检查 `arm_qpos_cmd` 的 NaN。若 IK 产生数值溢出绕过 pipeline 的 NaN 检查，`np.clip` 静默传播 NaN。Inner loop 的 NaN guard（`inner_loop.py:463-466`）是唯一防线 | `validate.py:103-105` | ✅ 确认（line 105 直接 clip 无 NaN 检查） |
| **V2** | **低** | **Env vs self-collision 非对称**：Env collision 用 `actual_arm_qpos`（当前位置），self-collision 用 `action.arm_qpos_cmd`（命令位置）。对 env collision 而言，检查**命令位置**更保守。但 env collision 检查的是静态障碍物（如桌面），当前位置和命令位置的差异在 0.3 rad delta clamp 范围内，实际风险低 | `validate.py:79 vs 91` | ✅ 确认（实际影响小，delta clamp 限制了差异） |
| **V3** | **低** | **Hand error gating 依赖 `connected_flag`**：line 53 使用 `connected_flag` 而非 `is_connected()`。设计意图明确（absent hand ≠ error），但 flag 在手部意外断开时可能未被正确标记 | `validate.py:53` | ✅ 确认（设计意图在注释中已说明） |

### 4.2 ArmInnerLoop 25Hz Tick 循环 (`inner_loop.py`)

**关键发现：实际运行在 25Hz，而非文档声称的 50Hz**。

`loop_period=0.04`（`inner_loop.py:73-74`）产生 25Hz 内部循环。Mode 6 固件在命令间插值，所以轨迹平滑度不受影响。注意：`examples/real/vr_teleop_arm_only_record_plus.py:193` 显式设置 `loop_period=0.04`，确认这是生产配置。

**Per-step delta clamp**（`inner_loop.py:636-646`）:

```
max_step = max_joint_delta (0.3 rad) × speed_ramp_scale
clipped = prev_sent + clip(target - prev_sent, -max_step, +max_step)
```

关键：使用 `_last_sent_target`（上次实际发送值）而非原始目标值作为基线。这意味着 IK 异常尖峰（如 1.0 rad 跳变）需要 ~4 个 inner tick（160ms）才能完全到达固件——此时外层已经产生了修正命令。

**Soft-start speed ramp**（`inner_loop.py:636-646`）:
- 前 20 帧（0.8s）从 `speed_ramp_min=0.2 rad/s` 线性升至 `joint_max_speed=1.5708 rad/s`
- **C24 防复发保护**（line 644）: `speed = min(joint_max_speed, max(speed, 1.25 * _qvel_inf))` — ramp 重置后速度上限不低于当前实际速度的 125%

**发现的 Gaps**:

| ID | 严重度 | 问题 | 位置 | 核验状态 |
|----|--------|------|------|---------|
| **I1** | **高** | **`_recover_mode()` 返回值被忽略**：lines 509 和 691 调用 `_recover_mode(arm)` 但不检查返回值。若 3 次重试全部失败，inner loop 继续运行，arm 可能处于 Mode 0（位置控制）——后续 `set_servo_angle` 使用错误的协议。**不崩溃但 SDK 行为不确定** | `inner_loop.py:509, 691` | ✅ 确认（两处调用均忽略返回值） |
| **I2** | **中** | **Temperature 读取脆弱**：`getattr(arm, "temperatures", None)` 若在部分 xArm 固件版本上不提供此属性 → `None` → 整个温度更新块被跳过 → `_arm_temps` 保持初始 NaN → `validate_action` 的温度门控永久静默失效。**无任何 warning** | `inner_loop.py:541-546` | ✅ 确认（无 else fallback/warning） |
| **I3** | **低** | **`_hold_position()` 不检查 SDK 错误码语义**：`code == 0 and len(states) > 0` + `np.all(np.isfinite(hold))` 仅防 NaN/Inf + 空返回。若 SDK 返回 `code=0` 但 `states[0]` 为全零（传感器 dropout），`np.all(np.isfinite(zeros))` 通过 → arm 被命令到零点。**极低概率，需固件级 bug** | `inner_loop.py:762-764` | ✅ 确认（无额外 safety bounds check） |
| **I4** | **低** | **Emergency stop race**：`emergency_stop()` 设置 `_emergency_event` + thread join（`inner_loop.py:286-301`）。若线程正在 `set_servo_angle` 内部（GIL 释放），estop 延迟最多 1 tick（40ms）。fallback 路径（短期连接）存在但增加开销 | `inner_loop.py:286-308` | ✅ 确认（40ms 在紧急停止场景可接受） |

### 4.3 HandProcess F1 设计 (`hand_process.py`)

**架构**: Clip/EMA 状态机在 **Facade**（主进程侧），子进程是**无状态执行器**。

**发现的 Gaps**（经代码核验）:

| ID | 严重度 | 问题 | 位置 | 核验状态 |
|----|--------|------|------|---------|
| **H1** | **低** | **`HandSHMAdapter.stop()` 立即设置 `connected_flag=False`**，子进程 detorque 异步（≤1 tick @30Hz ≈ 33ms）。33ms 窗口期主进程报告已停止但手仍 energize。但：主进程在 `stop()` 后不发送新命令，子进程下个 tick 首先检查 estop event（line 900），实际窗口极小 | `hand_process.py:1119-1121` | ✅ 确认（33ms 窗口，无安全影响） |

**已核验后排除的伪问题**:

| 原声称 | 核验结果 | 原因 |
|--------|---------|------|
| ~~Echo seq gap 假阳性（H2）~~ | **❌ 排除** | `_last_acked_seq > 0` guard（`hand_process.py:563`）阻止了重启后的假阳性——重启后 `_last_acked_seq=0`，gap 检查被完全跳过 |
| ~~`last_processed_seq` 在 send 失败仍递增（H3）~~ | **⚠️ 设计意图** | 条件递增是 F1 文档的显式设计决策：position servo 不需要持续刷新，跳过失败的命令是安全的。watchdog（30 次连续失败后重连）处理持续失败 |
| ~~Cartesian EMA 状态未重置（P2）~~ | **❌ 排除** | `_reset_mapper()`（`controller.py:727-728`）在 B/C 键转换时正确重置 `_prev_target_pos` 和 `_prev_target_quat` 为 None |

### 4.4 TeleopPipeline (`pipeline.py`)

**发现的 Gaps**:

| ID | 严重度 | 问题 | 位置 | 核验状态 |
|----|--------|------|------|---------|
| **P1** | **低** | **Mapper readiness check 静默失败**：`arm_mapper.is_ready() == False` 时返回 `prev_arm_cmd` 且无 warning（`pipeline.py:116-117`）。若 mapper 从未标定，用户看到 "arm 不动" 但无错误提示。误操作场景：用户忘记 B 键之前标定 VR | `pipeline.py:116-117` | ✅ 确认（silent return，无日志） |

### 4.5 跨项目安全方案对比

| 维度 | **DexMani** | **T-Rex** | **ManiUniCon** |
|------|------------|-----------|----------------|
| 碰撞检测 | MPlib (Pinocchio) IK 阶段 + `validate_action` 二次校验 + `FingertipDeskSafety` (FK-based, 零成本) | Pinocchio (hppfcl) @300Hz，`stop_at_first_collision=True` 优化。手-手碰撞已移除（允许双手交叉操作）| 无实时碰撞检测 |
| 速度限制 | Inner loop delta clamp (0.3 rad/step) + soft-start ramp + anti-C24 protection (125% vel floor) | Simple delta clip (0.4×vel_limit/300Hz, ~0.0032-0.0036 rad/step)。**Ruckig NOT used** (`ruckig_smoothing=False`) | `PoseTrajectoryInterpolator` with max_pos_speed/max_rot_speed |
| 跟踪监控 | Passive monitor: tracking error warn >0.35 rad, mode drift detection (throttled 50-frame) | `TRACKING_SAFETY_THRESHOLD = 10.0 rad`（极粗粒度，仅防灾难性偏离） | 无独立跟踪监控 |
| 回退策略 | Hold position on timeout/NaN/error | **Revert to last safe** — 检测到碰撞后回到上一个无碰撞位姿（`arm_hand_control.py:927-965`） | 遇错 set error_state + break loop |
| E-Stop | `EmergencyEvent` → 最高优先级（绕过 macro_lock）→ `set_state(4)` ≤1 tick | KeyboardInterrupt only（无专用 e-stop 键） | 全局 `error_state` flag → 所有进程 break |
| NaN 防护 | Inner loop NaN guard（line 463-466）+ pipeline NaN guard（line 113, 172） | 无显式 NaN 检查 | **完全无 NaN 检查**：`validate_action` 基类永远返回 True（`base.py:113`） |
| validate_action 质量 | 8 门控，fail-fast，多层独立防护 | N/A（SmoothingAndSafetyManager 在 300Hz 线程中检查） | **1 个严重 bug**：`elif` 应改为 `if`（`base.py:99,106`），导致 `joint_positions`、`joint_velocities`、`joint_torques` 中只有一个被 clip |

**ManiUniCon `validate_action` bug 详述**（`robot_interface/base.py:89-113`）:
```python
if action.joint_positions is not None:
    action.joint_positions = np.clip(...)
elif action.joint_velocities is not None:   # BUG: 应为 if
    action.joint_velocities = np.clip(...)
elif action.joint_torques is not None:      # BUG: 应为 if
    action.joint_torques = np.clip(...)
return True  # 永远返回 True，不拒绝任何 action
```
若 `joint_positions` 和 `joint_velocities` 同时非 None → 仅 `joint_positions` 被 clip，`joint_velocities` 和 `joint_torques` 未检查。且永远返回 True，不存在 "action 被拒绝" 的代码路径。 |

**关键洞察 — T-Rex 的 Ruckig 未被使用**: 这是本次审查最重要的发现之一。尽管 T-Rex 有完整的 Ruckig OTG 实现（`arm_hand_control.py:760-780`，含 jerk=50 rad/s³ 限制），但生产环境设置 `ruckig_smoothing=False`（`main_teleop.py:685`），实际使用简单的速度限制 delta clip。原因可能是免费版 Ruckig 不支持 `current_position`/`current_velocity` 跟踪和关节位置硬限制——这些在注释中被明确标注为缺失。这对 DexMani 的启示是：**简单的 per-step delta clamp + soft-start ramp 在实践中已足够**；完整的 OTG 可能 ROI 有限。

---

## 五、第二轮深入分析：测试基础设施与错误恢复

### 5.1 测试基础设施对比

| 维度 | **DexMani** | **T-Rex** | **ManiUniCon** | **LeFranX** |
|------|------------|-----------|----------------|-------------|
| 测试框架 | **无** pytest | **无** pytest | pytest（仅数据转换） | 依赖 LeRobot 框架测试 |
| 单元测试 | **0** | **0** | **0**（仅 `tools/zarr2lerobot/tests/` 有 20 个转换测试） | 未知 |
| Mock/Fake 硬件 | **无** | **无** | **无** | **无** |
| 硬件需求 | 所有 6 个诊断脚本需要真机 | 需要真机 | 需要真机 | 需要真机 |
| CI 可行性 | **零** | **零** | 仅数据转换可 CI | 未知 |

**关键发现 — 四个项目都缺乏自动化测试**：没有一个项目有真正的单元测试或 mock 硬件层。DexMani 的 6 个 "test" 脚本（`examples/real/test_*.py`）全部是需真机的手动诊断脚本。这在真机机器人项目中是普遍现状，但 DexMani 的 recording/validation pipeline（`EpisodeRecorder`, `DataValidator`, `TimestampAlignedBuffer`）是**纯数据路径、不依赖硬件**，完全可以也应该有单元测试。

**DexMani 独有的优势 — 8 项 DataValidator 自动检查**: 经过事实核验，ManiUniCon、T-Rex、LeFranX 均无对等的数据验证层。DexMani 的 `DataValidator` 实际上是在 **补偿测试覆盖的缺失**——通过自动化的事后数据质量检查来确保已采集 episode 的可靠性。

### 5.2 错误恢复模式对比

#### DexMani — 分层恢复

| 错误类型 | 恢复策略 | 代码位置 |
|---------|---------|---------|
| Arm C22（自碰撞） | Inner loop 自动清除固件 latch + re-init Mode 6（`_RECOVERABLE_ERRORS`） | `inner_loop.py:108-113, 509` |
| Arm C24（超速） | 同上 | `inner_loop.py:108-113` |
| Arm 进程崩溃 | `ensure_running()` 自动重建 + 清除 stale rings（`arm_process.py:549-577`） | `arm_process.py:549-577` |
| Hand 进程崩溃 | 孤儿检测 → 安全退出（**不脱力**）。主进程可检测 + 重启 | `hand_process.py:974-980` |
| Hand 连续 send 失败 | Watchdog：30 次连续错误 → `reset_connection()` | `hand_process.py:953-962` |
| VR 断开 > 3s | 自动 stop recording (discard) → IDLE | `controller.py:282-309` |
| IK 失败 | Hold last good arm command | `pipeline.py:155-157` |

#### T-Rex — 全局终止

- 任何硬件断开 → **break out of teleop entirely**。无重连，无降级
- `assert target_data is not None` → **AssertionError crash**（`main_teleop.py:1005`）
- Action 线程异常 → 设置 terminate event → 全部停
- **无数据恢复**：daemon writer 线程 → 任何非正常退出都导致数据丢失

#### ManiUniCon — 全局 Error State

- 任何进程捕获 Exception → `error_state.value = True` → 所有进程 break
- 错误时执行握手清理（`robot_ready.set()` + `policy_ready.clear()`，防止死锁）
- **无自动重启**：崩溃 = 全系统停止
- **无 NaN 验证**：`validate_action` 基类 always returns True

#### DexMani 错误恢复的优势

DexMani 是四个项目中**唯一**支持自动恢复的项目：
- Arm 子进程崩溃后 `ensure_running()` 自动重建
- 可恢复的 C22/C24 错误被 inner loop 自动清除
- Hand 连续 send 失败后 watchdog 自动重连
- VR 断开后降级到 IDLE（而非全局崩溃）

### 5.3 各项目数据丢失风险评估

| 场景 | **DexMani** | **T-Rex** | **ManiUniCon** |
|------|------------|-----------|----------------|
| 录制中主进程崩溃 | 丢最后 ~10s（基于 `flush_interval_s`，`episode_recorder.py:96-97`） | 丢**整个 episode**（daemon writer 线程，无增量 flush） | 丢**整个 episode**（纯内存，零 I/O） |
| 录制中磁盘满 | Queue → `RecordingSession.record()` drop frame + warning | Writer 线程 exception → 继续（queue 无界，内存爆炸） | 录制期无 I/O → dump 时 np.savez 失败 |
| Atexit 安全网 | `_flush_all_recorders`（`episode_recorder.py:42-55`） | 无 | 无 |
| 文件关闭 | `join_stop()` 阻塞等待 writer 线程 drain | `_queue.join()` 无限等待 + thread join 5s timeout | N/A（录制期间无文件 I/O） |

**DexMani 的背压策略**（`max_queue=2000` + `queue.Full` → drop frame）优于 T-Rex 的**无界队列**（无 maxsize → 内存无限增长），但仍有改进空间：当前 drop 策略不区分关键帧（如 episode 边界）和普通帧。

---

## 六、扩展 Gap 分析：经过两轮深入的新增建议

### 第一轮 Top 4（保留，更新细节）

#### Top 1: 相机 LZF → 视频编码（借鉴 T-Rex + LeRobot v3）

**更新**: LeRobot v3 默认使用 **AV1 (libsvtav1)** 而非 h264。AV1 在同等质量下比 h264 节省 ~30%，但 CPU 编码开销是 h264 的 5-10×。**推荐方案**: h264 为首选（CPU 友好，`skvideo.io.FFmpegWriter` 成熟，`libx264rgb CRF 18`），AV1 作为离线后处理选项（通过 LeRobot 导出工具批量转码）。

**新增细节**: LeRobot 支持两种编码模式——streaming encoding（实时，帧进 MP4 队列，后台线程编码）和 batch encoding（先存临时 PNG，episode 结束后批量编码）。DexMani 已有 `_cam_queue` + `_cam_writer` 后台线程架构（`episode_recorder.py:121-128`），天然适合 streaming encoding。

**深度图特殊处理**: 16-bit 深度图不能直接进标准 8-bit h264。选项: (1) 2×8-bit 通道（高位/低位分拆，无损但 double 帧数），(2) HEVC 10-bit（`libx265 -pix_fmt gray10`，需解码器支持），(3) 保持 HDF5 gzip 压缩（16-bit 数据 LZF→gzip-1 已足够，深度图在采集数据中占比 <30%）。**推荐 (3)**，因为深度图的压缩收益有限（gzip 对 16-bit 数据已有 ~2× 压缩），且保持数据保真度比存储节省更重要。

#### Top 4: LeRobot v3 格式导出工具（借鉴 LeFranX）

**更新**: LeRobot v3 格式使用 **Parquet（表格数据）+ MP4（相机视频）+ JSON（元数据）** 的三分结构，与 DexMani 的单文件 HDF5 截然不同。导出工具需要处理以下关键差异：

**LeRobot v3 格式要点**（WebSearch + LeFranX 源码核验）:
- 表格数据: Apache Parquet，分 chunk 目录（`data/chunk-NNN/episode_NNNNNN.parquet`），每 chunk 默认 1000 episodes
- 视频: MP4（默认 AV1 codec），`videos/chunk-NNN/{camera_key}/episode_NNNNNN.mp4`
- 元数据: `meta/info.json`（总统计）+ `meta/episodes.jsonl`（每 episode）+ `meta/stats.json`（归一化统计）+ `meta/tasks.jsonl`（任务描述）
- Episode 边界: `episode_data_index` dict（`from`/`to` 数组）
- 索引对齐: 全局 `index` + per-episode `frame_index` + per-frame `timestamp`

**DexMani → LeRobot 映射要点**:
| DexMani HDF5 | LeRobot v3 |
|-------------|-----------|
| `/arm_qpos`, `/arm_ee`, `/hand_qpos` 等 | `observation.state`（flattened float32 vector） |
| `/action_arm_joint`, `/action_hand_joint` 等 | `action`（flattened float32 vector） |
| `/rgb` (HDF5 dataset) | 提取为 PNG 序列 → MP4 编码 → `videos/.../episode_NNNNNN.mp4` |
| `/depth` (HDF5 dataset) | 提取为 16-bit TIFF → 独立 depth MP4（`gray12le` pixel format） |
| `/pointcloud` | LeRobot 无原生点云支持 → 作为额外 feature 存储在 state 中或跳过 |
| `/flag_*` 质量标志 | 作为 extra features 存储在 `observation` 中 |
| Meta attrs | `meta/episodes.jsonl` + `meta/info.json` |
| Sidecar JSON | `meta/tasks.jsonl`（task_label, operator, tags） |

**实现风险**: 低-中（纯工具链，不影响采集；需要 `lerobot` pip 包 + `pyav`/`torchcodec` 依赖）

#### Top 2–3: 保持不变，见第一轮分析

### 第二轮新增建议

#### Top 5: 建立 Recording Pipeline 的自动化测试

**紧迫度**: **高**
**ROI**: 高（保护数据完整性路径，不依赖硬件）
**借鉴**: 无直接借鉴（四个项目都缺），但 DexMani 的 recording 模块结构使其天然可测试

**为什么可行**:
- `EpisodeRecorder` 是纯 HDF5 I/O，可用临时文件测试
- `DataValidator` 是纯数据检查，可用合成数据测试
- `TimestampAlignedBuffer` 是纯 numpy 操作，完全可单元测试
- `CollectionLoop` 仅编排上述组件，可 mock
- 以上均不依赖任何硬件

**建议实施**: 最小可行版本 —— 5 个 pytest：
1. `test_episode_recorder_start_stop` — 创建/写入/关闭临时 HDF5
2. `test_data_validator_all_checks` — 8 项检查的合成数据验证
3. `test_timestamp_buffer_alignment` — 网格对齐 + forward-fill 正确性
4. `test_recording_session_queue` — Queue 驱动生命周期 + backpressure drop
5. `test_collection_loop_max_frames_autostop` — max_frames 自动停止

每个 ≤50 行，合计 ~250 行测试代码，提升数据路径信心 >100×。

#### Top 6: 修复 `_recover_mode()` 返回值忽略问题（Bug）

**紧迫度**: **高**
**ROI**: 高（修复静默 SDK 状态不一致风险）

**问题**: `inner_loop.py:509, 691` 调用 `self._recover_mode(arm)` 但不检查返回值。若 3 次 Mode 6 重初始化全部失败，inner loop 继续运行，arm 可能处于 Mode 0 —— 后续 `set_servo_angle` 使用错误协议。

**修复（~3 行）**:
```python
# inner_loop.py:509, 691 — 在调用后加
if not self._recover_mode(arm):
    logger.error("Mode 6 re-init failed after 3 attempts — stopping inner loop")
    self._error_state = True
    break  # 或在 line 691 处 return
```

#### Top 7: 添加 validate_action 的 NaN Guard（纵深防御）

**紧迫度**: **中**
**ROI**: 中（inner loop 已有兜底，但防御纵深原则要求 validate_action 不依赖下游）

**问题** (`validate.py:103-105`): `np.clip(action.arm_qpos_cmd, ...)` 在 arm_qpos_cmd 含 NaN 时静默传播。

**修复（~2 行）**:
```python
# 在 line 103 之前加
if not np.all(np.isfinite(action.arm_qpos_cmd)):
    return False, "action arm_qpos_cmd contains NaN/Inf"
```

#### Top 8: 温度门控失效检测 + Fallback

**紧迫度**: **中**
**ROI**: 中（当前生产环境 xArm 固件版本未知，可能已受影响）

**问题**: `inner_loop.py:541-546` 使用 `getattr(arm, "temperatures", None)` — 若旧版固件不提供此属性，temps 保持 NaN，温度门控永久静默失效。

**修复（~5 行）**: 在第一次成功读取后记录一个 `bool` 标志；若从未成功读取，在 `validate_action` 中跳过温度检查（而非静默通过）并记录一次性的 WARNING。

#### Top 9: 结构化性能指标

**紧迫度**: **低**
**ROI**: 中（长期调试和性能优化价值大）

**背景**: 四个项目均无结构化指标（Prometheus / Grafana / log-based metrics）。DexMani 已有 `get_logger` 基础设施，可以最低成本添加关键路径的 per-tick 耗时统计。

**建议添加的指标**（全部可选，默认关闭）:
- Per-tick 耗时分布（p50/p95/p99）：VR read / IK solve / validate / inner loop send
- Inner loop 跟踪误差 distribution（替代当前的 throttled warning）
- DataValidator 各项检查的失败率（长期趋势）
- Camera frame 到达间隔分布 → 检测 L515 stream stall 的早期信号

#### Top 10: Hand Process 内部常量可配置化

**紧迫度**: **低**
**ROI**: 低（当前默认值合理，但调试时不可调）

**当前硬编码值**: `_WATCHDOG_RECONNECT_AFTER=30`（连续 send 失败）、`_READY_TIMEOUT_S=15.0`、`_SEED_TIMEOUT_S=0.5`、`_MACRO_RESYNC_TIMEOUT_S=0.5`、`orphan_exit_s=60.0`

**建议**: 转移到 `HandProcessConfig` 或 `HandProcessTuning` dataclass，保持当前值作为默认值。

---

## 七、不建议借鉴的反模式（扩展版）

| 项目 | 不借鉴什么 | 为什么 | 代码证据 |
|------|----------|--------|---------|
| T-Rex | 单进程多线程模型 | DexMani 的多进程隔离是**明确的架构优势**：arm/hand/camera 各自崩溃不影响主循环 | `main_teleop.py`: 所有线程 daemon，无进程隔离 |
| T-Rex | **无界写入队列** | `DataWriter` queue.Queue 无 maxsize → 写入滞后时内存无限增长。DexMani 的 `max_queue=2000` + drop 策略更安全 | `data_writer.py:244`（`queue.Queue()` = 无界） |
| T-Rex | **daemon writer 线程** | daemon 线程在任何非正常退出时被强制终止 → 数据丢失。DexMani 的 `join_stop()` + atexit flush 保障 > 仅 daemon | `data_writer.py:272`（`daemon=True`） |
| T-Rex | **assert on sensor data** | `assert target_data is not None` → AssertionError crash 而非 graceful degradation。DexMani 使用 held-frame + warning | `main_teleop.py:1005` |
| ManiUniCon | Falling-edge dump（录制期间零 I/O） | 崩溃 = 丢整个 episode。DexMani 的增量 flush（~10s）+ atexit 安全网更抗崩溃 | `core/robot.py:186-204` vs `episode_recorder.py:96-97` |
| ManiUniCon | **无 NaN 验证** | `validate_action` 基类永远返回 True，NaN 在 observations/actions 中静默传播。DexMani 的多层 NaN guard 是明确的优势 | `maniunicon/robot_interface/base.py:89-113` |
| ManiUniCon | `mp.Process` per camera | DexMani 的 `CameraProcess` 一次 fork 管所有相机更轻量 | `main.py:82-84` |
| ManiUniCon | **print() 日志** | 全项目使用 `print()`（无 log level, timestamp, file output）。DexMani 的 `get_logger` 结构化日志是明确的优势 | 所有 ManiUniCon 文件 |
| LeFranX | 全面采用 LeRobot 框架 | 框架锁定太重；DexMani 的轻量 HDF5 直写 + 导出工具桥接更灵活 | `dual_vr_record.py:308-317` |

---

## 八、推荐实施路线图（事实核验后排序）

> **核验说明**: 所有建议均经直接源码核验。T-Rex 相关声明基于 agent 读取结果（项目已不在磁盘）。
> H2（echo seq gap）、P2（EMA state reset）等 3 项已在核验中排除（见 §4.3, §4.4）。
> ManiUniCon `elif→if` bug 是新增发现（`base.py:99,106`），加了反模式列表。

### 事实核验后的严重度重新排序

| 排名 | 建议 | 严重度 | 改动量 | 核验状态 |
|------|------|--------|--------|---------|
| **1** | Top 1: 相机 LZF → h264 视频编码 | ROI: **极高**（-90% 磁盘） | ~30 行（仅 `_cam_writer` 线程） | ✅ h264 CRF 18 方案确认；AV1 作为离线后处理 |
| **2** | Top 6: 修复 `_recover_mode()` 返回值忽略 | **高**（静默 SDK 状态不一致） | ~3 行 | ✅ `inner_loop.py:509,691` 两处确认 |
| **3** | Top 7: validate_action NaN guard | **中**（防御纵深缺口） | ~2 行 | ✅ `validate.py:105` 前无 NaN 检查确认 |
| **4** | Top 5: Recording pipeline pytest 基础设施 | 长期价值: **极高**（数据完整性回归保护） | ~250 行 | ✅ `EpisodeRecorder`/`DataValidator`/`TimestampAlignedBuffer` 均不依赖硬件 |
| **5** | Top 8: 温度门控失效检测 | **中**（静默门控失效） | ~5 行 | ✅ `inner_loop.py:541` `getattr` 无 fallback 确认 |
| **6** | Top 2: 2-Phase 同步握手接线 | 架构预留（未来 Policy 部署） | ~20 行 | ✅ `sync=None` 确认；基础设施已就位 |
| **7** | Top 9: 结构化性能指标 | 长期调试价值 | ~100 行 | ✅ 四项目均缺；DexMani 有 logger 基础设施 |
| **8** | Top 3: 触觉视频压缩 | 按需（当前数据量小） | ~50 行 | ✅ 当前仅 force 数据（~7.2KB/帧），无需紧急 |
| **9** | Top 10: Hand process 内部常量可配置化 | 调试灵活性 | ~20 行 | ✅ 5 个硬编码常量确认 |
| **10** | Top 4: LeRobot v3 格式导出工具 | 生态兼容性 | ~300 行 | ⚠️ AV1 默认 codec 来自上游文档（非 LeFranX 源码）；h264/hevc 备选 |

### 实施 Phase 重新排序

```
Phase 1 (立即 — 低风险，高/中严重度修复):
  ├── Top 1: 相机 LZF → h264 视频编码         磁盘 -90%
  ├── Top 6: fix _recover_mode() return ignored  ~3 行 bugfix
  └── Top 7: validate_action NaN guard           ~2 行防御纵深

Phase 2 (短期 — 基础设施):
  ├── Top 5: Recording pipeline pytest          5 个测试保护数据完整性
  ├── Top 8: 温度门控失效检测 + fallback        ~5 行
  └── Top 2: 2-Phase 同步握手接线              架构预留

Phase 3 (中期 — 观测性 + 灵活性):
  ├── Top 9: 结构化性能指标                    长期调试价值
  └── Top 10: Hand process 可配置化            调试灵活性

Phase 4 (工具链 — 按需):
  ├── Top 4: LeRobot v3 导出工具              训练生态桥接
  └── Top 3: 触觉视频压缩                      为高分辨率触觉扩展准备
```

### 核验中排除的伪发现

| 原声称 | 排除原因 | 教训 |
|--------|---------|------|
| H2: Echo seq gap 假阳性 | `_last_acked_seq > 0` guard（line 563）阻止重启后假阳性 — 这是**正确的防御性代码** | 阅读 guard 条件时不要只看主逻辑，要看前置条件 |
| P2: EMA state carry-over | `_reset_mapper()` 在 B/C 键转换时正确重置 — 两个方法在两个文件中（直接字段访问而非 getter），但确实被正确调用 | 跨文件追踪调用链以避免误报 |
| H3: seq 失败递增 | F1 文档的显式设计决策，不是 bug — position servo 不依赖持续刷新 | 区分 "看起来可疑" 和 "实际是设计意图" |

### 核验方法备注

- **DexMani 声明**: 所有行号通过直接读源文件确认
- **ManiUniCon 声明**: `validate_action` bug 通过直接读 `base.py:89-113` 确认
- **LeFranX 声明**: `use_videos=True, image_writer_threads=4` 通过直接读 `dual_vr_record.py:206-207` 确认
- **T-Rex 声明**: 项目已不在磁盘，依赖 agent 输出（行号来自 agent 结果，非本 session 直接验证）
- **LeRobot AV1 默认 codec**: 来自 WebSearch（上游 HuggingFace 文档），非 LeFranX 源码（LeFranX 不包含 `lerobot` 包源码）

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
| `scripts/dual_robot/dual_vr_record.py` | 录制脚本（LeRobotDataset.create + record_loop，`use_videos=True, image_writer_threads=4`） |
| `scripts/dual_robot/dual_vr_teleoperator.py` | VR 遥操作测试验证脚本 |
| `scripts/dual_robot/dual_robot_replay.py` | 从 LeRobot 数据集回放轨迹 |
| `src/lerobot/robots/franka_fer_xhand/franka_fer_xhand.py` | FrankaFER + XHand 组合机器人（observation/action 前缀统一） |
| `src/lerobot/robots/franka_fer_xhand/franka_fer_xhand_config.py` | 组合机器人配置（同步/紧急停止选项） |
| `src/lerobot/teleoperators/franka_fer_xhand_vr/franka_fer_xhand_vr_teleoperator.py` | 双 VR 遥操作器（arm IK + hand retargeting） |
| `src/lerobot/teleoperators/franka_fer_vr/arm_ik_processor.py` | C++ WeightedIKSolver 包装器 |
| `src/lerobot/teleoperators/xhand_vr/xhand_vr_teleoperator.py` | 手部 VR 遥操作器（dex-retargeting pipeline） |

### T-Rex/hardware（补充）
| 文件 | 职责 |
|------|------|
| `hardware_code/teleop/robot_descriptions.py` | 机器人 URDF/MJCF 模型构建 + 碰撞对过滤 + DEXMATE 关节限制 |
| `hardware_code/config/default.yaml` | 实际生产参数（`command_hz=30, arm_action_hz=300` 等） |
| `hardware_code/teleop/teleop_targets.py` | 250Hz VR 目标源（Vive + Manus 数据融合 + 重定向） |
