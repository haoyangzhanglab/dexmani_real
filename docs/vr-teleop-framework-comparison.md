# VR 遥操作框架对比分析

> **日期**: 2026-06-22 | **对比版本**: dexmani_real (main), LeFranX (Reference), BunnyVision Pro (Reference), Open-Teach (Reference), ManiUniCon (Reference)

---

## 1. Executive Summary

### 1.1 总览对比

| 维度 | dexmani_real | LeFranX | BunnyVision Pro | Open-Teach | ManiUniCon |
|------|-------------|---------|-----------------|------------|------------|
| **架构模式** | 单线程 50Hz 顺序循环 | Python + C++ pybind11 多线程 | 三机 ZMQ 部署 (VP→Docker→Client) | 多进程 ZMQ PUB/SUB 管线 | 多进程 + 共享内存 (lock-free) |
| **编程语言** | Python 3 | Python + C++ (pybind11) | Python (client) + Docker (server) | Python + C# (Unity VR) | Python 3 |
| **VR 输入** | Meta Quest (HTS SDK) | Meta Quest (raw TCP) | Apple Vision Pro (avp_stream) | Meta Quest (Unity + NetMQ) | Meta Quest (oculus_reader) |
| **VR 帧率** | 50 Hz (HTS 原生) | ~60 Hz (TCP) | 60 Hz (Vision Pro) | 60 Hz (Unity OVR) | 30 Hz (策略层) → 200 Hz (插值) |
| **手臂 IK** | DLS + MPlib Position IK 回退 | 解析 IK + Brent q7 优化 (C++) | DLS (Pinocchio, server) | 机器人原生 Cartesian servo | Pink QP (Pinocchio, FrameTask+PostureTask) |
| **IK 后端** | Pinocchio + MPlib | C++ 自定义 (geofik) | Pinocchio | 无自定义 IK | Pinocchio |
| **手部重定向** | DexPilot + XHandRefAdapter | DexPilot + adaptive pinky | SeqRetargeting (server-side) | KDL IK + 角度计算 | N/A (Robotiq 二指爪) |
| **通信协议** | 直接 Python API 调用 | TCP (VR raw + 控制) | ZMQ PUB/SUB + REQ/REP | ZMQ PUB/SUB + PUSH/PULL | Lock-Free SharedMemory (RingBuffer + Queue) |
| **安全机制** | ★★★★★ 四层 + 10bit 质量标记 | ★★☆☆☆ 基础位置限制 | ★★☆☆☆ 速度裁剪 + 初始化降速 | ★★☆☆☆ 暂停/恢复 + 分辨率缩放 | ★★★★☆ 三层 (workspace/VR delta/validate_action) |
| **数据录制** | HDF5 (episode + quality flags) | HuggingFace Dataset (LeRobot) | HDF5 + NPY | HDF5 + AVI + Pickle | NPZ → Zarr → LeRobot v3.0 |
| **双手支持** | 右手 only | 双手 (dual config) | 双手 (native bimanual) | 双手 (bimanual_right) | 右手 only |
| **进程模型** | 单进程 + 单线程 | 多进程 (franka_server) + 多线程 | 三机分布式 + 多线程 | 多进程 (ZMQ nodes) | 多进程 (相机/策略/机器人独立) |
| **成熟度评分** | ★★★★☆ (4/5) | ★★★★★ (5/5) | ★★★★☆ (4/5) | ★★★★☆ (4/5) | ★★★★☆ (4/5) |
| **机器人支持** | XArm7 + XHand | Franka FER + XHand | XArm7 + Ability Hand (双手) | Franka/Kinova/XArm + Allegro | XArm6/UR5/Franka + Robotiq |

### 1.2 Top 5 可采纳改进 — 全部已落地 ✅

> **核心结论**: dexmani 在**安全性**上远超五个参考框架。Phase 1-4 已将 Top 5 改进全部实施完毕，Phase 5 (2026-06-23) 完成最后的集成收尾。

| # | 改进项 | 来源 | 优先级 | 状态 |
|---|--------|------|--------|------|
| 1 | 遥操作 IK 加入可操作性评分 | LeFranX | **P0** | ✅ Phase 2.6 |
| 2 | VR 跟踪丢失时软减速保持 | BVPro | **P0** | ✅ Phase 1.4 |
| 3 | Cartesian Pose 插值（频率解耦） | ManiUniCon | **P1** | ✅ Phase 2.2 |
| 4 | ZMQ 进程分离（VR 解耦控制） | Open-Teach | **P1** | ✅ Phase 3.1 |
| 5 | VR per-step delta 旋转安全限制 + EEF 方向工作空间边界 | ManiUniCon | **P1** | ✅ Phase 1.2 + 1.3 |

### 1.3 Phase 5 新增能力 (2026-06-23)

| 能力 | 说明 |
|------|------|
| CollectionLoop 集成 | 统一录制生命周期管理，auto_stop_on_quality_drop |
| Episode sidecar JSON | stop_episode 时自动写入 metadata 到 `episode_NNN.json` |
| 多相机控制回路 | MultiCameraManager 集成到 TeleopController._tick() |
| 多相机 HDF5 写入 | `/camera/<serial>/rgb` + `/camera/<serial>/depth` per-camera paths |
|---|--------|------|--------|----------|
| 1 | 遥操作 IK 加入可操作性评分（`kin.compute_manipulability` 已实现但未在 teleop 热点使用） | LeFranX | **P0** | IK 鲁棒性 +20%，奇点规避更平滑 |
| 2 | VR 跟踪丢失时软减速保持（替代立即 hold） | BVPro | **P0** | 消除跟踪短暂丢失时的急停抖动 |
| 3 | Cartesian Pose 插值（频率解耦，消除 stale reuse） | ManiUniCon | **P1** | 消除 VR 帧重读抖动，200Hz 平滑控制 |
| 4 | ZMQ 进程分离（VR 解耦控制） | Open-Teach | **P1** | 消除 GIL 瓶颈，VR 解析可独立 60Hz |
| 5 | VR per-step delta 旋转安全限制 + EEF 方向工作空间边界 | ManiUniCon | **P1** | 防止 VR 跟踪跳变 + wrist 极值自碰撞 |

---

## 2. 架构总览

### 2.1 LeFranX — Python + C++ pybind11 双语言架构

```
┌─────────────────────────────────────────────────────────────────┐
│ Meta Quest VR App                                               │
│   raw TCP (port 8000), ADB reverse tunnel                       │
│   "Right wrist: x,y,z,qx,qy,qz,qw" + "Right landmarks: x1..z21"│
└───────────────────────────┬─────────────────────────────────────┘
                            │ TCP string stream
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ VRMessageRouter (C++ thread, pybind11)                          │
│   tcp_receiver_thread() ──► parse_vr_messages() (regex)         │
│   → current_messages_ (mutex-protected)                         │
│ File: franka_xhand_teleoperator/src/vr_message_router.cpp       │
│   :205 tcp_receiver_thread, :262 parse_vr_messages              │
└───────────────────────────┬─────────────────────────────────────┘
                            │ pybind11 get_messages()
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ VRRouterManager (Python singleton, thread-safe via Lock)         │
│   get_vr_data() → wrist + landmarks                             │
│ File: src/lerobot/teleoperators/vr_router_manager.py :188       │
└──────────┬──────────────────────────────┬───────────────────────┘
           │ wrist data                   │ landmarks
           ▼                              ▼
┌──────────────────────────┐  ┌───────────────────────────────────┐
│ ArmIKProcessor (Python)  │  │ VRHandDetectorAdapter (Python)    │
│   _compute_target_pose()  │  │   SVD frame estimation            │
│   coordinate transform:   │  │   OPERATOR2MANO_RIGHT             │
│     VR(rfu)→Robot(base)   │  │   adaptive pinky scaling (1.2-2.2x)│
│   differential delta rel. │  │   → ref_value                     │
│   workspace clamp@0.75m   │  └───────────────┬───────────────────┘
│ File: arm_ik_processor.py │                  │
│   :193 pos transform      │                  ▼
│   :269 quat transform     │  ┌───────────────────────────────────┐
│   :230 workspace clamp    │  │ dex-retargeting (Python)          │
│                           │  │   retarget(ref_value)             │
│                           │  │   → robot joint angles (16-DOF)   │
│                           │  │   → exponential smoothing         │
│                           │  │   → _map_to_xhand_order()         │
│                           │  └───────────────┬───────────────────┘
│                           │                  │
│                           ▼                  ▼
│  ┌──────────────────────────────────────────────────────────┐
│  │ WeightedIKSolver (C++/pybind11)                          │
│  │   solve_q7_optimized():                                  │
│  │   1. Brent's 1D optimization over q7 (elbow)             │
│  │   2. geofik.cpp → franka_J_ik_q7() → 0-8 analytic solns │
│  │   3. Score = w_manip*manip - w_neutral*neutral_dist      │
│  │              - w_current*current_dist                    │
│  │   manipulability = sqrt(det(J*J^T))  (Yoshikawa)        │
│  │   File: franka_xhand_teleoperator/src/weighted_ik.cpp    │
│  │     :19 manipulability, :71 scoring, :143 Brent opt      │
│  └──────────────────────────┬───────────────────────────────┘
│                              │ optimal 7-DOF qpos
│                              ▼
│  ┌──────────────────────────────────────────────────────────┐
│  │ FrankaFER.send_action()                                  │
│  │   TCP socket (port 5000) → "SET_POSITION p0..p6"         │
│  │ File: src/lerobot/robots/franka_fer/franka_fer.py :230   │
│  └──────────────────────────┬───────────────────────────────┘
│                              │
└──────────────────────────────┼───────────────────────────────────
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ franka_server (C++ 独立进程, RT)                                 │
│   Ruckig trajectory smoothing @ 1kHz                            │
│   500ms 命令超时 → hold in place                                │
│   File: franka_server/src/franka_server.cpp :411 control loop    │
└─────────────────────────────────────────────────────────────────┘
```

**关键设计模式**:
- **双语言热/冷路径分离**: C++ 处理实时热点（IK、VR 路由），Python 处理非实时冷路径（retargeting、生命周期）
- **多目标加权 IK**: 可操作性 × 中性位姿 × 当前距离，Brent 1D 优化
- **TCP 文本协议**: 简单可读，适合调试，但协议解析开销高于二进制

### 2.2 BunnyVision Pro — 三机 ZMQ 部署架构

```
┌──────────────────────────────────────────────────────────────────┐
│ Tier 1: Apple Vision Pro                                        │
│   Tracking Streamer app (avp_stream)                            │
│   → left_wrist/right_wrist (4×4 matrices)                       │
│   → left_fingers/right_fingers (25×4×4 matrices)                │
│   @ 60 Hz                                                        │
└────────────────────────────┬─────────────────────────────────────┘
                             │ local network IPv4
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ Tier 2: Docker Server (bunny_teleop_server)                      │
│                                                                   │
│  ┌─ run_vision_server (terminal 1) ─┐                            │
│  │  avp_stream receive              │                            │
│  │  coordinate alignment:            │                            │
│  │    hand_frame (avg 200 poses)     │                            │
│  │    → relative human hand delta    │                            │
│  │    → bimanual alignment mode      │                            │
│  │      (CENTER/LEFT/RIGHT/SEPARATELY) │                          │
│  └──────────────┬────────────────────┘                            │
│                 │                                                 │
│  ┌─ run_robot_server (terminal 2) ─┐                              │
│  │  Arm IK (DLS, Pinocchio):        │                             │
│  │    v = J^T·(J·J^T+λ²I)⁻¹·err    │                             │
│  │    λ²=1e-5, max 100 iters        │                             │
│  │  Hand retargeting (SeqRetarget): │                             │
│  │    OPERATOR2MANO → dex_retarget  │                             │
│  │  Output → joint qpos [0:7 arm,   │                             │
│  │                        7:17 hand]│                             │
│  └──────────────┬────────────────────┘                            │
│                 │                                                 │
│         ┌───────┴────────┐                                       │
│         ▼                ▼                                       │
│  ZMQ PUB (5500)   ZMQ REP (5501)                                  │
│  streaming cmds   init config                                     │
└─────────┼────────────────┼───────────────────────────────────────┘
          │ pickle dict     │ JSON config
          ▼                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ Tier 3: Teleop Client (this repo, Python)                        │
│                                                                   │
│  TeleopClient (ZMQ SUB + REQ)                                    │
│    update_teleop_cmd() → target_qpos (17,) × 2                   │
│  File: bunny_teleop/bimanual_teleop_client.py :74-95             │
│                                                                   │
│  XArm7Ability                                                      │
│    control_arm_qpos() → sets _arm_pos_target (:196)                 │
│    _internal_control_arm_qpos() → PID velocity (250Hz inner, :200) │
│    clip_arm_velocity() → max_arm_velocity scaling                  │
│      max_vel = [0.8,0.8,0.8,0.8,1.0,1.0,1.5] rad/s                │
│    compute_ik() → DLS for test motions (not in hot path, :136)     │
│  File: real_control/xarm7_ability.py :136, :196, :200              │
│                                                                   │
│  Main Control Loop (含 data recording) → HDF5 + NPY               │
│    teleop_data/<TASK>/robot_data_raw/episode_N/data.h5            │
│  File: real_control/teleop_bimanual_xarm7_ability.py :198-251    │
└──────────────────────────────────────────────────────────────────┘
```

**关键设计模式**:
- **三机责任分离**: Vision Pro (感知) → Docker Server (计算) → Client (执行)，各方独立升级
- **Server-side IK**: 计算密集的 IK 和 retargeting 从实时控制客户端剥离
- **250Hz 内环控制**: PID 速度环在高频下运行，50Hz 外环仅更新目标

### 2.3 Open-Teach — 多进程 ZMQ PUB/SUB 管线

```
┌──────────────────────────────────────────────────────────────────┐
│ VR Tier: Meta Quest (Unity C# app)                               │
│   Oculus OVR + OVRSkeleton → 24 hand bones                      │
│   GestureDetector.SendHandData()                                 │
│     NetMQ PushSocket → "absolute:x,y,z|x,y,z|..."                │
│   VR/Franka-Bot-Unity/Assets/Scripts/                            │
│     Gesture Detection/GestureDetector.cs :157 SendHandData       │
│     NetworkManager.cs :36 TCP config                             │
└──────────────┬─────────────────────┬─────────────────┬────────────┘
               │ PUSH (8087)         │ PUSH (8095)     │ PUSH (8100)
               │ raw keypoints       │ resolution btn  │ pause/reset
               ▼                     ▼                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ Python Pipeline (multiprocessing.Process × N)                    │
│                                                                   │
│  Process 1: OculusVRHandDetector                                 │
│    ZMQ PULL (8087) → parse keypoints                             │
│    ZMQ PUB (8088) topic="right" → keypoint array (1-byte type + 72 floats) │
│  File: openteach/components/detector/oculus.py :74-104           │
│                                                                   │
│  Process 2: TransformHandPositionCoords                          │
│    ZMQ SUB (8088) topic="right" → 24×3 keypoints                 │
│    _get_coord_frame(): palm normal from index+pinky              │
│    transform_keypoints(): rotate to invariant palm frame         │
│    ZMQ PUB (8089) topics="transformed_hand_coords/frame"         │
│  File: openteach/components/detector/keypoint_transform.py :55   │
│                                                                   │
│  Process 3: FrankaArmOperator (or Kinova/Bimanual)               │
│    ZMQ SUB (8089) → hand frame (4×3)                             │
│    _apply_retargeted_angles():                                   │
│      H_HT_HI = pinv(H_HI_HH) @ H_HT_HH  (relative hand delta)   │
│      H_RT_RH = H_RI_RH @ H_A_R @ H_HT_HI @ pinv(H_A_R)          │
│      → scaled Cartesian pose                                     │
│    robot.arm_control(final_pose)                                 │
│  File: openteach/components/operators/franka.py :181-226         │
│                                                                   │
│  Process N: Visualizers, Camera Recorders, Sim Env               │
│    (ZMQ SUB from respective topics, independent processes)       │
└──────────────────────────────┬───────────────────────────────────┘
                               │ robot.arm_control(cartesian_pose)
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ Robot Native Controller                                           │
│   Franka: libfranka Cartesian servo (ROS)                        │
│   Kinova: Cartesian velocity control (ROS)                       │
│   XArm: Direct TCP Cartesian pose                                │
│   (NO custom IK — robot own controller handles IK)               │
└──────────────────────────────────────────────────────────────────┘
```

**关键设计模式**:
- **纯管道架构**: 每个组件是独立 `multiprocessing.Process`，通过 ZMQ topic 连接，天然支持热插拔
- **Hydra 配置驱动**: `defaults:` 组合 + `_target_` 实例化，添加新机器人只需 YAML + Python 类
- **无自定义 IK**: 完全依赖机器人原生 Cartesian servo → 优点是零 IK 维护成本，缺点是缺少可操作性优化和奇点规避

### 2.4 dexmani_real — 单线程 50Hz 顺序循环

```
┌──────────────────────────────────────────────────────────────────┐
│ Meta Quest (HTS app)                                             │
│   TCP (port 8000), HandFrame stream                              │
│   hand_tracking_sdk HTS Client                                   │
└────────────────────────────┬─────────────────────────────────────┘
                             │ TCP binary (HTS protocol)
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ QuestHandTracker (Python, daemon thread)                         │
│   _receive_loop() → iter_events() → convert_frame()             │
│   → latest_frame (dict, mutex-protected)                         │
│   Output: wrist_pos(3) + wrist_quat_wxyz(4) + landmarks(21,3)   │
│ File: dexmani_real/teleop/vr/vr_tracker.py :220 _receive_loop   │
└────────────────────────────┬─────────────────────────────────────┘
                             │ get_latest()
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ TeleopController (主线程, 50Hz)                                   │
│                                                                   │
│  _tick() 每帧流程:                                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 1. _read_vr_frame()               (~1ms)  VR 帧接收      │    │
│  │ 2. TrackingQuality.check()        (~0.1ms) 帧新鲜度闸门  │    │
│  │    max_frame_age=0.2s, lost_timeout=1.0s → E-Stop       │    │
│  │ 3. robot.get_state()              (~2ms)  读取机器人状态  │    │
│  │ 4. _compute_action()              (~5-15ms) 核心计算      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  _compute_action() 内部:                                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ a. ArmWristMapper.map()         (~0.2ms) reset-relative  │    │
│  │    wrist deltas → target EEF pose                        │    │
│  │ b. planner.solve_teleop_ik()    (~1-10ms) IK 求解       │    │
│  │    → TeleopIKSolver (DLS primary, MPlib fallback)       │    │
│  │ c. EMA smooth (arm only)        (~0.05ms)               │    │
│  │ d. Workspace check               (~0.1ms)  EEF bounds   │    │
│  │ e. XHandRetargeter.retarget()   (~2-5ms) 手部重定向    │    │
│  │    estimate_frame_from_hand_points → MANO → DexPilot    │    │
│  │    → XHandRefAdapter (pinky scaling)                    │    │
│  │ f. Joint jump clamp              (~0.05ms) 5°/frame     │    │
│  │ g. 10bit QualityFlags            (~0.05ms) quality gate │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 5. Safety checks on state        (~0.3ms)               │    │
│  │    ARM_TORQUE(7×), HAND_CURRENT, HAND_TEMP, HAND_COMM   │    │
│  │    arm_joint_limits → E-Stop                             │    │
│  │    hand_joint_limits → Warning                           │    │
│  │ 6. robot.send_action(action)      (~1ms)  发送机器人指令  │    │
│  │    → XArm7._limit_joint_step() bottleneck scaling        │    │
│  │    → XHand direct position command                       │    │
│  │ 7. recorder.add_frame()           (~0.5ms) 录制帧       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│ File: dexmani_real/teleop/core/controller.py :199 _tick          │
└──────────────────────────────────────────────────────────────────┘
```

**关键设计模式**:
- **四层安全模型**: 驱动层 (torque/clip) → 接口层 (workspace/safety checks) → 控制器层 (jump clamp/quality flags) → 路径层 (desk FK/collision)
- **Hold-on-failure**: 任何管道失败返回 last_good_position，不发送危险指令
- **双 IK 策略**: DLS (确定性, <1ms) → MPlib Position IK (随机, ~10ms) → hold
- **每帧 10bit 质量标记**: TRACKING, IK, RETARGET, JUMP, WORKSPACE, TORQUE, CURRENT, TEMP, COMM, RETARGET_VALID

### 2.5 ManiUniCon — 多进程 + 共享内存 + 频率解耦

```
┌──────────────────────────────────────────────────────────────────┐
│ ManiUniCon — 多进程 + 共享内存 + 频率解耦                        │
│                                                                   │
│  main.py (RobotControlSystem)                                     │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │ Sensors  │  │ Policy       │  │ Robot (mp.Process)       │   │
│  │(Process) │  │ (Process)    │  │                          │   │
│  │          │  │              │  │  State Recv Thread (30Hz)│   │
│  │Realsense │  │ Quest/       │  │    ↓ write_state()       │   │
│  │(per cam) │  │ SpaceMouse/  │  │                          │   │
│  │          │  │ Keyboard     │  │  Main Thread (200Hz)     │   │
│  │          │  │  30Hz        │  │    read_all_action()     │   │
│  └────┬─────┘  └──────┬───────┘  │    → Interpolator       │   │
│       │                │          │    → IKSolver (Pink)     │   │
│       └────────┬───────┘          │    → send_action()      │   │
│                │                  │                          │   │
│       ┌────────▼────────────┐     │  File: core/robot.py     │   │
│       │  SharedStorage      │     │    :279 run() loop       │   │
│       │  (Lock-Free IPC)    │     │    :174 state thread     │   │
│       │                     │     │    :405 interpolator     │   │
│       │ RingBuffer(state)   │◄────┘                          │   │
│       │ Queue(action)       │───► read_all_action()          │   │
│       │ RingBuffer(camera)  │                                │   │
│       │ flags + events      │                                │   │
│       │                     │                                │   │
│       │ File: shared_memory/│                                │   │
│       │   shared_storage.py │                                │   │
│       └─────────────────────┘                                │   │
└──────────────────────────────────────────────────────────────────┘
```

**关键设计模式**:
- **Lock-Free 共享内存**: 自定义原子计数器 RingBuffer(FILO) + Queue(FIFO)，微秒级 IPC，无 GIL 竞争
- **频率解耦**: Policy 30Hz → Interpolator 200Hz → Hardware 200Hz，VR 慢输入被插值填满
- **Cartesian 空间插值**: 线性位置 + SLERP 旋转，消除 stale reuse
- **QP-based IK**: Pink 库 (Pinocchio)，FrameTask + PostureTask 多目标优化，QP 求解器自然处理秩亏
- **VR 安全限制**: max_delta_pos=0.5m, max_delta_rot=1.0rad per-step，防止跟踪 glitch
- **Hydra 配置**: eval resolver 支持表达式，CLI override

---

## 3. 维度对比

### 3.1 VR 输入

| 特性 | dexmani_real | LeFranX | BunnyVision Pro | Open-Teach | ManiUniCon |
|------|-------------|---------|-----------------|------------|------------|
| **头显设备** | Meta Quest | Meta Quest | Apple Vision Pro | Meta Quest | Meta Quest |
| **SDK** | HTS (Python) | 自定义 TCP + regex | avp_stream (Python) | Unity C# + OVR Skeleton | oculus_reader (Python) |
| **传输协议** | TCP binary (HTS) | TCP text (regex) | Network (avp_stream) | TCP (NetMQ) | raw TCP |
| **手部数据** | 21 landmarks (3D) | 21 landmarks (string) | 25×4×4 matrices | 24 bones (3D) | N/A (delta-based control, no hand) |
| **输出帧** | FLU (front-left-up) | RFU (right-forward-up) | Vision Pro native | 绝对/相对模式 | Quest native → R_ve transform |
| **帧率** | 50 Hz (HTS 原生) | ~60 Hz | 60 Hz | 60 Hz | 30 Hz (policy) |
| **坐标帧转换** | HTS Python SDK 内置 | Python manual (pos + quat) | Server-side | Python SVD palm frame | R_ve matrix ([[0,0,1],[0,1,0],[-1,0,0]] @ -60deg Y) |
| **跟踪丢失检测** | ✅ TrackingQuality (0.2s stale / 1.0s lost) | ✅ 无 VR 数据 → hold | ❌ 无显式检测 | ✅ NOBLOCK 重试 10 次 | N/A (VR 数据丢失 → 策略返回当前 pose) |
| **丢失处理** | **E-Stop** (连续 1s+) | Hold position | Hold last command | Hold last command | Hold current pose |

**dexmani 当前位置与差距**:
- dexmani 使用 HTS SDK 是最完善的选择，提供类型安全的 Python API，比 LeFranX 的 regex 解析更可靠
- 坐标帧转换已内置在 SDK 中 (`unity_left_to_flu_position/rotation`)，无需手动维护转换矩阵
- ManiUniCon 的 R_ve 矩阵变换是简洁的硬编码方案，对单机器人部署足够但不如 HTS SDK 通用
- **差距**: 跟踪丢失后的处理过于激进 — 连续 1s 丢失直接触发 E-Stop。BVPro 的方式（保持在最后有效指令）更平滑

**建议**: 跟踪丢失时采用分级策略：<200ms hold → 200ms-1s soft deceleration → >1s E-Stop

### 3.2 臂部 IK

| 特性 | dexmani_real | LeFranX | BunnyVision Pro | Open-Teach | ManiUniCon |
|------|-------------|---------|-----------------|------------|------------|
| **方法** | DLS (primary) + Position IK (fallback) | Analytic IK + Brent q7 opt | DLS (Pinocchio) | 机器人原生 Cartesian servo | Pink QP (Pinocchio task-driven) |
| **确定性** | DLS 确定 / Position IK 随机 | **确定性** | 确定性 | 确定性 (robot) | 确定性 (QP solver) |
| **后端** | Pinocchio (Jacobian) + MPlib | C++ 自定义 geofik | Pinocchio | libfranka / ROS / TCP | Pink + Pinocchio |
| **碰撞检测** | ✅ self-collision + FK desk | ❌ 无 | ❌ 无 | ❌ 无 | N/A |
| **奇异性处理** | DLS damping + position IK 回退 | 可操作性评分 (自动避免) | DLS damping (固定 λ²=1e-5) | 依赖机器人控制器 | QP solver 自然处理秩亏 (damping=1e-12) |
| **可操作性感知** | ⚠️ `compute_manipulability()` 有实现但**未在 teleop 使用** | ✅ Yoshikawa 可操作性是评分首要项 | ❌ 无 | ❌ 无 | QP 隐式处理 |
| **中性位姿偏好** | ❌ 无 (仅 planning 时使用) | ✅ 加权 neutral_dist | ❌ 无 | ❌ 无 | PostureTask (可选, cost=1e-3) |
| **硬件最近偏好** | ✅ position IK 回退时使用 | ✅ 加权 current_dist (base joints 更高权重) | ❌ 无 | N/A | N/A |
| **q7 冗余优化** | ❌ 无 (7 关节固定) | ✅ Brent 1D 优化 | ❌ 无 | N/A | N/A |
| **IK 失败处理** | hold last_good | hold current position | hold last target | 机器人自带处理 | return current qpos |

**dexmani 当前位置与差距**:
- dexmani 的 DLS + MPlib 回退在确定性方面是正确的架构选择 — 参考 BVPro 也使用纯 DLS
- **关键差距**: `kin.compute_manipulability()` (`kinematics.py:69-74`) 已精确实现 Yoshikawa 指标，且在 `PlanningProfile.ik_score_manipulability_weight=1.0` 中有配置项，但 `TeleopIKSolver.solve_differential_ik()` **未调用它**。LeFranX 的经验表明可操作性评分对奇点规避至关重要
- LeFranX 的解析 IK 是针对 Franka 7-DOF 运动学的特化解，xArm7 没有已知的解析解，所以 Brent 优化路径不可直接移植
- ManiUniCon 的 Pink QP 多目标优化 (FrameTask + PostureTask) 提供了一种更现代的 IK 方案: QP 求解器自然处理秩亏且收敛更好，是值得关注的替代路径

**建议**:
1. **P0**: 在 DLS 每步迭代中检查 `compute_manipulability()`，低于阈值时增加 damping 或拒绝该解
2. **P1**: 参考 BVPro 提供纯 DLS-only 模式（关闭 MPlib 回退）以减少延迟波动
3. **P1**: 参考 BVPro 实现自适应迭代 — 奇点附近降低步长、增加 damping

### 3.3 手部重定向

| 特性 | dexmani_real | LeFranX | BunnyVision Pro | Open-Teach | ManiUniCon |
|------|-------------|---------|-----------------|------------|------------|
| **算法** | DexPilot (dex_retargeting) | DexPilot (dex_retargeting) | SeqRetargeting (dex_retargeting) | KDL IK + 角度计算 | N/A (Robotiq 二指爪, 触发器 toggle) |
| **粉红指适配** | ✅ XHandRefAdapter (1.2x–2.2x) | ✅ adaptive_retargeting_xhand (1.2x–2.2x) | ❌ 无 (dexpilot 标准) | N/A (JointControl + KDL) | N/A |
| **SVD 手掌帧估计** | ✅ `estimate_frame_from_hand_points` | ✅ SVD `estimate_frame_from_hand_points` | ❌ 无 (使用 OPERATOR2AVP 固定矩阵) | ✅ palm normal from cross product | N/A |
| **坐标变换** | landmarks → wrist_rot → OPERATOR2MANO_RIGHT | 同 | OPERATOR2AVP_RIGHT (equiv.) | 24 bones → 归一化 palm frame | N/A |
| **处理位置** | 热路径 (每帧) | 热路径 (每帧) | Server-side (ZMQ) | 独立 Process | N/A |
| **smoothing** | dex_retargeting 内置 low_pass_alpha | 自定义 EMA | 未知 (server-side) | Moving avg + Slerp filter | N/A |
| **错误处理** | hold prev_hand_cmd | hold current qpos | hold last target | pause mode | N/A |

**dexmani 当前位置与差距**:
- **XHandRefAdapter 直接继承自 LeFranX** — 粉红指适配器 (`ref_adapter.py`) 的核心逻辑（extension_range 0.03-0.07m, pinky_scale 1.2-2.2x）与 LeFranX 的 `adaptive_retargeting_xhand` 几乎一致
- SVD 手掌帧估计 (`estimate_frame_from_hand_points`) 与 LeFranX 共享算法（`vr_hand_detector_adapter.py:294-342`）
- dexmani 的创新在于 `ref_adapter.py` 的 pinky_blend 参数化（默认 1.0）提供了比 LeFranX 更灵活的调参接口
- ManiUniCon 的 Robotiq 二指爪场景不涉及手部重定向，但 delta-based 手臂控制方式可部分借鉴（见 §5 P1-3）
- **差距**: BVPro 的 SeqRetargeting 可能对某些手势更准确（基于 keypoint 匹配序列而非向量差分），但 DexPilot 在 XHand 上已验证良好

**建议**: 维持现状，XHandRefAdapter 已是最佳实践

### 3.4 通信架构

| 特性 | dexmani_real | LeFranX | BunnyVision Pro | Open-Teach | ManiUniCon |
|------|-------------|---------|-----------------|------------|------------|
| **拓扑** | 单进程 + 1 daemon 线程 | 多进程 + 多线程 | 三机 ZMQ | 多进程 ZMQ | 多进程 + 共享内存 |
| **协议** | HTS TCP + Python 直接调用 | TCP text + Python 直接调用 | ZMQ PUB/SUB + REQ/REP | ZMQ PUB/SUB + PUSH/PULL | Lock-Free SharedMemory (RingBuffer + Queue) |
| **进程模型** | 主线程 50Hz | 主线程 + C++ 线程 | Client 线程 + Server Docker | 5+ 独立 Process | 相机 Process + 策略 Process + 机器人 Process |
| **VR 与控制解耦** | ❌ 同一进程 | ✅ C++ 线程 (独立) | ✅ 不同 machine | ✅ 不同 Process | ✅ 独立 Policy Process 30Hz + Robot Process 200Hz |
| **GIL 影响** | ⚠️ VR解析+IK+Retarget 在同一 GIL | ✅ C++ 部分无 GIL | ✅ 跨机器天然解耦 | ✅ 跨进程 | N/A (跨进程) |
| **重定向服务** | 本地 Python | 本地 Python | Docker 容器 | 独立 ZMQ node | N/A (无手部重定向) |
| **可扩展性** | ★★☆☆☆ | ★★★☆☆ | ★★★★☆ | ★★★★★ | ★★★★☆ |
| **部署复杂度** | ★★★★★ (单脚本) | ★★★☆☆ (编译 C++ + 多进程) | ★★☆☆☆ (Docker + VP + Client) | ★★☆☆☆ (Unity + 多终端 + ROS) | ★★★☆☆ (Python only, 无 Docker/Unity) |

**dexmani 当前位置与差距**:
- dexmani 的单进程架构是**最大瓶颈** — 所有计算（VR 解析、IK、retargeting、safety）在同一线程中串行执行
- 四个参考框架均实现了 VR 与控制解耦：LeFranX 用 C++ 线程，BVPro 用 ZMQ 节点，Open-Teach 用 ZMQ 进程，ManiUniCon 用共享内存进程
- ManiUniCon 的 Lock-Free SharedMemory 方案在纯 Python 环境下提供了接近 C++ 的 IPC 性能（微秒级），是一个值得关注的轻量级替代方案
- **GIL 影响评估**: VR 解析（HTS SDK, ~1ms）和 retargeting（dex_retargeting, ~2-5ms）均占用 GIL，与 IK 计算竞争 CPU。在 50Hz (20ms 预算) 下当前未超预算，但增加功能（如 camera 预处理）后会出问题

**建议**: 
- **P1**: 引入 ZMQ 进程分离 — VR tracker 独立进程发布帧，TeleopController 订阅并计算，参考 Open-Teach 的 PUB/SUB 模式
- **P2**: 长期考虑 Lock-Free SharedMemory IPC，参考 ManiUniCon 的 RingBuffer/Queue 设计

### 3.5 安全机制

| 安全检查 | dexmani_real | LeFranX | BunnyVision Pro | Open-Teach | ManiUniCon |
|----------|-------------|---------|-----------------|------------|------------|
| **VR 帧新鲜度超时** | ✅ 0.2s stale / 1.0s E-Stop | ⚠️ 基础 hold | ❌ 无检测 | ✅ NOBLOCK 10 次重试 | N/A |
| **关节力矩监控** | ✅ 7 关节 [50,50,30,30,30,20,20] Nm | ✅ 150-250 Nm 碰撞阈值 | ❌ 仅 return code | ❌ 无 | validate_action 统一检查 |
| **关节电流监控** | ✅ hand_current < 500mA | ❌ 无 | ❌ 无 | ❌ 无 | N/A |
| **关节温度监控** | ✅ hand_temp < 70°C | ❌ 无 | ❌ 无 | ❌ 无 | N/A |
| **手部通信状态** | ✅ hand_error flag | ❌ 无 | ❌ 无 | ❌ 无 | N/A |
| **关节限位** | ✅ E-Stop (arm) / Warning (hand) | ✅ position clamping | ❌ 无 | ✅ position clamping | ConfigurationLimit (IK solver) |
| **工作空间检查** | ✅ bounds check + clamp | ✅ 0.75m max offset | ❌ 无 | ❌ 无 | position + orientation (Euler) |
| **桌面 FK 碰撞** | ✅ FingertipDeskSafety | ❌ 无 | ❌ 无 | ❌ 无 | N/A |
| **自碰撞检测** | ✅ self-collision (MPlib) | ❌ 无 | ❌ 无 | ❌ 无 | N/A |
| **速度限制** | ✅ bottleneck scaling (driver) | ✅ Ruckig @ 1kHz | ✅ clip_arm_velocity | ✅ 分辨率缩放 0.6× | 插值器层 (0.25m/s, 0.5rad/s) |
| **Joint jump 保护** | ✅ 5°/frame arm, 10°/frame hand | ❌ 无 | ✅ clip_arm_next_qpos | ❌ 无 | N/A |
| **Retarget 质量检查** | ✅ physio range [-0.5, 2.5] rad | ❌ 无 | ❌ 无 | ❌ 无 | N/A |
| **每帧质量标记** | ✅ 10bit QualityFlags | ❌ 无 | ❌ 无 | ❌ 无 | N/A |
| **Hold-on-failure** | ✅ 所有管道失败 → hold | ✅ IK 失败 → hold | ✅ 无新数据 → hold last | ✅ pause mode | validate_action 失败 → hold |
| **E-Stop 升级** | ✅ 帧丢失持续/硬件错误 | ❌ 无 | ❌ 无 | ❌ 无 | error_state flag → 所有进程退出 |
| **命令超时** | ⚠️ 无 (依赖 hold-on-failure) | ✅ 500ms 命令超时 | ❌ 无 | ❌ 无 | N/A |

**结论**: dexmani 的安全系统是**五个框架中最全面的**，四层安全模型 + 每帧 10bit 质量标记在四个参考框架中均无等效机制。

**差距**:
- LeFranX 的 500ms 命令超时 (`franka_server.cpp:358-363`) 是一个有价值的额外保护层 — 如果 Python 进程崩溃，C++ server 会在 500ms 后自动 hold
- BVPro 的跟踪丢失软减速（不是立即 hold/E-Stop）在某些场景下更平滑
- ManiUniCon 的集中化 validate_action() 安全检查门是更清洁的架构模式，且方向 workspace 检查是 dexmani 缺失的维度

### 3.6 数据录制

| 特性 | dexmani_real | LeFranX | BunnyVision Pro | Open-Teach | ManiUniCon |
|------|-------------|---------|-----------------|------------|------------|
| **文件格式** | HDF5 (episode) | HuggingFace Dataset | HDF5 + NPY | HDF5 + AVI + Pickle | NPZ → Zarr → LeRobot v3.0 |
| **观测空间** | state + action + vr_frame + quality_flags + T_base_eef | arm_joint.pos + hand_joint.pos + ee_pose | joint_states + eef_pose + raw_hand | cartesian + joint + hand_joint | arm_qpos + hand_qpos (basic) |
| **质量元数据** | ✅ 10bit flags/frame | ❌ 无 | ❌ 无 | ❌ 无 | N/A |
| **相机支持** | ✅ camera extrinsics (T_base_eef) | ✅ multi-camera | ❌ 无 (仅 robot data) | ✅ multi-camera HDF5 + AVI | multi-camera (独立 Process) |
| **Episode 管理** | start/stop/success flag | LeRobot standard | keyboard (s/q/a/n) | directory per episode | A button toggle / B button drop |
| **压缩** | HDF5 默认 | HuggingFace built-in | 无压缩 | gzip level 6 | blosc (Zarr) |
| **频率** | 50Hz (同步控制频率) | 30-100Hz (可配置) | 50Hz (同步) | 30-300Hz (per-channel) | 30Hz (同步策略频率) |

**dexmani 当前位置与差距**:
- 10bit 质量标记是**独特优势**，可支持训练数据过滤（剔除低质量帧）
- **差距**: 不支持 LeRobot 格式 — 如果未来想与 HuggingFace 生态集成（预训练模型、社区数据集），需要格式转换层
- ManiUniCon 的 Zarr + blosc 压缩方案在文件大小和读取速度上有优势，其多 episode 导出模式也值得借鉴
- ManiUniCon 的 max_record_steps=5000 是一个实用的硬上限安全机制

**建议**: 
- **P2**: 添加 LeRobot 格式导出器（HDF5 → LeRobot HuggingFace Dataset），作为可选的后期转换步骤
- 维持 HDF5 作为原生录制格式（更低延迟，更少依赖）

### 3.7 性能与延迟

**dexmani 每帧延迟分解 (20ms 预算 @ 50Hz):**

| 步骤 | 延迟 | 波动源 | 占比 |
|------|------|--------|------|
| VR 帧读取 (`get_latest()`) | ~1ms | mutex 锁竞争 | 5% |
| 跟踪质量检查 | ~0.1ms | — | 0.5% |
| 机器人状态读取 (`get_state()`) | ~2ms | 串口通信 | 10% |
| ArmWristMapper | ~0.2ms | — | 1% |
| IK 求解 (DLS) | ~1-3ms | — | 5-15% |
| IK 求解 (MPlib 回退) | ~10ms | 随机种子 | **50%** |
| 手部重定向 | ~2-5ms | dex_retargeting 优化 | 10-25% |
| EMA/Workspace/Jump clamp | ~0.3ms | — | 1.5% |
| 安全检查 (torque/current/temp) | ~0.3ms | — | 1.5% |
| 发送指令 (`send_action()`) | ~1ms | 串口/网络 | 5% |
| **总计 (DLS 成功)** | **~7-12ms** | | 35-60% |
| **总计 (MPlib 回退)** | **~17-22ms** | | 85-110% ⚠️ |

**热点分析**:
- **MPlib position IK 回退** 是最大延迟波动源 — 当 DLS 失败时帧时间可能超出 20ms 预算
- 手部重定向 (2-5ms) 是第二大波动源 — dex_retargeting 内的优化求解可能因初始值不同而有明显变化
- **GIL 影响**: VR 解析线程（daemon）和主线程交替持有 GIL，但主循环中无明显 I/O 阻塞等待

**与其他框架对比:**

| 框架 | IK 延迟 | 总体控制延迟 | 瓶颈 |
|------|---------|-------------|------|
| dexmani | 1-10ms (DLS→MPlib) | 7-22ms | MPlib 回退 |
| LeFranX | <1ms (C++ 解析 IK) | ~5-10ms | TCP socket 通信 |
| BVPro | Server-side DLS (~1ms) | ~10-20ms (含网络) | ZMQ 网络延迟 |
| Open-Teach | N/A (Cartesian servo) | 1/60s (robot 内环) | 机器人控制器 |
| ManiUniCon | <1ms (QP, 200Hz) | ~5ms (200Hz) | 无显著瓶颈 |

**建议**:
1. **P0-2**: DLS-only 模式可消除 ~10ms 的 MPlib 回退开销
2. **P2**: 如 IK 延迟成为瓶颈，参考 LeFranX 用 C++/pybind11 实现 DLS solver
3. ManiUniCon 的 200Hz 插值器 + QP IK (<1ms) 组合展示了纯 Python 下的低延迟路径

---

## 4. dexmani_real 能力矩阵

### 8 维度雷达图（文本）

```
                       VR 质量 ★★★★☆ (4)
                           /\
                          /  \
            部署易用性 ★★★★★│    │★★★☆☆ IK 鲁棒性
                        │    │
                        │    │
                        │    │
      可扩展性 ★★★☆☆ ───┼────┼─── ★★★★☆ 手部重定向
                        │    │
                        │    │
                        │    │
         延迟 ★★★★☆ ────┼────┼─── ★★★★★ 安全性
                        │    │
                        │   /
             数据质量 ★★★★★   /
                          \/
```

**评分详情:**

| 维度 | 评分 | 相对位置 | 依据 |
|------|------|----------|------|
| **VR 质量** | ★★★★☆ (4) | = BVPro, > Open-Teach, = ManiUniCon | HTS SDK 类型安全，21 landmarks ↔ Vision Pro 25 matrices，但 Vision Pro 手指跟踪精度更高 |
| **IK 鲁棒性** | ★★★☆☆ (3) | < LeFranX (5), < ManiUniCon (4), = BVPro (3), > Open-Teach (N/A) | DLS+回退可靠但缺少可操作性评分和 q7 优化；ManiUniCon QP 求解器自然处理秩亏 |
| **手部重定向** | ★★★★☆ (4) | = LeFranX (4), >= BVPro (4), > Open-Teach (3), > ManiUniCon (N/A) | DexPilot + pinky 适配 = LeFranX 同款算法，远超 Robotiq 二指爪 |
| **安全性** | ★★★★★ (5) | > 所有参考框架 | 四层安全 + 10bit 质量标记，无框架可比 |
| **数据质量** | ★★★★★ (5) | > 所有参考框架 | 唯一带有 per-frame quality flags 的框架 |
| **延迟** | ★★★★☆ (4) | = LeFranX (4), < ManiUniCon (5), < BVPro (3 网络), > Open-Teach (N/A) | DLS 路径 7-12ms，回退时波动大；ManiUniCon 200Hz 插值更平滑 |
| **可扩展性** | ★★★☆☆ (3) | < Open-Teach (5), < ManiUniCon (4), < BVPro (4), = LeFranX (3) | 单线程架构限制了水平扩展；ManiUniCon 多进程共享内存更优 |
| **部署易用性** | ★★★★★ (5) | > 所有参考框架 | 单脚本运行，无 Docker/ROS/Unity 依赖 |

---

## 5. 可采纳改进（优先级排序）

### 优先级说明
- **P0** (关键): 低工作量、高影响，应立即实施
- **P1** (重要): 中低工作量、中高影响，下一个迭代
- **P2** (增强): 中高工作量、中影响，长期规划
- **P3** (可选择): 低影响或高工作量，视需求决定

### 改进清单

| # | 优先级 | 改进项 | 来源 | 工作量 | 影响 | 涉及文件 |
|---|--------|--------|------|--------|------|----------|
| 1 | **P0** | 遥操作 IK 加入可操作性评分 | LeFranX | 低 (~20 行) | **高** (IK 鲁棒性 +20%) | `planning/ik.py:231-288` `solve_differential_ik()` |
| 2 | **P0** | VR 跟踪丢失时软减速保持 | BVPro | 低 (~30 行) | 中 (消除急停抖动) | `teleop/core/controller.py:207-211` `_tick()` |
| 3 | **P1** | Cartesian Pose 插值 (频率解耦) | ManiUniCon | 中 (~200行) | **高** (消除 stale reuse, 平滑控制) | 新增 `teleop/vr/pose_interpolator.py` + `controller.py` |
| 4 | **P1** | ZMQ 进程分离（VR 解耦控制） | Open-Teach | 中 (~200 行) | **高** (消除 GIL 瓶颈) | `teleop/vr/vr_tracker.py` + 新增 `vr_publisher.py` |
| 5 | **P1** | DLS-only 模式（position IK 可配置关闭） | BVPro | 低 (~10 行 config) | 中 (降低延迟波动) | `planning/types.py:119` `TeleopProfile.use_position_ik` |
| 6 | **P1** | 近奇点自适应迭代 DLS | BVPro | 中 (~50 行) | 中 (奇点附近更平滑) | `planning/ik.py:231-268` `solve_differential_ik()` |
| 7 | **P1** | EEF 方向工作空间边界 | ManiUniCon | 低 (~50行) | 中 (防止 wrist 极值自碰撞) | `planning/planner.py:638-665` WorkspaceSafety |
| 8 | **P1** | VR per-step delta 旋转安全限制 | ManiUniCon | 低 (~20行) | 中 (防止 VR 跟踪跳变) | `teleop/vr/arm_mapper.py:49-75` `map()` |
| 9 | **P1** | 集中化 validate_action() 安全检查门 | ManiUniCon | 中 (~100行) | 中 (统一安全逻辑) | `robot/interface.py` + `controller.py` |
| 10 | **P2** | HTS 帧解析 C++ 扩展（pybind11） | LeFranX | 高 (~500 行 C++) | 中 (降低 ~1ms VR 解析) | 新增 `hts_bridge/` C++ 模块 |
| 11 | **P2** | XArm7 解析 IK（替代 MPlib position IK） | LeFranX | 高 (~1000 行) | **高** (完全消除 MPlib 回退开销) | 新增 `planning/analytic_ik.py` |
| 12 | **P2** | LeRobot 数据集格式支持 | LeFranX | 中 (~150 行) | 中 (HuggingFace 生态集成) | 新增 `recording/lerobot_exporter.py` |
| 13 | **P2** | Lock-Free 共享内存 IPC (进程解耦) | ManiUniCon | 高 (~500行) | **高** (进程间微秒级 IPC) | 新增 `shared_memory/` 模块 |
| 14 | **P3** | Docker 化重定向服务 | BVPro | 低 (~50 行 Dockerfile) | 低 (当前非瓶颈) | 新增 `Dockerfile.teleop` |
| 15 | **P3** | 双手（左手）支持 | BVPro | 中 (~300 行) | 低 (当前需求 single-hand) | `teleop/` 多处 + 新增左手配置 |

### 详细实施方案

#### P0-1: IK 可操作性评分

**参考**: LeFranX `weighted_ik.cpp:71-76`, dexmani `kinematics.py:69-74`

**当前位置**: `kin.compute_manipulability()` 已实现但 `TeleopIKSolver.solve_differential_ik()` 未调用。

**实施**:
```python
# planning/ik.py solve_differential_ik() 中，迭代循环内加入:
manip = self.kin.compute_manipulability(current_qpos)
if manip < profile.min_manipulability:  # 新增配置项
    damping = profile.differential_ik_damping * 10.0  # 自适应增大 damping
else:
    damping = profile.differential_ik_damping
```
- 新增 `TeleopProfile.min_manipulability` (默认 0.0001)
- 新增 `TeleopProfile.singularity_damping_scale` (默认 10.0)

#### P0-2: 跟踪丢失软减速

**参考**: BVPro `xarm7_ability.py:185-194` `clip_arm_velocity()`

**当前位置**: `controller.py:207-211` 连续丢失 → E-Stop，没有中间过渡。

**实施**:
```python
# controller.py _tick() 中:
if not tq_result.ok:
    self.error_handler.record_failure("vr_stale")
    if tq_result.tracking_lost:
        if tq_result.lost_duration_s < 1.0:  # <1s: 指数减速到零
            decay = np.exp(-tq_result.lost_duration_s * 3.0)
            self._last_arm_cmd = prev_arm_cmd + decay * (current_qpos - prev_arm_cmd)
        else:
            self._escalate_to_emergency(...)  # >1s: E-Stop
    return
```

#### P1-3: Cartesian Pose 插值 (频率解耦)

**参考**: ManiUniCon `pose_trajectory_interpolator.py:78-207`

**问题描述**: dexmani 的 50Hz 控制循环在 VR 更新帧率 ~25-30Hz 时会重读同一 VR 帧 2 次 (stale reuse)，导致:
1. 两帧相同的命令 → 不流畅的运动
2. 人手抖动 (~2-3mm, 8-12Hz) 和 VR 跟踪噪声直接传播到机器人
3. 关节空间 EMA (默认关闭) 只能后 IK 缓解，无法消除 Cartesian 源头的抖动

**实施**:

新建 `dexmani_real/teleop/vr/pose_interpolator.py`:

```python
"""Cartesian pose interpolator — smooths between discrete VR frames."""

from __future__ import annotations

import time
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class CartPoseInterpolator:
    """Interpolates between discrete VR-frame poses for smooth robot motion.
    
    Receives target poses at VR rate (~30 Hz) and produces interpolated
    poses at the controller's sampling rate (50 Hz) via:
      - Linear interpolation for position
      - SLERP for rotation
      - Speed-limited temporal scheduling
    """

    def __init__(
        self,
        max_pos_speed: float = 0.25,  # m/s
        max_rot_speed: float = 0.5,   # rad/s
        max_history: int = 5,
    ) -> None:
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self._waypoints: deque[tuple[float, np.ndarray, np.ndarray]] = deque(maxlen=max_history)
        self._last_pos: np.ndarray | None = None
        self._last_rot: Rotation | None = None
        self._earliest_arrival_time: float = 0.0

    def push_target_pose(
        self, pos: np.ndarray, quat_wxyz: np.ndarray, timestamp: float | None = None
    ) -> None:
        """Enqueue a new target waypoint (called at VR frame rate)."""
        ts = timestamp if timestamp is not None else time.monotonic()
        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
        quat_wxyz = quat_wxyz / np.linalg.norm(quat_wxyz)
        
        if self._last_pos is not None and self._last_rot is not None:
            pos_dist = float(np.linalg.norm(pos - self._last_pos))
            rot_dist = self._rotation_distance(quat_wxyz, self._last_rot)
            pos_time = pos_dist / self.max_pos_speed
            rot_time = rot_dist / self.max_rot_speed
            travel_time = max(pos_time, rot_time)
            self._earliest_arrival_time = max(ts, self._earliest_arrival_time + travel_time)
        else:
            self._earliest_arrival_time = ts
        
        self._last_pos = pos.copy()
        self._last_rot = Rotation.from_quat(
            np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        )
        self._waypoints.append((self._earliest_arrival_time, pos.copy(), quat_wxyz.copy()))

    def get_interpolated_pose(self, now: float | None = None) -> tuple[np.ndarray, np.ndarray] | None:
        """Get interpolated pose at current time (called at controller rate)."""
        if len(self._waypoints) < 2:
            if len(self._waypoints) == 1:
                _, pos, quat = self._waypoints[0]
                return pos.copy(), quat.copy()
            return None
        
        now = now if now is not None else time.monotonic()
        
        while len(self._waypoints) > 1 and self._waypoints[1][0] < now:
            self._waypoints.popleft()
        
        if len(self._waypoints) < 2:
            return None
        
        t_prev, pos_prev, quat_prev = self._waypoints[0]
        t_next, pos_next, quat_next = self._waypoints[1]
        
        if t_next <= t_prev:
            return pos_prev.copy(), quat_prev.copy()
        
        alpha = (now - t_prev) / (t_next - t_prev)
        alpha = max(0.0, min(1.0, alpha))
        
        interp_pos = pos_prev + alpha * (pos_next - pos_prev)
        
        rot_prev = Rotation.from_quat([quat_prev[1], quat_prev[2], quat_prev[3], quat_prev[0]])
        rot_next = Rotation.from_quat([quat_next[1], quat_next[2], quat_next[3], quat_next[0]])
        slerp = Slerp([t_prev, t_next], Rotation.concatenate([rot_prev, rot_next]))
        interp_rot = slerp(now)
        interp_quat_xyzw = interp_rot.as_quat()
        interp_quat = np.array([interp_quat_xyzw[3], interp_quat_xyzw[0], 
                                 interp_quat_xyzw[1], interp_quat_xyzw[2]])
        
        return interp_pos, interp_quat / np.linalg.norm(interp_quat)

    def _rotation_distance(self, quat_wxyz: np.ndarray, last_rot: Rotation) -> float:
        rot = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        delta = rot * last_rot.inv()
        angle = np.linalg.norm(delta.as_rotvec())
        return float(angle)

    def reset(self) -> None:
        self._waypoints.clear()
        self._last_pos = None
        self._last_rot = None
        self._earliest_arrival_time = 0.0
```

在 `controller.py` 中集成: `TeleopController.__init__` 添加 `use_cartesian_interpolation: bool = False` 参数和 `self._pose_interpolator = CartPoseInterpolator()`。在 `_compute_arm_command()` 中，`arm_mapper.map()` 后 `push_target_pose()` 再 `get_interpolated_pose()` 获取插值结果送入 IK。

- 新增 `TeleopProfile.use_cartesian_interpolation` (默认 `False`)

---

## 6. 代码路径对照

以下表格对照五个框架中关键控制流步骤的精确文件:行号：

| 步骤 | dexmani_real | LeFranX | BunnyVision Pro | Open-Teach | ManiUniCon |
|------|-------------|---------|-----------------|------------|------------|
| **VR 帧接收** | `teleop/vr/vr_tracker.py:220` `_receive_loop()` | `franka_xhand_teleoperator/src/vr_message_router.cpp:205` `tcp_receiver_thread()` | `bunny_teleop/bimanual_teleop_client.py:74` `update_teleop_cmd()` | `openteach/components/detector/oculus.py:74` `stream()` | `maniunicon/utils/quest_controller.py:107` `_update_internal_state()` |
| **VR 帧解析** | `teleop/vr/vr_tracker.py:252` `convert_frame()` | `vr_message_router.cpp:262` `parse_vr_messages()` | Server-side (external `bunny_teleop_server`) | `oculus.py:38` `_extract_data_from_token()` | `quest_controller.py:128` `_extract_vr_pose()` |
| **坐标帧变换** | `vr_tracker.py:274-289` `extract_geometry()` (RLHFU 内置) | `arm_ik_processor.py:193` position + `:269` quaternion | 无本地代码 (server-side) | `keypoint_transform.py:55` `transform_keypoints()` | `quest_controller.py:46-53` R_ve + `quest_controller.py:284-285` delta transform |
| **Wrist → EEF 映射** | `teleop/vr/arm_mapper.py:49` `map()` | `arm_ik_processor.py:129` `_compute_target_pose()` | Server-side bimanual alignment | N/A (Cartesian delta 直接计算) | `quest_controller.py:192-338` `_calculate_action()` (delta accumulation) |
| **手臂 IK** | `planning/ik.py:46` `solve()` → `:231` `solve_differential_ik()` / `:115` `solve_position_ik()` | `weighted_ik.cpp:232` `solve_q7_optimized()` (Brent + analytic) | Server-side `compute_ik()` (DLS, `lambda²=1e-5`, max 100 iters) | N/A (机器人 Cartesian servo) | `maniunicon/utils/ik_solver.py:171` `solve()` (Pink QP) |
| **手部重定向** | `teleop/vr/hand_retarget.py:93` `retarget()` | `xhand_vr_teleoperator.py:224` `self.retargeting.retarget(ref_value)` | Server-side `dex_retargeting.seq_retarget` | `allegro_retargeters.py:34` `calculate_finger_angles()` (KDL/angle) | N/A (Robotiq gripper, trigger toggle) |
| **手部适配器** | `teleop/vr/ref_adapter.py:32` `XHandRefAdapter.apply()` | `vr_hand_detector_adapter.py:27` `adaptive_retargeting_xhand()` | N/A | N/A | N/A |
| **平滑滤波** | `controller.py:380` `ema_smooth()` (arm) + dex_retargeting 内置 (hand) | `arm_ik_processor.py:360` exponential smoothing (arm) + hand smoothing | PID 内环自然平滑 | `keypoint_transform.py:84` moving avg + `franka.py:22` Slerp filter | `maniunicon/utils/filter.py:77` `JointSpaceSmoother.smooth()` + `maniunicon/utils/pose_trajectory_interpolator.py:187` `__call__()` |
| **安全: 帧新鲜度** | `teleop/core/tracking.py:49` (stale 0.2s) + `:71` (lost 1.0s) | `franka_fer_vr_teleoperator.py:213` if no VR data → hold | 无检测 (hold last by design) | `franka.py:103` NOBLOCK 10 次重试 | N/A (无显式检测) |
| **安全: 力矩** | `teleop/control/safety.py:21` `check_arm_torque()` | `franka_server.cpp:372` collision behavior thresholds | 无 | 无 | N/A |
| **安全: 工作空间** | `controller.py:321` `robot.check_workspace()` | `arm_ik_processor.py:230` 0.75m max offset | 无 | 无 | `maniunicon/policies/quest.py:110` `_clip_tcp_pose_to_bounds()` + `maniunicon/core/robot.py:72` `_clip_action_to_bounds()` |
| **安全: 桌面碰撞** | `planning/planner.py:668` `FingertipDeskSafety.check_hand_desk_clearance()` | 无 | 无 | 无 | N/A |
| **发送手臂指令** | `robot/interface.py:330` `send_action()` → XArm7 SDK | `franka_fer.py:230` `send_action()` → TCP "SET_POSITION" | `xarm7_ability.py:196` `control_arm_qpos()` → PID velocity (:200) → XArm SDK | `franka.py:226` `robot.arm_control(final_pose)` → ROS Cartesian servo | `maniunicon/robot_interface/xarm6_robotiq.py:202` `send_action()` |
| **发送手部指令** | `interface.py:330` `send_action()` → XHand SDK | `xhand.py:261` `send_action()` → XHand SDK | `xarm7_ability.py:230` `control_hand_qpos()` → `hand.set_joint_angle()` | `allegro.py` → ROS joint command | N/A (gripper on/off only) |
| **录制** | `controller.py:258` `recorder.add_frame()` | LeRobot `push_observation()` | `teleop_bimanual_xarm7_ability.py:206` list append → HDF5 save | `robot_state.py:69` → HDF5 per-channel | `maniunicon/core/robot.py:369-394` action_record_buffer (TimestampAlignedBuffer) |

---

## 7. 配置参数对比

### IK 参数

| 参数 | dexmani_real | LeFranX | BunnyVision Pro | Open-Teach | ManiUniCon |
|------|-------------|---------|-----------------|------------|------------|
| **IK 方法** | DLS + MPlib Position IK | Analytic + Brent q7 opt | DLS (Pinocchio) | 无自定义 IK | Pink QP |
| **DLS damping (λ²)** | 0.02 | N/A | 1e-5 | N/A | 1e-12 (QP regularization) |
| **DLS gain** | 1.0 | N/A | 1.0 (implied by error) | N/A | N/A |
| **DLS max iterations** | 单次 (Jacobian 伪逆) | N/A (analytic) | 100 | N/A | QP solve |
| **收敛阈值 pos** | 0.008 m | N/A (analytic exact) | 1e-3 (6D twist norm) | N/A | QP convergence |
| **收敛阈值 rot** | 0.08 rad | N/A | (包含在 twist norm) | N/A | QP convergence |
| **Max IK jump** | 30-60° per joint (7 joints) | N/A (position clamp only) | N/A (PID 速度限) | N/A | N/A |
| **可操作性权重** | 0 (未使用) | `weight_manip` (configurable) | 0 | 0 | QP implicit |

### 平滑/滤波参数

| 参数 | dexmani_real | LeFranX | BunnyVision Pro | Open-Teach | ManiUniCon |
|------|-------------|---------|-----------------|------------|------------|
| **Arm EMA alpha** | 1.0 (默认关闭) | configurable | N/A (PID 内环) | 0.8 (complementary filter) | JointSpaceSmoother (EWMA+velocity+accel+Kalman) |
| **Hand smoothing** | dex_retargeting low_pass_alpha | 自定义 EMA | server-side | Moving avg (window=5) | N/A |
| **滤波方法** | EMA (arm) + low_pass (hand) | EMA (arm+hand) | PID 衍生 | Moving avg + Slerp | JointSpaceSmoother + PoseTrajectoryInterpolator |

### 速度限制

| 参数 | dexmani_real | LeFranX | BunnyVision Pro | Open-Teach | ManiUniCon |
|------|-------------|---------|-----------------|------------|------------|
| **Joint vel limits** | [60,60,60,60,90,90,120]°/s | N/A (Ruckig 1kHz trajectory) | [0.8,0.8,0.8,0.8,1.0,1.0,1.5] rad/s | N/A (Cartesian servo) | 3.14 rad/s (config) |
| **Joint jump clamp** | 5°/frame arm, 10°/frame hand | ❌ | ✅ clip_arm_next_qpos | ❌ | N/A |
| **Cartesian vel limit** | N/A (joint level) | 0.75m max offset clamp | PID velocity clip | 0.6× resolution scale | 0.25 m/s pos, 0.5 rad/s rot (interpolator) |
| **Soft-start** | ✅ `reset_soft_start()` | ❌ (Ruckig 处理) | ✅ 1/3 speed during init | ❌ | `reset_to_init()` |

### 安全超时/阈值

| 参数 | dexmani_real | LeFranX | BunnyVision Pro | Open-Teach | ManiUniCon |
|------|-------------|---------|-----------------|------------|------------|
| **VR stale timeout** | 0.2 s | ❌ | ❌ | ❌ (NOBLOCK 10× retry) | N/A |
| **VR lost E-Stop** | 1.0 s | ❌ | ❌ | ❌ | N/A |
| **命令超时** | ❌ (hold-on-failure) | ✅ 500ms (franka_server) | ❌ (hold last) | ❌ | N/A |
| **Arm torque limit** | [50,50,30,30,30,20,20] Nm | 150-250 Nm (碰撞阈值) | ❌ | ❌ | N/A |
| **Hand current limit** | 500 mA | ❌ | ❌ | ❌ | N/A |
| **Hand temp limit** | 70°C | ❌ | ❌ | ❌ | N/A |
| **Workspace bounds** | [[0.28,0.72],[-0.45,0.45],[0.05,0.5]] m | 基础 0.75m offset | ❌ | ❌ | position + orientation (Euler) |
| **桌面 FK threshold** | configurable `fingertip_threshold` | ❌ | ❌ | ❌ | N/A |

### 录制参数

| 参数 | dexmani_real | LeFranX | BunnyVision Pro | Open-Teach | ManiUniCon |
|------|-------------|---------|-----------------|------------|------------|
| **格式** | HDF5 | HuggingFace Dataset | HDF5 + NPY | HDF5 + AVI + Pickle | NPZ → Zarr → LeRobot |
| **频率** | 50 Hz | 30-100 Hz | 50 Hz | 30-300 Hz (per-channel) | 30 Hz |
| **质量标记** | ✅ 10bit per frame | ❌ | ❌ | ❌ | N/A |
| **相机数据** | T_base_eef extrinsics | multi-camera | ❌ (仅 robot data) | multi-camera HDF5 + AVI | multi-camera Zarr |
| **Episode 管理** | start/stop/success flag | LeRobot built-in | keyboard s/q/a/n | directory per episode | A/B button toggle/drop |

---

## 8. 附录

### A. 术语表

| 术语 | 说明 |
|------|------|
| **DLS** | Damped Least Squares — 阻尼最小二乘法 IK，在奇点附近通过增大 damping 因子避免矩阵不可逆 |
| **DexPilot** | 一种基于向量差分的 hand retargeting 算法，计算人手指尖相对位移映射到机器人手 |
| **SeqRetargeting** | 基于序列 keypoint 匹配的 retargeting，优化人手指关键点与机器人 link 位置的匹配误差 |
| **Yoshikawa manipulability** | 可操作性度量 `μ = sqrt(det(J·J^T))`，衡量机器人配置距离奇异点的远近 |
| **Brent's method** | 一种 1D 优化算法，不需要导数，用于 LeFranX 中对 q7 肘关节的最优搜索 |
| **EMA** | Exponential Moving Average — 指数移动平均，用于平滑连续帧之间的命令跳变 |
| **ZMQ** | ZeroMQ — 高性能异步消息库，支持 PUB/SUB、REQ/REP、PUSH/PULL 等多种通信模式 |
| **Pinocchio** | 高效的刚体动力学库，提供 FK、Jacobian、动力学计算 |
| **MPlib** | Motion Planning Library — 基于 Pinocchio 的运动规划库，提供随机 IK 和路径规划 |
| **Ruckig** | 在线轨迹生成算法，可在 1ms 内生成 jerk-limited 平滑轨迹 |
| **HTS** | Hand Tracking SDK — Meta 官方的手部追踪 Python SDK |
| **Hydra** | Facebook 的配置管理框架，支持 YAML 组合、命令行覆盖和组件化实例化 |
| **GIL** | Global Interpreter Lock — Python 的全局解释器锁 |
| **Pink** | 基于 Pinocchio 的 task-driven IK 库，使用 QP 求解器进行多目标优化 |
| **QP** | Quadratic Programming — 二次规划，Pink 用于求解带约束的 IK 优化问题 |
| **SLERP** | Spherical Linear Interpolation — 球面线性插值，用于四元数旋转平滑插值 |
| **SharedStorage** | ManiUniCon 的无锁共享内存系统，包含 RingBuffer (FILO) + Queue (FIFO) + 原子计数器 |

### B. 参考框架源码目录树摘要

#### LeFranX 关键目录
```
LeFranX/
├── franka_server/src/          ← C++ 实时控制 (libfranka + Ruckig)
│   └── franka_server.cpp       ← 1kHz control loop, TCP server
├── franka_xhand_teleoperator/src/
│   ├── weighted_ik.cpp         ← WeightedIKSolver (Brent q7 opt)
│   ├── geofik.cpp              ← Analytic IK for Franka 7-DOF
│   └── vr_message_router.cpp   ← TCP VR receiver (regex parse)
├── src/lerobot/
│   ├── robots/
│   │   ├── franka_fer/         ← Franka FER robot interface
│   │   └── xhand/              ← XHand robot interface
│   └── teleoperators/
│       ├── franka_fer_vr/      ← Arm VR teleop pipeline
│       └── xhand_vr/           ← Hand VR teleop pipeline (DexPilot + pinky adapter)
└── vr-dex-retargeting/         ← Git submodule (dex_retargeting fork)
```

#### BunnyVision Pro 关键目录
```
BunnyVisionPro/
├── bunny_teleop/
│   ├── bimanual_teleop_server.py  ← ZMQ PUB (5500) + REP (5501) server
│   ├── bimanual_teleop_client.py  ← ZMQ SUB + REQ client
│   └── init_config.py             ← BimanualAlignmentMode, InitializationConfig
├── real_control/
│   ├── xarm7_ability.py           ← XArm7 + Ability Hand control (DLS IK, PID velocity)
│   └── teleop_bimanual_xarm7_ability.py ← Main teleop loop + HDF5 recording
├── examples/
│   └── retargeting/
│       └── retargeting.py         ← Offline SeqRetargeting (OPERATOR2MANO matrices)
└── docs/
    ├── system/overview.md
    └── advanced/initialization.md
```

#### Open-Teach 关键目录
```
Open-Teach/
├── teleop.py                      ← Main entry (Hydra @hydra.main)
├── data_collect.py                ← Data collection entry
├── configs/
│   ├── teleop.yaml                ← Defaults: network + robot
│   ├── network.yaml               ← All ZMQ port definitions
│   └── robot/                     ← Franka, Kinova, Bimanual, Allegro configs
├── openteach/
│   ├── components/
│   │   ├── detector/
│   │   │   ├── oculus.py          ← OculusVRHandDetector (ZMQ PULL→PUB)
│   │   │   └── keypoint_transform.py ← Coordinate frame transform
│   │   ├── operators/
│   │   │   ├── franka.py          ← FrankaArmOperator (Cartesian servo)
│   │   │   ├── kinova.py          ← KinovaArmOperator (velocity control)
│   │   │   ├── bimanual_right.py  ← Bimanual XArm operator
│   │   │   └── allegro.py         ← Allegro hand operator (KDL IK)
│   │   ├── recorders/
│   │   │   ├── robot_state.py     ← Robot state HDF5 recorder
│   │   │   └── image.py           ← Camera HDF5 + AVI recorder
│   │   └── initializers.py        ← Process spawning (5+ processes)
│   ├── robot/                     ← Robot wrappers (abstract RobotWrapper)
│   ├── ros_links/                 ← ROS interfaces to real hardware
│   └── utils/network.py           ← ZMQKeypointPublisher/Subscriber
└── VR/                            ← Unity C# Quest app (NetMQ + OVR)
```

#### ManiUniCon 关键目录
```
maniunicon/
├── main.py                             ← 163-170 RobotControlSystem
├── core/
│   └── robot.py                        ← 174-220 state_receiver thread
│                                       ← 279-466 run() control loop
│                                       ← 405-437 interpolator scheduling
├── policies/
│   └── quest.py                        ← 190-377 QuestPolicy.run()
├── utils/
│   ├── quest_controller.py             ← 46-53  R_ve construction
│   │                                   ← 134-158 _check_safety_limits()
│   │                                   ← 160-190 _apply_safety_limits()
│   │                                   ← 192-338 _calculate_action()
│   ├── ik_solver.py                    ← 103-124 _create_tasks()
│   │                                   ← 171-217 solve()
│   ├── pose_trajectory_interpolator.py ← 78-101 drive_to_waypoint()
│   │                                   ← 103-185 schedule_waypoint()
│   │                                   ← 187-207 __call__()
│   ├── filter.py                       ← 77-138 JointSpaceSmoother.smooth()
│   └── shared_memory/
│       ├── shared_storage.py           ← 382    write_action()
│       │                               ← 404    read_all_action()
│       └── shared_memory_queue.py      ← 88-107 put()
│                                       ← 140-149 get_all()
├── robot_interface/
│   └── xarm6_robotiq.py                ← 226-251 Cartesian IK in hot path
└── configs/
    ├── default.yaml                    ← 1-8   frequencies and buffer
    ├── robot/xarm6.yaml                ← 1-48  robot config
    └── policy/quest.yaml               ← 1-13  policy config
```

### C. dexmani 完整文件地图

```
dexmani_real/
├── teleop/
│   ├── core/
│   │   ├── controller.py          ← TeleopController: 主循环, 状态机, _tick()
│   │   ├── error_handler.py       ← TeleopErrorHandler: hold-on-failure
│   │   └── tracking.py            ← TrackingQuality: VR 帧新鲜度 + 丢失检测
│   ├── vr/
│   │   ├── vr_tracker.py          ← QuestHandTracker: HTS SDK 接收 (daemon thread)
│   │   ├── arm_mapper.py          ← ArmWristMapper: reset-relative wrist→EEF
│   │   ├── hand_retarget.py       ← XHandRetargeter: DexPilot + XHandRefAdapter
│   │   ├── ref_adapter.py         ← XHandRefAdapter: pinky scaling (LeFranX 同源)
│   │   └── dummy_tracker.py       ← DummyTracker: 测试用虚拟 VR 数据
│   ├── control/
│   │   ├── safety.py              ← 安全检查函数 (torque/current/temp/limits)
│   │   └── keyboard.py            ← KeyboardHandler: 键盘控制信号
│   └── __init__.py
├── planning/
│   ├── ik.py                      ← TeleopIKSolver: DLS (primary) + MPlib (fallback)
│   ├── ik_candidates.py           ← IKCandidateManager: IK 候选生成/过滤/评分
│   ├── kinematics.py              ← XArm7Kinematics: FK, Jacobian, manipulability
│   ├── planner.py                 ← XArm7MotionPlanner: IK, 路径规划, FingertipDeskSafety
│   ├── planner_interface.py       ← RobotPlanner: 统一规划接口
│   ├── types.py                   ← Pose, IKResult, TeleopProfile, PlanningProfile, XArm7PlannerConfig
│   ├── collision_config.py        ← CollisionConfig: 碰撞检测配置
│   ├── collision_links.py         ← 碰撞 link ID 常量
│   └── pose_utils.py              ← 姿态工具函数
├── robot/
│   ├── interface.py               ← RobotInterface: 统一机器人接口
│   ├── types.py                   ← RobotState, RobotAction, RobotInterfaceConfig
│   └── xarm7/ (xhand/)            ← 硬件驱动
├── recording/
│   ├── episode_recorder.py        ← EpisodeRecorder: HDF5 录制
│   ├── quality_flags.py           ← QualityFlags: 10bit 质量标记
│   └── recorder_config.py         ← RecorderConfig
├── sensor/                        ← RealSense 相机驱动
├── simulation/                    ← SAPIEN 仿真
├── config/                        ← CameraCalib, PipelineConfig
├── utils/
│   ├── rate_limiter.py            ← RateLimiter: 50Hz 帧率控制
│   ├── signal_utils.py            ← ema_smooth: 指数移动平均
│   └── hand_utils.py              ← estimate_frame_from_hand_points, OPERATOR2MANO_RIGHT
└── __init__.py
```

---

> **文档版本**: v2.0 | **最后更新**: 2026-06-22
> **分析覆盖率**: dexmani_real (17 文件深度阅读), LeFranX (15 文件), BunnyVision Pro (12 文件), Open-Teach (18 文件), ManiUniCon (16 文件)
