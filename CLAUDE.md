# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Python 环境**: `source /home/zhy/anaconda3/etc/profile.d/conda.sh && conda activate real`（`/home/zhy/anaconda3/envs/real/bin/python`）

## 项目简介

dexmani_real 是一个 xArm7 + XHand 灵巧操作机器人遥操作与数据采集系统。VR 输入通过 Meta Quest HTS SDK，支持仿真（SAPIEN）和真机两种模式。

## 开发进度

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 0 | 文档修正（6 项） | ✅ |
| Phase 1 | P0 安全修复（4 项） | ✅ |
| Phase 2 | P1 运动质量（6 项） | ✅ |
| Phase 3 | P2 架构增强（4+4 项） | ✅ |
| Phase 4 | P3 工程优化（3 项） | ✅ |
| Phase 5 | 集成收尾 & 生产就绪（6 项） | ✅ 2026-06-23 |
| Phase 6 | 高级特性（可选） | 📋 待排期 |
| Phase 7 | T-Rex → dexmani 优化移植（10 项） | ✅ 2026-06-23 |

### Phase 7 实施内容（2026-06-23）— T-Rex → dexmani 优化移植

**Phase 7.1 — 高 ROI（4 项）**
- **追踪安全误差监控**: controller.py `_tick()` 新增 command-vs-actual 偏差检查（|q_actual - q_cmd| > 5rad × 3 帧 → E-Stop），`TeleopControllerConfig.tracking_divergence_threshold_rad`
- **Episode 元数据富化 & 文件分类**: `CollectionLoop.stop_episode()` 写入 sidecar JSON（frame_count/duration/task_label/classification/ik_success_rate/vr_drop_rate），按 success/failure 路由到子目录，`CollectionConfig.success_dir/failure_dir/save_sidecar_json`
- **Resize-First 图像处理顺序（确认无需改动）**: RealSense SDK 在流配置层面指定分辨率，无软件 cv2.resize 调用；BGR→RGB 是唯一颜色转换（通道翻转）
- **Path shortcut 算法（确认已实现）**: `planner.py:366 shortcut_smooth_path()` 已实现贪心 shortcut + 密集碰撞检测（3 pass），与 T-Rex 等价

**Phase 7.2 — 中 ROI（4 项）**
- **速度限制平滑 + 碰撞回退**: `controller.py:_apply_velocity_limited_step()` 瓶颈缩放（复用 `_limit_joint_step` 算法），`TeleopControllerConfig.use_velocity_limited_smooth`
- **速率解耦 IK miss 计数器**: `_ik_miss_count` 追踪连续 IK 失败帧数，≥3 帧时降级警告；buffer 隔离 IK 抖动与指令发送
- **cbreak 模式键盘**: `keyboard.py` 用 termios cbreak + select 替代 pynput + multiprocessing.Queue，`KeyboardHandler.__enter__/__exit__` 确保终端恢复
- **None-Sentinel 动作缓冲区**: `XArm7.clear_target()` 设 PID 目标为 None → 内环发送零速度自然减速（velocity 模式）；PAUSED/软减速时使用

**Phase 7.3 — Nice-to-Have（2+1 项）**
- **预录制缓冲区**: `CollectionLoop.add_pre_frame()` + `collections.deque` ring buffer，`CollectionConfig.pre_record_duration_s`，start_episode 时 flush 到 HDF5
- **VR 帧模拟器**: `vr_tracker.py:VRFrameSimulator` — 正弦手腕轨迹 + 刚性手部 landmark，兼容 QuestHandTracker API
- ZMQ 相机流暂缓（SharedMemoryRingBuffer 延迟更低）

### Phase 5 实施内容（2026-06-23）
- CollectionLoop ↔ TeleopController 集成缝合
- 多相机集成到控制回路（MultiCameraManager + HDF5 per-camera paths）
- auto_stop_on_quality_drop（连续低质量帧自动停止）
- Episode sidecar annotation JSON（stop_episode 时写入 metadata）
- 仿真端到端验证（待手动运行）
- 真机测试脚本完善（scripts/real/ 受权限保护）

## 常用命令

### 仿真模式

```bash
# VR 遥操作仿真（dummy VR 数据，无需头显）
python scripts/sim/vr_teleop_sim.py --dummy

# VR 遥操作仿真（真实 Quest VR）
python scripts/sim/vr_teleop_sim.py

# 无头模式 + 指定数据目录
python scripts/sim/vr_teleop_sim.py --dummy --headless --data-dir ./my_episodes

# 键盘直接控制仿真（无 VR）
python scripts/sim/keyboard_teleop_sim.py

# VR 帧模拟器（正弦轨迹，无需 Quest 设备）
python scripts/sim/vr_teleop_sim.py --dummy --dummy-vr-sinusoidal
```

### 真机模式

```bash
# 真机遥操作（需连接 xArm7 + XHand + Quest VR）
python scripts/real/keyboard_teleop_real.py
```

### 运动规划测试

```bash
python scripts/sim/test_motion_planning_sim.py
```

### 代码格式化

```bash
black dexmani_real/ scripts/ --line-length 120
isort dexmani_real/ scripts/
```

## 架构总览

### 控制回路数据流

```
Meta Quest (HTS TCP, ~50Hz)
  │ 21 hand landmarks + wrist pose
  ▼
QuestHandTracker (daemon thread)
  │ get_latest() → vr_frame dict
  ▼
TeleopController._tick() (主线程, 50Hz)
  │
  ├─ 1. _read_vr_frame()          ← VR 帧 + 跟踪质量检查（四层分级）
  ├─ 2. TeleopPipeline.compute_action()
  │     ├─ ArmWristMapper.map()    ← reset-relative wrist→EEF delta
  │     ├─ CartPoseInterpolator    ← 可选，频率解耦（关闭默认）
  │     ├─ TeleopIKSolver.solve()  ← DLS primary → MPlib fallback
  │     ├─ ema_smooth (arm only)   ← 手部平滑由 dex-retargeting 内置
  │     ├─ XHandRetargeter.retarget() ← DexPilot + XHandRefAdapter
  │     ├─ joint jump clamp (5°/frame arm, 10°/frame hand)
  │     └─ 11-bit QualityFlags     ← 每帧质量标记（含 CAMERA_OK）
  ├─ 3. safety checks              ← torque/current/temp/comm/workspace/desk-FK
  ├─ 4. robot.send_action()        ← 伺服模式 / PID 速度模式
  └─ 5. CollectionLoop.add_frame() ← HDF5 录制 + 质量标记
```

### 关键设计决策

- **单步 DLS（非迭代）**: damping=0.02（λ²=0.0004），单次 Jacobian 伪逆。相比 BunnyVisionPro 的迭代 DLS（λ²=1e-5, 100 次迭代），牺牲 ~1-2mm 精度换取 <1ms 延迟。人手震颤（~2-3mm）主导误差，此方案合理。
- **自适应 damping**（`adaptive_damping=True`，默认开启）: 根据 manipulability 线性调整 damping（非奇异区 min=0.001，近奇异区 max=0.05），无需额外 FK 开销（Jacobian 复用）。
- **双 IK 策略**: DLS（确定性，<1ms）→ MPlib Position IK（随机，~10ms）→ hold。选中 hardware-closest 候选避免分支跳变。
- **速度限制在驱动层**: `XArm7._limit_joint_step()` bottleneck scaling。IK/planning 层不裁剪速度，职责分离（参考 BunnyVisionPro 架构）。
- **Hold-on-failure**: 任何管道失败返回 `last_good_position`，不发送危险指令。

### 四层安全模型

| 层 | 位置 | 检查项 |
|----|------|--------|
| 驱动层 | `xarm7.py`, `xhand.py` | 力矩裁剪、关节限位、step bottleneck、C31/C32 恢复 |
| 接口层 | `interface.py`, `safety.py` | workspace bounds、力矩/电流/温度/通信状态 |
| 控制器层 | `controller.py`, `pipeline.py` | jump clamp、VR 帧新鲜度、quality flags |
| 路径层 | `planner.py` | FingertipDeskSafety（桌面 FK 碰撞）、自碰撞检测 |

### 状态机

```
IDLE ──T/Enter──→ TELEOP ──R──→ RECORDING ──S──→ TELEOP
  │                  │              │
  H                  H              H
  │                  │              │
  ▼                  ▼              ▼
return_to_home    EMERGENCY_STOP (ESC / timeout)
```

- **C 键（Pause/Resume）**: 暂停时冻结 EEF、暂停录制；恢复时自动 re-anchor mapper 抵消暂停期间的漂移
- **SAVE_PROMPT 状态**: Q 退出时提示 S（保存）/ N（丢弃）当前 episode

### xArm7 控制模式

- **Mode 1（伺服，默认）**: `set_servo_angle_j` 直接位置指令。简单可靠。
- **Mode 4（速度，PID 内环）**: `vc_set_joint_velocity` @ 250Hz。设置 `XArm7Config.use_servo_control=False` 启用。PID 将位置误差转为连续速度，运动更平滑。参考 BunnyVisionPro `xarm7_ability.py`。

### 录制系统

- **格式**: HDF5（episode_NNN.h5）+ Episode sidecar JSON（episode_NNN.json）
- **质量标记**: 每帧 11-bit QualityFlags（TRACKING/IK/RETARGET/RETARGET_VALID/JUMP/WORKSPACE/CAMERA_OK/TORQUE/CURRENT/TEMP/COMM），支持训练时过滤低质量帧
- **多相机**: MultiCameraManager 集成到控制回路，HDF5 per-camera paths（`/camera/<serial>/rgb` + `/camera/<serial>/depth`）
- **生命周期**: CollectionLoop 管理 start/stop/auto_stop_on_quality_drop

## 关键文件速查

| 文件 | 职责 |
|------|------|
| `dexmani_real/teleop/core/controller.py` | TeleopController: 主循环、状态机、_tick() |
| `dexmani_real/teleop/core/pipeline.py` | TeleopPipeline: 无状态 action 计算（实机/仿真共用）|
| `dexmani_real/teleop/core/tracking.py` | TrackingQuality + FrameDropPolicy: 四层帧新鲜度分级 |
| `dexmani_real/teleop/vr/vr_tracker.py` | QuestHandTracker: HTS SDK 接收（daemon 线程）|
| `dexmani_real/teleop/vr/arm_mapper.py` | ArmWristMapper: reset-relative wrist→EEF 映射 |
| `dexmani_real/teleop/vr/hand_retarget.py` | XHandRetargeter: DexPilot + XHandRefAdapter |
| `dexmani_real/planning/ik.py` | TeleopIKSolver: DLS (primary) + MPlib Position IK (fallback) |
| `dexmani_real/planning/kinematics.py` | XArm7Kinematics: FK, Jacobian, manipulability |
| `dexmani_real/planning/planner.py` | XArm7MotionPlanner: IK, 路径规划, FingertipDeskSafety |
| `dexmani_real/planning/types.py` | Pose, IKResult, TeleopProfile, PlanningProfile, XArm7PlannerConfig |
| `dexmani_real/robot/xarm7/xarm7.py` | XArm7: 硬件驱动（伺服/PID 双模式，250Hz 内环）|
| `dexmani_real/robot/interface.py` | RobotInterface: 统一机器人接口 |
| `dexmani_real/recording/episode_recorder.py` | EpisodeRecorder: HDF5 录制生命周期 |
| `dexmani_real/recording/collection_loop.py` | CollectionLoop: 录制状态机 + auto_stop + sidecar JSON |
| `dexmani_real/recording/quality_flags.py` | QualityFlags: 11-bit 质量标记（含 CAMERA_OK bit 6） |
| `dexmani_real/teleop/control/safety.py` | 安全检查函数（torque/current/temp/limits）|
| `dexmani_real/teleop/control/keyboard.py` | KeyboardHandler: cbreak 模式键盘（termios + select）|
| `dexmani_real/teleop/vr/vr_tracker.py` | QuestHandTracker + VRFrameSimulator（正弦轨迹测试）|
| `dexmani_real/sensor/multi_camera_manager.py` | MultiCameraManager: 多相机进程管理 |

TeleopProfile 配置参数（`planning/types.py:112-188`）：所有 IK/servo/插值参数均在此 dataclass 中，支持 `FromDictMixin` 从 YAML/JSON 加载。仿真配置文件示例见 `configs/` 目录。

## 参考框架与设计来源

dexmani 的设计参考了五个同类框架，详细对比见 `docs/vr-teleop-framework-comparison.md`：

| 框架 | 主要借鉴 |
|------|----------|
| BunnyVision Pro | DLS IK、250Hz PID 速度内环、clip_arm_velocity、FrameAge gate |
| LeFranX | manipulability 评分、hardware-closest IK 候选、position IK fallback、XHandRefAdapter |
| Open-Teach | ZMQ 进程分离（可选）、VR 订阅进程 |
| ManiUniCon | PoseTrajectoryInterpolator（Cartesian 插值频率解耦）|
