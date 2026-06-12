# xArm7 + XHand VR 遥操作数据采集与策略部署开发计划

> **文件定位**：本文档是本项目的具体实施方案（工期、文件清单、验收标准）。所有实现必须遵循 CLAUDE.md 中的接口规范。
> 
> **与 CLAUDE.md 的映射**：

| 开发日 | 模块 | CLAUDE.md Section |
|--------|------|-------------------|
| Day 1 | RobotInterface + 配置 + 安全 | Section 2 (robot), Section 9 (配置) |
| Day 2 | IPC + TeleopController + 平滑 | Section 3 (IPC), Section 4 (controller) |
| Day 3 | HDF5 录制 + Episode 生命周期 | Section 5 (recording) |
| Day 4 | EpisodeReader + 数据验证 | Section 6 (data) |
| Day 5 | PolicyRunner + Action Chunk | Section 7 (deploy) |
| Day 6 | 延迟测量 + 集成测试 | Section 8 (utils) |

> **架构图详见 CLAUDE.md Section 0**（进程数据流、Episode 生命周期、状态机）。

## 概述

当前 `dexmani_real` 已具备底层驱动（xArm7 伺服、XHand 12-DOF、MPlib IK、Quest VR 追踪、DexPilot 重定向、RealSense 相机），缺少**真机实时控制闭环**和**同步多模态数据采集管线**。

本计划参考 5 个外部项目的核心设计（优先级 P1→P2→P3）：

**P1 — 架构方法论（决定"怎么做架构"）：**
- **LeFranX**：VR re-anchoring、episode 管理、hand 平滑、action chunk 部署
- **ManiUniCon**：多进程 shared_memory、数据质量过滤、IK solver

**P2 — 硬件控制模式（决定"怎么写硬件代码"）：**
- **BunnyVisionPro**：xArm7 + Ability Hand 真机遥操作、双手协调、dex-retargeting 重定向
- **Open-Teach**：多机器人抽象模式、VR 关键点检测、组件生命周期

**P3 — 补充场景（仅特定场景查阅）：**
- **Bidex_Manus_Teleop**：Manus 数据手套数据解析、PyBullet SDLS IK retargeting

不涉及策略训练代码。

> 详细的优先级检索协议（逐级检索、P2 选择规则、Fact-Check 规则）见 CLAUDE.md Section 0.5。

### LeFranX 中直接采纳的设计

| LeFranX 模式 | 采纳方式 |
|-------------|---------|
| VR reference re-anchoring（每 episode 重置） | TeleopController 进入 RECORDING 时调用 `arm_mapper.reset()` |
| Hand action smoothing（alpha=0.3） | Day 2 `_tick()` 仅对 hand_cmd 做指数平滑（arm 不平滑） |
| Episode 生命周期（disconnect→home→reconnect→reset→record） | Day 3 `EpisodeRecorder.start_episode()` 触发复位流程 |
| Action chunk 执行策略（query_freq + n_action_steps） | Day 5 PolicyRunner 支持 chunk 模型输出 |
| 手动加载模型（config.json + safetensors/stats.json） | Day 5 `PolicyLoader` 工具类 |
| 单 VR 进程服务 arm+hand（singleton router） | 架构中 VR 进程只维护一个 SharedRingBuffer |

### LeFranX 中不采纳的设计

| 模式 | 理由 |
|------|------|
| C++ 实时 server（libfranka + Ruckig） | xArm7 已通过 XArmAPI 内置伺服控制，不需要额外 server |
| 解析式 IK（geofik + Brent） | 已有 MPlib 数值 IK，且支持碰撞检测（Franka 解析 IK 不支持） |
| LeRobot Parquet 数据集 | 使用 HDF5，更通用，训练代码不依赖 LeRobot |
| LeRobot draccus CLI | 不引入该框架依赖 |

---

## 架构总览

> 以下架构图与 CLAUDE.md Section 0 保持同步。修改架构时需同时更新两个文件。

### 进程与数据流

```
VR 进程(80Hz) ──► ring["vr_frame"] ──┐
                                     ├── Controller 进程(50Hz)
Camera 进程(30Hz) ──► ring["camera"] ─┘     │
                                            ├─ _tick(): 同一份 VR frame 同时驱动 arm + hand
                                            │    arm: IK(pose, qpos, prev) → arm_cmd
                                            │    hand: retarget(landmarks) → EMA(alpha=0.3) → hand_cmd
                                            ├─ robot.send_action()
                                            └─ [RECORDING] recorder.add_frame()

Keyboard ──► multiprocessing.Queue ──► 控制信号(T/R/S/H/ESC)
--no-ipc 回退到单进程
```

### Episode 生命周期（LeFranX 风格）

```
for each episode:
  1. robot.return_to_home()           # 路径规划回 home + hand 复位
  2. arm_mapper.reset(vr_frame, eef)  # re-anchor VR reference
  3. recorder.start_episode()         # 创建 .h5
  4. [teleop 录制...]
  5. recorder.stop_episode()          # 写入 .h5，存入 norm_stats
```

### 控制器状态机

```
启动 → IDLE ──T──► TELEOP ──R──► RECORDING ──S──► IDLE
         │            │               │
         H            S               │ 追踪丢失>1s / IK连败>10
         │            │               │
         ▼            ▼               ▼
    return_to_home   IDLE         EMERGENCY_STOP → 退出
    (规划路径)
```

---

## Day 1：RobotInterface + 配置 + 安全 + 路径回 home

**工时** 4-5h | **P0 | 参考: [P1] ManiUniCon xarm6_robotiq/base, [P1] LeFranX franka_fer_xhand, [P2] BunnyVisionPro xarm7_ability**

### 文件

| 文件 | 说明 |
|------|------|
| `robot/__init__.py` | 导出 RobotInterface, RobotState, RobotAction |
| `robot/robot_interface.py` | arm+hand 统一接口 |
| `robot/workspace_safety.py` | workspace 安全 |
| `config/__init__.py` | |
| `config/pipeline_config.py` | 全 pipeline 配置 |

### 关键 API

```python
@dataclass
class RobotState:
    arm_qpos(7), arm_qvel(7), arm_tau(7), eef_pos(3), eef_quat_wxyz(4)
    hand_qpos(12), hand_current(12), hand_tactile_sum(5,3), hand_temperature(12)
    arm_connected, hand_connected, hand_error, timestamp

@dataclass
class RobotAction:
    arm_qpos_cmd(7), hand_qpos_cmd(12), target_eef_pose(7)|None

class RobotInterface:
    def connect(self) -> dict[str, bool]: ...     # {"arm": True, "hand": False}
    def return_to_home(self, use_planning=True, cancel_event=None) -> bool: ...
        # planner.plan_path(target_home, current) → 成功执行路径 → fallback reset()
        # 后跟 reset_hand()
    def reset_hand(self) -> bool: ...
    def get_state(self) -> RobotState: ...         # FK + 低通滤波
    def send_action(self, action: RobotAction) -> dict: ...
    def emergency_stop(self) -> None: ...
```

### 验收

- hand 断连降级运行；`return_to_home()` 路径规划逐点执行；规划失败 fallback 直线 reset；workspace 违规检测

---

## Day 2：IPC + TeleopController + 手部平滑 + 追踪质量 + 错误恢复

**工时** 6-8h | **P0 | 参考: [P1] LeFranX franka_fer_xhand_vr, [P1] ManiUniCon shared_memory + quest, [P2] BunnyVisionPro teleop_bimanual_xarm7_ability, [P2] Open-Teach operator**

### 文件

| 文件 | 说明 |
|------|------|
| `ipc/shared_ring_buffer.py` | seq_num 协议的 ring buffer |
| `controller/teleop_controller.py` | 主循环 + hand EMA 平滑 |
| `controller/tracking_quality.py` | 追踪 + retarget 质量检查 |
| `controller/error_handler.py` | hold-on-failure |
| `controller/keyboard_handler.py` | Queue 键盘事件 |
| `scripts/run_teleop.py` | |

### _tick 核心逻辑

```python
def _tick(self, vr_frame):
    # 0. 追踪质量门控
    quality = self.quality_checker.check(vr_frame)
    if quality.level == QualityLevel.REJECT:
        return self.error_handler.handle("tracking_lost")

    # 1. Arm: VR wrist → EEF → IK（不平滑，保留原始追踪）
    target = self.arm_mapper.map(vr_frame["wrist_pos"],
                                  vr_frame["wrist_quat_wxyz"])
    if target is not None:
        pose = Pose(p=target["pos"], q=target["quat_wxyz"])
        ik = self.planner.solve_teleop_ik(pose, self._prev_arm_cmd, self._prev_arm_cmd)
        if ik and ik.qpos is not None:
            arm_cmd = ik.qpos
            self._prev_arm_cmd = arm_cmd
        else:
            arm_cmd = self.error_handler.handle("ik_failed")
    else:
        arm_cmd = self.error_handler.handle("mapper_failed")

    # 2. Hand: landmarks → retarget → EMA 平滑（LeFranX: alpha=0.3）
    landmarks = vr_frame["landmarks"]
    mano_rot = estimate_frame_from_hand_points(landmarks)
    landmarks_mano = landmarks @ mano_rot @ OPERATOR2MANO_RIGHT
    raw_hand = self.hand_retargeter.retarget(landmarks_mano)

    if raw_hand is not None and self._is_hand_cmd_valid(raw_hand):
        # EMA: cmd = alpha * raw + (1-alpha) * prev
        hand_cmd = (self.hand_smooth_alpha * raw_hand +
                    (1 - self.hand_smooth_alpha) * self._prev_hand_cmd)
        self._prev_hand_cmd = hand_cmd
    elif raw_hand is not None:
        quality.add_flag(QualityFlags.RETARGET_INVALID)
        hand_cmd = self._prev_hand_cmd
    else:
        hand_cmd = self._prev_hand_cmd

    # 3. 关节跳变 clamp (arm + hand 各自独立限速)
    arm_cmd = self._clamp_jump(arm_cmd, self._prev_arm_cmd, self.arm_max_step)
    hand_cmd = self._clamp_jump(hand_cmd, self._prev_hand_cmd, self.hand_max_step)

    return RobotAction(arm_qpos_cmd=arm_cmd, hand_qpos_cmd=hand_cmd)
```

**为什么仅 hand 做平滑而 arm 不做？** LeFranX 的实践表明：arm IK 的输出本身已经平滑（IK solver 有内置限速），而 hand retargeting 从 21 个稀疏 landmark 映射到 12 DOF 关节角，输入噪声大，需要 EMA 过滤手指抖动。arm 不平滑是为了在录制数据中保留原始 VR→robot 映射关系，供策略学习真实的 arm 动态。

**VR re-anchoring**：`arm_mapper.reset(vr_wrist, eef)` 在进入 TELEOP 或 RECORDING 时调用，以当前 VR 手位和机器人 EEF 位姿为新的参考原点，消除累积偏移。

### 验收

- hand 重定向超限 → hold + flag；hand 关节平滑无抖动
- VR re-anchoring: 进入 RECORDING 前重置参考原点
- 追踪丢失 hold → 超时急停；kill camera 不崩溃

---

## Day 3：HDF5 录制 + Episode 生命周期 + 质量标记

**工时** 5-7h | **P1 | 参考: [P1] LeFranX dual_vr_record, [P1] ManiUniCon replay_buffer, [P2] BunnyVisionPro teleop_bimanual_xarm7_ability**

### 文件

| 文件 | 说明 |
|------|------|
| `recording/episode_recorder.py` | HDF5 录制器 |
| `recording/camera_recorder.py` | 后台相机进程 |
| `recording/quality_flags.py` | 11-bit 质量标记 |
| `utils/camera_calib.py` | 内外参加载（CameraCalib） |
| `config/calib/cameras.json` | 相机标定数据（外部标定脚本产出） |

### Episode 生命周期（LeFranX 风格）

```
TeleopController 中按 R 键后的流程:
  1. robot.return_to_home(use_planning=True)
  2. vr_frame = tracker.get_latest()
  3. arm_mapper.reset(vr_frame["wrist_pos"], robot.get_state().eef_pos)
     # ↑ LeFranX: reset_initial_pose(), 以当前位姿为 VR 参考原点
  4. calib = CameraCalib("config/calib/cameras.json")  # 加载标定
  5. recorder.start_episode(task_label, operator, tags, calib)
     # ↑ 将 camera_serial, camera_type, camera_K, camera_T_* 写入 /meta
  6. [在 TELEOP 状态下录制每一帧...]
     # ↑ add_frame() 内逐帧计算 /camera/extrinsics[t]
  7. 按 S 键 → recorder.stop_episode(success=True/False)
     # 写入 .h5 + norm_stats + meta
```

**为什么要 per-episode re-anchoring？** LeFranX 的实践中，操作员在 episode 之间会自然移动 VR 手的位置。如果不重置参考原点，新 episode 开始时机器人的初始位姿会与 VR 参考产生大跳变，导致 IK 失败或危险运动。

### HDF5 结构

```
episode_000.h5
  /meta: task_label, operator, tags, duration, fps,
         num_frames, num_valid_frames, success,
         camera_serial, camera_type,          # "eye_to_hand" | "eye_in_hand"
         camera_K,                            # [fx, fy, cx, cy]
         camera_T_base_camera | camera_T_eef_camera,  # 4x4 flat，标定原始值
         retargeting_config, pipeline_snapshot

  /obs/arm_qpos(7)  arm_qvel(7)  arm_tau(7)  eef_pos(3)  eef_quat(4)
  /obs/hand_qpos(12)  hand_current(12)  hand_tactile_sum(5,3)  hand_temperature(12)
  /action/arm_qpos(7)  hand_qpos(12)
  /vr/wrist_pos(3)  wrist_quat(4)  landmarks(21,3)
  /quality_flags(T,) uint16
  /camera/rgb(T,H,W,3)  depth(T,H,W)  timestamps(T)
  /camera/K(3,3)                        # 内参矩阵
  /camera/extrinsics(T,4,4)             # T_base_camera，逐帧外参
```

**相机内外参**：

标定由外部脚本完成，结果写入 `config/calib/cameras.json`。`CameraCalib` 在 `start_episode()` 时加载，数值直接写入 HDF5 meta（不存文件路径）。`add_frame()` 逐帧计算 `/camera/extrinsics`：

```
external calib script → config/calib/cameras.json
                             │
start_episode() ──► CameraCalib.load() ──► /meta camera_*
                             │
add_frame() ──► eye-to-hand: T_base_camera (static)
                eye-in-hand: FK(arm_qpos) @ T_eef_camera (per-frame)
                             │
                             ▼
                /camera/extrinsics[t] = T_base_camera (4x4)
```

| 字段 | 来源 | 说明 |
|------|------|------|
| `/meta/camera_serial` | cameras.json | 相机序列号 |
| `/meta/camera_type` | cameras.json | `"eye_to_hand"` 或 `"eye_in_hand"` |
| `/meta/camera_K` | cameras.json | `[fx, fy, cx, cy]` |
| `/meta/camera_T_base_camera` | cameras.json | eye-to-hand 时存，4x4 flat |
| `/meta/camera_T_eef_camera` | cameras.json | eye-in-hand 时存，4x4 flat |
| `/camera/K(3,3)` | cameras.json | 3x3 内参矩阵 |
| `/camera/extrinsics(T,4,4)` | 逐帧计算 | T_base_camera |

### EpisodeRecorder 接口

```python
class EpisodeRecorder:
    def start_episode(self, task_label: str = "", operator: str = "",
                      tags: list[str] | None = None,
                      calib: CameraCalib | None = None) -> bool: ...
        """创建 .h5，写入 meta（含相机标定数值）。"""

    def add_frame(self, state: RobotState, action: RobotAction,
                  vr_frame: dict, quality_flags: int,
                  camera_frame: dict | None = None,
                  T_base_eef: np.ndarray | None = None) -> bool: ...
        """追加一帧。eye-in-hand 时需要 T_base_eef 计算外参。"""
```

### QualityFlags

```python
TRACKING_OK(0), IK_SUCCESS(1), RETARGET_OK(2), RETARGET_VALID(3)
JOINT_JUMP_OK(4), IN_WORKSPACE(5), CAMERA_OK(6), ARM_TORQUE_OK(7)
HAND_CURRENT_OK(8), HAND_TEMP_OK(9), HAND_COMM_OK(10)
```

### 验收

- 按 R: home → 加载标定 → re-anchor → 录制开始
- 按 S: 写入 .h5 + meta 含 camera_K/camera_T_*/retargeting_config
- 连录 3 个 episode，每个 episode 独立 re-anchor，无位姿跳变
- eye-to-hand 外参逐帧一致；eye-in-hand 外参随 arm 运动变化

---

## Day 4：EpisodeReader + 数据验证 + 格式转换

**工时** 3-5h | **P1 | 参考: CLAUDE.md Section 6 接口规范，Open-Teach data_collect 数据组织参考**

### 文件

| 文件 | 说明 |
|------|------|
| `data/episode_reader.py` | 懒加载 HDF5 |
| `data/episode_replayer.py` | Rerun + hand skeleton |
| `data/data_validator.py` | 完整性验证 |
| `scripts/convert_data.py` | 训练格式 + per-joint 归一化 |
| `scripts/replay_episode.py` `scripts/validate_data.py` | |

### 关键 API

```python
class EpisodeReader:
    def read(self, key: str) -> np.ndarray: ...    # "obs/arm_qpos"
    def iter_frames(self, skip_rejected=True) -> Iterator[dict]: ...
    def get_valid_mask(self) -> np.ndarray: ...
    @property
    def num_frames / num_valid_frames / metadata: ...

class DataValidator:
    def validate(self, episode_path) -> ValidationReport: ...
    # 检查: nan/inf, shape 一致, 时间戳单调, 关节范围, 电流异常

# convert_data.py --norm-stats 输出 per-joint 归一化:
#   arm_qpos: {mean: [j0..j6], std: [j0..j6]}
#   hand_qpos: {mean: [j0..j11], std: [j0..j11]}
```

### 验收

- 回放含 hand skeleton；DataValidator 通过；per-joint 归一化正确

---

## Day 5：PolicyRunner + Action Chunk + SafetyMonitor

**工时** 5-7h | **P1 | 参考: [P1] LeFranX robot_client + deploy_act, [P1] ManiUniCon chunk_wrapper + torch_model, [P2] Open-Teach deploy_server**

### 文件

| 文件 | 说明 |
|------|------|
| `deploy/policy_runner.py` | 策略部署 + chunk 处理 |
| `deploy/observation_builder.py` | 观测构建 + 归一化 |
| `deploy/action_parser.py` | arm_only/hand_only/full |
| `deploy/safety_monitor.py` | arm+hand 完整安全 |
| `deploy/policy_loader.py` | 手动加载模型（LeFranX 模式） |
| `scripts/run_policy.py` | 启动脚本 |

### PolicyLoader（LeFranX 模式：手动加载，不依赖框架）

```python
class PolicyLoader:
    """从 checkpoint 目录加载策略模型。

    目录结构:
      checkpoint/
        config.json          # policy 配置（obs dims, action dims, chunk 等）
        model.safetensors    # 模型权重
        stats.json           # 归一化统计量

    加载后返回 (model, norm_stats, policy_config) 三元组。
    model 需实现 predict(obs: dict) -> np.ndarray。
    """
    @staticmethod
    def load(checkpoint_dir: str) -> tuple[Any, dict, dict]: ...
```

### Action Chunk 执行策略（LeFranX 模式）

```python
class PolicyRunner:
    def __init__(self, ...,
                 chunk_size: int = 1,         # 模型输出的动作长度
                 n_action_steps: int = 1,     # 每次执行几步
                 query_freq: int = 1,         # 每 N 步重新推理一次
                 hand_smooth_alpha: float = 0.5): ...

    def run(self) -> None:
        self._action_buffer = None
        for step in range(max_steps):
            if step % self.query_freq == 0 or self._action_buffer is None:
                # 重新推理
                model_output = self.model.predict(obs)
                self._action_buffer = self._extract_chunk(model_output)
                # 取前 n_action_steps 步
            action_raw = self._action_buffer[self._chunk_idx]
            # EMA 平滑（部署时 arm+hand 都做，LeFranX alpha=0.5）
            action = self._smooth(action_raw, prev_action)
            # safety check → send
            self._chunk_idx += 1
```

**为什么部署时 arm 也做平滑？** 遥操作时不加 arm 平滑是为了录原始数据。部署时策略推理可能有帧间抖动，EMA 平滑（alpha=0.5）可以抑制模型输出的高频噪声，保护真实机器人。

### SafetyMonitor

```python
class SafetyMonitor:
    def check(self, state, action) -> SafetyStatus:
        # Arm: workspace / 关节限位 / 力矩
        # Hand: 关节限位 / 电流(堵转) / 温度 / 通信
```

### 验收

- HoldPositionPolicy + ReplayPolicy 闭环
- `--chunk-size 8 --n-action-steps 3 --query-freq 3` chunk 模式正常
- PolicyLoader 从 checkpoint 目录加载模型
- 手指堵转 / 温度超限 → hard_stop

---

## Day 6：延迟测量 + 端到端集成测试

**工时** 4-6h | **P2 | 参考: [P1] ManiUniCon filter + math_utils, [P2] Open-Teach vectorops + timer**

### 文件

| 文件 | 说明 |
|------|------|
| `utils/hand_utils.py` | estimate_frame + OPERATOR2MANO_RIGHT |
| `utils/latency_bench.py` | 分阶段延迟统计 |
| `scripts/benchmark.py` | |

### 端到端测试清单

**遥操作 (Day 1-2)**

- [ ] arm+hand 连接，hand 断连降级
- [ ] H 键 return_to_home 路径规划 → 逐点执行，ESC 中断
- [ ] VR → IK + retarget(EMA 0.3) → 机器人闭环
- [ ] hand 重定向超限 → hold + flag
- [ ] kill camera 不影响 controller

**Episode 管理 (Day 3)**

- [ ] 按 R: home → re-anchor → 录制；按 S: .h5 写入
- [ ] 连录 3 个 episode，per-episode re-anchor 无跳变
- [ ] quality_flags 正确（含 hand 电流/温度/通信 bits）

**数据消费 (Day 4)**

- [ ] EpisodeReader 加载；回放含 hand skeleton
- [ ] DataValidator 通过（hand 关节+电流）
- [ ] convert_data.py per-joint 归一化

**策略部署 (Day 5)**

- [ ] `--action-mode full/arm_only/hand_only` 各自闭环
- [ ] chunk 执行: chunk_size=8, n_action_steps=3, query_freq=3
- [ ] PolicyLoader 加载 checkpoint
- [ ] 手指堵转/温度/workspace → hard_stop

**标定与延迟 (Day 6-7)**

- [ ] 相机标定（外部脚本 → config/calib/cameras.json）
- [ ] eye-to-hand: T_base_camera 正确，点云变换到基座坐标系
- [ ] eye-in-hand: FK @ T_eef_camera 正确，extrinsics 逐帧变化
- [ ] CameraCalib.to_meta_dict() 写入 HDF5 meta，数值与源文件一致
- [ ] pinky_scale 调优后手指贴合
- [ ] 端到端 < 30ms，> 40Hz，抖动 < 5ms

---

## 完整文件清单

```
dexmani_real/
  config/      __init__.py  pipeline_config.py
  robot/       __init__.py  robot_interface.py  workspace_safety.py
  ipc/         __init__.py  shared_ring_buffer.py
  controller/  __init__.py  teleop_controller.py  tracking_quality.py
               error_handler.py  keyboard_handler.py
  recording/   __init__.py  episode_recorder.py  camera_recorder.py  quality_flags.py
  data/        __init__.py  episode_reader.py  episode_replayer.py  data_validator.py
  deploy/      __init__.py  policy_runner.py  observation_builder.py
               action_parser.py  safety_monitor.py  policy_loader.py
  utils/       __init__.py  camera_calib.py  hand_utils.py  latency_bench.py
scripts/
  run_teleop.py  run_policy.py  convert_data.py
  validate_data.py  benchmark.py  replay_episode.py
config/
  calib/        cameras.json                      # 相机标定数据
```

**总计**：21 个包模块 + 6 个脚本 + 1 个标定数据文件
