# LeFranX XHand 知识体系文档

> 基于 LeFranX 代码库全面分析（commit: main 分支，2026-07-29）
> 对照项目: DexMani Real (dexmani_real)

---

## 目录

1. [系统架构总览](#1-系统架构总览)
2. [XHand 硬件驱动层](#2-xhand-硬件驱动层)
3. [Robot 组合模式（Franka + XHand）](#3-robot-组合模式franka--xhand)
4. [VR 手部追踪 → XHand 重定向全流程](#4-vr-手部追踪--xhand-重定向全流程)
5. [数据采集与录制](#5-数据采集与录制)
6. [训练与策略部署](#6-训练与策略部署)
7. [与 DexMani 的对比分析](#7-与-dexmani-的对比分析)
8. [值得借鉴的模式](#8-值得借鉴的模式)
9. [附录：XHand 命令/状态协议](#9-附录xhand-命令状态协议)

---

## 1. 系统架构总览

LeFranX 是 LeRobot 框架的扩展，支持 **Franka FER 机械臂（7-DOF）+ XHand 灵巧手（12-DOF）** 的 VR 遥操作与模仿学习。项目由三层组成：

```
┌─────────────────────────────────────────────┐
│                  Scripts 层                   │
│  xhand_vr_teleoperator.py (XHand-only)      │
│  dual_vr_teleoperator.py (Franka+XHand)     │
│  dual_vr_record.py (数据采集)                │
│  dual_robot_replay.py (轨迹回放)             │
│  train_act_policy.py / train_dp_policy.py   │
│  dual_robot_deploy_act.py / deploy_dp.py    │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────┴──────────────────────────┐
│            Teleoperator 层 (Python)          │
│  ┌─────────────────────────────────────┐    │
│  │ FrankaFERXHandVRTeleoperator (dual) │    │
│  │  ├─ FrankaFERVRTeleoperator (arm)   │    │
│  │  └─ XHandVRTeleoperator (hand)      │    │
│  └─────────────────────────────────────┘    │
│  VRRouterManager (Singleton, TCP 共享)       │
│  VRHandDetectorAdapter (landmarks→MANO)     │
│  dex-retargeting (MANO→robot joints)         │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────┴──────────────────────────┐
│              Robot 层 (Python)               │
│  ┌─────────────────────────────────────┐    │
│  │ FrankaFERXHand (composite robot)    │    │
│  │  ├─ FrankaFER (7-DOF arm)           │    │
│  │  └─ XHand (12-DOF hand)             │    │
│  └─────────────────────────────────────┘    │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────┴──────────────────────────┐
│              C++ 底层                         │
│  ┌─────────────────────────────────────┐    │
│  │ franka_server (C++, libfranka)       │    │
│  │  - TCP socket 接口 (SET_POSITION,    │    │
│  │    GET_STATE, MOVE_TO_START, STOP)   │    │
│  │  - 实时控制 (ruckig + libfranka)     │    │
│  ├─────────────────────────────────────┤    │
│  │ franka_xhand_teleoperator (C++)     │    │
│  │  - vr_message_router (VR TCP→landmarks) │
│  │  - weighted_ik (臂 IK 求解)          │    │
│  │  - geofik (位姿解算)                 │    │
│  ├─────────────────────────────────────┤    │
│  │ xhand_controller (RobotEra SDK .whl)│    │
│  │  - RS485/EtherCAT 直连              │    │
│  │  - read_state / send_command        │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

### 核心数据流（遥操作）

```
Meta Quest VR App
  │  (TCP: wrist pose + 21 hand landmarks)
  ▼
VRMessageRouter (C++, TCP server, port 8000)
  │
  ▼
VRRouterManager (Python singleton, ref-counted)
  ├──► FrankaFERVRTeleoperator.get_action()
  │      wrist pose → weighted IK → arm joint positions (7)
  │
  └──► XHandVRTeleoperator.get_action()
          21 landmarks → VRHandDetectorAdapter.process_landmarks_data()
          → coordinate transform (→MANO frame) → adaptive_retargeting_xhand()
          → dex-retargeting (dexpilot, POSITION mode) → 12 joint positions
          → joint reorder → action dict
```

---

## 2. XHand 硬件驱动层

### 2.1 驱动文件

| 文件 | 角色 |
|------|------|
| `src/lerobot/robots/xhand/xhand.py` | XHand 机器人实现（继承 `Robot`） |
| `src/lerobot/robots/xhand/xhand_config.py` | 配置数据类（`XHandConfig`） |
| `src/lerobot/robots/xhand/__init__.py` | 模块导出 |

### 2.2 连接方式

XHand 通过 RobotEra 的 `xhand_controller` Python SDK（闭源 `.whl`）连接：

```python
# 初始化
from xhand_controller import xhand_control
self._device = xhand_control.XHandControl()

# RS485 连接
response = self._device.open_serial("/dev/ttyUSB0", 3000000)  # 3M baud
hands_id = self._device.list_hands_id()
self._hand_id = hands_id[0]

# 命令结构体
self._hand_command = xhand_control.HandCommand_t()
# 每根手指: id, kp, ki, kd, position, tor_max, mode
```

**支持的协议**: RS485（默认）和 EtherCAT（未实现）

**与 DexMani 对比**:
- DexMani 也使用相同的 `xhand_controller` SDK
- LeFranX 额外提供了 stub 模式（无硬件测试），DexMani 没有
- DexMani 的连接在 `XHand.check_connection()` → `open_serial()`，更简洁

### 2.3 读取状态（read_state）

```python
error_struct, state = self._device.read_state(self._hand_id, True)
# state.finger_state[i]:
#   .position  # 弧度
#   .torque    # 实际是电流值(mA)，SDK 命名误导
```

**关键实现细节**:
- Torque 字段实际存储的是 **电流值（mA）**，不是扭矩（Nm）——代码中有注释说明: `# Actually current in mA (misnamed in SDK)`
- 读取时会**过滤已知的无害错误**（传感器力/温度读取失败、CRC 错误、硬件版本不支持力控），不中断控制
- 读取耗时约 1ms（perf_counter 计量）

### 2.4 发送命令（send_command）

```python
# 更新位置
for i in range(12):
    self._hand_command.finger_command[i].position = float(positions[i])

# 发送
error_struct = self._device.send_command(self._hand_id, self._hand_command)
```

**命令结构体字段**（12 根手指各自独立设置）:

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `id` | 0-11 | 手指/关节 ID |
| `kp` | 80 | 比例增益 |
| `ki` | 0 | 积分增益 |
| `kd` | 0 | 微分增益 |
| `position` | 0.0 | 目标位置（弧度） |
| `tor_max` | 400 | 最大力矩限制（mA） |
| `mode` | 3 | 控制模式 |

### 2.5 配置参数

```python
@dataclass
class XHandConfig(RobotConfig):
    # 通信
    protocol: str = "RS485"           # "RS485" | "EtherCAT"
    serial_port: str = "/dev/ttyUSB0"
    baud_rate: int = 3000000          # 3M baud
    hand_id: int = 0

    # 控制
    default_kp: int = 80
    default_ki: int = 0
    default_kd: int = 0
    default_tor_max: int = 400        # mA
    default_mode: int = 3
    control_frequency: float = 30.0   # Hz

    # 安全
    max_torque: float = 300.0         # 每关节最大力矩 (mA)

    # Home 位置（度）
    home_position_deg: tuple = (
        0, 80.66, 33.2, 0.00, 5.11, 5.0,
        6.53, 5.0, 6.76, 5.0, 10.13, 5.0
    )

    # 关节限位（度）— 手动防止机械卡死
    joint_limits_deg = [
        0, 105, -60, 90, -10, 105, -10, 10,
        0, 110, 5, 110, 0, 110, 5, 110,
        0, 110, 5, 110, 0, 110, 5, 110
    ]
```

**关节限位修正**: 原始 RobotEra 限位被手动调整为 `[5, 110]`（下限从 0 改为 5）和 `[-10, 10]`（从 `[-10, 105]` 改为 `[-10, 10]`），用于**防止机械卡死**（mechanical clogging）。

### 2.6 安全机制

LeFranX 的安全防护相对基础：

1. **关节限位裁剪**: `_apply_safety_limits()` 在发送前裁剪到 `position_limits`
2. **力矩限制**: `tor_max=400mA` 在硬件层面限制
3. **错误过滤**: 忽略已知无害的传感器错误（5 种）
4. **动作失败处理**: `send_command` 返回 `False` 时仅 `logger.warning`，不停止机器人

**与 DexMani 对比**:
- DexMani 有专门的 `validate_action()` 预发送门，包含误差检查、连接检查、**NaN 检查、力矩门（30N/指）、温度门（70°C）**
- LeFranX **缺少**：NaN 检查、力矩实时门、温度监控、连接丢失自动检测
- LeFranX 的 `stop()` 方法标记为 `NOT IMPLEMENTED`
- LeFranX 的 `recover_from_errors()` 标记为 `NOT IMPLEMENTED`

### 2.7 State 模型

```python
# 观测特征 (observation_features)
{
    "joint_0.pos": float,   # 位置 (rad)
    "joint_0.torque": float, # 电流 (mA)
    ...                      # 共 12 个关节
    "joint_11.pos": float,
    "joint_11.torque": float,
}

# 动作特征 (action_features)
{
    "joint_0.pos": float,   # 目标位置 (rad)
    ...
    "joint_11.pos": float,
}
```

**与 DexMani 对比**:
- DexMani 的 `RobotState` 包含更多字段：`arm_qpos(7)`, `arm_qvel(7)`, `arm_tau(7)`, `eef_pos(3)`, `eef_quat_wxyz(4)`, `hand_qpos(12)`, **`hand_tactile_sum(5,3)`**, **`hand_tactile_force(5,120,3)`**, **`fingertip_pos(5,3)`**
- LeFranX **不采集触觉数据**（`read_state` 只取 position 和 torque/current）
- LeFranX **没有手部速度反馈**

---

## 3. Robot 组合模式（Franka + XHand）

### 3.1 组合架构

```
Robot (abstract base)
  ├── FrankaFER      (arm only, 7-DOF)
  ├── XHand           (hand only, 12-DOF)
  └── FrankaFERXHand  (composition: arm + hand, 19-DOF)
       ├── self.arm = FrankaFER(config.arm_config)
       └── self.hand = XHand(config.hand_config)
```

**组合方式**: **Composition（组合）**，不是继承。`FrankaFERXHand` 同时持有 `FrankaFER` 和 `XHand` 实例。

### 3.2 命名空间前缀

在组合机器人中，所有观测和动作都加上 `arm_` / `hand_` 前缀：

```python
# 观测: arm_joint_0.pos, arm_joint_1.pos, ..., hand_joint_0.pos, hand_joint_0.torque, ...
# 动作: arm_joint_0.pos, ..., hand_joint_0.pos, ...
```

### 3.3 动作路由

```python
def send_action(self, action):
    arm_action = {key[4:]: val for key, val in action.items() if key.startswith("arm_")}
    hand_action = {key[5:]: val for key, val in action.items() if key.startswith("hand_")}

    if self.config.synchronize_actions:
        # 顺序发送（TODO: 真并行）
        self.arm.send_action(arm_action)
        self.hand.send_action(hand_action)
```

### 3.4 双层配置

```python
@dataclass
class FrankaFERXHandConfig(RobotConfig):
    arm_config: FrankaFERConfig   # Franka 配置（含 server_ip, joint_weights...）
    hand_config: XHandConfig      # XHand 配置（含 serial_port, protocol...）
    cameras: Dict[str, CameraConfig]   # 顶层相机
    synchronize_actions: bool = True   # 臂手同步
    emergency_stop_both: bool = True   # 任一出错双停
```

### 3.5 对比 DexMani

| 维度 | LeFranX | DexMani |
|------|---------|---------|
| 架构 | Composition (arm + hand 对象组合) | Process isolation (arm/hand 独立进程) |
| 通信 | XHand 直连串口, Franka 通过 TCP→C++ Server | SHM (SeqlockRingBuffer) + RPC |
| 臂手同步 | 顺序发送（同进程） | 独立进程，各自由 InnerLoop 控制 |
| 进程模型 | 单进程 | 多进程（arm_process, hand_process, camera_process） |
| 崩溃隔离 | 无（同进程，臂手耦合） | 强（fork 进程，崩溃不互相影响） |
| 实时性 | Franka 通过 C++ server 保证 | Arm Mode 6 固件轨迹规划 |

---

## 4. VR 手部追踪 → XHand 重定向全流程

### 4.1 数据流概览

```
Meta Quest VR App (franka-vr-teleop)
  │
  │  TCP (port 8000)
  │  发送: wrist pose (position + quaternion + fist_state)
  │        + 21 hand landmarks (3D coordinates)
  ▼
VRMessageRouter (C++, vr_message_router.cpp)
  │  - TCP server, 接收 VR 数据
  │  - get_messages() → {wrist_data, landmarks_data}
  ▼
VRRouterManager (Python singleton)
  │  - 单例模式, reference counting
  │  - ADB reverse port forwarding (Quest 连接)
  │  - 多个 teleoperator 共享同一 VR 数据源
  ▼
┌─ FrankaFERVRTeleoperator ─────────────────────
│  wrist_data → weighted IK → arm joint pos (7)
│
└─ XHandVRTeleoperator ─────────────────────────
   landmarks_data → VRHandDetectorAdapter
   → dex-retargeting → hand joint pos (12)
```

### 4.2 VRHandDetectorAdapter — 21 Landmarks → MANO 坐标

**核心文件**: `src/lerobot/teleoperators/xhand_vr/vr_hand_detector_adapter.py`

**处理流程** (`_process_landmarks_internal`):

```python
# Step 1: 缩放
keypoint_3d_array *= 1.05  # 匹配 MediaPipe 坐标范围

# Step 2: 右手翻转 X 轴
if self.hand_type == "Right":
    keypoint_3d_array[:, 0] = -keypoint_3d_array[:, 0]

# Step 3: 以手腕为原点
keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]

# Step 4: SVD 估计手部朝向
wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_array)
# 使用 wrist(0), index MCP(5), middle MCP(9) 三个点
# SVD 拟合平面 → normal vector
# Gram-Schmidt 正交化 → rotation matrix (3x3)

# Step 5: 变换到 MANO 坐标系
OPERATOR2MANO_RIGHT = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0],
])
joint_pos = keypoint_3d_array @ wrist_rot @ self.operator2mano

# Step 6: XHand 自适应小指重定向
if "xhand" in self.robot_name.lower():
    joint_pos = adaptive_retargeting_xhand(joint_pos)
```

### 4.3 adaptive_retargeting_xhand — XHand 特有的小指补偿

XHand 的小指与人手比例差异较大，LeFranX 实现了**自适应缩放**：

```python
def adaptive_retargeting_xhand(landmarks):
    """基于手指伸展状态的自适应缩放"""
    pinky_mcp, pinky_pip, pinky_dip, pinky_tip = 17, 18, 19, 20

    # 计算小指伸展程度
    pinky_extension = norm(landmarks[pinky_tip] - landmarks[pinky_mcp])
    extension_ratio = clip(
        (pinky_extension - 0.03) / (0.10 - 0.03), 0.0, 1.0
    )

    # 自适应缩放: 卷曲时 1.2x, 伸展时 2.2x
    adaptive_scale = 1.2 + (2.2 - 1.2) * extension_ratio

    # 沿运动链递进缩放 MCP→PIP→DIP→TIP
    landmarks[pinky_pip] = landmarks[pinky_mcp] + vector * adaptive_scale
    landmarks[pinky_dip] = landmarks[pinky_pip] + vector * adaptive_scale
    landmarks[pinky_tip] = landmarks[pinky_dip] + vector * adaptive_scale
```

**设计思路**: 小指卷曲时少缩放（保持抓握精度），伸展时多缩放（补偿长度差异）。

### 4.4 dex-retargeting — MANO → Robot Joints

使用 Yuzhe Qin 的 `dex-retargeting` 库：

```python
from dex_retargeting.constants import RobotName, RetargetingType, HandType
from dex_retargeting.retargeting_config import RetargetingConfig

# 加载配置
config = RetargetingConfig.load_from_file(config_path).build()

# 重定向类型: "POSITION" (dexpilot 方式)
ref_value = joint_pos[indices, :]
qpos = self.retargeting.retarget(ref_value)
```

**重定向类型**: `RetargetingType.dexpilot`（基于关键点位置的最优化方法）

### 4.5 Joint Reorder — 重定向输出 → XHand 关节顺序

重定向库的输出关节顺序与 XHand 实际关节顺序不同，需要映射：

```python
# XHand 期望的关节顺序
desired_xhand_joint_names = [
    'right_hand_thumb_bend_joint',    # 拇指弯曲
    'right_hand_thumb_rota_joint1',   # 拇指旋转1
    'right_hand_thumb_rota_joint2',   # 拇指旋转2
    'right_hand_index_bend_joint',    # 食指弯曲
    'right_hand_index_joint1',        # 食指侧摆1
    'right_hand_index_joint2',        # 食指侧摆2
    'right_hand_mid_joint1',          # 中指1
    'right_hand_mid_joint2',          # 中指2
    'right_hand_ring_joint1',         # 无名指1
    'right_hand_ring_joint2',         # 无名指2
    'right_hand_pinky_joint1',        # 小指1
    'right_hand_pinky_joint2',        # 小指2
]

# 建立索引映射 + 食指弯曲取反
xhand_joint_positions = qpos[self.retargeting_to_xhand]
xhand_joint_positions[3] = -xhand_joint_positions[3]  # 食指弯曲方向取反
```

### 4.6 平滑

```python
# EMA 平滑，alpha=0.3（可配置，dual 场景常用 0.5-0.6）
qpos = self.smoothing_alpha * current + (1 - self.smoothing_alpha) * previous
```

### 4.7 VR 数据丢失处理

当 VR 数据不可用时：
- 有上一帧数据 → 使用上一帧（hold-last）
- 无上一帧 → 返回全零位姿（开手）
- 同时打印 `logger.warning` 记录数据丢失时间戳

### 4.8 与 DexMani 手部重定向对比

| 维度 | LeFranX | DexMani |
|------|---------|---------|
| VR 手部追踪 | Meta Quest（MediaPipe landmarks 21点） | Meta Quest（MediaPipe landmarks 21点） |
| 坐标变换 | 自定义 `estimate_frame_from_hand_points` (SVD) | 使用 `mediapipe` 原生处理 |
| 重定向方法 | `dex-retargeting` 库（dexpilot, POSITION） | 自研 MANO + NLP 非线性优化 |
| 小指处理 | `adaptive_retargeting_xhand()` 自适应缩放 | 无特殊处理 |
| 关节映射 | 显式 joint reorder + 食指取反 | 隐式在 IK 优化中处理 |
| 平滑 | 简单 EMA (alpha=0.3-0.6) | Cartesian EMA (pos a=0.5, rot a=0.25) |
| 数据丢失处理 | hold-last / 回零 | hold-on-failure |
| C++ 加速 | VRMessageRouter (C++ TCP) | 无 |
| ADB 管理 | 自动 `adb reverse` setup/cleanup | 手动 |

---

## 5. 数据采集与录制

### 5.1 录制入口

`scripts/dual_robot/dual_vr_record.py`（Franka + XHand 双机器人录制）

### 5.2 录制流程

```python
# 1. 创建 dataset（LeRobotDataset）
dataset = LeRobotDataset.create(
    repo_id=dataset_path,
    fps=30,
    features=dataset_features,    # arm_* + hand_* + camera
    robot_type="franka_fer_xhand",
    use_videos=True,
    image_writer_threads=4,
)

# 2. 每 episode 的流程:
#    a) Home 臂和手
#    b) 连接 VR teleoperator
#    c) 重置 VR 参考系（reset_initial_pose）
#    d) record_loop(robot, dataset, teleop, control_time_s=60)
#    e) 断开 teleoperator
#    f) dataset.save_episode()

# 3. 支持 resume（断点续录）
```

### 5.3 数据格式

观测数据（以 `observation.` 前缀录制）:
```
arm_joint_0.pos ... arm_joint_6.pos    # 臂关节位置 (7)
arm_joint_0.vel ... arm_joint_6.vel    # 臂关节速度 (7)
arm_ee_pose.00 ... arm_ee_pose.15      # 末端位姿 (4x4=16)
hand_joint_0.pos ... hand_joint_11.pos # 手关节位置 (12)
hand_joint_0.torque ... hand_joint_11.torque  # 手关节力矩/电流 (12)
camera_images...                       # 相机图像
```

动作数据（以 `action.` 前缀录制）:
```
arm_joint_0.pos ... arm_joint_6.pos    # 臂动作 (7)
hand_joint_0.pos ... hand_joint_11.pos # 手动作 (12)
```

**录制帧率**: 30 Hz（与 DexMani 的 16 Hz 不同）

### 5.4 对比 DexMani

| 维度 | LeFranX | DexMani |
|------|---------|---------|
| 录制框架 | LeRobot（HuggingFace 标准） | 自研（HDF5 v8-10） |
| 录制帧率 | 30 Hz | 16 Hz |
| 异步写入 | LeRobot image_writer_threads | EpisodeRecorder (async + buffer) |
| Resume 支持 | 手动检查 parquet 文件 | dataset numbering |
| 触觉数据 | **不录制** | hand_tactile_sum(5,3), hand_tactile_force(5,120,3) |
| 手部速度 | **不录制** | hand_qvel (via SHM) |
| 指尖位置 | **不录制** | fingertip_pos(5,3) |
| 数据格式 | Parquet + 视频（LeRobot 标准） | HDF5 |
| Episode 管理 | 每 episode 确认输入（手动） | 连续录制 + 键盘控制 |
| 臂-手对齐 | LeRobot 自动时间戳对齐 | TimestampAlignedBuffer 精确对齐 |

---

## 6. 训练与策略部署

### 6.1 训练方法

支持两种主流模仿学习算法：

- **ACT (Action Chunking Transformer)**: `scripts/dual_robot/train_act_policy.py`
- **Diffusion Policy (DP)**: `scripts/dual_robot/train_dp_policy.py`

训练输出在 LeRobot 标准检查点目录。

### 6.2 策略部署

```python
# dual_robot_deploy_act.py / dual_robot_deploy_dp.py

# 1. 加载策略
policy = load_policy(checkpoint_path)

# 2. 推理 → 动作
observation = robot.get_observation()
action = policy.predict(observation)

# 3. 发送动作
robot.send_action(action)
#   arm_joint_0.pos ... arm_joint_6.pos  → Franka arm
#   hand_joint_0.pos ... hand_joint_11.pos → XHand
```

### 6.3 动作空间

- **总动作维度**: 19 DOF（arm 7 + hand 12）
- **手部动作**: 12 维关节位置（弧度），无速度或力矩控制
- **归一化**: 在训练 pipeline 中处理（LeRobot 标准）

### 6.4 轨迹回放

`scripts/dual_robot/dual_robot_replay.py` 支持从数据集重放轨迹，可调速度（`--speed`）。

---

## 7. 与 DexMani 的对比分析

### 7.1 架构差异总结

| 维度 | LeFranX | DexMani |
|------|---------|---------|
| **框架** | LeRobot 扩展（HuggingFace 生态） | 自研独立框架 |
| **机器人** | Franka FER (7-DOF) + XHand | xArm7 (7-DOF) + XHand |
| **进程模型** | 单进程（Python + C++ server） | 多进程隔离（arm/hand/camera 独立进程） |
| **臂控制** | C++ franka_server (TCP socket) | Mode 6 固件轨迹规划（无插值） |
| **臂 IK** | weighted IK (C++) | MPlib (Python) |
| **手控制** | xhand_controller SDK 直连 | xhand_controller SDK + hand_process 隔离 |
| **手重定向** | dex-retargeting (dexpilot) | MANO + NLP 非线性优化 |
| **触觉** | 无 | 5 指 × 120 点 × 3 轴 触觉 |
| **安全** | 基础限位 | 多层安全门（力矩/温度/NaN/跟踪误差） |
| **SHM** | 无 | SeqlockRingBuffer (odd/even torn-read) |
| **VR 通信** | TCP (C++ router) | 直接 VR SDK |
| **控制频率** | 30 Hz | 16 Hz |
| **数据格式** | Parquet + 视频（LeRobot） | HDF5（自研 Schema v8-10） |
| **成熟度** | 研究原型 | 生产级（多轮迭代修复） |

### 7.2 LeFranX 做得好的地方

1. **自适应小指重定向**: `adaptive_retargeting_xhand()` 基于伸展状态的自适应缩放，解决了 XHand 小指比例问题
2. **VR 路由器共享**: `VRRouterManager` singleton 模式，多个 teleoperator 共享同一 VR 源，避免端口冲突
3. **ADB 自动管理**: 自动 `adb reverse` 设置/清理，降低 Quest 连接门槛
4. **Stub 模式**: 无硬件也能测试 pipeline，降低开发门槛
5. **组合机器人抽象**: `FrankaFERXHand` 通过 composition 组合臂手，clean separation
6. **错误过滤而非中断**: 忽略已知无害错误（传感器 CRC 等），不中断控制（更激进但也更流畅）
7. **LeRobot 生态兼容**: 直接使用 HuggingFace 数据集、训练 pipeline、模型 zoo

### 7.3 LeFranX 缺少的（DexMani 已有）

1. **触觉数据采集**: DexMani 的 tactile 是核心差异化优势（5 指 120 点力分布）
2. **多进程崩溃隔离**: DexMani 的 arm/hand process 隔离使单组件故障不影响整体
3. **多层安全门**: NaN 检查、力矩门、温度门、跟踪误差门在 DexMani 中均有覆盖
4. **E-stop 机制**: LeFranX 的 `stop()` 方法标记为 NOT IMPLEMENTED
5. **断线自动检测**: DexMani 通过 SHM 心跳检测 arm/hand 连接状态
6. **数据质量检查**: DexMani 的 `check_episode_health.py`、`assess_trajectory_quality.py`
7. **录制后处理**: DexMani 的 post_processor (sigma_poly 深度校准等)
8. **手部动态控制**: DexMani 有速度、加速度相关的跟踪误差门
9. **温控保护**: DexMani 的 70°C 温度门
10. **录制时续对齐**: DexMani 的 16Hz 网格对齐 + TimestampAlignedBuffer

### 7.4 DexMani 可以借鉴的

1. **自适应小指重定向**: 移植到 DexMani 的手重定向 pipeline（当前 MANO 优化中无此处理）
2. **Stub 模式**: 可以给 DexMani 添加 `--mock-hand` 模式便于在没有硬件时测试 pipeline
3. **LeRobot 兼容层**: 可选支持 LeRobot 格式的数据导入/导出，扩大生态互操作
4. **VR ADB 自动管理**: 适合在 VR 遥操作入口脚本中使用
5. **组合机器人抽象**: 当前 DexMani 的 RobotInterface 是单例，组合抽象更灵活
6. **错误过滤策略**: 对于无害的传感器 CRC/温度读取错误，可以降级为 warning 而非阻断

---

## 8. 值得借鉴的模式

### 8.1 立即可以借鉴的

| 模式 | 文件 | 描述 | 建议移植到 |
|------|------|------|-----------|
| 自适应小指重定向 | `vr_hand_detector_adapter.py:27-84` | 根据伸展率自适应缩放 | `hand_retarget.py` |
| Stub 模式 | `xhand.py:158-163` | 无硬件测试支持 | `xhand.py` |
| 错误白名单过滤 | `xhand.py:231-241` | 已知无害错误不阻断 | `validate.py` |
| VR ADB 自动管理 | `vr_router_manager.py:64-72` | 自动 adb reverse | VR 遥操作入口 |
| Perf 计时日志 | `xhand.py:197-199` | 观测采集耗时跟踪 | `get_state()` |
| 关节限位防卡死 | `xhand_config.py:50-51` | 手动调整限位值 | `xhand_config` |

### 8.2 需要适配后借鉴的

| 模式 | 需要适配 |
|------|---------|
| VRRouterManager singleton | DexMani 的 VR 数据流不同（直接 SDK 而非 TCP） |
| Composition robot | DexMani 的 arm/hand 是独立进程，不适合直接组合 |
| LeRobot 录制格式 | 可以考虑作为可选导出格式 |
| 30Hz 控制频率 | DexMani 使用 16Hz (Mode 6 最优)，需要评估 |

### 8.3 不建议借鉴的

| 模式 | 原因 |
|------|------|
| 单进程控制 | 崩溃隔离弱，DexMani 多进程更安全 |
| 无 E-stop | 安全关键缺失 |
| 无触觉数据 | DexMani 触觉是核心优势 |
| 无手部速度反馈 | 丧失动态控制信息 |
| stop() 未实现 | 生产级应用必须实现 |

---

## 9. 附录：XHand 命令/状态协议

### 9.1 Python API（xhand_controller SDK）

```python
from xhand_controller import xhand_control

# 设备管理
device = xhand_control.XHandControl()
device.open_serial(port, baud_rate)        # RS485 连接
device.list_hands_id()                     # 枚举手 ID

# 数据读取
error_struct, state = device.read_state(hand_id, True)
# state.finger_state[i].position  # rad
# state.finger_state[i].torque    # mA (实际电流)
# error_struct.error_code         # 0=成功
# error_struct.error_message      # 错误描述

# 命令发送
error_struct = device.send_command(hand_id, hand_command)
# hand_command.finger_command[i]:
#   .id, .kp, .ki, .kd, .position, .tor_max, .mode
```

### 9.2 HandCommand_t 结构体

```c
// C 结构体（推断自使用方式）
typedef struct {
    int id;           // 手指/关节 ID (0-11)
    int kp;           // 比例增益 (default: 80)
    int ki;           // 积分增益 (default: 0)
    int kd;           // 微分增益 (default: 0)
    float position;   // 目标位置 (rad)
    int tor_max;      // 最大力矩限制 (mA, default: 400)
    int mode;         // 控制模式 (default: 3)
} FingerCommand_t;

typedef struct {
    FingerCommand_t finger_command[12];
} HandCommand_t;
```

### 9.3 控制模式

Mode 3 是默认模式（位置控制 + 力矩限制），具体模式含义在 RobotEra SDK 文档中定义。

### 9.4 已知错误（可安全忽略）

| 错误消息 | 说明 |
|---------|------|
| `Sensor fails to read the combined force` | 合力传感器偶发读取失败 |
| `Sensor fails to read the distributed force` | 分布力传感器偶发读取失败 |
| `Sensor fails to read temperature` | 温度传感器偶发读取失败 |
| `Communication data CRC error` | 串口 CRC 校验偶发错误 |
| `This hardware version does not support force control mode` | 硬件版本不支持力控 |

### 9.5 关节限位表（手动修正后，度）

| 关节 | Min | Max | 关节名 |
|------|-----|-----|--------|
| 0 | 0 | 105 | Thumb Bend |
| 1 | -60 | 90 | Thumb Rot1 |
| 2 | -10 | 105 | Thumb Rot2 |
| 3 | -10 | 10 | Index Bend |
| 4 | 0 | 110 | Index J1 |
| 5 | 5 | 110 | Index J2 |
| 6 | 0 | 110 | Mid J1 |
| 7 | 5 | 110 | Mid J2 |
| 8 | 0 | 110 | Ring J1 |
| 9 | 5 | 110 | Ring J2 |
| 10 | 0 | 110 | Pinky J1 |
| 11 | 5 | 110 | Pinky J2 |

注：5° 下限用于防止机械卡死（mechanical clogging），Index Bend 的 ±10° 是特殊限制。

---

## 10. C++ 层深度分析

### 10.1 franka_server（实时臂控制）

**文件**: `franka_server/src/franka_server.cpp`

`FrankaPositionServer` 是**实时 C++ TCP 位置服务器**，必须在机器人的 RTPC 上运行。核心通信协议：

```
客户端 ──TCP:5000──→ franka_server ──libfranka──→ Franka 机器人
```

**命令集**（纯文本协议）:

| 命令 | 格式 | 说明 |
|------|------|------|
| `SET_POSITION` | `SET_POSITION p0 p1 p2 p3 p4 p5 p6` | 设置目标关节位置（7 维） |
| `GET_STATE` | `GET_STATE` | 返回关节位置、速度、末端位姿（4x4） |
| `MOVE_TO_START` | `MOVE_TO_START p0 ... p6` | 回到起始位置（关节空间运动） |
| `STOP` | `STOP` | 停止运动 |
| `DISCONNECT` | `DISCONNECT` | 断开客户端连接 |

**响应**: `OK` 或 `ERROR <message>`

**实时控制循环**（1kHz）:
1. 使用 **Ruckig**（Online Trajectory Generation）平滑插值——输入目标位置，输出平滑速度指令
2. 首次调用用当前位置/零速度/零加速度初始化 Ruckig
3. 后续调用输入上一步 Ruckig 输出的速度和加速度以保持连续性
4. 输出 `franka::JointVelocities` 直接控制关节速度

**安全机制**:
- **命令超时**: 500ms 内未收到 `SET_POSITION` → 目标自动回到当前位置（hold）
- **断线安全**: 客户端断开 → Ruckig 以当前位姿为目标，平滑减速到零
- **会话循环**: 控制结束后重新实例化 `franka::Robot`，等待新客户端连接

### 10.2 Weighted IK（臂运动学求解）

**文件**: `franka_xhand_teleoperator/src/weighted_ik.cpp`, `geofik.cpp`

**Layer 1: Geometric IK（`geofik.cpp`）**
- 作者: Pablo Lopez-Custodio（全封闭解）
- 使用 DH 参数，支持多种冗余参数化:
  - `franka_ik_q7()`: q7 为自由变量（最多 8 个解）
  - `franka_ik_q4()`: q4 为自由变量
  - `franka_ik_swivel()`: swivel angle 为自由变量（1000 点离散 + 线性插值细化）
  - 各函数返回对应的 6x7 Jacobian
- 关节限位强制：超限解设为 NaN

**Layer 2: Weighted IK Optimizer（`weighted_ik.cpp`）**
- 使用 **Brent's Method**（黄金分割搜索 + 抛物线插值）在 `[q7_min, q7_max]` 内找最优 q7
- 评分函数:
  ```
  score = weight_manip * manipulability
        - weight_neutral * neutral_distance
        - weight_current * current_distance
  ```
  - `manipulability` = sqrt(det(J·J^T))（Yoshikawa 可操作度）
  - `neutral_distance` = 距中立位姿的距离（按关节范围归一化）
  - `current_distance` = 距当前位姿的距离（按关节权重加权）

- 关节权重示例: `[3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0]` — J1/J2 运动惩罚 3x

### 10.3 VR Message Router（C++ TCP 接收器）

**文件**: `franka_xhand_teleoperator/src/vr_message_router.cpp`

**功能**: TCP 服务器接收 Meta Quest VR 头显的追踪数据

- 监听配置端口（默认 8000），`SO_REUSEADDR`，backlog=1
- 独立 `tcp_receiver_thread`:
  - `accept()` 接受单个客户端
  - `recv()` 接收最多 4096 字节
  - 两个**预编译正则**解析:
    - **腕部**: `Right wrist:, x, y, z, qx, qy, qz, qw, leftFist: <state>`
    - **手部标记**: `Right landmarks: x1,y1,z1,x2,y2,z2,...`（最多 63 个浮点数 = 21 个标记点）
  - 结果存入 `VRMessages`（带时间戳 + mutex 保护）
- `get_messages()`: 消息超时检查（默认 100ms），超时则置 `valid=false`
- 客户端断开时自动返回 `accept()` 等待重连

通过 **pybind11** 绑定暴露给 Python，与 Python 进程共享同一地址空间。

### 10.4 Python 端 VR 路由管理器

**文件**: `src/lerobot/teleoperators/vr_router_manager.py`

`VRRouterManager` 是一个**单例（Singleton）**：

- 使用 `__new__` + class-level `_instance` 实现线程安全单例
- **引用计数**: 每个 teleoperator（arm control / hand control）注册时 ref+1，注销时 ref-1
- 第一个注册 → 创建 C++ `VRMessageRouter` + 启动 TCP server
- 最后一个注销（ref=0） → 关闭 TCP server + 清理 ADB
- **ADB 自动管理**: `setup_adb_reverse(port)` / `cleanup_adb_reverse(port)`
- 校验所有消费者使用相同 TCP 端口
- `get_vr_data()` → `(wrist_data, landmarks_data, status)` — 一次调用获取全部 VR 数据

---

## 11. 训练与策略部署细节

### 11.1 训练超参数

**ACT (Action Chunking Transformer)**:
```python
chunk_size=8, n_action_steps=8, n_obs_steps=1
dim_model=512, dim_feedforward=3200
n_heads=8, n_encoder_layers=4, n_decoder_layers=1
vision_backbone="resnet18", use_vae=True
batch_size=16, steps=100000, lr=1e-4
```

**Diffusion Policy**:
```python
horizon=16, n_action_steps=8, n_obs_steps=2
num_inference_steps=25, num_train_timesteps=100
spatial_softmax_num_keypoints=80  # 8x10 for 320x240
down_dims=(256, 512), n_groups=8
diffusion_step_embed_dim=128
```

### 11.2 动作空间归一化

| 特征类型 | 归一化方式 |
|---------|-----------|
| STATE（54 维） | MIN_MAX |
| VISUAL（图像） | MEAN_STD |
| ACTION（19 维） | MIN_MAX |

统计量从数据集 `stats.json` 加载。训练输出需要 `policy.unnormalize_outputs()` 后再发送到机器人。

### 11.3 策略部署控制循环

```
30 Hz 循环:
  1. robot.get_observation() → 54 维 state + camera images
  2. Normalize inputs
  3. Policy inference:
     ACT: predict_action_chunk → 取下一个 action (每3步 query 一次新 chunk)
     DP:  generate_actions (25 DDPM steps) → 取前 8 个 actions
  4. Unnormalize outputs
  5. EMA smoothing (alpha=0.5)
  6. Action scaling (*= ACTION_SCALE, default 1.0)
  7. Split: indices 0-6→arm, 7-18→hand
  8. robot.send_action(action_dict)
  9. Log to Rerun (visualization)
```

### 11.4 双机器人部署的特殊处理

由于 Draccus 在组合机器人上存在循环导入问题，双机器人脚本**完全绕过了** Draccus CLI 解析：

- XHand/Franka 单机器人 → shell wrapper 调用标准 `python -m lerobot.scripts.train`
- 双机器人（Franka+XHand）→ 直接构造 `TrainPipelineConfig` + 手动指定 input/output shapes
- 特征检测从自动（dataset metadata）变为**手动定义**（避免 shape mismatch）

---

## 参考资料

- LeFranX arXiv: [LeVR: A Modular VR Teleoperation Framework](https://arxiv.org/abs/2509.14349)
- LeFranX GitHub: [wengmister/LeFranX](https://github.com/wengmister/LeFranX)
- dex-retargeting: [dexsuite/dex-retargeting](https://github.com/dexsuite/dex-retargeting)
- XHand SDK: RobotEra 文档中心（闭源 .whl）
- Meta Quest VR App: [wengmister/franka-vr-teleop](https://github.com/wengmister/franka-vr-teleop)
- DexMani 对照: `/home/zhanghaoyang/Desktop/dexmani_real/`
