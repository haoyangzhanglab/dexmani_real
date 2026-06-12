# 外部 SDK 依赖参考

> **效力**: 本文档记录项目所有外部 SDK 的版本、API 签名、集成模式和已知陷阱。新增/修改硬件驱动时必须遵循此文档中的 API 约束。

---

## 1. xArm Python SDK

| 属性 | 值 |
|------|-----|
| **包名** | `xarm-python-sdk` (PyPI) / `xarm.wrapper` |
| **仓库** | https://github.com/xArm-Developer/xArm-Python-SDK |
| **最新版** | 1.18.4 |
| **Python** | 仅 Python 3 |
| **安装** | `pip install xarm-python-sdk` |

### 核心 API

```python
from xarm.wrapper import XArmAPI

arm = XArmAPI('192.168.1.113')          # IP 连接
arm = XArmAPI('192.168.1.113', is_radian=True)  # 使用弧度（本项目强制 True）

# ── 生命周期 ──
arm.connect()          # 建立连接
arm.disconnect()       # 断开连接
arm.emergency_stop()   # 急停

# ── 使能 ──
arm.motion_enable(enable=True)
arm.set_mode(0)        # 0=位置模式
arm.set_state(0)       # 0=启动

# ── 关节控制 ──
arm.set_servo_angle(angle=[0, 0, 0, 0, 0, 0, 0], wait=True)  # 阻塞式
arm.set_servo_angle_j(angles=[...])  # 伺服模式（本项目遥操作用这个）

# ── 笛卡尔控制 ──
arm.set_position(x=300, y=0, z=200, roll=180, pitch=0, yaw=0, wait=True)
arm.set_servo_cartesian(position=[...])

# ── 状态读取 ──
arm.angles            # property: 当前关节角 (rad, 当 is_radian=True)
arm.position           # property: 当前 EEF 位置
arm.error_code         # property: 错误码
arm.warn_code          # property: 警告码
arm.has_error          # property
arm.has_warn           # property
arm.get_state()        # 完整状态
arm.get_servo_angle()  # 关节角度
arm.get_position()     # EEF 位置

# ── 错误处理 ──
arm.clean_error()
arm.clean_warn()
arm.get_err_warn_code()

# ── 回调 ──
arm.register_report_callback(callback)
arm.register_error_warn_changed_callback(callback)
```

### 本项目使用模式（robot/xarm7.py）

```python
# ref: xArm-Python-SDK examples/2000-servo_controller.py
# 本项目使用 set_servo_angle_j() 做实时伺服控制

code = self.arm.set_servo_angle_j(angles=qpos_cmd.tolist(), is_radian=True)
if code == 0:
    # 成功
    return True
else:
    self.error_state = True
    self.last_error_message = f"set_servo_angle_j failed: code={code}"
    return False
```

### 已知陷阱

1. **`is_radian` 默认 False**: `XArmAPI` 的 `is_radian` 默认为 False（度）。本项目统一使用弧度，必须传 `is_radian=True`。
2. **`connect()` 行为**: `XArmAPI` 构造函数默认自动连接 (`do_not_open=False`)，但本项目包装层显式调用并做幂等检查。
3. **API 重命名 (1.17.0)**: 旧版 `set_impedance` → 新版 `set_ft_sensor_admittance_parameters`；本项目暂不使用 FT 传感器。
4. **servo 模式返回码**: `set_servo_angle_j` 返回 int (0=成功)，不是异常。

---

## 2. XHand SDK (xhand_controller)

| 属性 | 值 |
|------|-----|
| **包名** | `xhand_controller` (wheel) |
| **版本** | 1.1.8 |
| **本地路径** | `/home/zhy/Documents/硬件/Xhand/SDK/Python/xhand_control_sdk_py_x86_64_v118 (1)/xhand_control_sdk_py/` |
| **安装** | `pip install xhand_controller-1.1.8-cp312-cp312-linux_x86_64.whl` (按 Python 版本选择 wheel) |
| **文档** | `XHAND Python SDK接口说明文档 (1).pdf` |

### 核心 API

```python
from xhand_controller import xhand_control as xh

# ── 设备管理 ──
control = xh.XHandControl()
devices = control.enumerate_devices("RS485")   # 枚举 RS485 设备
devices = control.enumerate_devices("EtherCAT") # 枚举 EtherCAT 设备
control.open_serial("/dev/ttyUSB0", 3000000)     # RS485 连接
control.open_ethercat("device_name")             # EtherCAT 连接
control.close_device()

# ── 命令结构 ──
command = xh.HandCommand_t()
for i in range(12):
    command.finger_command[i].id = i
    command.finger_command[i].kp = 100       # 比例增益
    command.finger_command[i].ki = 0         # 积分增益
    command.finger_command[i].kd = 1         # 微分增益
    command.finger_command[i].position = 0.5 # rad（目标位置）
    command.finger_command[i].tor_max = 300  # 最大力矩
    command.finger_command[i].mode = 3       # 0=无力 3=位置 5=大力

# ── 发送控制 ──
err = control.send_command(device_id=0, command=command)
# err.error_code == 0 → 成功

# ── 读取状态 ──
err, state = control.read_state(device_id=0, force_update=True)
# state.finger_state[i] — 每个手指的状态:
#   .id, .position, .torque, .temperature, .raw_position
#   .commboard_err, .jonitboard_err, .tipboard_err
# state.sensor_data[i] — 指尖触觉传感器:
#   .calc_force (fx, fy, fz), .raw_force[120], .calc_temperature
```

### 控制模式（mode）

| mode | 名称 | 说明 |
|------|------|------|
| 0 | 无力模式 | 手指无力，自由移动 |
| 3 | 位置模式 | PID 位置控制（默认） |
| 5 | 大力模式 | 增大力矩的位置控制 |

### 本项目使用模式（robot/xhand.py）

```python
# ref: XHand SDK xhand_control_example.py L122-128
# 每帧更新 command 的 position 字段，重复使用同一 HandCommand_t 对象

def send_action(self, action: np.ndarray) -> bool:
    target_qpos = self.array12(action)
    target_qpos = self.limit_joint_range(target_qpos)   # range clip
    qpos_cmd = self.limit_joint_step(target_qpos)        # delta limit
    self.write_command_positions(qpos_cmd)
    err = self.control.send_command(self.config.device_id, self.hand_command)
    return self.error_ok(err)
```

### 关节映射（12-DOF）

```
索引   关节名              限位 [min, max] rad
0     thumb_abduction     [0.0, 1.832]
1     thumb_joint1        [-0.698, 1.57]
2     thumb_joint2        [0.0, 1.57]
3     index_abduction     [-0.174, 0.174]
4     index_joint1        [0.0, 1.919]
5     index_joint2        [0.0, 1.919]
6     middle_joint1       [0.0, 1.919]
7     middle_joint2       [0.0, 1.919]
8     ring_joint1         [0.0, 1.919]
9     ring_joint2         [0.0, 1.919]
10    little_joint1       [0.0, 1.919]
11    little_joint2       [0.0, 1.919]
```

### 已知陷阱

1. **SDK 缓存**: 调用 `open_serial()` 后 SDK 内部缓存可能为全零。`connect()` 必须用 `force_update=True` 多次读取来刷新硬件状态。
2. **通信板错误不抛异常**: `commboard_err`/`jointboard_err`/`tipboard_err` 字段需要手动检查。本项目 `is_error()` 会检查这些字段。
3. **mode=0 stop()**: `stop()` 发送 home 位置 + mode=0（无力），使手指无力。不是急停但比断电安全。
4. **sensor_id 映射**: 指尖传感器 ID 为 `[0x11, 0x12, 0x13, 0x14, 0x15]` 对应 thumb/index/middle/ring/little。

---

## 3. VR Hand Tracking SDK

| 属性 | 值 |
|------|-----|
| **包名** | `hand-tracking-sdk` (PyPI) |
| **仓库** | https://github.com/wengmister/hand-tracking-sdk |
| **许可证** | Apache-2.0 |
| **安装** | `pip install hand-tracking-sdk`（可选 `[visualization]` extra for Rerun） |
| **传输模式** | UDP / TCP server / TCP client |

### 核心 API

```python
from hand_tracking_sdk import (
    HTSClient, HTSClientConfig, StreamOutput, TransportMode,
    HandFilter, ErrorPolicy, HandFrame, JointName,
    unity_left_to_rfu_position, unity_left_to_rfu_rotation,
    unity_left_to_flu_position, unity_left_to_flu_rotation,
)

# ── 客户端 ──
client = HTSClient(HTSClientConfig(
    transport_mode=TransportMode("tcp_server"),
    host="0.0.0.0",
    port=8000,
    output=StreamOutput.FRAMES,
    hand_filter=HandFilter("right"),        # "left" | "right" | "both"
    error_policy=ErrorPolicy.TOLERANT,
))

# ── 接收循环 ──
for event in client.iter_events():
    if isinstance(event, HandFrame):
        # event.wrist — WristPose(x, y, z, qx, qy, qz, qw)
        # event.landmarks.points — 21 个 (x,y,z) landmark
        # event.side — Left/Right
        # event.recv_ts_ns, event.source_ts_ns, event.sequence_id
        pass

# ── 坐标变换 ──
# Unity left-hand → FLU (front-left-up): 本项目默认
pos_flu = unity_left_to_flu_position(wrist.x, wrist.y, wrist.z)
quat_xyzw = unity_left_to_flu_rotation(wrist.qx, wrist.qy, wrist.qz, wrist.qw)

# Unity left-hand → RFU (right-forward-up)
pos_rfu = unity_left_to_rfu_position(wrist.x, wrist.y, wrist.z)
```

### 数据格式

| 字段 | Shape | 说明 |
|------|-------|------|
| `wrist_pos` | (3,) | 腕部位置 (m) |
| `wrist_quat_wxyz` | (4,) | 腕部朝向 (w,x,y,z) |
| `landmarks` | (21, 3) | MediaPipe 21 个手部关键点 |
| `sequence_id` | int | 帧序号 |
| `recv_ts_ns` | int | 接收时间戳 (ns) |

### 21 个 MediaPipe Landmark

```
0=WRIST, 1=THUMB_CMC, 2=THUMB_MCP, 3=THUMB_IP, 4=THUMB_TIP,
5=INDEX_FINGER_MCP, 6=INDEX_FINGER_PIP, 7=INDEX_FINGER_DIP, 8=INDEX_FINGER_TIP,
9=MIDDLE_FINGER_MCP, 10=MIDDLE_FINGER_PIP, 11=MIDDLE_FINGER_DIP, 12=MIDDLE_FINGER_TIP,
13=RING_FINGER_MCP, 14=RING_FINGER_PIP, 15=RING_FINGER_DIP, 16=RING_FINGER_TIP,
17=PINKY_MCP, 18=PINKY_PIP, 19=PINKY_DIP, 20=PINKY_TIP
```

### 本项目使用模式（teleop/quest_hand_tracker.py）

```python
# ref: hand-tracking-sdk examples/stream_frames.py
# 后台线程接收 → lock 保护 latest_frame → get_latest() 暴露给主线程

class QuestHandTracker:
    def connect(self) -> None:     # 创建 client + 启动后台线程
    def get_latest(self) -> dict:  # 读取最新帧（线程安全）
    def read(self, timeout) -> dict:  # 阻塞等待新帧
    def disconnect(self) -> None:  # 停止线程 + 清理
```

### 已知陷阱

1. **坐标系**: HTS 默认输出 Unity left-handed 坐标系。本项目在 `teleop/quest_hand_tracker.py` 中转换为 FLU (front-left-up)。
2. **四元数顺序**: HTS 输出 xyzw，本项目统一使用 wxyz。`hand_retarget.py` 中的 `OPERATOR2MANO_RIGHT` 也是 wxyz 顺序。
3. **Landmark vs Wrist 是独立包**: HTS 底层两个独立的 UDP 数据包（wrist + landmarks），SDK 通过 `HandFrameAssembler` 按 `source_frame_seq` 组装。

---

## 4. dex-retargeting

| 属性 | 值 |
|------|-----|
| **包名** | `dex_retargeting` (PyPI) |
| **仓库** | https://github.com/wengmister/vr-dex-retargeting (VR 分支) |
| **上游** | https://github.com/dexsuite/dex-retargeting |
| **许可证** | MIT |
| **安装** | `pip install dex_retargeting` |
| **VR 示例依赖** | 需进入 `example/vector_retargeting/` 后 `pip install` 额外依赖 |

### 核心 API

```python
from dex_retargeting.retargeting_config import RetargetingConfig

# ── 加载配置 ──
RetargetingConfig.set_default_urdf_dir(str(ASSET_DIR / "robots"))
retargeter = RetargetingConfig.load_from_file(str(config_path)).build()

# ── 关节映射（关键！）──
# 不同 URDF 解析器的关节顺序不同，必须通过名称索引映射
retargeting_joint_names = retargeter.optimizer.robot.dof_joint_names
retargeting_to_sapien = np.array(
    [retargeting_joint_names.index(name) for name in sapien_joint_names]
).astype(int)

# ── 执行 retargeting ──
# ref_value: 21 个 landmark 的手部关键点坐标
# DexPilot 方法: ref_value = task_indices - origin_indices
qpos = retargeter.retarget(ref_value, fixed_qpos=fixed_joint_values)
qpos = qpos[retargeting_to_sapien]  # 按目标顺序重排
```

### 支持的优化器

| 类型 | 类 | 适用场景 |
|------|-----|---------|
| `dexpilot` | `DexPilotOptimizer` | VR 实时遥操作（本项目使用） |
| `position` | `PositionOptimizer` | 手-物姿态数据集后处理 |

### 本项目使用模式（teleop/hand_retarget.py）

```python
# ref: vr-dex-retargeting vr_realtime_retargeting.py
class XHandRetargeter:
    def __init__(self, hand_type="right", retargeting_type="dexpilot"):
        # 加载 config/retargeting/xhand_{hand_type}_{type}.yml
        config_path = CONFIG_DIR / "retargeting" / f"xhand_{hand_type}_{retargeting_type}.yml"
        RetargetingConfig.set_default_urdf_dir(str(ASSET_DIR / "robots"))
        self.retargeter = RetargetingConfig.load_from_file(str(config_path)).build()

    def retarget(self, hand_joint_pos: np.ndarray) -> np.ndarray:
        ref_value = self.build_ref_value(hand_joint_pos)
        qpos = self.retargeter.retarget(ref_value, fixed_qpos=self.fixed_joint_values)
        return qpos[self.retargeted_joint_order]  # 按 sapien_joint_names 重排
```

### 已知陷阱

1. **关节顺序不一致**: SAPIEN URDF vs ROS URDF 的 active joint 顺序可能不同。**必须通过关节名称索引映射**，不能直接按位置索引。
2. **assets 子模块**: VR 分支依赖 `wengmister/dex-urdf-plus` 更新后的 URDF 仓库，需 `git submodule update --init`。
3. **numpy 版本**: v0.5.0 起要求 `numpy >= 2.0.0`。

---

## 5. MPlib (Motion Planning Library)

| 属性 | 值 |
|------|-----|
| **包名** | `mplib` (PyPI) |
| **版本** | 0.2.1 |
| **文档** | https://motion-planning-lib.readthedocs.io/latest/ |
| **仓库** | https://github.com/haosulab/MPlib |
| **许可证** | MIT |
| **安装** | `pip install mplib` (预编译 wheel, Ubuntu 20.04+, Python 3.8+) |

### 核心 API

```python
import mplib as mp

# ── Planner 初始化 ──
planner = mp.Planner(
    urdf=str(urdf_path),           # URDF 文件路径
    srdf=str(srdf_path),           # SRDF 文件路径（可用 generate_srdf 自动生成）
    move_group=eef_link_name,      # 末端执行器 link 名
    use_convex=True,               # 使用凸包碰撞检测
    joint_vel_limits=[...],        # 关节速度限制 (rad/s)
    joint_acc_limits=[...],        # 关节加速度限制 (rad/s²)
)

# ── SRDF 生成 ──
mp.urdf_utils.generate_srdf(urdf_path, srdf_path)

# ── Pinocchio 模型 ──
pinocchio_model = planner.pinocchio_model
link_names = list(pinocchio_model.get_link_names())
joint_names = list(pinocchio_model.get_joint_names())
joint_limits = np.asarray(planner.joint_limits)

# ── IK ──
status, qpos = planner.IK(
    goal_pose=goal_pose,           # MPlib Pose 对象
    start_qpos=seed_qpos,         # 初始猜测
    n_init_qpos=n_init,           # 随机种子数
    return_closest=True,          # 失败时返回最近解
)
# status: "Success" / "IK Failed" 等字符串

# ── 路径规划 ──
result = planner.plan_screw(
    goal_pose=goal_pose,
    current_qpos=current_qpos,
    time_step=0.1,
    qpos_step=0.1,
    wrt_world=True,
)

result = planner.plan_qpos(
    goal_qposes=[goal1, goal2],
    current_qpos=current_qpos,
    time_step=0.1,
    rrt_range=0.1,
    planning_time=2.0,
    simplify=True,
)

# ── 碰撞检测 ──
planner.check_for_self_collision(qpos)
planner.check_for_env_collision(qpos)

# ── FK / Jacobian ──
planner.compute_forward_kinematics(qpos)   # link poses
planner.compute_single_link_local_pose(qpos, link_idx)
planner.compute_single_link_pose(qpos, link_idx)

# ── 基础位姿 ──
planner.set_base_pose(base_pose)  # 设置机器人底座在世界坐标系中的位姿

# ── Pinocchio 后端 ──
# MPlib 内部使用 pinocchio 做运动学
# planner.pinocchio_model 暴露 Pinocchio Model 对象
```

### 可选后端

| 后端 | 说明 |
|------|------|
| **Pinocchio** (默认) | 快速的刚体动力学库，用于 FK/Jacobian |
| **KDL** | OROCOS Kinematics and Dynamics Library（备选） |
| **OMPL** | 用于 motion planning 的随机采样算法（RRT 等） |
| **FCL** | Flexible Collision Library 用于碰撞检测 |

### 本项目使用模式（robot/planner/）

```python
# ref: MPlib docs Getting Started + IK tutorial
class XArm7MotionPlanner:
    def __init__(self, config, ...):
        # 1. 自动生成 SRDF
        mp.urdf_utils.generate_srdf(config.urdf_path, config.srdf_path)
        # 2. 创建 Planner（Pinocchio FK + FCL 碰撞检测）
        self.mp_planner = self.mp.Planner(
            urdf=str(config.urdf_path),
            srdf=str(config.srdf_path),
            move_group=config.eef_link_name,
            joint_vel_limits=joint_vel_limits.tolist(),
            joint_acc_limits=joint_acc_limits.tolist(),
        )
        # 3. 获取 Pinocchio model 用于自定义 FK/Jacobian
        self.pinocchio_model = self.mp_planner.pinocchio_model

    def call_mplib_ik(self, target_pose_base, seed_qpos, n_init_qpos, return_closest):
        status, raw_qpos = self.mp_planner.IK(...)
        return status, raw_qpos

    def plan_path(self, target_pose, current_qpos):
        # plan_screw → plan_qpos 两级策略
        ...
```

### 已知陷阱

1. **`IK()` 返回值**: 返回 `(status: str, qpos: np.ndarray)`。status 是字符串 `"Success"` 开头表示成功，不是 bool。
2. **SRDF 必须预生成**: URDF 路径必须可访问且 `generate_srdf` 生成的 SRDF 路径必须在同目录。
3. **joint_limits 格式**: Planner 的 `joint_limits` 返回 `(n, 2)` 数组，每行 `[min, max]`。
4. **路径规划结果**: `plan_screw()`/`plan_qpos()` 返回 dict，key `"status"` 和 `"position"`。
5. **Pinocchio model 不暴露 `forward_kinematics` 直接接口**: 使用 `planner.compute_forward_kinematics()` 或直接调 Pinocchio API。

---

## 6. SAPIEN

| 属性 | 值 |
|------|-----|
| **包名** | `sapien` (PyPI) |
| **版本** | 3.0.3 |
| **安装** | `pip install sapien==3.0.3` |
| **文档** | https://sapien.ucsd.edu/ |

### 本项目使用模式（robot/model/）

```python
# ref: robot/model/constructor.py — SAPIEN 场景 + URDF 加载
# ref: robot/model/xarm7_xhand.py — XArm7_XHand 组合机器人类
# ref: robot/model/sim_adapter.py — SimRobotInterface 适配 RobotInterface 接口

# SAPIEN 物理引擎 time_step (240Hz) > 控制频率 (50Hz)
# sim_adapter._step_physics(n=5): 240Hz → 48Hz 有效控制频率

class SimRobotInterface:
    def connect(self) -> bool:
        self.scene = setup_scene(time_step=1.0/240.0)
        self.robot = XArm7_XHand(self.scene, ...)

    def get_state(self) -> dict:
        # qpos = self.robot.get_qpos()  → (19,) [arm7 + hand12]
        # eef_pose = self.robot.get_eef_pose()  → sapien.Pose

    def send_action(self, action: np.ndarray) -> bool:
        self.robot.apply_action(target_qpos)
        self._step_physics()  # 推进物理仿真
```

### 已知陷阱

1. **物理步长 vs 控制频率**: SAPIEN 物理步长 1/240s，需要 `_step_physics(n=5)` 在每帧控制间推进 5 步。
2. **headless 模式**: 设置 `headless=True` 可在 CI 运行，无需 GPU/显示器。
3. **FK 坐标系差异**: SAPIEN `get_link_poses()` 和 Pinocchio `forward_kinematics()` 的 root_pose 变换可能导致 ~1.58m 的 FK 差异，这是坐标系 artifact，不是功能问题。IK 往返验证 (`validate_ik_roundtrip`) 是核心验证指标。

---

## 7. 版本兼容性矩阵

| SDK | 版本 | Python | 关键依赖 |
|-----|------|--------|---------|
| xarm-python-sdk | 1.18.x | 3.8+ | - |
| xhand_controller | 1.1.8 | 3.10/3.12 (wheel) | - |
| hand-tracking-sdk | latest | 3.8+ | - |
| dex_retargeting | latest | 3.8+ | numpy>=2.0.0, pinocchio |
| mplib | 0.2.1 | 3.8+ | pinocchio, FCL, OMPL |
| sapien | 3.0.3 | 3.8+ | - |

## 8. 新增 SDK 依赖检查清单

引入新的外部 SDK 到 `robot/` 或 `sensor/` 时：

- [ ] 版本号在此文档中记录
- [ ] API 签名与 CLAUDE.md Section 2 接口契约兼容
- [ ] 不引入 CLAUDE.md Section 0.5.6 中已禁用的依赖（ROS/Hydra/Pydantic/LeRobot）
- [ ] 安装命令在 CI 环境中可执行
- [ ] `send_action()` 包装层包含 joint limit + delta limit 安全裁剪
- [ ] 错误处理遵循 CLAUDE.md Section 12（捕获具体异常类型，设 error_state）
- [ ] 无可视化/CUDA 等重型依赖在 top-level import
