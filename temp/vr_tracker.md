# Quest Hand Tracking Teleop Adapter 使用说明

本文档对应当前最终版两文件实现：

```text
vr_utils.py
vr_tracker.py
```

目标是把 Quest / hand-tracking-sdk 的 VR 手部追踪数据整理成机器人工作流可直接使用的形式：

```text
Quest / HTS HandFrame
  -> HandData
      -> retargeting_joints  # 21x3 wrist-relative hand skeleton
      -> finger_curl         # 15-dim bend angles
      -> EEF target          # world-frame target pose for external IK
  -> external dex-retargeting / IK / robot command
```

设计原则：

- `vr_tracker.py` 不接入 dex-retargeting 对象。
- `vr_tracker.py` 不实现 IK。
- `vr_tracker.py` 不直接发 robot command。
- `VideoReturnService` 是可选视频回传模块，和 VR tracking 状态服务分离。
- 内部 quaternion 一律使用 `wxyz`。

---

## 1. 文件结构

```text
vr_utils.py
  - Vec3 / Quat type aliases
  - quaternion / Euler / matrix helpers
  - LeFranX-style VR->robot alignment helper
  - hand_data_to_retargeting_joints(...)
  - hand_data_to_wrist_relative_joints(...)
  - hand_data_to_wrist_local_joints(...)
  - hand_data_to_finger_curl_vector(...)

vr_tracker.py
  - VRTrackerConfig
  - VRTrackerStats
  - HandData
  - EefTarget
  - VRState
  - QuestHandReceiver
  - VRTrackerService
  - VideoReturnService
```

推荐 import：

```python
from vr_tracker import QuestHandReceiver, VRTrackerConfig, VRTrackerService
from vr_utils import lefranx_vr_to_robot_matrix
```

如果需要视频回传：

```python
from vr_tracker import VideoReturnService
```

---

## 2. 安装

### 2.1 基础依赖

```bash
pip install numpy transforms3d hand-tracking-sdk
```

当前代码使用：

- `numpy`：数组、landmarks、pose matrix。
- `transforms3d`：quaternion / Euler / matrix 旋转运算。
- `hand-tracking-sdk`：接收 Quest / HTS telemetry，组装 `HandFrame`。

`transforms3d` 的 quaternion API 使用 `w, x, y, z` 顺序，和本项目内部 quaternion 约定一致。

### 2.2 可选视频回传依赖

如果需要 `VideoReturnService`，安装 SDK 的 video extras：

```bash
pip install "hand-tracking-sdk[video]"
```

视频回传依赖 WebRTC / PyAV 等额外组件。普通 VR tracking、retargeting joints、finger curl、EEF target 不需要启动视频模块。

### 2.3 文件放置

把两个文件放在你的工作目录或包目录中：

```text
your_project/
  vr_utils.py
  vr_tracker.py
  main_teleop.py
```

---

## 3. 硬件通信连接

### 3.1 Quest / HTS 到 Host 的手部追踪通信

当前 `QuestHandReceiver` 默认使用：

```python
TransportMode.TCP_SERVER
host = "0.0.0.0"
port = 8000
output = StreamOutput.FRAMES
```

也就是说：

```text
Host Python 端作为 TCP server 监听 8000 端口
Quest / HTS app 作为 client 连接 Host IP:8000
Quest 持续发送 wrist packet + landmarks packet
hand-tracking-sdk 组装成 HandFrame
```

默认配置：

```python
config = VRTrackerConfig(
    host="0.0.0.0",
    port=8000,
    hand_filter="right",  # "left" / "right" / "both"
)
```

网络检查：

```bash
# Linux/macOS 查看本机 IP
ip addr
# 或
ifconfig
```

Quest 和 Host 需要在同一局域网，或者能互相访问。注意防火墙开放 TCP 8000。

### 3.2 视频回传通信

如果开启 `VideoReturnService`，Host 会启动 WebRTC signaling server：

```python
video = VideoReturnService(
    signaling_host="0.0.0.0",
    signaling_port=8765,
    width=1280,
    height=720,
    fps=30,
)
video.start()
```

Quest 端 video receiver / app 应连接：

```text
ws://<HOST_IP>:8765
```

注意：视频回传和手部追踪数据接收是两条独立通路。

```text
Hand tracking path:
  Quest -> TCP telemetry -> QuestHandReceiver -> VRTrackerService

Video return path:
  RGB frame -> VideoReturnService -> WebRTC -> Quest display
```

真实机器人部署时，视频可以降分辨率、降帧率、丢帧；robot command loop 不应该等待视频。

---

## 4. 核心数据结构

### 4.1 `VRTrackerConfig`

```python
@dataclass(frozen=True, slots=True)
class VRTrackerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    hand_filter: str = "both"
    timeout_s: float = 1.0
    smooth_alpha: float = 1.0
    max_frame_age_s: float = 0.0
    max_linear_speed: float = 0.0
    max_angular_speed: float = 0.0
    position_scale: float = 1.0
    rotation_scale: float = 1.0
    workspace_limits: tuple[float, float, float, float, float, float] | None = None
    orientation_limits: tuple[float, float, float, float, float, float] | None = None
    max_orientation_angle: float = 0.0
```

字段说明：

| 字段 | 含义 |
|---|---|
| `host` / `port` | TCP server 监听地址和端口。 |
| `hand_filter` | 追踪手：`"left"`、`"right"`、`"both"`。 |
| `timeout_s` | SDK socket timeout。 |
| `smooth_alpha` | EWMA smoothing 系数，`1.0` 表示不平滑，范围 `(0, 1]`。 |
| `max_frame_age_s` | stale frame watchdog。`0.0` 表示不启用。 |
| `max_linear_speed` | EEF target 位置限速，单位 m/s。`0.0` 表示不启用。 |
| `max_angular_speed` | EEF target 姿态限速，单位 rad/s。`0.0` 表示不启用。 |
| `position_scale` | VR wrist 位移到 robot EEF 位移的缩放。 |
| `rotation_scale` | VR wrist 姿态变化到 robot EEF 姿态变化的缩放。 |
| `workspace_limits` | 相对 home EEF position 的位置限制。 |
| `orientation_limits` | 相对 home EEF orientation 的 RPY 限制。 |
| `max_orientation_angle` | 相对 home orientation 的总旋转角限制。 |

`workspace_limits` 顺序：

```python
(min_x, max_x, min_y, max_y, min_z, max_z)
```

`orientation_limits` 顺序：

```python
(roll_min, roll_max, pitch_min, pitch_max, yaw_min, yaw_max)
```

单位均为 rad，使用 `sxyz` fixed-axis RPY。

示例：

```python
config = VRTrackerConfig(
    hand_filter="right",
    max_frame_age_s=0.2,
    max_linear_speed=0.25,
    max_angular_speed=1.0,
    position_scale=1.0,
    rotation_scale=1.0,
    workspace_limits=(-0.3, 0.3, -0.3, 0.3, -0.2, 0.4),
    orientation_limits=(-0.8, 0.8, -0.6, 0.6, -1.2, 1.2),
    max_orientation_angle=1.5,
)
```

---

### 4.2 `HandData`

```python
@dataclass(frozen=True, slots=True)
class HandData:
    side: str
    sequence_id: int
    frame_id: str | None
    wrist_pos: Vec3
    wrist_quat: Quat
    landmarks: tuple[Vec3, ...]
    recv_ts_ns: int
    recv_time_unix_ns: int | None
    source_ts_ns: int | None
    source_frame_seq: int | None
```

语义：

```text
HandData = VR / SDK 数据承载层
```

它只表示 hand-tracking-sdk 读出的数据，不知道 dex-retargeting、IK 或 robot driver。

约定：

- `wrist_pos`：FLU 坐标系下 wrist position。
- `wrist_quat`：FLU 坐标系下 wrist orientation，格式 `wxyz`。
- `landmarks`：FLU 坐标系下 21 个 MediaPipe-style hand landmarks。
- `recv_ts_ns`：monotonic receive timestamp，用于 stale 检查和 `dt`。
- `recv_time_unix_ns`：wall-clock timestamp，用于日志 / dataset 对齐。

常用方法：

```python
data.landmarks_np()  # np.ndarray, shape=(21, 3)
data.is_finite()
data.to_dict()
```

---

### 4.3 `EefTarget`

```python
@dataclass(frozen=True, slots=True)
class EefTarget:
    side: str
    position_world: Vec3
    quat_wxyz_world: Quat
    delta_pos_world: Vec3
    delta_quat_world: Quat
    sequence_id: int
    recv_ts_ns: int
    recv_time_unix_ns: int | None
    source_ts_ns: int | None
```

语义：

```text
EefTarget = robot world frame 下的末端目标位姿
```

常用方法：

```python
eef.position_world
eef.quat_wxyz_world
eef.quat_xyzw_world      # 给 ROS / scipy 风格接口用
eef.pos_quat_wxyz()
eef.pose_matrix_world()  # 4x4 transform matrix
```

---

### 4.4 `VRState`

```python
@dataclass(frozen=True, slots=True)
class VRState:
    data: HandData
    retargeting_joints: np.ndarray
    finger_curl: np.ndarray
    eef_target: EefTarget | None
    timestamp_ns: int
    is_fresh: bool
    valid: bool
```

语义：

```text
VRState = VRTrackerService 缓存的某一只手的最新状态
```

字段：

| 字段 | shape / 类型 | 含义 |
|---|---:|---|
| `data` | `HandData` | 原始 SDK/VR 数据。 |
| `retargeting_joints` | `(21, 3)` | wrist-relative hand skeleton。 |
| `finger_curl` | `(15,)` | 五指弯曲角，单位 rad。 |
| `eef_target` | `EefTarget | None` | 已标定且 fresh 时才有。 |
| `is_fresh` | `bool` | 当前帧是否通过 stale 检查。 |
| `valid` | `bool` | 当前帧是否适合外部控制使用。当前定义：`is_fresh and eef_target is not None`。 |

外部控制循环建议检查：

```python
state = vr.get_latest("right")
if state is None or not state.valid:
    continue
```

---

## 5. 核心函数接口

### 5.1 `hand_frame_to_data(frame)`

```python
from vr_tracker import hand_frame_to_data

data = hand_frame_to_data(frame)
```

作用：

```text
hand-tracking-sdk HandFrame
  -> HandData in FLU coordinates
```

转换内容：

- wrist position：Unity left-handed -> FLU。
- wrist quaternion：SDK `xyzw` -> 内部 `wxyz`。
- landmarks：逐点 Unity left-handed -> FLU。

---

### 5.2 `hand_data_to_retargeting_joints(data)`

```python
from vr_utils import hand_data_to_retargeting_joints

joints = hand_data_to_retargeting_joints(data)
```

输出：

```python
np.ndarray  # shape = (21, 3)
```

当前约定：

```text
origin = landmark[0] wrist
orientation = 保持 HandData.landmarks 所在 FLU frame
不估计 palm canonical frame
不使用 wrist_quat
不用于 EEF pose control
```

也就是：

```python
joints = landmarks - landmarks[0]
```

这与 LeFranX-style 分支一致：

```text
Arm branch: wrist pose -> EEF target
Hand branch: 21 landmarks -> hand retargeting
```

---

### 5.3 `hand_data_to_wrist_relative_joints(data)`

```python
from vr_utils import hand_data_to_wrist_relative_joints

joints = hand_data_to_wrist_relative_joints(data)
```

这是 `hand_data_to_retargeting_joints(data)` 当前内部调用的函数。语义更明确：只做 wrist-relative translation。

---

### 5.4 `hand_data_to_wrist_local_joints(data)`

```python
from vr_utils import hand_data_to_wrist_local_joints

local_joints = hand_data_to_wrist_local_joints(data)
```

输出：

```python
np.ndarray  # shape = (21, 3)
```

语义：

```text
landmarks - landmark[0]
再用 inverse(data.wrist_quat) 旋到 SDK wrist pose local frame
```

建议用途：debug / ablation / 可视化。默认 retargeting path 不使用它。

---

### 5.5 `hand_data_to_finger_curl_vector(data)`

```python
from vr_utils import hand_data_to_finger_curl_vector

curl = hand_data_to_finger_curl_vector(data)
```

输出：

```python
np.ndarray  # shape = (15,)
```

顺序：

```text
thumb 3 angles
index 3 angles
middle 3 angles
ring 3 angles
little 3 angles
```

角度定义：

```text
对于 finger chain 中的 interior joint：
angle = angle_between(incoming_bone, outgoing_bone)

straight finger ~= 0
larger angle = more curled
unit = radians
```

---

## 6. 核心类接口

### 6.1 `QuestHandReceiver`

```python
receiver = QuestHandReceiver(config)
```

职责：

```text
1. 从 hand-tracking-sdk 接收 HandFrame
2. 转成 HandData
3. 根据 wrist pose differential 计算 EEF target
```

常用接口：

```python
receiver.stream()                         # Iterator[HandData]
receiver.stop()
receiver.stats()                          # SDK ClientStats
receiver.reset_stats()

receiver.calibrate(data, home_pos, home_quat_wxyz)
receiver.clear_calibration(side=None)
receiver.compute_eef_target(data)

receiver.set_world_alignment(quat_wxyz)
receiver.set_world_alignment_yaw(yaw_rad)
receiver.set_world_alignment_matrix(R_world_from_flu)
receiver.set_workspace_limits(limits)
receiver.set_orientation_limits(limits)
receiver.set_max_orientation_angle(angle_rad)
```

同步 stream 示例：

```python
receiver = QuestHandReceiver(VRTrackerConfig(hand_filter="right"))

for data in receiver.stream():
    print(data.side, data.sequence_id, data.wrist_pos)
```

一般不建议真实 workflow 直接使用同步 stream。更推荐 `VRTrackerService` 后台线程。

---

### 6.2 `VRTrackerService`

```python
vr = VRTrackerService(receiver)
vr.start()
```

职责：

```text
后台线程持续读取 QuestHandReceiver.stream()
维护 latest state cache
外部 workflow 按需 get_latest(...)
```

常用接口：

```python
vr.start()
vr.stop()
vr.get_latest(side="right", copy_arrays=True)
vr.get_latest_all(copy_arrays=True)
vr.get_latest_data(side="right")
vr.get_latest_retargeting_joints(side="right", copy=True)
vr.get_latest_finger_curl_vector(side="right", copy=True)
vr.get_latest_eef_target(side="right")
vr.calibrate_from_latest(side, home_pos_world, home_quat_world)
vr.stats()
vr.sdk_stats()
vr.raise_if_failed()
```

推荐主循环：

```python
receiver = QuestHandReceiver(config)
vr = VRTrackerService(receiver)
vr.start()

while True:
    state = vr.get_latest("right")
    if state is None or not state.valid:
        continue

    joints = state.retargeting_joints
    curl = state.finger_curl
    eef = state.eef_target

    # 外部 dex-retargeting / IK / robot command
```

---

### 6.3 `VideoReturnService`

```python
video = VideoReturnService(
    signaling_host="0.0.0.0",
    signaling_port=8765,
    width=1280,
    height=720,
    fps=30,
)
video.start()
```

职责：

```text
RGB frame -> WebRTC video return -> Quest display
```

接口：

```python
video.start(timeout_s=5.0)
video.stop(timeout_s=5.0)
video.submit_frame(rgb)
video.is_running
```

`rgb` 要求：

```python
rgb.shape == (height, width, 3)
rgb.dtype == np.uint8
```

示例：

```python
video = VideoReturnService(width=1280, height=720, fps=30)
video.start()

while True:
    rgb = renderer.render_rgb()
    video.submit_frame(rgb)
```

视频回传是独立服务，不影响 `VRTrackerService` 的 state cache 逻辑。真实机器人中，视频帧提交最好放在独立相机 / 渲染线程，不要塞进 robot command loop。

---

## 7. 关键坐标系转换

### 7.1 原始 Quest / Unity left-handed 坐标

Quest / Unity 侧通常使用 Unity left-handed coordinate system。`hand-tracking-sdk` 输出的 wrist packet 是：

```text
x, y, z, qx, qy, qz, qw
```

landmarks packet 是：

```text
21 x (x, y, z)
```

SDK 文档中也说明 `HandFrame` 会包含 wrist pose、21 landmarks、`recv_ts_ns`、`source_ts_ns`、`sequence_id` 等元数据。

---

### 7.2 FLU 坐标系

本项目把 SDK 原始数据转换成 FLU 后存入 `HandData`：

```text
HandData.wrist_pos   # FLU
HandData.wrist_quat  # FLU, wxyz
HandData.landmarks   # FLU absolute positions
```

转换由 SDK 的 basis transform 完成：

```python
from hand_tracking_sdk.convert import BASIS_UNITY_LEFT_TO_FLU
```

内部函数：

```python
unity_lh_to_flu_position(...)
unity_lh_to_flu_quaternion(...)
```

---

### 7.3 Robot world 坐标系

机器人世界系由外部定义。`QuestHandReceiver.world_from_flu` 表示：

```text
FLU vector -> robot world vector
```

可以这样设置：

```python
receiver.set_world_alignment_yaw(yaw_rad)
receiver.set_world_alignment(quat_wxyz)
receiver.set_world_alignment_matrix(R_world_from_flu)
```

LeFranX-style 默认矩阵：

```python
from vr_utils import lefranx_vr_to_robot_matrix

receiver.set_world_alignment_matrix(lefranx_vr_to_robot_matrix())
```

矩阵语义：

```text
robot_x = vr_z
robot_y = -vr_x
robot_z = vr_y
```

---

### 7.4 EEF differential pose 转换

标定时记录：

```text
ref_hand_pos      = 当前 VR wrist position, FLU
ref_hand_quat     = 当前 VR wrist orientation, FLU, wxyz
home_pos_world    = 当前 robot EEF position, world
home_quat_world   = 当前 robot EEF orientation, world, wxyz
```

每一帧：

```text
delta_pos_flu = wrist_pos_t - ref_hand_pos

delta_quat_flu = wrist_quat_t * inverse(ref_hand_quat)
```

映射到 world：

```text
delta_pos_world = world_from_flu @ delta_pos_flu

delta_quat_world = R_world_from_flu * delta_quat_flu * R_world_from_flu^-1
```

作用到 robot home EEF：

```text
target_pos_world = home_pos_world + position_scale * delta_pos_world

target_quat_world = scaled(delta_quat_world, rotation_scale) * home_quat_world
```

之后再经过：

```text
linear speed limit
angular speed limit
workspace position clamp
relative RPY orientation clamp
max orientation angle clamp
```

---

### 7.5 Hand retargeting landmarks 转换

`retargeting_joints` 不是 EEF wrist frame。

当前定义：

```text
retargeting_joints = landmarks - landmarks[0]
```

也就是：

```text
origin = wrist landmark
orientation = FLU frame
shape = (21, 3)
```

这条线服务于 hand retargeting：

```text
retargeting_joints
  -> external build ref_value
  -> retargeting.retarget(ref_value)
  -> robot hand qpos
```

不要用 `retargeting_joints` 来计算 arm EEF pose。

---

## 8. 典型工作流

### 8.1 只读取 VR 最新状态

```python
from vr_tracker import QuestHandReceiver, VRTrackerConfig, VRTrackerService

config = VRTrackerConfig(hand_filter="right")
receiver = QuestHandReceiver(config)
vr = VRTrackerService(receiver)
vr.start()

try:
    while True:
        state = vr.get_latest("right")
        if state is None:
            continue
        print(state.data.sequence_id, state.data.wrist_pos, state.retargeting_joints.shape)
finally:
    vr.stop()
```

---

### 8.2 标定并输出 EEF target

```python
config = VRTrackerConfig(
    hand_filter="right",
    max_frame_age_s=0.2,
    max_linear_speed=0.25,
    max_angular_speed=1.0,
    workspace_limits=(-0.3, 0.3, -0.3, 0.3, -0.2, 0.4),
    orientation_limits=(-0.8, 0.8, -0.6, 0.6, -1.2, 1.2),
    max_orientation_angle=1.5,
)
receiver = QuestHandReceiver(config)
vr = VRTrackerService(receiver)
vr.start()

# 等第一帧
while vr.get_latest_data("right") is None:
    time.sleep(0.001)

# 当前机器人 EEF home pose
current_eef_pos = (0.4, 0.0, 0.3)
current_eef_quat_wxyz = (1.0, 0.0, 0.0, 0.0)

ok = vr.calibrate_from_latest("right", current_eef_pos, current_eef_quat_wxyz)
if not ok:
    raise RuntimeError("failed to calibrate")

while True:
    state = vr.get_latest("right")
    if state is None or not state.valid:
        continue

    eef = state.eef_target
    target_pos = eef.position_world
    target_quat = eef.quat_wxyz_world
```

---

### 8.3 接 dex-retargeting

`vr_tracker.py` 不直接 import 或调用 dex-retargeting。外部 workflow 自己构造 `ref_value`。

```python
state = vr.get_latest("right")
if state is None or not state.is_fresh:
    return

joints = state.retargeting_joints

retargeting_type = retargeting.optimizer.retargeting_type.lower()
indices = retargeting.optimizer.target_link_human_indices

if retargeting_type == "position":
    ref_value = joints[indices]
else:
    origin_indices = indices[0]
    task_indices = indices[1]
    ref_value = joints[task_indices] - joints[origin_indices]

hand_qpos = retargeting.retarget(ref_value)
```

如果同时需要手指弯曲特征：

```python
finger_curl = state.finger_curl  # shape=(15,)
```

---

### 8.4 接 arm IK

```python
state = vr.get_latest("right")
if state is None or not state.valid:
    return

eef = state.eef_target
arm_qpos = ik_solver.solve(
    target_pos=eef.position_world,
    target_quat_wxyz=eef.quat_wxyz_world,
)
```

---

### 8.5 完整 teleop 主循环示意

```python
from vr_tracker import QuestHandReceiver, VRTrackerConfig, VRTrackerService
from vr_utils import lefranx_vr_to_robot_matrix

config = VRTrackerConfig(
    hand_filter="right",
    max_frame_age_s=0.2,
    max_linear_speed=0.25,
    max_angular_speed=1.0,
    workspace_limits=(-0.3, 0.3, -0.3, 0.3, -0.2, 0.4),
    orientation_limits=(-0.8, 0.8, -0.6, 0.6, -1.2, 1.2),
    max_orientation_angle=1.5,
)

receiver = QuestHandReceiver(config)
receiver.set_world_alignment_matrix(lefranx_vr_to_robot_matrix())

vr = VRTrackerService(receiver)
vr.start()

try:
    # 机器人当前 EEF pose
    home_pos = robot.get_eef_position()
    home_quat = robot.get_eef_quat_wxyz()

    while vr.get_latest_data("right") is None:
        time.sleep(0.001)
    vr.calibrate_from_latest("right", home_pos, home_quat)

    while True:
        state = vr.get_latest("right")
        if state is None or not state.valid:
            robot.hold()
            continue

        joints = state.retargeting_joints
        eef = state.eef_target

        if retargeting_type == "position":
            ref_value = joints[indices]
        else:
            ref_value = joints[indices[1]] - joints[indices[0]]

        hand_qpos = retargeting.retarget(ref_value)
        arm_qpos = ik_solver.solve(
            target_pos=eef.position_world,
            target_quat_wxyz=eef.quat_wxyz_world,
        )

        robot.command(arm_qpos=arm_qpos, hand_qpos=hand_qpos)
finally:
    vr.stop()
```

---

## 9. 视频回传工作流

视频回传应当独立于 robot command loop。

```python
video = VideoReturnService(width=1280, height=720, fps=30)
video.start()

try:
    while True:
        rgb = camera.read_rgb()  # shape=(720, 1280, 3), dtype=uint8
        video.submit_frame(rgb)
finally:
    video.stop()
```

推荐：

```text
Thread A: VRTrackerService
Thread B: camera/render -> VideoReturnService.submit_frame(...)
Thread C: retargeting + IK + robot command
```

不要：

```text
robot command loop 内同步 render + submit_video_frame + IK + robot.command
```

---

## 10. Stats / diagnostics

### 10.1 VR stats

```python
stats = vr.stats()
print(stats)
```

字段：

```python
stats.running
stats.frames_received
stats.frames_dropped
stats.stale_frames
stats.update_hz
stats.last_frame_age_ms
stats.last_update_ns
stats.latest_sides
stats.last_error
```

### 10.2 SDK stats

```python
sdk_stats = vr.sdk_stats()
```

这来自 `hand-tracking-sdk` 的 `HTSClient.get_stats()`。

### 10.3 后台错误

```python
vr.raise_if_failed()
```

如果后台 loop 出现未恢复错误，会抛出 `RuntimeError`。

---

## 11. 注意事项

### 11.1 不要混淆 wrist pose 和 landmarks

正确分支：

```text
Arm / EEF:
  wrist_pos + wrist_quat

Hand retargeting:
  21 landmarks
```

错误做法：

```text
用 landmarks 估计 palm frame 再当作 EEF wrist pose
```

### 11.2 `retargeting_joints` 不是 robot hand qpos

```python
joints = state.retargeting_joints  # human hand 21x3 skeleton
```

它仍然是人手几何，不是机器人关节。你需要外部 retargeting：

```python
hand_qpos = retargeting.retarget(ref_value)
```

### 11.3 `valid` 的含义

```python
state.valid == state.is_fresh and state.eef_target is not None
```

如果你只做手指 retargeting，不控制 arm，可以检查：

```python
state is not None and state.is_fresh
```

如果要控制 arm，建议检查：

```python
state is not None and state.valid
```

### 11.4 真机部署要有额外 safety layer

当前代码只负责 VR data adapter 和 EEF target 生成，不是完整 robot safety controller。

真机上还需要外部：

```text
joint limit
velocity / acceleration / jerk limit
collision check
IK feasibility check
robot fault state check
physical emergency stop
hold-to-enable / clutch
```

### 11.5 视频回传会占资源

WebRTC 视频回传会占用 CPU / GPU / 内存带宽 / 网络。默认建议：

```text
720p30 起步
必要时降到 480p30 或 720p20
视频线程和 robot command loop 分开
```

### 11.6 时间戳语义

控制相关 stale / dt 使用：

```python
recv_ts_ns  # monotonic receive timestamp
```

日志 / dataset 对齐使用：

```python
recv_time_unix_ns
source_ts_ns
```

### 11.7 Quaternion 顺序

本项目内部一律：

```text
wxyz
```

如果接 ROS / scipy / 一些 robot SDK，需要确认是否使用 `xyzw`。`EefTarget` 提供：

```python
eef.quat_xyzw_world
```

---

## 12. 最小 smoke test

```python
from vr_tracker import VRTrackerConfig
from vr_utils import quat_from_rpy, quat_to_rpy

config = VRTrackerConfig(hand_filter="right")
q = quat_from_rpy(0.1, -0.2, 0.3)
rpy = quat_to_rpy(q)
print(config)
print(q)
print(rpy)
```

语法检查：

```bash
python3 -S -m py_compile vr_utils.py vr_tracker.py
```

---

## 13. 参考资料

- hand-tracking-sdk PyPI: https://pypi.org/project/hand-tracking-sdk/
- Unity XR Hands hand data model: https://docs.unity.cn/Packages/com.unity.xr.hands%401.4/manual/hand-data/xr-hand-data-model.html
- transforms3d quaternion docs: https://matthew-brett.github.io/transforms3d/reference/transforms3d.quaternions.html
- transforms3d Euler docs: https://matthew-brett.github.io/transforms3d/reference/transforms3d.euler.html
