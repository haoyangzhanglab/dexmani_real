# VR Teleoperation Input Utilities Guide

本文档对应五个轻量模块：

```text
quest_hand_tracker.py       # Quest/HTS 数据接收：wrist pose + 21 landmarks
arm_wrist_mapper.py         # VR wrist 相对运动 -> target EEF pose
quest_hand_visualizer.py    # Rerun 可视化：wrist、landmarks、骨架、腕部坐标轴
hand_retarget.py            # dex-retargeting: 21 landmarks → 12 DOF XHand 关节角
xhand_ref_adapter.py        # LeFranX 风格小指参考自适应适配器
```

设计目标：输入层简单、接口稳定、robot-agnostic。当前版本不包含 IK、不发送机器人控制命令。

> **相关文档**：XHand 硬件控制见 [xhand.md](xhand.md)，xArm7 硬件控制见 [xarm7.md](xarm7.md)，手部重定向与 retargeting 配置见本文档 Section 6。

---

## 1. 前置条件: ADB 连接与调试

### 1.1 目标状态

连接成功后执行 `adb devices -l` 应看到：

```text
340YC10GCD0RZV    device usb:2-1 product:xxx model:xxx device:xxx transport_id:1
```

关键字段是 `device`。常见异常状态：`no permissions`（Linux/udev 权限问题）、`unauthorized`（Quest 未授权 ADB RSA key）、`offline`（需重启 adb server 或重新插拔 USB）。

### 1.2 前置条件

**Quest 侧**：Meta 账号已启用开发者身份、已开启 Developer Mode、使用支持数据传输的 USB-C 线（不要只用充电线）、头显保持开机/解锁/亮屏、首次连接时在头显中允许 USB debugging（推荐选 "Always allow from this computer"）。

**Ubuntu/Debian 侧**：

```bash
sudo apt update
sudo apt install adb android-sdk-platform-tools-common
```

确认当前用户在 `plugdev` 组：

```bash
groups
# 如果没有 plugdev，执行 sudo usermod -aG plugdev $USER，然后重新登录
```

### 1.3 一键检查流程

```bash
# 1. 安装 adb 和常见 udev rules
sudo apt update
sudo apt install adb android-sdk-platform-tools-common

# 2. 清理 adb
sudo pkill adb || true
adb kill-server

# 3. 插上 Quest 后检查
lsusb
adb start-server
adb devices -l
```

如果出现 `no permissions`，添加 udev rule（vendor id 通常为 `2833`）：

```bash
sudo tee /etc/udev/rules.d/51-android-local.rules > /dev/null <<'EOF'
SUBSYSTEM=="usb", ATTR{idVendor}=="2833", MODE="0660", GROUP="plugdev", TAG+="uaccess"
EOF

sudo chmod a+r /etc/udev/rules.d/51-android-local.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
sudo systemctl restart udev

adb kill-server
adb start-server
adb devices -l
```

然后拔插 USB，并在 Quest 中允许 USB debugging。

### 1.4 连接状态速查

| adb 状态 | 含义 | 下一步 |
|----------|------|--------|
| 无设备 | 系统没识别到 USB 设备 | 换线、换口、检查 Quest 是否开机 |
| `no permissions` | Linux udev 权限未匹配 | 添加 udev rule |
| `unauthorized` | Quest 未授权 ADB RSA key | 在头显里允许 USB debugging |
| `offline` | adb server / 设备状态异常 | 重启 adb server，重新插拔 |
| `device` | 正常 | 可以跑 SDK、安装 APK、看 logcat |

### 1.5 常用 ADB 命令

```bash
adb devices -l                          # 查看设备列表
adb shell getprop ro.product.model      # 查看设备型号
adb shell                               # 进入设备 shell
adb install -r path/to/app.apk          # 安装 APK
adb logcat | grep -iE "hand|tracking"   # 查看 hand tracking 日志
adb logcat -d > quest_logcat.txt        # 截取日志到文件
```

**注意**：不要长期使用 `sudo adb`——可能启动 root 用户的 adb server，导致后续普通用户调用 SDK 时状态不一致。如果已误用，执行：`sudo pkill adb || true && adb kill-server && adb start-server`，并 `sudo chown -R $USER:$USER ~/.android` 修复本地 adb key 权限。

---

## 2. 软件安装与通信连接

### 2.1 Python 依赖

基础依赖：

```bash
pip install numpy transforms3d hand-tracking-sdk
```

可视化依赖：

```bash
pip install rerun-sdk
```

也可以直接安装 SDK 的可视化 extra：

```bash
pip install "hand-tracking-sdk[visualization]"
```

### 2.2 Quest 端 App

需要在 Meta Quest 上运行 **Hand Tracking Streamer / HTS**。该 App 会将以下数据流式发送到 PC：

```text
6-DoF wrist pose
21 hand landmarks
```

当前代码推荐使用 **TCP server** 模式：PC 端监听 `0.0.0.0:8000`，Quest/HTS 主动发送数据。

### 2.3 USB wired TCP / ADB reverse

推荐先用 USB + ADB reverse 调试，稳定性更高。

```bash
adb devices
adb reverse tcp:8000 tcp:8000
adb reverse --list
```

Quest/HTS 端通常设置：

```text
Protocol: TCP
Host: localhost
Port: 8000
Hand: Right
```

PC 端代码默认：

```python
tracker = QuestHandTracker(
    transport="tcp_server",
    host="0.0.0.0",
    port=8000,
    hand_side="right",
    output_frame="flu",
)
```

### 2.4 Wireless TCP / UDP

无线模式下，Quest/HTS 端 `Host` 应填写 PC 的局域网 IPv4 地址，例如：

```text
Protocol: TCP
Host: 192.168.x.x
Port: 8000
```

一般建议：

```text
优先：USB + TCP + ADB reverse，用于稳定调试与采集
其次：Wireless TCP，用于无线但仍需要较稳定传输
谨慎：Wireless UDP，延迟低但更容易受网络抖动影响
```

---

## 3. 文件说明

### 3.1 `quest_hand_tracker.py`

职责：

```text
连接 HTS stream
接收 HandFrame
转换坐标系
统一四元数顺序为 wxyz
缓存 latest frame
提供状态诊断
```

不负责：

```text
VR-to-robot 标定
IK
XHand retargeting
clutch / recenter
机器人安全限幅
数据录制
```

### 3.2 `arm_wrist_mapper.py`

职责：

```text
reset 时记录 wrist 初始 pose 与 robot EEF 初始 pose
map 时根据 wrist 相对 reset 的运动输出 target_eef_pose
```

输出只包含：

```python
{
    "pos": np.ndarray,        # shape (3,)
    "quat_wxyz": np.ndarray,  # shape (4,)
}
```

### 3.3 `quest_hand_visualizer.py`

职责：

```text
用 Rerun 可视化 QuestHandTracker 输出的 frame dict
显示 wrist 点、landmarks、手部骨架、可选 wrist 坐标轴
```

### 3.4 `hand_retarget.py`

职责：

```text
加载 dex-retargeting RetargetingConfig (DexPilot)
接收 21 个 MediaPipe hand landmarks
输出 12 DOF XHand 关节角
处理 retargeting → sapien 关节顺序映射
```

不负责：VR 数据接收、机器人控制、数据录制。

### 3.5 `xhand_ref_adapter.py`

职责：

```text
LeFranX 风格小指参考适配器
动态调整 retargeting 的 pinky reference 值
根据当前拇指-食指距离自适应调整小指姿态
```

---

## 4. 关键参数物理意义与默认值

### 4.1 `QuestHandTracker`

| 参数              |         默认值 | 物理/工程含义                                                |
| ----------------- | -------------: | ------------------------------------------------------------ |
| `transport`       | `"tcp_server"` | 通信模式。PC 端作为 server 监听 Quest/HTS 数据。可选取决于 SDK 支持：`tcp_server`、`tcp_client`、`udp`。 |
| `host`            |    `"0.0.0.0"` | PC 端监听地址。`0.0.0.0` 表示监听所有网卡。                  |
| `port`            |         `8000` | 通信端口。需要和 HTS App 中设置一致。                        |
| `hand_side`       |      `"right"` | 接收哪只手。当前推荐右手。                                   |
| `output_frame`    |        `"flu"` | 输出坐标系。`flu = forward-left-up`；也支持 `unity`、`rfu`。 |
| `max_frame_age_s` |         `0.20` | latest frame 最大可接受年龄。超过该时间，`get_latest()` 返回 `None`。 |
| `strict`          |        `False` | SDK 错误策略。调试阶段推荐 tolerant。                        |
| `verbose`         |        `False` | 是否打印简短连接信息。                                       |

### 4.2 `ArmWristMapper`

| 参数               |      默认值 | 物理/工程含义                                                |
| ------------------ | ----------: | ------------------------------------------------------------ |
| `pos_scale`        |       `1.0` | VR wrist 位移到 EEF 位移的比例系数。大于 1 会放大手部位移。  |
| `rot_scale`        |       `1.0` | VR wrist 旋转到 EEF 旋转的比例系数。小于 1 会降低旋转灵敏度。 |
| `vr_to_base_rot`   | `np.eye(3)` | 将 VR 坐标系中的运动增量旋转到机器人 base 坐标系。只处理方向，不处理平移。 |
| `eef_delta_bounds` |      `None` | EEF 相对 reset 初始位置的 XYZ 安全边界，shape `(3, 2)`。     |

`eef_delta_bounds` 示例：

```python
eef_delta_bounds = np.array([
    [-0.3, 0.3],  # x direction relative to reset EEF position
    [-0.3, 0.3],  # y direction
    [-0.2, 0.2],  # z direction
])
```

含义：

```text
target_eef_pos - reset_eef_pos
```

会被限制在这个 box 内。

### 4.3 `QuestHandVisualizer`

| 参数           |               默认值 | 含义                        |
| -------------- | -------------------: | --------------------------- |
| `app_id`       | `"quest-hand-debug"` | Rerun viewer 中的应用名。   |
| `spawn`        |               `True` | 是否自动启动 Rerun viewer。 |
| `show_axes`    |               `True` | 是否显示 wrist 坐标轴。     |
| `point_radius` |              `0.012` | landmarks 点半径。          |
| `wrist_radius` |               `0.02` | wrist 点半径。              |
| `axis_length`  |               `0.08` | wrist 坐标轴长度。          |

---

## 5. 主要 API 接口与返回值定义

### 5.1 `QuestHandTracker.connect()`

启动后台接收线程。

```python
tracker.connect()
```

重复调用不会重复启动。

### 5.2 `QuestHandTracker.disconnect()`

停止后台接收线程并释放本地状态。

```python
tracker.disconnect()
```

### 5.3 `QuestHandTracker.get_latest(max_age_s=None)`

非阻塞读取 latest frame。

```python
frame = tracker.get_latest()
```

返回：

```python
None | dict
```

当没有帧或帧过旧时，返回 `None`。

frame schema：

```python
{
    "side": str,
    "wrist_pos": np.ndarray,         # shape (3,), meter
    "wrist_quat_wxyz": np.ndarray,   # shape (4,), quaternion order wxyz
    "landmarks": np.ndarray,         # shape (21, 3), meter
    "recv_ts_ns": int,
    "source_ts_ns": int | None,
    "sequence_id": int | None,
    "source_frame_seq": int | None,
    "coordinate_frame": str,
}
```

### 5.4 `QuestHandTracker.read(timeout_s=1.0)`

阻塞等待新 frame。

```python
frame = tracker.read(timeout_s=1.0)
```

行为：

```text
如果有新 frame：返回 frame dict
如果超时：抛 TimeoutError
```

注意：`read()` 会根据 `(sequence_id, recv_ts_ns)` 避免重复返回同一帧。

### 5.5 `QuestHandTracker.get_status()`

返回状态诊断 dict。

```python
status = tracker.get_status()
```

返回字段：

```python
{
    "started": bool,
    "running": bool,
    "transport": str,
    "host": str,
    "port": int,
    "hand_side": str,
    "output_frame": str,
    "received_frames": int,
    "ignored_events": int,
    "malformed_frames": int,
    "sdk_lines_received": int | None,
    "sdk_parse_errors": int | None,
    "sdk_dropped_lines": int | None,
    "sequence_id": int | None,
    "frame_age_s": float | None,
    "last_error": str | None,
}
```

### 5.6 `ArmWristMapper.reset(...)`

记录当前 wrist pose 和当前 robot EEF pose，作为后续相对映射的零点。

```python
mapper.reset(
    wrist_pos=frame["wrist_pos"],
    wrist_quat_wxyz=frame["wrist_quat_wxyz"],
    eef_pos=current_eef_pos,
    eef_quat_wxyz=current_eef_quat_wxyz,
)
```

必须先 reset，才能调用 `map()` 得到有效输出。

### 5.7 `ArmWristMapper.map(...)`

将当前 wrist pose 映射为 target EEF pose。

```python
target = mapper.map(
    wrist_pos=frame["wrist_pos"],
    wrist_quat_wxyz=frame["wrist_quat_wxyz"],
)
```

返回：

```python
None | {
    "pos": np.ndarray,        # shape (3,)
    "quat_wxyz": np.ndarray,  # shape (4,)
}
```

如果没有 reset，返回 `None`。

### 5.8 `QuestHandVisualizer.log_frame(frame)`

将 frame 可视化到 Rerun。

```python
visualizer.log_frame(frame)
```

---

## 6. `hand_retarget.py` 使用说明

`hand_retarget.py` 基于 dex-retargeting 的 DexPilot 优化器，将 21 个 MediaPipe hand landmarks 映射为 12 DOF XHand 关节角。

### 6.1 `XHandRetargeter`

```python
from dexmani_real.teleop.hand_retarget import XHandRetargeter

retargeter = XHandRetargeter(
    hand_type="right",
    retargeting_type="dexpilot",
    enable_ref_adapter=True,
    pinky_extension_range=(0.03, 0.07),
    pinky_scale=(1.2, 2.2),
    pinky_blend=1.0,
)
```

参数：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `hand_type` | `"right"` | 手型：`"right"` 或 `"left"` |
| `retargeting_type` | `"dexpilot"` | 优化器类型：`"dexpilot"`（VR 遥操作）或 `"position"`（数据集后处理） |
| `enable_ref_adapter` | `True` | 是否启用 LeFranX 风格小指参考适配器 |
| `pinky_extension_range` | `(0.03, 0.07)` | 小指伸展检测的拇指-食指距离范围 (m) |
| `pinky_scale` | `(1.2, 2.2)` | 小指 landmark 缩放范围 |
| `pinky_blend` | `1.0` | 小指自适应混合权重 |

### 6.2 `retarget(hand_joint_pos) -> np.ndarray | None`

输入 21 个手部 landmark 坐标（shape `(21, 3)`，单位 m），输出 12 DOF XHand 关节角（shape `(12,)`，单位 rad）。

```python
frame = tracker.get_latest()
if frame is not None:
    qpos = retargeter.retarget(frame["landmarks"])
    # qpos: shape (12,), rad, XHand hardware order
```

返回 `None` 表示输入无效或 retargeting 失败。

### 6.3 `XHandRefAdapter`（xhand_ref_adapter.py）

小指参考适配器（LeFranX 风格），动态调整 retargeting 的 pinky reference 值：

- 根据当前拇指指尖到食指指尖的距离判断小指是否伸展
- 在小指伸展/弯曲时对 landmarks 17-20（pinky chain）做 scale 自适应
- 通过 `pinky_blend` 与原始 reference 混合，避免姿态突变

通常在 `XHandRetargeter` 内部自动启用，不需要单独调用。

---

## 7. 推荐使用流程

### 7.1 仅验证 Quest 数据

```python
from quest_hand_tracker import QuestHandTracker

tracker = QuestHandTracker(output_frame="flu", verbose=True)

with tracker:
    while True:
        frame = tracker.get_latest()
        if frame is not None:
            print(frame["wrist_pos"], frame["landmarks"].shape)
```

### 7.2 可视化调试

```python
from quest_hand_tracker import QuestHandTracker
from quest_hand_visualizer import QuestHandVisualizer

tracker = QuestHandTracker(output_frame="flu")
visualizer = QuestHandVisualizer(show_axes=True)

with tracker:
    while True:
        frame = tracker.get_latest()
        if frame is not None:
            visualizer.log_frame(frame)
```

### 7.3 wrist -> target EEF pose

```python
import numpy as np
from quest_hand_tracker import QuestHandTracker
from arm_wrist_mapper import ArmWristMapper

tracker = QuestHandTracker(output_frame="flu")
mapper = ArmWristMapper(
    pos_scale=1.0,
    rot_scale=1.0,
    eef_delta_bounds=np.array([
        [-0.3, 0.3],
        [-0.3, 0.3],
        [-0.2, 0.2],
    ]),
)

with tracker:
    frame = tracker.read()

    mapper.reset(
        wrist_pos=frame["wrist_pos"],
        wrist_quat_wxyz=frame["wrist_quat_wxyz"],
        eef_pos=np.array([0.4, 0.0, 0.3]),
        eef_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    while True:
        frame = tracker.get_latest()
        if frame is None:
            continue

        target_eef_pose = mapper.map(
            frame["wrist_pos"],
            frame["wrist_quat_wxyz"],
        )
```

---

## 8. 注意事项

### 8.1 四元数顺序

项目内部统一使用：

```text
wxyz
```

字段名也显式写为：

```python
wrist_quat_wxyz
eef_quat_wxyz
quat_wxyz
```

不要和 SDK 原始的 `xyzw` 顺序混用。

### 8.2 坐标系

默认输出：

```python
output_frame="flu"
```

`flu` 表示：

```text
x: forward
y: left
z: up
```

如果 robot base 坐标系与 FLU 不一致，应在 `ArmWristMapper` 中设置：

```python
vr_to_base_rot
```

不要在 `QuestHandTracker` 里写 robot-specific 坐标变换。

### 8.3 `ArmWristMapper` 必须 reset

`ArmWristMapper` 使用 reset-relative mapping，不是 absolute mapping。必须先记录：

```text
reset wrist pose
reset EEF pose
```

然后才能把 wrist 相对运动映射为 target EEF pose。

### 8.4 `get_latest()` 和 `read()` 不同

```text
get_latest(): 非阻塞，控制循环用，可能返回 None
read(): 阻塞等待新 frame，调试/初始化用
```

控制循环建议用 `get_latest()`，不要让机器人控制因 VR 暂时断流而阻塞。

### 8.5 当前代码不包含 robot safety policy

当前代码只提供输入和几何目标，不负责最终机器人安全。真实机器人上还需要额外层处理：

```text
stale frame hold
clutch / enable / recenter
per-step EEF delta limit
velocity / acceleration / jerk limit
IK fail hold
workspace absolute bounds
emergency stop
```

这些建议放到后续 `VRTeleopPolicy` 或 robot controller 层，不要塞进当前两个基础类。

### 8.6 不建议用右手 pinch 做 clutch

右手 landmarks 后续会用于 dex_retargeting 控制灵巧手。如果用右手 pinch 同时做 clutch，容易和真实抓取动作冲突。后续可以考虑：

```text
键盘快捷键
脚踏开关
左手 gesture
Quest controller button
```

---

## 9. 参考资料

- hand-tracking-sdk: https://github.com/wengmister/hand-tracking-sdk
- Hand Tracking Streamer / Quest wrist tracker: https://github.com/wengmister/quest-wrist-tracker
- dex-retargeting: https://github.com/dexsuite/dex-retargeting
- vr-dex-retargeting: https://github.com/wengmister/vr-dex-retargeting
- LeFranX: https://github.com/wengmister/LeFranX
- LeVR paper: https://arxiv.org/abs/2509.14349
- Rerun: https://github.com/rerun-io/rerun
- 相关文档：[xarm7.md](xarm7.md) | [xhand.md](xhand.md) | [realsense.md](realsense.md)