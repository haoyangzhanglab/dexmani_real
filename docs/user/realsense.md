# RealSense 单相机驱动与点云工具使用说明

> **相关文档**：L515 SDK 编译安装见 [L515 SDK.md](L515%20SDK.md)。

本文档对应当前项目中的 `realsense.py` 与 `pointcloud_utils.py` 设计。整体风格与 `xarm7.py`、`xhand.py` 保持一致：硬件驱动层尽量轻量、接口显式、默认状态简单、调试信息通过 `full=True` 返回，点云构造放在独立工具层。

---

## 1. RealSense 如何与本地台式机建立通信

### 1.1 基本连接关系

RealSense 相机通常通过 USB 连接到本地台式机。Python 侧使用 Intel RealSense SDK 2.0 的 Python wrapper：`pyrealsense2`。典型流程是：创建 `rs.pipeline()`，配置 depth/color stream，调用 `pipeline.start(config)` 启动相机，通过 `wait_for_frames()` 读取帧，最后调用 `pipeline.stop()` 释放设备。

推荐连接前先做以下检查：

```bash
lsusb | grep -i realsense
```

若能看到 Intel / RealSense 相关设备，说明系统层面已经识别到相机。

### 1.2 安装依赖

一般 Python 环境中需要：

```bash
pip install pyrealsense2 numpy
```

如果只使用 `realsense.py`，不需要安装 `torch`、`open3d` 或 `opencv-python`。这些依赖只与点云采样、可视化或调试工具有关。

如果需要使用 `pointcloud_utils.py` 中的可视化功能，再安装：

```bash
pip install open3d opencv-python
```

### 1.3 枚举相机

`RealSenseCamera.list_cameras()` 会列出当前连接的 RealSense 设备：

```python
from realsense import RealSenseCamera

for cam in RealSenseCamera.list_cameras():
    print(cam)
```

典型返回字段包括：

```python
{
    "serial": "123456789",
    "name": "Intel RealSense D435",
    "firmware": "...",
    "product_line": "D400",
}
```

如果只接一个相机，可以让 `serial_number=None` 自动选择第一个设备。若后续有多相机，必须显式指定 `serial_number`，避免 USB 插拔顺序导致相机角色错位。

### 1.4 最小连通测试

```python
from realsense import RealSenseCamera, RealSenseConfig

cam = RealSenseCamera(RealSenseConfig())

if not cam.connect():
    raise RuntimeError("RealSense connect failed")

state = cam.get_state()
print(state["color"].shape if state["color"] is not None else None)
print(state["depth"].shape)
print(state["timestamp"])

cam.disconnect()
```

默认输出：

```python
{
    "color": np.ndarray | None,   # H x W x 3, uint8, RGB
    "depth": np.ndarray,          # H x W, float32, meter
    "timestamp": float,           # host timestamp
}
```

---

## 2. `realsense.py` 的参数物理意义、主要 API 与返回值说明

### 2.1 设计定位

`realsense.py` 是单相机硬件驱动。它只负责稳定输出 RGB-D 与相机标定信息，不负责生成 policy point cloud，也不负责多相机同步、数据录制、OpenCV 可视化或策略预处理。

当前边界为：

```text
RealSenseCamera
    输入：相机配置
    输出：color / depth / timestamp / intrinsics / depth_scale

pointcloud_utils.py
    输入：depth + K + color + transform
    输出：point cloud
```

### 2.2 `RealSenseConfig`

推荐配置结构如下：

```python
@dataclass
class RealSenseConfig:
    camera_name: str = "realsense"
    serial_number: str | None = None

    depth_resolution: tuple[int, int] = (640, 480)
    color_resolution: tuple[int, int] = (640, 480)
    fps: int = 30

    enable_color: bool = True
    align_mode: str = "depth_to_color"
    color_format: str = "rgb"

    depth_hole_filling: bool = False
    enable_global_time: bool = True
    warmup_frames: int = 10
    timeout_ms: int = 1000

    frame_name: str | None = None
    transform: np.ndarray = np.eye(4)
    verbose: bool = False
```

字段说明：

| 参数                 | 含义                                   | 建议                                                     |
| -------------------- | -------------------------------------- | -------------------------------------------------------- |
| `camera_name`        | 相机逻辑名                             | 单相机默认 `"realsense"`，多相机时可用 `front/wrist/top` |
| `serial_number`      | RealSense 设备序列号                   | 单相机可为 `None`；多相机必须显式指定                    |
| `depth_resolution`   | 深度图分辨率，格式 `(width, height)`   | 默认 `(640, 480)`                                        |
| `color_resolution`   | 彩色图分辨率，格式 `(width, height)`   | 默认 `(640, 480)`                                        |
| `fps`                | 采集帧率                               | 默认 `30`                                                |
| `enable_color`       | 是否启用彩色流                         | RGB-D 策略通常设为 `True`                                |
| `align_mode`         | 对齐方式                               | 推荐 `"depth_to_color"`                                  |
| `color_format`       | 输出颜色顺序                           | 默认 `"rgb"`                                             |
| `depth_hole_filling` | 是否启用 RealSense hole filling filter | 调试时可开，训练/部署需保持一致                          |
| `enable_global_time` | 是否尝试开启 RealSense global time     | 多相机/日志场景更有意义                                  |
| `warmup_frames`      | 启动后丢弃前几帧                       | 默认 `10`，用于曝光/深度稳定                             |
| `timeout_ms`         | 等待帧超时时间                         | 默认 `1000 ms`                                           |
| `frame_name`         | 相机坐标系名称                         | 默认随 align mode 自动设置                               |
| `transform`          | 相机外参，4x4 齐次矩阵                 | 通常表示 `T_world_camera` 或 `T_base_camera`             |

### 2.3 `align_mode`

支持三种对齐方式：

| 模式               | 含义                                 | 适用场景                        |
| ------------------ | ------------------------------------ | ------------------------------- |
| `"depth_to_color"` | 将 depth 对齐到 color 分辨率与坐标系 | 最常用；生成彩色点云更方便      |
| `"color_to_depth"` | 将 color 对齐到 depth                | 需要保留 depth 原始视角时使用   |
| `"none"`           | 不做对齐                             | 只用 depth 或自行处理外参时使用 |

如果需要从 RGB-D 生成彩色点云，推荐使用：

```python
align_mode="depth_to_color"
```

这样 `state["color"]` 与 `state["depth"]` 的图像尺寸一致，`pointcloud_utils.rgbd_to_pointcloud()` 才能直接使用 RGB 作为点颜色。

### 2.4 主要 API

#### `connect() -> bool`

启动 RealSense pipeline。内部流程：

```text
创建 rs.pipeline
创建 rs.config
绑定 serial_number，如果提供
启用 depth stream
启用 color stream，如果 enable_color=True
启动 pipeline
创建 aligner
读取 depth_scale
读取 intrinsics
warmup
设置 connected_flag=True
```

用法：

```python
cam = RealSenseCamera(RealSenseConfig())
ok = cam.connect()
```

#### `disconnect() -> None`

停止 pipeline 并释放相机资源。

```python
cam.disconnect()
```

#### `stop() -> bool`

传感器侧的 `stop()` 语义是停止采集。当前实现等价于：

```python
cam.disconnect()
return True
```

注意：这不是机器人急停，不涉及执行器控制。

#### `get_state(full: bool = False) -> dict`

读取一帧 RGB-D 状态。

默认 `full=False`：

```python
state = cam.get_state()
```

返回：

```python
{
    "color": np.ndarray | None,  # H x W x 3, uint8, RGB by default
    "depth": np.ndarray,         # H x W, float32, meters
    "timestamp": float,          # host timestamp
}
```

`full=True`：

```python
state = cam.get_state(full=True)
```

额外返回：

```python
{
    "depth_raw": np.ndarray,       # uint16 raw z16
    "camera_timestamp": float,     # RealSense frame timestamp, seconds
    "host_timestamp": float,       # time.time()
    "frame_id": int,

    "K": np.ndarray,               # 3 x 3 camera intrinsic matrix
    "intr": np.ndarray,            # [fx, fy, cx, cy]
    "intrinsics_info": dict,
    "depth_scale": float,

    "camera_name": str,
    "serial_number": str | None,
    "align_mode": str,
    "frame_name": str,
    "transform": np.ndarray,       # 4 x 4

    "connected_flag": bool,
    "error_state": bool,
    "last_error_message": str,
    "meta": dict,
}
```

#### `is_connected() -> bool`

判断相机是否处于可读状态：

```python
if cam.is_connected():
    state = cam.get_state()
```

#### `is_error() -> bool`

判断本地相机驱动是否处于错误状态。

#### `clear_error() -> bool`

清除 Python 驱动层本地错误标志。若 pipeline 已断开，仍需重新 `connect()`。

#### `list_cameras() -> list[dict]`

枚举当前连接的 RealSense 相机。

```python
cameras = RealSenseCamera.list_cameras()
```

#### `get_intrinsics() -> np.ndarray`

返回当前有效内参矩阵：

```python
K = cam.get_intrinsics()
```

格式：

```python
[[fx, 0,  cx],
 [0,  fy, cy],
 [0,  0,  1 ]]
```

#### `get_depth_scale() -> float`

返回 RealSense 原始 Z16 depth 到米单位的尺度：

```python
depth_m = depth_raw.astype(np.float32) * depth_scale
```

### 2.5 错误与 stale state

如果 `get_state()` 读取失败，驱动会设置：

```python
error_state = True
last_error_message = str(error)
```

如果存在上一帧有效状态，可能返回带 `stale=True` 的旧状态。上层控制/记录模块应在使用状态前检查：

```python
state = cam.get_state(full=True)
if state.get("stale", False):
    print("warning: RealSense returned stale frame")
```

---

## 3. `pointcloud_utils.py` 点云工具用法

### 3.1 设计定位

`pointcloud_utils.py` 负责纯几何变换，不关心数据来自 RealSense、仿真还是离线数据集。RealSense 驱动不应该直接 import `pointcloud_utils.py`，点云构造由上层显式调用。

推荐调用链：

```text
RealSenseCamera.get_state(full=True)
    -> color / depth / K / transform
pointcloud_utils.rgbd_to_pointcloud(...)
    -> pointcloud, shape (N, 6), xyzrgb
```

### 3.2 `PointCloudConfig`

```python
@dataclass
class PointCloudConfig:
    npoints: int | None = 1024
    min_depth: float | None = 0.05
    max_depth: float | None = 1.5
    sampling: str = "random"       # "none", "random", "fps", "first"
    workspace: tuple[float, float, float, float, float, float] | None = None
    device: str = "cpu"
    return_tensor: bool = False
```

字段说明：

| 参数            | 含义                                                 |
| --------------- | ---------------------------------------------------- |
| `npoints`       | 输出点数；`None` 表示不固定点数                      |
| `min_depth`     | 最小有效深度，单位 m                                 |
| `max_depth`     | 最大有效深度，单位 m                                 |
| `sampling`      | 采样方式：`none/random/fps/first`                    |
| `workspace`     | 工作空间裁剪 `[x_min,y_min,z_min,x_max,y_max,z_max]` |
| `return_tensor` | 是否返回 torch Tensor；默认返回 numpy array          |

### 3.3 RGB-D 转点云

```python
from dexmani_real.utils.pointcloud_utils import PointCloudConfig, rgbd_to_pointcloud

state = cam.get_state(full=True)

pcd = rgbd_to_pointcloud(
    depth=state["depth"],        # meters
    K=state["K"],
    rgb=state["color"],
    T_out_camera=state["transform"],
    config=PointCloudConfig(
        npoints=1024,
        min_depth=0.05,
        max_depth=1.5,
        sampling="random",
        return_tensor=False,
    ),
)

print(pcd.shape)  # (1024, 6), columns = x,y,z,r,g,b
```

输出格式：

```text
pointcloud[:, 0:3] = xyz, unit = meter
pointcloud[:, 3:6] = rgb, range = [0, 1]
```

### 3.4 深度单位规则

当前设计中：

```text
realsense.py 输出的 state["depth"] 已经是 meters
pointcloud_utils.rgbd_to_pointcloud() 默认 depth 输入也是 meters
```

如果你只有 RealSense 原始 Z16 深度图，应显式传入 `depth_scale`：

```python
from dexmani_real.utils.pointcloud_utils import depth_raw_to_meters

depth = depth_raw_to_meters(depth_raw, depth_scale)
```

不要在点云函数里隐式假设 `uint16 depth` 一定是毫米。RealSense raw depth 的正确尺度应来自相机的 `depth_scale`。

### 3.5 使用 workspace 裁剪

```python
pcd = rgbd_to_pointcloud(
    depth=state["depth"],
    K=state["K"],
    rgb=state["color"],
    T_out_camera=state["transform"],
    config=PointCloudConfig(
        workspace=(-0.3, -0.4, 0.0, 0.5, 0.4, 0.8),
        npoints=2048,
        sampling="random",
        return_tensor=False,
    ),
)
```

workspace 的坐标系取决于 `T_out_camera`：

```text
T_out_camera is None:
    workspace 位于 camera frame

T_out_camera = T_base_camera:
    workspace 位于 base/world frame
```

### 3.6 点云可视化

```python
from dexmani_real.utils.pointcloud_utils import vis_point_cloud

vis_point_cloud(pcd, voxel_size=0.005)
```

该函数依赖 `open3d`，只用于调试，不应放进实时控制主循环。

---

## 4. 常见用法

### 4.1 只读取 RGB-D

```python
cam = RealSenseCamera(RealSenseConfig())
cam.connect()

while True:
    state = cam.get_state()
    color = state["color"]
    depth = state["depth"]
    timestamp = state["timestamp"]

cam.disconnect()
```

### 4.2 读取 RGB-D 并生成点云

```python
cam = RealSenseCamera(RealSenseConfig(align_mode="depth_to_color"))
cam.connect()

state = cam.get_state(full=True)

pcd = rgbd_to_pointcloud(
    depth=state["depth"],
    K=state["K"],
    rgb=state["color"],
    T_out_camera=state["transform"],
    config=PointCloudConfig(npoints=1024, return_tensor=False),
)

cam.disconnect()
```

### 4.3 指定相机序列号

```python
cam = RealSenseCamera(RealSenseConfig(
    camera_name="front",
    serial_number="123456789",
))
cam.connect()
```

多相机时必须显式指定 `serial_number`，后续不要依赖自动枚举顺序。

---

## 5. L515 材质、光照与对象选择

> 本节内容从 L515 SDK.md 迁移。L515 是 LiDAR depth camera，适合反射较稳定、表面偏漫反射的物体。

### 5.1 不适合的材质

| 材质/表面 | 问题 | 建议 |
|-----------|------|------|
| 透明玻璃、透明塑料、亚克力、水杯 | 深度可能穿透、缺失或落到背景上 | 不作为初期主训练对象；可贴哑光胶带或换哑光替代物 |
| 镜面金属、亮面陶瓷、反光桌面 | 入射光不一定反射回相机，深度空洞或跳变 | 改用哑光表面，调整视角 |
| 黑色/深色低反射物体 | 反射信号弱，深度噪声增加 | 换浅色物体，增加辅助标记 |
| 高光塑料包装、保鲜膜、塑料袋 | 透明 + 高光，深度和分割都不稳定 | 不建议作为早期学习对象 |
| 毛巾、衣物、软袋 | 形变大，3D 状态难定义 | 需要单独的柔性物体策略 |
| 细杆、细线、薄片边缘 | 点云稀疏，边缘容易断裂 | 近距离、多视角、避免作为关键抓取依据 |

### 5.2 不适合的光照

| 场景光 | 问题 | 建议 |
|--------|------|------|
| 太阳直射、窗边强日光 | 环境红外会降低深度质量 | 避开窗户，拉窗帘，使用稳定室内光 |
| LED/PWM 灯频闪 | RGB 画面可能出现水波纹、滚动暗带 | 设置 Power Line Frequency 为 50Hz 或 60Hz |
| 强背光 | RGB 曝光和白平衡漂移 | 光源放在相机同侧或侧前方 |
| 反光背景 | 深度和 RGB 都不稳定 | 使用哑光背景板和哑光桌面 |
| 频繁变化光照 | 训练数据分布漂移 | 固定灯光、曝光、白平衡 |

### 5.3 不适合的对象

| 对象 | 问题 | 建议 |
|------|------|------|
| 透明杯子、玻璃瓶 | 深度不可依赖 | 初期不要作为主要操作对象 |
| 镜面工具、金属杯 | 点云跳变、空洞 | 贴哑光胶带或换物体 |
| 黑色小零件 | 深度缺失概率高 | 换颜色或增加视觉标记 |
| 高反光包装盒 | 分割和深度都不稳定 | 使用哑光替代物 |
| 软袋、毛巾、衣物 | 状态空间复杂 | 后期单独建模 |
| 细小零件 | L515 点云分辨率和空洞影响定位 | 近距离、多视角、提高质量筛选 |

L515 有最小有效深度距离，过近物体不可靠。实际机器人操作中建议让目标处于约 `0.35 m ~ 1.2 m` 的稳定工作范围。

---

## 6. 3D 模仿学习数据采集建议

### 6.1 固定相机-机器人关系

必须稳定保存并复用：camera intrinsics、camera extrinsics (`camera_to_robot_base` 或 `camera_to_ee`)、depth scale、RGB-D 对齐方式、分辨率、FPS、桌面高度、物体初始区域、光照配置。建议每次任务开始前保存一次相机状态，采集中每帧保存时间戳。

### 6.2 不要让策略学习坏深度

坏深度会导致：抓取点错误、物体中心漂移、点云空洞、接触前距离估计错误、轨迹回放碰撞、策略学到错误 affordance。

建议过滤：`depth == 0` 的点、`< 0.25 m` 的点、`> 任务最大距离` 的点（如 `> 1.5 m`）、confidence 低的点、mask 外的点、反光边缘离群点。

### 6.3 固定 RGB 设置

为了减少视觉分布漂移，建议固定：Power Line Frequency（中国大陆 50Hz；北美 60Hz）、White Balance Auto: Off、Auto Exposure: Off（稳定采集时）、Color/Depth resolution、FPS。

如果 RGB 像水波一样波动，优先检查 `Power Line Frequency`，而不是 SDK 安装。

### 6.4 采集任务从易到难

1. 哑光、浅色、刚体、大物体
2. 不同颜色、不同形状刚体
3. 轻微反光物体
4. 小物体、细长物体、多物体遮挡
5. 透明、高反光、软体物体

早期不要直接使用：透明杯子、玻璃瓶、镜面金属、黑色小零件、塑料袋、毛巾衣物——否则很难区分是感知失败还是策略失败。

### 6.5 保留失败样本，标注失败原因

失败原因建议分为：`depth_missing`、`segmentation_error`、`gripper_slip`、`collision`、`occlusion`、`operator_error`、`object_moved`、`lighting_failure`、`calibration_drift`。训练时可以先只用成功样本；调试和后续提升时再使用失败样本做筛选、分类或数据增强。

### 6.6 单视角和多视角建议

单个 L515 推荐放置在斜上方 30°~60°，能看到桌面、目标物体和夹爪，不正对窗户/镜面反光面，尽量不被机械臂长期遮挡。如果条件允许，建议加一个辅助相机（主视角：策略输入；辅助视角：标注、调试、遮挡补偿）。

---

## 7. 设计原则回顾

`xarm7.py`、`xhand.py`、`realsense.py` 应保持一致的工程风格：

```text
xarm7.py:
    执行器，get_state + send_action

xhand.py:
    执行器，get_state + send_action

realsense.py:
    传感器，get_state，无 send_action
```

共同原则：

```text
独立单文件硬件驱动
清晰 dataclass config
connect / disconnect / stop
is_connected / is_error / clear_error
默认 get_state 轻量
full=True 返回调试和几何信息
不做 shared memory
不做 multiprocessing
不做 policy preprocessing
不做 dataset recording
不做默认 GUI 可视化
```

一句话总结：

```text
RealSenseCamera 负责把 RGB-D 与标定信息稳定读出来；
pointcloud_utils 负责把 RGB-D 转成点云；
策略模块再决定如何 crop、sample、merge、normalize。
```

---

## 8. 参考资料

- Intel RealSense SDK 2.0 / librealsense GitHub: https://github.com/realsenseai/librealsense
- pyrealsense2 Python package: https://pypi.org/project/pyrealsense2/
- RealSense depth-to-color alignment example: https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py
- RealSense projection/deprojection documentation: https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0