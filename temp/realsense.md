# RealSense MinFix 使用说明

对应文件：

- `realsense_refactor_minfix.py`
- `pcd_utils_refactor_minfix.py`

这版代码是一个轻量的 RealSense RGB-D 采集与点云生成 wrapper。主设计目标是：**接口少、路径直、依赖可降级、适合机器人实时实验**。

---

## 1. 功能概览

支持能力：

1. 枚举 RealSense 相机。
2. 根据 serial 启动指定相机。
3. 读取 RGB-D 帧。
4. 将 RealSense 原始 depth 转为米制深度。
5. 当 `enable_color=True` 时，将 color 对齐到 depth 坐标系。
6. 将 depth 根据 intrinsics 反投影为点云。
7. 可选执行 depth mask、pose transform、workspace crop、固定点数采样和 Open3D 可视化。

坐标系约定：

- `pose=None`：点云在 RealSense depth camera frame 下。
- `pose=T_out_camera`：点云会被变换到 `T_out_camera` 对应的坐标系，例如 robot base / world。

---

## 2. 依赖

必需依赖：

```bash
pip install numpy opencv-python torch pyrealsense2
```

可选依赖：

```bash
pip install open3d      # 仅 vis_point_cloud / demo 按 p 可视化时需要
pip install pytorch3d   # 用于 farthest point sampling；未安装时自动退化为 random sampling
```

---

## 3. 文件放置

```text
.
├── realsense_refactor_minfix.py
└── pcd_utils_refactor_minfix.py
```

`realsense_refactor_minfix.py` 内部导入：

```python
from pcd_utils_refactor_minfix import make_rays, rgbd_to_pointcloud
```

如果你想改回原文件名：

```text
pcd_utils_refactor_minfix.py  ->  pcd_utils_refactor.py
realsense_refactor_minfix.py  ->  realsense_refactor.py
```

同步把导入改为：

```python
from pcd_utils_refactor import make_rays, rgbd_to_pointcloud
```

---

## 4. 快速使用

### 4.1 查看相机

```python
from realsense_refactor_minfix import RealSense

cams = RealSense.list_cameras()
print(cams)
```

返回格式：

```python
[
    {
        "serial": "1234567890",
        "name": "Intel RealSense D435",
        "firmware": "5.xx.xx.xx",
    }
]
```

### 4.2 读取 RGB-D

```python
from realsense_refactor_minfix import RealSense

serial = RealSense.list_cameras()[0]["serial"]

with RealSense(serial=serial, enable_color=True) as cam:
    color, depth, timestamp = cam.read()

print(color.shape)  # (H, W, 3), RGB, uint8
print(depth.shape)  # (H, W), float32, meter
print(timestamp)    # second, RealSense frame timestamp
```

注意：`color` 是 RGB，不是 OpenCV 默认 BGR；`depth` 已经是 meter，不要再除以 1000。

### 4.3 直接读取点云

```python
with RealSense(serial=serial, enable_color=True) as cam:
    points, colors = cam.pointcloud(
        bound=None,
        npoints=1024,
        min_depth=0.1,
        max_depth=1.5,
        device="cpu",
        return_tensor=True,
    )

print(points.shape)  # torch.Size([1024, 3])
print(colors.shape)  # torch.Size([1024, 3]) or None
```

输出：

- `points`: `(N, 3)`，float32，单位 meter。
- `colors`: `(N, 3)`，uint8，RGB；若没有 color，则为 `None`。

### 4.4 从已读取 RGB-D 生成点云

机器人主循环中更推荐这种方式，因为可以保留同一帧的 timestamp：

```python
with RealSense(serial=serial, enable_color=True) as cam:
    color, depth, timestamp = cam.read()

    points, colors = cam.pointcloud_from_frame(
        color,
        depth,
        bound=[-0.3, 0.3, -0.3, 0.3, 0.1, 0.8],
        npoints=2048,
        min_depth=0.1,
        max_depth=1.0,
        device="cuda:0",
        return_tensor=True,
    )
```

### 4.5 只用 depth

```python
with RealSense(serial=serial, enable_color=False) as cam:
    color, depth, timestamp = cam.read()
    points, colors = cam.pointcloud()

print(color)   # None
print(colors)  # None
```

### 4.6 命令行 demo

```bash
python realsense_refactor_minfix.py
python realsense_refactor_minfix.py --no-color
python realsense_refactor_minfix.py --serial 1234567890
python realsense_refactor_minfix.py --npoints 2048 --min-depth 0.1 --max-depth 1.2
```

带 workspace crop：

```bash
python realsense_refactor_minfix.py \
  --npoints 2048 \
  --bound -0.3 0.3 -0.3 0.3 0.1 0.8
```

窗口快捷键：

- `q`: 退出。
- `p`: 使用 Open3D 显示当前点云，需要安装 `open3d`。

---

## 5. `RealSense` 类接口

### 5.1 构造函数

```python
RealSense(
    serial,
    depth_resolution=(640, 480),
    color_resolution=(1280, 720),
    fps=30,
    depth_hole_filling=False,
    pose=None,
    enable_color=True,
)
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `serial` | `str` | 必填 | 相机序列号 |
| `depth_resolution` | `tuple[int, int]` | `(640, 480)` | depth 分辨率，格式 `(width, height)` |
| `color_resolution` | `tuple[int, int]` | `(1280, 720)` | color 分辨率，格式 `(width, height)` |
| `fps` | `int` | `30` | depth/color stream FPS |
| `depth_hole_filling` | `bool` | `False` | 是否启用 RealSense hole filling filter |
| `pose` | array-like or `None` | `None` | 4x4 点云变换矩阵 |
| `enable_color` | `bool` | `True` | 是否启用 color stream |

说明：

- 参数里的 resolution 是 `(width, height)`。
- NumPy 返回的 `depth.shape` / `color.shape` 是 `(height, width)`。
- 当 `enable_color=True` 时，内部使用 `rs.align(rs.stream.depth)`，即把 color 对齐到 depth viewport。

### 5.2 `start()`

```python
def start(self):
    ...
```

启动 RealSense pipeline。主要行为：

1. 绑定指定 serial。
2. 启用 depth stream。
3. 若 `enable_color=True`，启用 color stream。
4. 读取 `depth_scale`。
5. 初始化 depth intrinsics。
6. 预热若干帧。

通常不手动调用，推荐：

```python
with RealSense(serial=serial) as cam:
    color, depth, timestamp = cam.read()
```

### 5.3 `stop()`

```python
def stop(self):
    ...
```

停止 RealSense pipeline。使用 `with RealSense(...) as cam:` 时会自动调用。

### 5.4 `read()`

```python
def read(self, timeout_ms=5000):
    ...
```

读取一帧 RGB-D。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `timeout_ms` | `int` | `5000` | 等待 frameset 的超时时间，单位 ms |

返回：

```python
color, depth, timestamp
```

| 返回值 | 类型 | shape | 说明 |
|---|---|---|---|
| `color` | `np.ndarray` or `None` | `(H, W, 3)` | RGB，uint8；`enable_color=False` 时为 `None` |
| `depth` | `np.ndarray` | `(H, W)` | float32，单位 meter |
| `timestamp` | `float` | scalar | RealSense frame timestamp，单位 second |

鲁棒性行为：

- 读不到 depth frame：抛出 `RuntimeError`。
- `enable_color=True` 但读不到 color frame：抛出 `RuntimeError`。
- depth intrinsics 变化时，自动更新 `self.depth_intr` 并清空 `rays_cache`。

### 5.5 `pointcloud_from_frame()`

```python
def pointcloud_from_frame(
    self,
    color,
    depth,
    *,
    bound=None,
    npoints=1024,
    min_depth=0.1,
    max_depth=5.0,
    device="cpu",
    return_tensor=True,
):
    ...
```

从一帧 RGB-D 生成点云。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `color` | `np.ndarray` or `None` | 必填 | RGB 图像，shape `(H, W, 3)` |
| `depth` | `np.ndarray` | 必填 | 深度图，shape `(H, W)`，单位 meter |
| `bound` | list/tuple or `None` | `None` | workspace crop，格式 `[x0, x1, y0, y1, z0, z1]` |
| `npoints` | `int` or `None` | `1024` | 输出点数；`None` 表示返回全部有效点 |
| `min_depth` | `float` | `0.1` | 最小有效深度，meter |
| `max_depth` | `float` | `5.0` | 最大有效深度，meter |
| `device` | `str` | `"cpu"` | PyTorch device，例如 `"cpu"` / `"cuda:0"` |
| `return_tensor` | `bool` | `True` | 是否返回 torch tensor |

返回：

```python
points, colors
```

| 返回值 | `return_tensor=True` | `return_tensor=False` | 说明 |
|---|---|---|---|
| `points` | `torch.Tensor` | `np.ndarray` | `(N, 3)`，meter |
| `colors` | `torch.Tensor` or `None` | `np.ndarray` or `None` | `(N, 3)`，RGB，uint8 |

内部处理顺序：

```text
depth + rays
  -> camera-frame points
  -> depth range mask
  -> optional pose transform
  -> optional workspace crop
  -> optional fixed-size sampling
```

### 5.6 `pointcloud()`

```python
def pointcloud(
    self,
    *,
    bound=None,
    npoints=1024,
    min_depth=0.1,
    max_depth=5.0,
    device="cpu",
    return_tensor=True,
):
    ...
```

直接读取一帧并转点云。等价于：

```python
color, depth, _ = cam.read()
points, colors = cam.pointcloud_from_frame(color, depth, ...)
```

适合快速测试；机器人主循环中更推荐显式 `read()` 后调用 `pointcloud_from_frame()`。

### 5.7 `rays()`

```python
def rays(self, shape, device="cpu"):
    ...
```

返回或缓存当前 depth intrinsics 对应的 camera rays。

| 参数 | 类型 | 说明 |
|---|---|---|
| `shape` | tuple | depth shape，通常是 `depth.shape`，即 `(H, W)` |
| `device` | str | PyTorch device |

返回：

```python
torch.Tensor  # shape (H, W, 3)
```

通常不需要外部手动调用。

### 5.8 `list_cameras()`

```python
@staticmethod
def list_cameras():
    ...
```

返回当前连接的 RealSense 相机列表：

```python
[
    {"serial": str, "name": str, "firmware": str},
    ...
]
```

---

## 6. `pcd_utils_refactor_minfix.py` 函数接口

### 6.1 `make_rays()`

```python
def make_rays(height, width, intr, device="cpu"):
    ...
```

根据 pinhole intrinsics 为每个像素生成 ray。

| 参数 | 类型 | 说明 |
|---|---|---|
| `height` | `int` | 图像高度 H |
| `width` | `int` | 图像宽度 W |
| `intr` | array-like | camera intrinsics matrix，shape `(3, 3)` |
| `device` | `str` | PyTorch device |

返回：

```python
torch.Tensor  # shape (H, W, 3)
```

### 6.2 `depth_to_points()`

```python
def depth_to_points(depth, rays, device="cpu"):
    ...
```

将 depth map 反投影成点云。

输入：

- `depth`: `(H, W)`，单位 meter。
- `rays`: `(H, W, 3)`。

返回：

```python
torch.Tensor  # shape (H * W, 3)
```

若 `rays.shape[:2] != depth.shape[:2]`，抛出 `ValueError`。

### 6.3 `image_to_colors()`

```python
def image_to_colors(color, device="cpu"):
    ...
```

将 RGB 图像 flatten 成点云颜色。

输入：

```python
color.shape == (H, W, 3)
```

返回：

```python
torch.Tensor  # shape (H * W, 3), dtype=torch.uint8
```

### 6.4 `mask_depth()`

```python
def mask_depth(points, colors=None, min_depth=0.1, max_depth=5.0):
    ...
```

过滤非法点和 depth range 外的点。

保留条件：

```python
torch.isfinite(points).all(dim=1)
points[:, 2] > min_depth
points[:, 2] < max_depth
```

返回：

```python
points, colors
```

### 6.5 `transform_points()`

```python
def transform_points(points, transform=None):
    ...
```

应用 4x4 齐次变换。

输入：

- `points`: `(N, 3)`
- `transform`: `(4, 4)` or `None`

若 `transform=None`，直接返回原点云。

### 6.6 `crop_points()`

```python
def crop_points(points, colors=None, bound=None):
    ...
```

按 workspace bound 裁剪点云。

`bound` 格式：

```python
[x0, x1, y0, y1, z0, z1]
```

保留条件：

```python
x0 < x < x1
y0 < y < y1
z0 < z < z1
```

### 6.7 `sample_points()`

```python
def sample_points(points, colors=None, npoints=1024):
    ...
```

将点云采样到固定数量。

| 情况 | 行为 |
|---|---|
| `npoints is None` | 返回全部点 |
| `npoints <= 0` | 抛出 `ValueError` |
| `N == 0` | 抛出 `ValueError` |
| `N < npoints` | 保留所有点，再随机重复补齐 |
| `N >= npoints` 且安装 PyTorch3D | farthest point sampling |
| `N >= npoints` 且未安装 PyTorch3D | random sampling |

### 6.8 `rgbd_to_pointcloud()`

```python
def rgbd_to_pointcloud(
    depth,
    intr=None,
    color=None,
    *,
    rays=None,
    bound=None,
    npoints=1024,
    min_depth=0.1,
    max_depth=5.0,
    transform=None,
    device="cpu",
    return_tensor=True,
):
    ...
```

RGB-D 到点云的主函数。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `depth` | ndarray/tensor | 必填 | `(H, W)`，meter |
| `intr` | array-like or `None` | `None` | camera intrinsics；未传 `rays` 时必填 |
| `color` | ndarray/tensor or `None` | `None` | `(H, W, 3)`，RGB |
| `rays` | tensor or `None` | `None` | `(H, W, 3)`；传入后不重复计算 |
| `bound` | list/tuple or `None` | `None` | `[x0, x1, y0, y1, z0, z1]` |
| `npoints` | int or `None` | `1024` | 输出点数 |
| `min_depth` | float | `0.1` | 最小深度，meter |
| `max_depth` | float | `5.0` | 最大深度，meter |
| `transform` | array-like or `None` | `None` | `(4, 4)` 点云变换矩阵 |
| `device` | str | `"cpu"` | PyTorch device |
| `return_tensor` | bool | `True` | 是否返回 torch tensor |

返回：

```python
points, colors
```

### 6.9 `vis_point_cloud()`

```python
def vis_point_cloud(points, voxel_size=None):
    ...
```

使用 Open3D 可视化点云。

输入：

- `points.shape == (N, 3)`：显示 xyz。
- `points.shape == (N, 6)`：前三维 xyz，后三维 RGB。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `voxel_size` | float or `None` | `None` | 若不为 `None`，先做 voxel downsample |

---

## 7. 典型机器人实验用法

### 7.1 接 policy observation

```python
from realsense_refactor_minfix import RealSense

serial = RealSense.list_cameras()[0]["serial"]

with RealSense(serial=serial, enable_color=True) as cam:
    while True:
        color, depth, timestamp = cam.read()

        try:
            points, colors = cam.pointcloud_from_frame(
                color,
                depth,
                bound=[-0.4, 0.4, -0.4, 0.4, 0.05, 0.8],
                npoints=1024,
                min_depth=0.05,
                max_depth=1.0,
                device="cuda:0",
                return_tensor=True,
            )
        except ValueError:
            continue

        obs = {
            "point_cloud": points,   # (1024, 3), float32, meter
            "rgb": colors,          # (1024, 3), uint8, RGB
            "timestamp": timestamp,
        }

        # action = policy(obs)
```

### 7.2 使用 camera-to-base 外参

```python
import numpy as np
from realsense_refactor_minfix import RealSense

T_base_camera = np.eye(4, dtype=np.float32)  # 替换为真实标定外参

with RealSense(serial=serial, pose=T_base_camera) as cam:
    points_base, colors = cam.pointcloud(
        bound=[-0.5, 0.5, -0.5, 0.5, 0.0, 0.8],
        npoints=2048,
        return_tensor=True,
    )
```

代码不会判断 `pose` 的语义，只执行齐次变换。若你传入 `T_base_camera`，输出就在 base frame；若传入 `T_world_camera`，输出就在 world frame。

### 7.3 返回全部有效点

```python
points, colors = cam.pointcloud(
    npoints=None,
    min_depth=0.1,
    max_depth=1.5,
    return_tensor=True,
)
```

适合调试、可视化或离线处理；实时 policy 通常建议固定 `npoints`。

---

## 8. 注意事项

### 8.1 color/depth 对齐

当 `enable_color=True` 时，模块使用：

```python
rs.align(rs.stream.depth)
```

即把 color 对齐到 depth viewport。因此 `read()` 返回的 `color.shape[:2]` 应该和 `depth.shape[:2]` 一致。

如果你绕过 `read()`，手动给 `pointcloud_from_frame()` 或 `rgbd_to_pointcloud()` 传入外部 RGB-D，需要确保 color/depth 已对齐且 shape 一致。

### 8.2 depth 单位

`read()` 返回的 `depth` 已经乘过 `depth_scale`，单位是 meter。

不要再次除以 1000。

### 8.3 RGB/BGR

`read()` 返回的是 RGB：

```python
color[..., 0]  # R
color[..., 1]  # G
color[..., 2]  # B
```

OpenCV 显示前需要转 BGR：

```python
cv2.imshow("rgb", color[..., ::-1])
```

### 8.4 `bound` 的坐标系

`bound` 在 `pose` 之后生效：

```text
points_camera -> pose transform -> crop by bound
```

所以：

- `pose=None`：`bound` 是 camera frame 下的 crop。
- `pose=T_base_camera`：`bound` 是 base frame 下的 crop。

格式固定为：

```python
[x_min, x_max, y_min, y_max, z_min, z_max]
```

### 8.5 空点云

以下情况可能导致空点云：

- `min_depth/max_depth` 太窄。
- `bound` 太小。
- `bound` 所在坐标系和当前点云坐标系不一致。
- 相机被遮挡。
- 场景中大面积无深度。
- `pose` 传反，导致 crop 全部裁掉。

此时会抛出：

```text
ValueError: No valid points after depth mask/crop. Check depth range, crop bound, and camera pose.
```

主循环里建议跳过当前帧或复用上一帧 observation。

### 8.6 timestamp

`read()` 返回的 `timestamp` 来自 RealSense frame timestamp，并转为 second。它不是系统 wall time。

如果要和 robot state、VR tracker 或 hand command 做严格同步，建议上层同时记录：

```python
import time
host_time = time.time()
```

### 8.7 PyTorch3D 和 Open3D 都不是强依赖

- 没有 PyTorch3D：固定点数采样退化为 random sampling。
- 没有 Open3D：采集和点云生成仍可用，只是不能调用 `vis_point_cloud()`。

---

## 9. 常见错误

### 9.1 `No RealSense camera found.`

没有检测到相机。检查 USB、权限、设备占用，以及 RealSense Viewer 是否能看到设备。

### 9.2 `Failed to get depth frame from RealSense.`

当前 frameset 中没有有效 depth frame。可能是设备掉线、stream 配置不支持、USB 带宽不足或 pipeline 尚未稳定。

可尝试降低分辨率或 FPS：

```bash
python realsense_refactor_minfix.py --depth-res 640x480 --color-res 640x480 --fps 15
```

### 9.3 `color shape must match depth shape`

说明 RGB 和 depth 没有对齐，或传入了外部分辨率不一致的图像。

若使用本模块的 `read()`，通常不会出现该问题；若使用外部 RGB-D，请先做 alignment。

### 9.4 `No valid points after depth mask/crop`

优先检查：

1. `min_depth/max_depth` 是否过窄。
2. `bound` 是否写错坐标系。
3. `pose` 是否传反。
4. 相机视野内是否有有效深度。

调试时可先关闭 crop：

```python
points, colors = cam.pointcloud(bound=None, min_depth=0.05, max_depth=2.0)
```

---

## 10. 推荐配置

### 桌面调试

```python
RealSense(
    serial=serial,
    depth_resolution=(640, 480),
    color_resolution=(1280, 720),
    fps=30,
    enable_color=True,
)
```

```python
points, colors = cam.pointcloud(
    npoints=1024,
    min_depth=0.1,
    max_depth=1.5,
    return_tensor=True,
)
```

### 实时机器人 policy

```python
RealSense(
    serial=serial,
    depth_resolution=(640, 480),
    color_resolution=(640, 480),
    fps=30,
    enable_color=True,
)
```

```python
points, colors = cam.pointcloud_from_frame(
    color,
    depth,
    bound=[-0.4, 0.4, -0.4, 0.4, 0.05, 0.8],
    npoints=1024,
    min_depth=0.05,
    max_depth=1.0,
    device="cuda:0",
    return_tensor=True,
)
```

---

## 11. 官方参考

- `pyrealsense2.align`: https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.align.html
- `pyrealsense2.depth_sensor.get_depth_scale`: https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.depth_sensor.html
- `pyrealsense2.depth_frame`: https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.depth_frame.html
- RealSense API How-To: https://dev.intelrealsense.com/docs/api-how-to
