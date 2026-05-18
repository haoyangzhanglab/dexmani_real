# RealSense V2: Thin RGB-D + Point Cloud Wrapper

这版代码面向机器人真机实验中的 RealSense 硬件交互，目标是 **直、薄、少包装**。只包含两个代码文件：

```text
utils.py       # RGB-D 几何、点云处理、可视化小工具、obs dict 打包
realsense.py   # RealSense 硬件交互、内参/ray 缓存、点云便捷接口、example 测试函数
```

不使用 dataclass，不引入线程、不引入 recorder、不引入 policy wrapper。

---

## 1. 标准数据格式

本版本统一采用下面的数据协议：

| 字段         |       shape | dtype                           | 单位 / 范围               | 说明                                             |
| ------------ | ----------: | ------------------------------- | ------------------------- | ------------------------------------------------ |
| `rgb`        | `(H, W, 3)` | `np.uint8`                      | `0–255`                   | **RGB** 通道顺序，保持 OpenCV 读取后已转换的现状 |
| `depth`      |    `(H, W)` | `np.uint16`                     | **mm**                    | 毫米深度图；无效深度为 `0`                       |
| `pointcloud` |    `(N, 6)` | `torch.float32` 或 `np.float32` | XYZ: **m**；RGB: `[0, 1]` | 单个数组/张量，列顺序为 `[x, y, z, r, g, b]`     |

重要约定：

```text
rgb:
    uint8 RGB, 0~255

depth:
    uint16 millimeter
    例如 depth[v, u] = 530 表示 530 mm = 0.53 m

pointcloud:
    float32 XYZRGB
    pointcloud[:, 0:3] 是 XYZ，单位 meter
    pointcloud[:, 3:6] 是 RGB，float32，范围 [0, 1]
```

RealSense 原始 Z16 depth 会先通过 `depth_scale` 转为米，再统一转成 `uint16` 毫米输出。Intel RealSense 的 `get_depth_scale()` 返回 raw depth 到 meters 的比例，SDK 的 3D 坐标也以 meters 表示。

---

## 2. 功能架构

```text
RealSense.start()
  -> configure depth/color streams
  -> config.resolve() 检查配置
  -> pipeline.start()
  -> enable global_time_enabled if supported
  -> read depth_scale
  -> read active intrinsics
  -> warmup

RealSense.read()
  -> wait_for_frames()
  -> optional align_to depth/color
  -> optional hole filling
  -> raw z16 depth -> meters -> uint16 millimeters
  -> BGR -> RGB
  -> return color, depth_mm, timestamp

RealSense.pointcloud_from_frame()
  -> cached rays
  -> utils.rgbd_to_pointcloud()
      -> depth_mm -> meters internally
      -> depth_to_points
      -> filter_depth in meters
      -> optional T_out_camera transform
      -> workspace crop in meters
      -> sample_points
      -> pack as XYZRGB float32, shape (N, 6)

RealSense.get_obs()
  -> read RGB-D
  -> optional point cloud
  -> pack as dict
```

---

## 3. 坐标系约定

### 3.1 RealSense 相机坐标系

RealSense optical frame 约定：

```text
origin: active stream 的 optical center / camera center
+X:     图像中向右
+Y:     图像中向下
+Z:     从相机向前，指向被观察物体
unit:   meter
```

像素坐标为 `(u, v)`：

```text
u: 图像水平方向，向右增大
v: 图像竖直方向，向下增大
```

反投影公式：

```text
[x, y, z]^T = depth_m * K^-1 [u, v, 1]^T
```

注意：虽然 `read()` 返回的 `depth` 是 `uint16 mm`，但点云反投影内部会自动转成 `meter`。

### 3.2 `align_to`

```python
align_to="depth"  # default
align_to="color"
align_to="none"
```

含义：

| align_to  | 行为               | active intrinsics | 推荐场景                       |
| --------- | ------------------ | ----------------- | ------------------------------ |
| `"depth"` | color 对齐到 depth | depth intrinsics  | 点云、几何操作、workspace crop |
| `"color"` | depth 对齐到 color | color intrinsics  | RGB-D image policy / VLA       |
| `"none"`  | 不做 SDK alignment | depth intrinsics  | 只用 depth、低延迟调试         |

注意：点云反投影使用的是当前 depth 图像坐标系下的 K，所以 `align_to="color"` 时，depth 已 warp 到 color viewport，active K 会变成 color intrinsics。

### 3.3 `T_out_camera` 和 workspace

```text
T_out_camera is None:
    pointcloud[:, 0:3] 在 camera frame
    workspace 也应该是 camera frame

T_out_camera is not None:
    pointcloud[:, 0:3] 先变换到 out/world/base frame
    workspace 也应该是 out/world/base frame
```

workspace 格式固定为：

```python
[x_min, x_max, y_min, y_max, z_min, z_max]
```

workspace 的单位始终是 **meter**。

---

## 4. 主要 API

### 4.1 初始化

```python
from realsense import RealSense

cam = RealSense(
    serial=None,                    # None 时自动选择第一台相机
    depth_resolution=(640, 480),
    color_resolution=(640, 480),
    fps=30,
    enable_color=True,
    align_to="depth",              # "depth" | "color" | "none"
    depth_hole_filling=False,
    T_out_camera=None,              # 4x4, e.g. T_base_camera
    enable_global_time=True,
    warmup_frames=10,
)
```

推荐使用 context manager：

```python
with RealSense(serial=None) as cam:
    color, depth, timestamp = cam.read()
```

---

### 4.2 读取 RGB-D

```python
color, depth, timestamp = cam.read()
```

返回：

```text
color:     (H, W, 3), np.uint8, RGB, enable_color=False 时为 None
depth:     (H, W), np.uint16, 单位 millimeter
timestamp: RealSense frame timestamp, 单位 second
```

返回 dict：

```python
frame = cam.read(return_dict=True)
```

包含：

```python
{
    "rgb": color,                 # np.uint8 RGB, 0~255
    "depth": depth,               # np.uint16 mm
    "timestamp": timestamp,
    "host_time": host_time,
    "intrinsics": K,
    "intrinsics_info": {...},
    "depth_scale": depth_scale,
    "meta": {...},
}
```

---

### 4.3 获取内参和 depth scale

```python
K = cam.get_intrinsics()           # 3x3 active intrinsics
info = cam.get_intrinsics_info()   # fx/fy/cx/cy/width/height
depth_scale = cam.get_depth_scale()
```

`K` 是 active intrinsics，语义跟随 `align_to`：

```text
align_to="depth" -> depth intrinsics
align_to="color" -> color intrinsics
align_to="none"  -> depth intrinsics
```

---

### 4.4 从已读取 RGB-D 生成点云

```python
color, depth, timestamp = cam.read()

pointcloud = cam.pointcloud_from_frame(
    color,
    depth,
    workspace=[0.2, 0.8, -0.4, 0.4, 0.0, 0.5],
    npoints=1024,
    min_depth=0.05,
    max_depth=1.5,
    sampling="random",             # "none" | "random" | "fps" | "first"
    device="cuda:0",
    return_tensor=True,
)
```

输出：

```text
pointcloud: (N, 6), float32, torch.Tensor 或 np.ndarray
pointcloud[:, 0:3]: XYZ, 单位 meter
pointcloud[:, 3:6]: RGB, float32, 范围 [0, 1]
```

---

### 4.5 直接读取点云

```python
pointcloud = cam.pointcloud(
    workspace=None,
    npoints=1024,
    sampling="random",
    device="cpu",
)
```

这等价于：

```python
color, depth, timestamp = cam.read()
pointcloud = cam.pointcloud_from_frame(color, depth, ...)
```

机器人主循环中更推荐显式 `read()`，这样可以保留同一帧的 timestamp。

---

### 4.6 打包 obs dict

```python
obs = cam.get_obs(
    mode="full",                   # "rgbd" | "pointcloud" | "full"
    workspace=[0.2, 0.8, -0.4, 0.4, 0.0, 0.5],
    npoints=1024,
    sampling="random",
    device="cuda:0",
    return_tensor=True,
)
```

三种 mode：

| mode           | 内容                          | 推荐场景                    |
| -------------- | ----------------------------- | --------------------------- |
| `"rgbd"`       | RGB-D + metadata，不生成点云  | VLA / RGB policy / 快速采集 |
| `"pointcloud"` | pointcloud + metadata         | 3D policy                   |
| `"full"`       | RGB-D + pointcloud + metadata | debug / 开发                |

obs 结构：

```python
{
    "rgb": color,                 # np.uint8 RGB, 0~255
    "depth": depth,               # np.uint16 mm
    "timestamp": timestamp,
    "host_time": host_time,
    "intrinsics": K,
    "intrinsics_info": info,
    "depth_scale": depth_scale,
    "pointcloud": pointcloud,     # (N, 6), float32, XYZ(m)+RGB[0,1]
    "meta": {
        "serial": serial,
        "frame_id": frame_id,
        "align_to": align_to,
        "depth_unit": "mm",
        "depth_dtype": "uint16",
        "pointcloud_format": "xyzrgb",
        "pointcloud_xyz_unit": "m",
        "pointcloud_rgb_range": [0.0, 1.0],
        "pointcloud_frame": "camera" or "out",
        "workspace": workspace,
        "workspace_unit": "m",
        "npoints": npoints,
        "sampling": sampling,
        "min_depth": min_depth,
        "max_depth": max_depth,
        "depth_valid_ratio": valid_ratio,
        "point_count": N,
    },
}
```

---

## 5. 命令行测试与依赖

### 5.1 命令行测试

```bash
python realsense.py
```

常用参数：

```bash
python realsense.py --fps 30 --depth-res 640x480 --color-res 640x480
python realsense.py --align-to depth --npoints 1024 --sampling random
python realsense.py --align-to color --npoints 1024 --sampling random
python realsense.py --all-points --sampling none
python realsense.py --device cuda:0 --sampling fps
python realsense.py --workspace 0.2 0.8 -0.4 0.4 0.0 0.5
python realsense.py --overlay target_frame.jpg --overlay-alpha 0.5
```

快捷键：

```text
q: 退出
p: Open3D 显示当前点云，需要安装 open3d
```

### 5.2 依赖

必需：

```bash
pip install numpy opencv-python torch pyrealsense2
```

可选：

```bash
pip install open3d      # 点云可视化
pip install pytorch3d   # farthest point sampling；没有时自动 fallback random
```

---

## 6. L515 与 D435：成像原理、工作范围和失效场景

这一节用于帮助你在机器人真机实验中选择和摆放 RealSense 相机。核心结论很简单：

```text
L515: LiDAR / ToF 主动扫描，相对适合室内、漫反射、静态或中低速桌面场景。
D435: Active IR Stereo 双目深度，相对适合宽视场、机器人导航、动态场景和更通用的室内/部分室外场景。
```

二者都不是“万能几何传感器”。透明、镜面、高反光、强吸光、细小薄边、强阳光和多红外设备干扰，都会让深度图明显变差。

### 6.1 L515 成像原理

L515 是 Intel RealSense 的 LiDAR 深度相机。它使用红外激光和 MEMS 微镜扫描场景，通过接收物体反射回来的红外信号估计距离。可以近似理解为：

```text
IR laser 发射
  -> MEMS mirror 扫描视场
    -> 物体表面反射 IR
      -> photodiode / receiver 接收回波
        -> ASIC 计算距离
          -> depth map / point cloud
```

因此，L515 深度质量非常依赖：

```text
1. 物体表面对 860 nm 左右红外光的反射能力
2. 反射光能否回到接收器
3. 环境红外噪声是否过强
4. 表面是否产生多路径、折射、镜面反射或次表面散射
```

L515 对普通室内漫反射物体通常能给出比较干净、稠密的深度图；但对透明、镜面、黑色吸光材质会明显不稳。

### 6.2 D435 成像原理

D435 属于 D400 系列 Active IR Stereo 深度相机。它有左右两个红外成像器，通过 stereo matching 计算左右图像中的视差，再由视差恢复深度。D400 系列还可以使用红外投影器给低纹理场景增加非可见 IR pattern，从而提高匹配质量。

可以近似理解为：

```text
left IR image + right IR image
  -> stereo rectification
    -> correspondence / disparity search
      -> disparity to depth
        -> depth map / point cloud
```

所以，D435 深度质量主要依赖：

```text
1. 左右 IR 图像能否看到同一个表面点
2. 表面是否有足够可匹配纹理，或者 IR projector pattern 是否有效
3. 物体边缘是否存在遮挡 / 半遮挡
4. 环境光是否破坏 IR 图像或 projector pattern
5. 距离越远，视差越小，深度误差通常越大
```

和 L515 不同，D435 的问题更像是“双目匹配问题”：低纹理白墙、重复纹理、细边缘、遮挡边界、强反光、透明体都会导致匹配失败或深度噪声。

### 6.3 工作范围与精度对比

| 项目         | L515                                                         | D435                                                         |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 深度技术     | LiDAR / ToF 主动扫描                                         | Active IR Stereo 双目深度                                    |
| 推荐环境     | 室内为主                                                     | 室内，也可在部分室外/弱阳光条件下使用                        |
| 典型工作范围 | 约 0.25 m 到 9 m，具体取决于分辨率和反射率                   | 官方产品页标称 ideal range 约 0.3 m 到 3 m；datasheet 写 range 可到 over 10 m，但随光照变化 |
| 近距离       | 官方范围从约 0.25 m 起；太近可能过饱和或无效                 | 官方 D400 datasheet 写 D435 range 约 0.2 m 到 over 10 m，随光照变化 |
| 视场         | 约 70° × 55°                                                 | 约 87° × 58°，宽视场                                         |
| shutter      | 深度为扫描/ToF 机制，单点曝光极短                            | depth imagers 为 global shutter                              |
| 官方精度表述 | VGA、95% 反射率下，avg depth accuracy < 5 mm @ 1 m，< 14 mm @ 9 m | D400 datasheet 中，2 m 内、80% FOV 区域，Z-accuracy ≤ 2%     |
| 主要优势     | 室内深度图干净、稠密；对低纹理表面不依赖双目纹理             | 宽 FOV、global shutter，适合机器人导航、动态场景、通用性更强 |
| 主要短板     | 对材质反射率、阳光、透明/反光物体敏感                        | 对纹理、匹配质量、遮挡边界、远距离视差敏感                   |

注意：官方精度通常是在特定实验条件下测得，真机实验中的实际精度会受到距离、角度、材质、光照、相机温度、曝光、preset、滤波和标定质量影响。不要把 datasheet 数值直接当作所有场景的误差上界。

### 6.4 L515 不适用或效果较差的材质 / 物体

L515 更怕“红外回波不可靠”的物体：

```text
透明 / 半透明：
    玻璃杯、透明塑料盒、亚克力、透明瓶、塑料袋
    -> 容易透射、折射、多路径，深度可能落到背景或出现大洞

镜面 / 高反光：
    镜子、不锈钢杯、镀铬工具、手机屏幕、亮面包装袋、亮面桌面
    -> 红外光被镜面反射到别处，局部无深度或出现飞点

黑色 / 低反射率：
    黑橡胶、黑布、深色泡棉、哑光黑塑料、黑色线缆
    -> 回波弱，距离稍远时 fill rate 和稳定性下降

半透明 / 次表面散射：
    硅胶、蜡、果冻状材料、半透明软管
    -> 深度可能偏移、表面变厚、边缘不稳定

细小 / 薄边：
    细绳、针状物、薄片边缘、薄膜、夹爪尖端
    -> 点云断裂、边缘飞点、被背景吞掉
```

L515 更怕的场景：

```text
1. 强阳光、窗边直射光、强红外照明
2. 多台主动红外深度相机互相照射同一区域
3. 物体距离小于最小有效距离
4. 大角度斜面 / 掠射角表面
5. 目标反射率很低且相机距离较远
```

### 6.5 D435 不适用或效果较差的材质 / 物体

D435 更怕“双目匹配不可靠”的物体和场景：

```text
低纹理 / 重复纹理：
    白墙、纯色桌面、纯色盒子、重复格纹材料
    -> 左右图像难以稳定匹配，深度噪声变大或空洞变多

透明 / 半透明：
    玻璃、透明塑料、亚克力、透明容器、塑料袋
    -> 左右图像看到的是折射/背景，匹配可能落到错误表面

镜面 / 高反光：
    金属、镜子、亮面塑料、屏幕、反光包装
    -> 高光和镜面反射破坏左右一致性

细小 / 遮挡边界：
    细绳、线缆、薄片边缘、夹爪尖、物体轮廓边缘
    -> 左右相机可见性不同，容易出现边缘毛刺和飞点

强阳光 / 强 IR 干扰：
    室外阳光、窗边直射、其他 IR 投影器
    -> IR pattern 被冲淡，IR 图像质量下降
```

相比 L515，D435 在机器人移动、宽视场、动态场景中更常用；但如果任务需要非常干净的桌面点云，且场景是室内漫反射物体，L515 可能更舒服。

### 6.6 对机器人真机实验的选型建议

桌面灵巧操作 / 模仿学习中，可以按下面思路选：

```text
优先选 L515 的情况：
    1. 室内固定相机
    2. 目标主要是纸盒、木块、哑光塑料、普通漫反射物体
    3. 需要较干净的局部点云
    4. 场景中阳光和透明/反光物体较少

优先选 D435 的情况：
    1. 需要更宽视场
    2. 机器人或物体运动较快
    3. 需要导航、避障、人体/大场景感知
    4. 室内外光照变化较大
    5. 可以接受双目边缘噪声和点云空洞
```

对你的代码接口，建议这样配置：

```python
# 点云 / workspace crop，优先保持 depth frame 几何一致
cam = RealSense(serial=None, align_to="depth")

# RGB-D policy / VLA，需要 RGB 和 depth 像素强对齐时
cam = RealSense(serial=None, align_to="color")
```

对两类相机都建议：

```text
1. 记录 depth_valid_ratio，用于过滤坏帧
2. 尽量避免镜面桌面、透明容器、强阳光直射
3. 对黑色/透明/反光物体，不要只依赖单帧点云
4. 调试 workspace 前先 npoints=None 看完整点云
5. 谨慎使用 hole filling，避免补出不存在的表面
6. 每次换相机位置后，重新确认 T_out_camera 和 workspace 坐标系
```

### 6.7 资料来源

```text
Intel RealSense LiDAR Camera L515 Datasheet:
https://realsenseai.com/wp-content/uploads/2025/06/Intel_RealSense_LiDAR_L515_Datasheet_Rev003.pdf

Intel RealSense D400 Series Datasheet:
https://cdrdv2-public.intel.com/841984/Intel-RealSense-D400-Series-Datasheet.pdf

Intel RealSense D435 Product Page:
https://realsenseai.com/products/stereo-depth-camera-d435/

Curto et al., 2022, An Experimental Assessment of Depth Estimation in Transparent and Translucent Scenes for Intel RealSense D415, SR305 and L515:
https://www.mdpi.com/1424-8220/22/19/7378
```

---

## 7. 注意事项

### 7.1 depth 单位

`read()` 返回的 depth 已经标准化为 `np.uint16`，单位是 **millimeter**。

```python
depth[v, u] = 530  # 530 mm = 0.53 m
```

点云生成时，代码内部会自动把 `depth` 从 mm 转成 meter。不要在调用 `pointcloud_from_frame()` 前手动除以 1000。

### 7.2 RGB/BGR

`read()` 返回的是 RGB。OpenCV 显示前需要转 BGR：

```python
cv2.imshow("rgb", color[..., ::-1])
```

### 7.3 alignment 和点云颜色

如果 `align_to="none"` 且 color/depth 分辨率不同，`pointcloud_from_frame(color, depth)` 会报错，因为无法一一对应上色。

常用选择：

```text
点云 / workspace crop: align_to="depth"
RGB-D image policy / VLA: align_to="color"
只用 depth: enable_color=False, align_to="none"
```

### 7.4 workspace 为空点云

出现：

```text
ValueError: No valid points after depth filter/crop.
```

优先检查：

```text
1. min_depth / max_depth 是否太窄
2. workspace 是否写错坐标系
3. T_out_camera 是否传反
4. 相机是否看到有效深度
5. align_to 是否符合当前点云需求
```

### 7.5 timestamp

`timestamp` 是 RealSense frame timestamp；`host_time` 是 Python 侧接收到 frameset 后的 `time.time()`。后续要和 robot state / teleop command 对齐时，建议优先记录二者。

### 7.6 sampling

```text
sampling="random": 推荐在线控制，耗时稳定
sampling="fps":    几何覆盖更好，但可能更慢；无 PyTorch3D 时 fallback random
sampling="first":  deterministic debug
sampling="none":   返回全部有效点
```

---

## 8. 设计边界

V2 明确不做：

```text
1. 线程 / multiprocessing
2. shared memory
3. 多相机同步
4. recorder
5. policy wrapper
6. LeRobot / RLDS adapter
7. dataclass 封装
```

它只负责：

```text
single RealSense hardware wrapper
+ RGB-D
+ active intrinsics
+ point cloud
+ workspace crop
+ sampling
+ obs dict
```