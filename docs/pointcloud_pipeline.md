# 点云生成与处理链路

本文描述 DexMani Real 当前的生产点云路径。运行行为以源码和
[`dexmani_real/config/pointcloud.py`](../dexmani_real/config/pointcloud.py) 为准；本文不替代
运行时配置或相机标定。

## 范围与不变量

生产点云使用 RealSense 的 `depth_to_color` 对齐结果，而不是原生 depth 像素网格。每帧输出
为 xArm-base 坐标系的连续 `float32[N, 6]`：

```text
[x_base, y_base, z_base, r, g, b]
```

- XYZ 单位为米；RGB 范围为 `[0, 1]`。
- `N` 由 `PointCloudConfig.num_points` 决定；实时 IPC 只允许 schema 支持的固定值。
- 对齐后，depth 与 RGB 都在 color 像素网格中，必须使用 color 内参和
  `T_xarm_base_from_color`。
- 原生 depth、原生几何和 `T_color_from_depth` 只作为设备 provenance 保留；它们不进入生产
  点云的反投影路径。

## 总体数据流

```text
Intel RealSense
  native depth + color frames
        │
        ├─ 保留 native payload / geometry 作为 provenance
        │
        └─ rs.align(depth → color)
                 │
                 ▼
        aligned uint16 depth + RGB, 均为 color 像素网格
                 │
                 ▼
        camera_worker → camera_ring
                 │  (最新帧、时钟/健康/generation 元数据)
                 ▼
        pointcloud_worker
                 │
                 ▼
        build_point_cloud_with_stats()
                 │
                 ▼
        fixed float32[N, 6], frame=xarm_base
                 │
                 ├─ pointcloud_ring → deployment / policy observation
                 └─ raw v24
                      ├─ raw visualizer → Rerun canonical preview
                      └─ offline process → processed HDF5 v11 → Policy Zarr v5
```

实时 worker 不排队旧帧：它只读取 `camera_ring` 的最新 sequence。构建前检查候选帧的相机健康、
generation、时钟重置、source/publish/now 因果关系和最大帧龄；构建后只复核 source frame age，
若结果已过期则不发布。`camera_frame_gap` 是 telemetry-only：reset/duplicate 时数值为 0，
恢复后的当前帧只要 payload 与时间戳仍新鲜，仍可进入构建。

## Raw episode 即时可视化

`examples/visualize_episode.py` 对 raw v24 默认启用点云。持久化边界由
`data/raw_pointcloud.py` 统一解析：它从 episode 读取 aligned RGB-D 几何、`depth_scale` 与
`T_xarm_base_from_color`，再调用和 offline processing、实时 worker 相同的
`sensor.pointcloud.build_point_cloud()`。可视化入口不维护第二套反投影、裁减或采样实现。

raw episode 保存记录期 resolved config 的 SHA-256，但不复制可反序列化的完整点云策略与桌面
平面。因此即时点云使用当前 resolved runtime 的 `PointCloudConfig` 和桌面标定；哈希与记录期不一致
时入口会 warning，结果应视为 current-config preview。需要可持久复现和完整 provenance 时，应生成
processed HDF5；其文件内保存 processing config、点云配置哈希与桌面平面身份。

运行时只读取 raw v24。`--no-point-cloud` 可关闭即时推导。
单帧构建结果为空时 Rerun 会显式清除该时间点的点云，避免沿用上一帧。eye-in-hand 仍会 fail closed，因为 raw schema 没有保存相机曝光
时刻对应的机械臂位姿。

## 坐标系与对齐语义

`rs.align(depth → color)` 将 Z16 深度采样重映射到 color 图像网格。生产路径构造：

```text
geometry.depth = color intrinsics
geometry.color = color intrinsics
T_color_from_depth = I
```

对于 aligned depth 像素 `(u, v, z)`：

```text
p_color = deproject_color_intrinsics(u, v, z)
p_base  = T_xarm_base_from_color · p_color
```

调用接口与局部变量因此显式使用 `T_xarm_base_from_color`。对 eye-to-hand 相机，该外参来自
`cameras.json` 中按序列号解析的静态标定；实时 point-cloud worker 会拒绝 eye-in-hand，因为
后者需要相机曝光时刻的同步机械臂位姿合同。

## 单帧生产处理

核心实现为
[`build_point_cloud_with_stats()`](../dexmani_real/sensor/pointcloud.py)。以下顺序在实时、离线
处理和点云诊断中共用。

```text
aligned uint16 depth + RGB
  → depth gate 与 2-D 支持过滤
  → 桌面高度迟滞预裁减
  → color-frame 反投影
  → xArm-base 变换与 workspace 裁减
  → XYZ/RGB 均值体素化
  → 单次 radius graph 邻居密度/连通域过滤
  → 空间候选上限
  → 粗体素分层的固定 N 采样或循环补齐
```

### 1. 深度 gate 与二维飞线过滤

先将 `uint16` 深度乘以 `depth_scale_m`，保留配置深度范围内的非零有限值。随后在 3×3 深度
邻域中：

- 深度跨度超过 `edge_jump_m` 时，删除位于前景/背景之间的 intermediate flying depth；
- 平坦区域要求至少 `depth_support_min_neighbors` 个同表面邻居；
- 深度突变边缘要求更严格的 `edge_support_min_neighbors`；
- 同表面由 `edge_surface_band_m` 判定。

在深度突变处还要求上、下、左、右四个直接邻居均属于同一表面。这会删除不可靠的一像素轮廓
层以及一至两像素飞线；非突变区域仍只使用原 3×3 支持阈值，因此已被深度相机解析、至少具有
内部像素的物体不会被全局收缩。代价是边缘处仅一至两像素宽的未解析结构会被视为不可靠深度，
这是清除飞线时有意采用的保守取舍。

该步骤针对单像素、双像素飞线，优先在二维深度图中以低成本处理。

### 2. 桌面预裁减

桌面平面在 xArm-base 中保存为：

```text
a x + b y + c z + d = 0
```

构建时先将平面变换到 color camera frame。对每个静态像素，系统缓存去畸变 ray
`r(u, v)=[x/z, y/z, 1]`，每帧直接计算：

```text
height(u, v) = z(u, v) · dot(n_color, r(u, v)) + d_color
```

这与反投影后代入平面等价，但可在分配三维点数组前删除桌面。随后对高于桌面核心的像素做
一次 8 邻接连通域标记：

- 高度不超过 `table_core_height_m=7 mm` 的点确定为桌面；
- 每个候选连通域必须包含至少 `table_object_seed_min_pixels=4` 个高度不低于
  `table_object_seed_height_m=13 mm` 的可靠物体种子；
- 满足种子条件时，保留该连通域中所有高于 7 mm 的像素，包括物体接触边缘的 7–13 mm
  低处表面；不满足时，整个低矮残留簇作为桌面删除。

这是一种高度迟滞：高阈值确认物体，低阈值恢复其连通表面。它不膨胀桌面掩膜，不会无条件
吃掉物体边缘；一次连通域标记为线性复杂度，也没有逐像素迭代生长。
低于或等于 7 mm 的表面与桌面在当前深度/平面观测中不可区分，仍按桌面处理；完全低于 13 mm
且没有可靠高点的扁平物体也无法由纯几何高度与桌面残留可靠区分，需要降低种子阈值或增加其他
感知先验。

该迟滞连通步骤在 640×480 合成高度图上的离线微基准为 p50 3.11 ms、p95 3.21 ms；这只衡量
连通保护本身，不包含 plane height 计算，也不能代替 L515 完整 `table_crop` 阶段的硬件基准。

桌面平面缺失或禁用时，该步骤保留全部已支持深度点。

### 3. 反投影、基座变换与 workspace

仅对通过桌面裁减的像素做 Brown-Conrady（或无畸变）反投影，随后使用
`T_xarm_base_from_color` 转到 xArm-base。`workspace` 是 base-frame 轴对齐包围盒；其外的
点不会进入体素化。当前感知范围为：

```text
x: [ 0.0, 0.8] m
y: [-0.5, 0.5] m
z: [ 0.0, 0.8] m
```

### 4. 体素与 RGB 聚合

`voxel_size_m` 将 base-frame 点分到整数体素键。同一体素内：

```text
voxel_xyz = mean(source_xyz)
voxel_rgb = mean(source_aligned_pixel_rgb)
```

因此生产点云不会将体素中心重新投影到 RGB 图像取色。颜色和几何具有同一批 source aligned
像素的 provenance，且移除了不必要的颜色重投影和 z-buffer 去重。

### 5. 三维离群过滤与固定输出

体素点只构建一次 `cKDTree`，并通过一次 `query_pairs(outlier_radius_m)` 得到无向 radius
graph。同一批邻接对依次提供：

- 每个体素的精确半径邻居数，要求至少 `outlier_min_neighbors`；
- 仅在通过邻居密度过滤的体素之间计算物理空间连通域，要求至少
  `outlier_min_component_points` 个体素点。这样稀疏桥接点不能把两个小孤立簇误连成大簇。

当前最小连通域为 10 点，能删除虽满足 6 邻居规则、但总规模只有 7–9 点的紧凑碎片。过滤在
全部体素上执行完成后，才以稳定空间哈希保留至多
`num_points × outlier_candidate_multiplier` 个候选，避免离群点提前占用候选名额。固定 N 采样先在
`sampling_coarse_voxel_stride=3` 的粗网格中各选一个确定性代表点；默认 5 mm 体素对应
15 mm 粗网格。粗网格数少于 `N` 时，再按细体素空间哈希填充剩余名额；多于
`N` 时按粗网格哈希截断。候选总数少于 `N` 时才循环重复已有有效点，而不插值或伪造点。
这在不引入最远点采样二次时间成本的前提下，减少局部过密区域抢占名额造成的空间空洞。

连通域和半径过滤只能删除小型或稀疏团块。密集、独立的假点团可能满足这两条纯几何规则；若
场景保证所有有效物体接触桌面，可在业务层额外引入“连通域桌面锚定”规则。该规则不适用于
抓取中或悬空的有效物体。

## 桌面标定

[`examples/pointcloud_process_example.py`](../examples/pointcloud_process_example.py) 首先询问：

```text
Run table calibration? [y/N]
```

- 默认跳过，使用运行时解析的 `desk_plane.json`；
- 选择标定后，操作者清空桌面并采集多帧 aligned depth；
- 每帧先经过深度 gate/二维支持过滤，再转换到 xArm-base；
- RANSAC 拟合平面并报告支持率、RMS 与倾角；
- 只有再次输入 `y` 才备份并发布 `desk_plane.json`。

发布新的桌面平面只影响随后启动或重新解析运行时配置的消费者；运行中的 worker 使用启动时的
不可变配置快照。

## 诊断与性能指标

`pointcloud_process_example.py` 显示：

- aligned RGBD 与深度 gate；
- 各过滤阶段的点数；
- 单帧各内部阶段耗时；
- 20 帧纯 build 的 p50/p95/max；
- 20 帧从 `camera.read()` 到点云结果的 capture-to-cloud p50/p95/max；
- 各内部阶段的 p95。

传入 `--save-dir <directory>` 时，入口会在该目录下原子发布一个 schema v2
post-calibration 同帧诊断快照：
`rgb.npy`、`depth_aligned_to_color_raw.npy`、`raw_point_cloud.npy`、
`processed_point_cloud.npy`、`T_xarm_base_from_color.npy` 与 `metadata.json`。metadata 保存 aligned
几何、`depth_scale_m`、桌面平面及来源、完整 `PointCloudConfig` 与 SHA-256、runtime SHA-256 和算法
语义。所有 `.npy` 均禁用 pickle；即使 processed 构建为空，原始 RGB-D、raw 点云和空输出仍会保存，
用于离线复现失败帧。该快照是算法诊断产物，不属于 raw episode 或 processed HDF5 schema。

纯 build 不包含等待相机帧、SDK 对齐和共享内存发布；capture-to-cloud 包含 `camera.read()`
及对齐/数组准备，但不包含实时 worker 的 ring 发布。需要评价部署总时延时，应同时读取实时
worker 的 `source_to_publish_ms_p95` 日志。当前纯构建目标是 p95 < 40 ms。

### 2026-08-26 当前离线验证

五个 L515 诊断快照覆盖常规桌面物体、机械臂轮廓和两个细小物体场景。当前默认策略使用
5 邻居边缘支持、7/13 mm 桌面核心/物体种子高度、5 mm 体素、12 mm 离群半径、6 邻居、
10 点最小连通域，以及 15 mm 粗体素分层的固定 N 采样。更严的桌面 8/14 mm 或半径 7 邻居
会开始删除盘沿、物体轮廓和机械臂细结构，因此未进入生产默认值。

`episode_20260826_152317` 的 240 个有效帧预先读入内存后，当前生产 builder 两轮结果如下：

| 指标 | 结果 |
|---|---:|
| pure-build p50 | 19.82--19.93 ms |
| pure-build p95 | 23.73--24.01 ms |
| 纯构建吞吐 | 56.8--57.5 FPS |
| `depth_filter` p50 | 2.54--2.58 ms |

OpenCV 3×3 depth fast path 与逐步 SciPy 参考实现的 240 帧最终点云 SHA-256 全部一致；五个
诊断快照的可信深度掩码和当前策略点云也逐元素一致。上述数字不包含相机采集、IPC 和系统调度
尾延迟；部署性能仍以实时 worker 的 `source_to_publish_ms_p95` 为准，真实硬件尚未按当前默认
参数重新测试。

[`examples/realsense_record_example.py`](../examples/realsense_record_example.py) 提供两种可视化：

| 模式 | 含义 | 用途 |
|---|---|---|
| RAW | 所有有效 aligned depth 像素直接反投影、同像素 RGB | 诊断传感器、对齐和飞线 |
| PROCESSED | 生产 builder 的固定 N 点云 | 验证策略实际输入 |

## 录制、离线处理与语义校验

录制 schema v24 保存 aligned raw depth、RGB、native depth/color 几何 provenance、帧号和时间
信息，并持久化与 camera source 对齐的 arm/hand policy observation。离线处理只接受 raw v24；
只有完整的对齐与动作限速合同的 pointcloud 产物可进入 Policy Zarr v5。

每个 processed 点云产物记录并在导出/可视化时校验：

- `point_cloud_frame=xarm_base` 与 `[N,6] float32` shape；
- `rgb`/`rgb_pc` profile 中 RGB 与 depth 的 `N/H/W` 完全一致，mismatch 在 admission 阶段拒绝；
- `point_cloud_color_source`；
- `point_cloud_policy_id`；
- `PointCloudConfig` 的 SHA-256；
- 桌面平面、采样和变换语义。

导出、可视化和部署只接受与当前 policy ID、配置哈希及桌面标定身份完全一致的产物；其他产物
必须从受支持的 raw v24 episode 重新处理。

## 代码所有权

| 责任 | 位置 |
|---|---|
| RealSense 采集、对齐、原生 provenance | `sensor/realsense.py` |
| aligned RGBD IPC 发布 | `sensor/camera_worker.py` |
| 纯几何/滤波/采样 | `sensor/pointcloud.py` |
| 最新帧消费、freshness 与 point-cloud IPC 发布 | `sensor/pointcloud_worker.py` |
| 点云参数与稳定 identity | `config/pointcloud.py` |
| 桌面 RANSAC 与 plane 文件 | `calibration/table.py`、`config/desk_plane.json` |
| raw v24 相机 metadata 与点云输入适配 | `data/raw_pointcloud.py` |
| raw v24 → processed v11 | `data/process.py` |
| raw current-config preview | `examples/visualize_episode.py` |

核心数学不访问 SDK、共享内存、文件或可视化；硬件采集、IPC、标定写入和 GUI 均在外围 owner 中。
