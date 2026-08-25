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
                 └─ raw v22 → offline process → processed HDF5 v7 → Policy Zarr
```

实时 worker 不排队旧帧：它只读取 `camera_ring` 的最新 sequence，并在构建前后检查相机健康、
generation、时钟重置和最大帧龄。结果在计算后已过期时不会发布。

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
  → 固定 N 空间采样或循环补齐
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

- 高度不超过 `table_core_height_m=6 mm` 的点确定为桌面；
- 每个候选连通域必须包含至少 `table_object_seed_min_pixels=4` 个高度不低于
  `table_object_seed_height_m=12 mm` 的可靠物体种子；
- 满足种子条件时，保留该连通域中所有高于 6 mm 的像素，包括物体接触边缘的 6–12 mm
  低处表面；不满足时，整个低矮残留簇作为桌面删除。

这是一种高度迟滞：高阈值确认物体，低阈值恢复其连通表面。它不膨胀桌面掩膜，不会无条件
吃掉物体边缘；一次连通域标记为线性复杂度，也没有逐像素迭代生长。
低于或等于 6 mm 的表面与桌面在当前深度/平面观测中不可区分，仍按桌面处理；完全低于 12 mm
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

当前最小连通域为 8 点，能删除虽满足 5 邻居规则、但总规模只有 6–7 点的紧凑碎片。过滤在
全部体素上执行完成后，才以稳定空间哈希保留至多
`num_points × outlier_candidate_multiplier` 个候选，避免离群点提前占用候选名额。最后按稳定
空间哈希输出 `N` 个点；候选少于 `N` 时循环重复已有有效点，而不插值或伪造点。

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

纯 build 不包含等待相机帧、SDK 对齐和共享内存发布；capture-to-cloud 包含 `camera.read()`
及对齐/数组准备，但不包含实时 worker 的 ring 发布。需要评价部署总时延时，应同时读取实时
worker 的 `source_to_publish_ms_p95` 日志。当前纯构建目标是 p95 < 40 ms。

### 2026-08-25 L515 当前 v8 硬件基准

当前结果来自 L515（serial `f1382055`，firmware `1.5.8.1`）、Short Range preset、640×480
depth-to-color aligned RGBD、桌面高度迟滞裁减、5 mm 体素、12 mm radius graph、5 邻居、
8 点最小连通域和 `N=1024`。workspace 为 `x=[0.0, 0.8]`、`y=[-0.5, 0.5]`、
`z=[0.0, 0.8]`；本次跳过桌面标定，使用已解析的 `desk_plane.json`。

| 指标 | 结果 | 诊断 |
|---|---:|---|
| 单帧 pure build | 26.9 ms | 内部阶段之和约 26.8 ms，计时一致 |
| 20 帧 pure build p50 / p95 / max | 27.5 / 28.5 / 32.9 ms | p95 比 40 ms 目标低 11.5 ms |
| 20 帧 capture-to-cloud p50 / p95 / max | 32.1 / 38.0 / 42.4 ms | p95 仍低于 40 ms；max 尾部来自采集/SDK 路径 |
| 单独 frame capture | 6.0 ms | 单次展示帧，不与独立分位数直接相减 |

构建阶段 p95 如下。各阶段分别取 p95，不能严格相加；其和为 29.0 ms，略高于总耗时 p95
28.5 ms 属于正常的分位数统计现象。

| 阶段 | p95 | 占 pure-build p95 | 诊断 |
|---|---:|---:|---|
| `depth_filter` | 10.3 ms | 36.1% | 当前最大热点；范围、飞线与二维支持过滤 |
| `table_crop` | 7.0 ms | 24.6% | plane height 与一次二维迟滞连通标记 |
| `spatial_outlier_filter` | 5.9 ms | 20.7% | 单次 radius graph 密度/连通域过滤 |
| `voxelization` | 2.3 ms | 8.1% | 23003 个裁减点聚合为 5126 个体素 |
| `deprojection` | 1.8 ms | 6.3% | 仅反投影通过桌面裁减的像素 |
| `base_workspace` | 1.4 ms | 4.9% | color → xArm-base 与 workspace 判断 |
| `color_sampling` | 0.3 ms | 1.1% | 组装 `[xyzrgb]` 与固定 N 采样 |

本帧点数变化为：

```text
valid 171,586
→ supported 168,682        (二维过滤删除 2,904，1.69%)
→ table reject 145,679     (删除 supported 的 86.36%)
→ workspace reject 0       (本帧 workspace 不是有效裁减边界)
→ crop 23,003
→ voxel 5,126              (保留 crop 的 22.28%)
→ density 4,991            (删除 135 个体素，2.63%)
→ component 4,963          (再删除 28 个体素，0.56%)
→ candidate 4,963
→ output 1,024             (稳定空间下采样，不是额外离群过滤)
```

`workspace_reject=0` 与当前扩大后的 workspace 一致，表示这帧所有桌面裁减后点都位于配置范围内，
不是 pipeline 漏执行。三维过滤共删除 163/5126 个体素（3.18%）；候选上限为 8192，因此本帧
没有触发 candidate cap。若可视化仍有密集桌面残留，应优先调整桌面高度策略或检查平面标定，
而不是继续提高半径邻居数，因为密集残留本来就能通过三维密度规则。

capture-to-cloud p50 比 pure-build p50 高约 4.6 ms，但其 p95 差约 9.5 ms，max 差约
9.5 ms。纯构建 max 只有 32.9 ms，因此 42.4 ms 的端到端尖峰主要来自 `camera.read()`、SDK
对齐或调度等待。评价实时部署仍应以 point-cloud worker 的 `source_to_publish_ms_p95` 为准。

[`examples/realsense_record_example.py`](../examples/realsense_record_example.py) 提供两种可视化：

| 模式 | 含义 | 用途 |
|---|---|---|
| RAW | 所有有效 aligned depth 像素直接反投影、同像素 RGB | 诊断传感器、对齐和飞线 |
| PROCESSED | 生产 builder 的固定 N 点云 | 验证策略实际输入 |

## 录制、离线处理与语义校验

录制 schema v22 保存 aligned raw depth、RGB、native depth/color 几何 provenance、帧号和时间
信息。离线处理只接受 raw v22，并使用同一个生产 builder 生成 processed HDF5 v7。

每个 processed 点云产物记录并在导出/可视化时校验：

- `point_cloud_frame=xarm_base` 与 `[N,6] float32` shape；
- `point_cloud_color_source`；
- `point_cloud_policy_id`；
- `PointCloudConfig` 的 SHA-256；
- 桌面平面、采样和变换语义。

导出、可视化和部署只接受与当前 policy ID、配置哈希及桌面标定身份完全一致的产物；其他产物
必须从受支持的 raw v22 episode 重新处理，不能通过兼容分支静默混用。

## 代码所有权

| 责任 | 位置 |
|---|---|
| RealSense 采集、对齐、原生 provenance | `sensor/realsense.py` |
| aligned RGBD IPC 发布 | `sensor/camera_worker.py` |
| 纯几何/滤波/采样 | `sensor/pointcloud.py` |
| 最新帧消费、freshness 与 point-cloud IPC 发布 | `sensor/pointcloud_worker.py` |
| 点云参数与稳定 identity | `config/pointcloud.py` |
| 桌面 RANSAC 与 plane 文件 | `calibration/table.py`、`config/desk_plane.json` |
| raw v22 → processed v7 | `data/process.py` |

核心数学不访问 SDK、共享内存、文件或可视化；硬件采集、IPC、标定写入和 GUI 均在外围 owner 中。
