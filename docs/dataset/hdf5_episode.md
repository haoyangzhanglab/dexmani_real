# DexMani Real v17 与 DexMani Sim HDF5/Zarr 完整数据字典

本文档同时记录两个彼此独立的数据合同：第 1–10 节描述 DexMani Real 当前运行时唯一
支持的录制格式 **schema v17**；第 11 节描述 DexMani Sim 当前 HDF5 episode、HDF5→Zarr
转换格式和现存数据审计。

Real 部分以
`recording/episode_recorder.py`、`recording/camera_stream_writer.py`、
`recording/episode_reader.py`、`recording/timestamp_buffer.py` 和实际生产端为准。
历史的单文件 HDF5 或 v17 以前格式不在本文档的兼容范围内，必须在运行时之外迁移
（旧 v16 episode 只需 bump `schema_version` 即可被新 reader/pipeline 处理，点云由 depth 无损重算）。

> **合同边界：** Real 与 Sim 的同名字段不能直接互换。读取方必须先识别容器来源，再应用
> 对应章节的 shape、单位、坐标系、时序和有效性规则。

## 0. 阅读指南

### 0.1 按任务查阅

| 目标 | 重点章节 |
|---|---|
| 判断文件属于 Real 还是 Sim | 0.2、0.3 |
| 查 Real episode 目录和 metadata | 1、3 |
| 查 Real dataset、相机侧车和有效性 | 4、5、6、7 |
| 读取或筛选 Real 数据 | 8 |
| 查 Real 已确认问题 | 10 |
| 查 Sim HDF5 dataset 和 attributes | 11.4、11.5 |
| 查 Sim Zarr array、chunk 和 episode 边界 | 11.7 |
| 查 Sim 实测规模、问题和读取约束 | 11.8、11.10、11.11 |
| 按 Sim/Policy 命名理解 Real episode | [Real→Sim/Policy 标签映射表](real_to_sim_mapping.md) |
| 按处置优先级查看 Real 已确认问题 | 0.4 |

### 0.2 覆盖范围与复核状态

| 合同 | 文档覆盖 | 复核基线 |
|---|---|---|
| Real v17 | 93 个基础 `data.h5` datasets、1 个标准 RecorderIO 条件 dataset、`/meta` attributes、1 个相机 HDF5 侧车 dataset（`depth.h5`），以及 `rgb.mp4` | writer、reader、producer、schema dtype 和离线 round-trip |
| Sim HDF5 | 13 个逐帧 datasets、46 个已观察根 attributes | 1113 个文件、237777 帧的只读 metadata 普查 |
| Sim Zarr v2 | 13 个 `data/*` arrays 和 `meta/episode_ends` | 8 个 stores、1000 个 episodes、210083 帧；边界及每个 task 首末样本核对 |

审计日期为 **2026-08-15**。Sim 的独立审计版位于
[`sim_hdf5_zarr.md`](sim_hdf5_zarr.md)。

### 0.3 Real 与 Sim 快速区分

| 维度 | DexMani Real v17 | DexMani Sim 当前格式 |
|---|---|---|
| episode 发布单元 | 一个目录：`data.h5`、`depth.h5`、`rgb.mp4` | 一个 `.h5` 文件；也可将同 task 的成功 episode 聚合成一个 Zarr store |
| schema 标记 | `/meta` attribute `schema_version=17` | HDF5 与 Zarr 均无 schema name/version |
| 行语义 | 固定控制网格上的对齐 sample，带逐行 timestamp 和有效性/质量信息 | `obs[t]` 为动作前状态，`action[t]` 驱动下一状态，`done[t]` 是动作后结果；无逐帧 timestamp |
| 当前默认/实测图像 shape | 默认 `(N,480,640,...)`，可由运行配置改变 | 实测固定为 `(N,240,320,...)` |
| 点云 | 不在 episode 中存储；从 `/depth` + 元数据确定性派生为 `(N,2048,6)`，默认点数可配置 | 可见点云 `(N,1024,6)`，另有手部理想点云 `(N,512,6)` |
| episode 元数据 | `data.h5` 的 `/meta` group attributes | HDF5 根 attributes；当前 Zarr 不保留这些属性 |
| 发布完整性 | 临时目录写入、校验并原子 rename | HDF5 直接写最终文件；Zarr 以 `mode="w"` 直接覆盖目标 store |

### 0.4 DexMani Real fact-check 问题优先级总表

本表只覆盖 **DexMani Real schema v17**，不包含第 11 节的 DexMani Sim 数据集问题。
优先级表示建议处置顺序：**P1** 应在依赖相关字段进行训练或回放前处理，**P2** 应在扩大
数据用途或下一次 schema 修订时处理，**P3** 属于扩展边界维护。本轮没有发现需要列为 P0
的现存 Real 数据不可逆损坏。运行复现均使用临时目录和 fake，不启动机器人、相机或 GUI。
本表复核日期为 **2026-08-16**。

| 顺序 | 优先级 | ID | 问题摘要 | 证据状态 | 主要影响 / 建议 |
|---:|---|---|---|---|---|
| 1 | P1 | H5-07 | reader 与 writer finalizer 没有共享完整 v17 dataset 合同，受损文件可被误判为 `VALID`。 | 已复现并修复 | 93 个基础字段和 `action_arm_joint_sent` 条件规则现由同一 schema 模块校验；未知历史扩展也必须首维为 `N`。 |
| 2 | P1 | H5-02 | align 会形成共同像素 viewport，但 aligned Z16 仍保留 source depth-camera 的轴向 Z；旧路径还允许把 depth-frame rays 套入 color 外参。 | producer、标定链及 librealsense 2.54.2 实现交叉核查；部分修复 | 生产已拒绝 `color_to_depth/none`，消除了更大的 frame mismatch；但当前 `depth_to_color` 点云仍是近似几何，严格修复需保存源 depth intrinsics、depth↔color extrinsic、distortion 并重投影。 |
| 3 | P1 | H5-03 | XHand 数量不足/缺失的 `sensor_data` 被零填充，甚至可能被标成 fresh/calibrated；同时 `0.1` 缩放不是已验证 SI 换算。 | fake 缺失传感器帧运行复现；已修复 | 严查 5×120×3 有限 payload；失效触觉立即发布 `fresh=False`，但不丢弃同帧有效关节反馈。 |
| 4 | P1 | H5-04 | held/failure 路径不存在求解器 raw 输出，但 v17 兼容 fallback 会把最终/hold 命令写入两个 `*_raw` 字段。 | recorder buffer 运行复现；已修复消费合同 | 保留 v17 存值，reader 提供 arm/hand 保守 raw-valid mask；不在 v17 内改成 NaN 或增加 dataset。 |
| 5 | P2 | H5-05 | `arm_tau` 单位未经合同确认，且 arm worker 启动首帧曾把 qvel/effort 人工置零却标有效。 | SDK 调用链、厂商说明和启动路径核查；已修复可修部分 | 启动即读取 `num=3` 真实反馈；metadata 明示 current-estimated effort、单位未验证，不做 N·m 换算。 |
| 6 | P2 | H5-01 | `arm_ee` 原生为 xArm base frame；当前 runtime 明确令 planner world 等于 base，因此不是现存数值 bug。 | FK、mapper、workspace 与 runtime 配置交叉核查；已加固 | 保持 v17 数值含义，修正误导命名并持久化 frame/invariant；未来支持非单位 base pose 必须升级合同并整体审计。 |
| 7 | P3 | H5-06 | 底层 recorder 对任意 diagnostics 既可覆盖核心字段，又可能形成未声明 dataset。 | writer/buffer 静态确认；已修复 | 固定 diagnostics allowlist，并冻结 source key/shape/dtype；reader 仍可读取首维为 `N` 的历史扩展。 |

H5-07 不表示既有标准 RecorderIO episode 普遍缺字段：标准路径启用
`arm_sent_stream=True`，会创建 93 个基础 dataset 加 `action_arm_joint_sent`，合计 94 个。
底层 `EpisodeRecorder(arm_sent_stream=False)` 的合法 v17 则只有 93 个基础字段。两种布局现均由
共享 schema 校验，不再把第 94 个条件字段误写成所有 v17 文件的无条件要求。

## 1. Episode 不是单个 HDF5 文件

一个成功发布的 episode 是一个目录：

```text
episode_YYYYMMDD_HHMMSS[_序号]/
├── data.h5          # 控制网格、机器人、VR、动作、质量和元数据
├── depth.h5         # 原始深度图
└── rgb.mp4          # 与 HDF5 逐帧对齐的 RGB 视频；不是 HDF5
```

录制期间先写入同级 `.tmp_episode_*` 目录；保存成功后经 fsync 和原子 rename
发布为 `episode_*`。失败或明确丢弃的录制不会保留部分 episode。三个文件在 v17
中都是必需文件，帧数必须一致。`EpisodeReader.h5f` 将 `data.h5` 和 `depth.h5`
暴露为一个只读的合并键空间；`rgb` 不属于该键空间，必须通过
`EpisodeReader.read_camera_frame()` 或 `read_camera_all()` 解码。世界点云不单独存储，
而是在消费边界从 `depth.h5`（配合内参/外参、desk-plane 和 `PointCloudProcessor` 配置）
确定性地派生。

## 2. 记号、坐标和存储约定

### 2.1 Shape 记号

| 记号 | 含义 | 当前默认值 |
|---|---|---:|
| `N` | episode 的控制网格槽数，即 `/meta.num_frames` | 最多由运行配置决定 |
| `A` | xArm7 关节数 | 7 |
| `H` | XHand 关节数 | 12 |
| `F` | 手指数，顺序由 XHand SDK/模型约定 | 5 |
| `T` | 每根手指的触觉采样点数 | 120 |
| `IH`, `IW` | 相机图像高、宽 | 默认 480、640，可由运行配置改变 |
| `M` | 定长点云点数 | 默认 2048，可由运行配置改变 |

文中 `(N,)` 表示每个网格槽一个标量；例如 `(N,A)` 表示每槽 7 个机械臂关节值。
除 `/meta` 外，所有 dataset 的第 0 维都应为同一个 `N`。

### 2.2 dtype 和压缩

- `data.h5` dataset 使用产生它的 NumPy dtype，均为可沿第 0 维扩展的 chunked
  dataset，并使用 h5py 默认级别的 gzip 压缩。
- `depth.h5` 每帧一个 chunk，使用 gzip level 1。
- 表中的 `float64`、`float32`、`int64`、`int32`、`uint8`、`uint16` 对应 NumPy/HDF5
  数值类型。`bool` 是 HDF5 可识别的 NumPy 布尔类型。
- 多数标识和纳秒时间在 IPC 边界是 `uint64`，但 `EpisodeRecorder` 将其转成 Python
  `int` 后交给当前 64-bit Python/NumPy 环境分配，因此 **HDF5 中实际为 `int64`**。
  读取器会在校验时安全地转换为 `uint64`。这些字段的合法值均非负，`0` 是“未知/无”。
- 浮点字段不可用时通常写 `NaN`；布尔、计数、标识和时间标识不可用时通常写
  `False` 或 `0`。相机图像/点云占位采用全零数组。

### 2.3 坐标和旋转

- `arm_ee` 的直接生产者输出 xArm base frame；`hand_fingertip` 由该 pose 链接 hand FK，
  因而继承相同基准。当前支持的 Real runtime 明确维持 `world == xarm_base`，所以状态、
  目标 EEF 和点云位于同一数值坐标系；这是一项运行不变量，不是对任意非单位
  `base_pose_world` 的支持。详见第 10.1 节。
- `vr_*` 是从 Unity 左手坐标转换后的 **FLU**（Forward、Left、Up）操作者/VR
  坐标，位置和 landmark 单位为米；它们尚未变换到机器人 world。
- 所有四元数字段均按 `[w,x,y,z]` 排列，无单位，正常值为单位四元数。
- 所有 `rot6d` 字段为旋转矩阵前两列的连续 6D 表示，无单位；恢复旋转时需要
  正交化，不能把六个分量当欧拉角。
- 点云每点固定为 `[x_world,y_world,z_world,r,g,b]`：XYZ 单位为米，RGB 为
  `[0,1]` 浮点值。

### 2.4 时间域

- 名称含 `monotonic` 的值属于主机单调时钟域，不是 Unix epoch，不能转换成日期。
- `_monotonic_ns` 以纳秒记录；`_monotonic_s` 和其余明确以 `_s` 结尾的时长以秒记录。
- `timestamp` 是控制网格单调时间（秒）。正常相邻槽间隔为 `grid_dt_s`；命令静默暂停
  会重新锚定网格，因此真实暂停表现为 timestamp 跳变，而不是合成动作。
- `camera_device_timestamp_s` 属于 RealSense 设备时钟；它与主机单调时钟不是同一原点。

## 3. `data.h5` 的 `/meta` group

`/meta` 本身没有 dataset，所有信息都是 HDF5 attribute。Python 字符串由 h5py
写成 UTF-8 字符串 attribute；表中数值标量通常对应 HDF5 `int64`、`float64` 或
`bool`。

### 3.1 录制、网格和完成状态

| Attribute | 类型/shape | 单位 | 物理意义 |
|---|---|---|---|
| `schema_version` | int 标量 | — | 固定为 `17`；读取器拒绝其他版本。 |
| `task_label` | UTF-8 字符串 | — | START 边界传入的任务标签。 |
| `operator` | UTF-8 字符串 | — | 操作者标识。 |
| `control_hz` | float64 标量 | Hz | 名义控制/录制网格频率。 |
| `fps` | float64 标量 | Hz | `control_hz` 的兼容别名，不是相机原始采集频率。 |
| `grid_dt_s` | float64 标量 | s | `1/control_hz`。 |
| `num_frames` | int64 标量 | 帧 | 发布 episode 中的网格槽数 `N`。 |
| `duration` | float64 标量 | s | 从 start 到 stop 的 wall/单调经过时间；兼容名称。 |
| `wall_duration_s` | float64 标量 | s | 与 `duration` 相同，包含暂停和未采样时段。 |
| `grid_duration_s` | float64 标量 | s | `max(0,N-1) * grid_dt_s`。 |
| `non_sampled_duration_s` | float64 标量 | s | `max(0,wall_duration_s-grid_duration_s)`。 |
| `wall_fps` | float64 标量 | 帧/s | `N/wall_duration_s`；包含暂停影响。 |
| `success` | bool 标量 | — | stop 是否请求保存成功；正常发布的 episode 应为 `True`。 |
| `min_frames_met` | bool 标量 | — | `N` 是否达到配置的最小帧数；是质量标签，不单独决定内部有效性。 |
| `truncated` | bool 标量 | — | 是否因达到 `max_frames` 截断。 |
| `stop_reason` | UTF-8 字符串 | — | 停止原因；未显式给出时通常为 `manual` 或 `max_frames`。 |
| `skip_initial_frames` | int64 标量 | 帧 | START 后跳过的过渡帧数；这些帧完全不进入网格。 |
| `has_timestamps` | bool 标量 | — | `data.h5` 是否含 `timestamp`，有效 v17 应为真。 |
| `resolved_config_sha256` | 64 字符字符串 | — | 已解析运行配置的 SHA-256，不是配置正文。 |
| `arm_sent_stream` | bool 标量，条件存在 | — | 表示存在 `action_arm_joint_sent`。标准 RecorderIO v17 录制会写 `True`。 |

### 3.2 相机身份、标定和编码

| Attribute | 类型/shape | 单位 | 条件与意义 |
|---|---|---|---|
| `camera_name` | UTF-8 字符串 | — | 条件存在；由在线序列号解析出的 `cameras.json` 条目名。 |
| `camera_serial` | UTF-8 字符串 | — | 条件存在；实际 RealSense 序列号。 |
| `camera_type` | 字符串 | — | 条件存在；`eye_to_hand` 或 `eye_in_hand`。 |
| `camera_K` | float64 `(9,)` | pixel | 条件存在；aligned synthetic frame/共同像素 viewport 的 3×3 内参。标准生产模式下是 color K；它不表示 aligned Z16 已成为 color-optical Z，也不包含 distortion。 |
| `camera_T_world_camera` | float64 `(16,)` | 平移 m | eye-to-hand 时条件存在；标定得到的 color-optical camera → world 变换。 |
| `camera_T_eef_camera` | float64 `(16,)` | 平移 m | eye-in-hand 时条件存在；标定得到的 color-optical camera → EEF 变换。 |
| `depth_scale` | float64 标量 | m/raw-unit | 条件存在；`depth * depth_scale` 得米，L515 常见值为 0.00025，但必须使用文件内值。 |
| `camera_firmware` | 字符串 | — | 相机固件版本；不可用时为 `unknown`。 |
| `camera_sdk_version` | 字符串 | — | pyrealsense2/SDK 版本；不可用时为 `unknown`。 |
| `camera_actual_profile_json` | JSON 字符串 | — | 实际 RGB/depth stream profile 与 align mode；标准生产 episode 的 mode 必须为 `depth_to_color`。 |
| `camera_alignment_mode` | 字符串 | — | 条件存在；标准生产值为 `depth_to_color`，并与实际 profile 交叉校验。 |
| `camera_common_viewport` | 字符串 | — | 条件存在；标准生产值为 `color`，说明处理后 RGB/depth 共用 color viewport。 |
| `camera_K_optical_frame` | 字符串 | — | 条件存在；标准生产值为 `camera_color_optical`。 |
| `camera_output_optical_frame` | 字符串 | — | 条件存在；当前 writer 声明值为 `camera_color_optical`。由于 aligned Z16 仍是 source-depth Z，现有点云算法并未严格证明这一 3D frame；消费方应把它视为 producer 声明而非几何真值。 |
| `camera_pointcloud_config_json` | JSON 字符串 | 多单位 | 点云过滤、裁剪和采样配置；若处理器未启用则通常为 `{}`。 |
| `camera_writer_queue_size` | int64 标量 | 项 | Recorder 侧车 writer 的有界队列容量。 |
| `camera_encoding_codec` | 字符串 | — | `rgb.mp4` 编码器名称。 |
| `camera_encoding_crf` | int64 标量 | — | RGB 编码 CRF。 |
| `camera_encoding_preset` | 字符串 | — | RGB 编码 preset。 |
| `camera_encoding_pixel_format` | 字符串 | — | RGB 编码输出像素格式。 |
| `camera_encoding_width` | int64 标量 | pixel | `IW`。 |
| `camera_encoding_height` | int64 标量 | pixel | `IH`。 |
| `camera_encoding_fps` | float64 标量 | 帧/s | RGB 侧车写入率，必须等于 `control_hz`。 |
| `camera_depth_storage` | 字符串 | — | 当前固定说明 `uint16/gzip-1`。 |
| `camera_health_taxonomy_json` | JSON 字符串 | — | 相机健康码表，当前为 `0..4`，见第 7 节。 |
| `has_camera` | bool 标量 | — | 是否写入了相机网格帧；不等价于每帧都 fresh。 |
| `camera_stream_frames` | int64 标量 | 帧 | 每个相机侧车实际写入帧数，必须等于 `N`。 |
| `camera_writer_error` | 字符串 | — | 已发布 episode 中应为空；非空会使 reader 判为 INVALID。 |

### 3.3 设备身份、资源 provenance 和 writer 指标

| Attribute | 类型 | 单位 | 意义 |
|---|---|---|---|
| `arm_device_identity_json` | JSON 字符串 | — | xArm 设备身份快照；不可用时含 `status=unavailable`。 |
| `hand_device_identity_json` | JSON 字符串 | — | XHand 设备身份快照；禁用时含 `status=disabled`。 |
| `provenance_arm_hand_collision_urdf_sha256` | 64 字符字符串 | — | 碰撞 URDF 的 SHA-256。 |
| `provenance_arm_hand_urdf_sha256` | 64 字符字符串 | — | 机器人 URDF 的 SHA-256。 |
| `provenance_arm_hand_srdf_sha256` | 64 字符字符串 | — | SRDF 的 SHA-256。 |
| `provenance_camera_calibration_sha256` | 64 字符字符串 | — | `cameras.json` 的 SHA-256。 |
| `provenance_vr_heading_calibration_sha256` | 64 字符字符串 | — | `vr_transform.json` 的 SHA-256。 |
| `camera_writer_queue_high_watermark` | int64 | 项 | 录制期间 writer 队列最高占用。 |
| `camera_writer_queue_capacity` | int64 | 项 | writer 队列容量。 |
| `camera_writer_close_s` | float64 | s | drain、编码器/HDF5 关闭总时长。 |
| `camera_encode_p50_s` | float64 | s/帧 | RGB 单帧编码耗时的 50 百分位。 |
| `camera_encode_p95_s` | float64 | s/帧 | RGB 单帧编码耗时的 95 百分位。 |
| `camera_encode_p99_s` | float64 | s/帧 | RGB 单帧编码耗时的 99 百分位。 |
| `camera_encode_max_s` | float64 | s/帧 | RGB 单帧编码耗时最大值。 |
| `camera_hdf5_p50_s` | float64 | s/帧 | depth 单帧 HDF5 append 耗时的 50 百分位。 |
| `camera_hdf5_p95_s` | float64 | s/帧 | depth 单帧 HDF5 append 耗时的 95 百分位。 |
| `camera_hdf5_p99_s` | float64 | s/帧 | depth 单帧 HDF5 append 耗时的 99 百分位。 |
| `camera_hdf5_max_s` | float64 | s/帧 | depth 单帧 HDF5 append 耗时最大值。 |

`EpisodeRecorder` 还允许调用方通过 `provenance` 添加任意
`provenance_<key>` 字符串 attribute，并允许 START 时的 `camera_metadata` 添加小型
attribute。本节各表列出当前标准采集入口实际提供的全部键；未知扩展键不能被当作跨版本固定合同。

### 3.4 Frame、raw action 与物理量 provenance

下列 additive attributes 自 2026-08-16 的 v16 writer 起写入，用于把既有数值语义机器可读化；
它们不改变 dataset layout，历史 v16 缺少这些属性仍可由 reader 读取。

| Attribute | 类型 | 当前值/语义 |
|---|---|---|
| `robot_world_frame` | 字符串 | `xarm_base`。 |
| `robot_world_equals_xarm_base` | bool | `True`；当前受支持 runtime 的显式不变量。 |
| `arm_ee_frame` | 字符串 | `xarm_base`。 |
| `action_arm_ee_frame` | 字符串 | `xarm_base`；等价于当前 runtime world。 |
| `hand_fingertip_frame` | 字符串 | `xarm_base`。 |
| `action_arm_joint_raw_validity_expression` | 字符串 | `flag_sample_valid & ~flag_held & flag_ik_ok`。 |
| `action_hand_joint_raw_validity_expression` | 字符串 | `flag_sample_valid & ~flag_held & flag_retarget_ok`。 |
| `tactile_sdk_scale_factor` | float64 | `0.1`；保持部署数值尺度，不声明为 SI conversion。 |
| `tactile_unit` | 字符串 | `sdk_scaled_unknown_si`。 |
| `tactile_si_unit_verified` | bool | `False`。 |
| `tactile_bias_semantics` | 字符串 | 启动软件 bias 在可用时扣除；逐行看 `tactile_calibrated`。 |
| `tactile_contact_metric` | 字符串 | 每指 `hand_contact` 三轴的 L2 norm。 |
| `tactile_contact_threshold` | float64 | 逐帧 `hand_tactile_contact` 判定阈值，当前生产值 `1.0`，单位同缩放后 SDK 值。 |
| `raw_force_contact_threshold` | float64 | 启动标定“真接触”门禁用逐点 raw force，当前生产值 `1.0`，单位同缩放后 SDK 值。 |
| `tactile_contact_comparison` | 字符串 | `strict_greater_than`。 |
| `arm_tau_source` | 字符串 | `xarm_get_joint_states_num_3_effort`。 |
| `arm_tau_unit` | 字符串 | `unknown`。 |
| `arm_tau_si_unit_verified` | bool | `False`。 |
| `arm_tau_sensor_semantics` | 字符串 | current-estimated effort，不是 direct torque sensor。 |

## 4. `data.h5` datasets（93 个基础字段 + 1 个条件字段）

93 个基础 dataset 是所有合法 v17 episode 的无条件合同。仅当
`/meta.arm_sent_stream=True` 时，第 94 个 `action_arm_joint_sent` 必须存在；该 dataset 存在而
marker 缺失/为假同样是无效布局。标准 RecorderIO 总是启用此流，因此其正常输出仍为
94 个 dataset。

### 4.1 控制网格与因果填充

| Dataset | Shape | dtype | 单位 | 物理意义 |
|---|---:|---|---|---|
| `timestamp` | `(N,)` | float64 | monotonic s | 每个输出槽唯一且严格递增的网格时间。 |
| `flag_sample_valid` | `(N,)` | bool | — | `True` 表示该槽直接写入了一个真实 source sample；不是“整帧所有传感器均健康”。 |
| `source_sample_index` | `(N,)` | int64 | source 序号 | 真实 sample 的递增索引；hold 槽沿用上一个索引，leading placeholder 为 `-1`。 |
| `source_timestamp` | `(N,)` | float64 | monotonic s | 产生该槽数据的 source sample 时间；hold 槽沿用旧值，leading placeholder 为 `NaN`。 |
| `fill_reason` | `(N,)` | uint8 | 枚举 | `0=SOURCE`、`1=CAUSAL_HOLD_LAST`、`2=LEADING_PLACEHOLDER`。 |

填充是严格因果的：过去的 source 可以向后 hold，但未来 source 不能反填更早的网格槽。
因此，对非 placeholder 槽总有 `source_timestamp <= timestamp`。

### 4.2 机械臂状态与反馈

| Dataset | Shape | dtype | 单位 | 物理意义 |
|---|---:|---|---|---|
| `arm_qpos` | `(N,A)` | float64 | rad | xArm7 实测关节位置，SDK/模型关节顺序。 |
| `arm_qvel` | `(N,A)` | float64 | rad/s | 实测关节速度。 |
| `arm_tau` | `(N,A)` | float64 | SDK effort；精确 SI 单位未验证 | xArm `get_joint_states(num=3)` 返回的电流估算 effort；不是直接力矩传感器真值。启动和正常反馈均读取真实 effort；失败占位只能结合无效状态使用。详见第 10.5 节。 |
| `arm_ee` | `(N,9)` | float64 | 前 3 列 m | 实测 EEF pose：`[x,y,z,rot6d(6)]`，原生为 xArm base frame；当前受支持 runtime 明确令 base/world 重合，详见第 10.1 节。 |
| `arm_connected` | `(N,)` | bool | — | 该状态构建时机械臂反馈是否连接/可读。 |
| `arm_last_cmd_seq` | `(N,)` | int64 | 序号 | arm worker 最近成功处理的命令序号。 |
| `arm_last_cmd_queue_latency_s` | `(N,)` | float64 | s | 命令创建到 arm queue 接收的延迟。 |
| `arm_last_cmd_apply_latency_s` | `(N,)` | float64 | s | 命令创建到 SDK 成功返回的累计延迟。 |
| `arm_last_cmd_sdk_duration_s` | `(N,)` | float64 | s | 最近一次 `set_servo_angle()` 调用耗时。 |
| `arm_last_cmd_is_hold` | `(N,)` | bool | — | 最近成功命令是否为安全/IK fallback 的 hold endpoint。普通命令静默不产生 endpoint。 |

### 4.3 XHand 状态、触觉和健康

| Dataset | Shape | dtype | 单位 | 物理意义 |
|---|---:|---|---|---|
| `hand_qpos` | `(N,H)` | float64 | rad | XHand 实测关节位置，SDK/模型关节顺序。 |
| `hand_current` | `(N,H)` | float64 | mA | 每个手部电机的电流；不可用时为 `NaN`。 |
| `hand_fingertip` | `(N,F,3)` | float64 | m | 由 `arm_ee` 和 hand FK 链式计算的 5 个指尖位置，继承 xArm base frame；当前 runtime world 与其相同。不可计算时为 `NaN`；详见第 10.1 节。 |
| `hand_contact` | `(N,F,3)` | float64 | SDK-scaled，物理单位未验证 | 每指三轴触觉合力/汇总值，经过 `0.1` 缩放和可选 bias 扣除；不是布尔接触结果。详见第 10.3 节。 |
| `hand_tactile_force` | `(N,F,T,3)` | float64 | SDK-scaled，物理单位未验证 | 每指 120 个触觉点的三轴值，经过 `0.1` 缩放和可选 bias 扣除；不得解释成 N。详见第 10.3 节。 |
| `hand_tactile_contact` | `(N,F)` | bool | — | XHand `detect_contact` 给出的逐指接触判断。 |
| `hand_tipboard_err` | `(N,H)` | int32 | SDK error code | 每关节 tip board 错误寄存器。 |
| `hand_commboard_err` | `(N,H)` | int32 | SDK error code | 每关节 communication board 错误寄存器。 |
| `hand_jointboard_err` | `(N,H)` | int32 | SDK error code | 每关节 motor-driver/joint board 错误寄存器。 |
| `hand_connected` | `(N,)` | bool | — | 手部反馈是否连接/可读。 |
| `hand_error_state` | `(N,)` | bool | — | 是否检测到手部板级硬件错误。 |
| `hand_qpos_stale` | `(N,)` | bool | — | v17 兼容保留位；当前正常运行保持 false，真实 freshness 应看 source 时间和 observation mask。 |
| `tactile_fresh` | `(N,)` | bool | — | 独立 tactile ring 数据在 anchor 前且年龄不超过 250 ms；合成 hold 槽会清为 false。 |
| `tactile_source_monotonic_ns` | `(N,)` | int64 | monotonic ns | 触觉帧的 source 时间；`0` 表示无来源。 |
| `tactile_calibrated` | `(N,)` | bool | — | SDK/driver 是否声明触觉已标定。 |
| `tactile_unit_code` | `(N,)` | int64 | 枚举 | 当前只写 `0=UNKNOWN`；用于防止把未验证的触觉数值误标成物理力。 |

区分断连与读取失败时，应联合检查 `*_connected`、数值是否有限、source 时间和
`observation_history_valid_mask`，不能只检查 qpos 是否为 NaN。

### 4.4 动作与目标

| Dataset | Shape | dtype | 单位 | 物理意义 |
|---|---:|---|---|---|
| `action_arm_joint_raw` | `(N,A)` | float64 | rad | 正常 IK 路径保存应用层逐帧 delta clamp 和 joint-limit clip 之前的关节解；held/failure 路径没有显式 raw 解时回退为当帧 hold/最终 arm 命令。 |
| `action_arm_joint` | `(N,A)` | float64 | rad | 经过 delta/joint-limit 处理并通过 SafetyGate 的有效机械臂命令；hold 槽保存有效 hold 目标。 |
| `action_arm_joint_sent` | `(N,A)` | float64 | rad | 实际转发给 arm worker 的安全命令流。标准 RecorderIO v17 存在；手工构造且未启用 `arm_sent_stream` 时可缺省。 |
| `action_hand_joint_raw` | `(N,H)` | float64 | rad | 正常 retarget 路径位于 startup ramp、operational clip 和后续校验之前；cache hit 复用该 VR observation 的 solved 值。held/failure 路径没有显式 raw 值时回退为当帧 hold/最终 hand 命令。详见第 10.4 节。 |
| `action_hand_joint` | `(N,H)` | float64 | rad | 经过 startup ramp、operational clip 和后续校验的有效手部命令；当前没有应用侧 hand command-to-command delta clamp。 |
| `action_arm_ee` | `(N,9)` | float64 | 前 3 列 m | IK 实际追踪的 `[target_pos(3), target_rot6d(6)]`；没有目标时为 NaN，held 帧尽量沿用最后有效目标。 |
| `target_eef_pos_raw` | `(N,3)` | float64 | m | VR 映射后的原始 EEF 位置目标，位于 EMA/workspace clamp 之前。 |
| `target_eef_rot6d_raw` | `(N,6)` | float64 | — | 与上一字段配套的原始 EEF 旋转目标。 |
| `target_pos_before_clamp` | `(N,3)` | float64 | m | 已经过 VR 映射和 EMA、但尚未 workspace clamp 的位置目标。 |

三条 arm joint 流的典型处理顺序为：

```text
action_arm_joint_raw
    → teleop per-frame delta clamp + application joint-limit clip + SafetyGate
    → action_arm_joint
    → 实际入队/hold endpoint
    → action_arm_joint_sent
```

该顺序只描述存在显式 IK raw 解的正常路径。对 `flag_frame_status != OK` 的行，不得仅凭
字段名把 `action_arm_joint_raw` 或 `action_hand_joint_raw` 解释为求解器原始输出；优先使用
`EpisodeReader.action_arm_joint_raw_valid_mask` 和
`EpisodeReader.action_hand_joint_raw_valid_mask`。

### 4.5 Observation/action 因果协议

以下四路数组的轴 1 顺序固定为 **`[arm, hand, vr, camera]`**。

| Dataset | Shape | dtype | 单位 | 物理意义 |
|---|---:|---|---|---|
| `observation_id` | `(N,)` | int64 | 标识 | observation 身份；通常与 action candidate 绑定，否则回退为 VR ring sequence 或 anchor ns。 |
| `observation_anchor_monotonic_ns` | `(N,)` | int64 | monotonic ns | 本控制槽的因果截止/网格锚点。每槽必须大于 0。 |
| `arm_source_sequence` | `(N,)` | int64 | ring sequence | 被选中 arm state 的 verified ring 序号。 |
| `hand_source_sequence` | `(N,)` | int64 | ring sequence | 被选中 hand state 的 verified ring 序号；手部禁用时可为 0。 |
| `vr_source_sequence` | `(N,)` | int64 | ring sequence | 被选中 VR frame 的 verified ring 序号。 |
| `camera_source_sequence` | `(N,)` | int64 | ring sequence | 被选中 camera frame 的 verified ring 序号。 |
| `arm_source_monotonic_ns` | `(N,)` | int64 | monotonic ns | arm 反馈的源采样时间。 |
| `hand_source_monotonic_ns` | `(N,)` | int64 | monotonic ns | hand 反馈的源采样时间。 |
| `vr_source_monotonic_ns` | `(N,)` | int64 | monotonic ns | VR 帧在本机接收的 source 时间。 |
| `camera_source_monotonic_ns` | `(N,)` | int64 | monotonic ns | 由设备时钟映射到主机 monotonic 域的相机源时间。 |
| `arm_publish_monotonic_ns` | `(N,)` | int64 | monotonic ns | arm state 发布到 shared ring 的时间。 |
| `hand_publish_monotonic_ns` | `(N,)` | int64 | monotonic ns | hand state 发布到 shared ring 的时间。 |
| `vr_publish_monotonic_ns` | `(N,)` | int64 | monotonic ns | VR frame 发布到 shared ring 的时间。 |
| `camera_publish_monotonic_ns` | `(N,)` | int64 | monotonic ns | camera frame 发布到 shared ring 的时间。 |
| `observation_source_receive_monotonic_ns` | `(N,4)` | uint64 | monotonic ns | 各 source 到达本机/owner 的时间；arm/hand 使用其 publish 时间，VR 使用本机接收时间，camera 使用 capture receive 时间。 |
| `observation_source_age_s` | `(N,4)` | float64 | s | `anchor - source_time`；该路无有效历史或不满足因果性时为 NaN。 |
| `observation_source_skew_s` | `(N,4)` | float64 | s | 最新有效 source 时间减去该路 source 时间；无效路为 NaN。 |
| `observation_history_valid_mask` | `(N,4,1)` | bool | — | 各路是否具有完整 `sequence/source<=receive<=publish<=anchor` 因果链。末尾 singleton 维是固定 schema 的一部分。 |
| `observation_valid` | `(N,)` | bool | — | 必需 source 均有效且最大 source skew 不超过 0.10 s；合成 hold 槽强制为 false。 |
| `observation_skew_s` | `(N,)` | float64 | s | 四路 `observation_source_skew_s` 的最大有效值。 |
| `action_id` | `(N,)` | int64 | 标识 | action candidate 身份；没有发布动作时为 0。合成 hold 槽清零。 |
| `action_created_monotonic_ns` | `(N,)` | int64 | monotonic ns | action candidate 创建时间；无动作时为 0。 |
| `action_target_monotonic_ns` | `(N,)` | int64 | monotonic ns | 动作预期应用/控制目标时间。 |
| `action_valid_until_monotonic_ns` | `(N,)` | int64 | monotonic ns | 动作有效期截止时间。 |
| `flag_action_queued` | `(N,)` | bool | — | 此 source 槽是否确实把 action 发布到命令边界；合成槽不会伪造发送事件。 |

合法的已入队动作满足
`created <= target <= valid_until` 且 `action_id != 0`。普通 active source 槽必须有动作；
held source 槽可以有明确 hold action，也可以有意不发布新动作。

### 4.6 VR 输入

| Dataset | Shape | dtype | 单位 | 物理意义 |
|---|---:|---|---|---|
| `vr_wrist_pos` | `(N,3)` | float64 | m | 右手腕在 VR/操作者 FLU 坐标系的位置。 |
| `vr_wrist_rot6d` | `(N,6)` | float64 | — | 右手腕 FLU 姿态；接收端原始四元数在写 HDF5 前转换为 rot6d。 |
| `vr_landmarks` | `(N,21,3)` | float64 | m | 右手 21 个 hand landmarks，在操作者/VR FLU 坐标系。 |
| `head_quat_wxyz` | `(N,4)` | float64 | — | 最近的头部 FLU 单位四元数，用于 heading/诊断；缺失时为 NaN。 |

HDF5 没有保存 VR 的 `head_pos`、原始 wrist quaternion、远端 source timestamp 或
HTS sequence id；上述四个 dataset 是 v17 中持久化的 VR 内容全集。

### 4.7 相机逐槽状态与 depth 质量

| Dataset | Shape | dtype | 单位 | 物理意义 |
|---|---:|---|---|---|
| `camera_health` | `(N,)` | int64 | 枚举 | 当前相机帧健康分类，见第 7 节；没有 camera frame 时默认 1。 |
| `flag_camera_fresh` | `(N,)` | bool | — | 帧为新 ring sequence、新 frame number、录制开始后、health=OK 且年龄不超阈值；合成槽清 false。 |
| `camera_frame_number` | `(N,)` | int64 | 设备帧号 | RealSense depth frame number。 |
| `camera_ring_sequence` | `(N,)` | int64 | ring sequence | shared camera ring 的 verified 序号。 |
| `camera_device_timestamp_s` | `(N,)` | float64 | device s | RealSense `get_timestamp()` 从 ms 转成 s；设备时钟域。 |
| `camera_capture_monotonic_s` | `(N,)` | float64 | monotonic s | 主机从 SDK 收到帧时的 capture/receive 时间。 |
| `camera_age_s` | `(N,)` | float64 | s | policy 检查时 `now - camera_source_time`，下限截为 0。 |
| `camera_generation` | `(N,)` | int64 | 标识 | 相机时钟映射 generation；时钟重置会更新 generation。 |
| `camera_clock_reset` | `(N,)` | bool | — | 本帧是否检测到设备时钟重置/重新锚定。 |
| `camera_duplicate` | `(N,)` | bool | — | 本帧是否被设备帧号/时钟分类为重复。 |
| `camera_frame_gap` | `(N,)` | int64 | 帧 | 相对预期序列缺少的设备帧数，0 表示连续。 |
| `camera_backlog_s` | `(N,)` | float64 | s | host receive 与映射 source time 的积压/传输延迟。 |
| `pointcloud_valid_depth_ratio` | `(N,)` | float64 | 比例 `[0,1]` | 原深度图中有限且大于 0 的像素比例，位于后续深度/空间过滤之前。它是 depth 派生质量指标，不是存储点云的有效性标志。 |

相机 payload 始终逐网格槽写入：真实 source 槽写当前 RGB/depth；因调度产生的 causal
hold 槽会沿用上一 payload；没有历史时用全零图像。点云不再作为 `pointcloud.h5` 存储；
消费方在需要时从 `depth.h5` 配合 `/meta` 内参/外参与 `camera_pointcloud_config_json`
确定性派生。

### 4.8 IK、安全、retarget 与性能诊断

| Dataset | Shape | dtype | 单位 | 物理意义 |
|---|---:|---|---|---|
| `flag_ik_ok` | `(N,)` | bool | — | 此 source 槽 IK 是否成功。 |
| `flag_ik_attempted` | `(N,)` | bool | — | 是否实际尝试 IK；普通主动帧默认 true，纯 hold 路径可为 false。 |
| `flag_retarget_ok` | `(N,)` | bool | — | 当前 VR ring sequence 是否有成功 solved hand endpoint；cache hit 可为 true，不表示本控制 tick 实际执行了 solver。 |
| `flag_held` | `(N,)` | bool | — | 是否记录的是 hold/fallback 行为。 |
| `flag_safety_reject` | `(N,)` | bool | — | SafetyGate/在线安全检查是否拒绝原动作。 |
| `flag_frame_status` | `(N,)` | int64 | 枚举 | 综合帧状态，见第 7 节。 |
| `tracking_error` | `(N,)` | float64 | rad | arm worker 的最大绝对关节跟踪误差。 |
| `ik_solve_time_ms` | `(N,)` | float64 | ms | 单次 teleop IK 求解耗时。 |
| `policy_map_time_ms` | `(N,)` | float64 | ms | VR wrist → EEF 映射、EMA 和 workspace clamp 阶段耗时。 |
| `hand_retarget_time_ms` | `(N,)` | float64 | ms | 本控制 tick 的 hand retarget/cache lookup 阶段耗时；新 VR ring sequence 通常包含 solver，cache hit 只测复用开销。 |
| `transition_check_time_ms` | `(N,)` | float64 | ms | 兼容保留字段；旧的 SafetyGate transition/collision 检查已移除，当前正常 source 帧写 0。 |
| `policy_compute_time_ms` | `(N,)` | float64 | ms | 本控制 tick 从 policy compute 起点到准备录制前的累计计算耗时。 |

诊断在某些 hold/错误路径不可用时会保留 `NaN`。分析耗时分布前应先筛选 finite 值，
再结合 `flag_sample_valid` 与 `flag_frame_status`。

## 5. 相机 HDF5 侧车（1 个 dataset）

### 5.1 `depth.h5`

| HDF5 path | Shape | dtype | chunk | 压缩 | 单位与意义 |
|---|---:|---|---|---|---|
| `/depth` | `(N,IH,IW)` | uint16 | `(1,IH,IW)` | gzip level 1 | 对齐处理后的 Z16。`depth_to_color` 时像素落点位于 RGB/color viewport，但数值仍是 source depth-camera 的轴向 Z；`color_to_depth` 时 viewport 为 depth。米值为 `depth.astype(float32) * /meta.depth_scale`，0 表示无效/占位。 |

不得固定假设 1 raw unit = 1 mm；L515 当前常见 scale 是 0.00025 m，但每个 episode
必须读取自己的 `/meta.depth_scale`。

还必须读取 `/meta.camera_actual_profile_json` 的 `align_mode`、两路 distortion，并区分
pixel viewport 与 depth-value optical frame；不能把“对齐到 RGB 像素”解释为“已得到严格
color-camera pinhole depth”。详见第 10.2 节。

### 5.2 派生点云（不再存储为 `pointcloud.h5`）

schema v17 不再保存 `pointcloud.h5`。世界点云是 `depth.h5` 的纯派生函数，由消费边界按需
计算：训练在 `data_processing` 离线派生，推理在视觉观测适配层在线派生。两者复用同一个
`PointCloudProcessor`，且派生是确定性的（无 RNG/seed）：

```text
depth + /meta 内参/外参 + desk_plane + camera_pointcloud_config_json
    → PointCloudProcessor → (M,6) float32，每点 [x_world,y_world,z_world,r,g,b]
```

处理流程包括深度有效性/范围与边缘过滤、camera→world 标定变换、桌面和 workspace
裁剪、体素/离群点过滤以及定长采样。点数多于 `M` 时确定性下采样；少于 `M` 时按固定顺序
循环补点（不再是随机重复）。派生点云仍标为 approximate：当前管线用 color K 的 ray 直接乘
aligned source-depth Z，且未把非零 color distortion 纳入反投影，因此不应作为标定级 world
geometry 真值（详见第 10.2 节）。

## 6. `rgb.mp4` 与 HDF5 的关系

RGB 不是 HDF5 dataset，但属于 v17 episode 必需侧车：

- 解码后逻辑 shape 为 `(N,IH,IW,3)`，输入 dtype 为 `uint8` RGB，数值 0–255。
- 编码 codec、CRF、preset、pixel format、宽高和网格 FPS 记录在 `/meta`。
- 编码帧数必须等于 `/meta.num_frames` 和 `/depth.shape[0]`，否则
  `EpisodeReader.validity` 为 INVALID。
- 有效性/新鲜度由 `flag_camera_fresh` 等逐槽字段表达；不能仅凭 MP4 中存在非黑图像
  推断它是该槽的新 source frame。

## 7. 枚举、标志与有效性

### 7.1 `fill_reason`

| 值 | 名称 | 含义 |
|---:|---|---|
| 0 | `SOURCE` | 该网格槽直接承载当前真实 sample，`flag_sample_valid=True`。 |
| 1 | `CAUSAL_HOLD_LAST` | 调度跨过网格 deadline，向后复制最近的过去 sample。 |
| 2 | `LEADING_PLACEHOLDER` | 尚无可因果使用的过去 sample；数值按 dtype 使用 NaN/0 占位。 |

### 7.2 `camera_health`

| 值 | 名称 | 含义 |
|---:|---|---|
| 0 | `OK` | 时钟和帧序列健康。 |
| 1 | `CLOCK_RESET` | 设备时钟重置或无相机帧时的保守默认。 |
| 2 | `DUPLICATE` | 重复设备帧。 |
| 3 | `FRAME_GAP` | 检测到设备帧号间隙。 |
| 4 | `BACKLOG` | backlog 超过运行时最大帧年龄阈值。 |

`camera_health` 门控 `flag_camera_fresh`：`flag_camera_fresh` 要求 health=OK。FRAME_GAP 只是
设备帧号跳号（相机循环短暂超预算），其 depth 仍新鲜有效，但 `flag_camera_fresh` 记 false，
以保留跳号信号供下游过滤。点云有效性由 depth 派生质量（`pointcloud_valid_depth_ratio`）与
相机 freshness 共同决定，不再有独立的 `flag_pointcloud_valid` 标志。

### 7.3 `flag_frame_status`

| 值 | 名称 | 含义 |
|---:|---|---|
| 0 | `OK` | IK 和必要的 retarget 正常，动作走正常路径。 |
| 1 | `HELD` | 主动 hold/fallback，通常未尝试 IK。 |
| 2 | `IK_FAIL` | IK 失败，机械臂保持，手部可能仍独立运动。 |
| 3 | `SAFETY_REJECT` | 在线安全检查拒绝原目标，记录安全 hold。 |
| 4 | `RETARGET_FAIL` | IK 成功但手部 retarget 失败/回退。 |

### 7.4 判断可训练/可回放数据

`EpisodeReader.validity == VALID` 至少要求：

1. schema 为 17，配置 SHA-256 长度正确，保存成功且 writer 无错误；
2. 必需 dataset 全部存在、首维等于 `N`，depth 与 MP4 帧数一致；
3. `timestamp` 有限、严格递增，填充理由与 source index/timestamp 自洽；
4. 有效 observation 满足四阶段因果时间链；
5. queued action 有非零 ID 且 action 时间有序；
6. source sample 的动作数组 shape 正确且数值有限。

`VALID` 只表示格式和核心因果合同成立。训练时仍应根据任务筛选
`flag_sample_valid`、`observation_valid`、`flag_camera_fresh`、
`flag_frame_status`、连接状态及触觉 freshness。

reader 与 writer finalizer 现复用 `recording.episode_schema`：无条件要求 93 个基础字段，
再按 `/meta.arm_sent_stream` 校验 `action_arm_joint_sent`。历史自定义 dataset 可以继续读取，
但也必须是逐帧 dataset 且首维等于 `N`。完整布局校验并不替代逐行质量筛选；例如 raw
action 的语义有效性、触觉 freshness 和 depth 派生质量仍须使用各自字段。

## 8. 读取示例

### 8.1 使用统一 reader

```python
from dexmani_real.recording.episode_reader import EpisodeReader

episode_dir = "data/episode_20260815_120000"

with EpisodeReader(episode_dir) as reader:
    reader.require_valid("offline analysis")
    f = reader.h5f

    arm_qpos_rad = f["arm_qpos"][:]          # (N, 7), float64
    depth_raw = f["depth"][:]                # (N, IH, IW), uint16
    pointcloud_world = f["pointcloud"][:]    # (N, M, 6), float32
    rgb = reader.read_camera_all("rgb")       # decoded (N, IH, IW, 3)

    meta = f["meta"].attrs
    depth_m = depth_raw.astype("float32") * float(meta["depth_scale"])
```

### 8.2 只选择真实、因果有效且相机新鲜的槽

```python
with EpisodeReader(episode_dir) as reader:
    f = reader.h5f
    keep = (
        f["flag_sample_valid"][:]
        & f["observation_valid"][:]
        & f["flag_camera_fresh"][:]
        & (f["flag_frame_status"][:] == 0)
    )
    observations = f["arm_qpos"][:][keep]
    actions = f["action_arm_joint_sent"][:][keep]
```

点云任务应从 `depth.h5` 派生点云，并按 `pointcloud_valid_depth_ratio` 筛选。如果读取的是手工创建的 v17
episode，应先检查 `"action_arm_joint_sent" in f`；标准 RecorderIO 录制包含该流。

### 8.3 只解释确实存在的 raw action

```python
with EpisodeReader(episode_dir) as reader:
    reader.require_valid("raw-action analysis")
    arm_raw_ok = reader.action_arm_joint_raw_valid_mask
    hand_raw_ok = reader.action_hand_joint_raw_valid_mask

    arm_raw = reader.h5f["action_arm_joint_raw"][:][arm_raw_ok]
    hand_raw = reader.h5f["action_hand_joint_raw"][:][hand_raw_ok]
```

这两个 mask 是保守语义边界，不会改写 HDF5。被排除的 held/failure 行仍保留 v17 的兼容
fallback 数值，但不能被解释为求解器/retargeter 的原始输出。

## 9. 容易混淆但未持久化的信息

- `data.h5` 没有保存完整 resolved config，只保存 SHA-256；配置还原需依赖外部实验配置。
- 没有保存 arm/hand SDK live object、共享内存 dtype、worker heartbeat 或主进程安全状态机轨迹。
- 没有保存 EEF quaternion dataset；`arm_ee` 和 `action_arm_ee` 使用 rot6d。
- 没有保存原始 VR wrist quaternion；`vr_wrist_rot6d` 是其持久化表示。
- 没有为 RGB 创建 `/rgb` HDF5 dataset；RGB 只在 `rgb.mp4`。
- `hand_contact` 不是 contact bool；布尔接触字段是 `hand_tactile_contact`。
- `has_camera=True` 不保证每槽相机数据 fresh；逐槽判断必须使用 `flag_camera_fresh`。
- `flag_sample_valid=True` 只说明槽来自真实 recorder source，不代表 observation、IK、相机、点云或触觉均有效。

## 10. 已知问题与解释边界

本节记录 2026-08-15 初审及 2026-08-16 fact-check 对 writer、producer、reader、标定程序、
已安装 SDK 和厂商公开合同进行交叉核查后确认的问题及修复。7 项均只涉及 DexMani Real；
未修改第 11 节的 DexMani Sim 合同。此次修复不改变 v17 dataset 的名称或数值含义，也不
伪造无法由现有数据证明的物理单位。

| ID | 优先级 | 影响字段 | 成因分类 | 修复状态 | 必要性结论 |
|---|---|---|---|---|---|
| H5-01 | P2 | `arm_ee`、`hand_fingertip`、`action_arm_ee` | 隐含 frame 不变量与误导命名 | 已修复 | 当前运行没有数值错误；为防未来非单位 base pose 被静默误用，需要自描述 metadata 和明确命名。 |
| H5-02 | P1 | `/depth`、RGB、`camera_K`、世界点云 | pixel viewport、depth scalar frame、distortion 与外参合同不完整 | 部分修复 | 已拒绝 `color_to_depth/none`；`depth_to_color` 仍原样携带 source-depth Z，且 K 不编码 distortion，现有点云只能视为 approximate。 |
| H5-03 | P1 | `hand_contact`、`hand_tactile_force`、`hand_tactile_contact` | 缺失 payload 被零填充；单位 provenance 不足 | 已修复可修部分 | 必须阻止缺失传感器被当作“有效零接触”；SI 换算仍无证据，故保留 unknown 而不是猜测。 |
| H5-04 | P1 | `action_arm_joint_raw`、`action_hand_joint_raw` | v17 fallback 混合了 raw 与最终/hold 语义 | 已修复消费合同 | 训练 clamp/retarget 差异前必须有逐行 mask；改变存值或新增 valid dataset 会改变 v17 合同，本轮不做。 |
| H5-05 | P2 | `arm_tau`、启动首帧 `arm_qvel` | 单位误注释；启动只读 position 后伪造零值 | 已修复可修部分 | 有效首帧不能包含人工 qvel/effort；精确物理单位仍待对应机型/固件的厂商合同。 |
| H5-06 | P3 | diagnostics 扩展 | 开放 key 覆盖与首帧动态 schema | 已修复 | 标准入口风险低，但底层 API 必须拒绝未知 key、核心冲突和 shape 漂移。 |
| H5-07 | P1 | 全部 `data.h5` datasets | reader/writer 各自维护不完整合同 | 已修复 | `VALID` 是所有消费入口的信任边界，必须覆盖完整 93+1 条件布局。 |

### 10.1 H5-01：`arm_ee` 与 `hand_fingertip` 的坐标系依赖

成因事实链如下：

1. arm worker 使用 `planning.kinematics.ArmFK.compute(qpos)` 生成 `eef_pos` 和
   `eef_rot6d`；该函数明确定义输出 arm base frame。
2. 标准 teleop 将 `base_pose_world` 固定为单位 pose，arm mapper 也明确假定
   `world == xarm_base`；workspace 和障碍定义在同一 base frame。
3. `teleop.episode_samples._build_robot_state()` 的数值链正确，但局部变量曾误命名为
   `T_world_*`，令读者误以为代码执行了额外 world 变换。

必要性评估为 P2：当前受支持 runtime 中这些数值一致，不应把 H5-01 当作现存数据损坏，
更不能在 v17 内突然变换 `arm_ee`。实施上仅把局部变量改为 `T_base_*`，并为新 episode
写入 `robot_world_frame=xarm_base`、`robot_world_equals_xarm_base=True`、
`arm_ee_frame`、`action_arm_ee_frame` 和 `hand_fingertip_frame`。未来若支持非单位 base pose，
必须升级坐标合同，并一起审计 mapper translation、指尖 FK、相机、点云和 replay。

代码定位：`robot/arm_loop.py::_arm_fk.compute`、
`planning/kinematics.py::ArmFK.compute`、`teleop/loop.py::XArm7PlannerConfig`、
`teleop/episode_samples.py::_build_robot_state`。

### 10.2 H5-02：共同 pixel viewport 不等于共同 depth optical frame

原问题表述“`color_to_depth` 时 RGB/depth 没有共同 viewport”不成立。RealSense align 会生成
共享 resolution/像素 viewport：`depth_to_color` 的目标是 color pixels，`color_to_depth` 的
目标是 depth pixels。[RealSense Projection 文档](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0)
说明了该投影关系。

但进一步核查本环境使用的 librealsense 2.54.2 实现后，发现共同 viewport 不能推出共同
depth-value frame：

1. `align_images()` 用 source depth intrinsics 和 depth→target extrinsic 计算目标 pixel；
2. `align_z_to_other()` 随后把原 `z_pixels[z_pixel_index]` 直接复制到目标 pixel，并未写入
   变换后 3D 点的 target-camera `z`；
3. synthetic aligned profile 却改用 target color intrinsics；当前 Real producer 再用这个
   color K 做 `K^-1 ray * source_depth_z`，忽略可能非零的 color distortion，最后应用
   color-optical 外参。

官方实现证据见
[librealsense v2.54.2 `align.cpp`](https://raw.githubusercontent.com/IntelRealSense/librealsense/v2.54.2/src/proc/align.cpp)
的 `align_z_to_other()` 与 `create_aligned_profile()`。因此当前生产点云是近似几何，不能把
`camera_output_optical_frame=camera_color_optical` 当作严格证明。

本轮代码已让生产 `camera_loop` 和 `RecorderIOConfig` fail-closed 为 `depth_to_color`，从而
消除了旧 `color_to_depth` 路径把 depth-frame rays 直接套入 color 外参的更大错误；START
也会交叉核验 mode/profile 并记录四个 camera geometry attributes。但这只是**部分修复**。

严格修复需要保存 source depth K、depth↔color extrinsic、两路 distortion 和未对齐 depth，
逐像素恢复 depth-camera 3D 点、变换到 color frame 并重算 `z_C`，再进行遮挡处理；或在采集
时保存 SDK 生成且 frame 明确的 3D vertices。现有历史 v16 缺少上述充分 provenance，无法
离线恢复严格 color-optical depth/pointcloud。转换和训练必须将其标为 approximate。

代码定位：`examples/calibrate_camera.py`、`sensor/realsense.py::RealSense.read`、
`utils/pointcloud_utils.py::make_rays`、`sensor/camera_process.py::camera_loop`、
`recording/io_process.py::_camera_geometry_from_profile`。

### 10.3 H5-03：触觉值不是未经处理的 SDK 原值

这里有两个成因。第一，`hand_tactile_force` 和作为 `hand_contact` 写入的逐指三轴汇总值都
经过以下处理：

1. SDK 各轴读数乘以 `0.1`；该因子用于兼容现场阈值，不代表已验证的 SI 单位换算；
2. 如果启动阶段成功估计了软件 bias，则逐点或逐指减去对应 bias；
3. `hand_tactile_contact[f]` 按
   `norm(hand_contact[f, :], ord=2) > tactile_contact_threshold` 计算；
4. 默认 `tactile_contact_threshold=1.0`，单位与缩放后的三轴汇总值相同，仍不得解释为 N。

启动标定的“真接触”门禁不用 `calc_force`，而用逐点 raw force（`raw_force_contact_threshold`，
默认 1.0）：部分传感器的 `calc_force` 通道带残余 DC 偏移，会把未加载的手误判为“有接触”
而拒绝标定。raw force 无该偏移，是更可靠的判据；raw force 缺失或无效时回退 `calc_force`
判据（fail-closed）。标定通过后，软件 bias 仍从 `calc_force` 均值扣除以吸收该 DC 偏移。

第二，旧 `_iter_sensors()` 接受少于 5 个传感器，`parse_tactile*()` 又以零初始化；因此完整
关节帧配上空 `sensor_data` 会被解释为全零力/无接触，甚至可能进入 calibration 流。这是
真实的数据真实性 bug，而不只是单位文档问题。

实施上，XHand 边界要求恰好 5 个 sensor、每个 `raw_force` 恰好 120 点，且 calc/raw 的
所有 xyz 有限。失败时进程内 `XHandSample.tactile_valid=False`，保留同帧完整关节反馈，
hand worker 立即发布 `fresh=False, calibrated=False` 的触觉帧；启动 calibration 也
fail-closed。v17 已有 freshness/calibration/unit 字段，因此无需新增 dataset。metadata 记录
缩放、bias、阈值规则和 `tactile_si_unit_verified=False`；消费方仍必须联合三个逐行标志，
不能把数值解释为 N。

代码定位：`robot/xhand.py::_parse_tactile_payload`、
`robot/xhand.py::XHandSample.tactile_valid`、`robot/hand_process.py::_build_tactile_frame`。

### 10.4 H5-04：两个 raw action 字段都是分路径定义

正常 action 路径会显式传入 raw 值：`action_arm_joint_raw` 是应用层 delta clamp 和
joint-limit clip 之前的 IK 关节解；`action_hand_joint_raw` 位于 startup ramp、operational
clip 和 `_sanitize_hand_command()` 后续校验之前，但其更上游含义取决于 retargeter：

- TAG 成功且 `last_raw_qpos` 有效：保存 SDK joint order 下的优化器输出；
- 非 TAG retargeter、retarget 失败、raw shape/finite 校验失败：保存 retargeter 返回的命令；
- 正常 action 调用方没有显式传值：录制层分别回退为当帧 `arm_cmd`/`hand_cmd`；
- `_record_held()` 不提供两个 raw 字段，因此 IK failure、safety reject 和其他主动 hold 行
  会把 hold/最终 arm、hand 命令写入对应的 `*_raw` dataset。

因果控制网格可以连续选择同一个 VR ring sequence。此时 stateful retargeter 不会再次运行：
成功 solve 的 raw endpoint 被复用，失败则继续记录当前合法 hold；因此相邻 source 行可以有
相同 `observation_id`/raw hand 值，而各自仍有独立 `action_id` 和 grid timestamp。

根因不只在 `_record_held()`：RecorderClient 和 EpisodeRecorder 都有兼容 fallback，而旧
reader 又要求所有 source 行的 raw 数值有限，三者共同迫使“不存在的 raw”伪装成最终命令。

必要性为 P1，但不能静默更改 v17 数值意义。实施上保留历史兼容存值，并由 reader 暴露
两个保守 mask：

```python
arm_raw_ok = reader.action_arm_joint_raw_valid_mask
hand_raw_ok = reader.action_hand_joint_raw_valid_mask
# arm  = flag_sample_valid & ~flag_held & flag_ik_ok
# hand = flag_sample_valid & ~flag_held & flag_retarget_ok
```

相同表达式也写入新 episode metadata。需要 NaN 语义或独立 raw-valid dataset 时，应升 schema
并同步 writer、reader、训练、可视化与 replay，而不是在 v17 中就地修改。

代码定位：`teleop/hand_retarget.py::TAGHandRetargeter.last_raw_qpos`、
`teleop/hand_control.py::_get_raw_hand_command`、`teleop/loop.py::teleop_loop`、
`teleop/episode_samples.py::_record_frame`、`teleop/episode_samples.py::_record_held`、
`recording/episode_recorder.py::EpisodeRecorder.add_frame`。

### 10.5 H5-05：`arm_tau` 是电流估算 effort，不是直接力矩测量

`arm_tau` 原样持久化 xArm SDK `get_joint_states(is_radian=True, num=3)` 返回值中的第三个
数组。SDK 将其命名为 `effort`；UFACTORY 的 ROS 说明进一步指出 effort 是基于电流的估计
值，而不是直接关节力矩传感器反馈，只适合参考。旧 `RobotState` 注释错误地写成 N·m，
但已核查的 SDK API 没有明确给出这个返回数组的精确单位。

另一个窄窗 bug 是 arm worker 启动只请求 `num=1` position，随后把 qvel/effort 人工置零并
发布 `state_valid=True`。标准人工开始录制通常晚于该窗口，但合法有效状态不应包含伪造值。

在获得与当前 xArm7 机型、固件版本对应的厂商单位合同前：

- 可把该字段用于同一设备、同一固件和相同配置下的相对趋势分析；
- 不应把它当成经过标定的关节力矩真值；
- 不应直接用于跨设备动力学标定、接触力反演或安全阈值设计。

实施上，启动 readiness 前即请求 `num=3`，通过与稳态相同的
`_decode_joint_state_feedback()` 严查并发布真实 qvel/effort；失败继续使用原有有界重试，
不再发布“有效零值”。注释与 metadata 改为 current-estimated effort、unit unknown、
SI unverified，未做任何 N·m 换算。

代码定位：`robot/arm_loop.py::_decode_joint_state_feedback`、
`robot/arm_loop.py::arm_loop`、已安装 xArm SDK `XArmAPI.get_joint_states`。
外部依据：[UFACTORY Python SDK `get_joint_states`](https://github.com/xArm-Developer/xArm-Python-SDK/blob/master/xarm/wrapper/xarm_api.py)、
[UFACTORY xarm_ros 状态反馈说明](https://github.com/xArm-Developer/xarm_ros#obtaining-status-feedback)。

### 10.6 H5-06：diagnostics 不再是隐式 schema 扩展口

旧底层 API 将任意 diagnostics 直接 `data[key]=value`。首帧额外 key 会冻结成自定义
dataset，后续新 key 被 buffer 静默忽略，而与核心字段同名的 key 还可覆盖默认语义。

必要性为 P3，因为标准 RecorderIO 只传固定字段，但开放的底层边界会造成难以审计的 v17
变体。现在 diagnostics 只允许 11 个已声明 value override，逐项转换为 float64 并检查固定
tail shape；未知 key、其他核心字段碰撞和错误 shape 在进入 buffer 前立即抛错。writer 在
buffer 首次接受 source layout 后，还会在每次写入前严格核对 key、shape 和 dtype，避免
静默 broadcast/cast；flush/finalize 再校验完整布局。reader 为兼容已经存在的历史扩展仍
允许未知 dataset，但要求其为逐帧数组且首维等于 `N`。

代码定位：`recording/io_process.py::_unpack_sample`、
`recording/episode_recorder.py::EpisodeRecorder.add_frame`、
`recording/episode_schema.py::normalize_diagnostics_v17`。

### 10.7 H5-07：`VALID` 现覆盖完整 93+1 布局

旧 `EpisodeReader.validity` 只硬编码 37 个核心 dataset；writer 发布前也只检查“已经存在”的
dataset 长度，而不检查缺项。

隔离复现构造了一个满足上述核心字段、相机侧车和 metadata 条件的 v17 episode，同时
省略 `arm_qpos`、`arm_qvel`、`flag_camera_fresh` 和 `flag_frame_status`，结果仍为
`VALID`。这主要影响手工构造、受损或未来发生 writer/reader
漂移的 episode；当前标准 RecorderIO 正常路径本身仍会创建 94 个 dataset。

必要性为 P1，因为所有训练/可视化/replay 都可能把 `VALID` 当信任边界。新
`recording.episode_schema` 定义 93 个基础 dataset 的确切 tail shape/dtype，以及
`arm_sent_stream=True ↔ action_arm_joint_sent 存在` 的条件规则。EpisodeRecorder 的 source
frame、flush、发布 finalizer 和 EpisodeReader 复用同一 validator。缺字段、错误
shape/dtype、marker 不一致或任意逐帧长度不等于 `N` 均会 fail-closed；不需要 schema bump，
因为这是恢复 v17 原有合同，而非改变存储意义。

代码定位：`recording/episode_reader.py::EpisodeReader.validity`、
`recording/episode_recorder.py::EpisodeRecorder._validate_and_sync_temp_episode`、
`recording/episode_schema.py::validate_data_layout_v17`。

---

<a id="sim-schema"></a>

## 11. DexMani Sim HDF5/Zarr 数据格式

本节是与前述 **DexMani Real schema v17 完全独立** 的 Sim 数据合同。为避免名称冲突，
本节沿用 `N` 表示单个 Sim episode 的动作数，使用 `T` 表示 Zarr 中拼接后的总帧数；
这里的 shape、单位、时序、压缩和 metadata 不能套用到第 1–10 节的 Real v17，反之亦然。

> **审计对象：** `/home/zhanghaoyang/Desktop/dexmani_sim`  
> **审计日期：** 2026-08-15  
> **结论来源：** 写入器、传感器生产链、转换器、回放器、可视化器的静态审计，以及
> 现存文件的只读元数据普查。  
> **执行边界：** 未启动 SAPIEN 场景、GPU 渲染、交互程序或机器人程序。

### 11.1 结论摘要

DexMani Sim 当前存在两层数据容器：

1. **HDF5 episode**：一个文件对应一个 episode，保存 13 个顶层逐帧 dataset 和 episode 级 HDF5 attributes。
2. **Zarr task dataset**：一个目录对应一个 task；默认只合并该 task 的成功 HDF5，将各 episode 沿第 0 维拼接，并用 `meta/episode_ends` 保存边界。

现存数据的实测结果为：

- 1113 个 HDF5 文件，合计 237777 帧；1000 个成功 episode、113 个失败 episode。
- 8 个 Zarr v2 store，各含 125 个成功 episode，合计 210083 帧。
- 1113 个 HDF5 的 13 个 dataset 键、单帧 shape、dtype 和 gzip 压缩级别完全一致。
- 所有 HDF5 内部 dataset 的首维长度一致，并与 `episode_steps` 一致。
- 8 个 Zarr 的 `episode_ends` 均与当前成功 HDF5 按文件名字典序累加得到的边界完全一致；每个 store 的首、末样本在全部 13 个 dataset 上与源 HDF5 精确相等。
- 当前实际图像分辨率是 **320×240（W×H）**，dataset shape 是 `(N,240,320,...)`。
  README 和 CLAUDE.md 中的 `480×640` 是过时说明。
- 当前全部 1113 个 HDF5 的 `action_space` 都是 `joint`；代码还支持 `ee`，但本次没有实际 `ee` 文件可用于样本级验证。

HDF5 和 Zarr 均没有显式的 schema 名称或 schema version。下文所称“标准格式”是当前生产代码和现存文件共同体现的事实，而不是由文件内版本标记保证的正式版本协议。

### 11.2 数据生成和转换链路

```text
env.reset(seed) -> 初始观测 s0
                    |
                    v
MimicGen joint path -> EnvRecorder -> episode_<seed>.h5
                                         |
                                         | 仅 succeeded_episode，按文件名字典序
                                         v
                              <task>.zarr/data/*
                              <task>.zarr/meta/episode_ends
```

关键实现：

| 环节 | 源文件 | 作用 |
|---|---|---|
| 环境观测 | `dexmani_sim/envs/base_env.py` | 组合相机、机器人、接触力和手部几何数据，完成点云与分割预处理 |
| RGB-D/分割/相机矩阵 | `dexmani_sim/sensors/camera.py` | 生成 `rgb`、`depth`、actor segmentation、相机内外参和原始世界点云 |
| 接触力 | `dexmani_sim/sensors/contact.py` | 将每个物理子步的接触 impulse 换算为 force，并对一个控制步求平均 |
| 理想手部点云 | `dexmani_sim/sensors/imagine_point_cloud.py` | 从 11 个手部 render mesh 采样固定 512 点并随 link pose 变换到世界系 |
| HDF5 写入 | `dexmani_sim/mimic_gen/utils/env_recorder.py` | 对齐观测、动作、done，写 dataset 和 attributes |
| episode 级扩展属性 | `dexmani_sim/mimic_gen/base_generator.py`、`dexmani_sim/gen_episode_data.py` | 写轨迹质量、抓取验证和场景随机化信息 |
| HDF5→Zarr | `dexmani_sim/data_process/hdf5_to_zarr.py` | 拼接顶层 dataset，写累计 episode 结束下标 |
| HDF5 派生工具 | `dexmani_sim/data_process/hdf5_custom_modify.py` | keep/delete/rename/merge/derive，可生成非标准 HDF5 |
| 回放 | `dexmani_sim/replay_episode.py` | 根据 `action_space` 选择 `action` 或 `action_ee` |
| 可视化 | `dexmani_sim/data_process/hdf5_visualize.py` | 读取标准 dataset 并在 Rerun 中显示 |

标准路径：

```text
mimic_data/episodes/<task>/succeeded_episode/episode_<seed:04d>.h5
mimic_data/episodes/<task>/failed_episode/episode_<seed:04d>.h5
mimic_data/dataset/<task>.zarr/
```

`<seed:04d>` 表示至少补足 4 位；超过 9999 的 seed 不会截断。

### 11.3 时间轴与行对齐语义

这是读取该格式时最重要的约定。

`BaseGenerator.reset()` 先保存初始观测 `s0`。随后每次 `env.step(action)` 返回动作执行后的下一状态。执行 N 个动作后，内存里有 N+1 个观测，但保存时丢弃最后一个观测，仅保留前 N 个。因此第 `t` 行的关系是：

```text
obs[t] = s_t
action[t] = a_t
done[t] = step(a_t) 返回的终止标志，即对应 s_(t+1)

s_t --a_t--> s_(t+1)
```

具体含义：

- `rgb/depth/.../joint_state` 的第 `t` 行是 **执行 `action[t]` 之前** 的观测。
- `done[t]` 是 **执行 `action[t]` 之后** 的结果，不与 `obs[t]` 处于同一个状态时刻。
- 最终 next observation `s_N` 不在文件中。
- 文件没有逐帧 timestamp。只能在固定步长假设下用 `time[t] = t * dt` 构造名义时间。
- `dt = frame_skip * physics_dt`。现存文件统一为 `frame_skip=15`、`physics_dt≈1/240 s`、`dt≈0.0625 s`，名义控制频率约 16 Hz。
- `obs_alignment="obs[t]_before_action[t]"` 只说明 observation/action 关系，没有单独说明 `done` 的 post-action 语义。

### 11.4 HDF5 容器结构（13 个 datasets）

#### 11.4.1 根结构

现存 1113 个文件的根层均只有 dataset，没有 group：

```text
/
├── action
├── action_ee
├── camera_extrinsic
├── camera_intrinsic
├── contact_force
├── depth
├── done
├── fingertip_points
├── imagine_point_cloud
├── joint_state
├── point_cloud
├── rgb
└── segmentation
```

所有 dataset：

- 使用 HDF5 gzip，`compression_opts=4`。
- 因启用压缩而采用 chunked layout；写入器未指定 chunks，实际 chunk shape 由 h5py 自动选择，会随 episode 长度变化，不能视为 schema 常量。
- 未启用 shuffle 或 Fletcher32 checksum。
- dataset 本身没有标准 attributes；episode 元数据位于文件根 attributes。

#### 11.4.2 Dataset 总表

下表中的 N 为本 episode 的动作数，也等于每个 dataset 的第 0 维和 `episode_steps`。

| Dataset | HDF5 shape | dtype | 单位 | 坐标系/范围 | 物理意义 |
|---|---:|---|---|---|---|
| `rgb` | `(N,240,320,3)` | `uint8` | 8-bit intensity | RGB，通常 `[0,255]` | SAPIEN `Color` render target 的前三通道，先乘 255、clip，再转 `uint8` |
| `depth` | `(N,240,320)` | `uint16` | mm | 相机光轴深度；0=无效/背景 | OpenGL camera-space `Position[...,2]` 取负后乘 1000；超出视锥或 actor id 为 0 的像素置 0 |
| `segmentation` | `(N,240,320)` | `uint8` | 无 | actor 级语义 id | 0=背景，1=桌面，2=机械臂，3=手；其余为按 scene actor id 偏移得到的物体 id |
| `point_cloud` | `(N,1024,6)` | `float32` | 前 3 列 m；后 3 列无量纲 | SAPIEN 世界系；RGB `[0,1]` | 相机可见前景点云，经世界系变换、工作空间 crop、2.5 mm voxel 聚合及 FPS/补采样 |
| `camera_intrinsic` | `(N,9)` | `float32` | 焦距/主点为 pixel | OpenCV intrinsic，row-major `3×3` | 每帧重复保存的相机 K 矩阵 |
| `camera_extrinsic` | `(N,12)` | `float32` | 平移为 m | OpenCV RDF camera；row-major `3×4` world→camera | `[R_cw \| t_cw]`；可视化器用 `R_wc=R_cw.T`、`t_wc=-R_wc@t_cw` 求 camera pose |
| `joint_state` | `(N,19)` | `float32` | rad | 自定义关节顺序 | 动作执行前的实际仿真 qpos：arm 7 + hand 12 |
| `contact_force` | `(N,15)` | `float32` | N | SAPIEN 世界系 xyz | 5 个手指 `link2` 的净接触力；每控制步对 15 个物理子步的 force 求平均 |
| `fingertip_points` | `(N,15)` | `float32` | m | SAPIEN 世界系 xyz | 5 个 fingertip link 原点的位置，flatten 为 5×3 |
| `imagine_point_cloud` | `(N,512,6)` | `float32` | 前 3 列 m；后 3 列无量纲 | SAPIEN 世界系；RGB `[0,1]` | 手部 render mesh 的完整“想象点云”；不受相机遮挡、视锥或工作空间 crop 限制 |
| `action` | `(N,19)` | `float32` | rad | 自定义关节顺序 | 关节目标：arm 7 + hand 12；是命令目标，不是反馈 qpos |
| `action_ee` | `(N,21)` | `float32` | pos 为 m；rot6d 无量纲；hand 为 rad | `custom_eef_link` 世界位姿 | `[eef_pos(3),eef_rot6d(6),hand_qpos(12)]` |
| `done` | `(N,)` | `bool` | 无 | `False/True` | `env.step(action[t])` 返回的 `success OR failed`；是 post-action transition 标签 |

SAPIEN 官方说明相机 `Position` texture 使用 OpenGL camera space（x 向右、y 向上、z 向后），model matrix 将其变换到世界系；官方 API 将 intrinsic/extrinsic 称为 OpenCV 格式。参见 [SAPIEN Camera 文档](https://sapien-sim.github.io/docs/user_guide/rendering/camera.html) 和 [CameraEntity API](https://sapien.ucsd.edu/docs/latest/apidoc/sapien.core.html)。

dataset 没有保存 unit、coordinate frame、列名或有效值规则等 attributes；上表语义来自生产代码，而不是文件内的机器可读描述。

坐标约定汇总：

- SAPIEN 世界使用机器人学右手系，可按 x 向前、y 向左、z 向上理解。
- 机器人 root 位于世界原点，并绕世界 z 轴 yaw `+π/6`；robot FK 输出再乘 root pose，所以 `action_ee`、fingertip 和两种点云都已在世界系，不是 robot-root local frame。
- 相机 `Position` texture 使用 OpenGL/Blender camera frame：x 向右、y 向上、z 向后；`depth=-z`，所以它是光轴深度而不是到相机中心的欧氏距离。
- `camera_extrinsic` 使用 OpenCV camera frame（RDF：x 向右、y 向下、z 向前），表示 world→camera。
- 相机可能在 reset 时按 episode 随机平移/重新朝向，但不会在标准 episode 内逐帧移动；K/extrinsic 仍被重复写入 N 行。

#### 11.4.3 关节和手指展开顺序

`joint_state` 与 `action` 的 19 维顺序完全相同：

| 索引 | 名称 | 部位 |
|---:|---|---|
| 0–6 | `joint1` … `joint7` | xArm7 |
| 7 | `right_hand_thumb_bend_joint` | thumb |
| 8 | `right_hand_thumb_rota_joint1` | thumb |
| 9 | `right_hand_thumb_rota_joint2` | thumb |
| 10 | `right_hand_index_bend_joint` | index |
| 11 | `right_hand_index_joint1` | index |
| 12 | `right_hand_index_joint2` | index |
| 13 | `right_hand_mid_joint1` | middle |
| 14 | `right_hand_mid_joint2` | middle |
| 15 | `right_hand_ring_joint1` | ring |
| 16 | `right_hand_ring_joint2` | ring |
| 17 | `right_hand_pinky_joint1` | pinky |
| 18 | `right_hand_pinky_joint2` | pinky |

`contact_force.reshape(5,3)` 和 `fingertip_points.reshape(5,3)` 的手指顺序均为：

```text
thumb, index, middle, ring, pinky
```

每个手指内部顺序为 `(x, y, z)`。接触力取各手指第二指节 `link2`，而 fingertip position 取各手指 tip link；二者不是同一个 link 原点。

#### 11.4.4 点云处理细节

##### `point_cloud`

原始输入来自相机每个有效前景像素：

```text
[x_world_m, y_world_m, z_world_m, r, g, b]
```

处理顺序：

1. 仅保留 `Position[...,3] < 1` 且 actor segmentation `> 0` 的像素。
2. 用相机 model matrix 从 OpenGL camera space 变换到 SAPIEN 世界系。
3. 进行闭区间 crop：`x∈[0.15,0.85] m`、`y∈[-0.5,0.5] m`、`z∈[table_surface_z+0.003,1.0] m`。
4. 用 `voxel_size=0.0025 m` 聚合，同 voxel 的 xyz 和 RGB 都取平均。
5. 若点数不少于 1024，对 xyz 做 farthest-point sampling，并携带对应 RGB。
6. 若点数不足 1024，用带 replacement 的随机索引复制已有点；随机源由 episode seed 设置。
7. 如果 crop 后没有任何点，整帧输出 1024×6 的 0。

代码注释写着“5 mm”，但实际调用值是 **2.5 mm**；格式解释应以执行代码为准。

##### `imagine_point_cloud`

它不是由 depth 反投影得到，而是对完整手部 render mesh 做表面采样：

- `right_hand_link`：192 点。
- 5 个手指的 `link1` 和 `link2`：10×32 点。
- 总数：`192 + 10×32 = 512`。
- 采样点保存在 link local frame，读取时随每个 link 的世界 pose 变换。
- RGB 来自 render mesh vertex/material colors。
- 该点云不包含被操作物体，也不表达可见性。

#### 11.4.5 `action` 与 `action_ee`

`action_ee` 的旋转 6D 表达不是 Euler angle，也不是 rotation vector：

```text
rot6d = concat(R[:, 0], R[:, 1])
```

即旋转矩阵前两列，列优先拼接为 6 个数。解码时使用 Gram–Schmidt 正交化并以叉乘恢复第三列。

两种 `action_space` 下文件仍同时保存两个动作 dataset：

| `action_space` | 实际送入 `env.step` | `action` | `action_ee` |
|---|---|---|---|
| `joint` | 19D joint target | 原始 joint target | 对该 joint target 做 FK 得到的 EEF+hand 表达 |
| `ee` | 21D EEF+hand target | normal motion 中为生成该目标所依据的 joint path | 从 joint path 计算并实际送入环境的 EEF+hand target |

注意：

- `action_dim` 始终取 `action.shape[1]`，因此即使 `action_space="ee"`，其值仍为 19，而不是 21。
- `action` 是 recorder 收到的目标值。`robot.apply_action()` 在真正下发 PD target 前还会按 joint limits clip，因此极端情况下文件中的 `action` 可能与最终应用值不同。
- EE `hold_phase` 在执行 EEF action **之后**重新调用 IK 来填充同一行 `action`，所以 hold 行的 19D 值不保证等于环境在该步执行前求得的 joint target。
- 现存 1113 个文件全部是 `action_space="joint"`；`ee` 行为是代码契约，未由现存样本覆盖。

#### 11.4.6 标准文件没有记录的信息

标准 HDF5 不包含：

- 逐帧 timestamp 或 wall-clock time。
- reward。
- 独立的 per-step `success`、`failed`、`truncated`、`success_condition` 或完整 `info`；`done` 不能区分成功与失败。
- terminal next observation `s_N`。
- joint velocity、acceleration、effort/torque 或实际 applied/clipped command。
- EEF feedback pose；`action_ee` 是由命令 joint target 做 FK 得到的动作表示。
- 被操作物体逐帧 pose、velocity、contact graph 或 segmentation label map。
- 相机 near/far、FOV、畸变系数或颜色空间标识；当前代码常量为 near=0.1 m、far=5.0 m、vertical FOV=58°，但文件只保存 K/extrinsic。
- 点云每点的 segmentation/object id、normal 或 visibility confidence。
- schema version、producer commit、依赖版本、生成时间或内容 checksum。

视频不嵌入 HDF5。启用 `save_video()` 时，它另存为：

```text
mimic_data/episodes/<task>/videos/episode_<seed:04d>_<succeeded|failed>.mp4
```

### 11.5 HDF5 根 Attributes 数据字典（46 个实测键）

#### 11.5.1 固定基础属性

| Attribute | 实测 dtype | 单位/格式 | 物理或逻辑意义 |
|---|---|---|---|
| `task_name` | UTF-8 string | snake_case | task 名称 |
| `seed` | `int64` | 无 | episode seed；与文件名编号一致 |
| `success` | `int64` | 0/1 | 最终用于分类和目录选择的 `wrapper.task_done`；可能受轨迹质量拒绝影响 |
| `episode_steps` | `int64` | frame | N，即动作数和每个 dataset 的首维 |
| `truncated` | `int64` | 0/1 | recorder 最后一次在 normal interaction phase 看到的时间上限标志 |
| `dt` | `float64` | s | 一个控制步的仿真时长，`frame_skip * physics_dt` |
| `physics_dt` | `float64` | s | 单个 PhysX step 时长 |
| `frame_skip` | `int64` | physics step/control step | 每个 action 执行的物理子步数 |
| `action_dim` | `int64` | dimension | `action` 的末维；当前恒为 19 |
| `action_space` | UTF-8 string | `joint`/`ee` | 环境实际采用的控制空间；回放器据此选动作源 |
| `obs_alignment` | UTF-8 string | 固定文本 | 当前为 `obs[t]_before_action[t]` |
| `success_once` | `int64` | 0/1 | recorder 是否曾观察到稳定成功 `info["success"]` |
| `success_at_end` | `int64` | 0/1 | 保存时 `env.success_condition` 的瞬时值，不等同于稳定保持后的 done |
| `success_frame_idx` | `int64` | action index | 首次观察到稳定成功的 action/done 行；从 0 开始，未成功为 -1 |
| `failed` | `int64` | 0/1 | finalize 时是否命中 task 的显式失败条件 |
| `failed_reason` | UTF-8 string | 文本 | 设计上用于失败原因；现存 1113 个文件全部为空字符串 |

这些逻辑量为了兼容写入都没有统一使用 HDF5 bool；除 `traj_quality_is_low_quality` 和 dataset `done` 外，多数布尔语义属性实际存为 `int64` 0/1。

#### 11.5.2 抓取、fallback 与轨迹质量属性

以下属性由 MimicGen 生成器通过 `extra_attrs` 写入。现存文件除 `fallback_reason` 外全部存在。

| Attribute | dtype | 单位/格式 | 定义 |
|---|---|---|---|
| `fallback_count` | `int64` | count | 规划完全失败后使用 2 帧 hold fallback 的次数 |
| `fallback_reason` | UTF-8 string | 文本，可缺省 | 最后一次 fallback 的原因；仅当存在原因时写入，不是全部原因列表 |
| `grasp_verified` | `int64` | 0/1 | 所有抓取验证是否都通过；没有验证项时按真处理 |
| `grasp_verified_list` | `int64[G]` | 每项 0/1 | 每次 grasp 的验证结果，按发生顺序；现存 shape 为 `(1,)` 或 `(2,)` |
| `traj_quality_is_low_quality` | HDF5 `bool` | True/False | 任一 primitive 被判为低质量 |
| `traj_quality_primitive_count` | `int64` | count | 参与质量日志的 primitive 数量 |
| `traj_quality_low_quality_count` | `int64` | count | 低质量 primitive 数量 |
| `traj_quality_min_efficiency` | `float64` | ratio | 有至少 4 个 waypoint 的 primitive 中，`straight_line/path_length` 的最小值 |
| `traj_quality_mean_efficiency` | `float64` | ratio | 上述效率的均值 |
| `traj_quality_total_large_turns` | `int64` | count | 全 episode 大转角数量之和；大转角阈值为 30° |
| `traj_quality_max_turn_deg` | `float64` | degree | 所有 primitive 的最大相邻路径方向转角 |

质量计算只使用 arm path 的 EEF 世界位置；至少 5 个 waypoint 时先做一次 3 点均值平滑。长度小于等于 1 mm 的 segment 不参与方向转角。低质量判定为：

```text
efficiency < 0.7
OR num_large_turns > 3
OR max_turn_deg > 60°
```

当直线位移小于 0.03 m 时，不使用 `max_turn_deg > 60°` 这一项。某些 task 可关闭 `enforce_path_quality`，因此 `traj_quality_is_low_quality=True` 并不必然意味着 `success=0`。现存数据中有 111 个成功 episode 同时标记为低质量。

#### 11.5.3 场景属性

| Attribute | dtype | 条件 | 定义 |
|---|---|---|---|
| `scene_info_json` | UTF-8 JSON string | `scene_info` 非空；当前始终存在 | 场景随机化的主要机器可读记录 |
| `scene_seed` | UTF-8 string | 当前始终存在 | `scene_info.seed` 的字符串副本；数值语义应优先用根 `seed` |
| `scene_table_height_offset` | UTF-8 string | 启用桌高随机化 | 桌高偏移的字符串副本，单位 m |
| `scene_skybox` | UTF-8 string | 启用 per-episode skybox | skybox 文件名或 `clean`；当前样本未出现 |
| `scene_objects_<key>` | UTF-8 string | 每个 manipulated/goal object | 对象信息 dict 的 Python `str()`；不是严格 JSON |

`scene_info_json` 的结构为：

```json
{
  "seed": 0,
  "table_height_offset": 0.009108850619643628,
  "skybox": "optional.exr",
  "objects": {
    "object_key": {
      "model_id": 3,
      "user_scale": 1.25
    }
  }
}
```

没有 `model_name` 的原生 SAPIEN entity 使用 `{"type": "ClassName"}`。`skybox` 和 `table_height_offset` 仅在对应随机化开关启用时存在。场景 JSON 不保存每个物体的初始 pose、质量、摩擦、灯光参数、材质参数、相机随机偏移或 clutter 的完整配置。

现存文件观察到的动态对象 attribute key 为：

| Task | `scene_objects_*` key（并集） |
|---|---|
| `multi_grasp` | `scene_objects_027_table-tennis`, `scene_objects_111_callbell` |
| `open_box` | `scene_objects_flip_box` |
| `peg_insertion` | `scene_objects_peg` |
| `pick_apple_messy` | `scene_objects_003_plate`, `scene_objects_035_apple` |
| `pick_bottle` | `scene_objects_001_bottle` |
| `place_milk_box` | `scene_objects_038_milk-box`, `scene_objects_074_displaystand` |
| `pour` | `scene_objects_002_bowl`, `scene_objects_019_coaster`, `scene_objects_021_cup` |
| `stack_cups` | `scene_objects_021_cup_0`, `scene_objects_021_cup_1`, `scene_objects_021_cup_3`, `scene_objects_021_cup_5` 的 episode 子集 |

读取场景信息时应解析 `scene_info_json`，不应解析 `scene_objects_*` 的 Python repr 字符串。

#### 11.5.4 `extra_attrs` 扩展边界

`EnvRecorder.save_episode(extra_attrs=...)` 会在固定属性之后执行：

```python
for k, v in extra_attrs.items():
    f.attrs[k] = v
```

因此它可以增加任意 HDF5 attribute，也可以覆盖同名基础属性。当前批量生产链使用的是上述质量与场景字段，但 writer 本身没有 whitelist 或冲突检查。第三方文件可能拥有更多属性或被覆盖的语义，不能仅凭扩展属性名认定为标准格式。

### 11.6 success、done、failed、truncated 的关系

这些字段不是同义词：

| 字段 | 粒度 | 来源 | 是否可能保持/过滤 |
|---|---|---|---|
| `success_condition` | 未直接保存 | task 每步计算的瞬时条件 | 可在下一步消失 |
| `done[t]` | 每 transition | 稳定成功或显式失败 | 成功在 env 内 latch 后通常持续为 True |
| `success_once` | episode | recorder 是否曾见过 `info.success` | latch |
| `success_at_end` | episode | 保存时瞬时 `env.success_condition` | 非 latch |
| `success` | episode | 最终 `task_done` | 可因 truncation、显式失败或低质量策略变为 0 |
| `failed` | episode | finalize 时 task 显式失败条件 | 与普通“未成功”不同 |
| `truncated` | episode | normal interaction phase 的 action limit | 与 success/failure 独立 |

目录与 `success` 在现存 1113 个文件中完全一致：`succeeded_episode` 都是 1，`failed_episode` 都是 0。但不要用 `failed=0` 推断成功；许多 episode 只是未达成功条件或被质量规则拒绝。

`hold_phase()` 不更新 `task_truncated`，也不因 `env.step()` 返回 truncated 而中止。因此成功 episode 的 `episode_steps` 可以略大于 task 的 nominal action limit，而 `truncated` 仍为 0。

### 11.7 Zarr 格式

#### 11.7.1 目录结构与版本

现存 store 均为 Zarr format 2：

```text
<task>.zarr/
├── .zgroup                  # {"zarr_format": 2}
├── data/
│   ├── action/
│   ├── action_ee/
│   ├── camera_extrinsic/
│   ├── camera_intrinsic/
│   ├── contact_force/
│   ├── depth/
│   ├── done/
│   ├── fingertip_points/
│   ├── imagine_point_cloud/
│   ├── joint_state/
│   ├── point_cloud/
│   ├── rgb/
│   └── segmentation/
└── meta/
    └── episode_ends/
```

根 group、`data` group、`meta` group 和全部 array 的 Zarr attrs 当前均为空。

#### 11.7.2 Array schema

设 E 为 episode 数、`N_i` 为第 i 个 episode 的长度、`T=sum(N_i)`。`data/*` 与 HDF5 dataset 数值和 dtype 相同，仅将首维从 episode 内的 N 改为全 task 的 T：

| Zarr array | Shape | dtype | Chunks |
|---|---:|---|---:|
| `data/rgb` | `(T,240,320,3)` | `uint8` | `(100,240,320,3)` |
| `data/depth` | `(T,240,320)` | `uint16` | `(100,240,320)` |
| `data/segmentation` | `(T,240,320)` | `uint8` | `(100,240,320)` |
| `data/point_cloud` | `(T,1024,6)` | `float32` | `(100,1024,6)` |
| `data/camera_intrinsic` | `(T,9)` | `float32` | `(100,9)` |
| `data/camera_extrinsic` | `(T,12)` | `float32` | `(100,12)` |
| `data/joint_state` | `(T,19)` | `float32` | `(100,19)` |
| `data/contact_force` | `(T,15)` | `float32` | `(100,15)` |
| `data/fingertip_points` | `(T,15)` | `float32` | `(100,15)` |
| `data/imagine_point_cloud` | `(T,512,6)` | `float32` | `(100,512,6)` |
| `data/action` | `(T,19)` | `float32` | `(100,19)` |
| `data/action_ee` | `(T,21)` | `float32` | `(100,21)` |
| `data/done` | `(T,)` | `bool` | `(100,)` |
| `meta/episode_ends` | `(E,)` | `int64` | 现存为 `(125,)` |

所有 array 使用：

- numcodecs Zstd codec：`id="zstd"`, `level=3`, `checksum=False`。
- C order。
- `filters=None`。
- 数值 fill value 为 0，bool fill value 为 False。

`chunk_size=100` 是第 0 维 chunk 长度；converter 把完整 sample shape 放入同一个 chunk。因此一个 RGB chunk 的未压缩大小约为 `100×240×320×3 ≈ 23.0 MB`，读取单帧时仍可能解压整个 100 帧 chunk。

#### 11.7.3 Episode 边界定义

`episode_ends` 保存的是 **exclusive cumulative end index**：

```text
episode_ends[i] = N_0 + N_1 + ... + N_i
start(i) = 0                       if i == 0
           episode_ends[i - 1]     otherwise
end(i) = episode_ends[i]

episode_i = data[key][start(i):end(i)]
```

最后一个值必须等于所有 `data/*` 的首维 T。当前 8 个 store 均严格递增且满足该不变量。

#### 11.7.4 转换规则

标准 `main(task_name)`：

1. 只读取 `mimic_data/episodes/<task>/succeeded_episode`。
2. 接受 `.h5` 和 `.hdf5` 后缀，按完整路径字符串字典序排序。
3. 从第一个 HDF5 的**顶层 dataset**推导 key 集合、单帧 shape 和 dtype；key 再按字典序排序。
4. 以 `mode="w"` 新建/覆盖目标 Zarr。
5. 预检查其余文件中这些 key 的 dtype 是否与第一个文件一致。
6. 对每个文件检查所选 dataset 的首维长度是否一致。
7. 不做数值 cast，直接 append 全量数组。
8. 每个文件 append 后累计长度，最终写 `meta/episode_ends`。

转换器不写入 HDF5 root attributes，也不保存源文件名、seed、task name、success、action space、dt、场景 JSON或质量指标。Zarr 中的 episode i 只能对应“转换时排序后的第 i 个源文件”；该映射没有在 store 内持久化。

#### 11.7.5 现存 Zarr 实测规模

| Task store | E | T | 第一个 episode end | 最终 end | 与当前成功 HDF5 边界一致 |
|---|---:|---:|---:|---:|---|
| `multi_grasp.zarr` | 125 | 23932 | 196 | 23932 | 是 |
| `open_box.zarr` | 125 | 27353 | 228 | 27353 | 是 |
| `peg_insertion.zarr` | 125 | 29326 | 250 | 29326 | 是 |
| `pick_apple_messy.zarr` | 125 | 21527 | 189 | 21527 | 是 |
| `pick_bottle.zarr` | 125 | 13542 | 114 | 13542 | 是 |
| `place_milk_box.zarr` | 125 | 29332 | 207 | 29332 | 是 |
| `pour.zarr` | 125 | 33365 | 246 | 33365 | 是 |
| `stack_cups.zarr` | 125 | 31706 | 324 | 31706 | 是 |

### 11.8 现存 HDF5 数据规模

| Task | Split | Episodes | Frames | N min | N max | N mean |
|---|---|---:|---:|---:|---:|---:|
| `multi_grasp` | failed | 2 | 452 | 205 | 247 | 226.00 |
| `multi_grasp` | succeeded | 125 | 23932 | 177 | 242 | 191.46 |
| `open_box` | failed | 2 | 480 | 240 | 240 | 240.00 |
| `open_box` | succeeded | 125 | 27353 | 178 | 255 | 218.82 |
| `peg_insertion` | failed | 44 | 12471 | 225 | 328 | 283.43 |
| `peg_insertion` | succeeded | 125 | 29326 | 207 | 275 | 234.61 |
| `pick_apple_messy` | failed | 18 | 3170 | 158 | 215 | 176.11 |
| `pick_apple_messy` | succeeded | 125 | 21527 | 152 | 209 | 172.22 |
| `pick_bottle` | failed | 3 | 322 | 100 | 122 | 107.33 |
| `pick_bottle` | succeeded | 125 | 13542 | 93 | 132 | 108.34 |
| `place_milk_box` | failed | 17 | 3916 | 172 | 306 | 230.35 |
| `place_milk_box` | succeeded | 125 | 29332 | 191 | 304 | 234.66 |
| `pour` | failed | 1 | 287 | 287 | 287 | 287.00 |
| `pour` | succeeded | 125 | 33365 | 240 | 296 | 266.92 |
| `stack_cups` | failed | 26 | 6596 | 214 | 320 | 253.69 |
| `stack_cups` | succeeded | 125 | 31706 | 205 | 324 | 253.65 |
| **合计** | — | **1113** | **237777** | — | — | — |

属性普查补充结果：

- `success=1`：1000；`success=0`：113。
- `success_once=1`：1008；其中 8 个最终 `success=0`。
- `traj_quality_is_low_quality=True`：138；其中 111 个仍是成功 episode。
- `failed=1`：12，但 1113 个 `failed_reason` 全为空。
- `truncated=1`：5。
- `fallback_count>0`：77；这 77 个均具有 `fallback_reason="AtomicMotion planning failed."`。

### 11.9 非标准派生 HDF5

`HDF5Pipeline` 不是标准 episode writer，而是通用转换工具。它可以：

- `KeepOnly`：只保留指定顶层 key。
- `Delete`：删除 key。
- `Rename`：重命名 key。
- `Merge`：拼接多个数组得到新 dataset。
- `Derive`：逐样本或整数组计算派生 dataset，并可改变 dtype/压缩。

仓库示例会生成仅含 `rgb`、`point_cloud`、重命名后的 `state`、带 normal 的点云和 DBSCAN labels 的 HDF5。这类文件不满足前述 13-key 标准 schema。

更重要的是，当前 `_process_one_file()` 没有复制 HDF5 文件根 attributes；dataset copy helper 也没有复制 dataset attributes。因此经该 pipeline 处理后，`task_name`、seed、时序、success、action space、场景 provenance 等 episode 元数据会全部丢失。

Zarr converter 是 data-driven 的：如果输入派生目录，第一个文件的顶层 dataset 会成为该 Zarr 的 schema。因此“Zarr 固定有 13 个 data array”只对当前标准输入和现存 8 个 store 成立，不是 converter 的硬编码保证。

### 11.10 已定位问题与解释边界

> **严重度统计：** 高 6 项、中 8 项、低 3 项，共 17 项。下表给出完整索引；后续小节
> 展开最容易导致错误读取、数据丢失或训练泄漏的 9 项。

| ID | 级别 | 类型 | 定位 | Fact-check 结论 |
|---|---|---|---|---|
| SIM-H5-01 | 高 | 文档错误 | README Observation Space、CLAUDE.md Obs keys | 文档写 480×640；`Camera` 构造为 320×240，1113 个 HDF5 全为 240×320 |
| SIM-H5-02 | 高 | schema 治理 | HDF5 writer、Zarr converter | 两种容器都没有 schema name/version；兼容性只能由 key/shape/dtype 猜测 |
| SIM-H5-03 | 高 | provenance 丢失 | `hdf5_to_zarr.py` | Zarr 不保存任何 HDF5 attrs、文件名、seed 或 episode id，只有边界 |
| SIM-H5-04 | 高 | 转换覆盖风险 | `zarr.open(..., mode="w")` | 转换开始即覆盖目标 store；没有临时目录、完成标记或原子 publish |
| SIM-H5-05 | 中 | 弱验证 | `HDF5ToZarrConverter` | key/shape 取第一个文件；仅预检 dtype，未显式比较 trailing shape、属性或语义；额外 key 被忽略 |
| SIM-H5-06 | 高 | 元数据丢失 | `HDF5Pipeline._process_one_file` | 自定义 HDF5 修改不复制 root attrs，派生文件失去 episode provenance |
| SIM-H5-07 | 中 | 标签不可自描述 | segmentation producer | 文件不保存 class-id→object 的映射；0–3 可解释，其余 id 无法仅靠 HDF5 稳健还原对象身份 |
| SIM-H5-08 | 中 | 失败诊断无效 | `EnvRecorder._failed_reason` | 字段只初始化/清空/写入，从未赋具体原因；现存 1113 个值全空 |
| SIM-H5-09 | 中 | 时序易误读 | recorder trimming | `obs[t]` pre-action、`done[t]` post-action，且 terminal next obs 丢弃；仅 `obs_alignment` 部分记录该关系 |
| SIM-H5-10 | 中 | 动作真实性边界 | `robot.apply_action` 与 recorder | recorder 保存 clip 前 target，文件没有 applied/clipped action 或 clip flag |
| SIM-H5-11 | 中 | EE 元数据歧义 | `action_dim` writer | `action_space=ee` 时实际控制向量 21D，但 `action_dim` 仍写 19；现存数据未覆盖该路径 |
| SIM-H5-12 | 低 | 存储效率 | 相机矩阵与 Zarr chunks | 每帧重复 K/extrinsic；RGB Zarr chunk 未压缩约 23 MB，随机单帧访问代价较大 |
| SIM-H5-13 | 中 | 完整性 | HDF5/Zarr writer | 没有 schema validator、checksum、源清单、转换 manifest 或 end-to-end 完成标志 |
| SIM-H5-14 | 低 | 代码/注释偏差 | `BaseEnv.preprocess_obs_data` | 注释称点云 voxel 为 5 mm，实际参数为 2.5 mm |
| SIM-H5-15 | 低 | legacy 边界 | replay `_try_settle` | reader 支持可选 `done_action` attribute，但当前 writer 从不写；1113 个文件均不存在 |
| SIM-H5-16 | 高 | 发布/覆盖风险 | `EnvRecorder.save_episode` | 直接以 `"w"` 打开最终 `.h5`；同 seed 会覆盖，异常可留下对外可见的半文件，没有临时文件和原子 publish |
| SIM-H5-17 | 中 | 未强制的不变量 | `record_initial_obs`、`obs_alignment` | pre-action 对齐依赖调用者先记录初始观测；writer 不校验，却始终写固定 alignment 文本 |

#### 11.10.1 SIM-H5-01：分辨率说明已过时

`Camera.__init__()` 明确使用 `width=320, height=240`。实测所有 `rgb` 为 `(N,240,320,3)`，`depth/segmentation` 为 `(N,240,320)`。任何根据 README 预分配 480×640 tensor 的 reader 都会失败。

#### 11.10.2 SIM-H5-03：Zarr 无法独立恢复 episode provenance

`meta/episode_ends` 只给边界，不给 episode 名称。即使当前可通过重新扫描源 HDF5 并重复相同排序来恢复 seed，这仍依赖外部目录保持不变。复制、过滤、增删或重命名 HDF5 后，Zarr 本身无法证明每段对应哪个 episode。

#### 11.10.3 SIM-H5-05：converter 的 schema 来自第一个文件

转换器只遍历第一个文件的顶层 dataset key：

- 后续文件缺 key 时会在访问阶段报错。
- 后续文件多出的 key 静默不进入 Zarr。
- dtype 会显式比较。
- trailing shape 不显式比较，通常在 Zarr append 时才因 shape 不兼容失败。
- 所有 HDF5 attrs 都不参与一致性验证。

因此转换成功只能说明被选择的数组可拼接，不能说明 episode 的 task、单位、action space 或时序语义一致。

#### 11.10.4 SIM-H5-07：分割标签缺少对象字典

预处理把背景/桌面/臂/手固定映射为 0/1/2/3，其他 actor id 通过 scene 内 id 偏移到 4 以上。但文件没有保存每个最终 label 对应的 entity 名称，也没有保证 scene entity 创建顺序跨代码版本不变。`scene_info_json.objects` 记录模型信息，但没有记录 segmentation label，二者无法可靠 join。

#### 11.10.5 SIM-H5-08：`failed_reason` 当前没有生产者

全仓库检索显示 `_failed_reason` 只在 recorder 初始化、reset 和保存时出现，没有赋值路径。属性存在并不代表诊断链已经实现。`failed=1` 的 12 个现存 episode 也全部是空原因。

#### 11.10.6 SIM-H5-09：训练窗口的正确切分

不得跨越 `episode_ends` 建立 sequence window。对于 observation-action pair，可使用同一行 `(obs[t], action[t])`；若需要监督 next state，应使用同 episode 的 `obs[t+1]`，但最后一个 action 的 next observation 不存在，应丢弃该 transition 或改变采集格式。

#### 11.10.7 SIM-H5-11：EE 路径仅完成代码级核实

代码允许 `action_space="ee"`，回放也会读取 `action_ee`，但当前 1113 个文件全部是 joint。文档对 EE 的 shape 和转换逻辑来自静态生产/消费链，不代表已经通过实际 episode round-trip 验证。尤其是 hold phase 的 `action` 在 step 后重新做 IK，不能当作该 transition 的精确 applied joint command。

#### 11.10.8 SIM-H5-16：HDF5 直接写最终路径

writer 使用 `h5py.File(episode_path, "w")`。这同时带来两个事实：已有同名 seed 文件会被截断覆盖；进程在 dataset/attribute 尚未写完时退出，目录中可能留下看似正式命名的半成品。当前 reader/converter 没有完成标记来排除它。

#### 11.10.9 SIM-H5-17：对齐语义由调用顺序保证

标准 `BaseGenerator.reset()` 正确调用 `record_initial_obs()`，因此现有生产链满足 pre-action 对齐。但 `EnvRecorder` 本身没有检查这个前置条件；若其他调用者直接 `interact_with_env()`，保存的观测将是 post-action，而 `obs_alignment` 仍会硬编码为 `obs[t]_before_action[t]`。

### 11.11 推荐的读取不变量

读取标准 HDF5 时至少检查：

1. 13 个必需 dataset 是否齐全。
2. 所有 dataset 首维是否相同并等于 `episode_steps`。
3. 单帧 shape 和 dtype 是否符合第 11.4.2 节。
4. `action_space` 是否为受支持值；`joint` 用 `action`，`ee` 用 `action_ee`。
5. `frame_skip * physics_dt` 是否与 `dt` 在容差内一致。
6. `success_frame_idx` 是否为 -1 或位于 `[0,N)`。
7. `success_once=0` 时 `success_frame_idx` 应为 -1。
8. 解析 `scene_info_json` 时处理字段缺省，不依赖 `scene_*` 字符串副本。
9. 将 `depth==0` 当作 invalid，不当作相机原点表面。
10. 不把 `action` 当实际反馈，不把 `done[t]` 当 `obs[t]` 的状态标签。

读取 Zarr 时额外检查：

1. `episode_ends` 为 `int64` 一维、非空、严格递增。
2. 所有 `data/*` 首维相同，且等于 `episode_ends[-1]`。
3. sequence sampler 不能跨 episode boundary。
4. provenance、dt、task、action space 和成功状态必须由外部 manifest 补充；当前 store 内无法恢复。
5. 不假设任意 Zarr 都有固定 13 key；先检查实际 array 列表和 schema。

### 11.12 本次验证范围

执行了以下只读验证：

- 静态追踪 HDF5 writer、全部数据生产者、Zarr converter、replay、visualizer 和 HDF5 派生 pipeline。
- 用 `/home/zhanghaoyang/miniconda3/envs/sim/bin/python`，在 `h5py 3.16.0`、`zarr 2.18.3`、`numpy 1.26.4` 下读取元数据。
- 遍历全部 1113 个 HDF5 的 key、shape、dtype、compression、首维一致性、attributes 类型与存在性。
- 核对 seed/文件名、success/目录、`dt=frame_skip*physics_dt`、success index 范围及对应 `done`，未发现不一致；1113 个文件的 dataset attributes 也全部为空。
- 遍历全部 8 个 Zarr 的 group/array、shape、dtype、chunk、codec、attrs 和 episode boundaries。
- 对每个 Zarr 核对当前成功 HDF5 的累计边界，并逐 task 比较首、末样本的全部 13 个 dataset，结果精确相等。

未执行：

- 未启动 SAPIEN 或创建环境。
- 未运行 MimicGen、episode generation、replay 或 Rerun GUI。
- 未对所有大体量 RGB/point-cloud 数值做全量 checksum；本次验证覆盖完整 metadata、完整边界和每个 task 的首末样本。
- 未生成或覆盖任何 `dexmani_sim` 数据文件。
