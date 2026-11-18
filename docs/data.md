# 实验数据

## Raw Episode

遥操作默认写入当前目录的 `episodes/<task_name>/`。每个已发布的 Episode 可独立读取和导出，包含 `data.h5`、有损 H.264 图像 `rgb.mp4` 和可选的 [HOI 标注](object_reconstruction.md#标注与查看) `hoi.h5`。用 `python scripts/visualize_episode.py EPISODE --info` 查看结构和录制截断原因。

`data.h5` 的 `meta` 顶层属性保存该段的相机内参、畸变、深度单位和已有外参。`effective_parameters` 是 JSON，保存实际使用的硬件、控制和算法参数，包括 `camera_config`、适用的 VR 标定和 `tactile_reset`。

Raw 按字段使用固定类型。关节、绝对关节目标、VR 腕部位姿和手部关键点使用 `float64`，触觉与 HOI 点云使用 `float32`，深度使用 `uint16`，VR 接收时间和序列号使用 `uint64`。提交状态使用 `int8`，`timing/segment_starts` 使用 `int64`，`timing/rows` 使用 `TIMING_ROW_DTYPE`。

## 时间与动作

`timing/rows` 保留控制采样时刻、观测完成时刻及各来源的主机单调时钟时间，单位为纳秒。相机派生观测沿用相机来源时间，重复读取保留原时间。多来源观测按这些时间做软件对齐，时间戳记录的是主机侧查询或接收时刻。

Raw 动作列保存尝试提交的绝对关节目标，缺失目标记为 NaN。完整命令在主机侧提交成功时，`action_send_accepted` 为 `1`，失败或未提交时为 `0`。实测关节位置另有反馈列。

`timing/segment_starts` 记录连续段首行。跳槽或 HOI HOLD 产生录制间断时，下一个接纳的行开始新段。队列满或录制来源失效则停止接纳并记录 `recording_truncated_reason`。Replay 只执行选定连续段的关节目标，运行前需将机器人就位到该段起点。

## 图像与模型输入

Raw 保留采集像素网格上的 RGB-D，解码后的有损 RGB 与现场像素值可能有差异。点云取色使用这一路 RGB-D。在线策略 RGB、Canonical RGB/depth 和 Raw 可视化使用同一针孔图像几何，沿用固定 K、原分辨率和光轴。RGB 双线性采样，depth 最近邻采样，越界填零。含图像的 Zarr 标记 `image_geometry="pinhole"`。

读取深度时要求 Raw 记录对齐到彩色网格的深度及其正尺度。读取 RGB 时检查像素尺寸和纳秒时基，逐帧核对时间戳是否等于该行 `control_ns` 减去首行 `control_ns`。完整读取时还检查 `rgb.mp4` 帧数与 `timing/rows` 一致。限帧可视化只检查所选前缀。

旧恒帧率录制的 RGB 会被拒绝。符合当前 Raw 数据结构的旧录制仍可读取关节和深度等独立字段。

模型所需模态经投影进入策略历史。Raw 录制使用本次采样的实测关节、触觉和原始 RGB-D。启用录制时，普通 RGB 策略与 Raw 共用相机源帧。HOI 或场景点云策略的 RGB 跟随对应的派生观测，Raw 从相机通道单独取帧。录制所需的传感器来源也成为本段依赖。

## 按需导出

- 每次 `export_zarr.py --fields ...` 发布一份固定字段的 Zarr。纯关节导出跳过相机，纯 RGB 跳过深度单位，选择 `object_pointcloud` 时才读取 HOI。用 `--output` 指定输出路径，否则默认写入当前目录的 `datasets/<输入目录名>.zarr`，单个 Episode 使用其父目录名。新产物完整后原子发布，路径须尚不存在。
- 同一产物的 `task_name` 取自各 Raw 一致的 `meta.task_label`。`action_space="joint"` 标记采集动作空间，训练用 `action_key` 选择 `action` 或经 FK 派生的 `action_ee`。
- 每个原生连续段成为一个 Zarr episode，`source_episodes` 和 `source_row_ranges=[start, stop)` 与 `episode_ends` 逐段对应。位置对齐和采样窗口限于段内。同一采集的各段属于同一个训练或验证分组，`val_episodes` 和 `max_train_episodes` 按采集组计数。
- 所选数据缺失、损坏或所需的 HOI 关节对齐无法覆盖整段时，整条 Raw 回滚并继续处理其他采集。完整有效的截断前缀仍可导出。没有可用采集时导出失败。
- 根属性 `rejected_episodes` 保存被拒的入口名和原因，`excluded_episodes` 保存显式排除的入口名。排除后仍有两个入口指向同一目录时会报错。

配套 Policy 的 Real 原生 ODE 读取 `action_ode`，使用 `export_zarr.py --fields joint_state point_cloud action_ode` 导出。导出器按逐行提交状态选择动作，失败行沿用上一条已接受目标并保留原时间轴。段首在首条成功命令前使用该段起始关节状态。

Zarr 的 `data_processing` 保存实际使用的处理参数。开启去桌面时，归一化系数写入 `data_processing.point_cloud.table_plane_abcd`。部署对应值见 `run_config.yaml` 的 `actual_processing.point_cloud.table_plane_abcd`。接触预训练按同一帧的手部和物体点云计算 1024 个有序标签。见[标注与查看](object_reconstruction.md#标注与查看)。
