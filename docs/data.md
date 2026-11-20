# 实验数据

## Raw Episode

遥操作默认写入当前目录的 `episodes/<task_name>/`。每个已发布的 Episode 可独立读取和导出，包含 `data.h5`、有损 H.264 图像 `rgb.mp4` 和可选的 [HOI 标注](object_reconstruction.md#标注与查看) `hoi.h5`。用 `python scripts/visualize_episode.py EPISODE --info` 查看结构和录制截断原因。

Raw 读取要求有效的 `meta.termination_reason`，已发布的有效截断前缀可读。读取动作要求 `meta.arm_action_semantics="q_goal_confirmed_submission"`。

`data.h5` 的 `meta` 顶层属性保存该段的相机内参、畸变、深度单位和已有外参。`effective_parameters` 是 JSON，保存实际使用的硬件和控制参数，包括 `camera_config`、适用的 VR 标定、五指 SDK `tactile_reset` 回执和 `tactile_calibration`。校准元数据包含本次状态、最后一次有效 Bias、该 Bias 的 Unix 时间（纳秒）和应用状态。

VR 录制的 `effective_parameters` 还保存 `hand_retargeting` 的实际求解参数、`hand_retargeting_calibration` 的手部骨长与掌宽，以及已安装的 `dex_retargeting_version`。 [手部重定向](hand_retargeting.md)

`meta.xhand_crc_counts_json` 记录录制窗口内的读写 CRC 回复数、读写恢复次数和因 CRC 未确认送达的写入次数（`unconfirmed_writes`）。

Raw 按字段使用固定类型。关节、绝对关节目标、VR 腕部位姿和手部关键点使用 `float64`，触觉与 HOI 点云使用 `float32`，深度使用 `uint16`，VR 接收时间和序列号使用 `uint64`。`timing/segment_starts` 使用 `int64`，`timing/rows` 使用 `TIMING_ROW_DTYPE`。

`hand_contact` 和 `hand_tactile_force` 保留现有形状与 `/ 255` 缩放。应用校准时先用 SDK 原始值减去 `tactile_calibration.bias_sdk`，否则直接缩放原始值。两种触觉独立按拇指到小指记录，缺失或过期的指尖存为 NaN。五指均有限时该模态有效，`timing/rows` 对应触觉时间记录最老的有效指尖来源时间。任一指尖缺失时，对应模态时间为 `0`。

## 时间与动作

`timing/rows` 保留控制采样时刻、观测完成时刻及各来源的主机单调时钟时间，单位为纳秒。相机派生观测沿用原始相机序列和来源时间，重复读取保留原时间。Arm 和 Hand 各自保留反馈来源时间。这些时间戳记录主机侧查询或接收时刻。软件按 Raw 行配对快照，允许相机、Arm 和 Hand 在不同物理时刻采样。

Raw 的 `action_arm_joint_target` 和 `action_hand_joint_target` 保存已在主机侧完整且按时提交的臂手绝对关节目标。VR 只写成功提交的行。策略正常等待且这一拍没有新指令时，保留观测行，两列目标均填 NaN。停止前未完成提交的终止拍省略。设备实际执行和到位以实测反馈为准。

Raw 的手部关节列通过 `meta.hand_joint_names` 和 `meta.joint_units` 校验，使用 SDK ID 顺序和弧度。Canonical Zarr 的 `joint_state` 和 `action` 在 7 个臂关节之后接 12 个手关节，`action_ee` 在 9 个末端位姿分量之后接同样的 12 列。

`timing/segment_starts` 记录连续段首行。跳槽或 HOLD 产生录制间断时，下一个接纳的行开始新段。队列满或录制来源失效则停止接纳并记录 `recording_truncated_reason`。

真机回放直接使用原始 Raw 所选连续段 `[start, end)` 的绝对关节目标。连接设备前，检查该段每行臂手目标的完整性、有限性和当前实验操作限位。发现缺失或无效的目标时，能定位则报告 Raw 内从 0 开始的绝对行号。运行前需将机器人就位到该段起点。播放结束并停稳后保持连接，按 H 回到初始位并退出。S/Q 停止并退出，Esc 急停。回放发送时遇到反馈不可用或提交未确认会结束回放。

## 图像与模型输入

Raw 保留采集像素网格上的 RGB-D，解码后的有损 RGB 与现场像素值可能有差异。点云取色使用这一路 RGB-D。在线策略 RGB、Canonical RGB/depth 和 Raw 可视化使用同一针孔图像几何，沿用固定 K、原分辨率和光轴。RGB 双线性采样，depth 最近邻采样，越界填零。含图像的 Zarr 标记 `image_geometry="pinhole"`。

读取深度时要求 Raw 记录对齐到彩色网格的深度及其正尺度。读取 RGB 时要求 `rgb.mp4` 包含视频流，并检查像素尺寸和纳秒时基，逐帧核对时间戳是否等于该行 `control_ns` 减去首行 `control_ns`。完整读取时还检查 `rgb.mp4` 帧数与 `timing/rows` 一致。限帧可视化只解码和检查所选视频前缀。

模型所需模态经投影进入策略历史。Raw 录制保留实测关节、触觉和原始 RGB-D。启用录制时，普通 RGB 策略与 Raw 共用相机源帧。HOI 或场景点云策略的 RGB 跟随对应的派生观测，Raw 从相机通道单独取帧。触觉缺失可随 Raw 保存。导出时只检查所选模态，现有无 Mask 的触觉字段要求所有指尖有效。

## 按需导出

- 每次 `export_zarr.py INPUT_ROOT` 发布一份固定字段的 Zarr。默认导出由 Raw 生成的全部字段，所选录制均有 `hoi.h5` 时一并导出 `object_pointcloud`。用 `--fields ...` 可指定字段。纯关节导出跳过相机，纯 RGB 跳过深度单位。关节观测和 FK 派生结果使用对应 Raw 行的原始关节状态，物体点云使用该行对应的 HOI 标注。
- 用 `--output` 指定输出路径，否则默认写入当前目录的 `datasets/<输入目录名>.zarr`，单个 Episode 使用其父目录名。新产物完整后原子发布，路径须尚不存在。
- 同一产物的 `task_name` 取自各 Raw 一致的 `meta.task_label`。`action_space="joint"` 标记采集动作空间。含动作的产物使用根属性 `action_semantics="held_accepted_joint_target"`。
- 选择 `action` 或 `action_ee` 时，导出器构造同一份关节目标序列。两列目标均有限的行使用原目标。两列目标均为 NaN 的等待行沿用本段上一条成功目标。首次成功提交前的等待行使用段首 Raw 行的实测关节状态。每段独立填充并保留原时间轴。`action_ee` 将同一序列的臂关节目标经 FK 转为末端位姿，并保留手关节目标。
- 每个原生连续段成为一个 Zarr episode，`source_episodes` 和 `source_row_ranges=[start, stop)` 与 `episode_ends` 逐段对应。采样窗口限于段内。同一采集的各段属于同一个训练或验证分组，`val_episodes` 和 `max_train_episodes` 按采集组计数。
- 所选数据缺失、损坏或所选 HOI 行含无效对象时，放弃该条 Raw 的全部结果，继续处理其他采集。完整有效的截断前缀仍可导出。没有可用采集时导出失败。权限、磁盘和输出写入错误会终止导出，并保留原始错误。
- 根属性 `rejected_episodes` 保存被拒的入口名和原因。臂手目标的有限值与 NaN 混杂或含无穷值时，原因给出首个问题的 Raw 绝对行号。`excluded_episodes` 保存显式排除的入口名。排除后仍有两个入口指向同一目录时会报错。

配套 Policy 的 Real 原生 ODE 使用同一关节 `action` 序列，可用 `export_zarr.py INPUT_ROOT --fields joint_state point_cloud action` 导出。

Zarr 的 `data_processing` 保存实际使用的处理参数。物体点云与关节观测或其 FK 结果一同导出时，`data_processing.object_pointcloud.joint_state_source` 为 `raw_observation_row`。开启去桌面时，归一化系数写入 `data_processing.point_cloud.table_plane_abcd`。部署对应值见 `run_config.yaml` 的 `actual_processing.point_cloud.table_plane_abcd`。接触预训练按同一观测行的手部和物体点云计算 1024 个有序标签。见[标注与查看](object_reconstruction.md#标注与查看)。

已有 `hoi.h5` 在相机来源时间、序列和外参通过核对后可随 Raw 复用。训练用关节和 FK 输入取自对应 Raw 行。[标注与查看](object_reconstruction.md#标注与查看)
