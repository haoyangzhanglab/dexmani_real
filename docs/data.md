# 实验数据

## Raw Episode

遥操作默认写入当前目录的 `episodes/<task_name>/`。每个已发布的 Episode 可独立读取和导出，包含 `data.h5`、有损 H.264 图像 `rgb.mp4` 和可选的 [HOI 标注](object_reconstruction.md#标注与查看) `hoi.h5`。用 `python scripts/visualize_episode.py EPISODE --info` 查看结构和录制截断原因。

`data.h5` 的 `meta` 顶层属性保存该段的相机内参、畸变、深度单位和已有外参。`effective_parameters` 是 JSON，保存实际使用的硬件、控制和算法参数，包括 `camera_config`、适用的 VR 标定和 `tactile_reset`。任务成败、控制结束原因与录制结果分别记录。任务成败可用 [label_episode.py](../scripts/label_episode.py) 标注。

Raw 写入时，关节、绝对关节目标、VR 腕部位姿和手部关键点使用 float64，聚合与稠密触觉使用 float32。读取时这些字段接受任意浮点精度，保留文件中的 dtype。深度须为 uint16，HOI 的 `object_pointcloud` 须为 float32，VR 接收时间和序列号须为 uint64，`timing/rows` 须符合 `TIMING_ROW_DTYPE`。

## 时间与动作

`timing/rows` 保留控制采样时刻、观测完成时刻及各来源的主机单调时钟时间，单位为纳秒。相机派生观测沿用相机来源时间，重复读取保留原时间。多来源观测按这些时间做软件对齐，时间戳记录的是主机侧查询或接收时刻。

实测关节和指令提交尝试分别保存。动作是尝试提交的绝对关节目标，包含发送失败的尝试。实际位置由反馈列提供。若一行没有提交某个执行器的目标，该执行器的动作列为 NaN。

`timing/segment_starts` 记录连续段首行。跳槽或 HOI HOLD 产生录制间断时，下一个接纳的行开始新段。队列满或录制来源失效则停止接纳并记录 `recording_truncated_reason`。Replay 只执行选定连续段的关节目标，运行前需将机器人就位到该段起点。

## 图像与模型输入

Raw 保留采集像素网格上的 RGB-D，解码后的有损 RGB 与现场像素值可能有差异。点云取色使用这一路 RGB-D。在线策略 RGB、Canonical RGB/depth 和 Raw 可视化使用同一针孔图像几何，沿用固定 K、原分辨率和光轴。RGB 双线性采样，depth 最近邻采样，越界填零。含图像的 Zarr 标记 `image_geometry="pinhole"`。

模型所需模态经投影进入策略历史。Raw 录制使用本次采样的实测关节、触觉和原始 RGB-D。启用录制时，普通 RGB 策略与 Raw 共用相机源帧。HOI 或场景点云策略的 RGB 跟随对应的派生观测，Raw 从相机通道单独取帧。录制所需的传感器来源也成为本段依赖。

## 按需导出

- 每次 `export_zarr.py --fields ...` 发布一份固定字段的 Zarr。纯关节导出跳过相机，纯 RGB 跳过深度单位，选择 `object_pointcloud` 时才读取 HOI。用 `--output` 指定输出路径，否则默认写入当前目录的 `datasets/<输入目录名>.zarr`，单个 Episode 使用其父目录名。新产物完整后原子发布，路径须尚不存在。
- 同一产物的 `task_name` 取自各 Raw 一致的 `meta.task_label`。`action_space="joint"` 标记采集动作空间，训练用 `action_key` 选择 `action` 或经 FK 派生的 `action_ee`。
- 每个原生连续段成为一个 Zarr episode，`source_episodes` 和 `source_row_ranges=[start, stop)` 与 `episode_ends` 逐段对应。位置对齐和采样窗口限于段内。同一采集的各段属于同一个训练或验证分组，`val_episodes` 和 `max_train_episodes` 按采集组计数。
- 所选数据缺失、损坏或所需的 HOI 关节对齐无法覆盖整段时，整条 Raw 回滚并继续处理其他采集。完整有效的截断前缀仍可导出。没有可用采集时导出失败。

`data_processing` 是普通训练和恢复可选的研究记录，保存实际使用的处理参数。开启 contact loss 的固定 1024 槽预训练及其完整续训要求共享的对象模板身份。见[模板摘要](object_reconstruction.md#标注与查看)。
