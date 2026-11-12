# CordViP 几何核心

`prepare_asset(mesh_path, output, scale_to_meters=...)` 或实测轴尺寸参数将源 Mesh 转为米制，保留原点。`load_asset(path)` 返回 `(mesh, metadata)`，不加载单物体采样文件。Mesh 顶点法向仍用于跟踪。

`RGBDFrame` 接收 color 网格 RGB、米制深度、内参与畸变参数；`from_camera()` 使用实测相机标定转换。离线标注逐行读取 Raw 时间和 RGB，仅在相机序列更新时读取深度并跟踪；重复序列复用上一行位姿。

`RGBDFrame`、`SceneTracker` 和 `TrackingFailure` 均可从 `cordvip_geometry` 导入。

`SceneTracker.open(assets, *, workspace)` 要求每个名字对应一个 Mesh 资产。首帧由 `detect_masks(frame, prompts)` 获取 initial_masks，调用 `step(frame, initial_masks=masks)`；后续直接 `step(frame)`。新 Episode 调用 reset()，模型权重和资产继续复用。

step 成功返回 `{object_name: pose_camera}`，位姿将原物体坐标系变换到相机系。任一物体预期跟踪失败立即抛出 TrackingFailure（含物体及原因），不返回部分结果。调用方结束当前 Episode，下一段先 reset；在线 Worker 保留整轮失败锁存。模型、IPC、文件和程序错误保持原异常传播。

跟踪使用 workspace 下默认模型目录、默认 SAM 配对、当前解释器和临时调试目录。语义支持与模型深度一致性阈值为 tracking 模块常量：pixel_stride=2、depth_tolerance_m=0.015、min_matched_pixels=30、min_matched_fraction=0.15，同时保留 FoundationPose 最小有效深度要求。内部只保留判定所需深度和像素，不输出展示 mask、颜色或三维诊断载荷。

正式离线入口为 annotate_episodes.py；它按 camera_sequence 去重并在首次失效停止该 Episode。结果通过主项目 visualize_episode.py 查看。重建选择支持图片或含 rgb 的 NPZ，选择结果使用固定的 mask.png、foreground_rgba.png、overlay.png 文件名。

FoundationPose 权重与同帧预处理复用、SAM 子进程隔离及有界时序状态继续保留。参见[实验命令](../../../scripts/cordvip/README.md)和[资产约定](../../../docs/object_reconstruction.md)。

输出使用尚不存在的新目标，可位于候选、资产或 Raw 目录内部，由研究者选择位置；已有输出仍拒绝覆盖。模型下载目录允许复用。
