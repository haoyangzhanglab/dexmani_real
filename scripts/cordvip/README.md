# CordViP 实验入口

## 任务级 HOI 流程

任务资产位于 `assets/pointcloud_template/object/<task_name>/<object_name>/`，每物体需要米制 Mesh 和 asset.json。任务配置、尺度与模板约定见[物体资产](../../docs/object_reconstruction.md)。模板由研究者固定，运行时只读。

```bash
python scripts/cordvip/build_task.py --task pnp_apple_banana
# 仅用于新实验首次冻结；已有模板只读取复用。
python scripts/cordvip/allocate_points.py assets/pointcloud_template/object/pnp_apple_banana
python scripts/cordvip/annotate_episodes.py --task pnp_apple_banana
python scripts/visualize_episode.py episodes/pnp_apple_banana/episode_20261101_165909
python scripts/export_zarr.py episodes/pnp_apple_banana \
  --output datasets/pnp_apple_banana.cordvip.zarr \
  --fields joint_state action hand_pointcloud object_pointcloud
```

默认无需 --config，execution.hoi_max_skew_s=None 关闭额外来源时间差限制。需要启用时，通过 --config 的 CONFIG.execution.hoi_max_skew_s 设置有限正数秒值；程序调用使用 ExportConfig.hoi_max_skew_s。标注可追加 `--episode` 指定单段。已发布结果不覆盖；首次 pose 失效即结束该段且不发布。导出遇到点云缺失、结构错误、非有限坐标或已启用的来源时间差检查超限时拒绝，不丢行拼接。

新模板直接按面积配额与固定 seed 表面采样；通过正式标注和 visualize_episode 查看结果。

手点云导出为 XYZ，普通 RGB-D point_cloud 仍为 XYZRGB。Zarr 可以直接用于 CordViP 训练，接触预训练为可选步骤。相机标定、动作语义和 Raw 连续段边界保持不变。

## 单物体重建

独立重建可直接指定图片和尺度。将图片路径、网格轴及尺寸替换为实际值，以下 `0.08` 仅为米制尺寸示例：

```bash
python scripts/cordvip/reconstruct_object.py \
  --image path/to/object.jpg --prompt apple \
  --size-axis x --size-meters 0.08 \
  --headless --output logs/cordvip/apple_build
```

提供尺度时，米制资产写入 `<output>/asset/`；省略尺度时保留未定尺度网格，后续需完成尺度校准。`--headless` 下多候选需要查看候选结果并明确指定零起始的 `--instance`；任务构建时对应 `task.json` 中的 `instance`。
