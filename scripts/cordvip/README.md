# CordViP 实验入口

从仓库根目录运行，使用 `real_robot` 环境。用 `--task-dir` 显式指定包含 `task.json`、对象目录和 `hoi/template.npz` 的任务资产目录，可以放在仓库外。所有显式路径支持 `~`，相对路径以 cwd 为准。

```bash
python scripts/cordvip/prepare_task.py --task-dir /data/tasks/pick
python scripts/cordvip/sample_template.py --task-dir /data/tasks/pick
python scripts/cordvip/annotate_episodes.py /data/episodes/pick --task-dir /data/tasks/pick
python scripts/cordvip/annotate_episodes.py /data/episodes/pick/episode_001 --task-dir /data/tasks/pick
python scripts/run_policy.py CHECKPOINT --task-dir /data/tasks/pick
```

`task.json` 的 `objects` 列表定义任务对象及顺序，每项包含 `name`。

每个对象的输入放在 `<task-dir>/<name>/`，参考图默认是 `reference.jpg`，已有米制网格是 `mesh/mesh_m.obj`。例如 `/data/tasks/pick/task.json` 可以包含 `{"objects": [{"name": "apple", "size_axis": "x", "size_meters": 0.08}]}`，参考图放在 `/data/tasks/pick/apple/reference.jpg`。这些路径与 Raw 及 checkpoint 中的任务标签无关。

- 分割：`prompt` 默认对象名，`image` 默认 `reference.jpg`。参考图有多个候选时，用 `instance` 指定去重、过滤空 mask 后按置信度降序排列的零起始索引。
- 尺度：需要重建的对象须提供 `scale_to_meters`，或配对的 `size_axis` / `size_meters`。
- 重建：可设置 `seed`、`steps`、`octree_resolution`、`num_chunks`。

| 入口 | 行为与输出 |
| --- | --- |
| `prepare_task.py` | 复用已有米制网格；其余对象共用模型完成分割、重建，保存 `mesh/mesh_m.obj` 和 `reconstruction/` 中间结果。 |
| `sample_template.py` | 按网格表面积分配配额，8 倍候选点经 FPS 选取 1024 个点，每对象至少 64 点，最多 16 个对象，保存到 `hoi/template.npz`。 |
| `annotate_episodes.py` | 跟踪已发布 Raw 的 RGB-D，完成逐帧标注后写入点云、位姿、质量和固定模板；暂时跟踪丢失记录无效帧并继续恢复，输入或文件错误直接报错。 |

米制网格保留源物体原点和朝向；模板固定对象顺序、点分段和点序，在线与离线共用。已有任务模板直接复用；修改物体 Mesh、手部 URDF 或 Visual Mesh 后，显式重建对应模板：

```bash
python scripts/cordvip/sample_template.py --task-dir /data/tasks/pick --overwrite
python scripts/generate_hand_template.py --overwrite
```

手模板保存 XYZ 和 Link 分段，任务模板保存 XYZ、对象顺序和分段。模板更新后重新生成对应标注和训练数据；写入通过临时文件原子替换，失败时保留原文件。

`prepare_task.py` 也检查复用的米制 Mesh；提供 `size_axis` / `size_meters` 时核对指定轴尺寸，容差为 `rtol=1e-4, atol=1e-6 m`。仅提供缩放因子时无法从米制文件反推实物尺寸。少量退化面会报告数量，整个网格无有限非零面积时拒绝。

## 离线标注

`annotate_episodes.py INPUT_ROOT --task-dir TASK_DIR` 接受一个含 `data.h5` 的 Episode，或直接包含 `episode_*` 子目录的数据目录；仅处理指定目录，不递归查找，也不从任务标签推导数据路径。无效路径或空目录直接报错。标注需要任务模板及 Raw 中的相机外参，已有 `hoi.h5` 时直接跳过；显式传入 `--overwrite` 重算。完成逐帧标注后写入基座坐标系下的米制 float32 点云；相同相机序列复用位姿、质量和有效性。遮挡、丢失和两帧恢复确认期间标为无效，后续新观测继续跟踪或恢复；时间间隔与 Mask 年龄使用 Raw 中真实相机时间和来源序列。

每个 `prompt` 须在去畸变后的首帧匹配唯一候选，缺失或歧义会使初始化失败，后续新帧重试。`instance` 仅用于参考图选择，不用于跟踪初始化。

标注保存在独立 `episode/hoi.h5` 的根目录，包括位姿、质量、有效性、模板和坐标变换。`data.h5` 和 `rgb.mp4` 只读。写入同目录临时文件并成功关闭后才发布；重算失败保留此前完整标注。Python 调用 `annotate_episodes(..., tracking_config=..., overwrite=True)` 可调整跟踪配置并重算，参数见后端的[质量检查](../../third_party/cordvip_geometry/README.md#质量检查)和[遮挡恢复](../../third_party/cordvip_geometry/README.md#遮挡恢复)。

## 分割间隔

离线标注通过 `--segmentation-interval-frames` 调整活动对象的 SAM2 最大处理帧间隔，默认 2；设为 1 时逐帧分割。在线 worker 的 `tracking_config` 参数接收相同配置。两者均仅对新的相机序列推进处理计数；来源序列的跳跃不补造 SAM2 图像。深度质量不通过、对象恢复到期或正在确认恢复时提前刷新；全部对象 LOST 时等待恢复到期。具体规则见[事件触发分割](../../third_party/cordvip_geometry/README.md#事件触发分割)。

## 导出与查看

使用 `scripts/export_zarr.py --fields ...` 显式选择一份输出的字段，使用 `scripts/visualize_episode.py` 查看点云与质量指标。仅选择 `object_pointcloud` 时读取 HOI；联合选择关节/FK 观测时按相机来源时间在原生连续段内对齐关节，前后反馈最大间隔由 `execution.hoi_max_joint_gap_s` 指定，默认 0.10 秒。每个 `timing/segment_starts` 区间成为一个训练 episode；不裁首尾、不按质量另切段。任一所选数据或对齐失败都拒绝整条 Raw，其余 Raw 继续导出。同源段在训练与验证之间按原始采集分组，避免泄漏。

数据表示与时间对齐见[物体重建与 HOI](../../docs/object_reconstruction.md)，后端安装与接口见 [cordvip_geometry](../../third_party/cordvip_geometry/README.md)。
