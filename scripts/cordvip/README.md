# CordViP 实验入口

从仓库根目录运行，使用 `real_robot` 环境。任务目录为 `assets/pointcloud_template/object/<task>/`。

```bash
python scripts/cordvip/prepare_task.py --task pnp_apple_banana
python scripts/cordvip/sample_template.py --task pnp_apple_banana
python scripts/cordvip/annotate_episodes.py --task pnp_apple_banana
```

`task.json` 的 `objects` 列表定义任务对象及顺序，每项包含 `name`。

- 分割：`prompt` 默认对象名，`image` 默认 `reference.jpg`。参考图有多个候选时，用 `instance` 指定去重、过滤空 mask 后按置信度降序排列的零起始索引。
- 尺度：需要重建的对象须提供 `scale_to_meters`，或配对的 `size_axis` / `size_meters`。
- 重建：可设置 `seed`、`steps`、`octree_resolution`、`num_chunks`。

| 入口 | 行为与输出 |
| --- | --- |
| `prepare_task.py` | 复用已有米制网格；其余对象共用模型完成分割、重建，保存 `mesh/mesh_m.obj` 和 `reconstruction/` 中间结果。 |
| `sample_template.py` | 按网格表面积分配 1024 个点，每对象至少 64 点，最多 16 个对象，保存到 `hoi/template.npz`。 |
| `annotate_episodes.py` | 跟踪已发布 Raw 的 RGB-D，整段成功后写入 `/hoi/object_pointcloud`；跟踪失败跳过该段，输入或文件错误直接报错。 |

米制网格保留源物体原点和朝向。模板固定对象顺序、点分段和点序，在线与离线共用。已有模板直接复用；需要重新采样时先移除旧模板，实验期间保持网格和模板不变。

标注支持 `--episode <episode目录>`，已有 HOI 的 Raw 会跳过。相同相机序列复用位姿；RGB 使用 4 个解码线程，物体点云按 128 帧分块变换、写入。在线与离线共用 `transform_task_points()`，输出为基座坐标系下的米制 float32 点云。

跟踪时，每个 `prompt` 须在去畸变后的首帧匹配唯一候选；缺失或歧义会终止当前段。`instance` 仅用于参考图选择，不用于跟踪初始化。检测去重、SAM2 初始化和深度门控见后端说明。

检查结果使用 `scripts/visualize_episode.py`，训练数据使用 `scripts/export_zarr.py` 导出。后端安装与单物体操作见 [cordvip_geometry](../../third_party/cordvip_geometry/README.md)。
