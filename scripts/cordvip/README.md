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
| `sample_template.py` | 按网格表面积分配配额，8 倍候选点经 FPS 选取 1024 个点，每对象至少 64 点，最多 16 个对象，保存到 `hoi/template.npz`。 |
| `annotate_episodes.py` | 跟踪已发布 Raw 的 RGB-D，整段成功后写入点云、位姿、质量和模板来源；跟踪失败跳过该段，输入或文件错误直接报错。 |

米制网格保留源物体原点和朝向；模板固定对象顺序、点分段和点序，在线与离线共用。已有模板直接复用；重新采样需显式移除旧模板，实验期间固定资产。已有手模板不会自动更换。

## 离线标注

`annotate_episodes.py` 默认处理任务下所有已发布 Raw，也可用 `--episode <目录>` 指定单段。标注需要任务模板及 Raw 中的相机外参，跳过已有 HOI 的 Raw，整段成功后写入基座坐标系下的米制 float32 点云；相同相机序列复用位姿与质量。

每个 `prompt` 须在去畸变后的首帧匹配唯一候选，缺失或歧义会终止当前段。`instance` 仅用于参考图选择，不用于跟踪初始化。

标注在 `/hoi` 中记录位姿、质量、模板和坐标变换，命令不会覆盖已有标注。

## 导出与查看

使用 `scripts/export_zarr.py` 导出训练数据，使用 `scripts/visualize_episode.py` 查看点云与质量指标。联合导出物体点云和关节/FK 观测时，会按相机来源时间插值所需设备的关节，前后反馈最大间隔由 `execution.hoi_max_joint_gap_s` 指定，默认 0.10 秒。仅裁去连续区间不可对齐的首尾，观测与动作同步裁剪；内部缺口报错。

数据表示、对齐和来源记录见[物体重建与 HOI](../../docs/object_reconstruction.md)，后端安装与接口见 [cordvip_geometry](../../third_party/cordvip_geometry/README.md)。
