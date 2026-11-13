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

## 离线标注

`annotate_episodes.py` 默认处理任务下所有已发布 Raw，也可用 `--episode <目录>` 指定单段。标注需要任务模板及 Raw 中的相机外参，跳过已有 HOI 的 Raw，整段成功后写入基座坐标系下的米制 float32 点云；相同相机序列复用位姿。

每个 `prompt` 须在去畸变后的首帧匹配唯一候选，缺失或歧义会终止当前段。`instance` 仅用于参考图选择，不用于跟踪初始化。

查看已保存的点云使用 `scripts/visualize_episode.py`，导出训练数据使用 `scripts/export_zarr.py`。安装、跟踪原理与单物体操作见 [cordvip_geometry](../../third_party/cordvip_geometry/README.md)。

## 导出训练数据

```bash
python scripts/export_zarr.py episodes/pnp_apple_banana \
  --output datasets/pnp_apple_banana.cordvip.zarr \
  --fields joint_state action hand_pointcloud object_pointcloud
```

`object_pointcloud` 必须显式选择；省略 `--fields` 时不导出该字段。Zarr 的 `task_name` 来自 Raw 的 `meta.task_label`，不会从可改名的目录推断。一次导出只接受同一任务、相同控制周期的 episode，并保留录制的连续区间边界。

导出 `action` 时，`action_space=joint`，对应 7 个机械臂关节和 12 个手关节的绝对目标；同时导出的 `action_ee` 是派生表示。仅导出 `action_ee` 时标记为 `eef`，这类数据不能直接通过当前 dexmani_policy 要求 `joint` 的训练入口。仅导出观测时不写 `action_space`。

新 HOI 标注在 Raw 的 `/hoi/template/` 保存实际使用的 `points_local`、`offsets` 和 `object_names`。导出物体点云时，将对象名与分段写入 Zarr 根属性 `object_names` / `object_offsets`，局部模板点写入 `meta/object_points_local`；跨 episode 的模板必须完全一致，包括点序。物体点云仍为基座系、米制的 `[T,1024,3]`，模型输入不变。

旧 HOI 没有模板快照时仍可导出，但会警告无法校验分段和点身份，也不允许与含快照的标注混合导出。导出器不会用当前资产补造历史元数据；需要这些信息时应重新生成 HOI 标注。这些元数据为后续接触标签预处理保留依据，本次不改变 dexmani_policy 的法向量或接触标签计算。
