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
| `annotate_episodes.py` | 跟踪已发布 Raw 的 RGB-D，完成逐帧标注后写入点云、位姿、质量和模板来源；暂时跟踪丢失记录无效帧并继续恢复，输入或文件错误直接报错。 |

米制网格保留源物体原点和朝向；模板固定对象顺序、点分段和点序，在线与离线共用。已有模板直接复用；重新采样需显式移除旧模板，实验期间固定资产。已有手模板不会自动更换。

## 离线标注

`annotate_episodes.py` 默认处理任务下所有已发布 Raw，也可用 `--episode <目录>` 指定单段。标注需要任务模板及 Raw 中的相机外参，校验并跳过已有 HOI 的 Raw，完成逐帧标注后写入基座坐标系下的米制 float32 点云；相同相机序列复用位姿、质量和有效性。暂时丢失的帧标为无效，后续帧继续尝试恢复。

每个 `prompt` 须在去畸变后的首帧匹配唯一候选，缺失或歧义会使初始化失败，后续新帧重试。`instance` 仅用于参考图选择，不用于跟踪初始化。

标注在 `/hoi` 中记录位姿、质量、有效性、模板和坐标变换，命令不会覆盖已有标注。已有标注缺少有效性字段或结构不符时直接报错。Python 调用 `annotate_episodes(..., tracking_config=...)` 可调整分割间隔与恢复参数，默认值见[后端说明](../../third_party/cordvip_geometry/README.md#遮挡恢复)。

## 分割间隔与只读 A/B

离线标注可传 `--segmentation-interval-frames 1` 使用逐帧基线，默认 2；在线 worker 的 Python 参数 `tracking_config` 接收同一配置。两者均仅对新的相机序列推进调度。

在现有 `real_robot` CUDA 环境、仓库根目录运行，选择尚不存在的输出目录：

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/cordvip/benchmark_tracking.py --output logs/sam2_ab_new
```

默认回放 pnp_apple_banana 全部已发布 Episode，对 interval=1/2/3 分别完整预热后各测量三次；可重复传 `--episode` 选择固定数据，`--repeats` 改重复次数。入口不写 Raw，不需要点云模板，不操作设备。所选 Episode 名称必须唯一，避免同名结果覆盖。

报告记录各阶段同步 CUDA 的墙钟时间、逐对象质量和有效性、LOST/恢复转换次数及视频/恢复 SAM2 调用数。总跟踪耗时排除读盘、初始化和独立参考复核；恢复帧包含在总跟踪分位数中。逐帧基线 mask 用于在计时外复核候选位姿；参考检查拒绝是需要调查的分歧，不是有真值支持的绝对位姿误差；参考帧缺失或汇总帧数不一致会直接报错。逐帧记录保存在各 Episode 的 JSON 中，`summary.json` 保存逐轮统计，`comparison.json` 保存跨轮汇总；环境、输入路径和对象列表见 `environment.json`。测量口径与已有结果见[后端实测](../../third_party/cordvip_geometry/README.md#p2-事件触发-sam2-实测)。

覆盖率修复的复现输出使用 `--output logs/sam2_depth_coverage_ab`，保留历史 `logs/sam2_ab`。`depth_quality_by_mask_source` 分别汇总两条路径的匹配点数、匹配比例和覆盖率 p50/p95；两条路径的覆盖率定义相同，匹配比例分母不同。

逐帧 `depth_checks` 保存刷新前的纯深度检查。`coverage_rejected_frames` 统计匹配点数和匹配比例原本通过、仅覆盖率失败的处理帧；`coverage_triggered_sam_frames` 统计实际刷新 SAM2 且至少一个候选覆盖不足的处理帧（可能同时存在其他质量异常）。两者均按共享帧去重，不等同于最终 LOST；最终新增无效帧须对照历史逐对象有效性。诊断只在评估入口收集，不改变跟踪状态机。

## 导出与查看

使用 `scripts/export_zarr.py` 导出训练数据，使用 `scripts/visualize_episode.py` 查看点云与质量指标。联合导出物体点云和关节/FK 观测时，会按相机来源时间插值所需设备的关节，前后反馈最大间隔由 `execution.hoi_max_joint_gap_s` 指定，默认 0.10 秒。关节对齐只裁去连续区间不可对齐的首尾，内部缺口报错。随后过滤感知无效行，观测与动作同步选择，并将有效连续区间分别导出为训练 episode，避免训练窗口跨越遮挡缺口。

数据表示、对齐和来源记录见[物体重建与 HOI](../../docs/object_reconstruction.md)，后端安装与接口见 [cordvip_geometry](../../third_party/cordvip_geometry/README.md)。
