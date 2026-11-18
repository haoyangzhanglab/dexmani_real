# 物体重建与 HOI

HOI（Hand–Object Interaction，手物交互）观测用手部与物体点云描述操作场景，供 CordViP 等策略训练与部署使用。

## 点云与资产

物体点云来自固定网格表面模板，由 RGB-D 跟踪位姿逐帧变换。手点云来自固定的 link 局部模板，由机械臂和手部关节通过 FK 变换。两者均为 xArm base 坐标系下的米制 float32 XYZ，保留点序和分段。

任务模板共 1024 点，按表面积分配配额，每对象至少 64 点，最多 16 个对象。新模板先按面积采样 8 倍候选点，再用固定种子和起点的 FPS 选点。手模板沿用各 link 的面积配额与 URDF visual 变换，使用同一采样函数。

手模板保存 float32 `points[N,3]`，按 `part_names` 和整数 `part_offsets` 对应 URDF link。`hand_geometry()` 接收 EEF 变换 `[T,4,4]` 与 SDK 顺序的手关节 `[T,12]`，通过批量 FK 返回所选的指尖 `[T,5,3]` 和表面点 `[T,N,3]`；导出、部署和可视化共用此接口，保留模板点序与分段。

通用场景 `point_cloud` 由 `PointCloudProcessor` 处理 RGB-D，包含深度滤波、桌面裁剪、体素降采样和离群点过滤；HOI 点云直接使用固定 CAD/URDF 表面模板。

任务模板保存 `points_local`、`object_names` 和 `offsets`。部署与离线标注共用任务解析，核对模板和 `task.json` 的对象集合，按模板顺序匹配网格和 prompt。

通用场景点云在过滤后无有效点时返回 float32 `[0,6]`；非空但不足目标点数时循环重复已有 XYZRGB，超过目标点数时执行 FPS。重复填充仅满足输出形状，不增加真实几何信息。

## 实验流程

从仓库根目录运行，使用 `real_robot` 环境。用 `--task-dir` 为准备、采样、标注和部署指定同一个任务资产目录。目录可放在仓库外，显式路径支持 `~`，相对路径以当前目录为准。Raw 和 checkpoint 中的任务标签用于标识实验。

### 任务资产

`task.json` 的 `objects` 列表定义对象名称和顺序，例如 `{"objects": [{"name": "apple", "size_axis": "x", "size_meters": 0.08}]}`。对象输入放在 `<task-dir>/<name>/`，参考图默认为 `reference.jpg`，已有米制网格放在 `mesh/mesh_m.obj`。

- 分割使用对象名作为默认 `prompt`，可用 `image` 指定其他参考图。有多个候选时，`instance` 指定去重并过滤空 mask 后按置信度排序的零起始索引。
- 重建保存 `reconstruction/` 下的中间结果。可设置 `seed`、`steps`、`octree_resolution` 和 `num_chunks`。
- 重建后使用 `scale_to_meters`，或配对的 `size_axis` 和 `size_meters` 生成米制网格，保留原点和朝向。复用已有网格时检查网格有效性和所给轴尺寸，尺寸容差为 `rtol=1e-4, atol=1e-6 m`。

### 采集、标注与部署

```bash
python scripts/collect_teleop.py pick --episodes-dir /data/episodes
python scripts/cordvip/prepare_task.py --task-dir /data/tasks/pick
python scripts/cordvip/sample_template.py --task-dir /data/tasks/pick
python scripts/cordvip/annotate_episodes.py /data/episodes/pick --task-dir /data/tasks/pick
python scripts/export_zarr.py /data/episodes/pick --fields joint_state action hand_pointcloud object_pointcloud
```

在 dexmani_policy 中训练后，用 `python scripts/run_policy.py CHECKPOINT --task-dir /data/tasks/pick` 部署。采样生成 `hoi/template.npz`，已有模板直接复用。修改物体网格后用 `sample_template.py --task-dir TASK_DIR --overwrite` 重建。修改手部 URDF 或 Visual Mesh 后，用 `python scripts/generate_hand_template.py --overwrite` 重建手模板。模板完整写入后原子替换，重建后重新生成相关标注和训练数据。

标注输入也可直接指定单个 Episode，如 `/data/episodes/pick/episode_001`。目录输入只处理其直接包含的 `episode_*` 子目录。标注读取 Raw 内的 RGB-D、深度单位与相机外参。离线用 `--segmentation-interval-frames` 设置活动对象的 SAM2 最大处理帧间隔，默认 2，设为 1 时逐帧分割。Python 入口 `annotate_episodes(..., tracking_config=..., overwrite=True)` 可调整同一套配置。见[后端安装、接口与分割规则](../third_party/cordvip_geometry/README.md)。

## 跟踪与时间对齐

在线 worker 加载并预热模型后宣告就绪。每段首帧对 RGB-D 去畸变，经 Grounding DINO、SAM2 和 FoundationPose 注册得到初始位姿。每个 `prompt` 须匹配唯一候选，参考图的 `instance` 只用于资产准备。后续对活动对象合批 refinement，使用 CAD、深度及当前 Mask 验证候选。见[后端质量规则](../third_party/cordvip_geometry/README.md#质量检查)。

当前按非对称物体跟踪；对称物体的旋转歧义见 [Object Symmetry](../third_party/cordvip_geometry/README.md#object-symmetry)。

模型跨段复用，初始化重置视频、位姿和相机缓存，段内相机参数及图像尺寸固定。在线、离线共用 SceneTracker，帧输入必须包含真实相机时间和来源序列；重复序列复用结果，跳帧不补造 SAM2 图像。

对象独立维护 TRACKING、OCCLUDED、LOST。证据不足时保留可靠内部位姿继续 Track，但当前观测无效；明确矛盾或遮挡超时才进入 LOST。恢复经过检测、分割、注册及下一处理帧的再次验证，两帧可信后才发布有效位姿。初始化因观测证据不满足要求而失败时，在后续新观测重试。状态和配置见[后端说明](../third_party/cordvip_geometry/README.md#遮挡恢复)。

检测、分割、注册和跟踪的后端异常直接传播，不转为无效观测，也不在合批失败后逐对象重跑。在线 Worker 因此退出，由会话现有故障路径停止设备并收尾；离线标注立即报错，已有完整 `hoi.h5` 保留。

在线任一必需对象无效时发布零点云和 `valid=False`，全部恢复后重新发布有效点云。感知 Worker 只在请求有效期间持续处理。部署控制层遇到临时 HOI 失效会保持机器人并保留当前请求，待感知恢复后重建策略上下文；恢复期限和停止条件见[策略部署的 HOI 暂停与恢复](policy_deployment.md#hoi-暂停与恢复)。

HOI 的 `joint_state`、`eef_pose`、指尖及手点云共用插值到相机来源时间的关节。插值要求前后反馈间隔不超过 `execution.hoi_max_joint_gap_s`，默认 0.10 秒；精确命中直接取样，禁止外推和跨连续区间插值。

在线每段按观测所需设备建立反馈历史，关节对齐暂不可用时进入 HOI 暂停与恢复。离线导出要求各原生连续段的所选行均能对齐。控制检查与 Raw 保留最新实测反馈。见[数据时间与导出规则](data.md)。

## 标注与查看

独立 `episode/hoi.h5` 完成逐帧标注并关闭后原子发布，`data.h5` 与 `rgb.mp4` 只读。默认跳过已有文件，`--overwrite` 显式重算；失败保留原文件并清理本次临时文件。遮挡与恢复期间保留原始行和时间戳，并记录无效状态。

| 内容 | 路径或位置 |
| --- | --- |
| 策略使用的基座系物体点云 | `hoi.h5:/object_pointcloud` |
| 物体局部到相机的位姿 | `hoi.h5:/object_pose_camera` |
| 逐对象有效性，bool `[T,O]` | `hoi.h5:/object_valid` |
| 逐帧、逐对象质量指标 | `hoi.h5:/quality/` |
| 固定模板、对象顺序及点分段 | `hoi.h5:/template/` |
| 相机到基座的变换 | `hoi.h5:/base_from_camera` |
| 实际跟踪配置 | `hoi.h5` 根属性 `tracking_config` |

`quality` 组始终存在，可以为空。缺失或为 `None` 的指标保存为 NaN；纯深度路径未计算的 Mask 指标为 −1，可视化均显示为曲线缺口。指标含义见[后端说明](../third_party/cordvip_geometry/README.md#质量检查)。

无效对象的位姿为 NaN；只有全部对象有效时才生成模板点云，否则该帧物体点云全零。整帧有效性由 `object_valid.all(axis=1)` 决定。

导出 `object_pointcloud` 要求所选 Raw 的每帧物体均有效。完整性检查及分段规则见[按需导出](data.md#按需导出)。

导出完成时，若接纳的采集具有相同的有序局部点与 offsets，则计算一次 `geometry_sha256` 供接触监督训练和续训使用。混用合法模板时记录 `null`。手模板的 link 名称须有效映射到 FK。

[visualize_episode.py](../scripts/visualize_episode.py) 使用相同插值规则。HOI 查看时，同一批时间行供时间轴、关节对齐和来源时差使用，逐物体质量和有效性按列发送。图像和几何逐帧显示。无效物体点云和不可对齐的手几何隐藏，保留对应行的时间位置。

可视化启动前要求展示范围内的 `observation_ns` 严格递增。查看 HOI 时同时校验控制时钟。

- `--max-frames N` 只加载展示前缀；关节插值可读取同一原生段内所需的邻点。
- `--no-hoi`（Python 参数 `hoi=False`）完全绕过标注，手部几何使用 Raw 关节；与 `--info` 组合也不打开 HOI 文件。

## 代码入口与参考

| 模块 | 职责 |
| --- | --- |
| [cordvip_geometry](../third_party/cordvip_geometry/) | 分割、重建、尺度调整和 RGB-D 位姿跟踪 |
| [geometry/hoi.py](../dexmani_real/geometry/hoi.py) | 任务资产、固定采样及点云坐标变换 |
| [perception/hoi_worker.py](../dexmani_real/perception/hoi_worker.py) | 在线物体点云生成 |
| [perception/hand_cloud.py](../dexmani_real/perception/hand_cloud.py) | 手部模板加载与 FK 几何 |
| [utils/timing.py](../dexmani_real/utils/timing.py) | 时钟调度、来源时效及离线批量关节插值 |
| [data/hoi.py](../dexmani_real/data/hoi.py)、[data/export.py](../dexmani_real/data/export.py) | 标注发布、区间对齐与训练数据导出 |

实现借鉴 [GALATEA `36b90481` 的遮挡感知轮廓评估](https://github.com/boyuan-an/GALATEA/blob/36b90481ba458d9beb5b6ac1396c639ea49a4747/recon/galatea_recon/pipeline/object_quality.py) 和固定表面点采样思路，以实测 RGB-D 和机器人 FK 为依据。

执行链路见[策略部署](policy_deployment.md)。返回[项目首页](../README.md)。
