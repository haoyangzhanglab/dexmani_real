# 物体重建与点云资产

物体重建属于独立的 CordVIP 离线几何流程，将单张物体图像和实测尺寸转换为米制网格及固定采样点云，用于物体几何分析与位姿跟踪。HOI 路径通过显式开关接入策略观测，常规采集不运行模型。

## 数据流

**物体图像 → 文本检测与分割 → 前景图像 → 网格生成 → 尺度校准 → 表面点云与法向**。

Grounding DINO 根据物体描述定位候选，SAM2 生成分割掩膜；确认前景后，Hunyuan3D 生成物体网格。网格依据实测尺寸转换为米制，再按表面面积采样点及对应法向，保留物体局部坐标系。

单图不可见部分由模型推断，真实尺度来自外部测量。输出资产包括网格、点云、法向和来源元数据；点云可独立用于几何分析，FoundationPose 位姿跟踪还需要网格。

## 核心模块

算法实现位于 [cordvip_geometry](../third_party/cordvip_geometry/)，主仓库通过 [scripts/cordvip](../scripts/cordvip/) 提供实验入口。

| 模块 | 职责 |
| --- | --- |
| [reconstruct_object.py](../scripts/cordvip/reconstruct_object.py) | 串联图像分割、网格重建与资产准备。 |
| [allocate_points.py](../scripts/cordvip/allocate_points.py) | 为多物体资产分配固定的总采样点数。 |
| [track_object.py](../scripts/cordvip/track_object.py) | 结合物体资产与 Raw RGB-D 数据进行位姿跟踪和离线几何分析。 |

## 与主项目的关系

机器人与手部几何由主包的 `geometry`、`perception` 提供，物体几何由 CordVIP 离线生成。两者可在离线分析中结合，但物体重建不是示教录制或常规 Zarr 导出的前置步骤。

已有物体模板存放于 [assets/pointcloud_template/object](../assets/pointcloud_template/object/)。

返回[项目架构](../README.md)。


## HOI 五阶段流程

在 `real_robot` 环境、仓库根目录运行。已有完整资产优先复用；缺 Mesh 时同步原始资产，不从点云伪造 Mesh。

```bash
# 1 单物体资产；读取各物体 reference.jpg 和可选 task.json
python scripts/cordvip/build_task.py --task pnp_apple_banana
# 2 表面积分配：每对象至少 64 点，总计 1024，冻结 hoi/ 下两个文件
python scripts/cordvip/allocate_points.py assets/pointcloud_template/object/pnp_apple_banana
# 3 写入原 data.h5 内 /hoi；也支持 --episode 单段路径
python scripts/cordvip/annotate_episodes.py --task pnp_apple_banana
python scripts/visualize_episode.py episodes/pnp_apple_banana/episode_20261101_165909
# 4 只读 Raw，显式导出 HOI；默认 13 个字段保持不变
python scripts/export_zarr.py episodes/pnp_apple_banana --output datasets/pnp_apple_banana.cordvip.zarr --fields joint_state action hand_pointcloud object_pointcloud
# 5 真机命令；执行运动前须取得硬件测试授权
python scripts/run_policy.py CHECKPOINT --enable_hoi_pcd
```

新资产的 `task.json` 示例（尺寸必须替换为实际测量值）：

```json
{"objects": [{"name": "apple", "prompt": "apple", "size_axis": "x", "size_meters": 0.08}]}
```

对象列表决定顺序；未提供时按目录名排序。尺度也可使用明确的 `scale_to_meters`，不能与轴尺寸混用。单图多个候选时检查保留的 `.build-*` 诊断并在对象配置中明确 `instance`；不自动选择最高检测分数。`asset.json` 及其合法 Mesh/点云是阶段 1 的复用依据，无需原图片或模型权重，也不另建构建指纹。图片和构建参数只用于尚未发布的资产；修改输入不会替换已发布资产，重新构建应使用新的资产目录。

`template.npz` 只保存 `points_local` 和 `offsets`；点顺序由数组行固定，不再保存未使用的法向、对象 ID 和源索引副本。`manifest.json` 只保存 `task_name`、`object_names`、`template_sha256`，一个指纹覆盖模板文件、物体顺序和实际跟踪 Mesh。尺度已应用到米制 Mesh 和点；固定的 1024 点、米制和 xArm base 是代码接口，不再作为配置或 schema 版本反复传递。

Raw `/hoi` 用原生 HDF5 属性保存上述三个字段和 `offsets`；Zarr 与 Checkpoint 直接沿用这三个字段作为 `hoi_pcd`，没有额外 recipe、来源证明或逐文件校验清单。启动时校验一次并把已加载模板交给 Worker。已有正式模板和标注不提供覆盖开关，不保留旧 HOI 派生格式的迁移分支。阶段 3 跨 Episode 复用模型，每段重置跟踪状态；中断结果只留在 HDF5 `_hoi_staging` 中，下次重算后发布。HDF5 group move 不提供断电级事务保证。

无效帧为显式 false 掩码和零点占位，Viewer 清除物体点，导出拒绝整个 Episode 并提示 `--exclude`。Raw 打开时只检查完成状态和数组结构；点云数值在写入、逐帧显示和分块导出时检查，不再全量扫描未使用的数据。原 Hand Point Cloud 标签按固定点分段着色；`--no-point-cloud` 只关闭普通场景点云。Raw 无 `/hoi` 保持原有读取路径；不完整 `/hoi` 报错。更早期缺少 `timing/rows` 的 Raw 不在本次迁移范围。

Policy 训练需把 `zarr_path` 显式指向上述导出路径（其默认路径是 `robot_data/`）。训练读取 Zarr 的 `hoi_pcd` 到 Checkpoint，恢复训练拒绝不同模板。HOI Checkpoint 缺少模板信息时拒绝真机部署。在线每轮 B 后重新初始化，真实观测 warmup 完成且新鲜有效才允许动作。HOI 缺失、过期或 Worker 故障终止当前动作计划。只有模型需要 RGB/深度时，HOI Ring 才携带相应的同源图像；安全检查不复制图像。Raw 独立读取最新 Camera Ring，其图像帧可以不同于模型 HOI 来源帧。首版不支持同时使用普通 `point_cloud`。普通 Camera / Point Cloud TTL 沿用现有配置；HOI 使用 `execution.hoi_max_age_s`，默认 250 ms，时间戳仍继承来源相机帧，不能把发布时刻当作采集时刻。

候选缺失或歧义、首帧空 mask／深度不足以及位姿门控失败会结束当前轮；Worker 记录原因并停止该轮视觉处理，保留模型，用户可重新按 B 初始化。已发布无效结果时不继续等待；每轮首次 HOI 初始化保留 120 秒预算，后续新鲜快照等待使用现有 1 秒预算。模型、CUDA、IPC 等故障仍通过原故障路径退出。

### 当前验证范围

- CPU/Mock 测试覆盖冻结模板确定性、偏移几何变换、Raw staging 与无效帧、显式导出、Viewer Clear、启用条件、来源 TTL、三种调度模式清空计划和 Worker 几何一致性；另覆盖无模型依赖的资产复用、批量标注共享模型、初始化期间切换 Episode、Worker 故障停止及按需传输同源图像。当前回归另覆盖预期初始化失败后的跨轮重试、独立 HOI 年龄预算、Raw 最新帧写入及动作映射期间的失效检查。
- RTX 4090：8 帧真实 Raw 副本完成三物体跟踪、内部 `/hoi` 写入、四字段 Zarr 导出、实际 CordViP Dataset 读取及重复标注跳过；另验证两轮 GPU 场景重置保留模型实例。完整 `episode_20261101_165909` 副本的 403 帧增强也已通过，三个物体均为 403/403 有效，离线总耗时约 79 秒（含模型加载）。
- 已验证既有资产复用及一个物体的真实 Hunyuan 新建资产 GPU 流程（1024 点和法向、重复运行缓存命中）。未执行完整任务数据质量评估、Rerun GUI 人工检查、策略与感知并发性能或真机运动测试。8 帧通过不代表遮挡场景精度或真机安全性能已证明。
- GPU 运行期间上游 Warp 输出驱动 API 告警，但上述实验正常完成；不以此宣称其他 CUDA 扩展路径均可用。

测试：`PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/test_hoi.py tests/test_freshness.py -q`。测试写入使用临时目录，不修改历史 Raw、冻结资产或机器人配置。

### 当前实现与性能结论

正式链路使用现有 Grounded-SAM-2 / SAM2.1 Small、Mask 引导 FoundationPose 两次细化和语义深度门控。DINO 只用于每轮首帧；模型权重跨轮复用，场景状态每轮清空。`foundationpose_tracking_cache.patch` 保留单姿态跟踪的 CUDA allocator 缓存，多候选注册仍释放缓存；安装脚本幂等应用，冲突直接报错。

清理后不保留完整 VOS 编译分支及其 CLI 参数：它的整帧收益约 6 ms，但需要额外 Torch 环境、约 223 秒冷编译和 CUDA Graph 张量处理，原速回放仍出现失效。EdgeTAM、EfficientTAM 和 TensorRT FP 均未接入正式后端；不增加必需依赖，也不自动切换模型。

以下是 2026-11-09 清理前的本机实验结论。RTX 4090、三物体、两个 Episode 各 403 帧（`20261101_165909`、`20261101_173701`），计时排除前 15 帧，包含分割、位姿、通信和质量门控，不包含文件解码与策略推理：

| 方案 | 第一段 P50/P95 | 两段全部物体有效帧数 | 当前状态 |
| --- | --- | --- | --- |
| 当前 SAM2 + 当前 FP | 79.04/81.86 ms | 403 / 146 | 正式实现 |
| 完整 VOS + 当前 FP | 73.18/76.67 ms | 403 / 146 | 实验分支已移除 |
| 当前 SAM2 + NVIDIA TensorRT FP | 53.79/55.35 ms | 403 / 147 | 仅 Benchmark |
| 完整 VOS + TensorRT FP | 47.20/49.12 ms | 403 / 146 | 仅 Benchmark |
| EdgeTAM + TensorRT FP | 32.96/34.62 ms | 403 / 143 | 仅 Benchmark |

有效帧数只表示通过当前门控，不是 6D 真值精度；失效后跳过物体的低耗时不能作为加速收益。EfficientTAM-S 512 在第二段苹果出现 252 帧空 Mask，未采用。后续优先验证保留当前 SAM2 的 TensorRT FP 路线。

原速 Camera Ring 回放使用第一段原始时间间隔，包含自动初始化、正式 HOI Worker 和固定点云变换；403 帧约 25.2 秒，Raw 本身约 16 Hz。每约 5 ms 采样最新结果：

| 跟踪实现 | 有效输出 | 新点云更新率 | 最新点云年龄 P50/P95 | 超过 133 ms 的采样比例 |
| --- | --- | --- | --- | --- |
| 当前 FP | 322/322 | 12.8 Hz | 150/200 ms | 69.5% |
| TensorRT FP 实验适配器 | 403/403 | 15.98 Hz | 91/120 ms | 0.46% |

HOI 默认 250 ms 依据上述回放 P95 约 200 ms 留出余量，是暂定研究参数，不代表真机安全保证。

回放没有策略 GPU 并发或真实设备，也缺少原相机 30 Hz 中未被记录的中间帧。正式策略仍在过期或失效时清空动作计划并停止，不能据此宣称满足连续真机执行要求。来源相机时钟用于 TTL；关节反馈目前仍为最新值，未做历史对齐或动力学补偿。

TensorRT 对照使用 [NVIDIA 固定版本](https://github.com/nvidia-isaac/foundation-pose-inference-library/tree/89dd4cfec75f944b9fca52d9cb282725f1666c12) 和[固定 ONNX 权重](https://huggingface.co/nvidia/foundationpose/tree/18d8309afc9790cddc03a1d50bc69954dc058693)，CUDA 12.4、TensorRT 10.16.0.72、FP16、252 个候选、5 次注册细化、2 次跟踪细化、micro-batch 42、关闭 CUDA Graph。原生实现没有 bbox XY 引导，tracking score 固定为 1.0；本次使用原语义/深度门控，不把 score 当可信度。尚未完成 FP32、遮挡真值精度和策略并发验证。

一次性对照脚本、逐帧结果副本、编译实验环境及临时日志已清理，不作为项目接口。保留上述结论和可复用的 NVIDIA 源码、库与权重缓存；正式命令仅使用仓库内现有入口。

### 改动入口

| 职责 | 文件 |
| --- | --- |
| 新入口 | `scripts/cordvip/build_task.py`、`scripts/cordvip/annotate_episodes.py` |
| 模板与 HDF5 读写 | `dexmani_real/geometry/hoi.py`、`dexmani_real/data/hoi.py`；扩展 `data/raw.py` |
| 分配与导出 | 原地修改 `allocate_points.py`、`data/export.py`、`scripts/export_zarr.py` |
| 原 Viewer | `scripts/visualize_episode.py`、`utils/visualization/rerun.py` |
| 在线 | 新增 `perception/hoi_worker.py`；扩展 IPC、Runtime observation、Deployment observation/preflight/session/rollout/run 和 CLI |
| 跟踪核心 | 现有 `tracking.py`、`sam2.py`、`sam2_worker.py`、`foundationpose.py` |
| 配套 Policy | `training/common/data.py`、`deployment/metadata.py`、`training/trainer.py`，保留已有用户改动 |
| 回归测试 | `tests/test_hoi.py` |
