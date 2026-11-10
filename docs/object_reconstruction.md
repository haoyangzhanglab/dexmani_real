# 物体重建与点云资产

CordVIP 将物体图像与实测尺寸转换为米制网格和固定点云，用于几何分析与位姿跟踪。HOI 根据 Checkpoint 的 `observation_fields` 自动接入策略；普通示教采集和常规 Zarr 导出无需运行该流程。

## 数据流

**物体图像 → Grounding DINO／SAM2 分割 → Hunyuan3D 网格 → 实测尺度校准 → 米制资产 → 固定任务模板 → Raw／Zarr／策略观测**。

单图不可见几何由模型推断，真实尺度依赖外部测量。资产保留物体局部坐标系；位姿跟踪需要 Mesh。实验命令见 [scripts/cordvip](../scripts/cordvip/README.md)，几何接口与坐标约定见[核心文档](../third_party/cordvip_geometry/docs/usage.md)，已有资产位于 [object](../assets/pointcloud_template/object/)。

## HOI 五阶段流程

在仓库根目录、已配置感知依赖的 `real_robot` 环境中运行，完整离线命令见[实验入口](../scripts/cordvip/README.md#任务级-hoi-流程)。

| 阶段 | 入口与产物 |
| --- | --- |
| 1. 资产构建 | `build_task.py` 构建或复用各物体米制 Mesh、1024 点及法向。 |
| 2. 模板冻结 | `allocate_points.py` 按表面积从已有点云选固定索引，总计 1024 点、每物体至少 64 点。 |
| 3. Raw 标注 | `annotate_episodes.py` 写入原 `data.h5:/hoi`，用 `visualize_episode.py` 查看。 |
| 4. Zarr 导出 | `export_zarr.py` 显式选择 `object_pointcloud`；默认导出不包含该字段。 |
| 5. 训练与部署 | 训练显式设置 `zarr_path`，部署使用匹配的 Checkpoint，自动识别 HOI 需求。 |

阶段 4 的常用字段与阶段 5 的真机部署命令：

```bash
python scripts/export_zarr.py episodes/pnp_apple_banana \
  --output datasets/pnp_apple_banana.cordvip.zarr \
  --fields joint_state action hand_pointcloud object_pointcloud
python scripts/run_policy.py CHECKPOINT
```

### 任务配置

任务目录为 `assets/pointcloud_template/object/<task_name>/`，各物体占一个子目录。新资产默认读取 `reference.jpg`；已有合法资产直接复用，无需原图或模型权重，需带齐 `asset.json` 指向的 Mesh、点云和法向。更改照片或参数不会替换已发布资产，重建应使用新的资产目录。

`task.json` 示例，尺寸需替换为实测值：

```json
{"objects": [
  {"name": "apple", "prompt": "apple", "size_axis": "x", "size_meters": 0.08}
]}
```

`objects` 决定对象顺序，省略时按目录名排序；Prompt 缺省为对象名。Prompt 须有实际文本，按 `strip().lower().rstrip(".")` 比较时不得重复，场景首帧中各自须匹配唯一候选。`instance` 仅选择重建参考图候选，不能在在线场景中区分同 Prompt 的多个实例。尺度也可用 `scale_to_meters`，与轴尺寸二选一。

### 模板与数据契约

`hoi/template.npz` 保存物体局部坐标的 `points_local` 与对象分段 `offsets`，点序固定。`hoi/manifest.json` 的三个字段贯穿 Raw、Zarr 和 Checkpoint：

| 字段 | 约定 |
| --- | --- |
| `task_name` | 非空任务名，与资产目录、Raw `/meta` 的 `task_label` 和训练任务一致。 |
| `object_names` | 非空、无重复的对象名列表，顺序固定。 |
| `template_sha256` | 模板发布时生成的小写 SHA-256，绑定模板文件、物体顺序与实际跟踪 Mesh；训练原样保存。 |

Raw `/hoi` 用 HDF5 属性保存上述字段及 `offsets`；Zarr 与 Checkpoint 保存为 `hoi_pcd`。物体点云为 `float32[T,1024,3]`，单位米、坐标系为 xArm base。恢复训练与部署均拒绝任务、对象顺序或模板失配，HOI Checkpoint 缺少元数据也会拒绝部署；普通非 HOI 数据无需该属性。

已有模板校验后复用，已发布且模板一致的标注跳过，不覆盖正式结果。模板身份检查只读取冻结点、对象清单及 Mesh 文件；跟踪器启动时再加载并检查完整资产。

标注先写入 `_hoi_staging`，所有数组写完并 flush 后发布为 `/hoi`，分组名是唯一的发布标志。中断可留 staging，下次重算后发布，HDF5 group move 不保证断电事务。RawReader 只识别已发布的 `/hoi`，首次请求 HOI 字段或元数据时检查数组结构与任务身份并缓存结果；打开基础 Raw、枚举字段和请求普通字段不会触发 HOI 校验；数值在写入、显示及分块导出时校验。

无效物体帧用 false 掩码和零点云占位，Viewer 清除物体点，导出拒绝整个 Episode 并提示 `--exclude`。`--no-point-cloud` 仅关闭普通场景点云显示，HOI 与手部几何仍按各自标签展示。

### 离线标注与在线观测

离线标注跨 Episode 复用模型，每段重置场景。`camera_sequence`、`camera_ns` 须为正且不回退；同一序列必须保留相同时间，重复帧复用结果。每次丢失最多允许三次连续重注册，成功后清零；终端汇总逐物体有效行、首次丢失、末态、失败原因与重注册预算，并输出相机与关节来源时间差的 P50／P95／Max。时间差仅统计正时间戳，描述主机记录关系。

在线 Worker 每轮 B 后重新初始化，模型保留。每段首次 HOI 初始化预算为 120 s，后续观测等待为 1 s；真实观测预热完成且新鲜有效才允许动作。在线不自动重注册，初始化或位姿失败结束本轮，模型执行、CUDA、IPC 和配置错误继续传播。

HOI 使用来源相机时间及独立的 `execution.hoi_max_age_s`（默认 250 ms）；缺失、过期或失效立即停止动作计划。模型需要的 RGB／Depth 与 HOI 同源，Raw 另取最新 Camera 帧；不支持同时输入普通 `point_cloud`。其余历史、时效和录制约定见[策略部署](policy_deployment.md#来源时间与时效)。

Episode 结束发布 `episode_id=0`，Worker 暂停读取相机和跟踪；下一轮新 ID 重置场景，旧轮在途结果不能进入新观测。

## 当前验证范围

已有验证覆盖 CPU／Mock、离线 GPU 与独立相机采集：

- 历史 CPU／Mock 验证覆盖模板、Raw／导出、训练元数据、来源时效与调度、Worker 重置和错误传播；另完成单物体 Hunyuan 新建资产及缓存复用验证。
- RTX 4090 上，8 帧 Raw 副本通过三物体标注、四字段导出及实际 CordViP Dataset 读取；完整 `episode_20261101_165909` 的三物体均为 403/403 有效，离线约 79 s（含模型加载），另验证两轮 GPU 场景重置保留模型。
- 2026-11-09，`episode_20261101_173701` 在临时副本重跑两次，均为 Apple 146/403、Banana 402/403、Bowl 403/403。Apple 第 143 行支持点 26（门槛 30），144 行恢复；147 行再失效，三次重注册支持点为 23、14、0，151 行起预算耗尽。RGB 显示手部遮挡；Banana 第 369 行失效、370 行恢复。未放宽阈值或增加重试。
- 上述重跑时原始 Episode 无 `/hoi`、正式目录无冻结模板，仅在临时资产副本生成模板，无法核对历史 Hash；原始 HDF5 与视频 SHA256 前后相同，临时副本已清理。
- L515 独立 RGB-D 采集 30 s：RGB 30.024 Hz、深度 29.991 Hz，帧号无跳跃或回退；这未覆盖策略与 HOI 并发。

有效率表示通过当前门控，不是真值位姿精度。完整任务质量评估、Rerun GUI 检查、TensorRT FP32 对照、策略／感知 GPU 并发、五分钟运行、两轮实机初始化和硬件故障注入仍待完成；尚未执行机器人运动。

## 当前实现与性能结论

正式后端为 Grounded-SAM-2／SAM2.1 Small、Mask 引导 FoundationPose 两次跟踪细化及语义深度门控；DINO 只用于首帧。模型跨轮复用，场景逐轮重置，单姿态跟踪使用缓存补丁。完整 VOS 编译分支已移除，EdgeTAM、EfficientTAM 与 TensorRT FP 均未接入正式后端。

2026-11-09 本机对照：RTX 4090、三物体、上述两段各 403 帧，排除前 15 帧计时。耗时包含分割、位姿、通信和门控，排除文件解码与策略推理：

| 方案 | 第一段 P50/P95 | 两段全物体同时有效帧数 | 状态 |
| --- | --- | --- | --- |
| 当前 SAM2 + 当前 FP | 79.04/81.86 ms | 403 / 146 | 正式实现 |
| 完整 VOS + 当前 FP | 73.18/76.67 ms | 403 / 146 | 已移除 |
| 当前 SAM2 + TensorRT FP | 53.79/55.35 ms | 403 / 147 | 对照实验 |
| 完整 VOS + TensorRT FP | 47.20/49.12 ms | 403 / 146 | 对照实验 |
| EdgeTAM + TensorRT FP | 32.96/34.62 ms | 403 / 143 | 对照实验 |

VOS 约节省 6 ms，但需额外环境、约 223 s 冷编译及 CUDA Graph 处理，原速回放仍失效。EfficientTAM-S 512 在第二段苹果出现 252 帧空 Mask，未采用；失效后跳过物体的低耗时不能作为加速收益。

第一段按原始时间间隔回放 Camera Ring：403 帧约 25.2 s、Raw 约 16 Hz，包含自动初始化、HOI Worker 与固定点变换，每约 5 ms 采样最新结果：

| 跟踪实现 | 有效输出 | 新点云更新率 | 年龄 P50/P95 | 超过 133 ms 的采样比例 |
| --- | --- | --- | --- | --- |
| 当前 FP | 322/322 | 12.8 Hz | 150/200 ms | 69.5% |
| TensorRT FP 实验适配器 | 403/403 | 15.98 Hz | 91/120 ms | 0.46% |

HOI 默认 250 ms 基于回放 P95 约 200 ms 留余量，仍是暂定研究参数。回放没有真实设备、策略 GPU 并发或原相机 30 Hz 中未记录的中间帧；关节反馈仍取最新值，未做历史对齐或动力学补偿，不能据此认定连续真机执行可用。

TensorRT 对照使用 [NVIDIA 固定源码](https://github.com/nvidia-isaac/foundation-pose-inference-library/tree/89dd4cfec75f944b9fca52d9cb282725f1666c12) 与[固定 ONNX 权重](https://huggingface.co/nvidia/foundationpose/tree/18d8309afc9790cddc03a1d50bc69954dc058693)：CUDA 12.4、TensorRT 10.16.0.72、FP16、252 个候选、5 次注册／2 次跟踪细化、micro-batch 42、关闭 CUDA Graph。原生实现无 bbox XY 引导，tracking score 固定 1.0；对照沿用语义／深度门控，不以 score 判断可信度。

一次性对照脚本、结果副本、实验环境和临时日志已清理，仅保留结论及可复用源码／权重缓存。后续先完成真实相机与已训练策略的并发验收，再决定是否继续评估 TensorRT。

返回[项目架构](../README.md)。
