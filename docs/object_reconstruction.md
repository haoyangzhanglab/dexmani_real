# 物体重建与点云资产

## HOI 五阶段流程

米制 Mesh → 固定任务模板 → Raw 离线标注 → Zarr → CordViP 训练与部署。命令见[实验入口](../scripts/cordvip/README.md)。重建 GUI、模型选择与 `--rgbd` 输入仍可使用。

`build_task.py` 构建或复用米制 Mesh。`asset.json` 只要求 `mesh_file`、`units=m`、`frame=object`，最后写入表示发布完成。加载已发布 Mesh 保留其尺度和原点；任务点云由 `allocate_points.py` 直接采样。

`allocate_points.py` 保留面积配额：总计 1024 点，每物体至少 64 点；固定 seed=0，直接在各 Mesh 表面采样最终点集。

### 任务配置

任务目录为 `assets/pointcloud_template/object/<task_name>/`，各物体占一个子目录。新资产默认读取 `reference.jpg`；已有合法资产直接复用，无需原图或模型权重，只需带齐 `asset.json` 指向的米制 Mesh。更改照片或参数不会替换已发布资产，重建应使用新的资产目录。

`task.json` 示例，尺寸需替换为实测值：

```json
{"objects": [
  {"name": "apple", "prompt": "apple", "size_axis": "x", "size_meters": 0.08}
]}
```

`objects` 决定对象顺序，省略时按目录名排序；Prompt 缺省为对象名。Prompt 须有实际文本，按 `strip().lower().rstrip(".")` 比较时不得重复，场景首帧中各自须匹配唯一候选。`instance` 仅选择重建参考图候选，不能在在线场景中区分同 Prompt 的多个实例。尺度也可用 `scale_to_meters`，与轴尺寸二选一。

### 资产与模板

研究者负责固定实验资产。已有 Raw、Zarr、checkpoint、Mesh 和冻结模板均不自动覆盖或重新生成。

示例任务三个米制 Mesh 在本机存在但未纳入 Git；任务级冻结模板缺失。新 checkout 需要自行提供 Mesh 与实验对应的冻结模板。

模板要求显式的 object_names 分段映射，资产文件不由运行时修改。手模板只消费 XYZ 通道。

### 模板与数据契约

`hoi/template.npz` 包含 `points_local[1024,3]`、`offsets`、`object_names`。名字关联 Mesh、Prompt 与跟踪结果；拼接顺序始终由模板决定，配置列表可以重新排序。加载边界检查必要形状、有限值、分段及对象对应关系。

Raw `/hoi` 只包含 float32 的 `object_pointcloud[T,1024,3]`，点云以 xArm base 为坐标系、米为单位，坐标必须有限。全段跟踪成功后才写入并发布；正式点云存在且满足结构要求即表示该段已完成标注。中间结果保留在 `_hoi_staging`，写完并 flush 后发布；分组移动不保证断电事务。损坏 HOI 只阻止 HOI 消费者。

离线跨 Episode 复用模型并重置场景，按 camera_sequence 去重；同序列的来源时间必须一致。首次初始化或 pose 失效即报告 Episode、行号与原因，不发布该 Episode，继续下一段。文件、模型或程序错误直接传播。在线失效后等待新请求，旧 Episode 在途结果不能被新请求接受。

手点云使用唯一的冻结手模板与 FK，运行时及导出为 XYZ。部署从 info.task_name 定位任务；手点数由 RealConfig.hand_pointcloud 决定，不从 checkpoint 覆盖。

### 可选来源时间差预算

最新关节与滞后物体点云的组合仍是实验近似；时间差预算只限制这种近似，不代表物理同步，关闭它也不会关闭年龄或跟踪有效性检查。

CONFIG.execution.hoi_max_skew_s 默认 None，关闭额外来源时间差限制；物体点云与手几何或 joint_state 联合消费时，可显式填写有限正数秒值启用。导出使用同一个 `--config`，程序调用可直接设置 ExportConfig.hoi_max_skew_s。

在线使用 HOI 原始相机来源时间与臂、手反馈时间；离线使用 camera_ns、arm_ns、hand_ns。启用检查时三者 max−min 不得超过预算（等于预算允许），超限拒绝整次导出或停止本轮。在线各来源年龄检查继续保留，不以年龄预算替代时间差预算。主机记录时间差不证明物理同步，不进行插值、补洞或重采样。

## 当前实现与性能结论

以下为历史性能测量，不能视为本次精简后 GPU 或真机验证。

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
