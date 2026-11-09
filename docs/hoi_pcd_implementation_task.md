# HOI 物体点云五阶段实施任务书

> **状态：** 五阶段初版接口已接入，部分 CPU/Mock 与短序列 GPU 链路已验证；本任务书的完整验收清单尚未全部完成，尤其未通过真机验证。实际测试范围见 object_reconstruction.md。
>
> **仓库：** `haoyangzhanglab/dexmani_real`；配套训练仓库 `haoyangzhanglab/dexmani_policy`。
>
> **阅读约定：** 下列清单保留验收目标，勾选状态不代表实现进度；当前行为与实测范围见 [object_reconstruction.md](object_reconstruction.md)。
>
> **总体目标：** 最少人工介入，利用已有 CordVIP 几何能力，让任务完整物体点云经历 **资产构建 → 任务点数分配 → Raw 离线增强/现有可视化 → Zarr 导出 → 真机实时推理** 五个独立阶段。优先正确性、可复现性、已有接口复用和少量清晰代码。

## 0. 最终约束与非目标

1. **严格保留五个阶段**，尤其阶段 3 和阶段 4 必须分开；HOI 数据必须能在不导出 Zarr 的情况下使用现有 `scripts/visualize_episode.py` 查看。
2. **不创建 Episode 外的标注目录或 Sidecar，不新建可视化脚本/窗口。** HOI 数据离线写入原 Episode 的 `data.h5`；采集时的 `RawWriter` 和 `rgb.mp4` 不变。
3. 在 `data.h5` 使用 **HDF5 内部 `/hoi` 分组**存储派生字段（这是文件内部 Group，**不是文件系统新文件夹**）。对外统一提供可选字段别名 `object_pointcloud`；不能破坏旧 Raw 的字段枚举与读取逻辑。
4. 策略输入固定为 **`object_pointcloud: float32[1024,3]`，单位米、xArm base frame**。各物体在同一任务内按固定顺序拼接；点索引、模板版本在训练与推理间必须完全一致。
5. 不改变遥操作时的传感器采集内容、机器人命令或安全约束；HOI 推理必须在独立 Worker 中运行，不能阻塞控制线程。
6. 旧 Episode 没有 `/hoi` 时，`visualize_episode.py` 完全保持现有行为；仅显式导出 `object_pointcloud` 才要求 HOI 数据。存在不完整/不合法 `/hoi` 时不得静默跳过；RawReader 统一要求其 `task_name` 与 Raw `/meta` 的 `task_label` 属性一致，缺少标签时报错。
7. 保留现有 `Grounding DINO + SAM2.1 + Hunyuan3D-2.1 + FoundationPose` 实现为默认基线。**不把 FoundationPose++、TensorRT、新的网络结构、SDF、非刚体跟踪引入首版必需依赖。**
8. 优先遵循根目录 `AGENTS.md`：少抽象、无额外框架、少新文件、错误透明、仅针对真实风险增加校验；修改前检查已有未提交改动。

### 已核实的现有代码基础

| 事实 | 实际入口 |
|---|---|
| 单图重建、米制资产与三角面采样已有 | `third_party/cordvip_geometry/src/cordvip_geometry/{reconstruction,assets,geometry}.py` |
| 多物体 Mask、RGB-D 实测点、FoundationPose 位姿、深度一致性检查已有 | `cordvip_geometry/{sam2,sam2_worker,tracking,model_depth,foundationpose}.py` |
| 离线 Raw 读取已有，`RawWriter` 不保存派生场景点云 | `dexmani_real/data/{raw,recorder}.py` |
| Raw Viewer 已使用 Rerun，在 `Hand Point Cloud` 标签显示手部 | `scripts/visualize_episode.py` + `dexmani_real/utils/visualization/rerun.py` |
| `RawReader.fields` 枚举根层 Dataset，并在 `/hoi` 完整合法时提供派生字段别名 | `dexmani_real/data/raw.py` |
| Zarr 默认保持原有 13 个字段，`object_pointcloud` 仅显式导出 | `dexmani_real/data/export.py` + `scripts/export_zarr.py` |
| `dexmani_policy` 支持 `object_pointcloud` 和 CordViP `pc_dim: 3`；Real 已通过 HOI Worker 生产该字段 | `dexmani_policy/configs/cordvip.yaml`、`dexmani_real/deployment/observation.py` |
| 现有重建 Mesh 被 `.gitignore` 排除，GitHub 克隆后可能缺失 | 根目录 `.gitignore`；模板 `asset.json` 仍引用 `mesh/mesh_m.obj` |

## 1. 数据布局

### 1.1 任务级资产

资产路径沿用 `assets/pointcloud_template/object/<task>/`：

```text
<task>/
  task.json                # 新增：用户输入（物体顺序可由对象配置明示）
  apple/
    reference.jpg
    asset.json             # 复用现有格式
    mesh/mesh_m.obj        # 必须存在于运行环境
    points.npy             # 物体局部米制 (1024,3)
    normals.npy            # 与点严格对齐
  banana/ ...
  bowl/ ...
  hoi/
    template.npz           # points_local, offsets
    manifest.json          # task_name, object_names, template_sha256
```

`template.npz` 存的是**多个物体各自局部坐标系的点**；不得把未变换的点直接当作统一坐标系下的场景点云。

`manifest.json` 为唯一任务级资产清单；**不要**再生成多个功能重叠的 `allocation.json`/`metadata.json`/`recipe.json`。`asset.json` 继续保留，因为现有加载器依赖它。

### 1.2 Raw 增强字段（物理文件不变）

一个 Episode 仍只有原有 `data.h5` 和 `rgb.mp4`。在 `data.h5` 内新增如下 HDF5 Group：

| HDF5 字段 | dtype / shape | 语义 |
|---|---|---|
| `/hoi/object_pointcloud` | `float32[T,1024,3]` | 完整物体点云，**xArm base frame、米** |
| `/hoi/object_pointcloud_valid` | `bool[T]` | 同一帧全部必需物体的模型点与位姿是否有效 |
| `/hoi/object_pose_camera` | `float32[T,M,4,4]` | 每个物体的原始 Mesh object→color camera 位姿；无效为 NaN |
| `/hoi/object_pose_valid` | `bool[T,M]` | 各物体位姿是否有效 |
| `/hoi/model_supported_fraction` | `float32[T,M]` | 现有模型深度一致性支持比例，**不是校准置信度** |

`T=RawReader.num_frames`，`M=manifest 中对象数量`。Group 属性为 `task_name`、`object_names`、`template_sha256`、`offsets`、`complete`。单位、坐标系和点数采用上述固定接口，不再传递自定义 schema、来源证明或另一份 recipe。相机内外参和原始时间戳以已保存的 Raw `meta` 和 `timing/rows` 为权威来源，不重复写一套时钟。

**无效帧：** `object_pointcloud` 写有限的全零占位，`object_pointcloud_valid=False`，对应无效物体位姿写 NaN 并明确 `object_pose_valid=False`。**占位数据绝不能用于可视化或训练。** `valid=True` 的帧必须所有坐标有限、全部必需物体位姿有效。采用明确的布尔掩码而非靠是否存在零值推断有效性。

**RawReader 接口：** 只在 `/hoi` Group 完整且校验通过时，让 `"object_pointcloud" in reader.fields` 成立；`reader["object_pointcloud"]` 读取 `/hoi/object_pointcloud`，并提供可选 validity/pose 访问。旧 Raw 缺少 `/hoi` 依然完全可用。不要更改 `RawWriter` 的字段写入清单。

### 1.3 统一坐标变换

对物体 `i`，固定点为 `P_i^O`，FoundationPose 估计 `T_CO_i(t)`，记录的外参为 `T_BC`：

```text
T_BO_i(t) = T_BC @ T_CO_i(t)
P_i^B(t)  = P_i^O @ R_BO_i(t).T + t_BO_i(t)    # 行向量点数组
P_task^B(t) = concatenate(P_1^B(t), ..., P_M^B(t))  # (1024,3)
```

将 **这一个纯几何函数**放在适当的共享模块（可选 `dexmani_real/geometry/hoi.py`），并在阶段 3 的 HDF5 生成及阶段 5 的 HOI Worker 中复用。固定索引不重采样、不均值中心化、不混用 camera/base frame。对于 FoundationPose 内部 Mesh 居中，确保输出 `T_CO` 是针对原始 Mesh 坐标系的位姿；用**非零原点偏移的非对称 Mesh**单测验证。

### 1.4 同一模板保证

一个 `template_sha256` 覆盖任务对象顺序、冻结模板文件和实际跟踪 Mesh；尺度已应用到米制几何中。Raw `/hoi`、Zarr 与 Checkpoint 共用 `task_name`、`object_names`、`template_sha256`，不再建立逐文件哈希或额外版本清单。模板加载时要求 Manifest 的 `task_name` 等于任务目录名。启动时校验任务和模板，再把已加载模板传给 Worker；失配、缺 Mesh、非法尺度均在运动开始前报错。不可在线自动重新重建替换资产。

## 2. 五阶段具体实施

### 阶段 1 — 图片到单物体资产

**入口建议：** 新增 `scripts/cordvip/build_task.py --task <task>`，只负责这一阶段；复用 `reconstruct_object.py`、`cordvip_geometry.cli.object`、`assets.prepare_asset`，不复制 Hunyuan3D 或 SAM2 算法。

**输入：** 物体子目录下 `reference.jpg`，可选任务 `task.json`（提示词、明确的尺度标定参数；已有合法 `asset.json` 可复用）。共享任务读取入口将缺省 Prompt 补为对象名，并拒绝非字符串、空白/标点内容和按检测器匹配规则等价的 Prompt；保留合法原文及顺序。不能凭图片自动假定公制尺度。

**完成条件：**
- [ ] 自动发现任务物体，直接复用已发布的合法资产；重复运行幂等且不依赖模型权重。图片和构建参数仅用于新资产，已发布资产不另建构建指纹或自动替换。
- [ ] 同一图像的分割候选无歧义时自动选择；歧义时保存诊断并显式要求少量人工确认，不静默选择错误物体。
- [ ] 只接受实测长度及其明确对应的 Mesh 测量方式；首版优先沿用已有 `scale_to_meters` 或 `size_axis + size_meters` 接口，不发明尚未实现的直径 API。
- [ ] 输出米制 Mesh、固定 1024 个表面点、法向和来源元数据；验证 Mesh 有效、尺度为正、数据有限且法向归一化。
- [ ] 资产发布后不能出现“有 points.npy、asset.json 却缺少用于跟踪的 Mesh”；若 Mesh 因 `.gitignore` 未同步，明确报错并给出资产同步要求。
- [ ] 无需在每个任务里重新安装或下载相同的模型权重；模型调用尽量复用已加载实例；不要为了减少加载而破坏当前 CUDA 环境隔离。

### 阶段 2 — 任务级固定 1024 点

**入口：** 在现有 `scripts/cordvip/allocate_points.py` 原地修改，不新增相同功能的新脚本。

**分配建议：** 新任务默认按 Mesh **表面积**配额，并对每物体保底 `64` 点，使用确定性最大余数取整；物体数多到超过预算则报错。面积比体积更贴近表面采样，也避免必须闭合 Mesh。既有已冻结任务的分配不应被静默改写。

```text
remaining = 1024 - 64 * M
quota_i  = 64 + remaining * area_i / sum(area_j)
N_i      = deterministic_largest_remainder(quota_i)
sum(N_i) == 1024
```

从每物体已有的 `points.npy (1024,3)` 中按确定性索引选取 `N_i` 个点（例如确定性 FPS；不新增依赖）。`template.npz` 只保存最终的 `points_local` 与分段 `offsets`；物体顺序在 `manifest.json`，不再保存未使用的法向、对象 ID 或源索引副本。单物体法向仍保留在原资产中。不复制 Mesh、图片或每物体 `asset.json` 到 `hoi/`。

**完成条件：**
- [ ] 严格恰好 1024 点、`float32`、所有点有限，物体顺序稳定。
- [ ] 同一输入、种子与配置多次运行的模板和哈希逐字节一致。
- [ ] 已冻结模板不自动覆盖；Manifest 绑定实际 Mesh、物体顺序与固定模板。
- [ ] 不在逐帧跟踪、Zarr 导出或策略推理时再次 FPS。

### 阶段 3 — 离线增强 Raw Episode（含现有可视化）

**入口：** `scripts/cordvip/annotate_episodes.py --task <task>`；支持 `--episode <path>` 仅处理单段。任务自动扫描 `episodes/<task>/episode_*`，直接在各 Episode 的 `data.h5` 写入 `/hoi`。无额外文件夹、不修改 `rgb.mp4`。

**算法：** 复用 `cordvip_geometry.SceneTracker`、`iter_raw_rows`/`RGBDFrame`、SAM2、FoundationPose、模型深度验证。同帧一次 Grounding DINO 多类检测及共享 SAM2 图片特征；每个不同 Prompt 必须对应唯一候选，跨帧维持固定 object ID，不支持通过相同 Prompt 区分多个实例。

**处理与写入步骤：**
1. 只接收已经发布的 Raw Episode；读取原始 `rgb.mp4`、深度、相机内外参和 `timing/rows`。
2. 自动初始化每个任务物体；对候选歧义、遮挡或缺深度明确报告。不能把文本类别置信度当作模型位姿正确性的证据。
3. 优先沿用当前 SAM2 每帧 Mask 引导与 FoundationPose 深度一致性检查。**不能在改为 Mask-free tracking 时又继续假设每帧有真实语义 Mask 来计算支持比例。**
4. 要求 `camera_sequence`、`camera_ns` 为正值且不回退，相同序列的来源时间必须一致。对合法重复相机帧只做一次视觉跟踪，保留逐 Raw 行的结果与原始相机采集时间；手部 FK 仍可按每行实测关节独立计算。
5. 在跟踪丢失时标记无效，不复用旧位姿。非法 FoundationPose 输出的 SE(3) 校验失败归类为位姿丢失，模型执行和 IPC 故障原样传播；离线使用现有 Mask + `register` 做最多 3 次连续重注册，在线默认不重注册。
6. 在 RawReader 关闭后，以 `h5py.File(..., "r+")` 在 HDF5 **临时 staging Group** 中分块写点云和姿态；检查数组结构和输出有限性后发布为 `/hoi`。已有同模板完整标注由批处理入口跳过，写入函数直接拒绝覆盖。Raw 打开时仅检查完成状态和数组结构，数值在显示或导出时按需检查。不实现覆盖或迁移分支，不能将 HDF5 Group move 夸大为断电级原子事务。
7. 大规模 Episode 不应把全部 `T×1024×3` 点云常驻内存：先收集轻量位姿/状态，再按块生成和写入点云。
8. 终端按 Raw 行输出逐物体有效数、首次丢失行、状态数、最近失败原因和重注册耗尽情况；重复相机帧不重复跟踪。主机记录时间戳差仅统计双方均为正的时间对，输出 P50/P95/Max，无有效样本时显示不可用；不能当作硬件同步误差。不保存额外诊断文件。

**可视化仅修改现有入口：**
- `dexmani_real/utils/visualization/rerun.py`：在现有 **Hand Point Cloud** 标签页的 `Spatial3DView.contents` 加入 `"/object_pointcloud/**"`；保留标签页名称与其他布局。
- `scripts/visualize_episode.py`：使用同一个 RawReader，有字段则读取 `/hoi/object_pointcloud` 和 `/hoi/object_pointcloud_valid`；按 HDF5 的 `offsets` 给物体着色，调用已有 `log_points("object_pointcloud", ...)`。
- 如果没有 `/hoi`，**完全跳过** HOI 绘制，旧视窗与旧功能不变；若当前帧 valid=False，`log_points("object_pointcloud", None)` 清除上一帧点；不对无效占位点绘制。
- 保持原命令 `python scripts/visualize_episode.py episodes/<task>/<episode>` 与 `--info` 可用；无需 `view_hoi.py` 或新的 Rerun 窗口。可选 `--no-point-cloud` 仍只控制原有场景点云，不应误伤手部与 HOI。

**验收：**
- [ ] 旧 Raw 无 `/hoi`，Visualizer、RawReader、默认 Zarr 导出不变。
- [ ] 新 Raw 在 **同一个 Hand Point Cloud 窗口** 同时展示手与物体（base frame），颜色明显区分多物体。
- [ ] 无效帧清除模型点，绝不显示原点占位；可按 Rerun 时间轴检查。
- [ ] Raw 中的原有 dataset、属性、RGB 视频、时钟和关节值不变。
- [ ] 模拟中断后 staging 不会被识别为正式 HOI；重新运行有明确恢复路径。

### 阶段 4 — 从 Raw 直接导出 Zarr

**入口：** 原地扩展 `dexmani_real/data/export.py` 和 `scripts/export_zarr.py`，不新增单独的 CordViP Zarr Writer。训练不重新执行 Hunyuan3D/SAM2/FoundationPose，也不重新从 Mesh 采样。

- [ ] 将 `object_pointcloud` 加入**显式可选**的 `--fields` 候选，并设置唯一 Raw 依赖 `/hoi/object_pointcloud`；训练数据形状 `float32[T,1024,3]`。
- [ ] **保留原来的 13 个默认导出字段。** 目前 `resolve_fields(None)` 会枚举 `FIELD_INPUTS`；新增可选项时必须单独固定旧默认列表，否则历史任务默认导出将被破坏。
- [ ] 通过 `RawReader` 检查 `/hoi` 完成状态和数组结构，块导出时检查 valid 与有限坐标；缺失或无效时明确指出 Episode/行及 `--exclude` 解决方式，**不将零占位导出训练**。
- [ ] 多 Episode 必须具有相同物体 Manifest 哈希、顺序、采样点和字段规范，否则拒绝合并；不要静默重新分配点数。
- [ ] HDF5 → Zarr 直接按块拷贝，**仅导出 HOI + state/action/hand 时不应为 HOI 重新解码 RGB 视频**。
- [ ] Zarr 的 `hoi_pcd` 属性直接保存三个模板字段：`task_name`、`object_names`、`template_sha256`。
- [ ] 保持已有 `hand_pointcloud` 的 FK、形状和训练 `pc_dim:3` 语义；`dexmani_policy` 当前 CordViP Dataset 已支持 `object_pointcloud`。
- [ ] 对所有任务样本继续用现有 `episode_ends` 管理训练边界；第一版可明确拒绝包含无效帧的整个 Episode，不在没有完整数据语义时悄悄拆分或插值。

**CLI（实现后目标）：**

```bash
python scripts/export_zarr.py episodes/pnp_apple_banana \
  --output datasets/pnp_apple_banana.cordvip.zarr \
  --fields joint_state action hand_pointcloud object_pointcloud
```

需配合检查 `dexmani_policy/configs/cordvip.yaml` 的默认路径 `robot_data/<task>.cordvip.zarr` 与本地实际 `datasets/` 是否一致；训练时显式提供数据路径或沿用现有同步约定。

### 阶段 5 — `--enable_hoi_pcd` 实时策略观测

**入口：** 原地扩展 `scripts/run_policy.py`、`dexmani_real/deployment/{run,session,observation,preflight}.py`、`runtime/observation.py` 与 `ipc/sensors.py`；新增一个 `perception/hoi_worker.py` 是合理的独立职责。

- [ ] **只有当 Checkpoint 要求 `object_pointcloud` 且显式传入 `--enable_hoi_pcd` 时启动**。Checkpoint 要求但没开开关，或开关已开但模型不使用，均在设备运动前明确报错。
- [ ] 配套 `dexmani_policy` 的训练入口要求 `hoi_pcd` 的任务名非空且匹配训练任务、对象名称列表非空且无重复、Hash 为 64 位十六进制字符串、点云为 `float32[T,1024,3]`；出口只传递三个 Manifest 字段并将 Hash 统一为小写，复用现有 Checkpoint 机制。部署前校验任务与模板，模板数据只加载一次；缺失直接拒绝。
- [ ] 复用**同一现有 RealSense Camera Ring**。HOI 需求必须将 RGB+对齐深度、内参、米制深度尺度、外参加入相机启动依赖，即使模型不使用普通 `point_cloud`。
- [ ] 一个 HOI Worker 处理所有物体，加载 Mesh、任务模板和跟踪模型一次。只处理新的相机序列，优先最新帧，不把旧相机帧排队形成假新鲜点云。
- [ ] 每个 Episode 按 B 后重置 SAM2 跟踪状态和 FoundationPose 位姿（复用模型权重），从该轮新的 RGB-D 帧初始化全部必需对象，验证后**再允许动作执行**。HOI Warmup 在 `Rollout.prepare()` 获得本轮有效观测后执行；非 HOI 路径沿用启动时 Warmup。结束时发送 `episode_id=0`，Worker 暂停相机读取和跟踪但保留模型；下一轮正整数 ID 重新初始化。
- [ ] 工作中按共享函数变换模板生成 `float32[1024,3]` 并通过独立 Ring 发布；Ring 保留 **来源相机时间戳和 `source_camera_sequence`**，HOI 使用独立的 250 ms 默认 TTL，仅对使用该模态的模型校验配置。模型图像由同一 HOI 快照携带，Raw 独立读取最新相机；当前拒绝 HOI 与普通 `point_cloud` 同时输入。
- [ ] 在 `read_observation`/`check_observation` 纳入 HOI 的缺失、过期、无效位姿和 Worker 退出检查；关键观测失效后禁止继续提交动作，并使现有控制调度停止/取消剩余动作计划，不单纯跳过下一次预测。
- [ ] 默认采用已有 Mask 引导 FoundationPose 和现有模型深度匹配；如果将来测试无 Mask `track_one()`，必须同步设计新的质量校验方式，而不是保留依赖每帧 SAM2 Mask 的旧检查却假设它有效。
- [ ] `NVIDIA foundation-pose-inference-library` 作为**隔离 Benchmark 候选**，不作为首版必需依赖；若升级需验证显卡驱动/CUDA/TensorRT、Mesh 中心坐标、原始点云位姿及真实遮挡精度。其 tracking score 不可直接视为逐帧可信度。

**CLI（实现后目标）：**

```bash
python scripts/run_policy.py <checkpoint> --enable_hoi_pcd
```

## 3. 最小改动清单

| 目标 | 文件 | 动作 |
|---|---|---|
| 资产构建 | `scripts/cordvip/build_task.py` | 新增轻量任务编排，复用现有模型代码 |
| 点数分配 | `scripts/cordvip/allocate_points.py` | 原地修改，生成唯一任务模板与 Manifest |
| 离线标注 | `scripts/cordvip/annotate_episodes.py` | 新增任务批量处理 + `data.h5` 写入 |
| 统一几何 | `dexmani_real/geometry/hoi.py`（若确有必要） | 实现一个共享的任务模板读取与 SE(3) 点云变换 |
| 跟踪 | `cordvip_geometry/{tracking,sam2,sam2_worker,foundationpose}.py` | 最小必要的批量检测、重置、失效及恢复扩展 |
| Raw 读写 | `dexmani_real/data/raw.py` | 仅新增可选 `/hoi` 校验和读取别名；不修改 `RawWriter` |
| Rerun | `scripts/visualize_episode.py`、`dexmani_real/utils/visualization/rerun.py` | 原视窗自动叠加，旧 Raw 跳过 |
| Zarr | `dexmani_real/data/export.py`、`scripts/export_zarr.py` | 新增显式可选字段并保持旧默认行为 |
| 在线 | `dexmani_real/perception/hoi_worker.py` + 现有 IPC/Runtime/Deployment | 一个 Worker + Ring + 质量/时钟/安全检查 |
| Checkpoint | 配套 `dexmani_policy` 的 data/metadata 代码 | 传递模板信息，复用现有 `object_pointcloud` 模态 |

**不要新增：** Episode 外 `annotations/`、`poses.npz` Sidecar、`view_hoi.py`、新 GUI、新 Zarr Writer、复杂 PoseBackend Registry、ROS2 服务、无限增长的状态管理框架。只有实测证明收益时才增加可选后端接口。

## 4. 实施顺序与交付门槛

### P0 — 不涉及真机的正确性闭环

- [ ] 先实现统一任务资产加载/哈希与阶段 2 冻结模板；验证已有 Apple/Banana/Bowl 资产的 Mesh 是否能从当前机器找到。
- [ ] 阶段 3 先通过**合成已知位姿/小 Raw fixture**验证 HDF5 完整写入、旧格式兼容、可视化叠加和无效帧清空，再做真实离线 GPU 推理。
- [ ] 完成阶段 4 的显式字段导出，并使用 `dexmani_policy` `DemonstrationDataset`/CordViP 配置做兼容测试。
- [ ] 确认原有 Raw 的所有数据集、字段默认、可视化、Zarr 导出均未变化。

### P1 — 感知与真实离线数据质量

- [ ] 在 Apple/Banana/Bowl Raw 上完成自动检测、实例匹配、SAM2、FoundationPose 与质量门控。
- [ ] 评估初始检测率、Mask/位姿丢失、残差分布、人工介入次数、处理速度和重复相机帧节省量。
- [ ] 将实际正确性问题修复到当前跟踪核心，而不是用后处理掩盖。

### P2 — 真机启用与安全

- [ ] 不连接机器人时通过 Mock/Fake Camera Ring 验证全部 HOI 观测路径。
- [ ] 使用合法、缺少 Manifest 及模板失配的 Checkpoint 测试 `--enable_hoi_pcd` 的启用/拒绝条件。
- [ ] 验证每轮 Episode 重置、从按 B 后新帧初始化、失效时动作停止、Worker 意外退出和时间戳过期。
- [ ] 只有在得到真实硬件测试授权后运行真机；记录 GPU 显存峰值、模型加载时间、跟踪 P50/P95 延迟、端到端 Pose Age，**不得把上游 Benchmark 当作本机性能**。

## 5. 必须编写的测试（覆盖五阶段）

1. **几何：** 采用非对称、原点偏移的合成 Mesh 和已知 `T_BC,T_CO`；离线与在线生成的同一个 `(1024,3)` 点云逐点一致（例如 `atol=1e-6`，同 dtype）。
2. **资产：** 重复构建/分配得到完全相同的结果与哈希；点数恰好 1024，法向对应，缺 Mesh、尺度未知、对象顺序改变均得到明确错误。
3. **Raw HDF5：** 旧文件无 `/hoi` 可读可看；完整 `/hoi` 验证通过；损坏/截断 Group 不伪装为旧格式；中途失败不发布半成品；重跑已完成资产不会重复写入。
4. **时间：** 同 `camera_sequence` 多 Raw 行只跟踪一次；输出保持 `T` 行，时钟不被覆盖；陈旧 HOI 不因 Worker 重发变新鲜。
5. **Viewer：** `Hand Point Cloud` 包含手和多色物体；无字段时无 HOI 绘图；`valid=False` 时 `rr.Clear` 生效而不是显示零点。
6. **Zarr：** 默认 13 字段不变；仅显式启用 `object_pointcloud`；缺失/无效/不同 Manifest 的 Episode 拒绝导出；CordViP Dataset 能读 `[T,1024,3]`。
7. **部署：** 正确/错误 Checkpoint、错误模板哈希、无相机外参、未初始化、位姿丢失、过期、Worker 退出等边界均在测试中明确处理；既有非 HOI Policy 路径不回归。
8. **真实模型：** GPU/权重/实机不可用时只报告“未执行”，不以 Mock 测试宣称感知精度已验证。

按根目录 `AGENTS.md`，测试时禁止产生 `.pytest_cache/`、`.mypy_cache/`、`.ruff_cache/`，必要时用 `PYTHONDONTWRITEBYTECODE=1` 或关闭对应缓存。

## 6. 对应用户体验（目标命令，未实现前不承诺可用）

```bash
# 1 物体重建
python scripts/cordvip/build_task.py --task pnp_apple_banana

# 2 任务级点数分配
python scripts/cordvip/allocate_points.py \
  assets/pointcloud_template/object/pnp_apple_banana

# 遥操作照旧：仅记录 Raw，不执行 HOI 模型

# 3 对已经存在的 Raw Episodes 写入 /hoi
python scripts/cordvip/annotate_episodes.py --task pnp_apple_banana

# 3 在已有 Raw Viewer 的 Hand Point Cloud 视窗查看
python scripts/visualize_episode.py \
  episodes/pnp_apple_banana/episode_0001

# 4 独立导出训练 Zarr
python scripts/export_zarr.py episodes/pnp_apple_banana \
  --output datasets/pnp_apple_banana.cordvip.zarr \
  --fields joint_state action hand_pointcloud object_pointcloud

# 5 Checkpoint 绑定版本的在线 HOI 推理
python scripts/run_policy.py <checkpoint> --enable_hoi_pcd
```

## 7. 不实施的优化与后续独立 Benchmark

- 可在隔离实验中比较 **SAM2 Small vs Tiny**、现有 Mask 引导 vs 原版 FoundationPose `track_one()`、NVIDIA TensorRT FoundationPose；选型应以实际 Episode 的 ADD-S/有真值数据、深度残差、跟踪失效率、恢复能力、P95 延迟、显存为准。
- 当前位姿质量验证依赖 SAM2 分割像素；**不允许仅为加速删除 SAM2 后无声保留不再可信的同名质量分数。**
- 暂不实施完整 FoundationPose++、ROS2、动态模型重建、复杂 Proximity/SDF 网络、云端 Grounding API。
- 不为未知将来场景设计任意维度、任意物体数或复杂注册中心；目前任务固定总 1024 点即可。

## 8. 完成定义（Definition of Done）

- [ ] 五阶段命令及旧工作流均经针对性测试；未经 GPU/设备测试的能力明确标注限制。
- [ ] HOI 点云只在**原 `data.h5`**，没有额外 Episode 标注文件夹；Rerun 使用**现有 Hand Point Cloud** 视窗，旧 Episode 可原样查看。
- [ ] `object_pointcloud` 从任务模板和位姿生成，训练、Raw 可视化、在线使用同一坐标系/单位/物体顺序与模板哈希。
- [ ] Zarr 不执行 GPU 跟踪，默认导出字段不变，错误数据不能进入训练。
- [ ] `--enable_hoi_pcd` 只服务需要该字段的 Checkpoint，每轮初始化，时钟、质量、动作停止策略可验证。
- [ ] 交付时提供变更文件、测试结果、未测试内容、数据/模型依赖和剩余风险；改动严格遵守 `AGENTS.md` 与最少必要复杂度原则。

## 参考实现与文献

- [DexMani Real 项目架构](../README.md) · [现有物体重建说明](object_reconstruction.md)
- [CordVIP 几何模块](../third_party/cordvip_geometry/docs/usage.md) · [现有 Raw Viewer](../scripts/visualize_episode.py)
- [CordViP 论文](https://arxiv.org/abs/2502.08449) · [ProxiDex 论文](https://arxiv.org/abs/2609.16586)
- [Meta SAM2](https://github.com/facebookresearch/sam2) · [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) · [FoundationPose](https://github.com/NVlabs/FoundationPose)
- [FoundationPose++](https://github.com/teal024/FoundationPose-plus-plus) · [NVIDIA Foundation Pose Inference Library](https://github.com/nvidia-isaac/foundation-pose-inference-library)
