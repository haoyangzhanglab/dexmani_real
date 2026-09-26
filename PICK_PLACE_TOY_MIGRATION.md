# pick_place_toy 数据审计与迁移适配方案

## 1. 文档范围与结论

本文件汇总 `episodes/pick_place_toy` 的离线检查结果、历史格式与当前代码的语义差异，以及后续迁移的实施条件。

- 检查日期：2026-09-26。
- 当前代码基线：`436f8ba7dd1e5b70762e6ca0789bb226d9557138`。
- 数据范围：61 个 episode，共 14,309 帧。
- 当前 raw schema：v33；当前 canonical Policy Zarr schema：v15。两者是不同的版本体系。
- 工作状态：已完成格式审计和迁移方案分析；尚未实施迁移、修改源数据或生成训练缓存。
- 检查方式：只读离线检查，没有连接或操作机器人、手部和相机。

**现有数据全部声明为 raw v30，不能被当前读取器直接接受。** 当前导出程序的 `--dry-run` 已验证：61 段全部因版本不匹配被拒绝，最终返回 `export produced no included episodes`。

迁移应区分两个目标：

| 目标 | 当前判断 |
| --- | --- |
| 完整迁入 raw v33，供当前原始数据工作流使用 | 现有文件缺少真实动作发布时间等信息，暂不能无损完成 |
| 保留旧 raw，生成符合当前训练语义的完整 Zarr | 有条件可行；必须先证明历史观测—动作对应、动作目标和模态语义，不能绕过证据缺失 |

建议优先评估第二条路径。训练缓存不保存运行时时间戳，不代表可以省略对原始数据时序正确性的审计。若无法证明训练所需的语义，也不能将数据标记为当前 canonical Zarr。

## 2. 已完成的检查

### 2.1 文件与数值完整性

| 项目 | 结果 |
| --- | --- |
| Episode 数量及帧数 | 61 段，14,309 帧；每段 165–331 帧 |
| 必需文件 | 每段均存在 `data.h5`、`depth.h5`、`rgb.mp4` |
| RGB | 全部 14,309 帧解码成功，尺寸为 640×480 |
| Depth | 全部帧读取成功，尺寸为 640×480，dtype 为 `uint16`，无全零深度帧 |
| 行数一致性 | 控制数组、深度数组和实际解码 RGB 帧数均与 `num_frames` 一致 |
| 共有字段 | 与 v33 同名字段的形状和 dtype 均匹配 |
| 相机几何 | 元数据通过当前相机模型和刚体变换校验 |
| 深度尺度 | 全部为约 `0.00025 m/unit`，不可按毫米直接解释原始数值 |
| 浮点有限性 | 除第 2.3 节列出的无效触觉行外，扫描到的其他浮点数组没有 NaN/Inf |

全部 episode 的相机内参、相关外参和深度尺度在本次比较中一致。这只说明已存元数据一致且满足数学检查，不证明历史实物标定准确，也不证明生成点云后满足当前裁剪配置。

### 2.2 时间与元数据

- 全部 `control_hz=16`；观测锚点严格递增，相邻间隔均为 62,500,000 ns。
- `timestamp` 同样严格递增，间隔为 0.0625 s。
- 已存 arm、hand、camera、VR 源时间戳全部非零，并且不晚于对应旧观测锚点。
- 相对于旧锚点，最大源样本年龄约为：arm 33.18 ms、hand 34.74 ms、camera 58.95 ms、VR 20.60 ms。
- 全部 `task_label=pick_place_toy`、`success=True`、`truncated=False`、`stop_reason=manual`、`min_frames_met=True`，相机 writer 错误字符串为空。
- 全部缺少 `technical_status`、`had_pause`、`provenance_workflow`、`termination_reason`。

上述旧标志不能替代当前技术有效性证明；固定网格间隔也不能单独证明没有暂停、重采样或历史数据处理。

### 2.3 已发现的异常

下表行索引均从 0 开始。状态码的含义需按旧代码解释，不能套用 v33 枚举。

| Episode | 帧数 | 已发现问题 |
| --- | ---: | --- |
| `episode_20260827_172305` | 304 | 行 264、265 的旧 `observation_valid` 和 `flag_camera_fresh` 为 false |
| `episode_20260827_175240` | 301 | 行 27、28 的上述两个标志为 false |
| `episode_20260827_194525` | 286 | 2 帧 `flag_frame_status=2` |
| `episode_20260827_195951` | 318 | 行 157 的聚合、密集触觉 validity 均为 false，两个触觉数组该行均含非有限值 |
| `episode_20260827_220112` | 223 | 行 220 的两个旧观测/相机标志为 false |
| `episode_20260827_220747` | 313 | 行 51 的两个旧观测/相机标志为 false |
| `episode_20260827_220919` | 225 | 行 94 的两个旧观测/相机标志为 false |
| `episode_20260827_223607` | 240 | 行 18 的两个旧观测/相机标志为 false |
| `episode_20260827_223729` | 192 | 行 111 的两个旧观测/相机标志为 false |
| `episode_20260827_224527` | 197 | 19 帧 `flag_frame_status=2`；行 133 的两个旧观测/相机标志为 false |

全量统计为 14,288 帧状态 0、21 帧状态 2；8 段共 10 帧带有上述旧观测/相机无效标志。8 段中有 1 段与控制失败 episode 重合。

这些相机异常行的源时间戳、深度帧号和彩色帧号都没有与前一行重复。仅凭当前文件无法解释所有 false 标志的来源，不能假定只是重复帧并直接放行。

### 2.4 检查边界

已执行逐段当前 `EpisodeReader` 尝试、HDF5 数组扫描、RGB-D 完整读取、相机元数据校验和官方导出 dry-run。

尚未执行迁移后的完整 FK/指尖/点云转换、Policy 数据加载验收、模型训练或物理回放。官方 dry-run 在 schema 检查阶段即拒绝全部数据，不能将其解释为已验证后续转换逻辑。

## 3. 当前格式和准入要求

### 3.1 Raw v33 结构要求

当前 `EpisodeReader` 要求版本恰好为 v33，并对 `data.h5` 的顶层数据集名称、行数、尾部形状和 dtype 严格校验；额外旧数据集也会被拒绝。

每段数据均缺少以下 10 个当前字段：

```text
action_arm_joint_target
action_hand_joint_target
action_timestamp_ns
arm_eef_intent
arm_effort
arm_timestamp_ns
camera_timestamp_ns
hand_timestamp_ns
observation_timestamp_ns
vr_timestamp_ns
```

每段均有以下 17 个不在当前 schema 中的旧字段：

```text
action_arm_ee
action_arm_joint_sent
action_hand_joint
arm_connected
arm_source_monotonic_ns
arm_tau
camera_health
camera_source_monotonic_ns
flag_action_queued
flag_camera_fresh
hand_connected
hand_qpos_stale
hand_source_monotonic_ns
observation_anchor_monotonic_ns
observation_valid
tracking_error
vr_source_monotonic_ns
```

因此，仅修改 `schema_version` 仍不能通过结构校验。

### 3.2 训练与回放要求

当前训练准入至少要求：

1. `technical_status=valid`，明确的 `provenance_workflow=teleop`，且没有暂停。
2. 非空 episode，每行控制状态均为 OK，聚合和密集触觉 validity 均为 true。
3. 要求有限的浮点数组不存在 NaN/Inf；当前检查豁免 `arm_eef_intent` 和 `head_quat_wxyz`。
4. 观测完成时间和动作发布时间均非零、严格递增，间隔不超过两个名义控制周期；动作时间不早于观测完成时间。
5. 必需源时间戳存在，相机几何、RGB-D、点云及其他派生模态通过后续检查。

`had_pause` 缺失在当前函数中默认按 false 读取，但这不能作为历史 episode 没有暂停的证据。`technical_status` 缺失默认视为 invalid，缺少 teleop provenance 也不能进入训练或回放。

物理回放另有目标、关节限位和起始姿态等要求。训练缓存迁移不等于授权或证明物理回放可用。

## 4. 必须明确的历史语义

### 4.1 历史源码证据的适用范围

已检查历史提交 `4bba54d` 中的 v30 schema、`recording/frame.py`、`teleop/episode_samples.py`、控制网格及相机 freshness 实现。

该提交提供 v30 字段的历史实现参考，**并非本批数据实际采集或迁移版本的证明**。目录日期、文件中的版本号和参考提交日期不能替代完整的数据来源记录。

### 4.2 观测锚点不等于观测完成时间

历史 `observation_anchor_monotonic_ns` 用于控制网格的因果样本选择；当前 `observation_timestamp_ns` 在读取当前观测并检查新鲜度时产生。当前 `timestamp` 来自该观测时间，而 `action_timestamp_ns` 来自动作目标发布后的主机单调时钟。

因此不得直接执行以下替代：

- 用旧网格锚点冒充真实观测完成时间。
- 用锚点、某个传感器时间或固定延迟生成动作发布时间。
- 将 `flag_action_queued=True` 解释为已记录动作发布时间或实际 SDK 执行成功。

严格的 raw v33 迁移需要找回真实记录，并证明它们与现有控制行一一对应。传感器时间不能恢复已经丢失的这些事件时间。

### 4.3 动作空间必须描述同一个最终目标

历史参考代码将 `action_arm_joint_sent` 和 `action_hand_joint` 描述为已提交的关节目标，将 `action_arm_ee` 保留为 Cartesian intent。后者可能与经过 IK、投影或其他处理后的最终关节目标不完全一致。

确认本批数据的动作来源后，训练动作应按下式构造：

```text
action    = concat(final_arm_joint_target, final_hand_joint_target)
action_ee = concat(FK(final_arm_joint_target), final_hand_joint_target)
```

旧 `action_arm_ee` 可保留为历史诊断意图，不直接充当训练 `action_ee`。观测 `eef_pose` 和 `fingertip_points` 则由观测关节状态计算。

### 4.4 状态码和相机标志不能原样解释

历史参考代码定义 `FRAME_IK_FAIL=2`、`FRAME_RETARGET_FAIL=4`；当前分别为 1、2。迁移诊断信息必须按旧枚举解释再转换，不能按数字直接复制。当前不存在对应值的旧状态必须保留其历史含义，不能映射为 OK。

旧 `flag_camera_fresh` 同时考虑新帧、帧健康、episode 开始时间和年龄等条件。当前主要消费最新样本并检查新鲜度，不能机械地把两种判定视为同一含义。已有 false 标志需要独立解释；无法解释时，相关 episode 保持未准入。

### 4.5 触觉、运动学和标定仍需追溯

还需确认触觉是否经过历史缩放、偏置校正或顺序变更，以及动作与机器人状态的单位、关节顺序、EEF 定义和手部安装变换。

60 段带有历史机器人模型和标定哈希；`episode_20260908_212811` 缺少本次比较的模型哈希。已有哈希是追溯线索，不自动证明与当前模型相同；缺失值也不能用当前文件哈希补成历史事实。

## 5. 推荐迁移路径

### 5.1 数据保护和输出边界

将现有目录作为只读历史资料保留。迁移工具放在独立离线入口，不给正常读取器、采集、回放或部署增加旧格式兼容分支。

转换前记录源文件哈希；转换后再次核对。输出写入新的目标目录，使用 staging 后发布，拒绝覆盖已有结果。旧诊断字段和真实时间戳继续保留在源数据及独立审计记录中；canonical Zarr 不增加运行时时间数组或 validity masks。

不得修改源文件版本号，不生成带有伪造时间戳的中间 v33 文件，也不得通过修改正常导出器的准入条件使本批数据强行通过。

### 5.2 准入分组

目前可建立以下临时分组，后续证据可以改变待核实组的判断：

| 分组 | Episode 数量 | 帧数 | 处理 |
| --- | ---: | ---: | --- |
| 已发现控制失败或无效触觉 | 3 | 801 | 按当前 clean demonstration 要求整段拒绝 |
| 其余带旧观测/相机无效标志 | 7 | 1,798 | 暂不准入，核实异常原因 |
| 未发现上述异常 | 51 | 11,710 | 仅为候选；继续验证来源、语义和派生模态 |
| 合计 | 61 | 14,309 | 当前均未通过 v33 原始数据准入 |

拒绝进入缓存不等于删除原始数据。所有决定以整个 episode 为单位，不删坏行、不拆段、不插值、不重采样，也不填补无效触觉。

### 5.3 训练缓存适配

一次性转换工具应先独立完成历史数据准入，再将已证明等价的数值字段交给当前派生逻辑。可复用当前 FK、指尖位置、RGB-D 和点云计算函数；不得构造虚假 reader 元数据来假装通过 v33 校验。

候选映射如下；“候选”表示需要先核实本批数据的真实来源：

| 旧内容 | 候选处理 |
| --- | --- |
| `arm_qpos`、`hand_qpos` | 拼接为 `joint_state`，保持 rad、关节顺序和行对应关系 |
| `arm_qvel`、`hand_current` | 保留物理语义，按当前缓存 dtype 输出 |
| `arm_tau` | 确认 SDK 来源后映射到 `arm_effort`，保持未验证 SI 单位的声明 |
| 两组旧关节动作 | 确认为同一步最终发布目标后构造 `action` |
| 最终关节动作的 FK | 构造 `action_ee`，不复用旧 Cartesian intent 作为最终目标 |
| 触觉及 validity | validity 用于整段准入；有效数值进入完整缓存 |
| RGB-D 和已存相机几何 | 保留逐行身份，由原始深度尺度和相机几何计算点云 |
| 观测关节状态 | 计算 `eef_pose`、`fingertip_points` |

缓存应包含全部学习相关模态，由 `dexmani_policy` 在加载时选择输入。点云参数、桌面平面、手部安装参数和模型文件必须显式确定，不能把当前环境标定未经核对地视为历史场景事实。

### 5.4 输出 schema 的决策

只有确认旧控制步满足当前训练的观测先于动作、因果样本选择、最终目标和模态语义，才可生成当前 canonical Zarr。迁移来源、工具版本、输入哈希、配置与逐段理由应记录在独立迁移报告中，不将缺失证据标为已验证。

若必须改变观测对齐、动作定义或其他持久化语义，则需要新的 schema 及对应的 Policy 契约评估。不能仅修改版本号，或填入 `control_step_latest_causal` 等字符串，使校验器接受语义不同的数据。

当关键语义无法确认时，应停止该 episode 的训练准入，保留诊断用途。另建 schema 也不能自动解决数据是否适合训练的问题。

## 6. 实施步骤与验收

### 6.1 先补齐来源证据

需要查明：

1. 本批文件是否经过旧脚本迁移、重采样或字段补写；最初录制文件、脚本、日志和配置是否还在。
2. 实际录制或迁移版本，以及各模态与最终动作如何对齐到当前行。
3. 暂停、异常退出和技术有效性的历史记录。
4. 触觉单位与校正过程、模型/EEF/手部安装定义、相机与桌面标定来源。

截至本文件编写时，尚未获得上述补充材料。缺失动作发布时间意味着不能仅凭现有文件完成严格 raw v33 迁移；训练缓存适配同样需要独立的语义证据。

### 6.2 先审计，再转换

第一阶段只输出迁移清单：逐段来源、字段检查、语义证据、准入结论、具体拒绝理由和无法恢复的信息。不得只输出一个整体成功标志。

第二阶段只转换通过审计的完整 episode。先用少量已通过审计的数据验证全链路，再执行全量转换；转换期间任何 RGB-D、FK、指尖或点云失败都导致对应 episode 整段拒绝，输出结果不含其部分行。

### 6.3 验收标准

- 源数据哈希不变；准入 episode 的顺序和行数完整保留。
- 直接映射字段符合预定 dtype 转换，动作没有新增裁剪、平滑、插值或偏移。
- `action` 与 `action_ee` 的 arm 目标经同一 FK 一致，hand 目标一致。
- 模态齐全，形状、dtype、有限性、旋转表示、点云范围和 RGB-D 对应关系正确。
- `episode_ends` 与完整准入 episode 的累计行数一致，不跨 episode 拼接观测历史。
- 通过 `dexmani_policy` 公共数据契约验证 joint/EEF 两种 action 布局和全部观测模态，并做最小数据加载检查。
- 保存最终转换配置、输入哈希、逐段决定和检查结果；未执行的项目明确标记为未验证。

验收不启动完整训练或真机回放。离线数据合格不等于硬件执行已验证。

## 7. 代码依据与检查记录

| 内容 | 当前源码 |
| --- | --- |
| Raw schema 和控制状态码 | [schema.py](dexmani_real/recording/storage/schema.py) |
| 严格读取和技术有效性检查 | [reader.py](dexmani_real/recording/storage/reader.py) |
| 当前录制字段来源 | [frame.py](dexmani_real/recording/frame.py) |
| 观测完成时间 | [observation.py](dexmani_real/runtime/observation.py) |
| 动作发布时间 | [commands.py](dexmani_real/robot/commands.py) |
| 控制步的观测、动作和录制关系 | [controller.py](dexmani_real/teleop/control/controller.py) |
| 训练准入和数值派生 | [processing.py](dexmani_real/dataset/processing.py) |
| Canonical Zarr 输出契约 | [contracts.py](dexmani_real/dataset/contracts.py)、[export.py](dexmani_real/dataset/export.py) |
| 相机几何读取与点云派生 | [pointcloud.py](dexmani_real/dataset/pointcloud.py) |
| 回放加载边界 | [trajectory.py](dexmani_real/replay/trajectory.py) |

历史参考可通过 `git show 4bba54d:<path>` 查看，重点路径为：

```text
dexmani_real/recording/storage/schema.py
dexmani_real/recording/frame.py
dexmani_real/teleop/episode_samples.py
dexmani_real/teleop/control_loop/grid.py
dexmani_real/teleop/control_loop/camera_freshness.py
```

本次还只读检查了相邻 `dexmani_policy` 仓库的 `dexmani_policy/datasets/real_policy_contract.py` 和 `base_dataset.py`，确认训练消费者要求明确的动作与观测对齐语义，而非仅满足张量形状。

已执行的导出检查命令如下；`--output` 指向当时不存在的临时目标，dry-run 没有生成 Zarr：

```bash
DEXMANI_LOG_DIR=/tmp/dexmani_audit_logs \
  /home/zhanghaoyang/miniconda3/envs/real_robot/bin/python \
  examples/export_policy_zarr.py episodes/pick_place_toy \
  --dry-run --output /tmp/pick_place_toy_format_audit_output.zarr
```

逐段临时检查记录位于 `/tmp/pick_place_toy_format_audit.json` 和 `/tmp/pick_place_toy_sidecar_audit.json`。这些文件可能被系统清理，不作为长期依据；本文件已记录关键结论、异常明细和实施条件。真正执行迁移时，应重新审计并将完整报告保存在持久化输出位置。
