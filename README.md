# DexMani Real

面向真实机器人灵巧操作研究的个人 PhD 代码库。

DexMani Real 聚焦 **real-world dexterous manipulation** 的实验闭环：真实机器人标定、遥操作示教、多模态数据采集、数据整理，以及 learned policy 的真机部署与评估。当前实验平台以 xArm7、XHand、RealSense RGB-D 与 VR / HTS 为核心；策略模型、训练与算法实验主要位于 `dexmani_policy`。

> 这是研究代码，不是通用机器人产品或 SDK 封装。仓库设计优先服务当前博士研究问题、实验可信度与快速迭代，而不是追求长期 API 兼容或通用平台化。

## 研究定位

本仓库主要承担四类工作：

1. **真实机器人实验基础设施**  
   管理机械臂、灵巧手、RGB-D 相机与 VR / HTS 等真实设备的实验入口和运行边界。

2. **多模态示教与实验数据**  
   采集真实机器人状态、RGB-D、点云、手部电流 / 触觉等与灵巧操作相关的观测，并保存可追溯的实验原始数据。

3. **从真实数据到策略训练的桥接**  
   将 Raw episode 整理为可供策略训练使用的派生数据，同时保留原始实验事实与必要的处理信息。

4. **learned policy 的真机闭环评估**  
   将 `dexmani_policy` 中训练得到的策略接入真实机器人，在当前实验标定和安全边界下进行 rollout 与评估。

本仓库**不试图**成为通用机器人中间件、硬件抽象框架、数据格式标准或模型训练框架。与学习算法直接相关的模型结构、训练流程和策略实现应优先放在 `dexmani_policy`。

## 研究工作流

典型实验链路为：

```text
实验环境与硬件准备
        ↓
相机 / 工作空间 / VR 等标定
        ↓
遥操作示教与 Raw episode 采集
        ↓
数据检查与离线处理
        ↓
Canonical training cache
        ↓
dexmani_policy 训练
        ↓
策略真机部署与 rollout
        ↓
实验结果分析
```

这个流程体现了本项目最重要的边界：

- **Raw 数据是实验事实**，应尽量保持原始、可追溯和不可变。
- **训练数据是可重建的派生物**，可以随研究需要重新处理和重新导出。
- **标定属于当前实验现场状态**。相机外参、桌面平面、手部安装等信息应在新的实验设置下重新标定或确认，而不是被当作跨实验永久不变的接口契约。
- **训练产物来自 `dexmani_policy`，真实执行边界属于 `dexmani_real`**。两个仓库通过必要的实验信息协作，不复制整套实现语义。

## 实验平台

| 组成 | 当前角色 |
|---|---|
| xArm7 | 机械臂运动与末端位姿控制 |
| XHand | 五指灵巧操作、关节状态与触觉 / 电流观测 |
| RealSense RGB-D | RGB、深度与 3D 几何观测 |
| VR / HTS | 人类示教与遥操作输入 |
| `dexmani_policy` | 策略训练、模型实现与推理逻辑 |

硬件组合和实验设置可能随研究推进而变化。代码中出现的具体设备、标定结果或处理参数，应理解为当前研究平台的一部分，而不是项目对外承诺的固定标准。

## 仓库结构

```text
dexmani_real/
├── calibration/    # 实验标定与标定状态
├── config/         # 实验配置
├── robot/          # 机器人与灵巧手接口
├── sensor/         # 真实传感器
├── teleop/         # 遥操作
├── runtime/        # 真机实验运行时
├── ipc/            # 运行时进程间通信
├── recording/      # 实验数据记录
├── dataset/        # Raw 读取、处理与训练数据导出
├── deployment/     # learned policy 真机接入
├── planning/       # 与真实机器人运动相关的规划能力
├── replay/         # 已录制轨迹的真实回放
└── utils/          # 小型通用辅助代码

examples/           # 面向实验人员的主要命令行入口
```

目录划分服务于“真实实验链路是否清晰”。除非研究问题确实需要，不应为了形式统一而增加额外 framework、registry、schema layer 或兼容层。

## 安装

要求 Python >= 3.10。

```bash
python -m pip install -e .
```

真实实验机还需要按当前硬件环境安装对应 SDK、几何 / 点云依赖以及 `dexmani_policy` 所需环境。仓库本身不会尝试把所有实验机软件依赖抽象成一个通用发行环境。

## 常用入口

以下入口覆盖当前最常见的研究流程：

| 任务 | 入口 |
|---|---|
| VR 遥操作与示教采集 | `python examples/collect_teleop.py --help` |
| 键盘调试 / 回零 | `python examples/keyboard_teleop.py --help` |
| 相机标定 | `python examples/calibrate_camera.py --help` |
| VR heading 标定 | `python examples/calibrate_vr_heading.py --help` |
| Raw episode 检查 | `python examples/visualize_episode.py --help` |
| 训练数据导出 | `python examples/export_policy_zarr.py --help` |
| 真实轨迹回放 | `python examples/replay_episode.py --help` |
| Policy rollout | `python examples/run_policy.py --help` |

具体参数、默认值和实验能力以入口的 `--help`、当前配置以及对应源码为准。README 只维护稳定的研究工作流，不复制容易过时的实现细节。

## 数据原则

### Raw episode

Raw 是真实实验的 source of truth。它用于回答“实验当时实际观测到了什么、发布了什么目标、以什么状态结束”。

基本原则：

- 已发布 Raw 不做原地修补或语义重写。
- 传感器缺测、动作未发布、异常终止等事实应被保留，而不是为了得到“更干净”的数据而事后改写。
- demonstration 与 policy rollout 都应保留真实实验边界；失败 rollout 同样是实验结果。
- 与实验可复现性相关的物理配置应保留必要 provenance，但不把一次实验的标定数值升级为全局固定协议。

### Canonical training cache

Canonical 数据用于训练，是从 Raw 重建得到的派生缓存。

它应满足：

- 对训练所需模态进行严格的一致性与有限值检查；
- 保留必要的数值处理配置；
- 不反向修改 Raw；
- 当处理逻辑发生实质变化时，优先从 Raw 重新生成，而不是长期维护旧缓存兼容链。

当前项目使用 Zarr 作为主要训练缓存载体。字段的科学含义应由生成 / 消费代码与稳定文档共同保持清晰，而不是依靠重复的语义注册表在多个仓库之间同步。

## 与 dexmani_policy 的关系

两个仓库的职责应保持简单：

```text
dexmani_real
真实传感器 / 机器人
        ↓
Raw experiment data
        ↓
processed training data
        ↓
dexmani_policy
Dataset → Policy → Training → Checkpoint
        ↓
dexmani_real
live observation → policy inference → safe robot execution
```

`dexmani_real` 不应复制策略网络、训练循环或通用 Dataset 逻辑；`dexmani_policy` 也不应承载真实硬件生命周期、现场标定和真机安全控制。

研究接口只保存真实复现实验所需的信息。对于点云、指尖几何等与训练数值处理直接相关的设置，可以随训练产物传递；相机外参、桌面平面等现场标定应使用当前实验环境的有效结果。

## 真机安全

本仓库能够连接并控制真实硬件。任何真机运行都应满足：

- 操作者在场并具备立即急停能力；
- 机械限位、工作空间、安装状态和实验环境已人工确认；
- 所需标定在当前实验设置下有效；
- 真机动作经过明确授权，不把离线测试结果当作真实安全证明；
- 首次运行新代码、新策略或新实验设置时采用保守条件逐步验证。

**不要把 import 成功、配置解析成功、compile / lint 通过或仿真结果视为真机安全验证。**

对于代码代理或自动化工具：除非用户明确授权，不得执行会连接设备、启动采集、写入标定、回零、遥操作、轨迹回放或 policy rollout 的命令。详细协作规则见 [AGENTS.md](AGENTS.md)。

## 开发原则

DexMani Real 是长期演化的个人研究代码。维护时优先考虑：

1. 实验安全和科学可信度；
2. 当前研究问题是否更容易验证；
3. 数据是否可追溯、可重建；
4. 代码是否简单、直接、容易检查；
5. 最后才是通用性和长期兼容。

如果一个旧抽象、兼容分支或文档已经不再服务当前研究，应优先删除，而不是继续包裹。对于一次性重构计划、验收清单和临时 Codex 任务文档，不应长期保留在仓库根目录。

代码代理的长期规则见 [AGENTS.md](AGENTS.md)。Claude 的入口说明见 [CLAUDE.md](CLAUDE.md)。

## 当前状态

本项目处于持续研究与快速迭代阶段。接口、数据处理方式和实验流程会随论文问题与真实机器人实验经验演化。

因此，使用某项能力前请优先确认：

- 当前分支与配置；
- 对应 `examples/` 入口；
- 相关源码中的实际行为；
- 当前实验平台与标定状态。

对于论文实验，应同时记录代码版本、训练配置、数据版本与真实实验设置，以便后续复现和分析。
