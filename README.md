# DexMani Real

**Real-world dexterous manipulation research stack for data collection, policy deployment, and robot evaluation.**

DexMani Real 是面向个人 PhD 研究的真实机器人代码库，当前围绕 **xArm7 + XHand + RealSense RGB-D + VR / HTS** 构建灵巧操作实验闭环。仓库负责真实硬件、标定、示教采集、Raw 数据、离线处理以及 learned policy 的真机执行；模型与训练主要位于 [dexmani_policy](https://github.com/haoyangzhanglab/dexmani_policy)。

> Research code under active development. 该仓库服务于当前论文研究与真实机器人实验，不以通用机器人平台、稳定 SDK 或长期向后兼容为目标。

## Highlights

- **Dexterous teleoperation**：VR / HTS 驱动的机械臂与五指灵巧手示教采集。
- **Multimodal real data**：机器人状态、RGB-D、点云、XHand 电流与触觉等真实观测。
- **Raw → training data**：保留不可变 Raw evidence，并生成可重建的训练缓存。
- **Real policy deployment**：将 `dexmani_policy` 中训练的策略接入真实机器人进行闭环评估。
- **Experiment-first design**：优先实验安全、数据可信度和研究迭代，不做无需求的平台化。

## System

| Component | Role |
|---|---|
| xArm7 | 机械臂控制与末端运动 |
| XHand | 五指灵巧操作、关节状态、触觉与电流 |
| RealSense RGB-D | RGB、depth 与 3D geometry |
| VR / HTS | 人类示教与遥操作输入 |
| `dexmani_policy` | Dataset、policy、training 与 checkpoint |

整体研究链路：

```text
Calibration
    ↓
Teleoperation / Data Collection
    ↓
Raw Episodes
    ↓
Offline Processing
    ↓
Canonical Zarr
    ↓
dexmani_policy Training
    ↓
Checkpoint
    ↓
Real-Robot Rollout
```

## Installation

要求 Python >= 3.10。

```bash
python -m pip install -e .
```

真实实验机还需要安装对应硬件 SDK、几何 / 点云依赖以及策略推理环境。不同机器的硬件依赖可能不同，本仓库不尝试提供一个覆盖所有设备的统一环境。

## Quick Start

### 1. 检查配置

支持 `--config` 的实验入口可先在不启动硬件的情况下检查 resolved config：

```bash
python examples/collect_teleop.py --print-config
python examples/collect_teleop.py --config experiment.yaml --print-config
```

### 2. 标定

```bash
python examples/calibrate_camera.py --help
python examples/calibrate_vr_heading.py --help
```

相机外参、桌面平面、hand mount、VR alignment 等属于**当前实验现场状态**。更换相机、桌面、机器人安装或实验布局后，应重新标定或确认，而不是沿用历史数值作为固定契约。

### 3. 采集示教

```bash
python examples/collect_teleop.py \
    --config experiment.yaml \
    --task-name <task>
```

Raw episode 是实验 source of truth。已发布 Raw 不做原地修补；暂停、失败、缺测和终止原因等实验事实应被保留。

### 4. 检查与导出数据

```bash
python examples/visualize_episode.py <episode> --info

python examples/export_policy_zarr.py episodes/<task>
```

Canonical Zarr 是从 Raw 生成的训练缓存。处理逻辑变化时，优先从 Raw 重新导出，而不是维护复杂的旧缓存兼容链。

### 5. 训练策略

策略模型、训练配置和 checkpoint 管理由 `dexmani_policy` 负责：

```text
dexmani_real Raw / Zarr
        ↓
dexmani_policy Dataset
        ↓
Policy Training
        ↓
Checkpoint
```

### 6. 真机评估

```bash
python examples/run_policy.py <experiment_dir> --checkpoint best
```

可用 `--n-action-steps N` 覆盖本次部署的执行段长度；必须满足
`n_obs_steps - 1 + N <= horizon`，不会改写训练配置或 checkpoint。
实际长度和覆盖参数会保存在 `run_config.yaml`。

需要定位预测换段或执行时序问题时，添加 `--diagnostics`。它在 session 的
`diagnostics/` 中单独保存完整预测、实际模型输入和主机侧发布／SDK 时间，默认关闭。
会话退出后离线分析：

```bash
python examples/analyze_rollout_diagnostics.py <session_dir>
```

报告包含诊断完整性、逻辑时间对齐的新旧预测、命令发送与关节反馈；SDK 调用时间不代表固件执行完成时间。

策略 rollout 使用当前真实实验环境的有效标定与硬件状态。离线测试、仿真结果或 checkpoint 可加载均不等价于真机安全验证。

## Main Entry Points

| Workflow | Entry |
|---|---|
| VR teleoperation / collection | `examples/collect_teleop.py` |
| Keyboard control / home | `examples/keyboard_teleop.py` |
| Camera calibration | `examples/calibrate_camera.py` |
| VR heading calibration | `examples/calibrate_vr_heading.py` |
| Raw episode inspection | `examples/visualize_episode.py` |
| Raw → Zarr export | `examples/export_policy_zarr.py` |
| Physical replay | `examples/replay_episode.py` |
| Policy rollout | `examples/run_policy.py` |
| XHand diagnostics | `examples/xhand_diagnostics.py` |
| RGB-D diagnostics | `examples/realsense_record_example.py` |
| Point-cloud diagnostics | `examples/pointcloud_process_example.py` |

具体参数与当前行为以各入口的 CLI、resolved configuration 和源码为准。

## Data

### Raw

Raw 保存真实实验中实际发生的观测与控制证据。

原则：

- Raw 一经发布保持不可变；
- 不用后处理掩盖传感器缺测、动作未发布或异常终止；
- demonstration 与 policy rollout 都保留真实实验边界；
- 失败 rollout 也是实验结果，不应只保留成功轨迹。

### Canonical Zarr

Canonical 数据是面向策略训练的派生缓存。

它负责：

- 统一当前训练所需的数据表示；
- 严格检查训练数据的一致性和数值有效性；
- 保存必要的 numerical preprocessing 信息；
- 保持可由 Raw 重新生成。

Canonical 不是新的实验事实，也不承担长期格式兼容。

一个 store 当前只组织一个 task；`dt` 必须一致，属于训练/部署数值语义。`depth_scale` 在 store 内保持一致以解释存储深度，不作为 Policy 兼容元数据。导出默认拒绝已有路径；`examples/export_policy_zarr.py --overwrite` 会先完整构建、验证 staging，再替换已有 canonical cache，始终不修改 Raw。

## Repository Layout

```text
dexmani_real/
├── calibration/    # calibration workflows and state
├── config/         # experiment configuration
├── robot/          # xArm / XHand interfaces
├── sensor/         # real sensors
├── teleop/         # teleoperation
├── runtime/        # experiment runtime
├── ipc/            # inter-process communication
├── recording/      # Raw recording
├── dataset/        # Raw loading and offline processing
├── deployment/     # policy deployment on real robot
├── planning/       # motion-planning utilities
├── replay/         # physical trajectory replay
└── utils/          # small shared utilities

examples/           # experiment-facing CLI entry points
```

目录组织围绕真实实验 workflow，而不是通用 framework taxonomy。新增模块前优先确认它是否对应真实的硬件、数据或实验责任边界。

## Relationship with Other Repositories

- [dexmani_policy](https://github.com/haoyangzhanglab/dexmani_policy)：策略模型、Dataset、训练、评估与 inference logic。
- [dexmani_sim](https://github.com/haoyangzhanglab/dexmani_sim)：仿真环境与相关灵巧操作实验。

推荐保持以下边界：

```text
dexmani_sim   → simulation experiments
dexmani_policy → learning algorithms
dexmani_real   → real-world experiments
```

Real 侧不复制模型训练内部实现；Policy 侧不承担真实硬件生命周期、现场标定和运动安全。

## Safety

⚠️ **This repository controls real hardware.**

运行任何真机流程前：

- 确认机械限位、安装状态和工作空间；
- 确认当前实验所需标定有效；
- 保持操作者在场并可立即急停；
- 首次运行新代码、新策略或新实验配置时采用保守条件逐步验证。

代码代理或自动化工具在没有明确授权时，不应执行设备连接、运动、回零、实时采集、标定写入、replay 或 policy rollout。

更详细的仓库协作规则见 [AGENTS.md](AGENTS.md)。

## Development

这是一个持续演化的研究仓库。修改时优先保持：

1. experiment safety；
2. scientific correctness；
3. data traceability；
4. simple and readable code；
5. fast research iteration。

长期文档只记录稳定的研究 workflow 和边界。具体实现细节放在源码、配置和 CLI help 中；一次性 task plan 与验收记录不进入长期文档。

面向代码代理的规则见 [AGENTS.md](AGENTS.md)，Claude 入口见 [CLAUDE.md](CLAUDE.md)。
