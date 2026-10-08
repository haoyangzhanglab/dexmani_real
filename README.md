# DexMani Real

**Real-world dexterous manipulation research stack for data collection, policy deployment, and robot evaluation.**

DexMani Real 是面向个人 PhD 研究的真实机器人代码库，当前围绕 **xArm7 + XHand + RealSense RGB-D + VR / HTS** 构建灵巧操作实验闭环。仓库负责真实硬件、标定、示教采集、Raw 数据、离线处理以及 learned policy 的真机执行；模型与训练主要位于 [dexmani_policy](https://github.com/haoyangzhanglab/dexmani_policy)。

> Research code under active development. 该仓库服务于当前论文研究与真实机器人实验，不以通用机器人平台、稳定 SDK 或长期向后兼容为目标。

研究链路：标定 → 遥操作/采集 → Raw → Canonical Zarr → Policy 训练 → 真机评估。

## Installation

支持源码 checkout + editable 安装（Python >= 3.10）；assets 按 checkout 路径读取，不宣称 wheel/PyPI 支持。

```bash
python -m pip install -e .
# 桌面交互入口另需：python -m pip install -e ".[interactive]"
```

真实实验机还需要安装对应硬件 SDK、几何 / 点云依赖以及策略推理环境。沿用实验机现有环境，不在实验过程中自动升级。
离线几何导出需要 Pinocchio，点云诊断视图需要 Open3D（公共点云导出使用 NumPy/SciPy/OpenCV）；retargeting 需要 NLopt/选定后端。
交互入口使用 pynput 全局键盘监听，需要桌面显示环境；`--info` 不需要 Rerun。

基础依赖与交互依赖在 `pyproject.toml` 中分开；几何、算法及厂商 SDK 使用实验环境中已核实的安装。

## Quick Start

### 1. 检查配置

采集与部署入口提供 `--print-config`，可在不启动硬件的情况下检查声明配置：

```bash
python examples/collect_teleop.py --print-config
python examples/run_policy.py --print-config

# 需要现场覆盖时，导出配置并编辑 IP、串口、相机序列号或执行参数。
python examples/collect_teleop.py --print-config > local.yaml
python examples/collect_teleop.py --config local.yaml --print-config
```

默认值统一由包内配置提供，`--config` 是可选 YAML 覆盖，只需写入待调整的字段；保留的 CLI 操作开关优先于 YAML。机械臂速度、加速度通过 YAML `arm` 配置，部署模式和预算通过 `execution` 配置。

配置加载检查字段名和数值类型，运行入口再检查所需参数、现场标定与策略执行条件。`--print-config` 只打印声明配置，不读取现场标定或加载设备、Policy、优化器、显示后端；它不代表运行时启动检查已通过。

### 2. 标定

```bash
python examples/calibrate_camera.py --help
python examples/calibrate_vr_heading.py --help
python examples/pointcloud_process_example.py --help
```

相机外参、桌面平面、hand mount、VR alignment 等属于**当前实验现场状态**。更换相机、桌面、机器人安装或实验布局后，应重新标定或确认，而不是沿用历史数值作为固定契约。首次桌面标定在 pointcloud 示例中选择 table calibration，不要求旧 plane 文件。
手基座/指尖 FK 使用当前安装补偿，碰撞与规划使用标称资产几何。
录制允许缺相机外参，重建点云需要对应的内外参和 depth scale。相机、桌面与 VR 标定在会话启动时读取；文件修改在下次会话生效。
路径可显式选择：采集 `--vr-transform` / `--camera-calibration`，policy `--camera-calibration` / `--output`，相机/VR 标定 `--output`；桌面使用配置的 `environment.table.plane_path`（相对 checkout），示教输出沿用 `policy.episodes_dir`。默认仍是既有 checkout 路径，不自动移动历史标定或数据。

两个相机诊断入口通过 `--config` 使用实验相机参数。RealSense 实时诊断退出时恢复可读取的原曝光优先级；图像读取、对齐或曝光恢复失败返回非零。

### 3. 采集示教

```bash
python examples/collect_teleop.py <task>
```

任务是位置参数，省略时沿用 `test`，同时用于录制元数据和 `episodes/<task>/` 目录。

启动会在停止监听建立后执行手部 HOME。触觉需在空闲、无 capture 且操作者确认手部无接触后按 **T** 采集并核验基线，S/Q/ESC 可取消。T 只做基线准备，SDK 阻塞期间取消仍需等待返回；未归零或失败的通道保留 NaN，不阻断关节任务。H 为整机 HOME，B 开始，C 暂停/恢复；恢复时保存上一段并开始新 episode。S 停止保存，D 丢弃当前 capture，Q 进入退出确认、再次 Q 保存退出（尚未开始且无 capture 时直接退出），ESC 急停。键盘 jog 入口仅在 R 时 HOME。
操作者结束原因和随后发生的停止故障分别留证；写盘或发布失败保留 staging 并报告会话失败，已发布 Raw 不回写。
录制启用全部已接入物理模态，aggregate/dense 触觉独立校准和判断可用性，缺测保留 NaN；驱动报告不可用状态，离线导出的 `export_report.json` 汇总缺测情况。相机无可用帧或任一路持续停帧时停止并保留前缀，不补黑图。

Raw episode 是实验 source of truth。已发布 Raw 不做原地修补；暂停、失败、缺测和终止原因等实验事实应被保留。

### 4. 检查与导出数据

```bash
python examples/visualize_episode.py <episode> --info
python examples/visualize_episode.py <episode> --max-frames 100

python examples/export_policy_zarr.py episodes/<task>
```

`--info` 只查看元数据和数值概要；交互 viewer 需要 Rerun，按所选帧数顺序读取 RGB-D。公共导出始终生成全模态缓存，下游在 `dexmani_policy` 中按模型所需字段筛选样本。处理逻辑变化后，从 Raw 重新导出。

Viewer 的 `--config` 指定当前实验的点云、桌面及手部安装参数，省略时使用默认配置；相机几何仍来自对应 Raw。

数据入口按当前 Raw 布局读取；缺失必要字段会明确报错，不自动补齐旧格式。旧实验数据保留原样，使用对应源码版本复现。

离线导出复用相同的 YAML 加载和覆盖规则，只检查实际处理所需参数；未使用的 TAG/DexPilot 或遥操作数值参数不会阻挡导出。仅在 `pointcloud.remove_table: true` 时读取当前桌面平面；设为 `false` 时无需桌面标定文件，也无需修改 `environment.table.enabled`。

### 5. 训练策略

策略模型、Dataset、训练与 checkpoint 由 [dexmani_policy](https://github.com/haoyangzhanglab/dexmani_policy) 管理。本仓提供可重建的 Canonical 数据。

### 6. 真机评估

```bash
python examples/run_policy.py <policy/task/experiment>
```

H 执行 HOME，空闲且确认手部无接触后 T 归零触觉，B 请求开始，S 停止，Q 退出，ESC 急停。HOME 结束后若要再次 HOME、归零或开始，需重新按 H/T/B；停止尚在处理时 H 被拒绝，完成后需重新发起。触觉归零可用 S/Q/ESC 取消。开始前检查当前 arm/hand HOME 姿态。默认每段运行预算 60 秒，可用 `--max-duration` 修改。

执行模式默认为 sync，预算使用包内针对当前 16 Hz、8 步 chunk 实验的起始值；更换策略节拍、执行长度或现场条件后需核对。通过可选 YAML `execution` 覆盖执行设置；async/RTC 必须指定预取步数，RTC 还需指定 guidance cap。启动检查保存的 Policy 节拍及 warmup 耗时，warmup 不自动放宽预算。模型加载与 warmup 在连接设备前完成，离线检查不能证明现场安全。

checkpoint 默认选择 `best`；策略对比参数在 `--help` 的 Policy research overrides 中列出。启用录制时，会话的 `run_config.yaml` 保存实际配置、checkpoint、策略覆盖值和源码版本。

`--no-record` 仍连接设备并执行动作，仅关闭录制；程序接口的 `execute=False` 仍连接并读取设备。它们都不是离线验证入口。

工作空间 bounds 只裁剪 EEF 意图，不证明最终 FK 或整条轨迹在界内。设备 SDK、GIL 或 CUDA 阻塞仍可能影响停止响应；现场需验证停止覆盖、反馈与工作预算。离线测试或 checkpoint 可加载不等价于真机安全验证。

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
| XHand action example | `examples/xhand_control_example.py` |
| RGB-D diagnostics | `examples/realsense_record_example.py` |
| Point-cloud diagnostics | `examples/pointcloud_process_example.py` |

`python examples/xhand_control_example.py` 连接 XHand1 后，顺序发送 HOME → fist → palm → V → OK → HOME，每个动作实测到位后停留 1 秒；失败或 Ctrl+C 中断时停止序列并请求停止。HOME/palm、增益和电流限制使用当前 `hand` 配置，支持 `--config`、`--comm serial|ethercat`、`--device` 和 `--hand-id`。
`--read-only` 只连接并读取一次；`--help` 和 `--print-config` 均不连接设备。

具体参数与当前行为以各入口的 CLI、resolved configuration 和源码为准。

## Data

Raw 保存真实观测、尝试目标与结束原因，发布后不可变。停止和失败保留已采前缀，仅操作者显式丢弃才删除当前 capture；写盘或发布失败保留 staging 并报错。

Canonical 是从 Raw 重建的训练缓存，公共导出保留全部 13 字段、缺测与 episode 边界，下游按模型所需字段筛选。重新导出使用新路径，不覆盖 Raw 或旧缓存。

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

Real 侧负责真实硬件生命周期、现场标定和运动安全；Policy 侧负责模型与训练，Sim 侧负责仿真实验。

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

离线检查命令与长期协作约束见 [AGENTS.md](AGENTS.md)，Claude 阅读入口见 [CLAUDE.md](CLAUDE.md)。当前行为以源码、配置和 CLI 为准。
