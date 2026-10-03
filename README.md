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

真实实验机还需要安装对应硬件 SDK、几何 / 点云依赖以及策略推理环境。沿用实验机现有环境，不在实验过程中自动升级。
离线几何导出需要 Pinocchio，点云诊断视图需要 Open3D（公共点云导出使用 NumPy/SciPy/OpenCV）；retargeting 需要 NLopt/选定后端。
交互入口使用 pynput 全局键盘监听，需要桌面显示环境；`--info` 不需要 Rerun。

## Quick Start

### 1. 检查配置

支持 `--config` 的实验入口可先在不启动硬件的情况下检查 resolved config：

```bash
cp experiment.example.yaml experiment.yaml
# 编辑 IP、XHand 串口和相机序列号，再核对当前标定。
python examples/collect_teleop.py --print-config
python examples/collect_teleop.py --config experiment.yaml --print-config
```

### 2. 标定

```bash
python examples/calibrate_camera.py --help
python examples/calibrate_vr_heading.py --help
python examples/pointcloud_process_example.py --help
```

相机外参、桌面平面、hand mount、VR alignment 等属于**当前实验现场状态**。更换相机、桌面、机器人安装或实验布局后，应重新标定或确认，而不是沿用历史数值作为固定契约。首次桌面标定在 pointcloud 示例中选择 table calibration，不要求旧 plane 文件。
物理录制允许缺相机外参，但点云需要数据对应的有效内外参和 depth scale。

### 3. 采集示教

```bash
python examples/collect_teleop.py \
    --config experiment.yaml \
    --task-name <task>
```

启动会在停止监听建立后执行手部 HOME。H 为整机 HOME，B 开始，C 暂停/恢复；恢复时保存上一段并开始新 episode。S 停止保存，D 丢弃当前 capture，Q 进入退出确认、再次 Q 保存退出（尚未开始且无 capture 时直接退出），ESC 急停。键盘 jog 入口仅在 R 时 HOME。
录制启用全部已接入物理模态，触觉持续不可用会在日志和 episode 摘要中显示。相机无可用帧或任一路持续停帧时停止并保留前缀，不补黑图。

Raw episode 是实验 source of truth。已发布 Raw 不做原地修补；暂停、失败、缺测和终止原因等实验事实应被保留。

### 4. 检查与导出数据

```bash
python examples/visualize_episode.py <episode> --info
python examples/visualize_episode.py <episode> --max-frames 100

python examples/export_policy_zarr.py episodes/<task> --config experiment.yaml
python examples/read_policy_windows.py datasets/<task>.zarr --horizon 2
```

`--info` 只查看元数据和数值概要；交互 viewer 需要 Rerun，按所选帧数顺序读取 RGB-D。公共导出始终生成全模态缓存，窗口示例再按策略所需字段选样。处理逻辑变化后，从 Raw 重新导出。

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
python examples/run_policy.py <policy/task/experiment> --config experiment.yaml --checkpoint best
```

H 执行 HOME，B 请求开始，S 停止，Q 退出；开始前检查当前 arm/hand HOME 姿态，不依赖历史 HOME 令牌。默认每段运行预算 60 秒，可用 `--max-duration` 修改。

可用 `--n-action-steps N` 覆盖本次部署的执行段长度；必须满足
`n_obs_steps - 1 + N <= horizon`，不会改写训练配置或 checkpoint。
实际长度和覆盖参数会保存在 `run_config.yaml`。

Policy、VR teleop、keyboard、replay 和 camera calibration 的正常动作统一由本地薄组合 Robot 调用现有 xArm / XHand 驱动。模型推理、设备读写及 HOME 在同一应用线程串行执行；传感器进程和录制 writer 保留。Policy 每轮读取观测、按需同步推理、发送一个目标，并只等待剩余周期预算。

历史仅使用最近 N 个真实观测，启动先采满，慢推理期间没有新控制观测；不复制当前帧掩盖时间缺口。
录制行是控制观测与尝试目标，并非每条 action 都有一次模型 query。rollout 录制独立保存完整 RGB-D/触觉，公共点云由保存行的 RGB-D 重建。
工作空间 bounds 只裁剪 EEF 意图；它不证明最终 FK 或整条轨迹都在界内。

单线程在推理或 SDK 阻塞期间不能并发调用物理停止，也不提供独立于推理的 Python 触觉采样。现场需另行确认停止覆盖、阻塞容忍、反馈/触觉需求与实际工作预算。`execute=False` 仍可能连接并读取设备。

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
| Offline policy-window example | `examples/read_policy_windows.py` |
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
- 停止、晚帧、控制失败均保留可保存前缀，显式丢弃才删除当前 capture；
- 写盘、编码或发布失败保留 staging 并报错，不报告为成功 Raw。

Raw 保存全量已接入物理模态，公共导出再重建末端、指尖与点云。动作是最终尝试的关节目标；未调用的设备目标为 NaN，dispatch 区分未调用、SDK 接受、CRC 不确定、拒绝和未知，均不证明物理到达。

逐行时间记录主机 monotonic 观测、机器人读取完成、相机队列返回及下发完成时间，并保留 RGB/depth 源帧号。时间 0 表示未知或未下发；旧 Raw 不伪造实测时间。latest-sample 不保证跨模态严格同步，重复源帧可以是正常采样结果。

### Canonical Zarr

Canonical 数据是面向策略训练的派生缓存。

它负责：

- 保留全部 13 个公共字段，训练端再选择所需字段；
- 检查实际 shape/dtype/行数；缺测保留 NaN，模型输入处检查所需数据；
- 保存必要的 numerical preprocessing 信息；
- 保持可由 Raw 重新生成。

Canonical 不是新的实验事实，也不承担长期格式兼容。

一个 store 当前只组织一个 task；`dt` 必须一致，属于训练/部署数值语义。`depth_scale` 在 store 内保持一致以解释存储深度，不作为 Policy 兼容元数据。导出默认拒绝已有路径；`examples/export_policy_zarr.py --overwrite` 会先完成分块转换，再替换已有 canonical cache；失败保留新 staging，始终不修改 Raw。
H/W、depth scale、名义 dt 不同的数据分开导出，不静默 resize 或插值。

13 字段包括 joint_state、arm_qvel、arm_effort、hand_current、action、action_ee、contact_force、tactile_force、fingertip_points、eef_pose、rgb、depth、point_cloud。
关节和动作单位为 rad，arm_qvel 为 rad/s，arm_effort 保持 SDK 原生单位，hand_current 为驱动的 mA 语义。聚合/稠密触觉独立有效，保存驱动已扣偏置的 SDK 值，导出不重复扣偏置；电流不是触觉力。
RGB 保持 RGB 顺序和采集分辨率，depth 为对齐彩色的 uint16 配 depth scale；RGB 视频编码与深度对齐不等于全部原生相机字节无损。

桌面和安装/运动学按当前代码配置重建，不要求历史参数快照；相机参数仍使用对应 Raw 的信息。缺重建参数或空点云只使点云为 NaN，并在 export_report 中说明。
Zarr 的 `row_info` 保留 Raw 的逐行时间、源帧号和 dispatch，`meta/episode_ends` 保留 episode 边界；`dt` 仅为名义周期。缺可解码 RGB-D 或有效 depth scale 时不能完整导出，数值字段仍可单独读取。

[离线窗口示例](examples/read_policy_windows.py) 对 RGB 与触觉策略分别选所需字段和连续原行号窗口，使用 dispatch，不跨 episode 或缺失所需样本的行；不把时间抖动当缺测。
相邻 `dexmani_policy` 的现有 SequenceSampler 和 normalization 仍按完整 buffer 取样/统计，尚未处理此处的 NaN、dispatch 与真实时间；使用新缺测数据训练前必须适配，不能直接喂给旧 loader。固定频率模型的时间选样仍由训练约定决定。

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
4. fast research iteration；
5. simple and readable code。

长期文档只记录稳定的研究 workflow 和边界。具体实现细节放在源码、配置和 CLI help 中；一次性 task plan 与验收记录不写入长期架构说明；用户指定保留的根目录任务书除外。

面向代码代理的规则见 [AGENTS.md](AGENTS.md)，Claude 入口见 [CLAUDE.md](CLAUDE.md)。
