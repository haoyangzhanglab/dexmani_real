# DexMani Real

DexMani Real 是面向灵巧操作研究的真实机器人运行时，围绕 **xArm7 + XHand + RGB-D + VR/HTS** 构建，覆盖从真实机器人交互、数据采集到 learned-policy 部署与实验评估的完整研究流程。

项目主要面向以下场景：

- VR / keyboard teleoperation 与 demonstration collection；
- raw episode 记录、物理 replay 与离线数据处理；
- Policy Zarr 数据集导出；
- learned-policy 在真实机器人上的 rollout；
- camera / VR 标定；
- 机器人 runtime、IPC、recording 与 safety boundary 的统一管理。

> **安全提示**：本仓库包含会连接和控制真实硬件的程序。运行 home、teleoperation、replay、policy rollout 或 calibration 前，应确认机器人工作空间、标定状态、急停状态与操作者授权。`examples/` 下的脚本不应默认视为离线工具，请先查看对应入口的 docstring 或 `--help`。

## Hardware / Software Stack

典型实验系统包括：

- **Robot Arm**: xArm7
- **Dexterous Hand**: XHand
- **Vision**: RealSense RGB-D
- **Human Input**: Quest / HTS hand tracking
- **Policy**: 通过相邻 `dexmani_policy` 仓库的 public deployment interface 接入

DexMani Real 负责真实机器人侧的数据、控制与实验运行；policy model、training 和 deployment artifact 由 `dexmani_policy` 侧维护。

## Installation

项目要求 Python `>=3.10`。

在仓库根目录安装：

```bash
python -m pip install -e .
```

`pyproject.toml` 提供通用 Python 依赖。xArm、XHand、RealSense、HTS、运动学/规划以及 learned-policy 工作流还需要各自的外部 SDK 或研究依赖，请根据实验环境安装。

运行时配置由 [`dexmani_real/config/experiment.py`](dexmani_real/config/experiment.py) 统一解析：

```text
CLI override > YAML file > dexmani_real/config/defaults.py
```

查看当前解析配置：

```bash
python examples/collect_teleop.py --print-config
```

## Main Workflows

下面列出主要研究入口。完整参数以各脚本当前的 `--help` 为准。

| Workflow | Entry Point | Hardware | Main Output |
|---|---|---|---|
| VR teleoperation / collection | `python examples/collect_teleop.py --task-name <task> --operator <name>` | Yes | raw episode |
| Keyboard teleoperation | `python examples/keyboard_teleop.py` | Yes | interactive robot control |
| Physical replay | `python examples/replay_episode.py <episode>` | Yes | physical replay / evaluation |
| Offline episode processing | `python examples/process_episodes.py episodes/<task> --dry-run` | No | processed HDF5 |
| Policy Zarr export | `python examples/export_policy_zarr.py episodes_processed/<task> --dry-run` | No | `datasets/<task>.zarr` |
| Learned-policy rollout | `python examples/run_policy.py <policy/task/experiment>` | Yes | recorded rollout session |
| Camera calibration | `python examples/calibrate_camera.py --hand-geometry {absent,secured-home}` | Yes | camera calibration |
| VR heading calibration | `python examples/calibrate_vr_heading.py` | HTS/VR only | VR transform calibration |

## Teleoperation and Data Collection

### VR Teleoperation

`collect_teleop.py` 是主要 demonstration collection 入口，可以控制 xArm7 / XHand，并在 recording 启用时将 raw episode 写入：

```text
episodes/<task>/episode_*
```

常用模式：

```bash
# 标准 VR teleoperation + recording
python examples/collect_teleop.py --task-name <task> --operator <name>

# Arm-only debugging
python examples/collect_teleop.py --task-name <task> --operator <name> --no-hand

# Teleoperation without recording
python examples/collect_teleop.py --task-name <task> --operator <name> --no-record
```

### Keyboard Teleoperation

`keyboard_teleop.py` 提供 Cartesian jog，用于机器人调试、实验准备和低频人工控制。

```bash
python examples/keyboard_teleop.py
```

具体按键、运行条件和限制以脚本当前帮助信息为准。

## Physical Replay

`replay_episode.py` 用于将已记录动作重新发送到真实机器人，是 hardware-affecting workflow。

Raw replay：

```bash
python examples/replay_episode.py <episode>
```

Processed replay：

```bash
python examples/replay_episode.py episodes_processed/<task>/episode_<timestamp>.h5 --processed
```

运行 replay 前应确认机器人初始状态、场景布局、工作空间与急停条件。

## Offline Data Pipeline

DexMani Real 的数据流保持为：

```text
raw episode
    ↓
offline processing
    ↓
processed HDF5
    ↓
Policy Zarr
```

### 1. Process Raw Episodes

推荐先执行只读 preflight：

```bash
python examples/process_episodes.py episodes/<task> --dry-run
```

确认后执行：

```bash
python examples/process_episodes.py episodes/<task>
```

输出：

```text
episodes_processed/<task>/episode_*.h5
```

### 2. Export Policy Dataset

同样建议先进行 preflight：

```bash
python examples/export_policy_zarr.py episodes_processed/<task> --dry-run
```

实际导出：

```bash
python examples/export_policy_zarr.py episodes_processed/<task>
```

输出：

```text
datasets/<task>.zarr
```

Raw / processed schema、dtype、shape 和 validation contract 由代码中的 schema 与 validator 定义，不在 README 中重复维护易漂移的结构快照。

### Offline Inspection

Raw episode：

```bash
python examples/visualize_episode.py <raw-episode> --info
```

Processed episode：

```bash
python examples/visualize_episode_processed.py <processed.h5> --info
```

去掉 `--info` 可进入对应可视化流程。

## Learned-Policy Rollout

`run_policy.py` 是 learned-policy 真机评估入口，通过：

```text
<policy/task/experiment>
```

选择 `dexmani_policy` 中的 deployment experiment。

基本用法：

```bash
python examples/run_policy.py <policy/task/experiment>
```

常用实验参数包括 artifact、inference steps、replan interval、seed、episode 数量、单 episode 时长和 device，例如：

```bash
python examples/run_policy.py <policy/task/experiment> \
  --num-episodes 2 \
  --inference-steps 4 \
  --seed 0
```

该入口 **始终连接真实硬件**。

### 导出并使用指定检查点

训练检查点需先在 `dexmani_policy` 中导出为部署文件。以下以 60% 进度的检查点为例，从本仓库根目录执行：

```bash
cd ../dexmani_policy
python -m dexmani_policy.deployment.export \
  experiments/maniflow/pick_place_toy/2026-09-15_01-28_0 \
  --checkpoint epoch=0539-step=00048000-milestone=60pct.pt
```

导出默认执行模型恢复和合成输入预测验证，在实验的 `checkpoints/` 下生成同名 `-deployment.pt` 文件，并将 `deployment_latest.pt` 更新为指向该文件。

导出成功后，回到本仓库运行真机评估：

```bash
cd ../dexmani_real
python examples/run_policy.py \
  experiments/maniflow/pick_place_toy/2026-09-15_01-28_0 \
  --num-episodes 10 \
  --artifact epoch=0539-step=00048000-milestone=60pct-deployment.pt
```

`--artifact` 只填写 `checkpoints/` 内的部署文件名，不带目录路径；省略时使用 `deployment_latest.pt`。普通训练检查点不能直接作为部署文件使用。

### 评估输出

每个 policy evaluation session 使用独立输出目录：

```text
rollouts/<policy>/<task>/<experiment>/session_*/
```

其中保存 rollout 数据、resolved run configuration 和用于分析 raw policy prediction 的 policy trace。

Policy deployment 使用与 teleoperation / replay 相同的 runtime safety 与 command publication infrastructure。Policy output 会在进入硬件执行路径前经过必要的表示转换、约束处理与安全验证；robot worker 保留硬件边界处的最终检查。无法安全继续的 rollout 会结束，而不会把未执行动作视作已完成的物理进度。

Rollout 结果可使用：

```bash
python examples/visualize_policy_rollout.py <rollout-episode> --info
```

进行离线检查。

## Calibration

### Camera Calibration

```bash
python examples/calibrate_camera.py --hand-geometry {absent,secured-home}
```

用于 xArm / RealSense eye-to-hand calibration。

`--hand-geometry` 描述实验时真实的 hand 安装状态：

- `absent`: 未安装 XHand；
- `secured-home`: XHand 已安装并固定在 configured home。

### VR Heading Calibration

```bash
python examples/calibrate_vr_heading.py
```

用于更新 VR heading transform，不控制机器人本体。

## System Architecture

DexMani Real 将真实机器人运行时划分为少数明确边界：

```text
Sensors / Human Input              Learned Policy
        │                               │
        └──────── Observation ──────────┘
                        │
                        ▼
                Teleop / Deployment
                        │
                        ▼
                Control / Safety
                        │
                        ▼
              Command Publication
                        │
                        ▼
                Robot Workers
                        │
                        ▼
                    Hardware

        Recording ──→ Raw Episode
                         │
                         ▼
                Offline Processing
                         │
                         ▼
                    Policy Zarr
```

总体职责：

- `config/`：runtime configuration；
- `ipc/`：跨进程通信与 shared-memory contract；
- `sensor/`：camera / VR / point-cloud runtime；
- `teleop/`：人类输入到机器人动作意图；
- `deployment/`：learned-policy runtime integration；
- `control/`：command admission、safety 与 publication；
- `robot/`：robot workers、drivers 与 hardware boundary；
- `recording/`：raw episode persistence；
- `dataset/`：offline processing、validation 与 policy export；
- `planning/`：kinematics / geometry / planning utilities；
- `replay/`：physical trajectory replay。

更细的 concurrency、timeout、schema、error-code 与 scheduling 语义属于当前 source，应以实现为准。

## Data and Outputs

主要 demonstration 数据：

```text
episodes/<task>/episode_*/
  data.h5
  depth.h5
  rgb.mp4
        │
        ▼
episodes_processed/<task>/episode_*.h5
        │
        ▼
datasets/<task>.zarr
```

Learned-policy rollout：

```text
rollouts/<policy>/<task>/<experiment>/session_*/
```

Canonical data contracts：

- raw episode schema：[`dexmani_real/recording/storage/schema.py`](dexmani_real/recording/storage/schema.py)
- processed validation：[`dexmani_real/dataset/`](dexmani_real/dataset)
- Policy Zarr export：[`dexmani_real/dataset/export.py`](dexmani_real/dataset/export.py)
- runtime / IPC contract：[`dexmani_real/ipc/`](dexmani_real/ipc)

## Repository Layout

```text
dexmani_real/
├── calibration/   # camera / VR calibration
├── config/        # runtime configuration
├── control/       # safety and command publication
├── dataset/       # offline processing and export
├── deployment/    # learned-policy deployment
├── ipc/           # inter-process communication
├── planning/      # kinematics / geometry / planning
├── recording/     # raw episode recording
├── replay/        # physical replay
├── robot/         # robot workers and drivers
├── runtime/       # process / lifecycle management
├── sensor/        # camera / VR / point cloud
├── teleop/        # teleoperation
└── utils/         # shared utilities

examples/           # research-facing entry points
assets/             # robot models and runtime resources
```

## Development

仓库级 coding-agent 约束见 [`AGENTS.md`](AGENTS.md)。

普通开发优先执行不触碰真实硬件的最低成本检查：

```bash
python -m compileall -q dexmani_real examples
git diff --check
git status --short
```

不要因为离线检查通过就声称完成真实硬件验证。
