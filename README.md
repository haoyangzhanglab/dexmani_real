# DexMani Real

DexMani Real 是面向灵巧操作研究的真实机器人运行时，覆盖 xArm7（7 DoF）、
XHand（12 DoF）、Quest/HTS 手部跟踪与 RealSense RGB-D 的遥操作、数据采集、
物理回放、离线处理和 learned-policy 部署。

> 这是安全敏感的真实硬件软件。除明确标注为离线的命令外，运行前必须确认
> 硬件连接、工作空间、标定状态、急停条件和操作者授权。

## 当前能力

- VR 或键盘控制 xArm7，并可选联动 XHand。
- 通过共享内存协调 arm、hand、VR、camera、recorder 与 policy 进程。
- 常规 arm/hand joint command 统一经过安全门，并在设备 worker 的 SDK 边界再次
  校验；homing 使用独立的碰撞检查路径和专用 queue。
- 事务式写入 raw episode；当前 writer 写 v18，reader 与处理管线支持 v17/v18。
- 将 raw episode 清洗为 processed HDF5 v3，再导出 Policy Zarr v2。
- 物理回放已记录 episode，并保存回放轨迹与一致性指标。
- 通过可替换 backend/adapter 运行 joint-action learned policy；仓库包含无模型的
  deterministic fake 实现和 `dexmani_policy` 集成适配器。

## 导航

| 目标 | 入口 | 主要实现 |
|---|---|---|
| 理解仓库与修改约束 | [`AGENTS.md`](AGENTS.md)、[`code_style.md`](code_style.md) | [`repo_map.md`](repo_map.md) |
| VR 遥操作与采集 | [`examples/collect_teleop.py`](examples/collect_teleop.py) | [`teleop/session.py`](dexmani_real/teleop/session.py)、[`teleop/loop.py`](dexmani_real/teleop/loop.py)、[`teleop/operator_controls.py`](dexmani_real/teleop/operator_controls.py)、[`teleop/control_grid.py`](dexmani_real/teleop/control_grid.py)、[`teleop/action_proposal.py`](dexmani_real/teleop/action_proposal.py) |
| 键盘遥操作 | [`examples/keyboard_teleop.py`](examples/keyboard_teleop.py) | [`teleop/keyboard_session.py`](dexmani_real/teleop/keyboard_session.py) |
| 物理回放 | [`examples/replay_episode.py`](examples/replay_episode.py) | [`robot/replay_trajectory.py`](dexmani_real/robot/replay_trajectory.py)、[`robot/episode_replay.py`](dexmani_real/robot/episode_replay.py)、[`robot/replay_controller.py`](dexmani_real/robot/replay_controller.py)、[`robot/replay_evaluation.py`](dexmani_real/robot/replay_evaluation.py) |
| raw episode 读取/录制 | — | [`recording/episode_frame.py`](dexmani_real/recording/episode_frame.py)、[`recording/episode_recorder.py`](dexmani_real/recording/episode_recorder.py)、[`recording/episode_data_writer.py`](dexmani_real/recording/episode_data_writer.py)、[`recording/episode_reader.py`](dexmani_real/recording/episode_reader.py) |
| 离线清洗与 Zarr 导出 | [`examples/process_episodes.py`](examples/process_episodes.py)、[`examples/export_policy_zarr.py`](examples/export_policy_zarr.py) | [`data_processing/`](dexmani_real/data_processing) |
| learned-policy 部署 | [`examples/run_policy.py`](examples/run_policy.py) | [`deployment/`](dexmani_real/deployment)、[`integrations/`](dexmani_real/integrations) |
| 相机、点云与 VR 标定 | [`examples/`](examples) | [`sensor/camera_calibration.py`](dexmani_real/sensor/camera_calibration.py)、[`sensor/camera_calibration_control.py`](dexmani_real/sensor/camera_calibration_control.py)、[`sensor/camera_calibration_session.py`](dexmani_real/sensor/camera_calibration_session.py)、[`sensor/`](dexmani_real/sensor)、[`config/`](dexmani_real/config) |

完整的逐文件职责见 [`repo_map.md`](repo_map.md)。

## 核心架构

```text
RealSense / Quest-HTS / xArm7 / XHand
                 │
                 ▼
        device-specific workers
                 │
                 ▼
 SharedStorage: typed rings + queues + lifecycle state
          │                         │
          ├─ teleop coordinator     └─ inference worker → policy plan ring
          │                                      │
          └──────────────────────────────────────┤
                                                 ▼
                                      SafetyGate validation
                                                 │
                                      command publication
                                                 │
                                      device workers + SDK checks

 teleop fixed-grid samples → RecorderIO → data.h5 + depth.h5 + rgb.mp4
 raw episode → offline processing → processed HDF5 → Policy Zarr
```

必须保持的边界：

- 跨进程状态通过 `SharedStorage`；固定 shape/dtype 由
  [`utils/schema.py`](dexmani_real/utils/schema.py) 定义。
- xArm、XHand、RealSense 和 HTS SDK 对象只存在于各自 owner/worker 内。
- teleop、replay 和 deployment 负责动作决策；候选先由
  [`policy/safety_gate.py`](dexmani_real/policy/safety_gate.py) fail-closed 校验，再由
  [`policy/command_publication.py`](dexmani_real/policy/command_publication.py) 发布；设备
  worker 在 [`robot/command_validation.py`](dexmani_real/robot/command_validation.py) 再次校验。
  [`policy/safety.py`](dexmani_real/policy/safety.py) 仅保留兼容性出口和 hand-home helper。
  homing 由 [`robot/homing.py`](dexmani_real/robot/homing.py) 规划、校验并排入专用 queue。
- Recorder 只持久化选定的固定网格样本，不拥有机器人控制。
- `run_generation`、freshness、heartbeat、safety state 与 worker 侧检查共同使
  暂停、回零或故障前的旧命令失效。

### 控制与录制边界

- [`teleop/action_proposal.py`](dexmani_real/teleop/action_proposal.py) 只计算并限幅
  EEF、arm 与 hand proposal；它不发布命令、不访问 shared memory，也不写录制数据。
- [`teleop/loop.py`](dexmani_real/teleop/loop.py) 构造资源、等待 readiness 并调度两个
  窄职责模块：[`teleop/operator_controls.py`](dexmani_real/teleop/operator_controls.py) 处理
  BEGIN/pause/home/quit 与录制决策，[`teleop/control_grid.py`](dexmani_real/teleop/control_grid.py)
  完成单个 causal tick 的读取、proposal、校验、发布和采样。pause、VR/hand feedback
  异常、BEGIN audio gate 或录制终结进入 command-quiescence；恢复前必须重新锚定，期间不发布命令。
- RecorderIO 从 fixed-size shared-memory record 解码为不可变、拥有自身数组副本的
  `EpisodeFrame`，并拥有 episode transaction、camera sidecar、验证和有界 finalize；其中
  [`recording/episode_data_writer.py`](dexmani_real/recording/episode_data_writer.py) 是单个
  `data.h5` handle、dataset append 与 offset 的唯一 owner。两者都不拥有机器人命令或
  episode 的开始/停止决策。
- 相机标定的纯 ArUco/hand-eye 计算在
  [`sensor/camera_calibration.py`](dexmani_real/sensor/camera_calibration.py)；
  [`sensor/camera_calibration_control.py`](dexmani_real/sensor/camera_calibration_control.py)
  拥有 arm motion state、gated publish 和归位，`camera_calibration_session.py` 持有设备、GUI、
  采样和失败 cleanup。

## 环境

项目声明 Python `>=3.10`，目标开发环境是 conda 环境 `real_robot`。从仓库根目录：

```bash
conda activate real_robot
python -m pip install -e .
```

`pyproject.toml` 提供基础 Python 依赖，但不是完整的硬件/研究环境锁文件。
实际工作流还可能需要对应设备或功能的外部包，例如 xArm SDK、XHand controller、
RealSense SDK、HTS hand-tracking SDK、Pinocchio、MPlib、NLopt、
`dex-retargeting`、Open3D、PyTorch/PyTorch3D、Rerun，以及外部
`dexmani_policy` 仓库。按当前任务安装这些依赖，不要假设一次基础安装即可连接硬件。

配置的唯一合并优先级是：

```text
CLI override > YAML file > dexmani_real/config/defaults.py
```

运行时配置由 [`config/runtime.py`](dexmani_real/config/runtime.py) 校验、冻结并生成
SHA-256；部署配置由 [`deployment/config.py`](dexmani_real/deployment/config.py)
独立解析。可在不启动硬件的情况下查看遥操作解析结果：

```bash
python examples/collect_teleop.py --print-config
```

## 常用命令

### 硬件工作流

| 用途 | 命令 | 主要副作用 |
|---|---|---|
| VR 遥操作采集 | `python examples/collect_teleop.py --task-name <task>` | 连接 arm/hand/VR/camera；写 raw episode |
| VR 遥操作但不录制 | `python examples/collect_teleop.py --task-name <task> --no-record` | 连接 arm/hand/VR；不启动 camera/recorder |
| 键盘遥操作 | `python examples/keyboard_teleop.py` | 连接并控制 xArm7，可选 XHand |
| 物理回放 | `python examples/replay_episode.py episodes/<task>/episode_*` | 预检后控制 xArm7/XHand；写 `results/` |
| learned policy | `python examples/run_policy.py --deployment-config <file.yml>` | 启动 arm、可选 hand、inference 与 coordinator |
| 相机标定 | `python examples/calibrate_camera.py --hand-geometry <absent or secured-home>` | 连接 xArm/RealSense；更新相机标定；参数必须反映真实 XHand 安装状态 |
| VR 朝向标定 | `python examples/calibrate_vr_heading.py` | 连接 HTS；更新 VR transform |
| RealSense 诊断 | `python examples/realsense_record_example.py` | 打开相机与 GUI |
| 点云/桌面平面诊断 | `python examples/pointcloud_process_example.py` | 打开相机与 GUI；确认后写桌面平面 |
| XHand 独立诊断 | `python examples/xhand_control_example.py` | 连接并控制 XHand |

支持 argparse 的入口应先用 `--help` 查看当前参数；
`examples/xhand_control_example.py` 没有 `--help` 模式，执行即进入硬件流程。
不要从旧文档复制硬件地址或运动参数。

相机标定的 `--hand-geometry` 是物理事实声明：未安装 XHand 时使用 `absent`；已安装时，
只有在它实际固定于配置的 home 姿态时才能使用 `secured-home`。它不绕过碰撞检查。

### 离线数据工作流

先审计一个任务目录，不写输出：

```bash
python examples/process_episodes.py \
  --input-root episodes/<task> \
  --profile rgb_pc \
  --dry-run
```

确认审计结果后去掉 `--dry-run`，默认发布到
`episodes_processed/<task>/`。可选 profile 为 `joint`、`rgb`、`pointcloud`
和 `rgb_pc`。随后导出一个全新的 Zarr 目标：

```bash
python examples/export_policy_zarr.py \
  --input-root episodes_processed/<task> \
  --output dataset/<task>.zarr \
  --task-name <task>
```

输出目标已存在时会拒绝覆盖。可视化 raw episode：

```bash
python examples/visualize_episode.py episodes/<task>/episode_*
```

## 数据布局

```text
episodes/<task>/episode_<timestamp>/
├── data.h5       # fixed-grid robot/action/VR/quality data and metadata
├── depth.h5      # grid-aligned uint16 depth stream
└── rgb.mp4       # grid-aligned RGB stream

episodes_processed/<task>/
├── episode_<timestamp>.h5
└── processing_index.json

dataset/<task>.zarr/
├── data/*
└── meta/episode_ends
```

正式 raw writer 写 schema v18；reader、replay、visualizer 和离线处理支持 v17/v18。
更早的 flat HDF5 或 pre-v17 数据需要在运行时之外显式迁移。离线处理默认保守：
硬无效行可被移除，时序异常先审计；压紧后产生危险动作跳变时默认拒绝该轨迹。

## 开发与验证

安全的最低成本检查：

```bash
python -m compileall -q dexmani_real examples
git diff --check
git diff --stat
git status --short
```

仓库当前没有通用单元测试套件。针对纯函数、schema、reader、生命周期和 IPC
边界，优先写或运行不连接硬件的 deterministic check。仓库级 agent 工作约定见
[`AGENTS.md`](AGENTS.md)，具体编码规范见 [`code_style.md`](code_style.md)。
