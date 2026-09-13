# DexMani Real

DexMani Real 是面向灵巧操作研究的真实机器人运行时，覆盖 xArm7、XHand、
Quest/HTS 手部跟踪与 RealSense RGB-D，主要用于遥操作与数据采集、物理回放、
离线数据处理、learned-policy rollout 和标定。

> **安全提示**：这是会连接并控制真实硬件的软件。运行任何可能连接设备、驱动机器人、
> home、replay、policy rollout 或写入标定的命令前，应先确认工作空间、标定、急停状态和
> 操作者授权。不要把 `examples/` 下的脚本默认当成离线工具；先看该入口的 docstring/`--help`。

## 环境与安装

项目要求 Python `>=3.10`。在已创建的 Python 环境中，从仓库根目录安装：

```bash
python -m pip install -e .
```

`pyproject.toml` 提供通用 Python 依赖。xArm、XHand、RealSense、HTS、运动学/规划以及
learned-policy 相关工作流还需要各自的外部 SDK 或研究依赖，请按实际任务安装。

运行时配置由 [`dexmani_real/config/experiment.py`](dexmani_real/config/experiment.py) 统一解析，
覆盖优先级为：

```text
CLI override > YAML file > dexmani_real/config/defaults.py
```

可以在不进入遥操作会话的情况下查看当前解析结果：

```bash
python examples/collect_teleop.py --print-config
```

## 主要工作流

下面只列 canonical entry point。完整参数以各脚本当前的 `--help` 为准。

| 工作流 | 入口 | 是否连接真实设备 | 主要结果 |
|---|---|---|---|
| VR 遥操作与采集 | `python examples/collect_teleop.py --task-name <task> --operator <name>` | 是 | raw episode（启用 recording 时） |
| 键盘遥操作 | `python examples/keyboard_teleop.py` | 是 | 交互式机器人控制 |
| 物理回放 | `python examples/replay_episode.py <episode>` | 是 | 物理 replay 与 replay evaluation 结果 |
| 离线 episode 处理 | `python examples/process_episodes.py episodes/<task> --dry-run` | 否 | 审计；去掉 `--dry-run` 后发布 processed HDF5 |
| Policy Zarr 导出 | `python examples/export_policy_zarr.py episodes_processed/<task> --dry-run` | 否 | 预检；去掉 `--dry-run` 后发布 `datasets/<task>.zarr` |
| Learned-policy rollout | `python examples/run_policy.py <policy/task/experiment>` | 是 | `rollouts/.../session_*` 下的记录与 resolved run config |
| Camera 标定 | `python examples/calibrate_camera.py --hand-geometry {absent,secured-home}` | 是 | xArm/RealSense eye-to-hand 标定并更新 camera calibration |
| VR 朝向标定 | `python examples/calibrate_vr_heading.py` | 仅 HTS/VR | 更新 `dexmani_real/config/vr_transform.json`；不控制机器人 |

### VR 遥操作与采集

`collect_teleop.py` 可以控制 xArm7/XHand，并在 recording 启用时将 raw episode 写入
`episodes/<task>/episode_*`。需要只调试 arm 时使用该入口提供的显式 `--no-hand`；需要关闭
recording 时使用 `--no-record`。两者的准确语义和其余参数以脚本 `--help` 为准。

### 物理回放

`replay_episode.py` 会把已记录轨迹真正发送到机器人，属于 hardware-affecting workflow。
它可以读取当前 raw episode，也可以通过脚本公开的 processed 模式回放 processed selection：

```bash
python examples/replay_episode.py episodes_processed/<task>/episode_<timestamp>.h5 --processed
```

运行前必须按真实机器人流程确认起始状态、碰撞环境和急停条件。

### 离线数据处理与导出

推荐先做只读 preflight：

```bash
python examples/process_episodes.py episodes/<task> --dry-run
python examples/export_policy_zarr.py episodes_processed/<task> --dry-run
```

确认后再执行实际发布：

```bash
python examples/process_episodes.py episodes/<task>
python examples/export_policy_zarr.py episodes_processed/<task>
```

processing 不修改原始 raw episode；export 从已验证的 processed HDF5 生成 policy Zarr。
当前 schema version、字段、dtype、shape 与 validation contract 由代码中的 schema/validator
定义，不在 README 复制一份易漂移的快照。

### 离线检查

raw 与 processed episode 都提供不连接硬件、不写文件的结构检查入口：

```bash
python examples/visualize_episode.py <raw-episode> --info
python examples/visualize_episode_processed.py <processed.h5> --info
```

去掉 `--info` 会打开对应的 Rerun viewer；完整参数以脚本 `--help` 为准。

### Learned-policy rollout

`run_policy.py` 运行一个 persistent recorded policy session。当前 CLI 直接以
`<policy/task/experiment>` 选择实验，并支持 artifact、inference steps、seed、episode 数量、
每个 episode 的运行时长预算和 device 等参数；完整接口以 `--help` 为准。

该入口 **始终连接真实硬件**。它会在启动 actuator/camera worker 之前先准备 inference child，
并为 session 写入 resolved `run_config.yaml`。任务成功与否在离线数据审核中判断；runtime 记录
技术停止原因和 episode 数据。

### Camera 标定

`--hand-geometry` 是必须由操作者显式给出的物理状态声明，而不是 geometry selector：
`absent` 仅用于没有安装 XHand；`secured-home` 仅用于已安装且物理固定在 configured home 的
XHand。当前 calibration collision checks 在两种声明下都使用 canonical fixed-home XHand
envelope，因此 `absent` 是保守建模。未来若引入经过验证的 arm-only collision model，这一用户
接口语义也必须同步更新。

## 核心架构

```text
xArm7 / XHand / RealSense / Quest-HTS
                │
                ▼
        device-specific owners
                │
                ▼
       RuntimeChannels / IPC
          │             │
          │             └─ policy inference
          │
          ├─ teleop
          └─ replay / deployment
                │
                ▼
       control safety boundary
                │
                ▼
       command publication
                │
                ▼
        robot worker / SDK

control-step recording → raw episode
raw episode → offline processing → processed HDF5 → Policy Zarr
```

长期应保持的边界只有少数几条：

- runtime 配置从 [`dexmani_real/config/`](dexmani_real/config) 的 canonical 定义解析；
- 跨进程通信与 shared-memory contract 由 [`dexmani_real/ipc/`](dexmani_real/ipc) 持有；
- live device/SDK state 留在对应 robot/sensor owner 中；
- teleop、replay 和 deployment 产生动作意图，安全与 command publication 由
  [`dexmani_real/control/`](dexmani_real/control) 的边界负责；
- recording 负责持久化数据，不拥有机器人动作决策；
- learned-policy 集成通过相邻 `dexmani_policy` 的 public deployment contract 完成，而不是依赖
  其私有 artifact 实现。

更细的 concurrency、freshness、schema、error-code 和 scheduling 语义属于当前 source，修改代码时
应直接追踪实际 producer/consumer，而不是依赖 README 中的历史描述。

## 数据与输出

主要数据流为：

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

learned-policy rollout 使用独立的：

```text
rollouts/<policy>/<task>/<experiment>/session_*/
```

当前数据合同的 source of truth：

- raw episode schema：[`dexmani_real/recording/storage/schema.py`](dexmani_real/recording/storage/schema.py)
- processed contract/validation：[`dexmani_real/dataset/`](dexmani_real/dataset)
- Policy Zarr export：[`dexmani_real/dataset/export.py`](dexmani_real/dataset/export.py)
- runtime/IPC wire contract：[`dexmani_real/ipc/`](dexmani_real/ipc)

历史数据迁移、一次性 salvage、实验统计和 incident 记录不属于 README 的长期接口；需要追溯时使用
Git history、issue/PR 或实验产物。

## Repository Layout

```text
dexmani_real/
├── calibration/   # camera / VR calibration workflows
├── config/        # canonical runtime defaults and resolved configuration
├── control/       # safety, homing, command publication
├── dataset/       # offline processing, validation, policy export
├── deployment/    # learned-policy runtime integration
├── ipc/           # shared-memory and process communication contracts
├── planning/      # kinematics, geometry, collision/path utilities
├── recording/     # raw recording and persistence
├── replay/        # physical replay
├── robot/         # robot workers, drivers, command validation
├── runtime/       # process/lifecycle supervision
├── sensor/        # camera, VR and point-cloud runtime code
├── teleop/        # VR/keyboard teleoperation
└── utils/         # small shared utilities

examples/           # user-facing entry points and diagnostics
assets/             # robot/resources used by the runtime
```

针对 coding agent 的仓库级工程与安全契约见 [`AGENTS.md`](AGENTS.md)。实现事实以当前 source、
schemas 和 resolved configuration 为准。

## 开发与验证

普通开发优先使用不触碰硬件的最低成本检查：

```bash
python -m compileall -q dexmani_real examples
git diff --check
git status --short
```

如果修改的子系统有现成的 focused offline validation，应按实际风险运行。不要把 example 程序当作
测试，也不要因为离线检查通过就声称完成了真实硬件验证。
