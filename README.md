# DexMani Real

DexMani Real 是面向灵巧操作研究的真实机器人运行时，覆盖 xArm7、XHand、Quest/HTS 手部跟踪与 RealSense RGB-D，主要用于：

- VR / keyboard teleoperation 与数据采集；
- raw episode 的物理回放、离线处理与 Policy Zarr 导出；
- learned-policy 在真实机器人上的 rollout；
- camera / VR 标定；
- 机器人 runtime、IPC、recording 与 safety boundary 的统一管理。

> **安全提示**：这是会连接并控制真实硬件的软件。运行任何可能连接设备、驱动机器人、home、replay、policy rollout 或写入标定的命令前，应先确认工作空间、标定、急停状态和操作者授权。不要把 `examples/` 下的脚本默认当成离线工具；先阅读对应入口的 docstring 或 `--help`。

## 环境与安装

项目要求 Python `>=3.10`。在已创建的 Python 环境中，从仓库根目录安装：

```bash
python -m pip install -e .
```

`pyproject.toml` 提供通用 Python 依赖。xArm、XHand、RealSense、HTS、运动学/规划以及 learned-policy 工作流还需要各自的外部 SDK 或研究依赖，请按实际任务安装。

运行时配置由 [`dexmani_real/config/experiment.py`](dexmani_real/config/experiment.py) 统一解析，覆盖优先级为：

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
| Learned-policy rollout | `python examples/run_policy.py <policy/task/experiment>` | 是 | `rollouts/.../session_*` 下的 rollout、trace 与 resolved run config |
| Camera 标定 | `python examples/calibrate_camera.py --hand-geometry {absent,secured-home}` | 是 | xArm/RealSense eye-to-hand 标定并更新 camera calibration |
| VR 朝向标定 | `python examples/calibrate_vr_heading.py` | 仅 HTS/VR | 更新 `dexmani_real/config/vr_transform.json`；不控制机器人 |

## 遥操作与采集

### VR 遥操作

`collect_teleop.py` 可以控制 xArm7/XHand，并在 recording 启用时将 raw episode 写入：

```text
episodes/<task>/episode_*
```

只调试 arm 时使用该入口提供的显式 `--no-hand`；关闭 recording 时使用 `--no-record`。准确语义和其余参数以脚本 `--help` 为准。

若 arm worker 在 SDK boundary 拒绝不连续目标，当前 motion 会被撤销，控制参考必须从新的有效 feedback 重新建立；不要把未被 robot 接受的 target 当作已经执行。

### 键盘遥操作

`keyboard_teleop.py` 使用 WASD / 方向键 / IJKL 进行 Cartesian jog。连续按键时，target 基于最新实测状态生成；正常释放按键后，控制端有界等待最后一个 arm target 的 acceptance，再结束当前 motion epoch。

IK、安全检查或 command acceptance 失败会结束当前 epoch。继续 jog 前应先释放当前按键组合，再重新输入命令。

## 物理回放

`replay_episode.py` 会把已记录轨迹真正发送到机器人，属于 hardware-affecting workflow。

它可以读取 raw episode，也可以通过 processed 模式回放 processed selection：

```bash
python examples/replay_episode.py episodes_processed/<task>/episode_<timestamp>.h5 --processed
```

运行前必须确认真实机器人起始状态、碰撞环境和急停条件。

如果 command 被 runtime/worker 拒绝，replay 不会把该失败伪装成成功执行，也不会自动跳过后继续补执行未来轨迹。硬件/controller fault 仍使用现有 fail-closed 路径。

## 离线数据处理与导出

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

当前 schema version、字段、dtype、shape 与 validation contract 由代码中的 schema / validator 定义，README 不复制易漂移的结构快照。

### 离线检查

raw 与 processed episode 都提供不连接硬件、不写文件的结构检查入口：

```bash
python examples/visualize_episode.py <raw-episode> --info
python examples/visualize_episode_processed.py <processed.h5> --info
```

去掉 `--info` 会打开对应 viewer；完整参数以脚本 `--help` 为准。

## Learned-policy rollout

`run_policy.py` 运行 persistent recorded policy session，通过：

```text
<policy/task/experiment>
```

选择相邻 `dexmani_policy` 中的 deployment experiment，并支持 artifact、inference steps、`--replan-steps`、seed、episode 数量、单 episode 运行时长和 device 等参数。

例如：

```bash
python examples/run_policy.py <policy/task/experiment> \
  --num-episodes 2 \
  --inference-steps 4 \
  --seed 0
```

该入口 **始终连接真实硬件**。session 会写入 resolved `run_config.yaml`；任务成功与否应在离线数据审核中判断，runtime 只记录技术停止原因与 rollout 数据。

### Policy action safety semantics

Learned-policy action 在真正进入 command publication 前先经过 physical-space shaping 和现有 safety validation。当前稳定语义是：

| 情况 | Runtime 行为 |
|---|---|
| 可安全修正的 joint / continuity 超界 | 在 physical action space 中 project/clip，再验证并发布 |
| 上一条 arm command 尚未被 worker 接受 | 等待；不发布下一条 arm command，不伪造执行进度 |
| feedback 暂时不可用或 stale | 等待；保持当前 trajectory state |
| action 因 wall-clock timing 已过期 | 按 timing stale 语义跳过 |
| EE IK 无可用解或实际 workspace violation | 结束当前 rollout，撤销到 `ARMED` |
| malformed target、post-projection invariant failure、SDK/controller/hardware fault | fail closed，进入现有 fault path |

Normal policy trajectory 只有在 command 成功进入 publication boundary 后才正常推进；**physical safety failure 不再等价于“跳过这个 waypoint，然后执行更未来的 waypoint”**。

Arm 的最终 command-continuity guard 仍由 arm worker 在 SDK boundary 持有；XHand 的逐 tick rate limiting 仍由 hand worker 持有。Policy executor 不复制底层 dynamics controller。

### Rollout diagnostics

Recorded policy session 使用独立目录：

```text
rollouts/<policy>/<task>/<experiment>/session_*/
```

每个 saved rollout 还会生成对应的 policy trace，保留 raw policy prediction；episode action 字段记录实际提交给 runtime 的 action target，因此可以对照 prediction 与 execution：

```bash
python examples/visualize_policy_rollout.py <rollout-episode> --info
```

这一区分很重要：生成模型输出可以超出 command envelope，而 runtime projection/validation 不应掩盖模型本身的分布质量问题。

## Camera 标定

`calibrate_camera.py` 的 `--hand-geometry` 是操作者对真实物理状态的显式声明：

- `absent`：未安装 XHand；
- `secured-home`：已安装且物理固定在 configured home。

标定 jog 从最新实测末端状态生成下一条增量 command；IK、安全检查、publication 或 acceptance 失败时结束当前 motion epoch，而不是继续累计未执行目标。

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

长期边界：

- runtime 配置由 [`dexmani_real/config/`](dexmani_real/config) 持有并统一解析；
- IPC / shared-memory contract 由 [`dexmani_real/ipc/`](dexmani_real/ipc) 持有；
- live SDK/device state 留在对应 robot/sensor owner 中；
- teleop、replay、deployment 产生动作意图，`control/` 与 worker boundary 负责 admission、publication 和最终硬件检查；
- arm worker 使用最后 SDK-accepted target 保持 command continuity；新 motion generation 从最新 measured state 重建参考；
- hand worker 负责 XHand 的 hardware-bound validation 与逐 tick command shaping；
- recording 只负责持久化，不拥有机器人动作决策；
- learned-policy 集成只依赖相邻 `dexmani_policy` 的 public deployment contract。

更细的 concurrency、timeout、schema、error-code 和 scheduling 实现属于当前 source；修改这些路径时应直接追踪 producer → representation → consumer → side effect，而不是依赖 README 中的历史描述。

## 数据与输出

主要数据流：

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

Learned-policy rollout 使用独立的：

```text
rollouts/<policy>/<task>/<experiment>/session_*/
```

当前数据合同的 source of truth：

- raw episode schema：[`dexmani_real/recording/storage/schema.py`](dexmani_real/recording/storage/schema.py)
- processed contract / validation：[`dexmani_real/dataset/`](dexmani_real/dataset)
- Policy Zarr export：[`dexmani_real/dataset/export.py`](dexmani_real/dataset/export.py)
- runtime / IPC wire contract：[`dexmani_real/ipc/`](dexmani_real/ipc)

历史 migration、一次性 salvage、实验统计和 incident 记录不属于 README 的长期接口；需要追溯时使用 Git history、issue/PR 或实验产物。

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

针对 coding agent 的仓库级工程与安全契约见 [`AGENTS.md`](AGENTS.md)。实现事实以当前 source、schemas 和 resolved configuration 为准。

## 开发与验证

普通开发优先使用不触碰硬件的最低成本检查：

```bash
python -m compileall -q dexmani_real examples
git diff --check
git status --short
```

如果修改的子系统有现成的 focused offline validation，应按实际风险运行。不要把 example 程序当作测试，也不要因为离线检查通过就声称完成了真实硬件验证。
