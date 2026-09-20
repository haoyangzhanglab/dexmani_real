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

相机与 recorder 在遥操作里只是录制证据角色：读失败、源停滞或 recorder 异常只把会话结果
标记为失败（会话自然结束时非零），不会中断遥操作——相机停帧时 recording 层按
`camera_stall` 保存已采集前缀，遥操作继续。只有 arm/hand/VR 等控制角色失效才走物理故障
路径。

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

`/meta` 标记 `provenance_workflow="policy_eval"` 的 raw rollout 记录的是同步推理的实际
发布时间，在出现明确的 resampling / time-aware 契约之前仅用于评估：offline
processing 与 fixed-rate physical replay 都会 fail-closed 拒绝它，而不是静默按
fixed-dt teleop 解释或按名义控制周期压缩时间。

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

常用实验参数包括 artifact、inference steps、seed、trial 数量（`--num-episodes` 选择要运行的
trial 数：真正 begin 的 trial 结束时恰好计一次，与保存的 episode 数独立；录制失败不会提前
终止控制，只会在会话自然结束时使结果非零）、单 trial 时长和 device，例如：

```bash
python examples/run_policy.py <policy/task/experiment> \
  --num-episodes 2 \
  --inference-steps 4 \
  --seed 0
```

该入口 **始终连接真实硬件**。

Policy deployment 使用同步推理：单个 policy 子进程拥有 model/CUDA 和动作调度，
先完成加载与 warmup，再启动硬件 workers。每次 `policy.predict()` 返回
`PolicySpec.n_action_steps` 个动作，放入进程内队列，按 `PolicySpec.control_dt_s`
逐个下发；动作节奏仅由实际发布时间决定。上一块最后一个动作发布满一个控制周期后
才重新观测并同步推理；推理结束后新块首动作立即下发，不再额外等待一个控制周期。
下一个动作最早在上次实际发布加一个控制周期后下发；推理提前完成则等待剩余时间，
超时则从新的实际发布时间重新计时，不跳动作、不补发追赶。各模态历史共用同一查询
时刻锚点与 reference grid：普通 slot 取 source<=reference 的最新因果 source 帧，
只要求 source<=commit<=查询锚点；必需历史不可得时显式等待（WAIT）并在下一拍重查，
不拼凑未来帧或替代 source，也不做通用 max-age/skew 否决——旧而因果的帧可用。
慢推理只要传感器源持续前进就不视为设备故障；真实源停滞由 producer（相机/状态
worker 的失速闩）暴露。policy 子进程由进程存活与父侧 `max_running_s` 总预算监管
（预算在阻塞推理期间也能 generation-checked 地 fence 当前 epoch），不设专用心跳
deadline。模型/合同违规按 session failure 结束会话，不判定为物理故障。
Arm/Hand workers 继续各自拥有 SDK，下发动作不等待物理收敛。

### 导出并使用指定检查点

训练检查点需先在 `dexmani_policy` 中导出为部署文件。以下以 60% 进度的检查点为例，从本仓库根目录执行：

```bash
cd ../dexmani_policy
python -m dexmani_policy.deployment.export \
  experiments/maniflow/pick_place_toy/2026-09-15_01-28_0 \
  --checkpoint epoch=0539-step=00048000-milestone=60pct.pt
```

导出**始终**执行模型恢复和合成输入预测验证（没有跳过验证的开关），成功后在实验的 `checkpoints/` 下生成同名 `-deployment.pt` 文件，并将 `deployment_latest.pt` 更新为指向该文件。因此导出成功即代表该文件已完成 safe reload、严格恢复和一次确定性合成预测。

导出的模型与数据语义（architecture / constructor、action/window、normalization、dataset/preprocessing）取自 **selected checkpoint 自己保存的** contract，而不是 `dexmani_policy` 当前的 experiment `config.yaml`；训练后修改 config 中这些部分不会改变旧检查点的部署行为。

当前 config 仍提供 experiment identity 与 inference recipe：非 `best` 选择器读取 `eval.use_ema` 与 `eval.denoise_steps`，`best` 选择器读取 `best_ckpt.json` 中记录的同一组设置。因此只有这两项会随当前 config 变化，决定导出使用 raw 还是 EMA 权重以及 artifact 的默认 NFE；artifact 内只保存被选中的那一套权重。

导出在 publish 之前失败时，本次生成的 candidate 会被删除且 `deployment_latest.pt` 保持原值，同一条命令可直接重试。

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

其中保存 raw rollout 数据和 resolved run configuration。同步推理期间采样 hook 暂停，
记录保留实际时间戳，不补造样本；动作记录后一个控制周期内不追加普通 held 行。
固定 FPS 的 `rgb.mp4` 不代表真实控制时间，时序分析以 HDF5 `timestamp` 为准。
不生成异步 prediction trace sidecar。

每个真正 begin 的 trial 结束时恰好计一次（与保存 episode 数独立）；录制（evidence）失败
会使会话最终结果非零，但不会结束当前 trial 或取消剩余 trials。证据服务不可用、START
失败或旧录制仍在收尾时，有效 B 可以启动不录制 trial；同一 recorder 始终只有一个事务。
录制容量耗尽只保存已采前缀，控制继续；有效 raw 前缀不代表覆盖完整 trial。迟到保存结果
按原录制 trial 归属计数，目录名按 trial 序号命名。Session End 分别输出 control reason、
recording status 和 cleanup status；资源已确认释放与会话总体成功是独立事实。

Policy deployment 使用与 teleoperation / replay 相同的 runtime safety 与 command publication infrastructure。命令经单一 owner 的软投影（2π canonicalization、操作关节界限、软命令步长一次裁剪）后进入有界有序命令 FIFO；FIFO 满时是 non-blocking 可恢复背压，producer 保留同一 candidate 按主循环节奏重试。robot worker 在硬件边界只保留硬限位等最终检查，不重复拒绝同一软阈值。普通 IK 无解或 workspace miss 只丢弃未发布的 chunk 后缀，并在同一 trial 内重新观测（已提交前缀与命令连续性参考保留）；模型/合同违规与硬件故障才会结束 rollout。未执行动作不会被视作已完成的物理进度。

已有 legacy policy trace sidecar 的旧 rollout 可使用：

```bash
python examples/visualize_policy_rollout.py <rollout-episode> --info
```

进行离线检查；该工具不适用于没有该 sidecar 的同步 rollout。

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
