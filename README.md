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
- VR/键盘遥操作与物理回放的轨迹回放环节保留机器人自碰撞和静态障碍检查，但桌面不作为
  动作拒绝条件，以允许近桌面的精细抓取；homing（含回放 return_home）保留桌面安全验证。
- XHand 已知的抓取接触过流码 `1501035` 在发送位置命令时会被接受，不中止抓取，并记入
  episode 质量指标；同一码在状态读取失败时不生成伪造的新鲜反馈。持续数据不可用仍由
  新鲜度和 watchdog 边界处理。
- XHand 通信 CRC 码 `1501070` 在发送路径表示交付状态不明：worker 记录告警但不置位全局
  fault、不退出，也不更新该 action 的 SDK-acceptance ACK；下一周期从新鲜实测状态和仍有效的
  latest target 继续。读取路径仅在 12 关节反馈完整且有限时继续，并将该帧触觉标为无效。
  其他未列入白名单的发送错误仍 fail closed。
- XHand worker 成功连接并完成触觉初始化后只建立反馈与命令边界，不会在发布 ready 前
  隐式发送 `home_qpos`。手部运动必须来自工作流 owner 显式发起的 home 或已校验命令。
- physical replay 以记录的首帧**实测 arm**关节状态作为起点：xArm 需先由操作者受监督地
  定位；XHand 连接后由 replay 显式发送首帧手部命令完成 3 秒安全 warm-up，并经实时
  自碰撞/静态障碍检查和新鲜反馈确认后才从 frame 0 开始重放。手部反馈不要求复现录制值。
- RealSense 相机按设备原生频率连续采集；16 Hz 控制网格只选择最新严格因果帧，
  不再将相机发布节拍绑定到控制频率。
- 事务式写入 raw v28：16 Hz control step 是 observation/action timeline，camera、arm、
  hand、aggregate contact 独立按 control anchor 取最新有效观测；保留 raw dense tactile、
  calibration、camera health 与真实 source timestamps。aggregate 有独立
  `hand_contact_source_monotonic_ns`，不伪装成更新的 hand qpos/dense 来源。
- raw → processed v18 → Policy Zarr v11 全程逐行对应：accepted N 行保持 N 行，一个
  processed 文件对应一个 Zarr episode。唯一自动行为质量拒绝是 persistent IK；
  技术损坏可使整条失败，camera/tactile timing jitter 不删行，也不跨行修复。
- processed 是完整 multimodal superset：joint/action/action_ee、aggregate contact（含 validity）、
  dense tactile（含 validity）、fingertip、native RGB-D、camera geometry、point cloud 与 timing。
  FK 保留，但不持久化可重算的 eef_pose。
- 物理回放已记录 episode，并保存回放轨迹与一致性指标。
- 通过 Policy-owned public runtime 与 Real-owned NumPy adapter 运行 joint/EE-action learned
  policy；Policy strict restore 模型，Real fail-closed 校验固定硬件、观测、IPC 与时序兼容。

## 导航

| 目标 | 入口 | 主要实现 |
|---|---|---|
| 理解仓库与修改约束 | [`AGENTS.md`](AGENTS.md)、[`code_style.md`](code_style.md) | [`repo_map.md`](repo_map.md)、[`user_design.md`](user_design.md) |
| VR 遥操作与采集 | [`examples/collect_teleop.py`](examples/collect_teleop.py) | [`teleop/session.py`](dexmani_real/teleop/session.py)、[`teleop/loop.py`](dexmani_real/teleop/loop.py)、[`teleop/control_loop/grid.py`](dexmani_real/teleop/control_loop/grid.py)、[`teleop/control_loop/action_proposal.py`](dexmani_real/teleop/control_loop/action_proposal.py) |
| 键盘遥操作 | [`examples/keyboard_teleop.py`](examples/keyboard_teleop.py) | [`teleop/keyboard_session.py`](dexmani_real/teleop/keyboard_session.py)、[`docs/teleop_jitter_incident.md`](docs/teleop_jitter_incident.md) |
| 物理回放 | [`examples/replay_episode.py`](examples/replay_episode.py) | [`replay/`](dexmani_real/replay) |
| raw episode 读取/录制 | — | [`recording/frame.py`](dexmani_real/recording/frame.py)、[`recording/recorder.py`](dexmani_real/recording/recorder.py)、[`recording/storage/hdf5_writer.py`](dexmani_real/recording/storage/hdf5_writer.py)、[`recording/storage/reader.py`](dexmani_real/recording/storage/reader.py) |
| 离线处理与 Zarr 导出 | [`examples/process_episodes.py`](examples/process_episodes.py)、[`examples/export_policy_zarr.py`](examples/export_policy_zarr.py) | [`dataset/`](dexmani_real/dataset) |
| 数据 schema 参考 | [`docs/data_schema.md`](docs/data_schema.md) | raw v28、processed v18 与 Policy Zarr v11 的字段、dtype、shape 与语义 |
| learned-policy 部署与评估会话 | [`examples/run_policy.py`](examples/run_policy.py) | [`deployment/`](dexmani_real/deployment)、[`deployment/inference/dexmani_policy.py`](dexmani_real/deployment/inference/dexmani_policy.py) |
| 相机、桌面与 VR 标定 | [`examples/`](examples) | [`calibration/`](dexmani_real/calibration)、[`sensor/`](dexmani_real/sensor)、[`config/`](dexmani_real/config) |
| 点云完整链路 | [`docs/pointcloud_pipeline.md`](docs/pointcloud_pipeline.md) | [`sensor/pointcloud.py`](dexmani_real/sensor/pointcloud.py)、[`sensor/pointcloud_worker.py`](dexmani_real/sensor/pointcloud_worker.py) |

稳定的运行拓扑、数据流和安全边界导航见 [`repo_map.md`](repo_map.md)。

## 核心架构

```text
RealSense / Quest-HTS / xArm7 / XHand
                 │
                 ▼
        device-specific workers
                 │
                 ▼
 RuntimeChannels: typed rings + queues + lifecycle state
          │                         │
          ├─ teleop control loop    └─ inference worker → prediction ring → policy executor
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

- 跨进程状态通过 `RuntimeChannels`；固定 wire shape/dtype 由
  [`ipc/schema.py`](dexmani_real/ipc/schema.py) 定义，机器人模型 shape 与关节顺序由
  [`robot/model.py`](dexmani_real/robot/model.py) 定义。
- xArm、XHand、RealSense 和 HTS SDK 对象只存在于各自 owner/worker 内。
- teleop、replay 和 deployment 负责动作决策；候选先由
  [`control/safety_gate.py`](dexmani_real/control/safety_gate.py) fail-closed 校验，再由
  [`control/publication.py`](dexmani_real/control/publication.py) 发布；设备
  worker 在 [`robot/command_validation.py`](dexmani_real/robot/command_validation.py) 再次校验。
  arm/hand homing 分别由 [`control/arm_homing.py`](dexmani_real/control/arm_homing.py) 和
  [`control/hand_homing.py`](dexmani_real/control/hand_homing.py) 拥有。
- Recorder 每个 controller sample 写一行，保留实际时间缺口，不二次对齐或生成 HOLD；不拥有机器人控制。
- `run_generation`、freshness、heartbeat、safety state 与 worker 侧检查共同使
  暂停、回零或故障前的旧命令失效。

### 控制与录制边界

- [`teleop/control_loop/action_proposal.py`](dexmani_real/teleop/control_loop/action_proposal.py) 只计算并限幅
  EEF、arm 与 hand proposal；它不发布命令、不访问 shared memory，也不写录制数据。
- [`teleop/session.py`](dexmani_real/teleop/session.py) 在父进程中分阶段启动 worker，并以
  ready、fault、liveness 和 timeout 组成有界启动屏障；[`teleop/loop.py`](dexmani_real/teleop/loop.py)
  构造控制资源，并由
  [`teleop/control_loop/grid.py`](dexmani_real/teleop/control_loop/grid.py) 完成单个 causal tick 的读取、
  proposal、校验、发布和采样。pause、VR/hand feedback 异常或录制终结进入 pause boundary；
  BEGIN 音频只是 best-effort 操作者反馈，恢复运动前必须等待 fresh causal feedback re-anchor，
  期间不发布命令。同步 home 是有意的控制静默区间，返回后由 loop 同时重锚控制 loop 和
  control-grid 时钟，不计作遥操作栅格丢失。
- RecorderIO 从 fixed-size shared-memory record 按逻辑 sequence 严格、连续地取得所有权；
  它只复制尚未确认的 slot，缺失或超出环容量时丢弃 active episode，绝不跳过样本。随后它将
  record 解码为拥有自身数组副本的 `EpisodeFrame`，并拥有 episode transaction、
  camera sidecar、验证和有界 finalize。其轮询是非执行器的服务循环；周期性批量持久化允许
  超过单次轮询周期，健康性以 sample backlog、sequence 连续性和 writer 状态为准，而不是
  轮询相位。其中
  [`recording/storage/hdf5_writer.py`](dexmani_real/recording/storage/hdf5_writer.py) 是单个
  `data.h5` handle、dataset append 与 offset 的唯一 owner。两者都不拥有机器人命令或
  episode 的开始/停止决策。
- 录制 START/STOP 与 Started/Finished 通过低频 Queue 传递，`RecorderClient` 是结果的唯一
  消费者。START 确认后才生产样本；STOP 先停止生产、快照 `through_sequence`，RecorderIO
  排空该边界内的全部样本后 flush、close、原子发布，再返回一个 Finished。启动超时由现有
  supervisor 中止会话，不建立迟到取消或 generation/FSM 协议。
- 相机标定的纯 ArUco/hand-eye 计算、运动控制与 side-effect lifecycle 分别位于
  [`calibration/camera/solver.py`](dexmani_real/calibration/camera/solver.py)、
  [`calibration/camera/motion.py`](dexmani_real/calibration/camera/motion.py) 与
  [`calibration/camera/session.py`](dexmani_real/calibration/camera/session.py)。采样仅在相机
  window 前后 arm feedback 都 fresh/healthy、速度满足既有 homing stationary bound、且 joint
  drift 不超过既有 convergence bound 时接受；失败只丢弃 sample。写入的
  `calibration_capture` 是 diagnostic provenance，不参与 runtime intrinsics 读取。

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

Real runtime 配置由 [`config/experiment.py`](dexmani_real/config/experiment.py) 加载为独立的 nested dataclass，并在加载边界统一校验；未知 YAML 字段会报错，`--print-config` 现场输出实际配置。learned-policy 的 observation freshness、command progress、action validity 和 watchdog timing
属于其 `policy` 段，模型 shape、modality、horizon 和 inference cadence 则来自 Policy public
API 的 `PolicySpec`。Real 只校验 `PolicySpec` 的公开契约字段（observation fields / shape /
dtype / 相应字段的 semantics、`requires_hand`、`chunk_size`、`n_action_steps`、`action_key`、`control_action_dim`、`control_dt_s`），
不解析 Policy artifact 内部、`best_ckpt.json` 或历史格式；调度也不会对 action chunk 做
temporal blending。recorded policy 会话使用唯一周期调度；每个 episode 使用 `--max-duration`
运行预算。
`pointcloud` 是与 EEF `policy.workspace` 分离的感知配置段；实时 worker、离线
重建和诊断入口只从该段派生参数，并把点云策略与桌面语义写入 processed/Zarr。
可在不启动硬件的情况下查看遥操作解析结果：

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
| 物理回放 | `python examples/replay_episode.py episodes/<task>/episode_*` | 回放 recorded published arm target 与 recorded logical hand target；当前 runtime/geometry 完整预检后控制 xArm7/XHand，hand worker 生成受限 SDK 中间 setpoint；output target 必须缺失或为空目录，默认写入 `replay_results/` |
| 回放 processed HDF5 | `python examples/replay_episode.py episodes_processed/<task>/episode_<timestamp>.h5 --processed` | processed 验证后按 `source_path` 读取完整 raw `float64` arm target 与 logical hand target；执行完整 live-start、limits、workspace 与 collision 预检，不发送 processed float32 action；当前 reader 不支持历史 v25/v26 physical replay |

| learned policy 评估会话 | `python examples/run_policy.py <policy/task/experiment> [--artifact A] [--inference-steps N] [--seed S] [--num-episodes N] [--max-duration SEC] [--device D]` | 先 inference restore+warmup 就绪，再启动 arm/hand/camera/recorder；H→B→S 循环录制 raw-v28 episode 到 `rollouts/.../session_*/episode_NNN/`；N/N 后自动 clean shutdown |
| 相机标定 | `python examples/calibrate_camera.py --hand-geometry <absent or secured-home>` | 连接 xArm/RealSense；更新相机标定；参数必须反映真实 XHand 安装状态 |
| VR 朝向标定 | `python examples/calibrate_vr_heading.py` | 连接 HTS；更新 VR transform |
| RealSense 点云交互诊断 | `python examples/realsense_record_example.py` | 只连接相机；GUI 切换完整 RAW/处理后点云，不写标定 |
| 桌面点云与标定诊断 | `python examples/pointcloud_process_example.py [--save-dir outputs/pointcloud_diagnostics]` | 只连接相机；可保存 aligned RGB-D、raw/processed 点云与离线重建 metadata；仅在显式确认后更新共享桌面标定 |
| XHand 独立诊断 | `python examples/xhand_control_example.py` | 连接并控制 XHand |

`--no-hand` 仅用于 arm bring-up/debug，必须同时禁用录制（`--no-record`）；项目不提供 arm-only dataset contract。

支持 argparse 的入口应先用 `--help` 查看当前参数；
`examples/xhand_control_example.py` 没有 `--help` 模式，执行即进入硬件流程。
不要从旧文档复制硬件地址或运动参数。

相机标定的 `--hand-geometry` 是物理事实声明：未安装 XHand 时使用 `absent`；已安装时，
只有在它实际固定于配置的 home 姿态时才能使用 `secured-home`。它不绕过碰撞检查。

### `run_policy.py` 用法

`run_policy.py` 只有一个 workflow：选择 experiment / artifact / inference 设置，然后完成若干
真实 rollout。

```bash
python examples/run_policy.py <policy/task/experiment>
```

论文实验：

```bash
python examples/run_policy.py <policy/task/experiment> \
  --artifact epoch_500-deployment.pt \
  --inference-steps 2 \
  --seed 0 \
  --num-episodes 10 \
  --max-duration 60
```

参数默认：`artifact=deployment_latest.pt`（父进程解析成真实文件名并 pin 给 inference child，
防止 inspect 后重新 export 造成 child 加载另一份）、`inference_steps=artifact default`、
`seed=0`、`num_episodes=1`、`max_duration=60 s`、`device=cuda:0`。`--seed` 是会话固定的
per-episode inference seed（每个 episode `reset_episode()` 恢复同一 seed）；比较不同随机 seed
应启动不同会话。

启动顺序是 inference-first：先启动 inference child，完成 strict restore + warmup 并置 ready，
才启动 arm/hand/camera/recorder。启动前父进程创建
`rollouts/<policy>/<task>/<experiment>/session_YYYYMMDD_HHMMSS/`（同秒冲突加 `_01` 后缀），
并写入只含 resolved 实验条件的 `run_config.yaml`（experiment、artifact、inference_steps、
seed、num_episodes、max_duration_s）；写失败则不启动任何硬件进程。不保存 git commit/SHA-256/
GPU/metrics 等运行元数据。

每个 episode 的流程是 `H → 人工摆场景 → B → S`：

```text
process start → inference/hardware ready → ARMED
  Episode i/N:  H → scene setup → B → RUNNING+recording → S/timeout → 回 ARMED
N/N 成功发布后自动 clean shutdown
```

按键只有五个：`B` 开始、`S` 停止并保存、`H` home、`Q` 退出会话、`ESC` 急停。
`S` 立即 fence motion、保存当前 raw episode 后回到 ARMED，**不表示 task SUCCESS**。
每个 episode 都必须先完成一次新的 `H` 才能 `B`。`Q` 在 ARMED 是 clean quit；在 RUNNING 先
fence motion、保存当前 partial episode、等待 recorder finalization 后退出。
`--max-duration` 是每个 episode 从 `B` 起的最大运行秒数，超时 fence + 保存 + `stop_reason=timeout`。

每个 B 在完成 `H → B` 的物理 home 前置条件后，必须先获得 RecorderIO 的 `RecordingStarted`
确认，才进入 RUNNING；这是唯一 startup recording barrier。控制先完成 publish/reject，再记录
结果，不另建第二套 evidence 事务。recording integrity failure（`saved=False` 或 `error`）
不计数、fail closed 终止会话，不自动重试。`num_episodes` 只统计成功 publish 到磁盘的 raw
episode；达到 N 后自动 quit。raw 的 `/meta/success` 仅表示 storage transaction 提交成功，
不表示 task success。RecorderIO 独占录制 lifecycle，并在后台 finalization 时继续
heartbeat/control polling；只有线程和文件资源已安全回收、完成消息已发布，才允许下一次 START。

### Learned policy 实时点云

完整的 experiment 选择与按键边界由 [`examples/run_policy.py`](examples/run_policy.py) 定义；
该命令会连接真实设备（inference preflight 优先于硬件），必须按硬件流程处理。

`PolicySpec.observation_fields` 包含 `point_cloud` 时，lifecycle 才启动 camera 与独立
point-cloud worker。worker 始终读取最新的 depth-to-color aligned RGB-D，旧帧不会排队；inference 仅在
当前 RUNNING epoch 之后的点云 T 历史窗完整时才推理。窗口以因果截点前最新已过去的控制 tick 为末端，按策略控制网格选取严格递增、
不重复的 camera frame；每帧不得晚于对应逻辑 tick，lag 不超过 `max_grid_lag_s`，且跨帧
`camera_generation` 一致。每个点云均因果配对到不晚于它、并处于
`max_observation_skew_s` 内的 arm/hand 状态，不插值、填充或复用旧 run 数据。每帧为 xArm-base
`float32 (N, 6)`，列语义为 `xyzrgb`，RGB 范围为 `[0,1]`。`pointcloud_num_points` 只允许
`1024`、`2048`、`4096`、`8192`，部署时必须与 PolicySpec 和 Real runtime 精确一致。

`PolicyExecutor` 将通过安全门的策略 endpoint 原子写为单条 coupled command，并立即返回，不在控制热路径
等待 worker 握手。ring sequence 是实际的传输 epoch；arm/hand worker 在各自 SDK 边界复核同一个
`(run_generation, ring_sequence, valid_until_monotonic_ns)` ticket，已被新 record 覆盖、过期或被 motion
permit 撤销的 endpoint 不再执行。B/S 请求与 RUNNING 状态转换由同一个 `motion_lock` 排序；candidate
保留其来源 prediction 的 generation，policy publication 在写 ring 的原子边界还必须满足 RUNNING + 同 generation。
`action_id` 保留在 record 与反馈中用于审计和 worker acceptance watermark，不参与 ownership 判定。这保证软件
IPC 记录一致且保持 latest-wins 实时性，不表示两个执行器物理同步或已完成动作。

实时、离线处理和诊断入口共用 `PointCloudConfig` 与同一个生产 builder。处理顺序为：
aligned depth 范围/OpenCV 3×3 局部支持与边缘四方向支撑过滤 → 使用缓存 color-camera ray 的桌面高度
迟滞裁减（在反投影前，以可靠高点连通保护物体低处表面）→ color-frame 反投影 → xArm-base
变换与 workspace 裁减 → XYZ/RGB 均值体素 →
单次 radius graph 邻居密度/连通域过滤 → 空间候选上限 → 15 mm 粗体素分层的确定性
固定 N 采样。感知
workspace 当前为 x `[0.0,0.8]`、y `[-0.5,0.5]`、z `[0.0,0.8]` m；体素 RGB 是其源
aligned 像素 RGB 的均值。processed HDF5 和 Policy Zarr 同时保存并校验算法、策略与桌面平面
身份，禁止混合不同点云语义的数据。

deployment lifecycle 从 Policy public API 取得只读 `PolicySpec`。唯一的
`dexmani.deployment.v3` artifact 只包含 `_format`、`contract` 和 `weights`；其
`data_contract.observation_fields` 按顺序声明每个原始模型
输入的名称、shape、dtype 与语义；训练 experiment 的 `dataset.sensor_modalities` 是导出时唯一的
人工选择入口，artifact 不再持久化第二份模态列表。`PolicySpec.observation_fields[*].semantics`
属于 Real public compatibility contract：EEF 的 frame/rotation/derivation、tactile 的
sensor/point/finger 顺序与单位声明、fingertip derivation，以及 point-cloud preprocessing
identity 都参与 fail-closed 兼容性校验。Real 验证自己要投影的 raw shape/dtype 和相应 semantics、
19/21D control action、XHand、控制周期、`chunk_size` 与 Prediction IPC capacity，并只启动所需 sensor worker。
若请求 point cloud，还会在 channel/worker 创建前逐项校验 preprocessing identity（frame、颜色来源、
policy/table/sampling/transform）；若请求 fingertip，则校验 derivation 与 policy ID。任一 mismatch
fail closed，deployment schema 仍保持 v3。
checkpoint/Hydra/EMA/normalizer/denoise 与 RGB model preprocessing 全部由
`dexmani_policy.deployment.load_experiment()` 拥有；Real adapter 只透传 canonical NumPy
observation 并转换 action。Policy export/restore 会核对模型 encoder 实际消费的字段；没有对应
encoder 的字段组合会 fail closed，不会静默忽略输入。
inference worker 完成 restore 与 warmup 后才置 ready，lifecycle 才允许启动其余
hardware workers。内部只传播 `execute: bool`：`False` 走完整 candidate validation 后返回，
不会调用 publication；`True` 发布同一 generation/ticket 的 arm + hand command，worker 仅在
candidate validity window 内接受目标。`PolicyExecutor` 的正常策略发布不逐 endpoint 等待 acceptance，
而是分别监控 arm `last_cmd_seq` 与 hand `accepted_target_action_id`；latest-wins 可跳过中间 ID，但持续
存在已发布目标且任一水位在 `command_progress_timeout_s` 内不前进会 fail closed。会话 `--seed`
默认 0，记录在 `run_config.yaml`；XHand 需求由 `PolicySpec.requires_hand` 决定，不由 CLI
重复声明。

inference 每次只发布一个带 observation provenance 和 inference timing 的 flat `Prediction` IPC record 到单槽 latest-wins
ring；`predict_action_chunk()` 的输出在 inference 边界转换并检查为有限 `float64[chunk_size,D]`。
三个必填 timing 字段 `inference_latency_ms`、`observation_age_ms`、`observation_skew_ms`
对应生成该 chunk 的同一次 observation / inference，IPC 使用 `float64`。metrics 只保存最新标量和计数器；无效诊断值不进入结果，也不使有效 rollout 失败。
inference 按 `n_action_steps * control_dt_s` 的绝对 cadence 查询，计算期间旧 future tail 继续可用。
新 prediction 的 `target_time <= now` 前缀直接丢弃；全过期 prediction 丢弃并保留已有合法计划或 hold。
executor 在每个 target 前一个控制周期内发布该未来目标，以 control grid 限制发布频率；
错过的 target 不补发，decode/IK 后再次检查 target 尚未过期，不产生 catch-up burst。
SDK 边界仍使用原有 command validity、generation 和最新命令检查。

实时策略发布不做软件碰撞插值，保留关节限位、工作空间、时效和 arm/hand 异步进度 watchdog；碰撞响应由
机械臂控制器灵敏度与现场操作员负责。`H` 的 `return_home` 仍使用手部、静态环境和桌面的
完整碰撞检查路径。

policy executor 每秒输出 live metrics，并在每个 B→停止/中止/故障边界输出 compact
episode summary。rejection 等计数覆盖本次 rollout，不会被每秒日志清零；timing snapshot 是最近
样本，不是全 episode 的 latency 分位数，也不写入磁盘。运行时只记录 technical stop reason——
操作员停止为 `operator`，时长上限为 `timeout`，首次命令与命令静默超时为
`first_command_timeout`/`command_silence_timeout`，其余为 `quit`/`hardware_fault`/`estop`/
`recorder_fault`/`recording_failure`/`max_frames`/`runtime_shutdown`。任务 SUCCESS/FAILURE/EXCLUDE
由离线分析（`EpisodeReader` + `visualize_episode.py` + 人工标注）决定，运行时不做判断。

点云缺失、过期、shape/dtype 错误、非有限值或颜色越界时 inference fail closed，不发布
新的 Prediction。实时路径当前仅支持静态 `eye_to_hand` 标定；`eye_in_hand` 需要另行建立与
相机帧同步的机械臂位姿合同。离线 IPC、合成 RGB-D 和已保存的真实 L515 帧均已验证；
实时 worker 不再收集滚动性能分位数，部署总时延仍须在完整硬件环境中单独测量。
`pointcloud_process_example.py` 同时报告纯构建与 capture-to-cloud 的 p50/p95/max，并报告
深度滤波、桌面裁减、反投影、体素和离群过滤等逐阶段 p95；纯构建目标为 p95 < 40 ms。
传入 `--save-dir outputs/pointcloud_diagnostics` 时，它还会把 post-calibration 同一帧的 aligned
RGB-D、完整 raw 点云、canonical processed 点云、相机几何、外参、桌面平面和点云配置原子保存为
独立目录快照；不传该参数时不写文件。

### L515 RGB/Depth 时序限制

暗场下 RGB 曾因 Auto-Exposure Priority=ON 把曝光拉到 ~60 ms 而降帧到 ~16.7 Hz。
根因已确认并修复：`RealSenseCameraConfig.auto_exposure_priority` 默认 `0.0`（OFF，Auto
Exposure 仍 ON），RGB 恢复 30 Hz，亮度由增益补偿、几乎不变（暗场噪声上升）。

深度与颜色流仍然不是同时曝光（两路曝光/时间戳存在 skew）。processed 的点云将
depth-to-color aligned 像素 RGB 聚合为体素颜色，但这不表示同步曝光；运动物体仍可能出现
颜色时间错位。

RGB-D 点云/桌面交互诊断入口为 `examples/realsense_record_example.py` 与
`examples/pointcloud_process_example.py`。前者可在完整 RAW 点云和 canonical 处理后点云之间
切换：`p` 显示点云，`s` 切换 RAW/PROCESSED，`f/r` 分别冻结和重置。后者先询问是否执行
桌面标定；选择后会提示清空桌面并采集 5 帧，以确定性 RANSAC 重新拟合桌面。新拟合在本次
运行立即使用，只有操作者输入 `y` 才会备份并原子更新
`dexmani_real/config/desk_plane.json`。该文件同时服务感知裁减和桌面碰撞几何。

### 离线数据工作流

先审计一个任务目录或单个 episode，不写输出：

```bash
python examples/process_episodes.py \
  episodes/<task> \
  --dry-run
```

传入 `episodes/<task>/episode_*` 时只处理该 episode；传入 `episodes/<task>` 时处理其下全部 `episode_*` 子目录。

确认审计结果后去掉 `--dry-run`，默认发布到
`episodes_processed/<task>/`。canonical 输出始终是完整 multimodal superset（不再有
modality-specific profile）；`--pointcloud-num-points` 可选 `1024`、`2048`、`4096` 或
`8192`，默认 `1024`。

当前 runtime reader、raw viewer 和 physical replay 只接受 raw v28，检查必需文件、
shape/dtype 与 sidecar 帧数，不在 reader/finalize 完整解码 RGB。
离线 processing 另有一个窄的 normalized-v26 salvage 路径；历史 v24/v25 必须先依据明确
lineage 使用冻结转换器生成 out-of-place v26，不能伪装成 v28 或改写原始数据。
见 [历史迁移](docs/raw_v24_migration.md) 与
[实际 salvage 证据](docs/invalid_frames_export_incident.md)。

要生成 learned-policy 训练数据，在发布
前先使用明确的 task identity 做 dry-run：

```bash
python examples/process_episodes.py \
  episodes/<task> \
  --task-name <task> \
  --dry-run
```

确认后去掉 `--dry-run`。每个 accepted episode 的 task identity 按全局 `--task-name`、
annotation `task_name`、raw `task_label` 的顺序解析；必须是非空、无首尾空白、无 ASCII 控制字符（C0/DEL）且
不为 `unknown` 的字符串。整批 accepted episodes 必须只有一个 task identity；canonical CLI
还要求 output root 的目录名与该 identity 相同（含显式 `--output-root`），不匹配时在发布前失败。
使用 `--task-name` 改名时，需同时指定匹配的新 output root；direct library 不限制输出目录名。
每个 accepted episode 保留全部 source rows，输出始终满足 `source_frames == episode_steps`；
原子发布前执行完整 schema/payload/semantics validation，不再提供 `--verify-output` 开关。
joint_state/contact_force/tactile 都直接来自同一 raw row；fingertip 由本行 joint_state
经共享 FK 计算。action 使用 `action_arm_joint_sent`，不能替换成未发布 candidate。
视觉时间与 tactile 时间均只需不晚于 control anchor，不要求彼此同步或 tactile 早于 camera。

raw v28 episode 可视化默认使用当前 resolved runtime 中的点云策略和桌面标定即时生成 canonical
`(N,6)` 点云；该路径与 offline processing、实时 deployment 共用同一个 `build_point_cloud()`
实现。使用 `--no-point-cloud` 可关闭即时点云，
`--pointcloud-num-points` 可选择 `1024`、`2048`、`4096` 或 `8192`。

查看持久化点云可运行 `python examples/visualize_episode_processed.py <processed.h5>`。
该离线 viewer 读取 processed v18 的 RGB-D/点云和 fingertip，使用逻辑 `arange(T)*dt`
显示时间；不加载 row provenance 或存储的 EEF pose。它按所用模态检查结构和渲染 payload；
完整 artifact 验证由 processing/export 边界承担。

物理回放始终以当前 geometry 和 runtime 完整验证 live start、joint limits、recorded
start→first target、workspace、collision 及全部相邻 transition。

处理入口对整个 batch 分析一次，只打印 whole-episode 汇总，不写 invalid-frame range JSON。
连续 1–4 个 IK_FAIL 保留；连续超过 4 个拒绝整条 episode。camera fresh=False、
observation_valid=False 单独出现、短 IK/tracking transient、dense 不可用但 aggregate 可用
均不删行；无法读取必需 payload、非有限必需值或严重 identity 损坏可使整条失败。

`--annotations` 仅支持 episode 的 `include` 与可选 `task_name`。include 默认为 true；
显式 include 的失败阻断整批，include:false 跳过。不支持 include_ranges/exclude_ranges。
CLI 跳过未标注 rejected episode；direct library 默认阻断，调用方可显式设置
`skip_rejected_unannotated=True`。CLI 只保留 input_root、output-root、
pointcloud-num-points、task-name、annotations、dry-run。

导出前可只读预检 processed schema/payload、task/dt/shape/dtype 和 modality
semantics 一致性；preflight 不创建 Zarr，也不扫描 raw 来重建 row-selection proof：

```bash
python examples/export_policy_zarr.py \
  episodes_processed/<task> \
  --dry-run
```

预检通过后导出到 `datasets/<task>.zarr`：

```bash
python examples/export_policy_zarr.py \
  episodes_processed/<task>
```

输入目录名决定导出任务名，且会与每个 processed HDF5 中的 `task_name` 校验；已有
`datasets/<task>.zarr` 文件、目录或符号链接（包括悬空链接）都会拒绝覆盖。
导出时在 stderr 显示输入校验、Zarr 写入的 tqdm 进度条；不会向 stdout
打印 JSON 报告。每个经过 processed validator 的文件完整追加为一个 Zarr episode，
`episode_ends` 是各文件行数的累积和。无效 processed input 使本次 export 整体失败；
没有 gap heuristic、按行 rejection report 或 partial export。

相邻 `dexmani_policy` 消费 control-step metadata（`schema_version` 是 informational，不作
exact-version gate），并拒绝 camera-master 合同；不提供 compatibility aliases。实际 salvage
Zarr 已通过消费端合同、ReplayBuffer 和离线采样验证；未执行真实 checkpoint export 或硬件 rollout。

可视化 raw episode：

```bash
python examples/visualize_episode.py episodes/<task>/episode_*
python examples/visualize_episode.py episodes/<task>/episode_* --no-point-cloud
```

可视化处理后的 processed HDF5（rgb/depth 与预计算点云已在文件内，离线、不连硬件）：

```bash
python examples/visualize_episode_processed.py episodes_processed/<task>/episode_<timestamp>.h5
```

## 数据布局

```text
episodes/<task>/episode_<timestamp>/
├── data.h5       # control-step robot/action/VR/contact and factual telemetry
├── depth.h5      # same-row aligned uint16 depth
└── rgb.mp4       # same-row RGB

episodes_processed/<task>/
└── episode_<timestamp>.h5

datasets/<task>.zarr/
├── data/*
└── meta/episode_ends
```

当前合同是 raw v28 → processed v18 → Policy Zarr v11。processed 保存 dense tactile（含
validity）、aggregate contact（含 validity）与 flat timing；不保存 eef_pose、row provenance、
quality JSON 或 segments。
录制时从 live ring 选择有效 aggregate，并保存其独立 source；这不是从其他 raw row
修补 dataset。历史 raw 不改写；详细字段和边界见 [data schema](docs/data_schema.md)。

实际 `pick_place_toy` salvage：61 条 / 14,309 行源数据；60 条 / 14,112 行保留；
只拒绝 `episode_20260827_224527` 的 persistent IK（197 行）。60 个 Zarr episodes 与
accepted processed 文件逐行一致，原始 183 个文件 SHA-256 未变。新产物分别位于
`episodes_processed/salvage_v18/pick_place_toy` 与
`datasets/salvage_v11/pick_place_toy.zarr`，不覆盖旧产物。
[小型 manifest](artifacts/pick_place_toy_salvage_manifest.json) 保留计数、lineage 和验证记录；
大型 HDF5/Zarr 不进入 git。

## 开发与验证

安全的最低成本检查：

```bash
python -m compileall -q dexmani_real examples tools
python -m pytest -q
git diff --check
git diff --stat
git status --short
```

`tests/` 提供纯函数、配置快照、IPC ABI、ring buffer、schema、生命周期、安全失败路径和
架构边界的离线回归门禁。example 程序不是测试，测试和静态检查通过也不等于完成真实硬件验证。仓库级
agent 工作约定见 [`AGENTS.md`](AGENTS.md)，具体编码规范见
[`code_style.md`](code_style.md)。
