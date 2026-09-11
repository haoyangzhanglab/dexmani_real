# DexMani Real Repository Map

本文件只记录稳定的运行拓扑、数据流和安全边界。运行行为以源码、schema 与配置为准；
本文件不维护逐文件职责清单。新增、删除或移动文件时，仅在改变下列边界时更新本文件。

## 仓库导航

| 文件 | 作用 |
|---|---|
| `AGENTS.md` | 仓库级工程、安全、范围与验收契约。 |
| `CLAUDE.md` | Claude Code 精简入口，具体规则委托给 `AGENTS.md`。 |
| `code_style.md` | 本研究代码库的具体编码与审查约定。 |
| `README.md` | 面向使用者的能力、环境、工作流与稳定架构。 |
| `repo_map.md` | 当前运行拓扑、核心数据流与边界索引。 |
| `tools/convert_raw_v24_to_v25.py` | 冻结的一次性历史 raw v24 → v25 转换器；不依赖当前 runtime。 |
| `tools/convert_raw_v25_to_v26_tactile.py` | 冻结的一次性 raw v25 → v26 tactile 表示迁移（`* 10` 还原 SDK 原生刻度、新增 `tactile_sum_fresh`）。 |
| `docs/control_step_dataset_simplification_plan.md` | 历史 control-step dataset 简化执行计划（SUPERSEDED）；当前实现为 v19/v12 multimodal dataset。 |
| `docs/invalid_frames_export_incident.md` | 历史 forensic evidence 与实际 salvage 验证记录。 |
| `docs/control_step_dataset_human_review_record.md` | Real/Policy control-step 修复交接：固定提交范围、证据边界、剩余限制与待人工签署的 review checklist。 |
| `artifacts/pick_place_toy_salvage_manifest.json` | 小型 source identity/lineage/whole-episode salvage manifest；无生成数据或机器绝对路径。 |
| `docs/raw_v24_migration.md` | 历史数据迁移与 processed/Zarr golden 回归步骤。 |
| `docs/refactor_contract_audit.md` | 验证器分类、producer/consumer 证据与 KEEP 决策。 |
| `docs/refactor_execution_plan.md` | 已执行重构的 canonical 方案、protected invariants 与分阶段验收规则；执行结果见 evidence。 |
| `docs/refactor_execution_evidence.md` | 本轮分阶段重构的 baseline、审查、离线验证和交付记录。 |
| `docs/tactile_unit_si_verification_pending.md` | 触觉力单位语义 fact-check 与决策记录：0.1 scale 考古、"单位=N"证据盘点、删除代价矩阵与 known-load 解锁协议；规则 owner 仍是 xhand 指南。 |
| `.codex/config.toml` | 项目级 Codex 权限、联网与子智能体并发配置。 |
| `.codex/agents/*.toml` | 项目级难度分档子智能体：`sol-high`、`terra-max`、`luna-max`。 |

## Process topology

```text
hardware SDKs
    │
    ▼
arm / hand / VR / camera / point-cloud / recorder workers
    │ typed rings, queues, flags and lifecycle state
    ▼
RuntimeChannels
    ├── teleop session + control loop
    ├── policy inference worker → prediction ring → policy executor
    └── replay / calibration control owner
    │
    ▼
control safety gate → command publication → arm / hand workers
```

- 每个硬件 SDK 只在其 owning worker 或 driver 内创建和使用；父进程只负责生命周期与监督。
- planning 配置由调用方直接构造 dataclass；不提供通用字典反序列化入口。
- `config/experiment.py` 拥有 YAML/CLI patch 和统一 runtime validation；每次加载得到独立配置，打印时才生成 YAML。
- `RuntimeChannels` 是跨进程状态的唯一 allocation owner；固定 wire shape、dtype 和持久化
  record layout 由 `dexmani_real/ipc/schema.py` 定义。
- `teleop/control_loop/grid.py` 拥有 feedback 连续错误计数和启动时解析的私有 command-limit
  cache；`action_proposal.py` 拥有 EMA pose smoothing。独立 health/smoothing 包装模块与无消费者的 StageTimer 模块已移除。
- `teleop/homing.py::do_configured_teleop_home` 直接读取 immutable runtime 并编排 hand-first
  homing；`tests/test_teleop_homing.py` 离线覆盖接受顺序、配置、freshness、取消和 completion audio。
- `runtime/operator_input.py::KeyboardInput` 统一拥有 listener、held keys 和原始事件；
  command workflow 使用 edge-triggered `poll()`，keyboard teleop/camera calibration
  关闭 command capture 并读取 held/event 状态，保留各自 repeat/release 语义。
- learned-policy lifecycle 先等待 inference restore/warmup，再启动所需传感器和执行器 worker；
  supervisor 只监督实际运行的 process heartbeat，readiness 只负责有界启动等待。
- teleop lifecycle 按 dependency → policy → VR 分阶段启动启用的 worker；父进程 readiness
  屏障同时检查 ready、sticky fault、所有已启动进程的 liveness 与 timeout。replay 与
  calibration 复用相同的 safety、publication 和 worker 边界。

## Policy flow

```text
causal observation history
    → Policy public runtime
    → Real NumPy adapter
    → flat Prediction IPC record [chunk_size, D] + provenance + inference timing
    → one-slot latest-wins prediction_ring
    → PolicyExecutor (timestamped future targets, endpoint decode, EE→IK)
    → physical SafetyGate
    → non-blocking coupled publication
    → arm / hand worker final checks
```

- `deployment/inference/observation.py` 的 builder 拥有 causal/freshness/history/skew/generation
  准入。camera、arm、hand、contact 均按 logical policy control-grid references 对齐，
  不以 camera exposure time 对齐机器人状态。SHM copy/seqlock、run-start、age/grid-lag、
  health/state/calibration/unit 与必要的 contact provenance source-match gates 保留。
- `Prediction` 是简单的内部策略输出容器；动作在 inference 输出边界校验为 finite
  `float64[chunk_size, D]`，序列化和 SHM 读取建立传输副本，
  `run_generation`、source timestamp 和 logical-step timestamp 随对象传播。
  三个 timing（inference latency、observation age/skew，单位 ms；无效诊断值忽略）
  随 exact chunk 经 IPC 进入 executor-owned `PolicyStats`（live 日志），
  表示当前 generation 最新收到的样本（含全过期 chunk），不是 full-episode percentile。
- Real 只校验 Policy 公开契约字段（observation fields/shape/dtype/相应 semantics、`requires_hand`、
  `chunk_size`、`n_action_steps`、`action_key`、`control_action_dim`、`control_dt_s`），不解析
  artifact 内部或 `temporal_ensemble_coeff` 等历史字段；Real 直接调度 Policy 给出的完整
  future action chunk，不在 deployment/sim runner 追加 overlap blending。
- `PolicySpec.observation_fields[*].semantics` 是 Real public compatibility contract 的一部分；
  EEF、tactile、fingertip 的 interpretation 和 point-cloud preprocessing identity 在
  `deployment/config.py` 校验，不能仅依据固定 shape/dtype 删除。逐类审计见
  [`docs/refactor_contract_audit.md`](docs/refactor_contract_audit.md)。
- inference worker 只负责 observation、策略调用和 prediction 发布；PolicyExecutor 独占动作
  horizon 解码、control-grid 调度、EE→IK、候选校验和 command-progress watchdog。
- `robot/model.py` 统一 anatomical finger、SDK tactile sensor ID 和 fingertip link 顺序；
  `tests/test_hand_finger_order.py` 固定其映射。部署 EEF/fingertips 从 policy-visible float32
  joint_state 派生；tactile full snapshot 同时提供 provenance，contact-only 使用
  `ipc/ring.py::get_last_k_fields()` 投影 metadata。`tests/test_ring_projection.py` 使用真实
  SHM 验证投影、副本所有权与 seqlock rejection，不启动硬件。
- 正常策略 tick 的路径是 `validate → publish → continue`；`execute=False` 完成同样的候选
  校验但不产生 actuator side effect。需要确认 SDK 接受的 home、calibration 和 replay 操作，
  才显式调用 blocking acceptance。
- learned-policy arm 动作是 reject-only：`wrap_nearest_equivalent` 只做关节表示 canonicalization，
  超 joint limit 或 per-joint jump 阈值直接拒绝，绝不 clip。observation freshness 在 inference
  边界检查，action 有效性由 logical target timestamp 的 stale 过滤决定，command 有效性由
  `action_validity_s` + worker guards 决定。
- B 只在 ARMED、physical home（物理运行）完成后进入 RUNNING；每个 episode 都必须重新 H 才能 B。
  唯一周期执行路径不追赶过期 deadline，也不制造超过 `control_hz` 的 command burst。
- `tests/test_policy_rollout.py` 用离线 fake 覆盖 Policy 公开契约兼容、timestamp 调度
  （stale-prefix/whole-stale/no-catch-up）、IK/SAFETY 归属、reject-only arm 与
  publish-then-record rollout 合同，不启动 worker 或硬件。
  `tests/test_prediction_boundary.py` 覆盖 inference 动作 shape/finite 准入，以及 warmup 失败与无效诊断的区别。
  `tests/test_observation_builder.py` 覆盖 sensor/camera 因果、freshness、generation、payload 与模型转换边界。
  `tests/test_control_safety.py` 覆盖 SafetyGate 目标准入、workspace/collision fail-closed，以及 worker mechanical/finite/jump、ticket/expiry 和 hand-home 边界。

## Recorded policy rollout flow

```text
B + completed H
    → RecorderIO START acknowledged as RECORDING
    → RUNNING generation
    → existing PolicyExecutor schedule / coupled publication
    → ordinary raw sample after publish/reject, or held control tick
    → operator S / timeout / watchdog / quit / estop / fault
    → motion fence
    → RecorderIO STOP → asynchronous finalize → saved=True → completed_episodes++
    → rollouts/<policy>/<task>/<experiment>/session_<ts>/episode_NNN/
    → (N/N) quit_requested → clean shutdown
```

- `deployment/config.py` 携带会话输出目录、task/operator、wall-clock timeout 与 episode 数的 narrow
  `RolloutRecordingConfig`（`deployment/evaluation.py` 已删除）；不写 result.json，也不判断 task success。
- executor 的时限与 watchdog 映射为 technical stop reason（`timeout` / `first_command_timeout` /
  `command_silence_timeout`）；`S`→`operator`、`Q`→`quit`、estop/hardware 故障为对应硬件原因，
  recorder/storage 失败为 `recording_failure` / `recorder_fault`，均 fail closed。
- 强制启动 camera 与 RecorderIO，即使 policy 是 state-only；camera 仍只在 `PolicySpec`
  请求 RGB/pointcloud 时进入 inference observation。`RuntimeChannels.evaluation_outcome` 已删除。
- START/RECORDING ACK 是唯一启动录制屏障。首条普通控制网格 sample 开始 raw evidence；没有 initial
  sample gate 或第二套 evidence 事务。recording failure 立即 fail closed 并撤销 generation，
  不等待 command acceptance。
- 停止调用 `stop_episode(save=...)`；`saved=True` 且无 error 是唯一计数证据（recorder client 的
  capacity 自停同样被计入，因为 RecorderIO 不会把 discard 升级为 save）。一个 session 跑 N 个
  episode，每个 episode 都需要 fresh H；第 N 个 publish 后自动 quit_requested，supervisor 走
  clean shutdown。`_CommandProgress` 继续承担 worker/SDK progress watchdog。
- `tests/test_policy_rollout.py` 用 fake/shared-memory boundary 覆盖 Policy 公开契约、timestamp
  调度、IK/SAFETY 归属、reject-only arm、多 episode 计数/超时/quit/estop 的 technical reason 与
  recording fail-closed；不启动任何 worker 或设备。
- `tests/test_recorder_io_boundary.py` 覆盖 RecorderIO capacity/stop reason 边界、显式 episode
  命名与真实 raw-v28 事务（临时目录写入 data.h5/depth.h5/rgb.mp4）；不连接设备。

## Teleop flow

```text
VR / keyboard input
    → teleop session lifecycle
    → control loop / causal fixed-grid tick
    → TeleopController + action proposal
    → SafetyGate + publication
    → arm / hand workers
    └→ fixed-grid record sample → RecorderIO
```

- `teleop/control_loop/grid.py` 持有 proposal、EMA、hand ramp/retarget、IK hold 与上一 endpoint 等
  算法状态；loop 持有 pause、recording、keyboard 和退出编排状态。
- feedback 缺失、过期、断开或录制边界进入 pause boundary：先使 generation 失效并清除
  controller reference，恢复时必须收到 pause 之后的新鲜因果 arm/VR/hand feedback，再重锚并
  继续发布。静默期间不发布旧目标。
- teleop 和 replay 的动作决策都必须经过共享 publication/safety 边界；worker 侧再次检查
  state、generation、ticket、shape、finite、limits 和 freshness。

## Command safety invariants

- runtime safety state 只有 `DISARMED → ARMED → RUNNING` 与 `FAULT` 终态路径；共享的
  `motion_lock` 同时保护 state、`run_generation` 与 coupled-command ring 的串行发布。
- 开始、停止、暂停和故障通过推进 `run_generation` 使旧命令失效；当前 ticket 由
  current generation 与 `coupled_cmd_ring.latest_sequence` 直接判定。stale、过期、被覆盖或不再拥有
  latest-wins slot 的命令不得跨 SDK 边界。
- `SafetyGate` 校验 joint limits、workspace、collision 和 command delta；它不把 feedback
  缺失当作安全，也不以隐式 clip 代替 reject。arm/hand worker 保留最后一道硬件调用前守卫。
- STOP、e-stop、worker death、heartbeat timeout 和 command-progress stall 都 fail closed；在所有
  child 已确认停止后发生的 IPC/resource cleanup error 仍记录为失败，但不重新赋予 physical `FAULT`。
  home 仍执行 hand + arm 的显式碰撞检查路径。
- 命令发布与物理接受是两个边界：实时路径非阻塞发布，只有确实需要确认的 home、calibration
  和 replay 路径等待明确 acceptance result。

## Recording/Data boundaries

- `recording/client.py` 定义四个低频 Queue 消息并独占结果消费；`recording/io_worker.py`
  按 STOP 的 `through_sequence` 排空 sample SHM 后关闭并原子发布。没有 recorder wire
  phase/generation；局部 pending finalization 不跨 IPC。`tests/test_recorder_queue_client.py`、
  `tests/test_recorder_queue_io.py`、`tests/test_recorder_queue_channels.py` 覆盖生产停止边界、
  FIFO/error/next episode 以及真实 spawn Queue；worker 非零退出由现有 session lifecycle 汇总。
- RecorderIO 是 episode lifecycle 的唯一 owner；`EpisodeRecorder.finish_episode()` 同步
  序列化/关闭/验证/发布。RecorderIO 的 finalizer 使用进程内 Queue 返回结果，主线程保持
  heartbeat/control polling，确认线程退出并 reap 后发布唯一完成消息，再允许 next START。
  `RecorderClient` 的 poll/join API 保留。安全回收的 episode-local failure 可恢复；
  timeout、transport failure、未回收 writer/文件资源保持 fatal，交由既有 verified shutdown 回收。
- Camera IPC 只携带 source/receive/publish 时间、generation、帧号和 health；保留 payload
  尺寸字段以支持 name-only ring attach。设备时钟映射留在 driver/worker，不作为录制审计传输。
  static shared metadata 只保留 serial、RGB-D geometry 和 depth scale；校准与研究 provenance
  继续写入 raw v28。`tests/test_camera_v25_telemetry.py` 覆盖 health、causality 与 seqlock 边界。
- teleop 只把已经选择并校验的 causal fixed-grid sample 交给 RecorderIO；RecorderIO 独占
  episode transaction、sidecar、sequence continuity、validation 和 atomic finalize，不决定
  机器人动作。
- recording startup 在创建 channel/worker 前使用已解析的运行时配置；录制数据只保留运行所需的
  source/publish provenance，不新增 episode sidecar 或进入 realtime loop。policy 会话的
  selector、pinned checkpoint name、inference steps、seed 和 wall-clock budget 通过
  recorder-owned `provenance_*` metadata attrs 保存（不含 git commit / SHA-256），不与 camera
  metadata 混用。
- raw episode 的 schema 与语义由 `recording/storage/schema.py` 与
  [data schema](docs/data_schema.md) 定义：raw v28 → processed v19 → Policy Zarr v12。
  每个 accepted episode 完整保留 raw 行，一个 processed 文件对应一个 Zarr episode。
- `ipc/causal.py::read_hand_contact_causal` 只为 recording 选择最新有效 causal aggregate；
  `recording/sample.py` / `recording/frame.py` 显式携带独立
  `hand_contact_source_monotonic_ns`，不修改 command hand feedback，也不借用 dense source。
  无有效 aggregate 保存 NaN/source0；age 仅是 recording telemetry。dense raw 数据保留。
- `EpisodeReader` 只接受 raw v28 并检查文件/layout/sidecar 帧数，不完整解码 RGB。
  `dataset/processing.py` 有一个窄的只读 normalized-v26 历史入口；冻结迁移独立存在，
  没有 v26/v27→v28 伪迁移或 runtime compatibility reader。
- `dataset/processing.py::process_episode_root` 独占 whole-episode admission 与 batch
  publication。persistent IK（连续 >4 行）是唯一自动行为拒绝；技术损坏失败，
  不做 timing-quality filtering、跨 raw row tactile repair 或 row compaction。
- `dataset/processed.py` 独占完整 schema/payload validation，证明
  `source_frames == episode_steps`。processed 是完整 multimodal superset：joint/action/action_ee、
  aggregate contact（含 `contact_force_valid`）、dense tactile（含 `tactile_force_valid`）、
  fingertip、eef_pose（与 fingertip 复用同一次 canonical arm FK）、native-resolution RGB-D、
  camera geometry、point cloud 与 flat timing arrays；
  无效 contact/dense 用 validity mask 表达，不整条拒绝。不再有 modality-specific profile。
- `dataset/export.py` 验证 processed 和跨文件一致性，完整追加并写 episode_ends 后
  transactional publish；无 gap、keep-mask、row-provenance 或 quality framework。
  `clean.py` 与 `quality.py` 已删除。CLI 只调用一次 batch，annotation 仅 include/task_name，
  无 compare/verify-output 开关或 invalid-frame JSON。
- `tests/test_control_step_dataset.py`、`test_control_step_export.py`、
  `test_control_step_observation.py`、`test_control_step_replay.py` 与
  `test_recording_control_contact.py` 覆盖当前数据/部署/录制边界；
  旧 selector、camera-aligned recording、processed-v15、gap projection 测试已移除。
- `tests/test_raw_v26_recording.py` 覆盖 direct-row batch、真实时间缺口、sidecar identity 与失败不发布；`recording/timeline.py` 的第二时间网格已删除。
- `tests/test_v26_producers.py` 用实际 sample/client 覆盖 sent target、head pose、active/IK hold 与 retarget-failure queued 语义。
- `recording/storage/hdf5_writer.py` 独占单个 `data.h5` handle；camera sidecar 和 video writer 不
  反向拥有控制状态。缺口、失败或未完成 finalize 不伪装成完整 episode。
- 物理回放读取 recorded published arm target 与 recorded logical hand target/provenance，并重新经过
  当前 runtime 的 preflight、safety、generation 与 worker 边界；当前 hand worker 由 logical
  hand target 生成受限 SDK intermediate setpoint，而不是回放 exact actuator setpoint。processed
  产物不能重新解释或替代 raw 命令事实；processed replay 只按 `source_path` 读取完整 raw float64 命令数据；当前 reader 不支持历史 v25/v26 physical replay。已有非空 replay
  output 在创建 channel 或 worker 前拒绝，避免覆盖实验结果。

源代码、schema 和 canonical config 是实现真相；本文件只帮助定位上述稳定边界。
