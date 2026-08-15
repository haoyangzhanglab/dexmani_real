# Phase C 实现与优化梳理（Implementation Map）

**来源文档：** `docs/DexMani Real 完整 A - B - C 整改与策略部署执行文档.md`
**当前基线：** A/B Runtime Freeze（`docs/ab_runtime_freeze_report.md`，offline 12/12）
**性质：** 只做梳理，不实施。每个 subphase 仍需按执行文档 §111 的报告格式单独 review + commit。

---

## 1. Phase C 意图

Phase C 不是「把 `dexmani_policy` 代码复制进 `dexmani_real`」，而是建立一个**通用
learned-policy 部署运行时**：能接入不同模型仓库（DexMani Policy / π0 / ACT /
Diffusion Policy / RDT / 其它），同时**完全复用 A/B 之后的机器人 runtime、安全、IPC、
生命周期机制**。

最终要交付的是执行文档 §119 的第五个稳定边界 —— **Learned-Policy Boundary**：

```text
Model 可替换            Control Source 可替换
Robot Runtime 不随模型变化      Safety Runtime 不随模型变化
Device Worker 不随模型变化      Recording Core 不随模型变化
```

核心架构（§47/§115）：

```text
Sensor Workers → SharedStorage → Inference Worker (ObservationAdapter → PolicyBackend → ActionAdapter)
        → policy_plan_ring (latest-wins)
        → Deployment Coordinator (scheduler → ActionCandidate → SafetyGate → transport)
        → arm queue / hand ring
```

最高原则（§48）一句话：**模型输出只是 proposal，不是 robot command**。Inference
Worker 禁止直接写 `arm_action_q` / `hand_cmd_ring` / SDK / `SafetyState` /
`run_generation`。Coordinator 是 learned-policy 唯一的 robot-action producer。

---

## 2. 必须冻结的地基（FROZEN RUNTIME CONTRACT）

`docs/ab_runtime_freeze_report.md` 已把 A/B 后的契约冻结。Phase C 的所有新代码必须
**搭在其上而不是改造它**：

| 冻结契约 | 位置 | Phase C 如何依赖 |
|---|---|---|
| `ActionCandidate`（joint_position / rad / robot_joint） | `policy/runtime.py:32` | C8 每个 due endpoint 都要转成它 |
| `SafetyGate.validate`（well-formed → joint limits → workspace） | `policy/safety.py:107` | C8 唯一安全边界 |
| `send_command`（fire-and-forget：arm queue + hand latest-wins） | `policy/safety.py:237` | C8 唯一发布路径 |
| `advance_run_generation`（cancellation token） | `policy/safety.py:32` | C7/C8 generation 取消 in-flight inference |
| `SharedMemoryRingBuffer`（seqlock + latest-wins + `get_last_k`） | `shm/ring_buffer.py:123` | C5.1 `policy_plan_ring` 直接复用 |
| `WorkerSpec` / `run_supervisor` / `wait_subsystem_ready` | `runtime/processes.py:26` / `runtime/supervisor.py:43` | C10 生命周期复用 |
| HDF5 schema v16 | `recording/` | C 初期**不改**（§95/§97） |
| XHand connect/disconnect（INIT + 2s watchdog） | `robot/xhand.py` | **禁止被 policy deployment 改动**（§109） |

---

## 3. 现状盘点（总表）

当前仓库处于 A/B Freeze 之后：`checks/offline/`（12/12）、`robot/homing.py`（B6）已存在；
**`deployment/` 和 `integrations/` 目录都不存在**，所有 Phase C 符号均未出现。

| Phase C 项 | 现状 | 性质 |
|---|---|---|
| C0/C3 部署契约（Protocols） | 无 | 🆕 实现 |
| C1 因果读取器 | 逻辑已在 `teleop/snapshot.py`（私有） | ♻️ 抽取 |
| C2 候选发布边界 | `send_command` + `publish_joint_targets` + `teleop/loop.py` 私有 `_safe_joint_publish` | ♻️ 合并/公开 |
| C4 loader / config | 无，但有 `config/runtime.py` 模式可镜像 | 🆕 实现（复用模式） |
| C5 `POLICY_PLAN_DTYPE` | 无 | 🆕 实现 |
| C5.1 `policy_plan_ring` + inference 心跳/就绪 | 无（`HEARTBEAT_FIELDS`/`READY_FIELDS` 缺 `inference`） | 🆕 实现 |
| C6 FakeBackend | 无 | 🆕 实现 |
| C7 inference worker | 无 | 🆕 实现 |
| C8 coordinator / scheduler | 无 | 🆕 实现 |
| C9 failure semantics | 无 | 🆕 实现 |
| C10 lifecycle + CLI | 无，但 `runtime/*` 可复用 | 🆕 实现（复用） |
| C11 DexMani Policy adapter | 无 | 🆕 实现 |
| C12/C13 backend swap + metrics | 无 | 🆕 实现 |
| C offline tests（§98） | 无 | 🆕 实现 |
| H0–H6 硬件门 | — | ⏸ 仅文档化，离线不可做 |

---

## 4. 需要「优化/抽取」的现有代码（♻️）

这些是 Phase C 里**非全新建、而是把已经存在的东西抽出来/公开化**的点，也是「优化」的落点。

### 4.1 C1 — 因果读取器抽取 `shm/causal_reader.py`

`teleop/snapshot.py:70-220` 已经有完整的因果读取原语，但全是私有 `_read_*`：

- `_read_causal_structured_frame`（`:70`）——核心不变量 `0 < source_ns <= publish_ns <= anchor`
- `_read_arm_state` / `_read_hand_state` / `_read_vr_frame` / `_read_hand_tactile` / `_read_camera_frame`

**优化动作：** 抽取到 `shm/causal_reader.py`，`teleop/snapshot.py` 变薄消费者，
`deployment/observation.py` 成为第二个消费者。

**必须保持（§51）：** 抽取前后「同一合成输入 → 同一选中帧」等价，不得改动 age
threshold / VR threshold / camera threshold / source precedence。注意 camera 链路还多一层
`receive_ns`（`source <= receive <= publish <= anchor`，见 `:176`）。

### 4.2 C2 — 合并/公开候选发布边界

发布逻辑今天分散在三处：

- `policy/safety.py:237` `send_command(shared, candidate, ...)` —— 已是「预构建候选 → transport」
- `policy/safety.py:479` `publish_joint_targets` —— 从**原始关节数组**构建候选再发（keyboard/replay/calibrate 用）
- `teleop/loop.py:1545/1571` `_safe_arm_queue_put` / `_safe_joint_publish` —— 私有，且被拆成两个函数

**优化动作：** 文档 §51 明确要一个 `validate_and_send_candidate(shared, candidate, *,
gate, prepare_timeout_s) -> ActionCandidate | None`，接收**已构建好的 `ActionCandidate`**
（coordinator 会从 due endpoint 自己构建候选）。把 teleop 里的私有发布逻辑收敛成这一个
公开函数，coordinator 复用而不是第三套重复实现。

### 4.3 C5.1 — `policy_plan_ring` 直接复用现有 ring

`SharedMemoryRingBuffer` 已实现 latest-wins + seqlock + `get_last_k`
（`ring_buffer.py:123`）。**不需要新 ring 类**，只需：

- 在 `shared_storage.py:170` `_RING_RESOURCE_NAMES` 加 `"policy_plan_ring"`
- 在 `SharedStorageConfig`（`:39`）加 `policy_plan_ring_maxlen`（2–4 slots）
- 在 `_allocate_resources`（`:324`）分配，`close()` 由现有 `_RING_RESOURCE_NAMES` 循环自动覆盖

### 4.4 C10 — 生命周期复用 `runtime/*`

`WorkerSpec` / `build_processes` / `start_processes` / `shutdown_processes_verified`
（`processes.py`）+ `run_supervisor` / `wait_subsystem_ready`（`supervisor.py`）就是 A/B
冻结的生命周期。`deployment/lifecycle.py` 只需**组合** arm/hand/camera/inference/coordinator
/recorder 的 `WorkerSpec`，**禁止**建第二个 process-health 系统（§64/§81）。

### 4.5 C4 — 配置解析模式镜像 `config/runtime.py`

`resolve_runtime_config`（`config/runtime.py:216`）的 `CLI > file > defaults` + canonical
SHA-256 是现成范式。部署配置要**独立**到 `deployment/config.py`，绝不能把模型内部参数
（transformer depth / diffusion schedule 等，§93）塞进冻结的 runtime config。

---

## 5. 需要「新建实现」的模块（🆕，按 C0→C13）

### 5.1 C0/C3 — `deployment/` 包骨架与协议

```text
deployment/
  contracts.py     PolicyBackend / ObservationAdapter / ActionAdapter Protocols（§52）
  config.py        DeploymentConfig 解析（§92 字段）
  loader.py        lazy backend loader（module:symbol 拆分 → import → instantiate → validate protocol，§58）
  observation.py   ObservationBatch + ObservationAdapter 消费端
  worker.py        inference_loop
  coordinator.py   scheduler + ActionCandidate + SafetyGate + publication
  lifecycle.py     resolve → SharedStorage → spawn → readiness → ARMED → supervise → shutdown
  metrics.py       普通 counters + structured logging
```

两个 process-local frozen dataclass（**不进 SharedStorage，无 IPC dtype**）：

- `ObservationBatch`（§54）：observation_id / run_generation / anchor + arm/hand/tactile/camera
  history + source/publish/valid_mask
- `JointActionChunk`（§56）：`arm_qpos[N,7]` / `hand_qpos[N,12]|None` /
  `target_monotonic_ns[N]` / `valid_mask[N]`

### 5.2 C5 — `utils/schema.py` 增加

- `MAX_POLICY_CHUNK_STEPS`（32 或 64，runtime transport capacity，≠ 某模型 horizon，§61）
- `POLICY_PLAN_DTYPE`（§60）：plan_id / run_generation / observation_id /
  observation_anchor / inference_started / inference_finished / num_steps / arm_present /
  hand_present / `target_monotonic_ns[MAX]` / `arm_qpos[MAX,7]` / `hand_qpos[MAX,12]` /
  `valid_mask[MAX]`

### 5.3 C5.1/C64 — `shm/shared_storage.py` 增量

- 新增 `policy_plan_ring`（latest-wins，2–4 slots）
- 在 `HEARTBEAT_FIELDS`（`:186`）与 `READY_FIELDS`（`:199`）各加 `"inference"`
  （注意 `collect_teleop.py` 目前复用了 `policy` 槽，C10 要分开）

### 5.4 C6 — FakeBackend（架构门）

`FakeObservationAdapter` / `FakePolicyBackend` / `FakeActionAdapter`，**CPU only、
deterministic、无 torch、无硬件**（§65）。它是 §66 的架构门：不 import `dexmani_policy`
也能端到端跑通 observation → backend → chunk → plan ring，否则说明 core 抽象失败。

### 5.5 C7 — Inference Worker

`inference_loop(shared, config)`（**普通函数，不是 `mp.Process` 子类**，§67）。启动顺序
（§68）：heartbeat → lazy import → 实例化 adapter/backend → `backend.load()` → 校验
horizon/output contract → `ready(inference)=true`（load 失败 = 进程失败，**不进入 dummy
safe mode**）。主循环（§69）末尾 **re-read generation，变了就 DROP，禁止贴新标签**（§70）。

### 5.6 C8 — Coordinator + scheduler（Phase C 核心）

Learned-policy 唯一 robot-action producer（§72）。必须实现：

- 本地 `active_plan` + `consumed step indices`（§75）
- 计划采纳校验：generation 当前？observation 更新？plan fresh？shape valid？（§75）
- **scheduler coalesce**：模型快于控制时，多个 overdue step 只发 latest due，不连发 3 条
  （§76）；模型慢时**不发命令、不插值、不重复 last command**（§77）
- endpoint → `ActionCandidate`（§79，`is_hold=False`）→ `validate_and_send_candidate` →
  SafetyGate → transport
- command silence watchdog（§82）：RUNNING 但超 `max_command_silence_s` 无有效 endpoint →
  advance generation → RUNNING→ARMED
- RUNNING ↔ ARMED control-source state

### 5.7 C9 — 失败语义分级（§80）

- **Drop-only**：generation mismatch / old observation / superseded / expired —— 计 metrics 继续
- **Abort policy run**：NaN / Inf / shape 错 / timestamp 乱序 / unsupported repr / SafetyGate
  reject / 反复 no-valid-action —— advance generation、RUNNING→ARMED、要求显式重启
  （**不必 FAULT**，硬件未必坏）
- **进程失败**：crash / CUDA fatal / backend exception / heartbeat timeout → 交给**现有
  supervisor**（§81）

### 5.8 C10 — `deployment/lifecycle.py` + `examples/run_policy.py`

lifecycle 组合（§83）；CLI 只做 argparse → config override → resolve → run → exit code
（§84，禁止塞模型/调度/安全/SharedStorage 业务逻辑）。**默认不启动 VR worker**，只有
adapter 声明需要 VR 才启用（§85）。

### 5.9 C11 — `integrations/dexmani_policy.py`

FakeBackend 全通过后才做（§86）。

- DexMani Policy Backend（§87）：load Hydra config → Agent → checkpoint → normalizer →
  `predict_action` → model-native output
- ObservationAdapter（§88）：`ObservationBatch → policy 期望的 dict/tensors`（RGB /
  pointcloud / joint 模态差异全部留在 adapter）
- ActionAdapter（§89）：denormalize → `arm[N,7]` / `hand[N,12]` → `JointActionChunk`
- **首版只允许 native joint action**；EE-action checkpoint 且无已验证 EE→joint 转换 →
  startup reject（§90）

### 5.10 C12/C13 — backend swap 验证 + metrics/provenance

- swap 只允许改 config / adapter / checkpoint / env，**禁止改** robot/sensor/SafetyGate/
  SharedStorage/coordinator（§100）
- metrics：observations_built / observation_age / observation_skew / inference_ms /
  plans_created / plans_superseded / plans_stale / plans_generation_dropped / plan_age /
  endpoints_due / endpoints_coalesced / endpoints_published / safety_rejections /
  policy_aborts / command_silence_abort（§94），先用 counters + structured logging，
  **不引入 Prometheus/OpenTelemetry**
- provenance 只记日志（commit/hash/target/checkpoint），**不塞进高频 IPC payload**（§96）

### 5.11 C offline tests（§98）

```text
check_deployment_contracts.py
check_policy_plan_dtype.py
check_causal_observation.py
check_fake_backend.py
check_inference_generation.py
check_plan_scheduler.py
check_candidate_publication.py
check_policy_failure_semantics.py
check_backend_swap.py
```

放进 `checks/offline/` 并纳入 `run_all.py`。

---

## 6. 关键设计决策点（实现时需拍板，文档留白处）

| 决策 | 文档建议 | 备注 |
|---|---|---|
| `MAX_POLICY_CHUNK_STEPS` | 32 或 64 | adapter requested N ≤ MAX 否则 fail，**禁止 silently truncate**（§61） |
| `policy_plan_ring` 槽数 | 2–4 | latest-wins（§63） |
| observation horizon 不足 | startup fail | 不 duplicate latest / 不为一个模型放大所有 ring（§55） |
| VR worker | 默认不启动 | 仅 adapter 声明需 VR（§85） |
| hand 可选 | `hand_enabled` / `--no-hand` | 对应现有 `HandParams` / `PolicyParams.hand_enabled` |
| recording | 初期不动 v16 schema | Policy recording 单独开 C-Recording Migration（§95/§97） |

---

## 7. 硬性禁止清单（§109 Reject List）

新代码一旦出现下面任一项即 REJECT：

1. Inference Worker 写 `arm_action_q` / `hand_cmd_ring`
2. Inference Worker 拥有 `SafetyState` / import robot SDK
3. model adapter 持有 `SharedStorage`
4. coordinator 里出现 model-specific branch（`if backend == "dexmani_policy"`）
5. core 包 import-time import torch（必须 child lazy import）
6. 整段 action chunk 直接 dump 进 robot transport（§73/§74）
7. application-side arm 插值（§78）
8. 并行 process watchdog / 并行 recording framework / 新 global plugin registry
9. 高频 IPC 用 JSON/object dtype
10. XHand lifecycle 被 policy deployment 改动

---

## 8. 离线可做 vs 硬件门

**离线可做**：C0–C13 全部代码 + `checks/offline/*` + `compileall`。故障矩阵（§99）——
Model / Observation / Generation / Scheduler / Runtime 五类——全部用 fake 设备离线验证，
**fail closed**。

**硬件门 H0–H6**（§101–107）离线不可做，只能文档化 + 手工 checklist：

- H0 no-command（观察 inference/plan/candidate，不动）
- H1 connected dry-run
- H2 arm-only restricted
- H3 arm+hand
- H4 pause-during-inference（验证 old plan never executes）
- H5 杀 inference 进程验证 supervisor fault
- H6 soak

---

## 9. 建议执行顺序（对齐 §108/§110）

```text
C0/C3 契约+协议 → C1 因果读取器抽取 → C2 候选发布合并
→ C4 loader/config → C5 dtype+ring → C6 FakeBackend
→ C7 inference worker → C8 coordinator/scheduler → C9 failure semantics
→ C10 lifecycle+CLI → C11 DexMani adapter → C12 backend swap → C13 metrics/provenance
→ C offline gate（checks/offline/run_all.py 全绿）→ 文档化 H0–H6 硬件门
```

每个 subphase 按执行文档 §111 的报告格式单独 review + commit，**不合并成巨大 diff**。

---

## 附：提交拆分建议（对齐 §108）

```text
C0  docs: freeze A/B runtime contracts
C1  refactor(shm): extract causal observation reader
C2  refactor(policy): add reusable candidate publication boundary
C3  feat(deployment): add backend/adapter contracts
C4  feat(deployment): add lazy backend loader and configuration
C5  feat(shm): add policy plan dtype and ring
C6  feat(deployment): add deterministic fake backend
C7  feat(deployment): add inference worker
C8  feat(deployment): add coordinator and endpoint scheduler
C9  feat(deployment): add lifecycle and thin CLI
C10 test(deployment): add failure and generation regression checks
C11 feat(integration): add DexMani Policy adapter
C12 test(deployment): verify backend replacement
C13 docs: document deployment runtime and hardware gates
```
