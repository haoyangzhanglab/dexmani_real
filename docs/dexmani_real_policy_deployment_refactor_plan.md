# DexMani Real 策略部署与推理重构方案

> 版本：2026-08-29 Batch 5 time-bounded shadow revision
> 主仓库：`/home/zhanghaoyang/Desktop/dexmani_real`
> 模型仓库：`/home/zhanghaoyang/Desktop/dexmani_policy`
> 唯一操作者入口：`examples/run_policy.py`
> 文档用途：后续分批实现、review、离线验证和真机放行的统一规格

---

## 1. 结论与实施主线

本轮不重写现有 deployment runtime。当前进程隔离、generation、freshness、
SafetyState、coupled publication 和硬件 worker 二次校验都是正确边界，应继续保留。

需要完成的主线只有四条：

1. `dexmani_policy` 生成可独立部署、可校验的 checkpoint contract；
2. `examples/run_policy.py --experiment-dir` 确定性选择并预检最新 checkpoint；
3. 用 rolling `ActionBuffer` 替换 coordinator 的整段 chunk commitment；
4. 简化 learned-policy motion gate，并用 shadow 与分段指标证明闭环可运行。

推荐实施顺序：

```text
artifact contract + legacy export
→ experiment entry + offline preflight
→ PolicyPrediction + full actionable future
→ rolling ActionBuffer + typed safety disposition
→ shadow + readiness metrics
→ profile-driven micro-optimization
→ physical validation ladder
```

首轮明确不实现 temporal ensemble。先采用可证明、可测试的 `latest-wins`；只有在
rolling latest 模式通过 shadow 和低风险真机验证后，才单独评估同一 target 上的
temporal aggregation。

---

## 2. 2026-08-28 源码事实基线

实施必须从以下事实出发，不能把本文目标状态误当成已完成代码。

### 2.1 仓库版本与工作样本

本节记录 2026-08-28 开始实施前的历史基线；当前通过 gate 的 artifact 与状态见
第 13 节和第 21.9 节 receipt，不得再把以下“尚无 deployment artifact”描述当作现状。

审查时版本：

```text
dexmani_real   fd7d275
dexmani_policy 7e31d10
```

固定部署样本：

```text
/home/zhanghaoyang/Desktop/dexmani_policy/
  experiments/dp3/pick_place_toy/2026-08-28_13-59_42
```

当时最新 training selector：

```text
checkpoints/latest.pt
→ epoch=1126-step=00080000-milestone=100pct.pt
```

源 checkpoint 大小为 `1,100,542,334` bytes。当前 experiment 尚无
`deployment_latest.pt` 和相邻 `.deployment.json`，因此它只能作为 migration/export
输入，不能直接绕过新 contract 进入真机 lifecycle。

该实验配置的固定事实：

```text
task_name       pick_place_toy
action_key      action
action_dim      19
horizon         16
n_obs_steps     2
n_action_steps  8
pad_before      1
pad_after       7
point count     1024
use_ema         true
denoise_steps   10
```

完整 actionable future 为：

```text
horizon - (n_obs_steps - 1) = 16 - 1 = 15 steps
```

在 16 Hz 训练/控制时间网格上，对应约 `0.9375 s` 的预测覆盖。

### 2.2 当前 `dexmani_real` 行为

| 边界 | 当前事实 | 目标 |
|---|---|---|
| CLI | `run_policy.py` 要求 runtime/checkpoint/task/action 等重复参数 | 正常路径只接收 experiment identity 和 Real/operator 参数 |
| artifact | parent 直接依赖 checkpoint 路径 | experiment selector + hash-bound sidecar |
| adapter | 仍导入并处理 FAAS | 删除全部 FAAS 分支 |
| checkpoint load | adapter 存在重复加载风险 | worker 内单次 deserialize/restore |
| model output | adapter 返回带物理时间的 `JointActionChunk` | model 只返回无时间的 `PolicyPrediction` |
| output horizon | 只消费 8-step `control_action` | 消费 15-step full actionable future |
| coordinator | `active_plan + pending_plan`，旧 chunk 结束后新 plan 才接管 | rolling latest-wins |
| safety disposition | gate reject 通常直接结束 run | motion reject discard endpoint；fatal contract/checker error abort |
| arm delta | coordinator 8°，worker 20° | learned gate 与 worker 均使用 canonical 20°；删除 8° |
| hand delta | coordinator 0.3 reject，worker 0.3 ramp | coordinator 不做 delta reject；worker 保留 measured-state ramp |
| metrics | latency gauge 主要保存最新值 | bounded p50/p95/p99 + usable horizon |
| inference trigger | 5 ms polling 后才判断 logical step 是否重复 | 新 logical step/pointcloud sequence 再构建 observation |

### 2.3 当前正确行为，禁止回归

- inference worker 是唯一拥有 model/CUDA 的进程；
- policy 输出只是 proposal，不能写 command ring；
- coordinator 是 learned-policy 唯一 command producer；
- 每个 control tick 最多发布一个 coherent arm-hand endpoint；
- action 时间属于 observation logical grid，不能按 inference finish time 平移；
- 过期 endpoint 只能 drop/mask，不能 retime；
- generation、motion permit、ticket、freshness 在 publication 和 worker 边界继续复核；
- EE action 必须先经 IK，再进入最终 SafetyGate；
- collision、workspace、joint bounds、hardware worker validation 继续 fail closed；
- 构造器保持无硬件副作用，设备连接由显式 worker lifecycle 拥有。

---

## 3. 固定设计决策

以下结论已经完成 review，实施期间不再反复重新设计。

### 3.1 FAAS 不属于部署合同

`dexmani_policy` 已经没有 FAAS。`dexmani_real` 中以下内容全部删除：

- `inject_faas_into_agent` import/call；
- `use_faas` manifest 字段；
- FAAS normalizer 维度分支；
- FAAS hand mapping；
- 对应 stale tests/config/comments。

缺失或不兼容的 checkpoint 应明确拒绝，不保留 FAAS compatibility mode。

### 3.2 Real 数据固定 `pad_before=1`

Real dataset contract 必须记录：

```text
pad_before = 1
padding = repeat-edge
```

它属于训练 sampler contract，不是 runtime 使用 B 前 feedback 的授权。

第一版 runtime 仍等待两个完整、不同且都属于 B 后 epoch 的 causal observation frame。
不得把 reset-home 或 ARMED 阶段 feedback 填入 policy episode。

### 3.3 单一操作者入口

正常部署只使用：

```bash
conda run -n real_robot python examples/run_policy.py \
  --experiment-dir \
  /home/zhanghaoyang/Desktop/dexmani_policy/experiments/dp3/pick_place_toy/2026-08-28_13-59_42 \
  --device cuda:0 \
  --execution-mode shadow \
  --hand \
  --max-running-seconds 120
```

`run_policy.py` 不再要求操作者重复填写：

```text
runtime_target
checkpoint
task_name
action_key
observation_horizon
observation_fields
pointcloud_num_points
action dimensions
```

这些值由 artifact contract 提供。项目内部 fake runtime 使用单测和纯 helper，不通过
真实操作者 CLI 暴露第二套部署协议。

### 3.4 XHand reset-home 时序

当 `--hand` 开启时：

```text
connect
→ tactile calibration
→ issue reset_home exactly once
→ obtain valid initial feedback
→ hand ready
→ all subsystems ready
→ ARMED
→ B becomes available
```

`reset_home` 的完成条件只是 SDK 接受/下发该动作：

- 不等待关节收敛；
- 不检查 home tolerance；
- 不因 measured qpos 偏离 home 而阻止 ready；
- 不为等待收敛重复下发 reset；
- SDK 明确拒绝则 fail closed；
- CRC 无法确认时记录告警，若后续 feedback 有效则允许 readiness。

按 B 后才创建新的 `run_generation` 和 `run_started_monotonic_ns`。

### 3.5 SafetyGate 简化边界

learned-policy motion gate 保留：

- representation / units / frame contract；
- finite、shape、generation；
- arm/hand joint bounds；
- canonical arm command jump 20°；
- workspace segment；
- full 19-DoF collision transition；
- hand mechanical preflight；
- freshness、deadline、motion permit、ticket checks。

删除：

- coordinator 的 arm 8° policy-specific delta；
- coordinator 的 hand 0.3 rad delta reject。

hand worker 继续用 measured-state 0.3 rad/tick ramp。SafetyGate 不 clip policy target。

### 3.6 首轮只做 latest-wins

首轮 ActionBuffer 不做 weighted average，不做 SO(3) blend，不做 uncertainty routing。

原因：latest-wins 已能解决当前 `active_plan/pending_plan` 引入的 chunk 接管延迟，且不会
创造需要再次解释的新动作。完成该闭环后再用数据判断 temporal ensemble 是否必要。

---

## 4. 范围与非目标

### 4.1 本轮支持

```text
single task
Real Zarr v5
joint_state + point_cloud
native XHand state
joint action 19D
EE action 21D（保留现有 contract；DP3 joint 优先验证）
local single-host inference worker
shadow / execute
```

首个真实模型固定为上述 DP3 joint checkpoint 的 deployment export。

### 4.2 本轮不做

- remote inference server / gRPC；
- multitask routing；
- adaptive inference frequency；
- temporal ensemble；
- CUDA graph、encoder cache；
- 自动 FP16/bfloat16；
- 自动 `torch.compile`；
- 减少 diffusion denoise steps；
- 修改 16 Hz policy control grid；
- 用插值或重定时补发过期 action；
- 为兼容旧 checkpoint 猜测缺失 metadata。

---

## 5. Artifact contract 与 experiment 解析

### 5.1 产物布局

当前已实现的 retrofit publisher 生成独立的 deployment-only v2：

```text
<experiment>/checkpoints/
  epoch=...-deployment-v2.pt
  epoch=...-deployment-v2.pt.deployment.json
  deployment_latest.pt -> epoch=...-deployment-v2.pt
  latest.pt            -> training-selected checkpoint
```

当前 training `simple.v1` checkpoint 包含 monitor/optimizer/scheduler，不能直接成为
deployment selector target。`deployment_latest.pt` 只允许指向 `dexmani.deployment.v2`；
v2 仅保留 JSON contract、model 与 EMA（normalizer 随 state dict 保留），并由 restricted
loader 读取。native v2 producer 尚未实现；在实现并单独 review 前，所有部署 artifact 都
必须走 retrofit publisher。legacy migration 不覆盖源 checkpoint，只新增 deployment
artifact。

selector 发布顺序：

```text
write checkpoint temp
→ fsync/close
→ atomic rename checkpoint
→ write sidecar temp
→ atomic rename sidecar
→ atomic replace deployment_latest.pt symlink
```

selector 永远不能指向半成品。

### 5.2 Checkpoint 内嵌 contract

checkpoint 是模型恢复的唯一事实源，至少包含：

```text
schema_version
resolved inference_config
data_contract
train_params
model weights
EMA weights when eval.use_ema=true
normalizer
producer provenance
```

deployment v2 明确不包含 `monitor`、optimizer 或 scheduler；这些只属于 source training
checkpoint 与 source-integrity evidence。training loader 拒绝 v2，deployment loader 也拒绝
`simple.v1`，禁止在训练恢复与部署恢复之间静默互换格式。

`data_contract` 至少覆盖：

```text
domain = real
task_name
action_key
control_dt_s
sensor_modalities
joint/action dimensions
point-cloud count, feature dimension, frame and preprocessing identity
Real Zarr schema/version/hash identity
pad_before = 1
padding = repeat-edge
```

`train_params` 至少覆盖：

```text
horizon
n_obs_steps
n_action_steps
action_dim
control_action_dim
action_key
```

不允许：

```text
missing metadata → read experiment config.yaml and guess
missing EMA → silently use raw model
missing normalizer → construct identity normalizer
```

### 5.3 Sidecar 只服务 parent allocation

parent 必须在 inference worker 启动前分配 shared memory，但不能为了获得 shape 在 parent
进程加载 1.1 GB checkpoint。因此相邻 JSON 是 checkpoint contract 的小型、hash-bound
projection，不是第二份模型 YAML。

推荐结构：

```text
{
  "schema_version": 1,
  "checkpoint": {
    "filename": "epoch=1126-step=00080000-deployment-v2.pt",
    "size_bytes": <actual exported artifact integer>,
    "sha256": "..."
  },
  "embedded_contract_sha256": "...",
  "allocation": {
    "task_name": "pick_place_toy",
    "action_key": "action",
    "action_dim": 19,
    "n_obs_steps": 2,
    "n_action_steps": 8,
    "horizon": 16,
    "required_action_steps": 15,
    "control_dt_s": 0.0625,
    "sensor_modalities": ["joint_state", "point_cloud"],
    "observation_fields": ["arm_qpos", "hand_qpos", "point_cloud"],
    "requires_hand": true,
    "point_cloud_num_points": 1024,
    "point_cloud_feature_dim": 6
  },
  "producer": {
    "repository": "haoyangzhanglab/dexmani_policy",
    "commit": "...",
    "metadata_provenance": "native-or-retrofitted"
  }
}
```

Real IPC capacity 继续由 `MAX_POLICY_CHUNK_STEPS=32` 拥有。artifact 只声明
`required_action_steps=15`，parent 校验它不超过 32；checkpoint 不能定义 Real ring 容量。

### 5.4 确定性 selector

```text
1. 若 deployment_latest.pt directory entry 存在，选择它；
2. 否则选择 latest.pt；
3. 两者都不存在则拒绝。
```

已选择的 entry 无效时立即拒绝，不能 fallback 到另一个 selector。特别是 dangling
`deployment_latest.pt` 不能被忽略。

解析纯函数返回：

```python
@dataclass(frozen=True)
class ResolvedPolicyArtifact:
    experiment_dir: Path
    selector_path: Path
    checkpoint_path: Path
    sidecar_path: Path
    selector_name: str
    checkpoint_size_bytes: int
    checkpoint_sha256_from_index: str
    index_sha256: str
    allocation_contract: PolicyArtifactContract
```

解析阶段必须校验：

- experiment root、`config.yaml`、`checkpoints/` 存在；
- selector 不是 dangling symlink；
- resolved checkpoint 为 regular、非空、可读 `.pt`；
- selector 和 resolved checkpoint 均不越出 experiment `checkpoints/`；
- sidecar 存在、不是 out-of-tree symlink、canonical JSON 合法；
- sidecar filename/size 与固定 resolved checkpoint 一致；
- `required_action_steps <= 32`；
- allocation fields 内部一致。

完整 SHA-256 在 preflight/shadow/execute 计算并校验。`--print-config` 不扫描 1.1 GB
文件，只显示 sidecar 声明值并明确：

```text
checkpoint_sha256_verified=false
```

preflight hash/load 前后都要比较固定 file identity；run 内不得重新解析已变化的 symlink。

---

## 6. `examples/run_policy.py` 最终 CLI 与生命周期

### 6.1 CLI

保留/新增：

```text
--experiment-dir PATH        正常路径必填
--config PATH                Real hardware/runtime config
--deployment-config PATH     只允许 Real-owned timing/readiness overrides
--device DEVICE              operator-owned inference device
--hand                       显式 XHand hardware acknowledgement
--execution-mode shadow|execute
--max-running-seconds FLOAT  operational B-relative limit (shadow or bounded H4 execute)
--execute-max-published-endpoints 1  H4 execute 的固定 one-publication bound
--execute-ack-timeout-seconds FLOAT  H4 arm+hand ACK 的有限等待上限
--print-config
--preflight-only
```

规则：

- `--execution-mode` 默认 `shadow`；
- execute 必须显式写在 CLI，YAML 不能暗中开启物理发布；
- `--max-running-seconds` 必须为有限正数。在 shadow 中它是 operational B-relative limit；在
  H4 execute 中它被冻结到 immutable bounds。`print-config`/`preflight-only` 可以显示和校验
  H4 bounds，但不声称已完成 operational timed run；
- execute 还必须显式传入 `--hand`、`--execute-max-published-endpoints 1` 和有限正数
  `--execute-ack-timeout-seconds`；YAML 不能暗中开启或放宽该 profile；
- duration 从 `SafetyState.RUNNING` 的 atomic B epoch 开始。到期时 supervisor 写 typed
  `RUN_TIME_LIMIT` stop request，由 coordinator 正常撤销 motion、生成 `reason="run time limit"`
  的 receipt 后再结束 session；超过 policy heartbeat grace 未回到 ARMED 则 FAULT；
- artifact 要求 hand，而 CLI 未给 `--hand` 时，在启动硬件前拒绝；
- artifact-owned YAML 字段只能作为相等 expectation，不能 override；
- `run_policy.py` 只做参数解析、纯 resolution 和 lifecycle dispatch；
- `--print-config` 和 `--preflight-only` 分支不得导入 hardware lifecycle；
- experiment path 不加入 `sys.path`，不执行 experiment 中的 Python 文件；
- `dexmani_policy` 必须通过 editable install 或 wheel 进入环境。

### 6.2 三种执行路径

#### `--print-config`

```text
parse CLI
→ resolve Real config
→ resolve artifact/index
→ render canonical run identity
→ exit
```

禁止 torch load、worker、相机或机器人 import/连接。

#### `--preflight-only`

```text
resolve
→ verify checkpoint SHA-256
→ start isolated inference worker
→ load checkpoint once
→ cross-check embedded/index contracts
→ restore EMA + normalizer
→ run deterministic fake-observation shape/finiteness preflight
→ close worker
→ exit
```

禁止构建 hardware worker specs。

#### `shadow` / bounded H4 `execute`

```text
resolve + verify artifact
→ allocate IPC from validated projection
→ start inference worker only
→ load/check/preflight model once; inference ready
→ start camera/pointcloud/arm/hand/coordinator workers
→ hand reset_home before readiness/B
→ all workers ready
→ ARMED
→ enable keyboard operator
```

H4 execute 在此之后最多发布一条完整 arm+hand coupled record，等待两个 worker 都确认同一
action id，并在成功时回到 ARMED。ACK、feedback、generation、SafetyGate、worker 或超时故障
均进入 sticky FAULT，不重试、不发送第二条 policy publication。

正常 lifecycle 不先运行一个临时 preflight worker再加载第二份模型。正式 inference worker
在 hardware startup 前完成相同 preflight，然后继续服务该 run。

### 6.3 Artifact-owned 与 Real-owned 配置

```text
artifact-owned:
    checkpoint identity
    task/action/observation contract
    model horizon and dimensions
    required action steps
    training dt
    point-cloud model contract
    padding and normalizer/EMA semantics

Real/operator-owned:
    hardware runtime config
    device
    inference_hz
    freshness/deadline/silence limits
    execution_mode
    hand acknowledgement
```

`runtime_target` 固定为：

```text
dexmani_real.integrations.dexmani_policy:DexManiPolicyRuntime
```

不再是正常部署配置项。

---

## 7. B 前后完整链路

### 7.1 启动到 ARMED

```text
run_policy.py
→ pure config/artifact resolution
→ hash-bound inference preflight
→ inference ready
→ hardware workers start
→ XHand connect/calibrate/reset_home once/valid feedback
→ all subsystem ready
→ SafetyState.ARMED
→ operator may press B
```

inference worker 在 ARMED：

- 不构建 observation；
- 不调用 model；
- 只维持 heartbeat/lifecycle。

### 7.2 按 B 后

```text
B
→ begin_motion under motion lock
→ invalidate old coupled ticket
→ increment run_generation
→ set run_started_monotonic_ns
→ SafetyState.RUNNING
→ accept only source timestamps >= run start
→ wait complete 2-frame post-B causal history
→ build ObservationBatch
→ PolicyRuntime.predict
→ full 15-step PolicyPrediction
→ worker stamps immutable logical-grid targets
→ mask expired endpoints
→ publish plan
→ ActionBuffer.push
→ latest-wins peek_due
→ EE→IK if required
→ SafetyGate/publication validation
→ shadow: no coupled write, commit token
→ execute: coupled write succeeds, commit token
→ worker revalidates permit/generation/ticket
→ SDK command
→ fresh feedback
```

S、fault、e-stop、shutdown 或 timeout 都必须 revoke motion permit 并推进 generation。
旧 observation、prediction、ActionBuffer endpoint 和 coupled ticket 随即失效。

---

## 8. Model/runtime 边界重构

### 8.1 当前问题

当前 adapter 根据 `InferenceContext` 生成物理 target timestamp，因此 model integration
同时拥有模型解码和 Real 时间语义。它还只返回 `control_action` 8 步。

### 8.2 新 `PolicyPrediction`

```python
@dataclass(frozen=True)
class PolicyPrediction:
    arm_qpos: np.ndarray | None       # [N, 7]
    hand_qpos: np.ndarray | None      # [N, 12]
    ee_pos: np.ndarray | None = None  # [N, 3]
    ee_rot6d: np.ndarray | None = None  # [N, 6]
```

约束：

- joint 与 EE 二选一；
- N 必须等于 artifact `required_action_steps`；
- shape 和 finite 在构造边界校验；
- 不包含 monotonic timestamp、valid mask、generation 或 shared-memory 状态。

接口收敛为：

```python
class PolicyRuntime(Protocol):
    def load(self) -> None: ...
    def reset_episode(self) -> None: ...
    def predict(self, observation: ObservationBatch) -> PolicyPrediction: ...
    def close(self) -> None: ...
```

### 8.3 adapter 解码 full actionable future

DexMani Policy 已返回：

```text
pred_action
control_action
tail
```

adapter 使用：

```text
pred_action[:, n_obs_steps - 1 :, :control_action_dim]
```

或严格校验后拼接：

```text
control_action + tail
```

首选直接从 `pred_action` 取单一连续 slice，并用 `control_action/tail` 做 parity test，避免
双路径语义漂移。当前 DP3 必须得到 `[1, 15, 19]`。

### 8.4 worker 拥有 physical timing

新增纯 helper：

```python
def stamp_prediction_timing(
    prediction: PolicyPrediction,
    *,
    logical_step_ns: int,
    step_dt_ns: int,
    inference_finished_ns: int,
    command_lead_ns: int,
) -> JointActionChunk | None:
    ...
```

时间规则：

```text
target[i] = logical_step_ns + i * step_dt_ns
valid[i]  = target[i] > inference_finished_ns + command_lead_ns
```

`logical_step_ns` 必须是由当前 `run_started_monotonic_ns` 和整数 control step 推导出的
逻辑网格时间，而不是未经对齐的 sensor arrival time。这样不同 prediction 对相同未来
step 产生完全相同的 target key，latest-wins 才能做确定性替换。

全过期返回 `None`。严禁把第一个可用 action 平移到 `now + lead`。

---

## 9. Rolling ActionBuffer

### 9.1 替换范围

新增：

```text
dexmani_real/deployment/action_buffer.py
```

删除 coordinator 中：

```text
active_plan
pending_plan
active_plan_id
pending_plan_id
next_step
_ready_to_replan
promotion branches
```

ActionBuffer 只做纯 scheduling，不导入 hardware、planner、SafetyGate 或 shared memory。

### 9.2 数据结构

```python
@dataclass(frozen=True)
class BufferedPlan:
    plan_id: int
    run_generation: int
    observation_id: int
    observation_latest_source_ns: int
    inference_finished_ns: int
    deadline_ns: int
    chunk: JointActionChunk


@dataclass(frozen=True)
class EndpointToken:
    token_id: int
    target_ns: int
    plan_id: int
    observation_id: int
    step_index: int
```

buffer 数量必须有显式上限，且该上限由 Real runtime 计算，不进入 checkpoint contract：

```text
max_buffered_plans = min(32, max(2, ceil(max_plan_age_s * inference_hz) + 2))
```

默认 `max_plan_age_s=1.0`、`inference_hz=10` 时保留最多 12 个 plan。`push` 前先 prune；
仍达到上限时按 `(observation_id, plan_id)` 确定性移除最旧 plan 并计数。所有 endpoint
同时受 32-step IPC 上限、deadline 和 generation 限制，禁止无界 Python list。

### 9.3 最小 API

```python
class ActionBuffer:
    def reset(self, *, run_generation: int) -> None: ...
    def push(self, plan: BufferedPlan, *, now_ns: int) -> PushResult: ...
    def peek_due(self, *, now_ns: int) -> PeekResult: ...
    def commit(self, token: EndpointToken) -> None: ...
    def discard(self, token: EndpointToken, *, reason_code: str) -> None: ...
    def coverage(self, *, now_ns: int) -> BufferCoverage: ...
```

`peek_due` 不改变 buffer。只有成功 shadow validation/physical publish 后才能 `commit`；
明确 motion rejection 走 `discard`。

### 9.4 latest-wins 算法

每个 coordinator tick：

1. prune generation mismatch、expired、deadline closed、已 finalize endpoint；
2. 收集 `target_ns <= now_ns` 的 due endpoint；
3. 选择其中最大的 `target_ns`，更早 overdue endpoint 计为 coalesced；
4. 同一 target 若有多个 candidate，选择最大 `observation_id`，再以 `plan_id` 破平局；
5. 返回 candidate + immutable token；
6. SafetyGate 后 commit/discard。

buffer 维护当前 generation 的单调 `finalized_through_ns`。无论 `commit` 还是 motion
`discard`，都将 selected target 以及本次被 coalesce 的更早 due target 一并 finalize；
以后到达的任何 plan 都不能重新激活 `target_ns <= finalized_through_ns` 的 endpoint。
transient publication deferral 不推进该 watermark。

新 prediction 可立即替换未执行的重叠未来 endpoint，不等待旧 chunk 完成。新 prediction
首个可用 target 尚未到达时，旧 plan 的仍合法 endpoint 可提供 fallback。

### 9.5 不允许的 fallback

若当前 target 的最新 candidate 因 motion safety 被拒绝：

```text
discard target
→ no command
→ do not try an older candidate at the same target
→ wait fresh prediction
```

原因：旧 candidate 并未在当前最终仲裁结果下得到安全证明。silence watchdog 负责在持续
无合法动作时终止 run。

### 9.6 三种 coverage 状态

```text
DUE        当前有 endpoint 可评估
FUTURE     没有 due，但仍有未过期未来覆盖
EXHAUSTED  没有合法未来覆盖
```

`FUTURE` 只等待，不 abort；`EXHAUSTED` 也不自动发 hold，由 first-command 或 command-
silence watchdog 决定是否结束 run。

---

## 10. Safety disposition 重构

### 10.1 验证顺序

```text
ActionBuffer latest-wins
→ optional EE IK
→ ActionCandidate
→ runtime/fresh-feedback snapshot
→ SafetyGate
→ hand preflight
→ shadow validation or coupled publish
```

不能对每个 prediction 先过 gate，再从多个结果中选择；最终 candidate 才是需要验证的
物理动作。

### 10.2 Motion reject：discard，不终止整次 run

以下属于 policy endpoint 的可恢复 motion rejection：

```text
arm joint bound
hand operational/mechanical bound
arm canonical 20° jump
workspace violation
collision transition
IK no valid solution / IK geometry reject
```

处理：

```text
no publish
→ ActionBuffer.discard(token)
→ metric by reason code
→ wait next prediction
```

### 10.3 Fatal：立即 abort 到 ARMED

以下表示 contract、checker 或 runtime 完整性失效：

```text
unsupported representation/units/frame
invalid action shape
NaN/Inf action
embedded/index contract mismatch
workspace checker exception
collision checker exception
invalid SafetyGate configuration
publication invariant violation
```

需要为 collision exception 新增独立 code，例如：

```text
COLLISION_CHECK_FAILED
```

不能继续复用 `COLLISION_TRANSITION`，否则无法区分“确实碰撞”和“检测器失效”。

### 10.4 Transient：不 finalize，受 watchdog 约束

短时 feedback 不可用、feedback stale 或 publication 暂时失去可用 snapshot 时：

- 不发布；
- 不假装 safety reject；
- token 可在有效窗口内保留到下一 tick；
- generation 改变立即清空；
- first-command/silence timeout 最终收敛。

如果底层 worker 已明确报告 hardware fault，沿现有 fault lifecycle 处理，不归类为
policy motion reject。

### 10.5 Canonical arm/hand 规则

```text
learned coordinator arm delta = runtime.arm.max_servo_command_jump_rad = 20°
arm worker final jump check    = same canonical 20°
learned coordinator hand delta = disabled
hand worker measured ramp      = 0.3 rad/tick
```

20° 在 coordinator 是为了在进入 command ring 前拒绝明显不连续 endpoint；worker 的同值
检查仍作为最终设备边界，不应删除。

---

## 11. Shadow publication

`execution_mode` 必须进入 immutable resolved run config，并由 publication owner 执行，
不能在 coordinator 到处散布条件分支。

推荐抽取：

```python
def validate_or_publish_candidate(
    ...,
    execution_mode: ExecutionMode,
) -> CommandPublishResult:
    ...
```

两种模式共享：

- observation/provenance checks；
- ActionBuffer selection；
- IK；
- SafetyGate；
- hand preflight；
- metrics；
- commit/discard semantics。

差异只有最后一步：

```text
shadow  → do not write coupled_cmd_ring; return validated
execute → write coupled_cmd_ring; return published
```

shadow 仍会连接真实传感器/robot feedback，并在 B 前执行一次 XHand reset_home。这一点必须
在 CLI 日志中明确提示。B 后 shadow 不得产生任何 policy coupled command write。

---

## 12. 可观测性与回路优化

### 12.1 必须先增加的指标

当前单值 gauge 无法判断尾延迟。使用每个 worker 私有的有界 deque/rolling stats，不引入
Prometheus/OpenTelemetry。

至少记录：

#### Observation

```text
observation_build_ms p50/p95/p99
observation_age_ms p50/p95/p99
observation_skew_ms p50/p95/p99
observation_build_skipped_same_step
effective_observation_hz
```

#### Inference

```text
host_to_device_ms p50/p95/p99
model_inference_ms p50/p95/p99
device_to_host_ms p50/p95/p99
inference_total_ms p50/p95/p99
effective_inference_hz
usable_action_steps p50/p05
usable_horizon_ms p50/p05
fully_expired_predictions
```

#### Scheduling/publication

```text
plans_ingested
plans_superseded
buffer_plan_count
buffer_future_endpoint_count
endpoints_due
endpoints_coalesced
endpoints_committed
endpoints_discarded_by_code
transient_publish_deferrals
plan_to_command_ms p50/p95/p99
source_to_command_ms p50/p95/p99
first_command_ms
command_gap_ms p95/p99
```

#### Safety

```text
motion_reject_by_code
fatal_reject_by_code
checker_failure_by_code
IK failure by reason
first-command timeout
command-silence timeout
```

### 12.2 Observation build 优化

当前逻辑先构建 observation，后判断 `logical_step_ns` 是否变化。目标顺序：

```text
read latest sequence/time anchor cheaply
→ if no new eligible logical step: skip
→ build and align complete history once
→ validate
→ assign observation_id
→ infer
```

同时：

- `parse_observation_fields` 在 worker startup 缓存一次；
- freshness/skew/grid budgets 预计算为 ns；
- observation_id 只在完整 observation 被接受时递增；
- 先保留现有 causal alignment 算法，不在同一批改写 ring API；
- 只有 profile 证明 ring copy 显著后，才增加按时间锚点读取 API。

### 12.3 Adapter 内存路径

在 correctness 完成后做低风险调整：

- 已是 float32 时使用 `astype(np.float32, copy=False)`；
- joint/pointcloud 各组装一次 contiguous array；
- 删除不必要的中间 stack/copy；
- 只有 H2D 指标明显时才引入 pinned memory/non-blocking transfer；
- `.cpu().numpy()` 继续作为 CUDA completion boundary，保证 inference timing 真实。

### 12.4 Checkpoint startup

- parent `--print-config` 不读完整 checkpoint；
- preflight/shadow/execute 必须完成一次内容 SHA-256；
- inference worker 只 deserialize 一次；
- 从已加载 checkpoint object 恢复 agent/EMA/normalizer；
- 不调用会再次按 path load 的 helper。

完整 hash 是部署完整性校验，不以“优化启动速度”为由删除。

### 12.5 不应先做的优化

- 不先把 inference_hz 从 10 提到 16；
- 不改变训练/执行 16 Hz logical grid；
- 不减少 denoise steps；
- 不直接开启 mixed precision/compile；
- 不关闭 collision/freshness/generation checks；
- 不强制处理每个 camera frame；pointcloud 继续 latest-only；
- 不把相机采样频率等同于 policy control frequency。

滚动调度落地后，再依据 `usable_horizon` 和 inference p95 判断 10 Hz 是否需要调整。

---

## 13. 分批实施计划

每个批次完成后必须停下 review 和验证。后一批不能掩盖前一批失败。

当前状态与依赖关系如下：

| Batch | 当前状态 | 前置条件 |
|---:|---|---|
| 0 | GATE PASS | 109 tests + compileall + 双 review + 独立 validation |
| 1 | GATE PASS | P1a/P1b contract + v2 restricted artifact：Policy 82 tests、双 review、独立 validation；reference v2 receipt PASS |
| 2 | GATE PASS | R2a resolver + R2b CLI/preflight：Real 141 tests、双 review、独立 validation；CPU/CUDA H0 PASS |
| 3 | GATE PASS | PolicyPrediction + 15-step future + worker timing：Real 145 tests、双 review；CPU/CUDA H0 15×19 PASS |
| 4 | GATE PASS | ActionBuffer、coordinator 一次性切换、typed disposition：Real 172 tests、双 review、独立 validation 均通过 |
| 5 | H2/H3 shadow zero-write PASS；boundary retest 与 B-relative watchdog 真机验证 PASS | 三次限定真机 shadow receipt；float32 endpoint roundoff 仅在 policy publication boundary 规范化，最近一次由内置 watchdog 在 B 后 120.040 s 正常停止 |
| 6 | pending | 已有 profile evidence；H2/H3 定时监督已验证，仍需先从现有指标确认是否存在值得优化的实际瓶颈 |

### Batch 0 — Regression baseline

仓库：`dexmani_real`

目标：只补纯测试，不改变行为。

锁定：

- immutable logical timeline；
- expired endpoint mask，不 retime；
- one endpoint per control tick；
- generation invalidation；
- policy 不能绕过 publication/SafetyGate；
- hand reset_home 在 ready/B 前只下发一次且无 tolerance 判定；
- ARMED 不 inference；
- B 前 feedback 不进入新 observation epoch。

Gate：相关原有测试和新增测试全部通过，focused diff 无 production behavior change。

### Batch 1 — Policy artifact producer + legacy migration

仓库：`dexmani_policy`

修改：

- Real Zarr root attrs/data contract 读取与严格验证；
- Real dataset 固定并记录 `pad_before=1/repeat-edge`；
- checkpoint 内嵌完整 inference/data/train contract；
- sidecar projection/hash；
- atomic `deployment_latest.pt` 发布；
- legacy exporter；
- EMA/normalizer fail-closed restore helper 接受已加载 checkpoint object。

使用固定 step-80000 checkpoint 生成 deployment artifact。源文件保持不变。

Gate：

- export 可重复，输出 contract/hash 一致；
- source model/EMA/normalizer tensor key 与 shape 完整保留；
- source config、checkpoint tensor contract、Real dataset contract 任一冲突都拒绝；
- `metadata_provenance=retrofitted`；
- `deployment_latest.pt` 指向新 artifact。

P1b 已通过最终 gate：legacy exporter 使用单次、`O_NOFOLLOW` source deserialize，
checkpoint 目录内的临时文件、rename、sidecar 和 selector transaction 全部绑定同一
directory fd；两个独立进程对同一 synthetic source 的 checkpoint/sidecar 字节和 SHA-256
一致。严格 legacy retrofit 只允许从 canonical inference receipt 补入唯一缺失的
`use_aux_ee=false`，其他缺失、多余或冲突字段均拒绝。Policy 全量离线测试为 61 passed，
P1b focused 为 33 passed，compileall、diff check、双 review 和独立 validation 均通过。
最初的 v1 migration 保留了 training-only optimizer state，其中 Hydra `ListConfig` 使
restricted loader 正确拒绝该文件；v1 因此只作为历史失败样本保留，不是当前 deployment
target。随后发布 deployment-only v2，去除 monitor/optimizer/scheduler，并补齐严格 v2
schema、空 safe-globals、`weights_only=True` 单次 load、v1 不覆盖与 selector rollback 测试。
Policy 最终全量离线测试为 82 passed；双 review 与独立 validation 均为 PASS。真实
step-80000 v2 migration/material validation 已完成；source、v1 和训练 selector `latest.pt`
均未改变。

### Batch 2 — Experiment resolver + offline preflight

仓库：`dexmani_real`

修改：

- 新增纯 artifact resolver/index validator；
- 重构 `run_policy.py` CLI；
- `--print-config` 与 `--preflight-only`；
- artifact-owned values 注入 immutable config；
- adapter 删除全部 FAAS；
- checkpoint 单次 load/restore；
- import/package provenance 校验；
- run identity 记录 checkpoint/index/config/repo hashes。

本批 `--experiment-dir` 只允许 print/preflight。shadow/execute 暂时明确拒绝，避免半完成
CLI 落入当前仍会发布 command 的 coordinator。

R2a 已通过最终 gate：resolver 以 held experiment/checkpoints directory fd 和
`openat(..., O_NOFOLLOW)` 固定不可信文件系统边界；selector/sidecar 只允许 direct regular
或严格一跳 relative basename symlink，且在返回前复核目录、selector、checkpoint、sidecar
entry/target 身份。`deployment_latest.pt` 存在但 dangling/非法时 fail closed，不回退
`latest.pt`。resolver 只读取最多 64 KiB 的 canonical sidecar，不读取、哈希或反序列化
checkpoint，也不导入 Policy、torch、deployment lifecycle/worker 或硬件模块。

R2a 全量离线验证为 124 passed、58 subtests；focused 为 15 passed、40 subtests。双 review
和独立 validation 均为 PASS。独立 `strace` 确认 reference checkpoint 仅被 no-follow
open/close，sidecar 实际读取 799 bytes；连续 128 次 resolve 的 fd 数保持 4→4。

Gate：

```bash
conda run -n real_robot python examples/run_policy.py \
  --experiment-dir <reference-experiment> --print-config

conda run -n real_robot python examples/run_policy.py \
  --experiment-dir <reference-experiment> --preflight-only
```

两者都没有 hardware import/worker/connection；preflight 成功恢复最新 deployment artifact。

R2b 已通过最终 gate：CLI 只接受 experiment identity 与 Real/operator 参数；print 路径不
导入 torch、Policy、worker、sensor 或 robot，也不扫描 checkpoint。preflight 在 spawn
child 中持有固定 no-follow fd，依次完成内容 hash、Policy package provenance gate、单次
restricted deserialize、exact embedded/index contract、EMA/normalizer restore 和一次确定性
fake-observation prediction；joint/EE 输出表示必须与 artifact `action_key` 一致。IPC 使用
16 KiB 有界 JSON，timeout/EOF/hung child 均执行 terminate/kill/join/close。

Real 全量离线验证为 141 tests（58 subtests），Policy v2 全量为 82 passed；contract、
filesystem/security 双 review 与独立 validator 均为 PASS。reference v2 在 CPU 与 CUDA
`--preflight-only` 均通过，当时的 runtime 输出为 8×19 `control_action`；该历史结果已由
Batch 3 的 15×19 full-future H0 receipt 取代。CUDA 首次在 sandbox 内失败仅因 spawn child
无 GPU device，可见性探针与同一 preflight 在非 sandbox 环境均通过。以上均为 H0
offline 验证，不是 hardware validation。Batch 2 仍明确拒绝 operational shadow/execute。

### Batch 3 — Full future + timing ownership

仓库：`dexmani_real`

修改：

- 引入 `PolicyPrediction`；
- `PolicyRuntime.predict` 不再拥有 physical timing；
- adapter 解码 15-step full actionable future；
- worker `stamp_prediction_timing`；
- transport/output capacity cross-check；
- prediction/timing parity tests。

Gate：

- 当前 DP3 输出 `[15, 19]`；
- `pred_action` slice 与 `control_action + tail` 数值一致；
- target grid 与旧实现前 8 步完全一致；
- inference latency 只 mask，不平移；
- oversize/nonfinite output fail closed。

Batch 3 gate receipt（2026-08-29）：

```text
focused timing + preflight + artifact: 57 passed
full Real offline suite:                145 passed
compileall / diff --check:              PASS
contract/safety review:                 PASS
numerical/parity review:                PASS
reference CPU preflight:                15×19 PASS
reference CUDA preflight:               15×19 PASS
checkpoint SHA-256:                     b174bd483b64090cd3f5dbe0a5bfadd10998f5d27d43fc9aca06efb82242484c
Policy Python tree SHA-256:              b568a8c9c5885eb5953040fa1ad55e06f27a71f2c7faf866291c91b638fdb589
Real Python tree SHA-256:                a587de9593e9efa7943320a84db389580a3644d476694b9918e7f2e324da3a11
hardware / camera / shadow / execute:    not run
```

### Batch 4 — Rolling ActionBuffer + minimal motion gate

仓库：`dexmani_real`

修改：

- 新增纯 `action_buffer.py`；
- coordinator 删除 active/pending state；
- latest-wins/coalesce/coverage；
- peek/commit/discard；
- typed motion/fatal/transient disposition；
- arm gate 20°；
- hand coordinator delta disabled；
- split collision transition/checker failure codes；
- scheduling/safety metrics。

Gate：

- 新 plan 在旧 chunk 未结束时接管未执行 future；
- 新 plan 未覆盖时旧合法 future 仍可用；
- 同 target 最新 observation 胜出；
- 已 commit/discard target 不重复发；
- safety reject 不 fallback 到旧 candidate；
- 10° arm endpoint 在其余检查通过时被接受；
- >20° arm endpoint 被 motion-discard，command ring 不写；
- 大但合法 hand target 被 coordinator 接受，worker 单测证明仍按 0.3 ramp；
- checker exception 立即 abort；
- generation reset 清空 buffer。

Batch 4 gate receipt（2026-08-29）：

```text
ActionBuffer focused suite:            17 passed
R4 focused scheduler/safety suite:     69 passed, 12 subtests
full Real offline suite:               172 passed, 67 subtests
compileall / diff --check:             PASS
focused black / isort:                 PASS
scheduler/lifecycle review:            PASS
safety/IK review:                      PASS
hardware / camera / shadow / execute:  not run
```

其中 ActionBuffer 以 immutable copy、opaque token 和 finalized watermark 保证同一 logical
endpoint 恰好一次；新 plan 对相同 target 永久胜出，且不能在新 candidate motion-discard 后
回退到旧 candidate。`CHECKER_FAILURE`、非有限 IK output、坏 feedback contract 均是 fatal；
正常运动范围、delta、collision transition 与可解性 reject 只丢弃当前 endpoint，不终止 run。

### Batch 5 — Shadow + readiness evidence

仓库：`dexmani_real`

修改：

- publication owner 支持 shadow；
- CLI 开放 `--execution-mode shadow`；
- H4 execute 仅以固定 one-publication profile 开放，要求显式 `--hand`、ACK timeout 与
  B-relative duration；仍须单独取得 physical authorization；
- bounded latency stats；
- usable horizon/readiness report；
- startup reset-home/B epoch integration tests。

Gate：shadow RUNNING 中：

```text
real sensors                    yes
model inference                 yes
ActionBuffer                    yes
IK/SafetyGate                   yes
B 前 reset_home                 exactly once
home convergence/tolerance wait no
B 后 coupled command write      zero
fatal/checker error visibility  yes
```

需要形成一份 reference experiment 的 shadow report。阈值在获得真实分布后 review，不在
代码中凭空写死。

Batch 5 offline implementation receipt（2026-08-29）：

```text
execution mode:                       immutable shadow or bounded H4 execute; no unbounded execute
full validation:                      runtime + feedback + SafetyGate + hand + temporal
B 后 policy coupled command write:    structurally zero; sequence mutation or unobservable baseline latches FAULT
run receipt:                          canonical JSON, start/end coupled sequence + run totals
timing evidence:                      bounded 256-sample p50/p95/p99 for observation/inference/plan/horizon
hand acknowledgement:                 required before any IPC/process for hand-enabled shadow
reset_home:                           one startup request before ready/B; no tolerance wait
CRC_UNCONFIRMED reset response:       warn and continue after send; only REJECTED/exception fail closed
H key in policy lifecycle:             disabled (prevents post-B home publication)
B-relative duration limit:            shadow or immutable H4; coordinator-acknowledged or FAULT
H4 publication bound:                 exactly one complete coupled record; dual worker ACK required
focused R5 offline suite:             76 passed, 7 subtests
full Real offline suite:               187 passed, 67 subtests
compileall / diff --check:             PASS
focused black / isort:                 PASS
scheduler/metrics review:              PASS
safety/lifecycle review:               PASS
hardware / camera / shadow / execute: see H2/H3 report below / execute not run
```

Time-bounded shadow watchdog update（2026-08-29）：

```text
CLI:                                  --max-running-seconds FLOAT (operational shadow only)
B-relative start:                     shared RUNNING epoch monotonic timestamp
stop request / receipt reason:        RUN_TIME_LIMIT / "run time limit"
ack failure:                          policy heartbeat grace 后 FAULT
focused timing + preflight suite:     59 passed
full Real offline suite:              191 passed
compileall / black / isort / diff:    PASS
hardware watchdog validation:         H2/H3 PASS；见下方 time-bounded report
```

### H2/H3 初始 reference shadow report（2026-08-29）

在限定授权（reference v2 checkpoint、`--hand`、shadow、B 后只采集/验证、禁止 execute）下，
真机 session 的启动和 B epoch 均成功：XHand serial `012R320220251128022` ready、camera/pointcloud/
inference ready，且 `reset_home` 在 B 前被 hand worker 接受。B 于 `15:16:14` 进入 RUNNING；
`15:18:31` 收到 stop，实际运行 137 s（目标 120 s，超过 17 s；下次需以 B 为起点严格计时）。

```text
execution_mode:                       shadow
coupled_command_start/end sequence:   0 / 0
coupled_command_writes:                0
arm servo_calls:                       0
plans_ingested:                        1365
endpoints_due / shadow_validated:      2184 / 1158
endpoints_motion_discarded:            1026
plan_age p50 / p95:                    29.835 / 60.997 ms
usable_horizon p50 / p95:              568.506 / 641.042 ms
fatal/checker/freshness abort:         none
```

因此 H2/H3 的设备、feedback、ActionBuffer、SafetyGate 和 zero-write shadow contract 均有实测
证据；它不构成 H4 execute 授权。该 session 的 1026 个 motion discard 全部为 hand operational
lower-bound reject，集中在 `j5/j7/j9/j11`，而不是 collision、freshness 或 normalizer 双反归一化。

checkpoint 的 action normalizer 为 float32；由 scale/offset 还原的训练下界相对 runtime float64
operation lower bound 的差值为：j5 `-3.9791546e-08 rad`，j7/j9/j11 各
`-9.9892238e-09 rad`。policy publication boundary 现在只对仍在严格 mechanical/rated envelope
内、且距 operational boundary 不超过 `1e-6 rad` 的 policy endpoint canonicalize 到精确
operational boundary；其它越界仍拒绝，manual/teleop/replay 仍保持 reject-only。receipt 增加
`hand_policy_endpoint_roundoff_canonicalized` 计数，SafetyGate 对真正越界日志输出 target/bound/delta
的 17 位有效数值。

### H2/H3 post-fix reference shadow report（2026-08-29）

同一 frozen reference v2 checkpoint 在 `--hand --execution-mode shadow` 下完成了修复后的 H2/H3
复测。XHand 在 `15:28:31` 于 B 前接受一次 startup `reset_home`；操作员按 B 后，session 于
`15:28:38` 进入 RUNNING。roundoff canonicalization 在 policy publication boundary 生效，所有
due endpoint 都通过 shadow validation；日志中没有
`safety_reject_hand_joint_limit_violation`、`endpoints_motion_discarded` 或
`HAND_PREFLIGHT_REJECTED`。

```text
execution_mode:                       shadow
coupled_command_start/end sequence:   0 / 0
coupled_command_writes:                0
zero_coupled_command_writes:           true
arm servo_calls:                       0
plans_ingested:                        1505
endpoints_due / committed:             2409 / 2409
endpoints_shadow_validated:            2409
endpoints_motion_discarded:            0
hand_limit / preflight rejects:        0 / 0
roundoff canonicalized:                958
plan_age p50 / p95 / p99:              31.686 / 58.895 / 61.357 ms
usable_horizon p50 / p95 / p99:        588.023 / 655.425 / 667.560 ms
fatal/checker/freshness abort:         none
```

该结果验证了 checkpoint float32 反归一化端点与 runtime float64 operational lower bound 的微小
表示差已被正确处理：只修正距 operational boundary `<=1e-6 rad`、同时仍处于 strict mechanical/
rated envelope 内的 learned-policy endpoint；真正越界与 manual/teleop/replay command 不改变原有
reject-only 语义。它继续只构成 H2/H3 zero-write evidence，绝不构成 H4 execute 授权。

本次授权目标为 B 后 120 s，但外置的自动停止监视器未触发，Root 在发现后于 `15:31:09` 发送
`SIGINT`；实际 B→stop 为 151 s，超出授权 31 s。虽然期间没有 coupled write（arm `servo_calls=0`），
该时限偏差必须作为验证流程缺陷记录，不能宣称“严格 120 s”通过。随后已将 watchdog 收回
`run_policy → lifecycle → supervisor`：显式 `--max-running-seconds 120` 从 B epoch 计时，写 typed
`RUN_TIME_LIMIT` request，等待 coordinator 产生 `run time limit` receipt 并回到 ARMED；未在 policy
heartbeat grace 内确认则 FAULT。下方第三次 H2/H3 session 已验证这条内置链路；后续任何 hardware
session 仍必须重新取得限定授权，并使用这一内置 watchdog；不得再依赖 detached helper。

### H2/H3 time-bounded watchdog shadow report（2026-08-29）

在新的限定授权下，以同一 frozen reference v2 checkpoint 执行
`--hand --execution-mode shadow --max-running-seconds 120`。XHand 在 `15:42:53` 于 B 前接受一次
startup `reset_home`；操作员于 `15:43:47` 使系统进入 RUNNING。supervisor 在 `15:45:47` 请求
`RUN_TIME_LIMIT`，实测 B-relative duration 为 `120.040 s`（limit `120.000 s`）；coordinator 同秒撤销
motion、产生 `reason="run time limit"` receipt 并回到 ARMED。随后所有 worker 均 graceful exit，
session 于 `15:45:49` cleanly DISARMED。

```text
execution_mode:                       shadow
checkpoint:                           frozen reference v2 (SHA-256 b174bd…2484c)
coupled_command_start/end sequence:   0 / 0
coupled_command_writes:                0
zero_coupled_command_writes:           true
arm servo_calls:                       0
plans_ingested:                        1197
endpoints_due / committed:             1914 / 1914
endpoints_shadow_validated:            1914
endpoints_coalesced:                   2
hand roundoff canonicalized:           687
hand-limit / motion-discard logs:      0 / 0
plan_age p50 / p95 / p99:              32.807 / 60.663 / 62.324 ms
usable_horizon p50 / p95 / p99:        596.576 / 664.447 / 674.209 ms
supervisor exit / final safety:        run time limit reached / DISARMED
worker exit codes:                     inference, arm, camera, pointcloud, policy, hand = graceful:0
```

这验证的是 B epoch → shared timestamp → typed stop request → coordinator receipt → clean shutdown 的
真机闭环，且整个窗口未产生 policy coupled command write。它消除了上一次 detached helper 的时限
偏差，但仍只构成 H2/H3 zero-write shadow evidence，绝不构成 H4 execute 授权。

H4 的前置 gate、现场授权边界、stop 条件与 receipt 内容见
[`policy_h4_execute_runbook.md`](policy_h4_execute_runbook.md)。H4 的 CLI/lifecycle 已实现一个
严格 single-publication profile（`--hand`、publication bound=1、ACK timeout 与 B-relative
duration 均显式）；不得以这份 runbook、离线测试或 H2/H3 evidence 代替单独的硬件授权。

### Batch 6 — Profile-driven performance

只有 Batch 5 的数据证明具体瓶颈后才实施。

优先顺序：

1. 跳过重复 logical step 的 observation build；
2. checkpoint 单次 deserialize 的剩余路径清理；
3. adapter copy/H2D；
4. pointcloud compute cadence；
5. inference_hz 调整；
6. 最后才是 precision/compile/denoise 实验。

任何影响模型数值的优化都需要 recorded-observation parity 和任务离线评估，不能作为纯
runtime micro-optimization 合并。

---

## 14. 文件级修改地图

### 14.1 `dexmani_real`

| 文件 | 批次 | 责任 |
|---|---:|---|
| `examples/run_policy.py` | 2/5 | thin CLI、纯模式分流、execution acknowledgement |
| `dexmani_real/deployment/artifact.py`（新） | 2 | experiment selector、sidecar validation、run identity |
| `dexmani_real/deployment/config.py` | 2/5 | artifact/operator ownership 合并、execution mode |
| `dexmani_real/deployment/lifecycle.py` | 2/5 | inference-first preflight、hardware startup sequencing |
| `dexmani_real/deployment/contracts.py` | 3 | `PolicyPrediction`、timing-free runtime protocol |
| `dexmani_real/integrations/dexmani_policy.py` | 2/3/6 | 无 FAAS、single load、full future、encode/decode |
| `dexmani_real/deployment/worker.py` | 3/6 | timing stamp、sequence-triggered observation、latency stages |
| `dexmani_real/deployment/action_buffer.py`（新） | 4 | 纯 rolling scheduler |
| `dexmani_real/deployment/coordinator.py` | 4/5 | buffer consumer、typed disposition、shadow dispatch |
| `dexmani_real/control/safety_gate.py` | 4 | reject code split、canonical minimal gate |
| `dexmani_real/control/publication.py` | 4/5 | typed result、shadow final boundary |
| `dexmani_real/deployment/metrics.py` | 4/5 | bounded percentiles 与新 counter |
| `dexmani_real/robot/hand_worker.py` | 0/5 | reset-home contract verification，保留 measured ramp |
| `dexmani_real/ipc/schema.py` | 3 | 仅在 contract 需要时调整字段；capacity 保持 32 |
| `tests/test_run_policy.py`（新） | 2/5 | CLI/mode/import side-effect |
| `tests/test_policy_artifact.py`（新） | 2 | selector/index/path/hash |
| `tests/test_action_buffer.py`（新） | 4 | pure scheduler cases |
| `tests/test_deployment_timing.py` | 0/3/4 | timeline/full future/generation |
| `tests/test_safety_gate_command_delta.py` | 0/4 | 20° arm、无 coordinator hand delta |
| `tests/test_coupled_command_publication.py` | 0/4/5 | typed result/shadow/no-write |
| `tests/test_worker_command_validation.py` | 0/4/5 | worker 20°与 hand ramp/reset |

若新增/删除 tracked file，按仓库约定更新 `repo_map.md`。用户工作流稳定后再更新 README；
不要在实现尚未放行时提前宣称 execute 可用。

### 14.2 `dexmani_policy`

| 文件/区域 | 批次 | 责任 |
|---|---:|---|
| `datasets/pc_dataset.py` | 1 | Real contract、pad_before=1 |
| `datasets/replay_buffer.py` | 1 | 保留/验证 Real Zarr root attrs |
| `datasets/sampler.py` | 1 | repeat-edge semantics 测试 |
| `common/checkpoint_io.py` | 1 | 完整 checkpoint metadata、loaded-object restore |
| `training/workspace.py` | 1 | atomic artifact/index/selector publication |
| `training/eval_utils.py` | 1 | EMA fail-closed、避免二次 load |
| deployment exporter（新薄入口） | 1 | legacy training checkpoint → deployment-only v2 |
| policy tests | 1 | data/checkpoint/index/export round trip |

具体修改前仍须以当时源码 owner 为准；不要为了匹配这张表移动无关文件。

---

## 15. 测试矩阵

### 15.1 Artifact/config

| Case | Expected |
|---|---|
| valid deployment DP3 artifact | accept |
| no selector | reject before lifecycle |
| dangling `deployment_latest.pt` | reject; no latest fallback |
| selector escapes checkpoints dir | reject |
| missing/malformed sidecar | reject |
| size/hash/filename mismatch | reject |
| embedded/index contract mismatch | reject |
| `required_action_steps > 32` | reject |
| missing inference/data/train contract | reject |
| Real `pad_before != 1` | reject |
| padding policy not repeat-edge | reject |
| task/action/dt/pointcloud mismatch | reject |
| `use_ema=true` but EMA missing | reject |
| missing normalizer | reject |
| FAAS field/import path | absent, not accepted as variant |

### 15.2 CLI/lifecycle

| Case | Expected |
|---|---|
| print-config | no torch load/worker/hardware import |
| preflight-only | inference only, then clean exit |
| shadow default | explicit log of reset-home hardware effect |
| YAML requests execute without CLI acknowledgement | reject |
| artifact commands hand without `--hand` | reject before hardware |
| inference load failure | hardware workers never start |
| hand reset rejected | no ready/ARMED/B |
| reset CRC unconfirmed + valid feedback | warn, may become ready |
| reset measured qpos outside home tolerance | no tolerance check; may become ready |
| pre-B feedback | excluded from observation epoch |
| ARMED | no model call |
| shadow after B | coupled ring sequence unchanged |

### 15.3 Prediction/timing

| Case | Expected |
|---|---|
| DP3 output | 15×19 finite prediction |
| first endpoints expired | mask only |
| all endpoints expired | drop plan |
| inference slower than period | no retime |
| repeated logical step | no observation rebuild/model call |
| previous generation prediction | drop |
| nonfinite/wrong shape | fatal |

### 15.4 ActionBuffer

| Case | Expected |
|---|---|
| old future only | use old valid endpoint |
| overlapping fresh plan | fresh observation wins |
| multiple overdue endpoints | select latest due, count coalesced |
| no due but future exists | FUTURE |
| no valid future | EXHAUSTED |
| commit token twice | reject invariant violation |
| discard then older same-target candidate exists | no fallback/no publish |
| generation changes | reset all plans/tokens |
| plan deadline closes | prune |
| buffer capacity reached | deterministic oldest/stale prune; no unbounded growth |

### 15.5 Safety/publication

| Case | Expected |
|---|---|
| 10° arm target otherwise valid | accept |
| >20° arm jump | motion discard, no publish |
| large in-range hand target | coordinator accept; worker ramps |
| joint/workspace/collision violation | motion discard |
| collision checker throws | fatal abort |
| NaN/shape/contract error | fatal abort |
| transient stale feedback | no finalize, watchdog remains active |
| execute publish success | commit exactly once |
| shadow validation success | no ring write, commit exactly once |

---

## 16. 验证命令与安全限制

普通代码阶段只运行离线检查：

```bash
git status --short
python -m compileall -q dexmani_real examples tests
python -m pytest -q tests/test_deployment_timing.py
python -m pytest -q tests/test_worker_command_validation.py
python -m pytest -q tests/test_coupled_command_publication.py
python -m pytest -q tests/test_deployment_manifest.py
python -m pytest -q tests/test_policy_artifact.py
python -m pytest -q tests/test_run_policy.py
python -m pytest -q tests/test_action_buffer.py
python -m pytest -q
git diff --check
git diff --stat
```

不存在的新增测试只在对应 batch 落地后加入命令。

以下命令在 Batch 1/2 完成后才有意义，且仍不连接硬件：

```bash
conda run -n real_robot python examples/run_policy.py \
  --experiment-dir \
  /home/zhanghaoyang/Desktop/dexmani_policy/experiments/dp3/pick_place_toy/2026-08-28_13-59_42 \
  --print-config

conda run -n real_robot python examples/run_policy.py \
  --experiment-dir \
  /home/zhanghaoyang/Desktop/dexmani_policy/experiments/dp3/pick_place_toy/2026-08-28_13-59_42 \
  --preflight-only
```

不得把 preflight 报告为 hardware validation。

所有 `examples/` 入口在检查前均视为可能影响硬件。没有用户针对该次验证的明确授权，
不得运行 shadow/execute、连接设备或下发 reset/home/servo command。

---

## 17. Physical validation ladder

代码完成不等于允许 full rollout。实际放行顺序固定为：

```text
H0 offline artifact/preflight
H1 recorded observation replay
H2 real sensor shadow inference
H3 shadow + ActionBuffer + IK + SafetyGate
H4 one short full-coupled execute in clear free space
H5 repeated arm + hand no-object motion
H6 simple large-object task
H7 full dexterous task
```

每一级都要求：

- 用户明确授权本次硬件动作；
- 保存 run identity 和 metrics；
- STOP/e-stop 可用；
- 上一级没有未解释 fatal/checker error；
- review 后再进入下一级。

DP3 joint 是第一批 physical model。其他 agent 或 EE policy 不与第一次真机闭环同时引入。

---

## 18. Definition of Done

### Artifact

- Real Zarr → dexmani_policy → checkpoint/index → dexmani_real 闭环；
- `pad_before=1/repeat-edge` 被内嵌并严格校验；
- 当前 step-80000 latest 已生成 immutable deployment export；
- checkpoint self-describing，sidecar 只做 allocation projection；
- EMA、normalizer、contract、hash 全部 fail closed；
- Real 和 Policy 都不存在 FAAS deployment branch。

### Entry/lifecycle

- `examples/run_policy.py --experiment-dir` 是唯一操作者入口；
- print/preflight 不产生 hardware side effect；
- 正式 lifecycle 只加载一次模型，并在 hardware 前 ready；
- hand reset_home 在 B 前只下发一次，不做位置容差判定；
- B 后才开始 observation/inference/action epoch。

### Runtime/control

- PolicyRuntime 只产生无物理时间的 proposal；
- worker 依据 immutable observation grid stamp 15-step future；
- ActionBuffer 完全取代 active/pending；
- fresh plan 可接管未执行未来，旧合法 future 可短时 fallback；
- 每 tick 最多一个 endpoint；
- expired 不 retime，commit/discard 不重复。

### Safety

- learned gate 不再使用 arm 8° / hand 0.3 reject；
- canonical arm 20°、joint bounds、workspace、19-DoF collision 保留；
- hand worker 0.3 measured-state ramp 保留；
- motion reject discard，fatal contract/checker error abort；
- safety reject 后不 fallback 到旧同-target candidate；
- shadow B 后 coupled command write 为零。

### Evidence

- targeted/full offline tests 通过；
- reference checkpoint preflight 通过；
- shadow report 含 p50/p95/p99、usable horizon、discard reason 和 command-write proof；
- 未经真实硬件执行不得声称 hardware validation；
- README/repo_map 与最终实际支持状态一致。

---

## 19. 实施时的停止条件

遇到以下任一情况，停止当前 batch，不继续扩大修改：

- source checkpoint/data/config 三方 contract 无法一致证明；
- exporter 必须猜测缺失的 action/frame/unit 语义；
- 新实现同时保留 active/pending 和 ActionBuffer；
- print/preflight 路径不可证明无 hardware import side effect；
- full future 与 Policy slicing 数值不一致；
- SafetyGate 无法区分 motion collision 与 checker exception；
- shadow 出现 B 后 coupled command write；
- optimization 改变模型数值但没有 parity/eval 证据；
- 工作区存在无法安全避让的用户修改。

---

## 20. 最终架构

```text
┌──────────────── dexmani_policy ────────────────┐
│ Real Zarr v5                                   │
│   ↓ pad_before=1 contract                      │
│ Train → model/EMA/normalizer                   │
│   ↓                                            │
│ self-describing checkpoint + hash-bound index  │
│   ↓                                            │
│ deployment_latest.pt                           │
└──────────────────────┬─────────────────────────┘
                       │ experiment-dir
                       ▼
              examples/run_policy.py
                       │
          resolve → fixed artifact receipt
                       │
                       ▼
┌──────────── dexmani_real / inference ───────────┐
│ checked no-follow fd → hash → provenance → load  │
│ post-B causal ObservationBatch                  │
│   ↓                                             │
│ DexManiPolicyRuntime                            │
│   ↓ full 15-step PolicyPrediction               │
│ worker stamps immutable logical-grid timing     │
│   ↓ mask expired, never retime                  │
│ policy_plan_ring                                │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌───────────── dexmani_real / control ────────────┐
│ ActionBuffer latest-wins                        │
│   ↓ one due endpoint/tick                       │
│ EE→IK when required                             │
│   ↓                                             │
│ minimal SafetyGate + publication validation     │
│   ├─ motion reject → discard                    │
│   ├─ fatal checker/contract → abort              │
│   ├─ shadow → no write, commit                  │
│   └─ execute → coupled publish, commit          │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
       arm worker 20° final check + hand 0.3 ramp
```

设计原则：

> Policy 负责预测未来动作；Real Runtime 负责证明这些动作在当前 observation 时间轴、
> lifecycle 和物理安全边界下仍然有效，并且每个 control tick 只发布一个最终动作。

---

## 21. Terra xhigh Agent 实施与验证 Workflow

### 21.1 总体编排

所有后续 spawned agent 固定使用：

```text
model: gpt-5.6-terra
reasoning_effort: xhigh
```

采用“主 agent 编排、批内单写者、双重只读 review、独立验证”的模式：

```text
Root Orchestrator
    ↓ freeze Batch scope / file ownership / Gate
Implementation Agent（唯一写者）
    ↓ focused diff frozen
Architecture/Contract Reviewer ─┐
Safety/Validation Reviewer ─────┤ parallel, read-only
                                ↓
Root triage
    ↓ CHANGES_REQUIRED
原 Implementation Agent 修复
    ↓ reviewers re-check
Independent Validator
    ↓ exact offline commands + evidence receipt
Root Gate decision
    ↓ PASS only
next Batch
```

审查 agent 永不直接修代码。发现问题统一交回原 implementation agent，避免第二个写者
根据不同理解重做边界。

### 21.2 四槽并发使用

系统最多四个并发槽，包括 Root。推荐使用方式：

#### Implementation wave

```text
slot 1  Root Orchestrator
slot 2  当前 Batch Implementation Agent（唯一写者）
slot 3  可选只读 source scout
slot 4  可选只读 test/contract scout
```

只在任务确实可独立时启动 scout；不要为了占满并发而增加 agent。

#### Review wave

```text
slot 1  Root Orchestrator
slot 2  Architecture/Contract Reviewer
slot 3  Safety/Validation Reviewer
slot 4  batch-specific Reviewer（仅 artifact、ActionBuffer、shadow 等高风险批次）
```

#### Fix/validation wave

```text
原 Implementation Agent follow-up 修复
→ reviewers follow-up 复审
→ Independent Validator 单独执行测试
```

pytest、compileall 和 checkpoint preflight 不与 production 写入并行，避免共享工作区、
cache、artifact 或日志造成不确定状态。

### 21.3 写入与分支规则

- 同一时刻整个任务最多一个 production 写 agent；
- 不允许一个 agent 同时修改两个仓库；
- 每个 agent 开始前运行 `git status --short`，保留用户未提交内容；
- 当前方案文档是未跟踪文件，任何 agent 都不得自动 add/commit/delete；
- 是否创建 branch/commit 由用户授权决定，agent 不自行提交；
- 若用户允许 commit，每个可独立回滚单元一个 commit，不 squash 跨仓证据；
- `dexmani_policy` 当前不在本 workspace writable root 内，Batch 1 写入前由 Root 获取明确
  文件系统授权；不能让 agent 用复制仓库或临时脚本绕过权限；
- P1b 先用 synthetic/minimal fixture 在获准目录完成 exporter 单测；向 reference experiment
  写入 deployment artifact、sidecar 或 selector 属于单独的 material write，Root 必须在
  展示精确目标路径和新增/替换项后获得用户授权；
- reviewer、validator 默认只读 production files；测试产生的普通 cache 不构成代码所有权；
- 任一 agent 发现与本 Batch 重叠的未知用户修改，立即停止并交回 Root。

### 21.4 跨仓依赖 DAG

跨仓接口只能由 Policy producer 先冻结，Real consumer 后实现。禁止两个仓库并行猜合同。

```text
R0   Real regression baseline
 ↓
P1a  Policy canonical contract + Real data proof
 ↓
P1b  legacy exporter + reference deployment artifact
 ↓
R2a  Real pure artifact resolver
 ↓
R2b  Real CLI + inference-only preflight + single load
 ↓
R3   PolicyPrediction + 15-step future + timing owner
 ↓
R4   ActionBuffer + typed safety disposition
 ↓
R5   shadow + metrics + readiness evidence
 ↓
R6   one profile-proven optimization at a time
```

其中：

- `P1a/P1b` 分开，避免 schema 设计与 1.1 GB migration 问题混在同一 review；
- `R2a/R2b` 分开，先验证不可信 filesystem/index 边界，再接 CLI、spawn 和模型；
- `R4` 的纯 ActionBuffer 可以先在工作分支完成测试，但必须与 coordinator 的一次性切换
  同批交付；不允许合入新旧 scheduler 并行接入状态；
- `R6` 没有 Batch 5 profile 数据时不创建 implementation agent。

### 21.5 Agent roster

| Agent | 仓库 | 写权限 | 任务 | 专项 reviewer |
|---|---|---:|---|---|
| `r0_baseline_impl` | Real | tests only | 锁定 timing/generation/publication/reset-home | `r0_invariant_review` |
| `p1a_contract_impl` | Policy | yes | canonical contract、Real Zarr、padding、checkpoint metadata | `p1_contract_review` |
| `p1b_export_impl` | Policy | yes | atomic exporter、sidecar、selector、reference migration | `p1_artifact_review` |
| `r2a_resolver_impl` | Real | yes | pure selector/path/index/capacity validation | `r2_filesystem_review` |
| `r2b_preflight_impl` | Real | yes | thin CLI、lazy import、inference-only preflight、去 FAAS | `r2_side_effect_review` |
| `r3_prediction_impl` | Real | yes | PolicyPrediction、15-step slice、worker timing | `r3_numerical_review` |
| `r4_action_buffer_impl` | Real | yes | rolling scheduler、typed disposition、minimal gate | `r4_scheduler_review` + `r4_safety_review` |
| `r5_shadow_impl` | Real | yes | shadow publication、metrics、readiness | `r5_no_write_review` + `r5_lifecycle_review` |
| `r6_perf_impl` | owning repo | yes | 单个量化瓶颈 | `r6_parity_review` |
| `batch_validator` | current repo | no production edit | 独立复跑 Gate、生成 evidence receipt | Root |

每个名字表示一个 bounded task，不表示常驻 agent。一个 agent 完成后释放槽位；需要修复时
优先 `followup_task` 给原 agent，不创建竞争实现者。

### 21.6 每个 Batch 的状态机

```text
PREPARE
→ IMPLEMENTING
→ DIFF_FROZEN
→ REVIEWING
→ CHANGES_REQUIRED → IMPLEMENTING
→ REVIEW_PASS
→ VALIDATING
→ GATE_PASS / GATE_FAIL / BLOCKED
```

只有 Root 可以宣布 `GATE_PASS` 并派发下一批。

#### PREPARE

Root 提供：

- 精确 Batch 和目标仓库；
- 允许修改的文件/目录；
- 必读入口、边界两侧和测试；
- 本文对应 Gate；
- 已知用户修改；
- 禁止运行的 hardware/example 命令；
- 上游 evidence receipt。

#### DIFF_FROZEN

Implementation agent 停止编辑并交付 focused diff。review 开始后，除非 Root 返回
`CHANGES_REQUIRED`，implementation agent 不继续“顺手优化”。

#### REVIEWING

两个 reviewer 并行但分工固定：

```text
Architecture/Contract:
    ownership, dependency direction, schema/config, duplicated logic,
    current-source consistency, scope

Safety/Validation:
    generation, freshness, ticket, collision/workspace, publication,
    fail-closed, negative tests, hardware side effects
```

#### VALIDATING

Independent Validator 从干净的 frozen diff 复跑：

```text
focused tests
→ relevant integration tests
→ full pytest
→ compileall
→ diff --check/stat/status
```

实现 agent 自己跑过的测试不能替代独立验证。

### 21.7 通用 implementation prompt contract

Root 派发时使用以下模板：

```text
你是 DexMani <Real|Policy> <Batch ID> implementation agent。
模型固定 gpt-5.6-terra，reasoning_effort=xhigh。

只完成：<精确目标>。
目标仓库：<repo>。
允许修改：<file allowlist>。
禁止修改：<explicit exclusions>。

开始前必须：
1. git status --short，保留无关修改；
2. 完整阅读适用 AGENTS.md；
3. 阅读计划的 <sections>；
4. 按 definition→producer→transformation→consumer→side effect 跟踪本批值；
5. 检查本批跨越边界的两侧。

硬约束：
- 不运行硬件、shadow/execute、reset/home/servo；
- 不弱化 generation/freshness/ticket/collision/workspace/worker validation；
- 不猜测缺失 contract，不增加 silent fallback；
- 不重构 allowlist 外代码；
- 最小完整垂直修改；
- 若计划与源码冲突，提供文件:行号证据并停止；
- 新增/删除 tracked file 时检查 repo_map；README 只反映已支持流程；
- 完成后停止编辑，提交 handoff，不 commit，除非 Root 明确授权。

本批 Gate：
<粘贴完整 Gate 和测试矩阵条目>
```

### 21.8 通用 reviewer prompt contract

```text
你是 <Batch ID> independent reviewer。
模型固定 gpt-5.6-terra，reasoning_effort=xhigh。
只读；绝不编辑文件。

审查 base→working diff 和边界两侧源码。按严重度输出 findings，必须带文件:行号和
可复现证据。检查：
- 本 Batch Gate 是否逐项成立；
- 是否与当前源码事实一致；
- contract/schema/config 是否只有一个 owner；
- generation/freshness/ticket/publication 是否回归；
- fail-closed 和 negative tests 是否充分；
- 是否存在 silent fallback、重复 abstraction、越界修改；
- 实现 agent 声称的测试是否真的覆盖风险。

结论只能是 PASS / CHANGES_REQUIRED / BLOCKED。
没有 finding 时明确写“未发现阻断项”，不要为了产出而虚构问题。
```

### 21.9 Handoff 与 evidence receipt

Implementation handoff：

```text
Batch / target repo / base commit:
范围与明确未做项:
修改文件及责任变化:
保持的安全不变量:
运行的离线测试及结果:
未运行项，尤其 hardware:
计划/源码冲突与风险:
git status + focused diff summary:
需要 reviewer 逐项回答的问题:
```

Reviewer handoff：

```text
审查对象 base/diff identity:
结论: PASS / CHANGES_REQUIRED / BLOCKED
findings by severity, file:line:
Gate evidence:
safety/contract/side-effect review:
缺失测试或复现命令:
允许 Root 通过的条件:
```

Validator receipt：

```text
Batch / repo commit or base+diff identity:
environment identity:
commands and exit codes:
focused/full test counts:
artifact/run hashes when applicable:
git status / diff --check / diff stat:
explicitly not validated:
verdict: PASS / FAIL / BLOCKED
```

P1b 跨仓交接必须额外记录：

```text
policy commit/base+diff identity
source checkpoint lstat identity + SHA-256
deployment checkpoint SHA-256
embedded contract SHA-256
sidecar canonical SHA-256
model/EMA/normalizer tensor key+shape digest
deployment_latest selector target
metadata_provenance=retrofitted
```

R2 之后不得重新从 experiment config 推导这些值；只消费 frozen receipt 和 artifact。

历史 v1 reference receipt（2026-08-28；保留作迁移证据，restricted loader 不接受）：

```text
policy HEAD:
  7e31d10e7a31ff3d12df31b8683c9c90b357cbc5
policy tracked diff SHA-256:
  bfccb6de0e088046a08eca78b1c869035385f3e38a8e64f92aac56e766b5c191
source checkpoint:
  epoch=1126-step=00080000-milestone=100pct.pt
  size=1100542334
  sha256=0e5615cc3be4e5299791aae24c412df3667027b06ade6cad266be48e50150e84
deployment checkpoint:
  epoch=1126-step=00080000-deployment-v1.pt
  size=1100517666
  sha256=d15a447a9584dc511f3c9554011af9841b21808e74077eb174cb968a70c29e9b
embedded contract sha256:
  ab0d058b8876336daeb250de0eb8a8d1e0c4de84782f49e000d00ddd9e58466b
sidecar canonical sha256:
  060ff47663be75b4cf27796d4ceff1fc91d17594c46672a53d3dfc4f9145abee
resolved / inference / data contract sha256:
  ebece14da8264455afcae2e32773a16aed5ba82c30f30e165f8a5b00edb2db86
  f72d2614811d1034226e2932fbb2bc73ef5be2f824eba59ccceda17cde0c404e
  ded880bd3c8a1994b6bc766efc07956ed99b6ec96e50a5b69cd478cf30e13f61
selector:
  deployment_latest.pt -> epoch=1126-step=00080000-deployment-v1.pt
metadata_provenance=retrofitted
retrofitted_train_params_fields=[use_aux_ee]
```

独立 `DexManiP1bTensorDigestV1` receipt：

```text
model:            tensors=178 numel=68763243  sha256=631403981e0e56e7b3a21de75663960678740c224a475f473ddb2651d39f6a28
EMA:              tensors=178 numel=68763243  sha256=085a40f7419a42d1ea2b49bab5d03e2c6e25799e7105675825c7a801c10d4892
model normalizer: tensors=6   numel=88        sha256=6b4b5ccde4ee5fdf54eadd052dccb6c0bf70b14c16590a247b53c410fef1be06
EMA normalizer:   tensors=6   numel=88        sha256=1afd853fa6a5ec12a36316e8aec65d1b801ac9b6b293c8b1508fbd56d4e276f7
optimizer:        tensors=516 numel=137526482 sha256=5cbbf7f83aa5beec0de9c4ebe8df5786561daa0978d5d41c0f6d1fa26dbc36c7
scheduler:        tensors=0   numel=0         sha256=a3c3ca953ebce96daeadfc4aa32f67820be0171b36c66fd83b09dcfe1ea8286b
```

这组 digest 描述完整 source-training state；其中 optimizer/scheduler 只属于 source
integrity evidence，不进入 deployment v2。历史 validator 与发布者的 digest framing 不同，
因此旧 digest 不与下面的 v2 内容 digest 逐字比较；model/EMA tensor 数量与 numel 一致。

当前 deployment-only v2 receipt（2026-08-29）：

```text
source checkpoint:
  epoch=1126-step=00080000-milestone=100pct.pt
  size=1100542334
  sha256=0e5615cc3be4e5299791aae24c412df3667027b06ade6cad266be48e50150e84
historical v1 (unchanged):
  checkpoint sha256=d15a447a9584dc511f3c9554011af9841b21808e74077eb174cb968a70c29e9b
  sidecar sha256=060ff47663be75b4cf27796d4ceff1fc91d17594c46672a53d3dfc4f9145abee
deployment v2:
  epoch=1126-step=00080000-deployment-v2.pt
  format=dexmani.deployment.v2
  size=550236074
  sha256=b174bd483b64090cd3f5dbe0a5bfadd10998f5d27d43fc9aca06efb82242484c
  sidecar size=798
  sidecar sha256=d67bf21394c8d79239dd0f116baf25a681c3e4d0964569ce47a520f62e047c6e
  embedded contract sha256=ab0d058b8876336daeb250de0eb8a8d1e0c4de84782f49e000d00ddd9e58466b
selector:
  latest.pt -> epoch=1126-step=00080000-milestone=100pct.pt
  deployment_latest.pt -> epoch=1126-step=00080000-deployment-v2.pt
producer:
  repository=haoyangzhanglab/dexmani_policy
  commit=7e31d10e7a31ff3d12df31b8683c9c90b357cbc5
  metadata_provenance=retrofitted
  retrofitted_train_params_fields=[use_aux_ee]
```

独立 `DexManiDeploymentTensorDigestV1` receipt：

```text
model:            tensors=178 numel=68763243 sha256=f47832ffc871ca7af288321f79ecbb49c145024812d441d2f6ae6a0908f16d64
EMA:              tensors=178 numel=68763243 sha256=7d397a800e5506f9469fa5b5d81bc9869b1332496d07c3f0f7a09ef5450ad45f
model normalizer: tensors=6   numel=88       sha256=31483a6ede79e4d6877dfa1fed6496e3e2cbc59cdadd45a561c6707b7952dbec
EMA normalizer:   tensors=6   numel=88       sha256=31483a6ede79e4d6877dfa1fed6496e3e2cbc59cdadd45a561c6707b7952dbec
```

digest 以 `DexManiDeploymentTensorDigestV1\0` 初始化；按 tensor key 排序，依次加入
length-prefixed key、dtype、shape 和 contiguous CPU raw bytes。v2 static pickle audit 只出现
`collections.OrderedDict`、`torch.FloatStorage` 与 `torch._utils._rebuild_tensor_v2`，没有
OmegaConf/custom globals；restricted material load 为单次 `weights_only=True`，fd 4→4。
training-only monitor/optimizer/scheduler 均为空，`pad_before=1`。CPU 与非 sandbox CUDA
preflight 都验证实际/expected checkpoint SHA 一致；Batch 2 当时输出 8×19，Batch 3
切换 full actionable future 后重新验证为 15×19。

R2a Real resolver 历史 receipt（2026-08-29，发生在 v2 selector 切换前）：

```text
full offline tests:    124 passed, 58 subtests
focused resolver:      15 passed, 40 subtests
reference checkpoint:  epoch=1126-step=00080000-deployment-v1.pt
reference size:        1100517666
reference index sha:   060ff47663be75b4cf27796d4ceff1fc91d17594c46672a53d3dfc4f9145abee
checkpoint content IO: openat(O_NOFOLLOW) + close only; no read/hash/load
sidecar content IO:    799 bytes
fd stability:          4 -> 4 after 128 resolves
review / validation:   contract PASS; filesystem PASS; independent PASS
hardware:              not run
```

### 21.10 Batch-specific review focus

- `P1a`：所有字段都有显式数据来源；`pad_before=1/repeat-edge`；没有 FAAS；
- `P1b`：源 checkpoint 不变；checkpoint→sidecar→selector 原子顺序；重复 export contract
  hash 稳定；
- `R2a`：不导入 Policy/torch/lifecycle；dangling/out-of-tree/mismatch 全拒绝；
- `R2b`：print 不 hash/load；preflight 无 hardware spec/import；single deserialize；失败无
  selector fallback；
- `R3`：15×19 parity；logical grid immutable；expired 只 mask；
- `R4`：active/pending 已删除；watermark exactly-once；motion discard；checker abort；无旧
  same-target fallback；
- `R5`：shadow 复用完整 validation；B 后 policy coupled write 为零；reset-home 一次且在 B 前；
- `R6`：存在 Batch 5 profile 证据；一次只改一个瓶颈；数值变化有 replay parity/eval。

### 21.11 验证授权边界

```text
H0 artifact/resolver/preflight     offline
H1 recorded observation replay    offline
H2 real sensor/feedback shadow    hardware authorization required
H3 shadow + IK/SafetyGate         hardware authorization required
H4–H7 execute                     per-level explicit authorization required
```

R2b 已落地并通过 import-sentinel、受限单次加载与 CPU/CUDA H0 预检。R5 已在代码中开放
`examples/run_policy.py --execution-mode shadow`，但它仍是 hardware-affecting 入口：只可在取得
H2/H3 限定授权后启动；hand-enabled artifact 还必须显式传入 `--hand`。`--print-config` 与
`--preflight-only` 继续是无 hardware 的入口。H4 execute 仅开放 immutable one-shot profile，
仍不能据此越过单次 H4 authorization、场地确认或后续 physical gate。

H2–H7 每次授权必须限定：

- checkpoint/index/config hash；
- execution mode；
- physical validation level；
- 场地和对象状态；
- 预计持续时间；
- STOP/e-stop/operator readiness。

agent 可以启动已获授权的进程、保存日志和离线分析，但不能代替操作员按 B、选择升级
level 或判断场地安全。shadow 也会连接设备并在 B 前下发 XHand reset-home，因此不是
无硬件模式。

出现以下任一情况立即停止当前 level，保存证据，禁止自动升级：

- B 后 shadow coupled write 非零；
- hardware fault、freshness/watchdog、checker exception；
- STOP/e-stop；
- reference identity 与授权值不一致；
- H4 首次 execute 出现 motion reject；
- 连续或无法解释的 safety rejection。

### 21.12 Root 的下一步

R0–R4 均已完成并通过各自 gate。R4 已删除 coordinator 的 `active/pending` 双状态，以
`ActionBuffer` 的 logical target latest-wins、commit/discard watermark 和 generation reset
实现有界且恰好一次的 endpoint 仲裁；typed disposition 也已区分 motion discard、stale
discard、transient defer 与 fatal abort。该结果经过 scheduler/lifecycle 与 safety/IK 两路
独立 review，且全量离线验证通过。

R5 的离线实现已完成：immutable `shadow` mode 完整通过 model/IK/SafetyGate/generation/
freshness validation，但结构上不能写 policy coupled command；每个 B epoch 都记录 bounded
timing/horizon 指标与 start/end ring sequence 的 canonical receipt。hand-enabled shadow 必须
在创建 IPC 或进程前显式 `--hand`，startup reset-home 只在 B 前下发一次且没有 home tolerance
wait；为维持 B 后 zero-write contract，shadow 禁用 H/home。H4 execute 同样禁用 operator H/home，
并以 coordinator 的 one-publication + dual-worker ACK boundary 管理唯一 physical command。

2026-08-29 的首次 H2/H3 shadow 启动尝试在 inference-only gate 被安全拦截：worker 调用了
已禁用的 path-based `DexManiPolicyRuntime.load()`，故 inference 子进程退出；lifecycle 随即关闭
RuntimeChannels，未启动 arm/hand/camera/pointcloud/policy worker，未连接设备、未下发
reset-home、未进入 B epoch，也没有 shadow receipt。它不是 hardware validation。

该加载断链现已修复：artifact-bound inference worker 与 `--preflight-only` 共享固定 entry 的
no-follow fd、identity、SHA-256、package provenance、单次 stream deserialize 与 restore 链路；
worker 不再调用 path-based `load()`。离线 host-CUDA H0 已用下方 frozen v2 artifact 通过（实际
SHA 与 index SHA 均为 `b174bd…2484c`，15×19 fake prediction 通过）。

随后已在同一 reference identity 上完成初始、post-fix 和 time-bounded 三次限定 H2/H3 shadow，并取得上文
receipt：三次均在 B 前 `reset_home` accepted、B 后 `coupled_command_writes=0`、arm
`servo_calls=0`。初始 session 的 1026 个 hand lower-bound reject 已归因于 checkpoint float32
normalizer 到 runtime float64 boundary 的微小表示差，不是归一化/反归一化链路重复或单位错误；
post-fix session 则以 958 次 `hand_policy_endpoint_roundoff_canonicalized` 完成 2409/2409 个
endpoint shadow validation，hand-limit discard 为 0。该 session 的 detached B-relative stop watchdog
失效，实际运行 151 s（授权目标 120 s）；内置 typed watchdog 随后在新的限定 H2/H3 session 中实测为
B 后 `120.040 s` 正常 receipt、零 coupled write 且全部 worker clean exit。不得据此启动 H4 或推断
execute 已放行。

已冻结的 reference 路径为：

```text
experiments/dp3/pick_place_toy/2026-08-28_13-59_42/checkpoints/
  epoch=1126-step=00080000-deployment-v2.pt
  epoch=1126-step=00080000-deployment-v2.pt.deployment.json
  deployment_latest.pt -> epoch=1126-step=00080000-deployment-v2.pt
```

源 `epoch=1126-step=00080000-milestone=100pct.pt`、历史 v1 产物及训练 selector
`latest.pt` 保持只读、不变。下游 agent 不得绕过 frozen v2 receipt 自行解释 experiment
config。R5 的离线实现不可运行真实 hardware；只有同时取得本节 H2/H3 所列限定授权后，
才允许用该 reference identity 产生一次 shadow report。H4 implementation 已完成离线验证，
但一次 H4 真机运行仍须满足专属 runbook，并重新取得明确的 one-shot execute 授权。

### H4 software closeout（2026-08-29，无硬件）

本轮只做软件边界收口，没有启动 `run_policy` operational lifecycle，也没有连接或驱动任何
设备。新增/复核的 H4 不变量如下：

```text
execute profile:                     --hand + bound=1 + ACK timeout + B-relative limit
publication invariant:               coupled ring sequence 从 B 基线只增加 1
candidate provenance:                ACK/previous-command 使用实际发布的 canonical candidate
pending ACK + time limit:             未完成双 worker ACK 直接 sticky FAULT，不正常收尾
extra/missing ring write:             运行中拒绝并 sticky FAULT
```

离线验证结果：

```text
full Real offline suite:             205 tests, OK
focused H4 timing/preflight/IPC:     OK
compileall (dexmani_real/examples/tests): PASS
focused black / isort:                PASS
git diff --check:                     PASS
mypy focused H4 files:                无本地 H4 类型错误；环境缺少 scipy/pinocchio/hppfcl/yaml stubs
```

同一 frozen reference v2 experiment 的无硬件检查也重新通过：

```text
checkpoint:                           epoch=1126-step=00080000-deployment-v2.pt
checkpoint SHA-256:                   b174bd483b64090cd3f5dbe0a5bfadd10998f5d27d43fc9aca06efb82242484c
action_dim / required_action_steps:   19 / 15
pad_before / pad_after:               1 / 7
preflight:                            passed
H4 bounds:                            publication=1, ACK=2.0 s, B-relative=30.0 s
hardware lifecycle:                   not run
```

因此当前状态是“execute 软件路径具备 one-shot fail-closed guard，H4 真机仍未授权”。下一步只能
在新的、单独限定的 H4 授权下运行一次现场实验；授权前继续使用 `--print-config` 或
`--preflight-only`，不得以本节离线结果替代物理安全案例或操作员确认。
