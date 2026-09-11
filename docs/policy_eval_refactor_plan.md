# DexMani Policy Deployment / Evaluation 重构执行方案

> 状态：**实施计划 / Claude Code 执行依据**  
> 事实核查基线：`dexmani_real/main` @ `3ad3019383cc7cad79af95b2d984af67b8d01408`（2026-09-11）  
> 关联仓库：`haoyangzhanglab/dexmani_policy`  
> Hardware validation：**NOT RUN**

本文定义 DexMani 真实机器人 Policy deployment / evaluation 的目标形态、职责边界、跨仓库依赖、修改顺序、测试要求与禁止事项。实施时以本文为主，不要从历史 refactor plan、旧 README 示例或已 superseded 文档反推目标行为。

目标不是建设 production deployment platform，而是为博士研究中的真实机器人论文实验提供一个：

- 简洁、直观、低操作负担的唯一入口；
- 安全、因果、可复现的真机执行边界；
- 支持连续多 episode 的 persistent evaluation session；
- 与 teleop 共用 canonical raw episode 的数据记录方式；
- 足以复现实验条件、但不过度生产工程化的 session metadata；
- 不破坏当前已经建立的 Point Cloud / Fingertip / Tactile scientific contract。

---

## 1. 设计原则

发生取舍时按以下顺序判断：

```text
真实机器人安全与论文结果正确
→ train / deploy scientific semantics 一致
→ 主数据流清晰
→ 真机实验操作简单
→ 数据可复查、可复现
→ 通用扩展性
```

核心原则：

1. **删 workflow 管理，不删 scientific correctness。**
   Git provenance、checkpoint SHA-256、online task outcome 属于本项目不需要的 production 管理；Point Cloud numeric preprocessing、Fingertip mount geometry、Tactile calibration/unit identity 会改变模型实际看到的 observation，必须保留。
2. **一个真机 Policy 入口。**
   `run_policy.py` 的用户职责只有“选择 experiment / artifact / inference setting，然后完成若干真实 rollout”。
3. **一个 session，多次 episode。**
   Policy、CUDA、camera、arm、hand、RecorderIO 只初始化一次；每局通过已有 `run_generation` 开启新的 inference epoch。
4. **Eval episode 就是 raw physical episode。**
   不新增 `eval.h5`、`policy_episode.h5`、`result.json`、`metrics.json` 等第二套格式。
5. **Realtime runtime 不判断 task success。**
   Runtime 只记录“为什么停止”；任务 SUCCESS / FAILURE / EXCLUDE 由离线人工判断。
6. **不要借重构扩大 scope。**
   RunPolicy/Eval 重构不应顺手修改 raw v28、dataset schema、observation assembler、scheduler、safety gate、tactile unit contract。

---

## 2. 当前事实基线

### 2.1 `examples/run_policy.py`

当前仍提供：

```text
list
check
shadow
run
eval
```

并承担：

- `--eval-seed`、`--max-duration`、`--task-label`、`--operator`；
- checkpoint SHA-256；
- run/eval mode；
- workflow provenance；
- Policy summary / check / shadow / physical run / formal eval 多套分支。

这些职责已经超过论文真机实验的必要入口复杂度。

### 2.2 Lifecycle

当前 `run_policy_deployment()` 的重要正确行为：

```text
resolve config
→ create RuntimeChannels
→ build workers
→ start inference worker alone
→ strict Policy load + warmup
→ wait inference READY
→ only then start hardware/camera/recorder/executor
→ wait remaining readiness
→ ARMED
→ operator / supervisor
→ verified shutdown
```

**必须保留 inference-first startup。** 独立 `check` CLI 不应通过破坏这条路径来替代。

### 2.3 Executor

当前 `PolicyExecutor` 已拥有：

- latest-wins prediction；
- stale prefix skip；
- timestamp-based scheduling；
- generation fence；
- EE IK；
- reject-only safety；
- command progress watchdog；
- raw rollout recording；
- `H → B` physical start gate。

但存在明确 one-shot blocker：

```python
self.rollout_started = False
```

再次 `B` 会提示：

```text
rollout already completed; restart run_policy.py
```

这是 multi-episode 的主要结构性问题。

### 2.4 Evaluation outcome

当前 deployment 维护：

```text
NONE
SUCCESS
FAILURE
INVALID
STOPPED
```

Operator 在 eval 中把：

```text
S → SUCCESS
C → FAILURE
D → INVALID
Q → INVALID / stop
```

Timeout / first-command timeout / command-silence timeout 也会被映射成 task outcome。

这混淆了两个不同问题：

```text
Runtime: 为什么 rollout 停止？
Paper evaluation: manipulation task 是否成功？
```

应彻底拆开。

### 2.5 Recorder / raw episode

当前 Policy rollout 已复用 canonical `EpisodeRecorder`，一个 published raw episode 是：

```text
episode_*/
├── data.h5
├── depth.h5
└── rgb.mp4
```

Recorder 已具备：

```text
START acknowledgement
→ append source rows
→ STOP
→ close writers
→ structural validation
→ atomic publish
```

RecorderClient 也已经可在一次进程生命周期内 start / stop / finalize 后再次 start。

因此 multi-episode **不需要新 Recorder manager，也不需要新 eval data format**。

### 2.6 Real online observation capability

当前 Real deployment 已支持 canonical fields：

```text
joint_state       [19]        float32
rgb               [H,W,3]     uint8
point_cloud       [N,6]       float32
contact_force     [5,3]       float32
tactile_force     [5,120,3]   float32
fingertip_points  [5,3]       float32
eef_pose          [9]         float32
```

在线 EEF / fingertip 使用 canonical FK；dense tactile / aggregate contact 使用 causal/calibrated/unit provenance gates。

**RunPolicy 重构不需要重写 `deployment/inference/observation.py`。**

### 2.7 Dataset gap

Processed v18 / Policy Zarr v11 当前已有 tactile/fingertip，但 processed 不持久化 `eef_pose`。Processing 实际已经调用 `compute_eef_pose_history_xarm_base()` 给 fingertip FK 使用。

这是一个真实的数据侧 gap，但它与 RunPolicy multi-episode 没有运行时依赖，应拆成独立工作包，见 §15。

---

## 3. 最终用户体验

最终用户侧 `run_policy.py` **不再有 subcommand**。

正常运行：

```bash
python examples/run_policy.py action_flow/pick_cube/exp_001
```

论文评测：

```bash
python examples/run_policy.py \
    action_flow/pick_cube/exp_001 \
    --artifact epoch_500-deployment.pt \
    --inference-steps 2 \
    --seed 0 \
    --num-episodes 10 \
    --max-duration 60
```

最终 CLI：

```text
run_policy.py EXPERIMENT
    [--artifact ARTIFACT]
    [--inference-steps N]
    [--seed SEED]
    [--num-episodes N]
    [--max-duration SEC]
    [--device DEVICE]
```

推荐默认值：

```text
artifact         = deployment_latest.pt
inference_steps  = artifact default
seed             = 0
num_episodes     = 1
max_duration     = 60 s
device           = cuda:0
```

### 3.1 参数语义

#### `EXPERIMENT`

Canonical Policy selector，例如：

```text
action_flow/pick_cube/exp_001
```

#### `--artifact`

指定该 experiment `checkpoints/` 下已经 export / qualify 的 deployment artifact。

默认使用 `deployment_latest.pt`，但父进程必须把它解析成真实 target filename，并把**同一个 resolved artifact**传给 inference child，防止：

```text
parent inspect → deployment_latest → checkpoint_A
期间重新 export
child load → deployment_latest → checkpoint_B
```

#### `--inference-steps`

同一份 weights 下的 runtime inference step override。

- ActionFlow / Flow Matching：等价于 NFE；
- Diffusion family：等价于 inference denoise steps；
- 不改变 architecture / weights / observation contract；
- 不应为了 NFE=1/2/4 重新 export 三份 artifact。

Artifact 中已有 `denoise_steps` 应保留为 **default inference steps**，不是 immutable runtime requirement。

#### `--seed`

定义为 **per-episode Policy reset seed**。

本次重构保持一个 session 内所有 episodes 使用同一 seed：

```text
session seed=0
episode_001 reset(seed=0)
episode_002 reset(seed=0)
...
```

这是有意的实验定义：同一 session 中 inference stochastic setting 不变。需要比较不同 stochastic seeds 时，应启动独立 session：

```bash
run_policy.py EXP --seed 0 ...
run_policy.py EXP --seed 1 ...
```

**不要在本轮引入 implicit `seed + episode_index`。** 否则一个 session 内会混入多个 seed，增加分析与复现复杂度。

#### `--num-episodes`

目标成功 publish 到磁盘的 raw episode 数量，默认 `1`。

只在 Recorder 返回：

```text
RecordingFinished
saved=True
error=None
```

后增加计数。

它不是：

```text
按 B 的次数
按 S 的次数
task SUCCESS 数量
```

#### `--max-duration`

每个 physical episode 从 `B` 开始后的最大运行时长。

例如：

```text
--num-episodes 10 --max-duration 60
```

表示：

```text
目标保存 10 个 raw episodes
每个 episode B 后最多运行 60 秒
```

人工 reset scene / homing 的时间不计入。

#### `--device`

纯工程运行参数，例如 `cuda:0`。默认无需写入论文 session config。

---

## 4. 最终 operator workflow

用户只需要记住：

```text
H = Home
B = Begin
S = Stop + Save
Q = Quit session
ESC = Emergency stop
```

推荐实际流程：

```text
process start
→ inference/hardware ready
→ ARMED

Episode 1 / N
H
→ robot home
→ manual scene setup
→ B
→ RUNNING + recording
→ S or timeout
→ motion fenced
→ recorder finalize
→ episode_001 published
→ ARMED

Episode 2 / N
H
→ scene setup
→ B
...

N / N published
→ automatic clean shutdown
```

### 4.1 为什么是 `H → scene setup → B`

默认先 Home，再摆场景。Home planner 不知道操作者随后放入的动态物体；先摆物体再执行大幅 Home motion 会增加碰撞/扰动风险。

一局结束后，如果物体位于 Home trajectory 附近，应先人工清理，再按 `H`。

### 4.2 每局结束必须重新 H

当前 `_end_policy_run()` 已清除 `physical_home_completed`。保留该行为。

因此：

```text
episode_001 stop
→ ARMED
→ B without new H = reject
→ H
→ B = accept
```

这保证每个 trial 都从 canonical training start pose 开始。

---

## 5. Multi-episode 状态机

最终状态机：

```text
PROCESS START
    │
    ├─ inspect + pin artifact
    ├─ create session/run_config
    │
    ▼
INFERENCE START
    │
    ├─ strict restore
    ├─ effective inference_steps
    └─ warmup
    │
    ▼
HARDWARE START
    │
    ▼
ARMED
    │
    │ H
    ▼
HOME READY
    │
    │ manual scene setup
    │ B
    ▼
RUNNING
    │
    ├─ S ────────────────┐
    ├─ max_duration ─────┤
    └─ technical stop ───┤
                         ▼
                 MOTION FENCED
                         │
                         ▼
                 RECORDER FINALIZE
                         │
                  saved successfully
                         │
                         ▼
                 completed_episodes++
                         │
             ┌───────────┴───────────┐
             │                       │
         count < N               count == N
             │                       │
           ARMED                 request quit
             │                       │
             └──── H → ...           ▼
                                 CLEAN SHUTDOWN
```

### 5.1 删除 one-shot gate

从 `PolicyExecutor` 删除：

```python
self.rollout_started
```

以及所有：

```text
rollout already completed; restart run_policy.py
```

分支。

保留每局现有 episode-local reset：

```text
stats reset
prediction state reset
scheduler state reset
recorded action reset
run_generation new epoch
Policy runtime.reset_episode()
```

### 5.2 计数点

**唯一合法计数点是 recorder finalization success。**

伪代码：

```python
def _complete_recording(result):
    if result.error or not result.saved:
        ...
        return

    completed_episodes += 1
    print(f"Episode {completed_episodes}/{num_episodes} saved")

    if completed_episodes >= num_episodes:
        shared.quit_requested.value = True
```

不要在 `B` / `S` / `stop_episode()` 时提前计数。

### 5.3 Recording failure

如果本来要求保存的 episode 最终：

```text
saved=False
or error != None
```

则：

- 不增加 `completed_episodes`；
- 不伪造成功 episode；
- recording transport / persistent storage failure 应 fail closed，终止 session；
- 不设计 automatic retry manager。

Recorder START 在 motion 之前已有 ACK barrier，必须继续保留：**没有 recording ACK，不允许进入 physical RUNNING。**

---

## 6. Stop / Quit / Timeout 语义

Runtime 只记录 technical stop reason，不判断 task outcome。

推荐 canonical reasons：

```text
operator
timeout
quit
first_command_timeout
command_silence_timeout
hardware_fault
estop
recorder_fault
recording_failure
max_frames
runtime_shutdown
```

不再使用：

```text
eval:success:operator
eval:failure:timeout
eval:invalid:...
run:stopped:...
```

### 6.1 `S`

语义固定为：

```text
fence motion
→ stop current recording with save=True
→ finalize raw episode
→ ARMED
```

`S` **不表示 SUCCESS**。

### 6.2 `Q` while ARMED

直接请求结束整个 session。

### 6.3 `Q` while RUNNING

```text
immediate motion fence
→ save current partial raw episode
→ wait recorder finalize
→ quit session
```

最终 session 可以显示例如：

```text
4 / 10 episodes saved; operator quit
```

不产生 task INVALID。

### 6.4 `max_duration`

属于 evaluation protocol：

```text
max duration reached
→ stop motion
→ save raw episode
→ stop_reason=timeout
→ ARMED
```

如果 raw 成功 publish，这一局属于 `num_episodes` 的一个 saved episode。

### 6.5 first-command / silence watchdog

它们是 technical boundary，不是 task failure：

```text
first_command_timeout
command_silence_timeout
```

可保存 raw evidence；是否在论文统计中 exclude，由离线人工分析决定。

不要恢复 online `INVALID` framework。

---

## 7. 删除 realtime task outcome system

删除 deployment-only outcome 链：

```text
EvaluationOutcome

evaluation_outcome_stop_reason()
write_rollout_result()

RuntimeChannels.evaluation_outcome

_request_evaluation_outcome_and_stop()
executor.rollout_result
result.json
```

对应修改：

```text
dexmani_real/deployment/evaluation.py
dexmani_real/deployment/operator.py
dexmani_real/deployment/executor.py
dexmani_real/ipc/channels.py
tests/test_policy_rollout.py
repo_map.md / relevant docs
```

### 7.1 不要删除全局 C / D command

`OperatorCommand.PAUSE` / `OperatorCommand.DISCARD` 被 teleop 正式使用。

因此：

```text
runtime/operator_input.py
```

的：

```text
C → PAUSE
D → DISCARD
```

必须保留。

Policy deployment 中：

- UI 不再宣传 C/D；
- 收到 C/D 时可忽略并打印一次明确 warning；
- 不赋予 success/failure/invalid 意义。

不要为了 Policy Eval 简化破坏 teleop。

### 7.2 S/Q immediate callback

删除 outcome ordering 后，Policy operator 可以重新统一使用 immediate `stop_callback` / `quit_callback`。

这是 desirable 的：当 operator thread 正阻塞在 `H` homing 时，`S/Q/ESC` 仍可立即 fence / quit / estop，而不必等待 Home 完成。

---

## 8. Session 目录与数据格式

最终输出结构：

```text
rollouts/
└── <policy>/
    └── <task>/
        └── <experiment>/
            └── session_YYYYMMDD_HHMMSS/
                ├── run_config.yaml
                ├── episode_001/
                │   ├── data.h5
                │   ├── depth.h5
                │   └── rgb.mp4
                ├── episode_002/
                │   ├── data.h5
                │   ├── depth.h5
                │   └── rgb.mp4
                └── ...
```

不要再构造：

```text
eval/
seed_000/
nfe_2/
solver_x/
```

参数属于 config，不属于目录 hierarchy。

### 8.1 Session naming

使用 wall-clock：

```text
session_YYYYMMDD_HHMMSS
```

若极端情况下同秒冲突，可使用最简单的 `_01`, `_02` 后缀。

不要建设 UUID/session database。

### 8.2 `run_config.yaml`

只保存**本次真正执行的 resolved experimental conditions**：

```yaml
experiment: action_flow/pick_cube/exp_001
artifact: epoch_500-deployment.pt
inference_steps: 2
seed: 0
num_episodes: 10
max_duration_s: 60
```

规则：

- 即使 CLI 没显式写 `--inference-steps`，也要保存 artifact default 的 resolved value；
- 即使用户用默认 `deployment_latest.pt`，也要保存 parent 已 resolve 的真实 artifact filename；
- 在任何 hardware process 启动之前写好 config；
- config write 失败则不要启动硬件。

不保存：

```text
Git commit
Git dirty/origin
checkpoint SHA256
full Real config
full PolicySpec
operator
GPU / CUDA version
online success count
runtime metrics summary
```

`device` 属于工程运行参数，默认不进入论文实验 config。

---

## 9. Episode naming 与 Recorder 修改边界

当前 teleop 依赖 timestamp episode naming，不能全局改成顺序编号。

为 generic recorder 增加窄的 optional explicit name：

```python
RecorderClient.start_episode(
    *,
    task_label: str = "",
    operator: str = "",
    episode_name: str | None = None,
) -> bool:
    ...
```

一路透传：

```text
StartRecording
→ RecorderIO
→ EpisodeRecorder.start_episode(..., episode_name=None)
```

行为：

```text
episode_name=None
→ 保持当前 teleop timestamp naming

policy eval episode_name="episode_003"
→ session_dir/episode_003/
```

显式 name 已存在时 fail loudly；不要覆盖，也不要自动改成一个意外 suffix。

Policy executor 使用：

```python
episode_index = completed_episodes + 1
episode_name = f"episode_{episode_index:03d}"
```

---

## 10. Eval episode 必须继续使用 canonical raw v28

不要创建第二套 Policy Eval storage schema。

当前 raw 已保存足够的 physical facts：

```text
timestamp
arm_qpos
arm_qvel
arm_tau
hand_qpos
hand_current
hand_contact
hand_tactile_force

action_arm_joint_sent
action_hand_joint
action_arm_ee

flag_action_queued
flag_frame_status
tracking_error
sensor source timestamps
camera freshness/health
tactile provenance
...
```

Camera：

```text
rgb.mp4
depth.h5
```

加上 data.h5 中的 camera calibration / intrinsics / extrinsics / depth scale。

### 10.1 不保存派生 observation duplicate

即使 Policy 使用：

```text
eef_pose
fingertip_points
point_cloud
```

raw 仍不新增这些 duplicate truth sources。

离线确定性重建：

```text
arm_qpos
→ canonical Arm FK
→ eef_pose

arm_qpos + hand_qpos
→ canonical arm/hand FK
→ fingertip_points

RGB + depth + calibration
→ canonical point-cloud builder
→ point_cloud
```

### 10.2 不默认保存 prediction chunk

不要本轮新增：

```text
pred_action_chunk
prediction_id
future_actions
inference trace
```

canonical eval raw 首先记录：

```text
机器人实际状态
机器人实际收到的 command
sensor facts
technical stop reason
```

未来若研究 replanning / temporal consistency，再设计独立 debug trace，不污染 canonical raw。

### 10.3 本轮不修改 raw v28

明确禁止为 RunPolicy 重构修改：

```text
EPISODE_SCHEMA_VERSION
fill_reason
flag_sample_valid
observation_valid
meta.success
```

已有设计记录明确指出不要为了清几个 scalar 单独制造 raw v29。

`meta.success` 当前含义实际上是 storage `save`，不是 task success；本轮保持不动，只禁止任何 offline task evaluator 把它解释为 manipulation success。

`observation_valid=False` 在 current dataset admission 中也不是单独 rejection gate；本轮不要猜测并强行改成 True。

---

## 11. Scientific contract：哪些“严格检查”必须保留

不要把“metadata 很多”直接等同于过度工程。

判断标准：

> 如果某配置变化会让同一个 raw scene / robot state 生成不同的 model observation 数值或物理含义，则属于 scientific contract，应保留；如果只是软件来源、代码状态、管理 provenance，则可以删除。

### 11.1 必须保留

#### Basic tensor/action contract

```text
observation modality names
shape
dtype
action representation
action/control dimensions
horizon
n_obs_steps
n_action_steps
control_dt
normalizer compatibility
encoder consumed_observation_fields
```

#### Point Cloud

保留足以防止 train/deploy numeric drift 的完整 numeric/geometry contract，包括当前 Real 所依赖的：

```text
frame
units
representation
color semantics
point count
workspace-related preprocessing
table plane when used
numeric pointcloud processing config
sampling / transform identity needed to prove same producer
```

Point Cloud 即使始终是 `[N,6]`，改变 crop / voxel / table / outlier / sampling 后输入分布也已经改变。

#### Fingertip

保留：

```text
frame
units
finger order
canonical derivation
fingertip link names
handbase position in EEF
handbase quaternion in EEF
```

这些值会直接改变 `[5,3]` XYZ。

#### EEF

保留：

```text
position + rot6d representation
xarm_base frame
meters
canonical FK derivation/model identity sufficient to prevent silent frame change
```

#### Tactile / Contact

**禁止简化最新 unit/calibration contract。**

当前事实：

```text
XHand values = SDK native values
SDK native unit has NOT been experimentally proven to be Newton
```

必须继续诚实声明：

```text
xhand_sdk_native_unknown_si
si_verified = false
```

并继续保留：

```text
calibration status
unit identity
finger/sensor/point ordering
axis labels
spatial_geometry_verified=false
```

当前 `contact_force_valid` 对 calibration/native-unit 的要求也必须保持。

### 11.2 可以删除/降级

这些不决定 model observation 的科学含义：

```text
Git repository identity
Git origin
clean working tree
Git commit as export hard gate
checkpoint SHA256 in run_policy
operator-facing workflow provenance
online task outcome/result summary
```

Policy exporter 的 Git 简化属于 `dexmani_policy` 独立改动，Real 不应重新实现一套 provenance。

---

## 12. 用户 CLI 简化不等于删除内部 dry-run/test capability

删除用户侧：

```text
list
check
shadow
run
eval subcommand
```

但不要因此删除 lifecycle / executor 内部：

```python
execute=False
```

已有 no-hardware deterministic tests 依赖该能力。

正确边界：

```text
删除 shadow CLI
≠
删除 internal non-publishing execution path
```

同样，独立 `check` CLI 删除后，仍保留：

- Policy export / qualify strict restore；
- inference child strict load；
- inference startup warmup；
- inference ready before hardware。

---

## 13. `dexmani_policy` 前置依赖（Work A）

Real 的最终 CLI 依赖 Policy public runtime 提供两个能力：

1. explicit deployment artifact selection；
2. runtime `inference_steps` override。

### 13.1 Artifact selection

建议 public API：

```python
def inspect_experiment(
    selector,
    *,
    artifact: str | None = None,
) -> ExperimentInfo:
    ...


def load_experiment(
    selector,
    device: str = "cuda:0",
    seed: int = 0,
    *,
    artifact: str | None = None,
    inference_steps: int | None = None,
) -> LoadedPolicy:
    ...
```

规则：

- `artifact=None`：解析 `checkpoints/deployment_latest.pt`；
- explicit artifact：只能解析当前 experiment `checkpoints/` 内文件；
- 禁止 path escape；
- `ExperimentInfo.checkpoint_name` 返回真实 resolved filename；
- Real parent inspect 后，把该 resolved filename传给 inference child。

### 13.2 Runtime inference steps

Artifact 中 persisted `denoise_steps` 保持兼容，但 public semantics 改为 default inference steps。

建议 `PolicySpec` 暴露：

```python
default_inference_steps: int
```

`LoadedPolicy` 保存 effective value：

```python
effective = (
    spec.default_inference_steps
    if inference_steps is None
    else inference_steps
)
```

真正调用：

```python
agent.predict_action(
    tensors,
    denoise_timesteps=effective,
)
```

warmup 与实际 prediction 必须使用同一 effective value。

### 13.3 不做 generic override system

不要增加：

```text
--policy-override key=value
InferenceOptionsRegistry
arbitrary dict[str, Any]
```

以后真的需要 solver/guidance/temperature sweep 时，再逐个增加有明确科学语义的参数。

---

## 14. `dexmani_real` P0 实施步骤（Work B）

按以下顺序执行；每一步测试通过再继续。

### B0. 基线确认

修改前：

```bash
git status --short
git rev-parse HEAD
```

只作为开发过程事实，不写进 runtime business logic。

运行当前相关纯软件 tests，建立 baseline。

### B1. 重写 `examples/run_policy.py`

删除：

```text
subparsers
list/check/shadow/run/eval handlers
statistics/check warmup output
checkpoint SHA-256
mode
--eval-seed
--task-label
--operator
workflow provenance
```

保留薄入口职责：

```text
parse args
→ inspect/pin Policy artifact
→ resolve effective inference steps
→ resolve Real config
→ create session dir
→ write run_config.yaml
→ construct narrow configs
→ call run_policy_deployment()
```

内部 recording metadata 可继续：

```text
task_label = info.task_name
operator = getpass.getuser()
```

只是不要暴露成论文实验 CLI，也不要写进 `run_config.yaml`。

### B2. 收缩 `deployment/evaluation.py`

删除 task outcome / result writer。

保留一个具体、pickle-safe 的 session/recording config 即可。为降低 churn，可以继续使用现有 `RolloutRecordingConfig` 名称，但字段缩为实际需要：

```python
@dataclass(frozen=True)
class RolloutRecordingConfig:
    data_dir: str
    task_label: str
    operator: str
    max_running_s: float
    num_episodes: int
```

删除：

```text
mode
provenance
EvaluationOutcome
write_rollout_result
```

不要创建 Manager/Service。

### B3. 删除 outcome IPC

从 `RuntimeChannels` 删除：

```text
evaluation_outcome
```

同步 tests / repo map。

### B4. 简化 `deployment/operator.py`

Policy operator：

```text
B start
S immediate stop/save
H home
Q immediate quit
ESC e-stop
```

C/D 在 Policy deployment 中 warning/ignore；全局 enum/key mapping 保留给 teleop。

删除所有 `_request_evaluation_outcome_and_stop()` 相关逻辑。

### B5. Multi-episode `PolicyExecutor`

删除：

```text
rollout_started
rollout_result
EvaluationOutcome imports/branches
mode-specific stop mapping
```

增加：

```python
self.completed_episodes = 0
self.num_episodes = recording_config.num_episodes
```

开始新 episode 时使用：

```python
next_index = self.completed_episodes + 1
```

Recorder ACK 成功后再进入 RUNNING。

Recorder finalize success 后：

```python
self.completed_episodes += 1
```

达到目标后请求 clean quit。

### B6. Simplify stop reasons

删除 `_rollout_timeout_result(mode, ...)`。

改成纯 technical mapping，例如：

```python
def _timeout_stop_reason(reason: str) -> str:
    return {
        "run time limit": "timeout",
        "first command timeout": "first_command_timeout",
        "command silence timeout": "command_silence_timeout",
    }[reason]
```

不要返回 task outcome。

### B7. Recorder explicit episode name

最小扩展：

```text
StartRecording.episode_name
RecorderClient.start_episode(... episode_name=None)
RecorderIO forward
EpisodeRecorder.start_episode(... episode_name=None)
```

Policy 使用 `episode_001` 等；teleop `None` 保持现状。

### B8. Lifecycle

保留：

```text
inference-first startup
readiness
supervisor
heartbeat
verified shutdown
camera always on for recorded Policy eval
```

更新 operator help：

```text
Episode 1/N
[B] begin  [S] stop/save  [H] home  [Q] quit  [ESC] e-stop
```

不再打印 success/failure/invalid keys。

### B9. `run_config.yaml`

Session dir 创建后、启动 inference/hardware 之前写入。

不要建设通用 config serializer；使用 PyYAML 的简单 deterministic dump 即可。

### B10. Visualizer 最小修复

至少修：

```text
raw schema-v25
→ current raw schema / raw v28
```

P1 可增加：

```text
arm actual vs command
hand actual vs command
actual EEF vs target EEF
```

不要创建 `visualize_eval_episode.py`。

---

## 15. Dataset modality completion 独立工作包（Work C）

**不要与 Work B 同一个大 patch。**

目标：让 future Policy 可以训练时直接读取 canonical：

```text
eef_pose [N,9] float32
```

### C1. Processed

当前 processing 已经为了 fingertip 做：

```python
eef_pose = compute_eef_pose_history_xarm_base(joint_state[:, :7])
```

应复用这个一次计算结果：

```python
eef_pose = compute_eef_pose_history_xarm_base(joint_state[:, :7])
output["eef_pose"][:] = eef_pose.astype(np.float32)

output["fingertip_points"][:] = compute_fingertip_history_xarm_base(
    ...,
    eef_pose_history=eef_pose,
)
```

不要重复 Arm FK。

### C2. Schema bump

因为这是新增 required dataset：

```text
processed v18 → v19
Policy Zarr v11 → v12
```

Raw v28 不变。

### C3. EEF semantics

至少保留：

```text
representation = position_m_rot6d
frame = xarm_base
position_units = m
rotation_representation = rot6d
canonical FK identity/derivation sufficient to prevent silent drift
```

### C4. Policy exporter

随后在 `dexmani_policy` exporter 补齐 canonical observation support：

```text
eef_pose
tactile_force
```

Tactile validity filter/mask strategy属于具体 dataset/model policy，不要再次改 Real online deployment infrastructure。

---

## 16. 明确禁止事项

Claude Code 实施 Work B 时 **DO NOT**：

```text
修改 raw EPISODE_SCHEMA_VERSION
修改 recording/storage/schema.py 的字段集合
删除 observation_valid/fill_reason/flag_sample_valid
删除/改写 raw meta.success
修改 deployment/inference/observation.py
修改 causal observation alignment
修改 PointCloud preprocessing
弱化 PointCloud numeric compatibility contract
弱化 Fingertip mount geometry contract
弱化 Tactile calibration/unit/order contract
声称 tactile unit 是 Newton
修改 action scheduler / stale-prefix skip
增加 temporal ensemble / RTC
修改 IK/safety semantics
修改 arm/hand worker
删除 internal execute=False
删除 teleop C/D
设计 generic policy override
设计 registry/plugin/strategy/manager framework
建设 success-rate/metrics/result database
自动执行真实硬件测试
```

Work C 才允许显式修改 processed/Zarr schema，并必须单独 review/version bump/rebuild。

---

## 17. 测试计划

### 17.1 `dexmani_policy`（Work A）

至少覆盖：

```text
artifact=None resolves deployment_latest
explicit artifact resolves exact file
artifact cannot escape experiment/checkpoints
resolved ExperimentInfo exposes pinned filename

inference_steps=None uses artifact default
inference_steps=N overrides default
zero/negative inference_steps rejected
warmup uses effective steps
predict_action_chunk uses effective steps

strict restore unchanged
normalizer validation unchanged
qualify direct/export parity unchanged
```

如果同时处理 Policy exporter Git 简化：

```text
export does not call git
export works with dirty/non-git/fork-like working tree
scientific observation/data contract remains enforced
```

### 17.2 Real Policy rollout tests

重点修改 `tests/test_policy_rollout.py`，覆盖：

#### Multi-episode

```text
B → S → recorder finalize → ARMED
then H → B can start second episode
no rollout_started one-shot state remains
```

#### H gate

```text
initial B without H rejected
H completed → B accepted
end episode clears physical_home_completed
next B without H rejected
```

#### Count

```text
RecordingFinished(saved=True,error=None) → completed += 1
saved=False → no increment
error != None → no increment / fail closed as appropriate
```

#### Automatic completion

```text
num_episodes=2
save episode_001
save episode_002
→ quit_requested=True only after second finalization
```

#### Timeout

```text
max_duration boundary
→ stop_reason=timeout
→ no EvaluationOutcome
→ raw save requested
```

#### Command watchdog

```text
first command timeout
command silence timeout
→ technical stop reasons only
```

#### Q

```text
Q while ARMED → clean quit
Q while RUNNING → fence + save/finalize + quit
```

#### Internal dry-run

Ensure `execute=False` deterministic tests remain valid。

### 17.3 Recorder tests

覆盖：

```text
start_episode(episode_name=None)
→ existing timestamp naming unchanged

start_episode(episode_name="episode_001")
→ exact directory

duplicate explicit name
→ fail loudly / no overwrite

first finalize then second start
→ both published correctly
```

### 17.4 Operator tests

覆盖：

```text
Policy: S/Q immediate semantics
Policy: C/D ignored or warned
Teleop/global KeyboardInput: C/D mapping unchanged
```

### 17.5 Raw reader/viewer

```text
Policy-created episode opens with EpisodeReader
raw structural validation passes
visualize_episode.py --info works
```

不要为了新的 session hierarchy 修改 `EpisodeReader`：它只应该收到一个 episode directory。

### 17.6 Work C dataset tests

独立覆盖：

```text
processed eef_pose shape [N,9]
dtype float32
finite
eef_pose == canonical FK(joint_state[:,:7])
rot6d valid
fingertip reuses same EEF result
Zarr contains eef_pose
schema versions bumped correctly
```

---

## 18. 软件验收标准

Work A + B 完成后，不连接硬件也必须证明：

```text
run_policy parser has no subcommands
num_episodes default = 1
max_duration default = 60

resolved artifact pinned parent → inference child
resolved inference_steps recorded in run_config

multi-episode executor has no rollout_started gate
completed count happens only after recorder finalization

no EvaluationOutcome chain remains in Policy deployment
no result.json writer remains
RuntimeChannels no evaluation_outcome

teleop C/D behavior unchanged
raw schema version unchanged
scientific modality compatibility unchanged
```

全文搜索应确认不再存在 deployment task-outcome remnants，例如：

```text
EvaluationOutcome
write_rollout_result
eval:success
eval:failure
rollout_started
```

但不要机械删除历史文档中的 forensic mentions；只修改 current spec / code ownership 文档。

---

## 19. 第一次人工硬件验收

Claude Code **禁止自动运行**。代码完成后只输出：

```text
Hardware validation: NOT RUN
```

人工第一次使用短时、双 episode：

```bash
python examples/run_policy.py \
    action_flow/pick_cube/exp_001 \
    --num-episodes 2 \
    --max-duration 10
```

依次验证：

1. inference READY 出现在 arm/hand/camera startup 之前；
2. ARMED 后未按 H，B 被拒绝；
3. H 成功后，人工设置 scene；
4. B 开始 Episode 1/2；
5. S 立即 fence motion；
6. `episode_001` 完整 publish；
7. 自动回 ARMED；
8. 未重新 H，B 被拒绝；
9. H → scene setup → B；
10. Episode 2/2 正常运行；
11. S 后 `episode_002` publish；
12. 只有在第二局 finalization 完成后自动 clean shutdown；
13. session 中存在且仅需：

```text
run_config.yaml
episode_001/
episode_002/
```

14. 两个 episode 都可以被 `EpisodeReader` 打开；
15. `examples/visualize_episode.py` 可直接读取两个 episode；
16. `data.h5:/meta/stop_reason` 正确；
17. 没有 `result.json`；
18. 没有 task SUCCESS/FAILURE runtime label。

随后再人工验证：

```text
--inference-steps 1
--inference-steps 2
--inference-steps 4
```

只确认 runtime override 确实生效；Policy 性能比较不属于本重构验收。

---

## 20. 推荐提交 / PR 划分

不要把全部工作压进一个巨大 commit。

### Work A — `dexmani_policy`

建议内容：

```text
runtime: support explicit deployment artifact
runtime: support inference step override
runtime: expose resolved default inference steps
(optional separate commit) export: remove Git provenance gates
```

### Work B — `dexmani_real`

建议按可 review 单元拆 commit：

```text
1. refactor: simplify run_policy research CLI
2. refactor: remove deployment task outcomes
3. feat: support persistent multi-episode policy eval
4. feat: add explicit recorder episode names
5. docs/tests: update policy eval workflow and regressions
6. viz: fix raw viewer current-schema docs / optional command overlays
```

### Work C — dataset / Policy modality completion

独立 PR：

```text
data: persist canonical eef_pose in processed/Zarr
policy: export eef_pose/tactile_force deployment contracts
```

这样可以独立回归：

```text
A = Policy runtime boundary
B = Real eval lifecycle
C = dataset schema evolution
```

---

## 21. Claude Code 执行要求

执行时遵循以下顺序：

```text
1. 先阅读当前源代码，不以本文伪代码代替事实。
2. 检查当前 main 与本文事实基线之间是否有新 commit。
3. 若当前代码已发生相关变化，先报告差异，再按目标语义最小适配。
4. 每个 phase 只修改该 phase 的 owner 文件。
5. 每一步先写/更新 deterministic tests，再推进下一层。
6. 不自动运行真实硬件。
7. 不为了“清理”顺手删除 current scientific contracts。
8. 不引入 generic framework。
9. 修改 current behavior 后同步 current docs/repo_map；不要改写历史 forensic evidence。
10. 最终报告：修改文件、行为变化、测试结果、未运行的硬件验证。
```

实现前应重点重新阅读：

```text
examples/run_policy.py

dexmani_real/deployment/config.py
dexmani_real/deployment/lifecycle.py
dexmani_real/deployment/executor.py
dexmani_real/deployment/operator.py
dexmani_real/deployment/evaluation.py
dexmani_real/deployment/inference/worker.py

dexmani_real/ipc/channels.py

dexmani_real/recording/client.py
dexmani_real/recording/io_worker.py
dexmani_real/recording/recorder.py
dexmani_real/recording/storage/schema.py
dexmani_real/recording/storage/reader.py

examples/visualize_episode.py

tests/test_policy_rollout.py
tests/test_recorder_io_boundary.py
tests/test_recorder_queue_io.py
tests/test_keyboard_input.py
```

Work C 再额外阅读：

```text
dexmani_real/dataset/processing.py
dexmani_real/dataset/processed.py
dexmani_real/dataset/export.py
docs/data_schema.md
```

---

## 22. 最终目标架构

```text
dexmani_policy
    train
      ↓
    export / qualify
      ↓
    deployment artifact
      │
      │ weights + scientific model contract
      │ default inference steps
      ▼

examples/run_policy.py EXP
    --artifact
    --inference-steps
    --seed
    --num-episodes
    --max-duration
      │
      ▼

one persistent Real session
    │
    ├─ inference process loaded once
    ├─ hardware/camera/recorder loaded once
    │
    ├─ H → scene → B → S
    │        episode_001 raw
    │
    ├─ H → scene → B → S
    │        episode_002 raw
    │
    └─ ... N episodes
      │
      ▼

session_<timestamp>/
    run_config.yaml
    episode_001/{data.h5,depth.h5,rgb.mp4}
    episode_002/{data.h5,depth.h5,rgb.mp4}
    ...
      │
      ▼

offline
    EpisodeReader
      ├─ visualize_episode.py
      ├─ manual SUCCESS/FAILURE/EXCLUDE labels
      └─ paper analysis scripts
```

最终职责边界一句话：

> **Policy deployment 只保证模型与真实 observation/action 的科学语义一致；`run_policy` 只负责安全、高效地完成指定数量的真实机器人 trials；每个 trial 保存 canonical raw physical evidence；任务成功率与论文统计完全留给 offline analysis。**
