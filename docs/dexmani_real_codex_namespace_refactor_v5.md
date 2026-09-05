# DexMani Real 机器人学命名与包结构重构方案 v5

## 面向 Codex 的执行指导文档

> **Review 基线**：`haoyangzhanglab/dexmani_real@f3e3cc064cb5537d605a06c383263ec776164c6c`
> **基线提交**：`refactor: unify real robot runtime contracts`
> **目标**：优化 Python package / module / class 命名、文件归属、import dependency 与少量明确的职责拆分；**不改变机器人运行语义、控制策略、安全边界、IPC schema、数据 schema 或配置字段语义**。
> **适用对象**：Codex 主代理及项目内 `sol-high` / `terra-xhigh` / `luna-max` 子代理。
> **重要规则**：如果执行时 `HEAD` 已不是上述基线，先对新增提交重新做 scope audit，再决定是否继续；不得机械套用本方案。

---

# 1. Executive Summary

本次 v5 在 v4 的基础上只做一类结构升级：**允许 `teleop/`、`deployment/`、`recording/` 各自再显式建立一个稳定二级 subsystem package**。这不是放宽“多建目录”的限制，而是利用已经存在于源码中的自然边界，把 workflow、model inference 与 persisted storage 三类复杂度分别收拢。

核心判断保持不变：

**真实机器人安全边界与完整 vertical data/control flow 优先于目录整齐。只有当一组文件共享稳定 owner、内部数据流和依赖方向时才形成二级 package。**

最终策略：

1. 保留顶层 `planning/`、`control/`、`deployment/`、`ipc/`、`runtime/`，不引入 `core/common/service/manager` 等技术模式顶层。
2. 最终只保留以下有明确领域意义的二级 package：
   - `robot/drivers/`：hardware backend；
   - `sensor/camera/`：camera backend、geometry、clock、worker；
   - `planning/kinematics/`：pose/FK/IK/IK candidate/fingertip；
   - `teleop/control_loop/`：causal realtime teleoperation control algorithm；
   - `teleop/retargeting/`：hand retargeting backends；
   - `deployment/inference/`：model-facing observation/runtime/adapter/inference child；
   - `recording/storage/`：persisted raw episode schema/read/write/encoding；
   - `calibration/camera/`：camera calibration subsystem。
3. `teleop/` root 只保留 workflow/lifecycle/operator/recording orchestration；`control_loop/` 收拢一次实时遥操作控制 tick 的算法依赖，但 **`grid.py` 本身不再继续拆**。
4. `deployment/` 明确形成两半：
   ```text
   inference/ → Prediction → executor.py → control/ → robot/
   ```
   `Prediction` 留在 deployment root，作为 inference 与 executor 的共享边界。
5. `recording/` 明确形成：
   ```text
   runtime transaction → storage/
   ```
   `EpisodeRecorder` 留在 root 作为 transaction owner，`storage/` 只拥有 persisted representation 与 file IO。
6. 将 `data/` 重命名为 `dataset/`，并把 processed artifact contract 从 conversion transaction 中分离为 `dataset/processed.py`。
7. 删除 `planning/types.py`、`robot/types.py` 这类 catch-all owner，将类型放回真实领域。
8. 将跨 workflow 的 keyboard/operator input 从 `teleop/` 移到 `runtime/operator_input.py`，Cartesian jog 纯函数放 `control/jog.py`，消除 `deployment/replay/calibration -> teleop` 反向依赖。
9. 保留最新源码已经形成的安全边界：
   ```text
   ActionCandidate → SafetyGate → publication → worker final SDK guard
   ```
   不合并 `action.py/publication.py`，不重命名 `ActionCandidate`、`SafetyGate`、`CollisionModel`。
10. 所有变更必须是 **behavior-preserving namespace / ownership refactor**；不得引入 compatibility shim、temporary alias、registry/factory、第二套 validation 或额外 runtime abstraction。

# 2. Source of Truth 与 Review 方法

Codex 开始前必须按下列优先级理解仓库：

```text
runtime behavior / source code
    ↓
IPC / persisted schema / canonical configuration
    ↓
README / focused implementation docs
    ↓
repo_map.md
    ↓
agent / style guidance
```

本次 review 特别参考：

```text
AGENTS.md
code_style.md
repo_map.md
user_design.md
docs/action_clip_mechanisms.md
docs/data_schema.md
docs/policy_deployment.md
```

其中必须重点保护 `user_design.md` 与 `docs/action_clip_mechanisms.md` 已明确的行为，例如：

- teleop 不以 table collision 作为动作拒绝条件；
- replay 主轨迹同样不以 table collision 拒绝，但 return-home 保留 table-aware safety；
- XHand startup 不允许隐式运动；
- learned-policy arm spike 是 producer-side clip；
- learned-policy hand jump 是 reject-only；
- worker 仍保留独立 SDK-side final guard。

这些都不是本次重构可以“顺手优化”的对象。

---

# 3. Community Convention Calibration

本方案借鉴社区规范，但不复制大型通用框架。

## 3.1 MoveIt / MoveIt2

MoveIt 使用明确机器人学领域术语，例如：

```text
robot_model
kinematics
collision_detection
planning_interface
trajectory_processing
```

对本仓库的启示：

- 保留 `planning`；
- `kinematics` 是自然二级 package；
- `robot/model.py` 比 `robot_spec.py` 更符合 robotics vocabulary；
- `CollisionModel` 是合法且标准的名词，因为它确实拥有 collision geometry/model/data；
- 不应为了避免 `Model` 而强行改成 `Checker`。

## 3.2 ros2_control

ros2_control 明确区分 controller、hardware interface、joint limits、command/state interfaces。

对本仓库的直接启示是保留：

```text
producer action/proposal
    → safety validation
    → command publication
    → worker / hardware SDK final guard
```

因此当前：

```text
control/action.py
control/safety_gate.py
control/publication.py
robot/command_validation.py
robot/*_worker.py
```

的分层合理，不应为了“文件少”而合并。

## 3.3 LeRobot

LeRobot 的 RealSense backend 使用明确命名：

```text
RealSenseCamera
RealSenseCameraConfig
```

同时 backend-specific dependency 不应被父 package 自动 import。

因此建议：

```text
sensor/camera/realsense.py
    RealSenseCamera
    RealSenseCameraConfig
```

并让 consumer 显式：

```python
from dexmani_real.sensor.camera.realsense import RealSenseCamera
```

而不是让 `dexmani_real.sensor.camera` 隐式加载 `pyrealsense2`。

## 3.4 Robot Learning research repositories

研究代码的主要价值是：

- 入口可追踪；
- 数据流线性；
- experiment 修改成本低；
- 关键算法与边界容易定位；
- 有时少量重复优于 manager/service/factory 链。

因此：

- 大文件不等于应该拆；
- `teleop/control_grid.py`、`deployment/executor.py`、`recording/recorder.py` 这类完整 vertical owner 可以保留；
- 只在职责真的分裂时拆文件。

## 3.5 Policy repository 与 real deployment

模型 policy 与 real-robot deployment 是两个概念。

本仓库不拥有 policy model architecture，而是负责：

```text
observation assembly
model runtime adaptation
inference
prediction IPC
real-robot scheduling
safety
hardware execution
```

所以顶层继续叫 `deployment/`，而不是 `policy/`。

---

# 4. 旧方案 Review：保留、修订与撤销

| 旧方案决策 | v5 结论 | 原因 |
|---|---|---|
| 保留顶层 `planning/` | **保留** | 标准 robotics domain |
| 保留顶层 `control/` | **保留** | 共享 command/safety/publication owner |
| 保留顶层 `deployment/` | **保留** | real-robot deployment，不是 policy model implementation |
| 保留顶层 `ipc/` | **保留** | typed shared-memory infrastructure 是稳定独立层 |
| `data/ -> dataset/` | **保留** | 更准确表达 offline dataset construction |
| `robot/drivers/` | **保留** | hardware backend family |
| `sensor/camera/` | **保留** | camera backend/geometry/clock/worker family |
| `planning/kinematics/` | **保留并修订** | 用 `arm_fk.py` / `hand_fk.py`，不缩成 `arm.py` / `hand.py` |
| `teleop/retarget -> retargeting/` | **保留** | 稳定 hand-retargeting subsystem |
| `teleop/control_loop/` | **新增** | 明确分离 workflow orchestration 与 causal realtime control algorithm |
| `deployment/inference/` | **新增，优先级最高** | 明确分离 model-facing inference 与 physical execution |
| `recording/storage/` | **新增** | 明确分离 transaction/runtime 与 persisted artifact implementation |
| 删除 `planning/types.py` | **保留** | catch-all owner 不合理 |
| 删除 `robot/types.py` | **保留** | 内容实为 recording episode sample |
| `IKCandidateManager -> IKCandidateSearch` | **保留** | 更准确描述 generate/filter/score/search |
| `TeleopIKSolver -> OnlineIKSolver` | **保留** | deployment EE action 也消费 |
| `RealSense -> RealSenseCamera` | **保留** | 社区 camera naming 更明确 |
| `ArmWristMapper -> VRWristMapper` | **保留** | input source 明确 |
| `XHandRetargeter -> DexPilotHandRetargeter` | **保留** | 当前类具体属于 DexPilot backend |
| `control/action.py + publication.py -> command.py` | **撤销** | 最新源码已有清晰 contract/gate/publication 分层 |
| `ActionCandidate -> JointCommand` | **撤销** | candidate 在 publication 前仍是 proposal |
| `SafetyGate -> CommandSafetyChecker` | **撤销** | `Gate` 准确表达 fail-closed crossing boundary |
| `GateResult/GateRejectCode` 重命名 | **撤销** | 现有 typed gate vocabulary 已清楚 |
| `CollisionModel -> CollisionChecker` | **撤销** | 当前类真实拥有 Pinocchio geometry/model/data |
| `runtime/safety.py -> motion_state.py` | **撤销** | 模块还拥有 generation/ticket/atomic publication permit |
| `runtime/status.py -> supervisor.py` | **撤销** | `ExitReason` 是稳定独立 status vocabulary |
| `teleop/control_grid.py` 拆 controller/observation | **仍撤销** | 只整体移动到 `control_loop/grid.py`；一次 causal tick 继续单 owner |
| `recording/schema.py` 拆 schema/validation | **仍撤销** | 只整体移动到 `storage/schema.py`；raw contract 保持单一 owner |
| `deployment/worker.py` 完全保持单文件 | **修订** | 创建 `inference/` 后，将 process-local observation assembly 与 types 收到 `inference/observation.py`；`worker.py` 保留 child lifecycle/predict/publish |
| `deployment/contracts.py` 保留 | **撤销** | `Prediction` 与 `PolicyRuntime` owner 不同，应分别进入 root `prediction.py` 与 `inference/runtime.py` |
| `deployment/lifecycle.py -> session.py` | **撤销** | lifecycle 准确描述 startup/readiness/ARMED/supervision/shutdown |
| `utils/smoothing.py -> action_proposal.py` 合并 | **修订** | 移到 `teleop/control_loop/smoothing.py`，保持纯算法独立可测 |
| `examples/ -> scripts/` | **撤销** | `code_style.md` 已明确 examples 是薄入口/诊断 |
| 全量 `*Params -> *Config` | **延期** | config schema/name migration 应独立处理 |
| calibration JSON 一并迁移 | **延期** | package-data/provenance/hash 风险 |

# 5. 最终推荐 Package Tree

```text
dexmani_real/
├── config/
│   ├── __init__.py
│   ├── experiment.py
│   ├── defaults.py
│   ├── pointcloud.py
│   ├── cameras.json
│   ├── desk_plane.json
│   └── vr_transform.json
│
├── robot/
│   ├── __init__.py
│   ├── model.py
│   ├── arm_worker.py
│   ├── hand_worker.py
│   ├── command_validation.py
│   └── drivers/
│       ├── __init__.py
│       ├── xarm7.py
│       └── xhand.py
│
├── sensor/
│   ├── __init__.py
│   ├── vr_worker.py
│   ├── pointcloud.py
│   ├── pointcloud_worker.py
│   └── camera/
│       ├── __init__.py
│       ├── realsense.py
│       ├── geometry.py
│       ├── clock_sync.py
│       └── worker.py
│
├── planning/
│   ├── __init__.py
│   ├── planner.py
│   ├── collision.py
│   ├── paths.py
│   └── kinematics/
│       ├── __init__.py
│       ├── pose.py
│       ├── arm_fk.py
│       ├── hand_fk.py
│       ├── ik.py
│       ├── ik_candidates.py
│       └── fingertip.py
│
├── control/
│   ├── __init__.py
│   ├── action.py
│   ├── publication.py
│   ├── safety_gate.py
│   ├── jog.py
│   ├── arm_homing.py
│   └── hand_homing.py
│
├── ipc/
│   ├── __init__.py
│   ├── channels.py
│   ├── schema.py
│   ├── ring.py
│   ├── camera_ring.py
│   └── causal.py
│
├── runtime/
│   ├── __init__.py
│   ├── safety.py
│   ├── status.py
│   ├── processes.py
│   ├── supervisor.py
│   └── operator_input.py
│
├── teleop/
│   ├── __init__.py
│   ├── config.py
│   ├── session.py
│   ├── loop.py
│   ├── keyboard_session.py
│   ├── homing.py
│   ├── health.py
│   ├── episode_samples.py
│   ├── audio_feedback.py
│   ├── vr_transform.py
│   ├── control_loop/
│   │   ├── __init__.py
│   │   ├── grid.py
│   │   ├── action_proposal.py
│   │   ├── vr_mapping.py
│   │   ├── hand_control.py
│   │   ├── smoothing.py
│   │   ├── camera_freshness.py
│   │   └── timing.py
│   └── retargeting/
│       ├── __init__.py
│       ├── retargeter.py
│       ├── dexpilot.py
│       ├── tag_optimizer.py
│       └── pin_grad.py
│
├── deployment/
│   ├── __init__.py
│   ├── config.py
│   ├── prediction.py
│   ├── executor.py
│   ├── lifecycle.py
│   ├── operator.py
│   ├── metrics.py
│   ├── timing.py
│   └── inference/
│       ├── __init__.py
│       ├── observation.py
│       ├── runtime.py
│       ├── dexmani_policy.py
│       └── worker.py
│
├── recording/
│   ├── __init__.py
│   ├── sample.py
│   ├── frame.py
│   ├── client.py
│   ├── recorder.py
│   ├── io_worker.py
│   ├── timeline.py
│   └── storage/
│       ├── __init__.py
│       ├── schema.py
│       ├── reader.py
│       ├── hdf5_writer.py
│       ├── camera_writer.py
│       └── video.py
│
├── dataset/
│   ├── __init__.py
│   ├── contracts.py
│   ├── processed.py
│   ├── processing.py
│   ├── clean.py
│   ├── quality.py
│   ├── pointcloud.py
│   ├── transforms.py
│   └── export.py
│
├── replay/
│   ├── __init__.py
│   ├── session.py
│   ├── replayer.py
│   ├── trajectory.py
│   ├── capture.py
│   └── evaluation.py
│
├── calibration/
│   ├── __init__.py
│   ├── table.py
│   └── camera/
│       ├── __init__.py
│       ├── extrinsics.py
│       ├── solver.py
│       ├── motion.py
│       └── session.py
│
└── utils/
    ├── __init__.py
    ├── atomic_io.py
    ├── feedback.py
    ├── limits.py
    ├── log.py
    ├── rate.py
    └── serialization.py
```

## 5.1 二级 package 约束

v5 允许的二级 package **仅限**：

```text
robot/drivers
sensor/camera
planning/kinematics
teleop/control_loop
teleop/retargeting
deployment/inference
recording/storage
calibration/camera
```

其中 `teleop/` 是唯一拥有两个二级 package 的 workflow domain，因为 realtime control algorithm 与 hand retargeting 是两个独立且稳定的子系统。

不要继续建立：

```text
teleop/control_loop/observation/
deployment/inference/adapters/
recording/storage/io/
control/homing/
runtime/ipc/
dataset/validation/
core/
common/
services/
managers/
```

**规则：二级 package 是最终 namespace boundary，不作为继续分层的理由。**

## 5.2 三个新增 subsystem 的内部主链

### Teleop

```text
teleop/loop.py                     # operator / pause / recording / cadence
    ↓
teleop/control_loop/grid.py        # one causal realtime tick
    ↓
control/SafetyGate + publication
```

`teleop/retargeting/` 是 hand-retarget backend subsystem，由 workflow 构造/配置并供 control loop 使用；不要建立 `control_loop/retargeting/`。

### Deployment

```text
inference/observation.py
    → inference/runtime.py
    → inference/dexmani_policy.py
    → inference/worker.py
    → deployment/prediction.py
    → deployment/executor.py
    → control/
    → robot/
```

这里 `prediction.py` 必须位于 root：它是 model-facing inference 与 real execution 的 shared boundary。

### Recording

```text
frame/sample/client
    → io_worker.py + recorder.py
    → storage/schema.py + writers
```

`recorder.py` 是 transaction owner；`storage/` 是 persisted representation owner。不要把 `recorder.py` 下沉到 `storage/`。

# 6. Exact Path Migration Matrix

## 6.1 Robot

| Current | Target | Action |
|---|---|---|
| `dexmani_real/robot_spec.py` | `dexmani_real/robot/model.py` | move |
| `dexmani_real/robot/xarm7.py` | `dexmani_real/robot/drivers/xarm7.py` | move |
| `dexmani_real/robot/xhand.py` | `dexmani_real/robot/drivers/xhand.py` | move |
| `dexmani_real/robot/types.py` | `dexmani_real/recording/sample.py` | split/move then delete |

保持 `robot/arm_worker.py`、`robot/hand_worker.py`、`robot/command_validation.py`。其中 `command_validation.py` 虽短，但它是 SDK crossing 前的 final guard，不得并入 worker。

## 6.2 Sensor

| Current | Target |
|---|---|
| `sensor/realsense.py` | `sensor/camera/realsense.py` |
| `sensor/camera_geometry.py` | `sensor/camera/geometry.py` |
| `sensor/clock_sync.py` | `sensor/camera/clock_sync.py` |
| `sensor/camera_worker.py` | `sensor/camera/worker.py` |

保持 `sensor/vr_worker.py`、`sensor/pointcloud.py`、`sensor/pointcloud_worker.py`。

`sensor/camera/__init__.py` 不得 import `realsense.py`，避免 `import dexmani_real.sensor.camera` 隐式要求 `pyrealsense2`。

## 6.3 Planning

| Current | Target |
|---|---|
| `planning/poses.py` | `planning/kinematics/pose.py` |
| `planning/arm_fk.py` | `planning/kinematics/arm_fk.py` |
| `planning/hand_fk.py` | `planning/kinematics/hand_fk.py` |
| `planning/ik.py` | `planning/kinematics/ik.py` |
| `planning/candidates.py` | `planning/kinematics/ik_candidates.py` |
| `planning/fingertip.py` | `planning/kinematics/fingertip.py` |
| `planning/types.py` | distributed to owner modules | delete |

保持 `planning/planner.py`、`planning/collision.py`、`planning/paths.py`。

不要把 `arm_fk.py` / `hand_fk.py` 简写成 `arm.py` / `hand.py`；FK 是标准 robotics vocabulary，且明确说明职责。

## 6.4 Control

| Current | Target |
|---|---|
| `control/arm_home.py` | `control/arm_homing.py` |
| `control/hand_home.py` | `control/hand_homing.py` |
| `teleop/keyboard.py` 中 Cartesian key mapping | `control/jog.py` |

保持：

```text
control/action.py
control/safety_gate.py
control/publication.py
```

不要创建 `control/command.py` 或把 `safety_gate.py` 泛化成 `safety.py`。

## 6.5 Runtime

| Current | Target | Note |
|---|---|---|
| `runtime/workers.py` | `runtime/processes.py` | process construction/shutdown |
| `runtime/workers.py::supervisor_exit_reason` | `runtime/supervisor.py` | move function |
| `teleop/keyboard.py` 中 keyboard capture | `runtime/operator_input.py` | cross-workflow input |

保持 `runtime/safety.py`、`runtime/status.py`、`runtime/supervisor.py`。

## 6.6 Teleop

### Workflow root 保留

```text
teleop/config.py
teleop/session.py
teleop/loop.py
teleop/keyboard_session.py
teleop/homing.py
teleop/health.py
teleop/episode_samples.py
teleop/audio_feedback.py
teleop/vr_transform.py
```

这些模块属于 operator/workflow/lifecycle/recording orchestration，而不是 realtime control algorithm。

### 新建 `teleop/control_loop/`

| Current | Target |
|---|---|
| `teleop/control_grid.py` | `teleop/control_loop/grid.py` |
| `teleop/action_proposal.py` | `teleop/control_loop/action_proposal.py` |
| `teleop/arm_mapper.py` | `teleop/control_loop/vr_mapping.py` |
| `teleop/hand_control.py` | `teleop/control_loop/hand_control.py` |
| `utils/smoothing.py` | `teleop/control_loop/smoothing.py` |
| `teleop/camera_freshness.py` | `teleop/control_loop/camera_freshness.py` |
| `teleop/timing.py` | `teleop/control_loop/timing.py` |

`control_grid.py` 只是**整体移动并改名为 `grid.py`**，不要再拆 controller/observation/publication 文件。它继续拥有一个 causal teleop tick 的 vertical invariant。

### Retargeting

```text
teleop/retarget/ → teleop/retargeting/
teleop/retarget/facade.py → teleop/retargeting/retargeter.py
```

### 其他 ownership cleanup

```text
teleop/safety.py
    → teleop/homing.py
    → teleop/health.py

teleop/recording_session.py
    → merge into teleop/loop.py

teleop/keyboard.py
    → runtime/operator_input.py
    → control/jog.py
```

完成后删除旧 `teleop/keyboard.py`、`teleop/safety.py`、`teleop/recording_session.py`、`teleop/retarget/`。

## 6.7 Deployment

新结构明确分为：

```text
model-facing inference subsystem
        ↓
Prediction boundary
        ↓
physical execution subsystem
```

### 新建 `deployment/inference/`

| Current | Target | Ownership |
|---|---|---|
| `deployment/observation.py` | `deployment/inference/observation.py` | process-local causal observation contract + assembly |
| `deployment/worker.py` | `deployment/inference/worker.py` | spawned child lifecycle, runtime load/warmup/reset/predict/publish |
| `integrations/dexmani_policy.py` | `deployment/inference/dexmani_policy.py` | concrete `PolicyRuntime` adapter |
| `deployment/contracts.py::PolicyRuntime` | `deployment/inference/runtime.py` | model runtime protocol |

`deployment/inference/observation.py` 可以同时拥有当前 `observation.py` 的 dataclasses 与从旧 `worker.py` 迁出的 **process-local observation read/alignment/build helpers**。不要继续拆 `history.py`、`alignment.py`、`builder.py`。

### Root prediction boundary

```text
deployment/contracts.py::Prediction
    → deployment/prediction.py
```

`Prediction` 留在 deployment root，因为它是：

```text
inference/worker.py
        ↓
Prediction
        ↓
executor.py
```

的双侧共享 contract，而不是 inference 私有类型。

### Root 保持

```text
deployment/config.py
deployment/executor.py
deployment/lifecycle.py
deployment/operator.py
deployment/metrics.py
deployment/timing.py
```

完成后删除：

```text
deployment/contracts.py
integrations/
deployment/worker.py
deployment/observation.py
```

动态字符串 `FIXED_POLICY_RUNTIME_TARGET` 必须更新到新的 `deployment.inference.dexmani_policy` 路径。

## 6.8 Recording

新结构分离：

```text
recording runtime / transaction
        ↓
recording/storage persisted artifact layer
```

### Root 保持 runtime/transaction owner

```text
recording/sample.py
recording/frame.py
recording/client.py
recording/recorder.py
recording/io_worker.py
recording/timeline.py
```

其中：

- `recorder.py` 是 episode transaction coordinator，不属于 storage；
- `io_worker.py` 是 process/runtime owner，不属于 storage；
- `frame.py` 是 in-memory/shared recording boundary；
- `timeline.py` 是 temporal alignment/sequence semantics。

### 新建 `recording/storage/`

| Current | Target |
|---|---|
| `recording/schema.py` | `recording/storage/schema.py` |
| `recording/reader.py` | `recording/storage/reader.py` |
| `recording/hdf5_writer.py` | `recording/storage/hdf5_writer.py` |
| `recording/camera_writer.py` | `recording/storage/camera_writer.py` |
| `recording/video.py` | `recording/storage/video.py` |

`schema.py` **整体移动，不拆分**。它继续是 raw persisted contract 的唯一 owner。

另：

```text
recording/worker.py → recording/io_worker.py
robot/types.py → recording/sample.py
```

依赖方向必须是：

```text
recording/recorder.py
        ↓
recording/storage/*
```

而不是 storage 反向依赖 recorder/client/io_worker。

## 6.9 Dataset

整个 package：

```text
data/ → dataset/
```

其中：

| Current | Target |
|---|---|
| `data/process.py` | `dataset/processing.py` + `dataset/processed.py` |
| `data/raw_pointcloud.py` | `dataset/pointcloud.py` |
| `data/contracts.py` | `dataset/contracts.py` |
| `data/clean.py` | `dataset/clean.py` |
| `data/quality.py` | `dataset/quality.py` |
| `data/transforms.py` | `dataset/transforms.py` |
| `data/export.py` | `dataset/export.py` |

`processed.py` 统一拥有：

```text
PROCESSED_SCHEMA_NAME
PROCESSED_SCHEMA_VERSION
ProcessedProvenance
processed HDF5 dataset specifications
processed semantic attrs/constants
strict processed attr readers
validate_processed_payload()
validate_processed_provenance()
processed structure validation helpers
```

`processing.py` 只保留 raw episode → processed artifact transaction。这样 `export.py` 不再从 `processing.py` import 私有 validation constants/helper。

## 6.10 Replay

```text
replay/controller.py → replay/replayer.py
```

保持 `EpisodeReplayer`、`ReplayStatus`、`ReplayOutcome` 以及 `session.py`、`trajectory.py`、`capture.py`、`evaluation.py`。不要把 `capture.py` 并入 replayer。

## 6.11 Calibration

| Current | Target |
|---|---|
| `config/camera_calib.py` | `calibration/camera/extrinsics.py` |
| `calibration/camera/control.py` | `calibration/camera/motion.py` |

保持 `solver.py`、`session.py`、`calibration/table.py`。

本任务**不移动** `config/cameras.json`、`config/desk_plane.json`、`config/vr_transform.json`，也不修改 JSON 内容。`CameraExtrinsics` 移动后必须继续读取同一份 `config/cameras.json`，source bytes 与 SHA-256 不变。

## 6.12 Config

推荐：

```text
config/runtime.py → config/experiment.py
ResolvedRuntimeConfig → ExperimentConfig
resolve_runtime_config → resolve_experiment_config
```

理由：该对象实际是 `CLI/file/defaults → immutable validated experiment snapshot → canonical JSON/YAML → SHA-256 identity`，被 teleop/deployment/replay/calibration/dataset 共同消费。

但本任务不做 config schema 重构；保持 YAML section/key：

```text
arm
hand
policy
keyboard_teleop
vr
safety
camera
pointcloud
tag_retargeting
dexpilot_retargeting
environment
```

---

# 7. Class Rename Matrix

## 7.1 应执行

| Current | Target | Reason |
|---|---|---|
| `ResolvedRuntimeConfig` | `ExperimentConfig` | canonical experiment snapshot |
| `IKCandidateManager` | `IKCandidateSearch` | generate/filter/score/search，不是 generic manager |
| `TeleopIKSolver` | `OnlineIKSolver` | deployment EE action 也使用 |
| `TeleopProfile` | `OnlineIKConfig` | same reason |
| `PlanningProfile` | `MotionPlanningConfig` | motion planning config |
| `RealSense` | `RealSenseCamera` | community camera naming |
| `RealSenseConfig` | `RealSenseCameraConfig` | explicit backend config |
| `CameraFrame` | `RGBDFrame` | payload actually RGB-D |
| `ArmWristMapper` | `VRWristMapper` | input semantics |
| `XHandRetargeter` | `DexPilotHandRetargeter` | implementation belongs to DexPilot backend |
| `DexManiPolicyRuntime` | `DexManiPolicyAdapter` | adapter, not model runtime owner |
| `PolicyWorkerConfig` | `InferenceWorkerConfig` | specifically inference child |
| `WorkerSpec` | `ProcessSpec` | process descriptor |
| `RobotState` | `EpisodeState` | recording sample aggregate |
| `RobotAction` | `EpisodeAction` | recording sample aggregate |
| `CameraCalib` | `CameraExtrinsics` | module owns extrinsics |
| `CameraCalibEntry` | `CameraExtrinsicsEntry` | same |
| `ControlSignal` | `OperatorCommand` | used by multiple workflows |
| `KeyboardHandler` | `KeyboardInput` | external input owner |
| `GlobalKeyState` | `KeyboardState` | held-key state |
| `eef_delta_from_keys` | `compute_cartesian_jog_delta` | pure Cartesian jog mapping |

## 7.2 明确保留

```text
ActionCandidate
SafetyGate
GateResult
GateRejectCode
CollisionModel

XArm7MotionPlanner
XArm7Kinematics
ArmFK
Pose
IKResult
PathResult
XArm7PlannerConfig

XArm7
XHand

RuntimeChannels
RuntimeChannelsConfig
SafetyState

TeleopController
CameraFreshnessTracker
StageTimer

PolicyRuntime
Prediction
PolicyObservation
PolicyExecutor
PolicyDeploymentConfig
FingertipAssemblerConfig

RecorderIOConfig
RecorderClient
EpisodeRecorder
EpisodeReader

EpisodeReplayer
ReplayStatus
ReplayOutcome
```

特别不要执行：

```text
Prediction → PolicyPrediction
CollisionModel → CollisionChecker
SafetyGate → CommandSafetyChecker
ActionCandidate → JointCommand
```

---

# 8. 必须拆分 / 合并 / 删除

## 8.1 `planning/types.py`

拆到真实 owner：

```text
CollisionPair / CollisionInfo
    → planning/collision.py
Pose
    → planning/kinematics/pose.py
IKResult / IKFailureKind
    → planning/kinematics/ik.py
PathResult
    → planning/paths.py
XArm7PlannerConfig / MotionPlanningConfig
    → planning/planner.py
OnlineIKConfig
    → planning/kinematics/ik.py
```

完成后删除 `planning/types.py`。

## 8.2 `robot/types.py`

```text
RobotState  → EpisodeState
RobotAction → EpisodeAction
```

移动到 `recording/sample.py`，然后删除 `robot/types.py`。

## 8.3 `teleop/safety.py`

拆为：

```text
teleop/homing.py
teleop/health.py
```

不要叫 `teleop/feedback.py`，避免与 `utils/feedback.py` basename 冲突。

## 8.4 `teleop/keyboard.py`

拆为：

```text
runtime/operator_input.py
    OperatorCommand
    KeyboardInput
    KeyboardState
    pynput lifecycle / terminal handling

control/jog.py
    compute_cartesian_jog_delta()
```

完成后删除 `teleop/keyboard.py`。

## 8.5 `runtime/workers.py`

重命名为 `runtime/processes.py`，保留 process construction/shutdown owner；将 `supervisor_exit_reason()` 移到 `runtime/supervisor.py`。完成后删除旧 module。

## 8.6 `deployment/contracts.py`

这是 v5 新增的明确拆分：

```text
Prediction
    → deployment/prediction.py

PolicyRuntime
    → deployment/inference/runtime.py
```

两个对象跨越不同边界，不应继续放在 generic `contracts.py`。

## 8.7 `deployment/worker.py` 与 `deployment/observation.py`

创建 `deployment/inference/` 后：

```text
observation dataclasses + causal observation read/alignment/build
    → deployment/inference/observation.py

spawned child lifecycle + model load/warmup/reset/predict/publish
    → deployment/inference/worker.py
```

这是**一次职责拆分**，但不得进一步拆 history/alignment/builder。

## 8.8 `data/process.py`

package rename 后拆为：

```text
dataset/processing.py
dataset/processed.py
```

`processed.py` 拥有 processed schema/provenance/spec/validation；`processing.py` 只拥有 raw → processed transaction。

## 8.9 `teleop/recording_session.py`

合并进 `teleop/loop.py`，然后删除。

## 8.10 只移动、不拆的关键文件

```text
teleop/control_grid.py
    → teleop/control_loop/grid.py

recording/schema.py
    → recording/storage/schema.py
```

二者 owner 仍完整，不因新 package 再次拆分。

## 8.11 最终应消失的旧路径

```text
planning/types.py
robot/types.py
teleop/safety.py
teleop/keyboard.py
teleop/recording_session.py
teleop/retarget/
runtime/workers.py
deployment/contracts.py
deployment/observation.py
deployment/worker.py
integrations/
recording/worker.py
recording/schema.py
recording/reader.py
recording/hdf5_writer.py
recording/camera_writer.py
recording/video.py
data/
```

这里的 recording 文件均为 move 到 `recording/storage/` 或 `io_worker.py`，不是功能删除。

# 9. 明确“不拆”的文件

不要依据行数机械拆分。v5 最终结构下，下列文件继续保持完整 owner：

```text
teleop/control_loop/grid.py
teleop/control_loop/action_proposal.py

deployment/executor.py
deployment/inference/observation.py
deployment/inference/worker.py
deployment/lifecycle.py

recording/recorder.py
recording/storage/schema.py

planning/kinematics/ik.py
planning/collision.py
planning/kinematics/ik_candidates.py

dataset/clean.py
sensor/camera/realsense.py
teleop/keyboard_session.py
```

关键理由：

- `control_loop/grid.py`：一个 causal teleop tick 的 vertical owner；
- `executor.py`：prediction decode/schedule/EE→IK/shaping/safety/publication/progress watchdog 的 execution state machine；
- `inference/observation.py`：process-local model observation 的唯一 owner，允许包含 types + read/alignment/build；
- `recording/recorder.py`：episode transaction coordinator；
- `storage/schema.py`：raw persisted contract single source of truth；
- `realsense.py`：单一真实 camera backend 的完整 device lifecycle/config/frame contract。

新建二级 package 的目的不是创造更多可拆分层级，而是让这些完整 owner 获得正确 namespace。

# 10. Dependency Law

总体方向：

```text
robot/model + planning/kinematics + ipc/schema
        │
        ▼
robot/drivers + sensor + ipc + runtime
        │
        ▼
planning + control
        │
        ▼
teleop / deployment / replay / calibration
        │
        ├──→ recording
        │
        ▼
offline dataset
```

## 10.1 Workflow 之间禁止反向 import

禁止：

```text
deployment → teleop
replay → teleop
calibration → teleop
teleop → deployment
teleop → replay
```

shared operator input 必须通过 `runtime/operator_input.py` / `control/jog.py` 解决。

## 10.2 Teleop 内部依赖

推荐：

```text
teleop/loop.py
    ├──→ teleop/control_loop/*
    └──→ teleop/retargeting/*
```

`control_loop/` 不应反向依赖 `loop.py/session.py/keyboard_session.py`。不要为了消除少量参数传递建立 teleop manager/service。

## 10.3 Deployment 内部依赖

目标：

```text
deployment/inference/observation.py
        ↓
deployment/inference/runtime.py
        ↓
deployment/inference/dexmani_policy.py
        ↓
deployment/inference/worker.py
        ↓
deployment/prediction.py
        ↓
deployment/executor.py
```

允许 `inference/*` 依赖 `deployment/config.py` 和 `deployment/prediction.py`；禁止 inference 依赖 `executor.py`、`operator.py`、`lifecycle.py`。

`executor.py` 可以依赖 `prediction.py`，但不得 import concrete policy adapter/Torch。

## 10.4 Recording 内部依赖

目标：

```text
client/frame/sample
       ↓
io_worker + recorder
       ↓
storage/*
```

`recording/storage/` 可以互相依赖 schema/read/write helpers，但禁止反向 import：

```text
storage → recorder
storage → io_worker
storage → client
```

`dataset/` 应从 `recording/storage/reader.py` / `schema.py` 消费 persisted artifact contract，而不是依赖 RecorderIO runtime。

## 10.5 Control 与 Recording 的顶层规则

- `control/` 不 import `teleop/deployment/replay/calibration`；
- `recording/` 不拥有 workflow action decision；
- `dataset/` 是 offline-only，不依赖 hardware worker 或 deployment executor。

## 10.6 Backend dependency isolation

`__init__.py` 默认 empty/docstring-only，不通过 package root 隐式加载：

```text
pyrealsense2
vendor robot SDK
Torch / CUDA
MPlib / Pinocchio
```

内部 import 优先 explicit full path，例如：

```python
from dexmani_real.teleop.control_loop.grid import TeleopController
from dexmani_real.deployment.inference.observation import PolicyObservation
from dexmani_real.recording.storage.reader import EpisodeReader
```

# 11. Latest Behavior Invariants — Codex 不得改变

这是整个任务的硬边界。

## 11.1 Action semantics

保持 teleop arm：

```text
joint bound
→ wrap-aware producer delta shaping
→ safety gate
→ publication
→ arm worker final guard
```

保持 learned-policy arm：

```text
decode
→ wrap/canonical representation
→ absolute joint-limit admission
→ producer-side neural-spike clip
→ workspace/safety
→ publication
→ arm worker final guard
```

保持 learned-policy hand：

```text
decode
→ reject-only endpoint jump gate
→ publication
→ hand worker SDK-level slew
```

不得把 clip、shaping、reject、worker slew 统一成一个 helper 或一个 gate。

## 11.2 Safety state / command identity

不得改变：

```text
SafetyState
StopRequest
run_generation
motion_lock
CoupledCommandTicket
latest-wins command ownership
valid_until_monotonic_ns
generation invalidation
```

不得修改 state transition semantics。

## 11.3 Hardware SDK ownership

必须保持：

```text
xArm SDK object
    → arm driver/worker process only

XHand SDK object
    → hand driver/worker process only

RealSense device/pipeline
    → camera worker process only

Policy Torch/CUDA runtime
    → inference child only
```

不得因为 import 整理把重型对象提前创建到 parent process。

## 11.4 Teleop pause / re-anchor

保持：

```text
pause boundary
→ invalidate old generation/reference
→ command silent
→ require fresh post-pause causal feedback
→ re-anchor
→ resume
```

## 11.5 Table collision policy

保持已确认设计：

```text
teleop:
    table collision NOT action reject criterion

replay main trajectory:
    table collision NOT reject criterion

return-home / explicit table-aware path:
    retain table safety
```

## 11.6 XHand startup

保持：

```text
worker ready
≠
home position reached
```

XHand startup 不发送隐式 home target。

## 11.7 Data schema

保持：

```text
raw v24
processed HDF5 v12
Policy Zarr v6
```

不得改 field name、dtype、shape、semantic attr、unit、coordinate frame、schema version、manifest semantics、raw/processed provenance meaning。本任务只允许 Python import/module path 改变。

## 11.8 Config serialization

同一份 defaults + YAML + CLI 输入必须继续产生等价的：

```text
canonical_json
canonical_yaml
sha256
```

`ResolvedRuntimeConfig -> ExperimentConfig` 不得改变 serialized payload。

---

# 12. Codex Execution Protocol

## 12.1 启动检查

主代理先执行：

```bash
git status --short
git rev-parse HEAD
```

如果 HEAD 不是：

```text
f3e3cc064cb5537d605a06c383263ec776164c6c
```

则必须：

1. 查看 `f3e3cc..HEAD`；
2. 找出是否修改本方案涉及的 path/symbol；
3. 对受影响 phase 重新 audit；
4. 只在结论仍成立时继续。

不得因为本文给出 target tree 就覆盖更新后的设计。

## 12.2 必读文件

修改前：

```text
AGENTS.md
code_style.md
repo_map.md
user_design.md
docs/action_clip_mechanisms.md
```

涉及 dataset 时读 `docs/data_schema.md`；涉及 deployment 时读 `docs/policy_deployment.md`。

## 12.3 使用本文驱动 workflow

本文是 namespace refactor 的唯一 phase brief。用户在 v5 review 后要求增加自动推进、phase acceptance
与 compact checkpoint；具体执行循环见第 22 节，持久状态只记录在
`docs/dexmani_real_codex_namespace_refactor_progress.md`。不要创建第二套 workflow framework。

## 12.4 Baseline validation

任何 edit 前：

```bash
conda run -n real_robot python -m compileall -q dexmani_real examples
conda run -n real_robot python -m pytest -q
```

baseline 如果失败：

- 判断与当前 scope 是否相关；
- 不顺手修 unrelated failure；
- 如果 failure 阻止行为等价验证，停止并报告。

## 12.5 Symbol inventory

每个 phase 开始前：

```bash
rg "<old module path|old class name|old function name>" \
  dexmani_real examples tests docs README.md repo_map.md code_style.md user_design.md
```

必须覆盖 production、tests、examples、docs、dynamic import strings、`FIXED_*_TARGET`。

尤其注意 `FIXED_POLICY_RUNTIME_TARGET`，IDE/static rename 可能漏掉。

---

# 13. Codex Agent 分工

## 主代理

拥有：

```text
phase order
scope
cross-package dependency judgment
final diff review
test acceptance
stop decision
```

## terra-xhigh

优先用于：

```text
package/file move
import rewrite
planning types co-location
sensor camera namespace
teleop namespace cleanup
dataset package rename/split
recording/replay/calibration mechanical rename
config experiment mechanical rename
```

## sol-high

用于以下 boundary 的修改后 read-only review：

```text
control/
runtime/safety
runtime/process lifecycle
deployment executor/inference boundary
hardware worker import boundary
operator e-stop path
```

本任务原则上不改这些模块逻辑；`sol-high` 负责确认 rename 没有改变 safety semantics。

## luna-max

用于：

```text
rg inventory
remaining old import search
documentation path update
focused compile/test
stale-file detection
```

## 并行规则

可并行 read-only inventory、disjoint docs、non-overlapping audit。不要让两个 agent 同时修改同一个 package/import graph。shared module 只能有一个 write owner。

---

# 14. Recommended Phase Plan

## Phase 0 — Freeze Baseline

### Goal

确认基线、tests、invariants、现有 import graph。

### Actions

```text
read required docs
git status
git rev-parse HEAD
baseline compile
baseline pytest
inventory old symbols
```

### Acceptance

```text
baseline understood
no unrelated user changes overwritten
test state recorded
```

---

## Phase 1 — Planning + Robot Model Ownership

### Scope

```text
planning/
robot_spec.py
robot/types.py
robot/
recording/sample.py
```

### Actions

1. 创建 `planning/kinematics/`、`robot/drivers/`、`recording/sample.py`、`robot/model.py`。
2. 移动 planning modules。
3. 消除 `planning/types.py`。
4. 移动 xArm/XHand drivers。
5. `robot_spec.py -> robot/model.py`。
6. `robot/types.py -> recording/sample.py`。
7. class rename：
   ```text
   IKCandidateManager -> IKCandidateSearch
   TeleopIKSolver -> OnlineIKSolver
   TeleopProfile -> OnlineIKConfig
   PlanningProfile -> MotionPlanningConfig
   RobotState -> EpisodeState
   RobotAction -> EpisodeAction
   ```

### Must preserve

```text
MPlib behavior
IK candidate scoring/order
collision pair semantics
URDF/SRDF paths
joint order
recorded data values
```

### Validation

```bash
conda run -n real_robot python -m compileall -q dexmani_real examples
conda run -n real_robot python -m pytest -q
git diff --check
rg "planning\.types|robot_spec|robot\.types|IKCandidateManager|TeleopIKSolver|TeleopProfile|PlanningProfile" .
```

---

## Phase 2 — Sensor Camera Namespace

### Actions

创建 `sensor/camera/`，移动 camera backend/geometry/clock/worker，并 rename：

```text
RealSense → RealSenseCamera
RealSenseConfig → RealSenseCameraConfig
CameraFrame → RGBDFrame
```

### Must preserve

```text
RealSense lifecycle
camera timestamp mapping
RGB-D alignment
camera generation/reset semantics
shared-memory layout
```

父 package 不 re-export RealSense backend。

---

## Phase 3 — Runtime / Operator Input / Shared Control Naming

### Actions

```text
runtime/workers.py → runtime/processes.py
WorkerSpec → ProcessSpec
supervisor_exit_reason → runtime/supervisor.py

teleop/keyboard.py
    → runtime/operator_input.py
    → control/jog.py

control/arm_home.py → control/arm_homing.py
control/hand_home.py → control/hand_homing.py

teleop/safety.py
    → teleop/homing.py
    → teleop/health.py
```

rename：

```text
ControlSignal → OperatorCommand
KeyboardHandler → KeyboardInput
GlobalKeyState → KeyboardState
eef_delta_from_keys → compute_cartesian_jog_delta
```

### Critical review

此 phase 涉及 e-stop input、stop/quit callbacks、process lifecycle imports、homing imports。完成后必须由 `sol-high` 做 read-only safety review。

### Forbidden

不得改变 keyboard edge/debounce、ESC latch、callback timing、homing protocol、SafetyState transition、worker stop order。

---

## Phase 4 — Teleop Subsystems

### Actions

创建两个明确 subsystem：

```text
teleop/control_loop/
teleop/retargeting/
```

执行：

```text
control_grid.py → control_loop/grid.py
action_proposal.py → control_loop/action_proposal.py
arm_mapper.py → control_loop/vr_mapping.py
hand_control.py → control_loop/hand_control.py
utils/smoothing.py → control_loop/smoothing.py
camera_freshness.py → control_loop/camera_freshness.py
timing.py → control_loop/timing.py

retarget/ → retargeting/
facade.py → retargeting/retargeter.py
ArmWristMapper → VRWristMapper
XHandRetargeter → DexPilotHandRetargeter
recording_session.py → merge into loop.py
```

保持 `grid.py` 内部 vertical control tick，不再拆 observation/controller/publication。

### Forbidden

不得修改 EMA、hand ramp、retarget failure semantics、pause/re-anchor、control cadence 或 SafetyGate/publication 行为。

---

## Phase 5 — Deployment Inference Boundary

### Actions

创建：

```text
deployment/inference/
```

执行：

```text
deployment/contracts.py::Prediction
    → deployment/prediction.py

deployment/contracts.py::PolicyRuntime
    → deployment/inference/runtime.py

deployment/observation.py
    → deployment/inference/observation.py

deployment/worker.py
    → deployment/inference/worker.py

integrations/dexmani_policy.py
    → deployment/inference/dexmani_policy.py

DexManiPolicyRuntime → DexManiPolicyAdapter
PolicyWorkerConfig → InferenceWorkerConfig
```

将旧 `worker.py` 中属于 causal observation read/alignment/build 的 process-local helpers 移到 `inference/observation.py`；worker 只保留 spawned inference child lifecycle、runtime load/warmup/reset/predict/publish。

更新 `FIXED_POLICY_RUNTIME_TARGET`、spawn imports、tests、docs，删除 `deployment/contracts.py`、旧 observation/worker 与整个 `integrations/`。

### Must preserve

```text
inference child is only model/Torch/CUDA owner
parent does not import Policy/Torch
PolicyObservation tensor contract unchanged
Prediction IPC unchanged
sync/async scheduling unchanged
arm clip / hand reject semantics unchanged
```

完成后需要 `sol-high` 对 inference→Prediction→executor boundary 做 read-only review。

---

## Phase 6 — Recording Storage + Replay + Calibration

### Recording

创建：

```text
recording/storage/
```

移动：

```text
recording/schema.py → recording/storage/schema.py
recording/reader.py → recording/storage/reader.py
recording/hdf5_writer.py → recording/storage/hdf5_writer.py
recording/camera_writer.py → recording/storage/camera_writer.py
recording/video.py → recording/storage/video.py
recording/worker.py → recording/io_worker.py
```

保持 `recording/recorder.py` 在 root，作为 transaction coordinator；`storage/schema.py` 整体移动，不拆 validation/metadata。

### Replay

```text
replay/controller.py → replay/replayer.py
```

### Calibration

```text
config/camera_calib.py → calibration/camera/extrinsics.py
CameraCalib → CameraExtrinsics
CameraCalibEntry → CameraExtrinsicsEntry
calibration/camera/control.py → calibration/camera/motion.py
```

### Resource invariant

保持 `config/cameras.json` bytes/path semantics 不变；default extrinsics loader 仍解析同一 artifact，source SHA-256 不变。

---

## Phase 7 — Dataset Namespace + Processed Contract Owner

### Actions

```text
data/ → dataset/
raw_pointcloud.py → pointcloud.py
process.py → processing.py + processed.py
```

`processed.py` 统一拥有 processed schema/version/provenance/spec/validation；`processing.py` 只拥有 raw → processed transaction。

### Must preserve

```text
raw v24
processed v12
Policy Zarr v6
```

禁止 rename HDF5 fields、`data.h5`、dataset order、drop semantics、quality thresholds、export schema。

---

## Phase 8 — Experiment Config Naming

### Actions

```text
config/runtime.py → config/experiment.py
ResolvedRuntimeConfig → ExperimentConfig
resolve_runtime_config → resolve_experiment_config
```

### Must preserve

完全相同的：

```text
CLI > file > defaults precedence
validation
canonical_json
canonical_yaml
sha256
YAML section names
```

本 phase 不改 `ArmParams`、`HandParams`、`PolicyParams`、`SafetyParams` 等。

---

## Phase 9 — Documentation / Examples / Hygiene

更新：

```text
README.md
code_style.md
repo_map.md
user_design.md       # only necessary path references
docs/*.md
examples/*.py
tests/*.py
```

`code_style.md` 将旧 `data/` / `integrations/` 描述更新为 `dataset/` / `deployment/inference/dexmani_policy.py`，并补充 `teleop/control_loop/`、`recording/storage/` 的稳定职责；但不要把 style doc 变成逐文件 inventory。

可选诊断文件 rename（仅在确认仍存在时）：

```text
realsense_record_example.py → diagnose_realsense.py
pointcloud_process_example.py → diagnose_pointcloud.py
xhand_control_example.py → diagnose_xhand.py
```

这不是 Definition of Done 的必要条件。


---

# 15. Validation Matrix

每个 phase 至少执行：

```bash
conda run -n real_robot python -m compileall -q dexmani_real examples
conda run -n real_robot python -m pytest -q
git diff --check
git status --short
```

同时检查 focused diff：

```bash
git diff --stat
git diff -- <phase-scope>
```

## 15.1 不运行 hardware

禁止运行：

```text
teleop physical control
home
physical replay
camera calibration write
xArm/XHand connection
RealSense connection
policy execute=True
```

除非用户对本次 Codex 执行明确授权实机验证。

## 15.2 不添加 implementation-shape-only tests

不要为了 rename 加大量只断言 module/class path 的测试。

应优先更新现有 behavior tests 的 imports。只有下列 namespace-refactor 风险值得新增 focused regression：

```text
dynamic import target
resource path identity
backend-heavy import isolation
config canonical identity
camera calibration source identity
```

---

# 16. Cross-Phase Invariants

最终 diff 中不得出现：

```text
numerical threshold changes
control frequency changes
timeout changes
joint limit changes
workspace changes
collision policy changes

new clipping
removed clipping
clip → reject semantic change
reject → clip semantic change

new fallback
new retry
new multiprocessing mechanism
new device lifecycle

new persistence format
new schema field
new config YAML key
```

如果为了完成 rename 发现必须改变上述任何一项：

```text
STOP
```

作为独立问题报告，不要把逻辑修改藏在 namespace refactor 中。

---

# 17. Stop Conditions

Codex 必须停止当前 phase，而不是猜测或扩大 scope，当出现：

1. `HEAD` 在相关文件上发生新的未 review 更新；
2. baseline tests 出现相关失败且无法证明与 refactor 无关；
3. 发现外部仓库通过旧 Python module path 直接 import 本包，而用户明确要求兼容；
4. rename 需要改变 persisted schema 或 YAML key；
5. 需要改变 action shaping/reject/safety 语义才能让 tests 通过；
6. 某 SDK import/constructor 在新 package import 时产生额外 side effect；
7. calibration resource move会改变 artifact identity；
8. DexMani Policy public contract 与当前假设不一致；
9. 只有真实硬件才能判断行为是否等价；
10. 两个 owner 对同一 mutable state 的职责无法从源码确认。

停止时报告：

```text
evidence
affected files
why current plan cannot safely decide
smallest next investigation
```

不要添加 compatibility shim、temporary alias 或 fallback 绕过问题。

---

# 18. Definition of Done

## 18.1 Structure

```text
target package tree reached
only the eight approved second-level packages exist
no stale obsolete package
no accidental deeper subsystem package
```

## 18.2 Old import/path cleanup

以下旧路径不再作为 active production/test/example import：

```text
dexmani_real.robot_spec
dexmani_real.robot.xarm7
dexmani_real.robot.xhand
dexmani_real.robot.types

dexmani_real.planning.types
dexmani_real.planning.arm_fk
dexmani_real.planning.hand_fk
dexmani_real.planning.poses
dexmani_real.planning.ik
dexmani_real.planning.candidates

dexmani_real.sensor.realsense
dexmani_real.sensor.camera_geometry
dexmani_real.sensor.clock_sync
dexmani_real.sensor.camera_worker

dexmani_real.teleop.control_grid
dexmani_real.teleop.action_proposal
dexmani_real.teleop.arm_mapper
dexmani_real.teleop.hand_control
dexmani_real.teleop.camera_freshness
dexmani_real.teleop.timing
dexmani_real.teleop.keyboard
dexmani_real.teleop.safety
dexmani_real.teleop.retarget
dexmani_real.teleop.recording_session

dexmani_real.runtime.workers

dexmani_real.deployment.contracts
dexmani_real.deployment.observation
dexmani_real.deployment.worker
dexmani_real.integrations

dexmani_real.recording.worker
dexmani_real.recording.schema
dexmani_real.recording.reader
dexmani_real.recording.hdf5_writer
dexmani_real.recording.camera_writer
dexmani_real.recording.video

dexmani_real.data
dexmani_real.replay.controller
dexmani_real.config.runtime
dexmani_real.config.camera_calib
```

历史文档若明确讨论旧版本可以保留旧路径字符串，但不得继续作为 current architecture guidance。

## 18.3 Behavior

```text
full pytest passes
compileall passes
config canonical identity unchanged
schema versions unchanged
action clip/reject semantics unchanged
hardware SDK ownership unchanged
```

## 18.4 Hygiene

```text
git diff --check passes
no wildcard compatibility reexports
no new generic manager/service/factory abstraction
no temporary aliases
no unrelated formatting churn
```

---

# 19. Explicitly Deferred Work

以下事项**不要混入本次 Codex task**。

## 19.1 Config schema redesign

当前 `PolicyParams` 仍然混合 deployment timing、teleop、recording、IK、workspace、hand retarget settings。

这是一个真实设计问题，但解决它需要：

```text
YAML schema migration
canonical hash consideration
many access-path changes
experiment config compatibility
```

应作为独立设计任务处理。

## 19.2 `SafetyParams` 重命名

它当前更接近 runtime health / supervisor config，而不是 physical safety。

未来可以研究：

```text
SafetyParams → RuntimeHealthConfig
```

或：

```text
SupervisorConfig
```

但因为 YAML section 仍叫 `safety`，本次不动。

## 19.3 Calibration JSON relocation

未来可考虑：

```text
config/cameras.json
    → calibration/camera/cameras.json
```

以及 VR/table calibration artifact 的统一目录。

但当前 package data、provenance、hash、default path 依赖旧位置，本次不动。

## 19.4 Generic plugin / adapter framework

当前只有一个固定 DexMani Policy integration。

不要为假设中的未来 backend 增加：

```text
adapter registry
entry-point plugin
factory
dependency injection
abstract service
```

## 19.5 Safety/control algorithm redesign

本任务不研究：

```text
action clip threshold
hand jump threshold
worker slew
IK acceptance
collision policy
command watchdog
control frequency
latency
```

这些应分别作为独立 `Problem → Hypothesis → Implementation → Experiment` 任务处理。

---

# 20. Recommended Commit Strategy

如果用户明确授权 Codex commit，建议每个 ownership boundary 一个可独立编译/测试的 coherent commit：

```text
refactor(planning): organize kinematics and robot model ownership
refactor(sensor): group camera backend modules
refactor(runtime): centralize operator input and process lifecycle
refactor(teleop): isolate realtime control loop and retargeting subsystems
refactor(deployment): isolate model inference from physical execution
refactor(recording): isolate persisted storage from recording runtime
refactor(replay): clarify physical replay ownership
refactor(calibration): clarify camera extrinsics and motion modules
refactor(dataset): separate processed contract from processing transaction
refactor(config): rename resolved experiment configuration
docs: update architecture and namespace guidance
```

不要一个 200-file mega commit，也不要为每个单独 `mv` 建 commit。每个 commit 必须对应一个可解释、可验证的 subsystem/ownership change。

如果 invoking request 未授权 commit，则完成 worktree edit + validation 后停止，不自行 commit。

# 21. Final Codex Instruction

本 v5 相对 v4 的核心升级是显式建立三个 subsystem boundary：`teleop/control_loop/`、`deployment/inference/`、`recording/storage/`。这些 package 只负责 namespace/ownership 收拢，不能成为继续分层或重新设计 runtime 的理由。

Codex 应把本任务理解为：

```text
A behavior-preserving robotics namespace and ownership refactor.
```

而不是：

```text
a cleanup campaign
a code-golf task
a framework redesign
a generic modularization task
```

优化顺序始终是：

```text
domain naming
→ ownership
→ dependency direction
→ mechanical move/rename
→ remove obsolete wrapper
→ verify behavior
```

而不是：

```text
shorter files
→ more subpackages
→ more abstractions
```

最终理想状态：

> 第一次重新打开仓库时，能够仅凭 package/module/class 名称，在少量跳转内找到
> `observation → planning/retarget → action proposal → safety → publication → worker → hardware`，
> 以及 `recording → processed dataset → export` 的完整主链；同时最新真实机器人 safety/data semantics 完全不变。

---

# 22. Automatic Execution Workflow

本节由用户在 v5 review 后明确要求。本文仍是唯一 phase brief；
[`dexmani_real_codex_namespace_refactor_progress.md`](dexmani_real_codex_namespace_refactor_progress.md)
仅保存执行状态和 compact checkpoint，不复制设计或阶段说明。

## 22.1 Parent state machine

主代理按固定顺序自动推进：

```text
Phase 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → Final DoD
```

用户启动执行后，除非命中第 17 节 stop condition，主代理不得在 phase 间等待新的确认。
每个 phase 执行同一个闭环：

```text
读取 v5 与 progress checkpoint
→ 确认 worktree、HEAD、phase scope 和原始用户改动
→ luna-max 做旧符号、动态字符串和消费者 inventory
→ 主代理选择唯一 write owner 并发出 bounded phase brief
→ write owner 完成一个 coherent vertical change 和 focused check
→ luna-max 做残留路径、stale file 和 focused diff audit
→ 必要时 sol-high 做只读 boundary review
→ write owner 或主代理只修复 concrete finding
→ 主代理执行一次 phase acceptance gate 并检查 focused diff
→ 将 phase 标为 accepted，更新 progress checkpoint
→ 结束已完成的 child task，compact context
→ 从 progress checkpoint 自动进入下一 phase
```

同一个 package/import graph 同时只有一个 write owner。可并行的工作限于 read-only inventory、
互不重叠的文档盘点和 boundary review。主代理独占 phase order、scope、跨 package 依赖判断、
review 接受、progress 更新、compact 和最终 DoD。

## 22.2 Agent tier selection

主代理按实际任务难度选择 agent，不按 phase 编号机械分配：

- `sol-high`：安全、进程、IPC、生命周期、硬件或 Policy/Real 跨仓边界存在歧义时使用；默认做
  read-only review，只有主代理明确转移唯一写权限时才修改代码。
- `terra-xhigh`：清晰的多文件 namespace move、import rewrite、类型归属、职责拆分和已知 invariant
  下的集成修改，是 Phase 1–8 的默认 write owner。
- `luna-max`：旧符号清单、机械 rename、互不重叠的文档更新、残留搜索、stale-file 检查和单个
  focused check；发现 ownership、schema 或 safety 歧义时立即上交。

默认分配如下：

| Phase | Write / inventory owner | Required specialist review |
|---|---|---|
| 0 | `luna-max` inventory；主代理验收 | 无 |
| 1 | `terra-xhigh` | driver import 或 final guard 边界实际变化时 `sol-high` |
| 2 | `terra-xhigh` | 无；出现设备 import/lifecycle 变化时升级 |
| 3 | `terra-xhigh` | `sol-high` 强制只读审查 operator e-stop、process lifecycle、homing |
| 4 | `terra-xhigh` | 仅在 control/runtime safety 边界实际变化时 `sol-high` |
| 5 | `terra-xhigh` | `sol-high` 强制只读审查 inference → Prediction → executor |
| 6 | `terra-xhigh`；`luna-max` 检查资源身份 | 出现 transaction/hardware 歧义时升级 |
| 7 | `terra-xhigh` | 出现 persisted schema 语义歧义时升级 |
| 8 | `terra-xhigh`；纯路径盘点可给 `luna-max` | 出现 canonical identity 歧义时升级 |
| 9 | `luna-max` 文档与残留清理 | 主代理最终审查，不重复例行 safety review |

Phase 5 必须只读核对 `/home/zhanghaoyang/Desktop/dexmani_policy` 的 public contract。除非用户另行
授权，该仓库不属于 write scope。

## 22.3 Efficient acceptance

子代理只运行自己任务直接相关的 focused check，并回传精确命令与结果。主代理复用通过的证据，
不重复同一 focused check。每个 phase 的代码和 concrete review finding 完成后，主代理只执行一次：

```bash
conda run -n real_robot python -m compileall -q dexmani_real examples
conda run -n real_robot python -m pytest -q
git diff --check
git status --short
git diff --stat
git diff -- <phase-scope>
```

若 gate 失败，先运行最窄的相关检查定位和修复；修复后只重跑受影响检查，再运行一次完整 gate。
没有新 edit 或具体未决风险时不得重复 passing check。不要为了 namespace rename 新增仅断言路径或
类名的测试。

所有 Python 命令使用 `conda run -n real_robot`。不得运行硬件、RealSense、homing、physical replay、
calibration write 或 `execute=True`。sandbox 中 CUDA 不可见是已知运行环境限制，不是产品 defect，
不得因此增加 CPU fallback、兼容分支或临时代码；本机 GPU 能力只在 sandbox 外的既有授权流程中验证。

## 22.4 Acceptance and compact contract

只有 phase gate 通过、focused diff 符合本 phase scope、review findings 已处理，并且主代理在 progress
中写入 `accepted` 后，才可 compact 并进入下一 phase。Compact 前 checkpoint 必须记录：

```text
baseline commit and current HEAD
original user worktree changes
accepted/current/next phase
single write owner and actual changed files
move/rename/split mapping and dynamic import updates
preserved ownership, data-flow and behavior invariants
focused review findings and resolution
exact validation commands and outcomes
old active path residue versus valid historical references
resource identity/schema/config identity evidence when applicable
known unvalidated items, including no hardware validation
next phase entry points, inventory targets and risks
continuing constraints: no shim, alias, fallback, temporary code or new abstraction
```

若当前 Codex host 提供显式 compact 操作，主代理在写入 checkpoint 后调用它；若 host 只支持自动
compaction，主代理丢弃已完成 phase 的详细日志，以 progress checkpoint 作为恢复源并继续。Compact
不是 commit，不得清理或覆盖任何 worktree 内容。

## 22.5 Agent return contract

所有 child task 最终只返回 `READY` 或 `BLOCKED`，并附：

```text
assigned scope and files changed/read
preserved invariants
exact focused checks and outcomes
concrete findings or unresolved risk
focused diff summary
```

`BLOCKED` 还必须提供第 17 节规定的 evidence、affected files、why unsafe to decide 和 smallest next
investigation。主代理不得用 compatibility shim、alias、fallback、额外 retry、重复 validation 或防御性
分支绕过 blocker。
