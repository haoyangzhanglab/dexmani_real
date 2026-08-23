# DexMani Real Policy Deployment 重构与完善方案

## 0. 目标

建立 Real-only、模型无关、可反复 START/STOP、便于后续 dexmani_policy
模型接入测试的策略部署框架。

核心链路：

    Real Dataset Contract
            ↓
    dexmani_policy checkpoint + DeploymentManifest
            ↓
    PolicyRuntime
            ↓
    Real causal observation window
            ↓
    Model inference
            ↓
    Timestamped action trajectory
            ↓
    Scheduler
            ↓
    Safety Gate
            ↓
    Robot Command

------------------------------------------------------------------------

# 1. 核心原则

## 1.1 Real-only

真机部署禁止：

-   import dexmani_sim
-   instantiate sim env_runner
-   使用 sim observation preprocessing
-   使用 sim frequency 推导 real control timing
-   自动继承 sim temporal ensemble

sim 只能用于理解模型 API。

------------------------------------------------------------------------

## 1.2 职责边界

dexmani_policy：

-   checkpoint loading
-   EMA/raw weights
-   normalizer
-   model config
-   observation contract
-   action decoding

dexmani_real：

-   sensor
-   timestamp alignment
-   SharedMemory
-   scheduler
-   operator control
-   safety
-   robot command

------------------------------------------------------------------------

## 1.3 Generation 是跨进程失效边界

所有策略相关对象必须绑定：

    run_generation
    observation_id
    action_id
    target_time

以下事件必须增加 generation：

-   B START
-   S STOP
-   FAULT
-   新 run

旧：

-   policy plan
-   arm command
-   hand command

必须立即失效。

------------------------------------------------------------------------

# 2. Operator 控制设计

第一版：

  按键   功能
  ------ ----------------------
  B      START new policy run
  S      STOP current run
  H      HOME
  Q      QUIT
  ESC    Emergency Stop

不实现 C PAUSE/RESUME。

原因：

PAUSE 会导致 observation history 跨 temporal boundary，需要重新设计。

------------------------------------------------------------------------

# 3. 生命周期

    STARTUP
        |
        v
    ARMED
        |
        | B
        v
    RUNNING
        |
        | S
        v
    ARMED

ARMED：

-   model loaded
-   GPU allocated
-   sensor active
-   no inference
-   no policy command

RUNNING：

-   inference enabled
-   scheduler enabled
-   command publication enabled

------------------------------------------------------------------------

# 4. dexmani_policy Deployment API

禁止：

-   build_agent()
-   agent.reset()
-   env_runner dependency

建立：

    PolicyRuntime

接口：

``` python
load(
    model_config,
    checkpoint,
    manifest,
    device
)

predict(obs)

reset_episode()
```

输入：

``` python
{
    "joint_state": [B,T,19],
    "point_cloud": [B,T,P,6]
}
```

输出：

``` python
future_action [H,19]
```

------------------------------------------------------------------------

# 5. Deployment Manifest

模型必须提供：

-   domain
-   task
-   dt
-   n_obs_steps
-   modalities
-   action_dim
-   control_action_dim
-   action representation
-   point cloud contract
-   normalizer info
-   inference config

启动必须验证：

    domain == real

禁止 sim checkpoint 进入 real deployment。

------------------------------------------------------------------------

# 6. Observation Assembly

模型输入：

    joint_state:
    [B,T,19]

    point_cloud:
    [B,T,P,6]

禁止：

    arm_qpos
    hand_qpos
    latest frame stacking

------------------------------------------------------------------------

## Temporal Sampling

必须使用：

    policy timestamp grid

而不是：

    ring 最近 T 帧

每个 timestamp：

    causal arm state
    causal hand state
    causal pointcloud

检查：

-   source age
-   publish time
-   cross modal skew

------------------------------------------------------------------------

# 7. Action Timing

禁止：

    target = inference_finish + dt

必须：

    target_time =
    observation_anchor + k * dt

Inference latency：

通过：

    stale prefix rejection

处理。

不要平移整个 trajectory。

------------------------------------------------------------------------

# 8. Scheduler

不要使用：

    inference_hz

改为：

    replan_stride_steps

例如：

    dt = 62.5ms

    stride=8:
    500ms replanning

    stride=4:
    250ms replanning

------------------------------------------------------------------------

Scheduler 使用：

    active_plan
    pending_plan

避免：

    new plan arrival
    ↓
    old plan discard
    ↓
    command gap

------------------------------------------------------------------------

# 9. Safety Gate

检查顺序：

1.  shape
2.  finite
3.  generation
4.  command time
5.  joint limits
6.  delta limits
7.  workspace
8.  collision

------------------------------------------------------------------------

禁止：

    clip learned action

正确：

    unsafe
     ↓
    reject

------------------------------------------------------------------------

Collision：

使用：

    check_transition_collision_free()

检查：

    arm current → target
    hand current → target

------------------------------------------------------------------------

# 10. Arm / Hand Command Contract

ARM 和 HAND 必须统一：

    run_generation
    observation_id
    action_id
    created_time
    target_time
    valid_until

Arm 不能只有 sequence + age。

------------------------------------------------------------------------

# 11. 测试计划

## Generation

验证：

-   STOP 后旧 plan 无法执行
-   STOP 后旧 arm command 无法执行
-   STOP 后旧 hand command 无法执行

------------------------------------------------------------------------

## Keyboard

测试：

    B
    S
    B
    S
    Q

循环运行。

------------------------------------------------------------------------

## Timing

注入：

    0ms
    30ms
    70ms
    130ms

验证 action timestamp。

------------------------------------------------------------------------

## Model Smoke Test

三层：

### Level 1

Policy-only：

    checkpoint
    +
    manifest
    +
    sample observation

验证：

-   load
-   forward
-   shape
-   finite

### Level 2

Offline integration：

    real observation
    → adapter
    → scheduler
    → safety dry-run

### Level 3

Live dry-run：

真实：

-   arm
-   hand
-   camera
-   pointcloud
-   model

但是：

    disable command publication

------------------------------------------------------------------------

# 12. 实施顺序

## Phase 1

Generation safety：

-   ARM command generation
-   validation
-   coordinator generation check

## Phase 2

Lifecycle：

-   ARMED idle
-   RUNNING active
-   B/S

## Phase 3

Keyboard：

-   B
-   S
-   H
-   Q
-   ESC

## Phase 4

PolicyRuntime：

-   manifest
-   loader
-   prediction contract

## Phase 5

Observation：

-   causal grid
-   history reset
-   pointcloud history

## Phase 6

Scheduler：

-   timestamped trajectory
-   active/pending plan
-   stride scheduling

## Phase 7

Safety：

-   delta limit
-   collision transition
-   publication gate

------------------------------------------------------------------------

# 13. Definition of Done

完成后：

启动：

    model loaded
    hardware ready
    Safety=ARMED
    robot static

按 B：

    new generation
    new observation epoch
    RUNNING
    inference active

按 S：

    generation++
    old command invalid
    old plan invalid
    RUNNING -> ARMED
    model remains loaded

再次 B：

    new run
    new history
    no reload

新模型接入：

只需要：

    checkpoint
    model config
    DeploymentManifest

不修改：

    scheduler
    safety
    robot worker
    operator control

------------------------------------------------------------------------

# 14. 审查修订（2026-08-23，结合 dexmani_policy 实况）

本节由对 `~/Desktop/dexmani_policy` 的逐条核对得出，**优先于**第 4/5/6/9/10
节中与此冲突的表述。三个待决点已定：PolicyRuntime 放 `dexmani_real`、manifest
由 real 侧产出、支持 EE。

## 14.1 关键事实（`dexmani_policy` 真实契约）

| 方案原假设 | 实况 | 结论 |
|---|---|---|
| §4 `PolicyRuntime` 替换 `build_agent()` | `__init__.py` 为空，无 `build_agent`；真实入口 `BaseAgent.predict_action`（`agents/core/base.py`） | 现有 `integrations/dexmani_policy.py` 建在臆想 API 上，须**整替换** |
| §4 输入 `joint_state [B,T,19]` | `_validate_obs_dict` 要求每个 obs tensor `(B,T,...)`,`T>=n_obs_steps`；`state_dim:19`、`sensor_modalities:["joint_state","point_cloud"]` | 精确吻合；`T=n_obs_steps=2`（非 4） |
| §4 输出 `future_action [H,19]` | `predict_action` 返回 `{"pred_action"(B,16,A),"control_action"(B,8,control_action_dim),"tail"}`；`control_action_dim`=19 原生关节 | 吻合；`H=n_action_steps=8`；EE 时 control_action_dim=21 |
| §5 manifest `control_action_dim` | 已在 `base.py` 属性与 checkpoint `train_params` | 现成 |
| §1.1 禁止 sim/env_runner/temporal ensemble | `env_runner/sim_runner.py:7` `from dexmani_sim import DATA_DIR`；eval 默认 `temporal_ensemble_coeff=0.01` | 该条 load-bearing |

关键推论：

- **normalizer 在 checkpoint 里**：`self.normalizer` 是 agent 子模块，随
  `model_state` 存盘；`load_ckpt_for_inference`（`training/eval_utils.py`）
  只 `instantiate(cfg.agent)` + `load_state_dict` + `is_fitted(["action"])`，
  **不碰 dataset/DATA_DIR**。故 `PolicyRuntime.load` 可脱离数据集。
- **`dt` 不在模型配置里**：模型只懂步数（`horizon=16/n_obs_steps=2/
  n_action_steps=8`），时间尺度是 sim/real 控制栅格概念。manifest 的 `dt`
  必须由 real 侧提供。
- **`train_params` 已含 9 个 manifest 类字段**：`n_obs_steps / n_action_steps /
  action_dim / horizon / action_key / tcp_dim / use_faas / hand_dim /
  control_action_dim`（`common/checkpoint_io.py`）。manifest 大半可从它派生。

## 14.2 三个决策的落地形态

### 决策 1 — PolicyRuntime 放 `dexmani_real`

`dexmani_real/integrations/dexmani_policy.py` 整替换为 `PolicyRuntime`：

```python
load(model_config, checkpoint, manifest, device)
predict(obs)          # -> agent.predict_action(...)["control_action"]  [H, control_action_dim]
reset_episode()
```

`load` 只 import `dexmani_policy.common.*`、`training/build_utils`、
`training/eval_utils`、`agents.*`；**绝不 import `env_runner`**（其顶层
`from dexmani_sim import DATA_DIR`）。流程：`OmegaConf.load(model_config)` →
`register_resolvers` + `normalize_action_key` → `instantiate(cfg.agent)` +
`inject_faas_into_agent` → `load_ckpt_for_inference`（EMA/raw）→ 校验
`train_params == manifest`。`reset_episode` 无循环态（diffusion/flowmatch
无状态），主要是 real 侧观测历史重置 + 可选 `ChunkOverlapBlender.reset`。

### 决策 2 — manifest 由 real 侧产出

`DeploymentManifest` 是 real 侧 frozen dataclass，启动时由 loader 组装：

- 来源：checkpoint `train_params`（9 字段）+ config.yaml（`sensor_modalities/
  pc_dim/num_points/normalizer_mode`）+ runtime config（`dt/control_hz`）。
- `domain` 恒为 `real`（无 sim 组装路径，故 §5 的 domain==real 由构造保证）。
- 校验：`action_key ∈ {action, action_ee}`、`use_faas/use_aux_ee ∈ {T,F}`、
  `dt` 与录制/控制栅格一致。

### 决策 3 — 支持 EE（`action_ee`, 21D）

`manifest.action_key` 分派：

- **joint（`action`）**：`control_action [H,19]` = arm7 ‖ hand12，arm 直接进
  joint safety gate，hand12 直接。
- **ee（`action_ee`）**：`control_action [H,21]` = pos3 ‖ rot6d6 ‖ hand12。
  arm 走 `pose_utils.rot6d_to_quat_wxyz` → `Pose(p, q)` →
  `planner.solve_teleop_ik(target_eef_pose_world, current_qpos, prev_cmd)`
  （先 `planner.set_hand_qpos(hand12)` 做 19-DoF 碰撞）；IK fail → reject
  （符合 §9）。hand12 直接。新增的只有 coordinator 里一段 EE→IK→joint
  分派。
- **FAAS**：透明。`predict_action` 内部做 19↔39 转换，`control_action`
  仍是 native 19/21，real 侧无需特殊处理。

EE 坐标帧（pos3/rot6d 的参考系）必须与 real 点云的 xArm-base 帧一致，锁进
manifest 的 action representation + point cloud contract 字段。

## 14.3 需补进对应章节的缺口

1. **§5 domain 载体**：manifest 由 real 侧构造，`domain=real` 为常量；补
   “谁产出、在哪校验”的说明（= 决策 2）。
2. **§5 dt 来源**：`dt` 不在模型 config，由 real 控制栅格提供并校验。
3. **§5/§6 point cloud contract 展开**：不止 `N` 个点，还含坐标帧、颜色
   归一（normalizer 把 PC 归一化到 [-1,1]）、以及训练 FPS 采样 vs real 确定性
   采样的一致性。dp3 训练用 `fps_random_config`（FPS+随机），real 侧
   `pointcloud_process` 是确定性 `float32[N,6]`——此差异需在 manifest 显式
   声明并校验，否则有 train/deploy 预处理不一致的隐性精度损失。
4. **§6 观测组装**：real 侧需把 arm_qpos(7)/hand_qpos(12) 拼成 19D
   `joint_state`（FAAS mapper 也按 `[...,:7]`/`[...,7:]` 切分）；点云从
   latest-only 改成 `T=2` 历史窗（`pointcloud_process` 目前只有 latest）。

## 14.4 修订后的实施顺序

Phase 1–3 不变（纯 real 侧，与模型无关）：

- Phase 1：ARM command generation（§10，arm dtype 补齐六字段）
- Phase 2：Lifecycle（ARMED idle / RUNNING active / B/S）
- Phase 3：Keyboard（B/S/H/Q/ESC）

Phase 4 拆两步：

- Phase 4a：整替换 `integrations/dexmani_policy.py` 为 `PolicyRuntime`
  （现状是坏的，须先替换再谈其余）
- Phase 4b：`DeploymentManifest`（real 侧组装 + 校验，含 EE 分派）

Phase 5–7 追加 EE 与点云历史：

- Phase 5：观测（19D joint_state 拼装 + 点云 `T=2` 历史 + causal grid）
- Phase 6：Scheduler（`replan_stride_steps` + active/pending + EE→IK 分派）
- Phase 7：Safety（delta limit + collision transition + publication gate）
