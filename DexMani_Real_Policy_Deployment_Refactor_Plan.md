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
