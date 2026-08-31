# Learned Policy 单次任务执行 Runbook

> 状态：**首次完整 task rollout 已在 58/331 次发布时 fail closed；其后的 observation 与
> CPU inference 问题均已定位修复。current Python tree 已通过 `cuda:0` H2/H3 shadow，并在
> `a697480` sealed 完成 H4 one-endpoint。** task profile 的 deterministic seed=1066 也已在任务场景
> 完成 zero-write shadow，见
> [`deployment_reference_task_scene_h2h3_shadow_2026-08-31_6b976f8.json`](deployment_reference_task_scene_h2h3_shadow_2026-08-31_6b976f8.json)。
> 剩余 gate 是独立 task review 和新的 task 授权。
> 本文给出独立于 H4 one-shot 的 bounded task profile；文档和既有硬件证据均不构成新的真机授权。

## 1. 根因与修复边界

当前 joint checkpoint 输出 19-DoF **绝对关节目标**。训练 Zarr 的 59 个 episode
均从 canonical arm home 附近开始：episode 首帧 arm state 与 home 的最大偏差约
`0.004984 rad`，首个 action 与 home 的最大偏差约 `0.065262 rad`（3.74°）。旧部署
只在启动时下发 XHand `reset_home`，却允许机械臂从任意姿态按 B；首个模型目标因此
可能相对实测机械臂超过 20°，被原有 arm per-tick delta gate 正确拒绝。不同运行时机械臂
起点不同，解释了同一 checkpoint 有时通过、有时连续拒绝。

修复不放宽 SafetyGate：

- `execute` 与 `task` 在接受 B 前要求本进程的 H home 链路已成功完成，同时读取新鲜、健康的
  xArm feedback，并要求机械臂处于 runtime canonical home 的 homing tolerance 内；任一不满足
  都保持 ARMED、忽略 B；
- 物理模式启用 H：先要求 XHand home command 被 SDK 接受，再通过完整碰撞模型规划并
  执行 arm home；按既定要求，不判断 XHand 关节是否落在 home tolerance 内；
- inference worker 在构造 agent 前按 Policy `set_seed` 约定用 receipt 中的
  `inference_seed` 初始化一次 Python、NumPy 与 Torch/CUDA RNG streams，后续 prediction
  自然推进；本 checkpoint 采用 `training.seed + 1024 = 42 + 1024 = 1066`；
- H4 `execute` 仍严格只发布 1 个 endpoint；完整 rollout 使用独立 `task` 模式，不能用
  增大 H4 bound 的方式绕过 review；
- `task` 每次只允许一个 coupled command 在途；arm 与 hand 对同一 action id 均 ACK 后，
  coordinator 才处理下一个 endpoint。超时、feedback、generation、sequence 或 receipt
  异常继续 fail closed。
- 策略 endpoint 在最终 IPC 写入前必须至少保留一个完整 policy control tick 的 immutable
  delivery window（当前 16 Hz 为 62.5 ms）。不足时按 stale 丢弃并等待新 plan；不延长
  原 plan deadline，也不放宽 worker freshness 或 ACK gate。

归一化与反归一化仍由 checkpoint 恢复后的 Policy agent 完成；Real adapter 只拼接
`joint_state=[arm7, hand12]`、传入 point cloud，并拆分反归一化后的 `pred_action`。本次没有
修改训练仓库或 normalizer，也没有改变已训练仿真策略。

## 2. 软件完成与物理任务成功

`task` 的 `completed=true` 只证明：达到授权的 endpoint 上限、每个 coupled command 均获得
双 worker ACK、未越界且 clean shutdown。当前 checkpoint 没有 `done`/成功分类输出，Real
也没有任务成功传感器，因此软件不能自动证明物体已被成功 pick/place。物理任务结果仍须由
现场操作者观察记录；策略能力本身不能由运行命令保证。

当前数据 episode 最大长度为 331 control ticks。首次完整 rollout 不越过训练长度，使用
331 endpoints（20.6875 s @ 16 Hz）和 25 s B-relative watchdog；任何更长或重复运行都需要
重新 review 与授权。

2026-08-30 在 `77d8c44` 上的首次 task 尝试发布 58 个 coupled endpoints，前 57 个获得双
worker ACK。第 58 个动作仅剩 `8.517 ms` plan validity，短于 30 Hz worker 的单周期，因而被
arm 和 hand 同时判为 expired；2 秒 ACK watchdog 随后正确转入 FAULT。该运行不是任务成功，
没有自动重试。原始事实与 hash 记录在
`deployment_reference_task_execute_failure_2026-08-30_77d8c44.json`。

随后在 `bf79d4f` 上进行的 H2/H3 shadow revalidation 没有构建出首个 observation/plan，5 秒
first-command watchdog 将 RUNNING 退回 ARMED；coupled ring 保持 zero-write。该 revision 的
no-feedback 分支没有定期 flush 分类指标，无法从 receipt 继续区分具体 observation 前置条件。
当前实现只增加每秒一次的 pointcloud-grid/stale、arm/hand history 与 grid-advance 等待计数，
不改变任何 observation 条件或动作路径。失败事实记录在
`deployment_reference_h2h3_shadow_failure_2026-08-30_bf79d4f.json`。

`acc2cc1` 上的后续 H2/H3 已证明 observation 能构建并产生 plan，但该 invocation 未传
`--device`，实际回落到 CPU；同一 inference child 在点云等 CPU worker 并发后出现
`0.35–3.69 s` 的间歇推理延迟，固定 action grid 全部或大部过期，最终两次触发 command-silence
abort。两段 receipt 均为 zero coupled writes，操作员随后从 ARMED 按 Q 正常退出。第二个
RUNNING epoch 的 start-request 来源无法由旧日志归因，但旧 coordinator 确实允许一个 shadow
进程内重复启动。离线同 checkpoint/seed 基准确认 CPU 稳态约 196 ms，而 `cuda:0` 首次约
140 ms、随后约 18–20 ms。因此当前入口要求显式 device，本 reference 固定 `cuda:0`；inference
ready 前还会完成 5 次无硬件 warmup，要求最后 3 次均落在 artifact action horizon 的可用窗口
内，并在 warmup 后恢复所有 RNG 状态；同时所有模式每个进程只接受第一次 B。完整事实见
`deployment_reference_h2h3_shadow_failure_2026-08-30_acc2cc1.json`。

`6349147` 上使用显式 `--device cuda:0` 的后续 H2/H3 已通过完整 120 秒验证：启动 warmup
稳定最大值为 `20.036 ms`，在线报告的最大 inference sample 为 `24.318 ms`、滚动 p99 最大值
为 `25.957 ms`；`1916/1916` 个 endpoint 完成 shadow validation，SafetyGate reject、motion
discard、inference failure 和 coupled command write 均为 0，arm `servo_calls=0`，最后所有
worker graceful shutdown。事实与日志 hash 记录在
`deployment_reference_h2h3_shadow_2026-08-30_6349147.json`。该结果只解除 H2/H3 阻塞，
不构成 H4 或 task 真机授权。

## 3. 真机前离线检查

在干净且已 review 的 DexMani Real revision 上执行；两条命令都不会连接硬件：

```bash
cd /home/zhanghaoyang/Desktop/dexmani_real

common_args=(
  --experiment-dir /home/zhanghaoyang/Desktop/dexmani_policy/experiments/dp3/pick_place_toy/2026-08-28_13-59_42
  --device cuda:0
  --execution-mode task
  --hand
  --inference-seed 1066
  --max-running-seconds 25
  --task-max-published-endpoints 331
  --task-ack-timeout-seconds 2
  --task-expected-checkpoint-sha256 b174bd483b64090cd3f5dbe0a5bfadd10998f5d27d43fc9aca06efb82242484c
)

PYTHONPATH=/home/zhanghaoyang/Desktop/dexmani_policy \
/home/zhanghaoyang/miniconda3/envs/real_robot/bin/python \
  examples/run_policy.py "${common_args[@]}" --print-config

PYTHONPATH=/home/zhanghaoyang/Desktop/dexmani_policy \
/home/zhanghaoyang/miniconda3/envs/real_robot/bin/python \
  examples/run_policy.py "${common_args[@]}" --preflight-only
```

确认 checkpoint SHA、projection SHA、Real commit/dirty、Policy package provenance、seed、task
bounds 与 preflight prediction 全部符合本次批准内容。dirty/unknown Real revision会在连接硬件前
拒绝物理模式。

## 3.1 task seed 的场景 shadow

H4 使用的默认 seed=0 只证明一个已封存的硬件 publication；它不证明 task 的 seed=1066
diffusion sampling trajectory。获得单独的 H2/H3 shadow 授权后，在任务物体按训练场景摆放、但
场地无人且无非任务障碍物时，运行以下 command。它只允许启动时的 XHand `reset_home`；B 后不
发布 policy action、不触发 H/arm home，也不接触物体：

```bash
cd /home/zhanghaoyang/Desktop/dexmani_real

PYTHONPATH=/home/zhanghaoyang/Desktop/dexmani_policy \
PYTHONUNBUFFERED=1 \
/home/zhanghaoyang/miniconda3/envs/real_robot/bin/python \
  examples/run_policy.py \
    --experiment-dir /home/zhanghaoyang/Desktop/dexmani_policy/experiments/dp3/pick_place_toy/2026-08-28_13-59_42 \
    --execution-mode shadow \
    --hand \
    --device cuda:0 \
    --inference-seed 1066 \
    --max-running-seconds 120
```

验收条件是：checkpoint/runtime/source identity 与本节 task profile 一致，warmup 通过，B-relative
120 秒内 endpoint 均只完成 shadow validation，coupled writes、arm servo、hand policy SDK send、
SafetyGate reject、motion discard 和 worker fault 均为零，并 clean shutdown。该 shadow 不替代
task execute 的独立场景 checklist 或授权。

## 4. 单次完整 rollout 命令

只有在获得一份新的、明确包含 `task execute`、`--hand`、一次 arm+hand home、331 endpoints、
25 秒、场地/设备/e-stop 状态的真机授权后，才可执行：

```bash
cd /home/zhanghaoyang/Desktop/dexmani_real

receipt_dir=/home/zhanghaoyang/.dexmani/receipts
log_dir="$(mktemp -d /tmp/dexmani-task-execute.XXXXXX)"
mkdir -p "$receipt_dir"

PYTHONPATH=/home/zhanghaoyang/Desktop/dexmani_policy \
PYTHONUNBUFFERED=1 \
DEXMANI_RECEIPT_DIR="$receipt_dir" \
/home/zhanghaoyang/miniconda3/envs/real_robot/bin/python \
  examples/run_policy.py \
    --experiment-dir /home/zhanghaoyang/Desktop/dexmani_policy/experiments/dp3/pick_place_toy/2026-08-28_13-59_42 \
    --device cuda:0 \
    --execution-mode task \
    --hand \
    --inference-seed 1066 \
    --max-running-seconds 25 \
    --task-max-published-endpoints 331 \
    --task-ack-timeout-seconds 2 \
    --task-expected-checkpoint-sha256 b174bd483b64090cd3f5dbe0a5bfadd10998f5d27d43fc9aca06efb82242484c \
    2>&1 | tee "$log_dir/terminal.log"
```

操作顺序不可省略：

1. 等全部 subsystem ready，系统显示 ARMED。
2. 操作者按一次 H；该操作先下发 XHand home，再执行 collision-checked arm home。只要求 hand
   command accepted，不检查手指 home tolerance。本进程不接受第二次 H 尝试。
3. 等日志明确显示 `physical home sequence completed` 并保持 ARMED；缺少完成标志或 arm 不在
   home 时，coordinator 都会忽略 B。H 失败后退出，不在同一进程重试。
4. 操作者确认场地状态没有变化后按 B，开始一次 bounded task rollout。
5. 任意异常立即按 S；人员/物体进入、失控趋势或紧急危险立即按 ESC/e-stop。不得自动重试。

正常软件终态应包含 `physical_home_completed=1`、`reason="task publication bound reached"`、`completed=true`、
`coupled_command_writes=331`、`execute_acknowledged=331`、verified clean shutdown，并写出
`task_execute_*.json`。退出码非零、FAULT、少于上限、ACK/sequence/freshness/safety reject 持续、
receipt 缺失或物理结果异常均不是成功。
