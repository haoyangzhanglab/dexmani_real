# Learned Policy 单次任务执行 Runbook

> 状态：**仅完成离线实现，尚未获得真机授权或真机验证。** 本文给出独立于 H4
> one-shot 的 bounded task profile；文档和命令本身不构成硬件授权。

## 1. 根因与修复边界

当前 joint checkpoint 输出 19-DoF **绝对关节目标**。训练 Zarr 的 59 个 episode
均从 canonical arm home 附近开始：episode 首帧 arm state 与 home 的最大偏差约
`0.004984 rad`，首个 action 与 home 的最大偏差约 `0.065262 rad`（3.74°）。旧部署
只在启动时下发 XHand `reset_home`，却允许机械臂从任意姿态按 B；首个模型目标因此
可能相对实测机械臂超过 20°，被原有 arm per-tick delta gate 正确拒绝。不同运行时机械臂
起点不同，解释了同一 checkpoint 有时通过、有时连续拒绝。

修复不放宽 SafetyGate：

- `execute` 与 `task` 在接受 B 前读取新鲜、健康的 xArm feedback，并要求机械臂处于
  runtime canonical home 的 homing tolerance 内；不满足时保持 ARMED、忽略 B，并输出
  current/home/delta/tolerance 数值；
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

## 3. 真机前离线检查

在干净且已 review 的 DexMani Real revision 上执行；两条命令都不会连接硬件：

```bash
cd /home/zhanghaoyang/Desktop/dexmani_real

common_args=(
  --experiment-dir /home/zhanghaoyang/Desktop/dexmani_policy/experiments/dp3/pick_place_toy/2026-08-28_13-59_42
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
2. 操作者按 H；该操作先下发 XHand home，再执行 collision-checked arm home。只要求 hand
   command accepted，不检查手指 home tolerance。
3. 等 arm home 流程明确完成并保持 ARMED；如果误按 B 且 arm 不在 home，coordinator 会忽略 B
   并打印逐关节差值。
4. 操作者确认场地状态没有变化后按 B，开始一次 bounded task rollout。
5. 任意异常立即按 S；人员/物体进入、失控趋势或紧急危险立即按 ESC/e-stop。不得自动重试。

正常软件终态应包含 `reason="task publication bound reached"`、`completed=true`、
`coupled_command_writes=331`、`execute_acknowledged=331`、verified clean shutdown，并写出
`task_execute_*.json`。退出码非零、FAULT、少于上限、ACK/sequence/freshness/safety reject 持续、
receipt 缺失或物理结果异常均不是成功。
