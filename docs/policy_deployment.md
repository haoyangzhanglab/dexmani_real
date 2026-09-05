# Learned-policy experiment workflow

本文说明 DexMani Policy experiment 在 DexMani Real 中的检查、shadow 和物理运行流程。
运行行为和安全边界以源码、`PolicySpec` 与 Real runtime config 为准。

## 1. 边界与前提

Policy 仓库拥有 experiment 解析、checkpoint 选择、模型构造、EMA/normalizer 恢复和推理；
Real 仓库拥有传感器、因果 observation、控制时序、共享内存、SafetyGate、命令发布和设备 acceptance/progress 观察。
Real 不解析 Policy checkpoint，也不复制模型 shape、history 或 action horizon。

一个可部署 experiment 可以使用已有目录，或短 selector：

```text
policy/task/experiment
```

该目录必须包含 `config.yaml`，以及固定 selector
`checkpoints/deployment_latest.pt`。没有隐式 `latest` experiment 或时间戳猜测。
所有 artifact 使用 `dexmani.deployment.v2`，根 payload 只有 `_format`、`contract` 和
`weights`。其中有序 `data_contract.observation_fields` 是观测输入的唯一持久化声明；每项记录
名称、raw shape、dtype 和语义。它们都要求 xArm7 + XHand、兼容的控制周期与单槽 Prediction IPC，
使用 point cloud 时还必须匹配点数和 xyzrgb feature。

## 2. 从离线检查到真机

先在 Real 仓库列出 experiment：

```bash
python examples/run_policy.py list
python examples/run_policy.py list <filter>
```

`list` 只做 Policy 文件系统发现，不加载 Torch、checkpoint 或硬件 SDK。

再严格恢复并执行合成 observation smoke test：

```bash
python examples/run_policy.py check <experiment> --device cuda:0
```

`check` 验证 checkpoint restore、normalizer、warmup 和 prediction，不连接机器人或相机。
无 GPU 的开发机可显式使用 `--device cpu` 做离线检查；这不能替代目标 GPU 的时延验证。

`--device` 只属于 `check`、`shadow` 和 `run`。`--runtime-config`、
`--inference-mode` 与 `--max-action-steps` 只属于
`shadow/run`，不会被 `list/check` 静默接受。选项必须跟在其所属子命令之后，不能置于子命令前。

以下两个命令都会进入真实设备 lifecycle：

```bash
python examples/run_policy.py shadow <experiment> [--runtime-config <runtime-overrides.yaml>]
python examples/run_policy.py run <experiment> [--runtime-config <runtime-overrides.yaml>]
```

- `shadow` 连接相机、xArm 与 XHand feedback，运行 inference、IK 和 SafetyGate，但
  `execute=False` 从结构上禁止 actuator publication，也禁用 H/home。
- `run` 连接并控制 xArm/XHand；只有它设置 `execute=True`，并允许非阻塞 coupled arm/hand
  publication。正常策略 tick 不等待同 ticket acceptance；`PolicyExecutor` 以 arm
  `last_cmd_seq` 和 hand `accepted_target_action_id` 监控进度。

`shadow` 不是纯离线命令。首次真机回归必须人工确认连接阶段 arm 与 XHand 均不运动，
再用 B/S 执行短 episode，并确认没有 actuator SDK command。

## 3. Readiness 与操作者流程

inference child 先 strict restore 并 warmup。只有它 ready 后，
lifecycle 才启动硬件 workers；任何模型加载失败都会在硬件连接前 fail closed。
其余 workers ready 后系统从 DISARMED 进入 ARMED。

按键语义：

| 按键 | 行为 |
|---|---|
| `B` | 从 ARMED 开始一个新 episode；推进 generation。 |
| `S` | 立即撤销当前 motion permit，结束 episode 并回到 ARMED。 |
| `H` | 仅 `run` 可用；在 ARMED 中执行 XHand home 与 collision-checked arm home。 |
| `Q` | 结束 session 并执行有界 shutdown。 |
| `ESC` | 锁存 emergency stop/fault。 |

物理 `run` 中，每个 episode 都需要新的 `H → B`：H 成功只授权下一次新鲜 B，B 开始后
授权立即消费。S 后可以在同一进程再次 H→B；旧 generation 的 Prediction、ticket 和 command
不会进入新 episode。不要把 H 与 B 放在同一批键盘输入中。

推荐逐级真机验证：

1. `shadow` 启动并人工确认 startup no-motion。
2. `shadow` 中 B→S，确认 arm/hand 均零发布。
3. `run` 中 H→短 B→S，观察 SafetyGate、coupled ticket 与 arm/hand command-progress watermark。
4. 同进程重复 H→B→S，确认每个 episode 都需要 home 且 generation 前进。
5. 在安全条件下验证 S、Q、ESC；不要故意制造危险硬件故障。

## 4. 配置与因果 observation

Real 配置优先级固定为：

```text
CLI override > YAML file > dexmani_real/config/defaults.py
```

input freshness、observation skew、source-to-command freshness、command-progress timeout、action validity 和
watchdog 属于 `ResolvedRuntimeConfig.policy`。模型 action/observation shape、history、horizon 和
`control_dt_s` 只来自 Policy `PolicySpec`；lifecycle 启动前必须与 Real 配置精确兼容。

`PolicyExecutor` 独占调度。默认 `sync`：它在 episode 开始和当前 Prediction 完成或因 source stale
丢弃后请求一次推理。可选 `--inference-mode async` 按 `n_action_steps * control_dt_s` 的绝对 cadence
推理，较新的同 generation Prediction latest-wins 替换尚未消费的 suffix。executor 以
`executor_poll_hz=128` 轮询，每轮最多发布一个到期 endpoint，绝不追赶过期 slot。

策略 arm endpoint 经过 `arm.max_servo_command_jump_rad=20°` 的 reject-only SafetyGate；hand target
经过独立的 `policy.hand_max_action_jump_rad=1.0 rad` reject-only 检查。首条动作以 measured feedback 为
reference，后续动作以上一条成功发布的 target 为 reference；通过后原样发布。hand worker 仍在 SDK
边界用 `hand.hand_max_delta_rad_per_tick` 生成受限中间 setpoint。任一 coupled target 被拒绝时均不发布。
`--max-action-steps N` 限制每个 episode 的 terminal action steps；达到 N 时以 `TRUNCATED`
结束，而不是触发 FAULT。示例：

```bash
python examples/run_policy.py shadow <experiment> --max-action-steps 32
python examples/run_policy.py run <experiment> \
  --inference-mode async
```

artifact 的唯一观测合同是有序 `observation_fields`。Policy export 从 experiment 的
`dataset.sensor_modalities` 读取人工选择，并只在该选择、训练数据与 encoder 实际消费字段一致时
导出；artifact 不再保留重复的 `sensor_modalities` 或 contract 外版本字段。Real 按字段顺序构造精确的
`PolicyObservation.arrays`：RGB 保持 causal raw HWC `uint8`，model-specific preprocessing
只在 Policy runtime/encoder 内执行；RGB 与 point cloud 共存时必须来自同一 camera sequence、
generation 与 source timestamp；contact 必须有 fresh/calibrated/unit/source-match 证明；
fingertip 使用同步 arm/hand state 和本地 FK 输出 `xarm_base` 米制坐标。

字段组合的可用性由实际 encoder 决定。一个 encoder 未消费的字段不能被导出或部署，避免静默
转发、忽略或填充输入。

point-cloud observation 保留 `source <= publish <= anchor`、run-start 下界、camera generation、
最大 age/skew、控制网格选择和 logical-grid advance。缺帧、过期、shape/dtype 错误、非有限值
或颜色越界时不会推理或发布新 Prediction；这些门禁不能为提高运行通过率而放宽。

## 5. 运行诊断与失败语义

inference 和 `PolicyExecutor` 周期性输出 observation age/skew、inference、Prediction publish/ingest/
stale/drop、endpoint、SafetyGate/IK reject、publication 与 command-progress 等 live metrics。
`PolicyExecutor` 在每个 episode 结束时另输出一条 `episode summary`，包含 generation、状态、原因、
duration、累计 counters 和固定容量 timing quantiles。

这些 summary 是 log-only experiment diagnostics：不落盘、不做 hash、不生成 receipt、sidecar
或部署资格证明。SDK acceptance 只表示 worker/SDK 接受目标，不表示物理到位；它保留给 home、
calibration 和 replay 等明确的 blocking acceptance 流程。正常策略 tick 只发布并继续，由独立的
arm/hand progress watchdog 检测持续已发布命令却没有 acceptance watermark 前进的故障。是否允许运动
仍只由当前 safety state、generation、freshness、SafetyGate 和 worker validation 决定。

B/S 请求、run generation 与 RUNNING 转换在同一个 `motion_lock` 下排序；新鲜 S 不能被正在消费的
B 覆盖。每个 candidate 固定继承来源 Prediction 的 generation，最终 policy publication 仅在原子快照仍为
RUNNING 且 generation 相同时写入 coupled-command ring。home 等其他控制路径仍使用各自既有状态合同。

策略 step 的 SafetyGate/IK rejection 只拒绝当前 step；后续由 action-step limit、command silence 或
command-progress timeout 决定 episode 何时回到 ARMED，而不是走旧的立即 abort 路径。无效 IPC、
fatal worker feedback、worker progress 停滞以及 error/e-stop 才进入 FAULT。S 是 episode stop，不要求
重启进程。Q/ESC、worker 退出、heartbeat/readiness 失败和 shutdown 异常仍由 lifecycle/supervisor
fail closed 处理。

## 6. 验证状态

离线回归覆盖 startup no-motion、shadow no-publication、inference-first readiness、PolicySpec
兼容、因果 observation、generation 隔离、SafetyGate、command-progress/watchdog、重复 episode 和 diagnostics。
离线测试不等于真机验证。

截至 2026-09-05，集中真机回归已完成：CUDA `check`、连接设备的 `shadow`（无执行器发布）、
低风险 `run` 的 H→B→S、同进程重复 H→B→S、RUNNING 中 S，以及 RUNNING 中 ESC/e-stop。各次
结束后 arm/hand worker 均已干净退出，没有遗留 actuator command。

尚未覆盖的边界包括：人为注入超过 policy hand jump 阈值的模型动作、SDK CRC/拒绝、持续 sensor
失效、接触导致的 measured-feedback lag，以及真实相机/point-cloud 的端到端时延测量。这些需要在
受控实验条件下单独验证，不能由上述常规回归替代。
