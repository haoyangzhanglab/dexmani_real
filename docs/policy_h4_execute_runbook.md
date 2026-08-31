# H4 首次物理 coupled execute Runbook

> 状态：`3488063` 的 H4 runtime receipt 虽显示 completed，但 hand shutdown counter 被 generation
> revoke 清零，evidence sealer 已正确拒绝封存，详见
> [`deployment_reference_h4_execute_unsealed_2026-08-31_3488063.json`](deployment_reference_h4_execute_unsealed_2026-08-31_3488063.json)。
> 修复后的当前 Python deployment tree 已完成 H2/H3 shadow；下一步是独立 H4 review 和新的明确
> H4 授权。当前 `cuda:0` 120 秒 zero-write baseline 是
> [`deployment_reference_h2h3_shadow_2026-08-31_42bb288.json`](deployment_reference_h2h3_shadow_2026-08-31_42bb288.json)：
> 1,915/1,915 个 endpoint 完成 shadow validation，且所有 policy physical command 计数均为零。
> `dfad0e7` receipt 保留为修复前的历史审计证据，不能替代 current-tree baseline。
> `3b2615f` 已完成 65 秒、1038 endpoint、zero-write 的短 H2/H3 shadow，见
> [`deployment_reference_h2h3_shadow_2026-08-31_3b2615f_short.json`](deployment_reference_h2h3_shadow_2026-08-31_3b2615f_short.json)；
> 它不替代当前 revision 所需的 120 秒 baseline。
> `02f88e6` 的单 endpoint 与双 worker ACK 均成功，但日志没有 H home 链路证据，因此不计为
> H4 通过；事实见
> [`deployment_reference_h4_execute_failure_2026-08-30_02f88e6.json`](deployment_reference_h4_execute_failure_2026-08-30_02f88e6.json)。
> 本文不是启动授权。没有独立 review 与明确、限定的 H4 真机授权，任何人不得运行 execute
> lifecycle。

## 1. 目标与范围

H4 只回答一个问题：经过现有完整 validation 的 learned-policy endpoint 能否在空旷、无物体的
工作空间中产生**有限且可证明**的 physical coupled command publication，并由 arm/hand worker
在各自 SDK 边界复核。

本次不验证任务成功、抓取、接触、桌面操作、长时间稳定性或性能。不得将本 runbook 或 H2/H3
shadow 证据解释成 execute 授权。

## 2. 已实现的软件硬门

`examples/run_policy.py` 和 lifecycle 仅接受以下不可放宽的 H4 profile：

- inference 默认 `cuda:0`；CUDA 不可用会 fail closed，不会隐式落到 CPU。本 frozen reference
  的 operational invocation 仍须显式写出 `--device cuda:0`，以使人工审计和 receipt 对齐；
- `--execution-mode execute --hand`，且 artifact 必须要求 hand；
- `--execute-max-published-endpoints 1`，不能使用其他数值；
- `--execute-ack-timeout-seconds <finite positive>`；
- `--execute-expected-checkpoint-sha256 <64-hex>` 必须等于本次批准的 frozen
  reference；dirty 或无法识别的 Real source revision 会在连接硬件前被拒绝；
- B 前必须由操作员按 H，先确认 XHand home command accepted，再完成 collision-checked
  canonical arm home；本进程只允许一次 H 尝试。coordinator 同时要求 H 链路完成标志与 fresh
  arm feedback 的 home tolerance，任一不满足都忽略 B；
- XHand 仍不要求关节落入 home tolerance，只要求 home command 被接受；
- `--max-running-seconds <finite positive>`，其值被冻结进 H4 runtime receipt；
- B 后 coordinator 最多写一条 complete coupled arm+hand record，随后停止调度；
- 一个 execute lifecycle 只接受第一次 B；即使 supervisor 尚未观察到 ARMED，后续 B 也会被
  coordinator 忽略；
- arm `last_cmd_seq` 与 hand `accepted_target_action_id` 都须确认同一个 action id；
- 确认成功即撤销 motion 并回到 ARMED；超时、superseded、freshness/feedback 或 publication
  异常均进入 sticky FAULT，绝不重试或发送第二条 policy command。
- ACK 在 coordinator 轮询到 `APPLIED` 后仍须处于 deadline 内；晚到的 `APPLIED` 按 timeout
  处理。进程退出码 0 还要求 `completed=true`，人工 S/Q 的干净停止不是 H4 成功。

`--print-config` / `--preflight-only` 可审计该 H4 projection，但 operational invocation 会连接
真实设备。因此，**不得**绕过 lifecycle、直接调用 coordinator、复用 teleop/replay 入口，或把
离线通过当作真机授权。

## 3. H4 前的软件 gate

执行前必须逐项满足，并把结果附在新的 H4 receipt：

| Gate | 需要的证据 | 失败处理 |
|---|---|---|
| reference identity | checkpoint SHA-256 与 [当前 H2/H3 artifact](deployment_reference_h2h3_shadow_2026-08-31_42bb288.json) 均为 `b174bd483b64090cd3f5dbe0a5bfadd10998f5d27d43fc9aca06efb82242484c` | 停止，不选取“最新” checkpoint |
| H2/H3 baseline | `42bb288` 的 120 s shadow receipt 证明 1,915/1,915 endpoint validated、zero coupled writes、clean shutdown | Python deployment tree、runtime config 或 artifact 改变即停止并重跑 shadow |
| execute enablement diff | H4 保持 one-publication 与原 SafetyGate/worker limits；新增 seed receipt、H home 和 B 前 canonical arm-home gate，不修改 normalizer、collision、freshness 或 generation 语义 | 任一未解释差异都停止 |
| bounded execute guard | execute 的 publication bound 固定为 `1`；ack timeout 与 B-relative duration 均为有限正值，并写入 coordinator receipt | 不允许用无界 execute 替代 |
| offline publication | fake ring 的 full validation 成功只写一条 coherent arm+hand record；gate 后 generation 变化不得写入 | 停止并修复 |
| fault paths | 预期覆盖 stop/e-stop、generation revoke、worker ack 超时、feedback stale、hand preflight reject、collision checker exception、publication failure | 任一路径不 fail-closed 即停止 |
| static checks | focused 与全量离线 tests、`compileall`、Black、isort、`git diff --check` 均通过 | 修复后重跑 |

H4 software guard 必须覆盖 fake-ring、arm-home start gate、coordinator acknowledgement/timeout、
CLI/lifecycle 与 receipt，并对同一 frozen reference 重跑 `--print-config` / `--preflight-only`。
离线通过不替代独立 H4 review、现场 checklist、更新后的 H2/H3 shadow 或明确 H4 授权。

## 4. 现场限定授权应包含的内容

只有在第 3 节全部通过后，才向操作者请求一次新的、明确且单独的授权。授权至少应说明：

- H4、frozen reference v2 SHA、`--hand`、free-space、**execute**；
- 操作员负责确认场地无人、无物体、无障碍，e-stop 就绪，并在 ARMED 后按 B；
- B 前允许启动时 XHand `reset_home` 和操作员按 H 触发的 home command；不要求手指到达
  home 容差。H 同时执行 collision-checked arm home；
- 最大物理 publication 数、最大 B-relative 运行时长、是否允许 one run；
- 操作员随时按 STOP/S，任何异常立即 e-stop；
- 禁止自动升级为 H5、接触、抓取或重复运行。

授权不能由日志、runbook、先前 H2/H3 结果或模型 checkpoint 代替。

## 5. 现场 checklist

在启动硬件前，实施者与操作员逐项确认：

1. 恢复干净、已 review 的 execute enablement revision；无未解释的配置或 artifact 差异。
2. 用该 revision 和与 operational invocation 相同的显式 `--device cuda:0`，对同一 frozen
   artifact 运行无硬件 `--print-config` 与 `--preflight-only`；记录 checkpoint/index/runtime/
   source identity。
3. 确认 xArm、XHand、RealSense、网络/串口、calibration 与 worker SDK 状态均健康；任一初始化
   fault 均取消本次 H4。
4. 清空机械臂与手指潜在 sweep 空间；移除物体；人员退出；e-stop 可立即触及。
5. 操作员明确读回本次 publication/duration 上限与 STOP/e-stop 职责。
6. 系统 ARMED 后，确认日志显示 startup `reset_home` 已下发；由操作员按 H，等待 hand home
   command accepted、arm canonical home 流程完成和 `physical home sequence completed`。只检查
   arm home tolerance，不判断 hand home tolerance。H 失败或误触后退出，不在同一进程重试。
7. 仍处于 ARMED 且场地条件未变化时，仅由操作员按 B。实施者不代按 B，也不自动 begin。

## 6. 运行期间的停止规则

下列任一事件出现即停止当前 level；不得自动重试或升级：

- 操作员 STOP/S、e-stop、人员或物体进入工作空间；
- publication count 或 B-relative duration 达到本次授权上限；
- fatal `SafetyGate`/contract/checker exception 或 worker 反馈故障；typed 的 joint/workspace/
  transition motion reject 与 hand preflight reject 只丢弃当前未发布 endpoint，并计入 receipt；
  若一直没有可发布 endpoint，则由 first-command/B-relative timeout 结束为 FAULT；
- arm/hand/camera/pointcloud/inference/policy heartbeat 或 freshness 故障；
- generation mismatch、ticket superseded/timeout、SDK error、worker exit、unexpected command count；
- reference identity、runtime config 或 calibration provenance 不匹配。

停止后 coordinator 必须撤销 motion，worker 必须完成 verified shutdown。停止原因不是成功指标；
只有完整 receipt 和日志能证明实际 publication 边界。

## 7. H4 evidence bundle 的最低验收内容

coordinator runtime receipt 会作为原子 JSON 文件写到 `$DEXMANI_RECEIPT_DIR`（默认
`~/.dexmani/receipts`）；写入失败会把 session 标为 FAULT。它证明 coupled publication
边界、ACK、metrics 和 monotonic B/publication/receipt timeline。verified shutdown、原始
terminal log 和操作者确认发生在 coordinator 已退出之后，不能伪称由该子进程持久化。

因此 H4 完成后必须运行无硬件的 `examples/seal_h4_evidence.py`，把 completed runtime
receipt、已关闭的 terminal log 和单独准备的 operator-record JSON 原子封存为
`h4_evidence_*.json`。该脚本会拒绝没有 home/run/publication/clean-session 标记、没有完整
monotonic timeline、或没有 one-write/ACK/home gate 的 runtime receipt。runtime receipt、sealed
evidence 和启动时保存的 `--print-config` / `--preflight-only` receipt 一起构成 H4 审计材料。

sealed evidence bundle 至少须持久化以下内容：

- frozen artifact checkpoint/index SHA-256、runtime/source/config identities、准确 argv；
- reset-home accepted、hand home accepted、arm path selected/home reached、B/RUNNING、first/last
  publication、stop、DISARMED 的 terminal-log evidence；runtime receipt 的 B/publication/receipt
  monotonic timeline 与 sealed evidence 的 UTC seal time；
- publication 上限、实际 `coupled_command_writes`、sequence start/end、每个 worker 的 SDK-side
  accepted/executed/duplicate/rejected 计数；
- endpoint due/committed/published/discarded/fatal、roundoff canonicalization、所有 safety reject code；
- `physical_home_completed=1`，且原始日志包含 hand home accepted、arm path selected、arm home
  reached 和 physical home sequence completed；
- p50/p95/p99 observation/inference/plan/horizon，worker exit status 与 final safety state；
- 原始日志的 path、size、SHA-256，及人员/设备/场地确认的操作者记录。

操作者在启动前准备一个不含密钥的 JSON record，至少包含 `operator`、`scene`、
`e_stop_ready: true` 和本次 `authorization`。H4 process 已退出、`tee` 已关闭后，执行：

```bash
python examples/seal_h4_evidence.py \
  --runtime-receipt /absolute/path/to/h4_execute_*.json \
  --terminal-log /absolute/path/to/terminal.log \
  --operator-record /absolute/path/to/operator_record.json \
  --output-dir "$DEXMANI_RECEIPT_DIR"
```

命令只读取既有证据并写入新的 `h4_evidence_*.json`，不导入硬件 SDK，也不连接设备。

H4 的唯一通过判定是：实际 publication 数不超过授权上限、每条 publication 与 receipt/worker
证据一致、没有 safety/freshness/checker/hardware fault、按授权正常停止并 verified clean shutdown，
且 sealed evidence bundle 成功写入。
这不自动放行 H5 或任务级实验；下一 level 仍需要独立 review 和授权。
