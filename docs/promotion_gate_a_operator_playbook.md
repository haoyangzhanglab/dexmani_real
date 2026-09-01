# Promotion Gate A：操作者行动手册

> 目标：用最少人工阅读完成剩余真机 Gate，并把 evidence、Git、merge、rebase 等工作继续交给 Codex。
>
> **一次只授权一个物理阶段。Live Shadow 的授权不包含 H4；H4 的授权不包含 task rollout。**

---

## 1. 当前状态

| 项目 | 状态 |
|---|---|
| R1 `control_action` 语义 | PASS / MERGED |
| R2 timing / readiness | PASS / MERGED |
| Gate A offline | PASS |
| frozen pre-live inspect | PASS |
| frozen isolated CUDA preflight | PASS |
| R3 CLI / logging | APPROVED / FROZEN / UNMERGED |
| Gate A live shadow | **下一步** |
| fresh H4 one endpoint | BLOCKED：等待 live shadow review |
| task rollout | BLOCKED |
| R4 / R5 | BLOCKED：等待完整 Gate A |

### 冻结身份

当前 Gate A 真机验证只接受以下 baseline：

```text
Real
  effe745c68847a4b32ed1e4680041a350da4f4fe

Policy
  fc6b7dfb45748f4187f2e82b5425721ed02b028e

Checkpoint SHA-256
  28ff79a6ca5d5b746bbde877ff96abbb88543539f4c73ef554348184f446effc

Device
  cuda:0

Inference seed
  1066
```

相关冻结分支：

```text
Gate A offline evidence
  codex/real-gate-a-offline
  a70df31aae6d00a004f927a679ece813efc1a4d7

R3
  codex/real-r3-cli-logging
  a0d9972bce160c3bbf61072749196d7bc434f6b9
```

---

## 2. 已完成，不要重复

只要上述 identity 没有变化，以下工作已经完成：

- artifact inspect；
- exact producer strict restore；
- isolated CUDA preflight；
- recorded observation replay；
- hardware-free multiprocess shadow；
- timing / generation / no-write negative checks；
- R3 offline check、CLI、logging review；
- R3 XHand native SDK no-hardware audit hardening。

不要为了“更保险”重复这些步骤。若 identity 发生变化，则停止并重新 review，不自动换 baseline。

---

# Stage 1 — Gate A Live Shadow

## 3. 目标

Live Shadow 只验证 offline 无法证明的部分：

```text
真实 camera/state
    ↓
causal ObservationBatch
    ↓
GPU Policy inference
    ↓
8-step control_action
    ↓
R2 timing / deadline
    ↓
ActionBuffer
    ↓
SafetyGate
    ↓
SHADOW_VALIDATED
```

预期 learned coupled command writes 必须为 0。

这不是 task success test，也不要求完成抓取。

---

## 4. 操作者需要确认的内容

无需阅读源码，只确认：

### Identity

- [ ] Real SHA 与本手册一致
- [ ] Policy SHA 与本手册一致
- [ ] 两个 repo clean
- [ ] checkpoint SHA 与本手册一致
- [ ] device = `cuda:0`
- [ ] seed = `1066`

### Physical environment

- [ ] e-stop 可用
- [ ] workspace 无人员、无临时障碍
- [ ] 已知晓 hand startup 可能产生 reset/home 相关副作用

任一项不满足：不要运行。

---

## 5. 如何执行

不要重新拼接或修改真机命令。

使用冻结 Gate A evidence 中已经生成并 review 过的 **Live Shadow operator command**：

```text
branch:
  codex/real-gate-a-offline

commit:
  a70df31aae6d00a004f927a679ece813efc1a4d7

file:
  docs/policy_gate_a_offline_report.md

section:
  Operator-only next steps → Live shadow
```

该命令是 operator-only，会连接真实硬件；不要交给 Codex 自动执行。

---

## 6. Live Shadow PASS 条件

只关注以下结果：

```text
inference ready                     YES
required workers ready              YES
ARMED → RUNNING                      正常
valid policy plans                   > 0
SHADOW_VALIDATED                     > 0
coupled command writes               0
inference failure                    0
first-command timeout                0
command-silence abort                0
source/deadline/generation fatal     0
bounded shutdown                     clean
```

### 立即停止条件

若出现以下任一情况，停止并进入 review，不要现场放宽参数：

- unexpected coupled command write；
- FAULT / supervisor crash；
- 非预期机械运动；
- source/deadline/generation 持续异常；
- inference/coordinator exception；
- sensor freshness 长时间无法满足；
- operator 无法确认设备状态。

**Live Shadow 后必须先 review，不能直接继续 H4。**

---

## 7. 需要保存的 evidence

运行结束后保留：

- terminal output；
- session log；
- shadow receipt / metrics；
- Real / Policy SHA；
- seed；
- checkpoint SHA。

优先检查：

```text
inference_ms p50/p95/p99
observation_age_ms
observation_skew_ms
plans_created
plans_ingested
plans_stale
plans_generation_dropped
endpoints_shadow_validated
safety_rejections
coupled_command_writes
```

将 evidence 提交给 reviewer，等待 `LIVE SHADOW PASS / READY FOR FRESH H4`。

---

## 8. 授权模板

操作者只需要明确授权本阶段：

```text
授权执行 Gate A Live Shadow。
使用冻结 baseline，只执行已 review 的 bounded shadow，不进入 H4。
```

---

# Stage 2 — Fresh H4 One Endpoint

## 9. 当前状态

```text
BLOCKED ON LIVE SHADOW REVIEW
```

只有 reviewer 明确给出：

```text
LIVE SHADOW PASS / READY FOR FRESH H4
```

之后才进入本阶段。

H4 只验证：

```text
same frozen identity
    ↓
SafetyGate
    ↓
exactly one learned coupled publication
    ↓
arm + hand acknowledgement
    ↓
stop
```

PASS 条件：

- exactly 1 learned coupled endpoint published；
- arm ACK success；
- hand ACK success；
- publication bound严格生效；
- 没有第二 endpoint；
- 无 SafetyGate / expiry / generation fault；
- clean shutdown。

若 endpoint 被 SafetyGate合理拒绝，保存 evidence并 review，不放宽 safety limits。

H4 命令应在 Live Shadow review 后，根据同一冻结 identity 重新确认后使用；不要提前执行历史 H4命令。

### H4 授权模板

```text
授权执行 Gate A Fresh H4。
保持 Live Shadow 相同 identity，只允许 1 个 learned coupled endpoint；不进入 task rollout。
```

---

# Stage 3 — Complete Gate A Evidence

## 10. H4 PASS 后交给 Codex

H4 review通过后，让 Codex完成：

```text
Gate A offline evidence
+ live shadow evidence
+ fresh H4 evidence
        ↓
final Gate A receipt / report
        ↓
Promotion Gate A = COMPLETE
```

Codex只做 evidence / Git 工作，不运行任何真机命令。

最终 evidence至少冻结：

- Real / Policy SHA；
- checkpoint SHA；
- seed；
- offline replay PASS；
- multiprocess shadow PASS；
- live shadow PASS；
- live shadow coupled writes = 0；
- fresh H4 publication = 1；
- arm/hand ACK evidence；
- clean shutdown evidence。

---

# Stage 4 — Merge Gate A，然后集成 R3

## 11. 顺序

```text
Gate A COMPLETE
    ↓
merge Gate evidence
    ↓
rebase / transplant frozen R3 onto Gate-complete main
    ↓
rerun offline regression + representative check
    ↓
review
    ↓
merge R3
```

不要先 merge R3 再完成当前 Gate A，否则会改变 physical baseline。

R3 integration 后至少重新运行：

```text
compileall
policy CLI tests
preflight tests
timing tests
ActionBuffer tests
artifact / manifest tests
hardware-isolation tests
representative offline check
```

如果 rebase只包含预期 CLI/logging diff，通常不需要重新做完整 H4；是否需要短 shadow smoke由最终 diff review决定。

---

# Stage 5 — R4 / R5

## 12. R4

完整 Gate A 后才正式开始 causal reader consolidation：先 differential equivalence test，再去重，不改变 observation selection semantics。

## 13. R5

继续延后。在 R4完成并重新建立稳定 baseline前，不修改：

- ActionBuffer；
- transport `valid_mask`；
- deadline ownership；
- command fences。

---

# 14. 最短操作路径

如果只看这一节：

```text
现在：
  授权并执行一次 frozen Live Shadow
        ↓
  保存 log / receipt
        ↓
  review

通过后：
  授权 Fresh H4 one endpoint
        ↓
  保存 evidence
        ↓
  review

通过后：
  Codex 完成 Gate A evidence / merge
        ↓
  Codex rebase / validate / merge R3
        ↓
  开始 R4
```

**当前唯一需要人工执行的下一步：Gate A Live Shadow。**
