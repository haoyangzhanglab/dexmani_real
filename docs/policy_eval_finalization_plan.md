# DexMani Policy Eval 最终收尾方案（Research-Focused）

> 状态：**当前执行依据 / finalization plan**  
> 项目定位：**个人博士论文实验仓库，不是 production deployment platform**  
> Fact-check 基线：
> - `dexmani_real/main @ 6634b79e1770c96ccdcae461facb07d4c992c5db`
> - `dexmani_policy/main @ 4588bb254dfd01e8f329d574b6efcf40c4a377ec`
> - 日期：2026-09-11
> - Hardware validation：**NOT RUN**
>
> 本文用于指导当前 `run_policy` / dataset / deployment 收尾。`docs/policy_eval_refactor_plan.md` 记录了主重构设计；主架构现已基本落地，后续剩余工作以本文为准。不要再根据旧计划扩大重构范围。

---

## 0. 最终判断

当前主架构已经正确，不需要重新设计：

```text
Policy artifact
→ explicit artifact pinning
→ runtime inference_steps
→ inference-first startup
→ one persistent real-robot session
→ H → scene → B → S/timeout
→ canonical raw v29 episode
→ recorder finalize
→ completed_episodes++
→ next episode
→ offline task evaluation
```

现阶段只处理会影响以下四件事的问题：

```text
1. 论文实验结果是否正确
2. 真机执行是否安全可靠
3. train/deploy scientific semantics 是否一致
4. NFE / latency 结论是否被明显工程 overhead 污染
```

不再追求：

```text
production-grade workflow framework
通用 experiment manager
通用 retry/recovery system
完整 observability/provenance platform
未来所有 modality/model 的提前支持
```

---

# 1. 研究仓库下的优先级原则

按照下面优先级做决定：

```text
P0  会让实验链路直接失败或产生错误 scientific semantics
P1  会让实验结果/测量不可信，但不会立即 crash
P2  小范围 operator / robustness bug，修复成本很低
P3  纯文档/可读性问题
```

额外原则：

> **能通过一个小 patch + 一个针对性 test 解决的问题直接修；需要新增 framework 才能“更完善”的问题不做。**

> **性能优化先测量，再决定是否修改。没有实测 bottleneck，不为“可能更快”增加 runtime 分支。**

---

# 2. 本轮必须完成的内容

## Work A — EEF data / deployment contract 闭环（P0/P1）

这是本轮唯一真正的 P0。

### A1. 修复 processed → Zarr EEF semantics 丢失

当前事实：

```text
processed v19:
    data/eef_pose                  ✅
    eef_pose_frame                 ✅
    eef_pose_components            ✅
    eef_pose_derivation            ✅
    eef_pose_algorithm_id          ✅

Policy Zarr v12:
    data/eef_pose                  ✅
    上述四个 EEF attrs             ❌ 当前 export allowlist 未透传

Policy deployment exporter:
    _validate_eef_pose(...)        ✅ 严格要求四个 attrs
```

因此真实链路现在会在 Policy export 边界失败。

修改：

```text
dexmani_real/dataset/export.py
```

在 `_inspect_artifact()` 的 `semantic_keys` 中增加：

```python
"eef_pose_frame",
"eef_pose_components",
"eef_pose_derivation",
"eef_pose_algorithm_id",
```

只做 semantic propagation，不新增新的 contract abstraction。

### A2. processed validator 校验 `eef_pose` rot6d

修改：

```text
dexmani_real/dataset/processed.py
```

当前 `action_ee[..., 3:9]` 已调用 `validate_canonical_rot6d()`；`eef_pose[..., 3:9]` 应使用同一 validator：

```python
if key == "eef_pose":
    validate_canonical_rot6d(
        block[:, 3:9],
        label=f"{label}: eef_pose rot6d",
    )
```

目的不是增加复杂 validation，而是让 persisted representation 的实际 payload 与其声明的 `position_m_rot6d` 一致。

### A3. 不 bump schema

本修复只是把已经声明的：

```text
processed v19
Policy Zarr v12
```

contract 实现完整。

**不要再 bump：**

```text
raw v29
processed v20
Zarr v13
```

### A4. EEF tests

最少增加以下 tests：

```text
1. processed canonical FK → eef_pose [N,9] PASS
2. finite 但非法的 eef_pose rot6d → validator FAIL
3. processed eef_pose_* attrs → Zarr attrs verbatim propagation
4. Real-produced Zarr → dexmani_policy _build_observation_contract(... eef_pose ...) PASS
```

第 4 项是最重要的跨仓库 contract test。

---

# 3. 明确边界：不要伪造 EEF / Tactile model support

当前 `dexmani_policy` exporter 已经能描述：

```text
eef_pose
tactile_force
```

但当前主要 Policy encoders 实际仍只消费例如：

```text
joint_state + point_cloud
joint_state + rgb
```

而 restore 会严格检查：

```text
artifact observation_fields
==
agent.obs_encoder.consumed_observation_fields
```

因此当前正确状态是：

```text
EEF/Tactile data contract available
≠
current ActionFlow/ManiFlow/DP3 already consumes EEF/Tactile
```

本轮 **DO NOT**：

```text
为了让 EEF artifact restore 成功，直接把 eef_pose 塞进 consumed_observation_fields
给现有 encoder 增加一个没有研究假设的临时 concat
增加“通用 modality adapter”
增加 generic fusion framework
```

如果论文后续真的研究：

```text
PCD + EEF
PCD + Tactile
PCD + EEF + Tactile
```

再以独立研究任务设计 encoder/fusion/normalizer/ablation。

当前 qualification 只需要证明：

```text
Real data producer
→ processed
→ Zarr
→ Policy data/deployment contract parser
```

是闭环的。

---

# 4. Work B — 两个低成本 Runtime 边界修复（P2）

## B1. Policy 中 C/D 必须是真正 no-op

当前 Policy UI 定义：

```text
C = unsupported / ignored
D = unsupported / ignored
```

但 `operator.py` 的 batch logic 仍把 `PAUSE/DISCARD` 算进会阻止 `H` 的 `stop_in_batch`。

这是语义不一致。

修改：

```text
dexmani_real/deployment/operator.py
```

让 Home blocking 只受真正 terminal/control signals 影响：

```text
STOP
QUIT
EMERGENCY_STOP / estop latch
```

不要让：

```text
PAUSE
DISCARD
```

影响 H。

保持 global `runtime/operator_input.py` 的 C/D mapping 不变，因为 teleop 仍使用它们。

测试：

```text
[C, H] → C warning，H 仍执行
[D, H] → D warning，H 仍执行
[S, H] → H blocked
[Q, H] → H blocked
ESC during/around H → existing estop path unchanged
```

不要为 keyboard batch 新建状态机。

## B2. Recorder START error 做最小分类

当前 `RecorderClient.start_episode() == False` 可能表示：

```text
A. 暂时不能开始 / sequencing rejection
B. RecorderIO 已返回明确 terminal start error
```

二者不应完全等价。

目标行为：

```text
start=False + last_error=None
→ reject this B
→ remain ARMED
→ allow retry

start=False + last_error!=None
→ no physical motion has started
→ latch recording fault
→ supervisor cleanly ends session
```

典型 persistent error：

```text
explicit episode name collision
EpisodeRecorder start resource failure
invalid immutable recording metadata
```

不要新增 Recorder error enum、retry manager 或 recovery framework。

测试：

```text
retryable false/no-error → no motion, no error_state
terminal false/error → no motion, error_state=True
```

---

# 5. Work C — Policy inference efficiency 改为“先测后改”

这是相对上一版方案最重要的收缩。

当前 `LoadedPolicy.predict_action_chunk()` 通过完整 `validate_prediction()`，会执行：

```text
shape validation
finite validation
control_action == canonical pred slice
pred_action CPU clone
control_action CPU clone
```

其中 Real 最终只需要 future `pred_action` chunk，因此存在固定 overhead 的可能。

但是本项目是个人论文实验仓库：

> **没有证明是 bottleneck，就不要为了理论上的几百微秒增加第二套 runtime validation path。**

## C0. 先 benchmark，不先改代码

对目标真实 Policy / GPU 做一次本地 benchmark：

```text
NFE = 1 / 2 / 4 / artifact default
warm GPU
固定 observation
```

至少比较：

```text
T_model_like      = agent.predict_action / solver 主体
T_runtime_total   = LoadedPolicy.predict_action_chunk wall time
fixed_overhead    = T_runtime_total - T_model_like
```

不需要建设 benchmark framework；一次研究脚本或临时 profiling 即可。

### 决策规则

如果 fixed overhead：

```text
< 0.5 ms
或
< 5% of runtime_total
```

则：

```text
NO CHANGE
```

保留当前严格 path，优先可靠性。

如果 1-step / 2-step 下 fixed overhead 明显影响论文 latency 结论，再做 C1。

## C1. 如确有 bottleneck，先只消除无用 CPU copy

优先目标：

```text
保留当前完整 tensor validation
但 realtime predict_action_chunk 不 materialize 不需要的完整 PredictionSnapshot CPU copies
```

建议把：

```text
GPU tensor validation
```

与：

```text
CPU PredictionSnapshot materialization
```

拆开。

完整 `validate_prediction()` 继续服务：

```text
export smoke
qualify
predict()
offline parity
```

`predict_action_chunk()` 可在同一 validated GPU tensors 上：

```text
slice future pred_action
→ one GPU→CPU transfer
→ return owned NumPy chunk
```

不要一开始就删除：

```text
finite check
canonical control slice check
```

## C2. aggressive fast path 只在实测需要时做

只有 profiling 明确证明 GPU validation synchronization 本身成为 1-step/2-step 的主要固定开销，才考虑进一步简化 realtime validation。

该优化必须单独 review，不和 Work A/B 混在同一 patch。

---

# 6. Work D — 只修会误导下一轮开发的 stale docs

个人研究仓库不需要清理所有历史文档，只修“仍像 current instruction”的错误描述。

## dexmani_real

### `docs/multimodal_dataset_last_mile_repair_plan.md`

顶部明确增加：

```text
STATUS: SUPERSEDED
Current implementation: raw v29 → processed v20 → Policy Zarr v13
```

历史正文可以保留。

### `docs/tactile_unit_si_verification_pending.md`

未来 SI migration 不要再写死：

```text
processed v19 / Zarr v12
```

改成：

```text
next processed / Policy-Zarr schema version
```

因为 v19/v12 已被 EEF 使用。

### `dexmani_real/deployment/lifecycle.py`

把 stale：

```text
single-rollout policy deployment lifecycle
```

改成当前真实的 persistent multi-episode wording。

## dexmani_policy

### `docs/policy_deployment_simplification_guide.md`

当前仍有旧描述称 exporter 不支持 `eef_pose/tactile_force`。

更新为：

```text
exporter/data contract now admits eef_pose/tactile_force;
existing model encoders do not automatically consume them.
```

这是很重要的职责边界，避免后续 Claude 误判“contract support = model support”。

---

# 7. Work E — 最小但真实的 Software Qualification

不要再靠两个仓库各自 synthetic fixture 推断 integration 一定正确。

## E1. Data / contract qualification

执行：

```text
synthetic current raw v29
→ process_episode_root()
→ processed v20
→ validate_processed_hdf5()
→ export_processed_hdf5_to_zarr()
→ Policy Zarr v13
→ inspect data/eef_pose
→ inspect eef_pose_* attrs
→ dexmani_policy _build_observation_contract(... eef_pose ...)
```

必须 PASS。

同时：

```text
invalid finite rot6d
→ processed validation FAIL
```

## E2. 不要求不存在的 EEF model consumer

当前不要把 qualification 写成：

```text
EEF Zarr
→ current ActionFlow restore with eef_pose
→ PASS
```

这是错误目标，因为当前 encoder 没有消费 EEF。

真正 model-level qualification 只对当前已经实际支持的 modality combination 进行。

---

# 8. Real v19/v12 数据 rebuild 改成“论文需要时才做”

当前 tracked salvage evidence 仍是历史：

```text
processed v18
Policy Zarr v11
60 accepted episodes
14112 rows
```

代码已经进入 v19/v12，但是否立即重建全部历史数据取决于当前论文实验是否真的要使用 `eef_pose`。

## 如果当前论文近期要训练/分析 EEF

则在 Work A + E 通过后：

```text
rebuild accepted raw episodes
→ processed v19
→ Zarr v12
→ verify row counts unchanged
→ verify eef_pose == canonical FK(joint_state[:,:7])
→ verify Zarr arrays exactly equal concatenated processed arrays
→ update/add manifest evidence
```

## 如果当前论文暂时不用 EEF

则：

```text
不需要为了“版本整齐”立刻重建 60 episodes
```

保留现有 v18/v11 salvage artifact 作为历史已验证 evidence；新数据/需要 EEF 的实验按 v19/v12 重新构建即可。

这比为了 schema version 一致性消耗时间更符合个人研究仓库定位。

---

# 9. Hardware smoke 与数据 rebuild 解耦

当前 RunPolicy 主链的 hardware smoke 不依赖 EEF dataset rebuild。

Work A/B software tests 通过后，使用一个当前已经 qualify、真实 encoder 支持其 observation_fields 的 Policy artifact 做：

```bash
python examples/run_policy.py \
    <existing-supported-experiment> \
    --num-episodes 2 \
    --max-duration 10
```

人工检查：

```text
1. inference ready before hardware
2. B without H rejected
3. H → scene → B
4. S fences motion
5. episode_001 finalizes and publishes
6. returns ARMED
7. B without fresh H rejected
8. H → scene → B
9. episode_002 finalizes and publishes
10. only after 2/2 publication → clean shutdown
11. run_config.yaml correct
12. no result.json / online task outcome
13. EpisodeReader can open both episodes
```

Hardware validation 仍必须人工执行；Claude Code 不自动连接机器人。

---

# 10. 本轮明确不做

以下事项即使“更完整”，当前都不值得做：

```text
新的 Manager / Service / Registry / Plugin
通用 Recorder retry/recovery framework
通用 modality fusion adapter
为了 EEF/Tactile 修改现有 encoder 而没有研究假设
为了版本整齐强制重建所有历史 dataset
新的 raw schema
新的 eval storage format
prediction trace framework
experiment database
Git/SHA provenance 恢复
online SUCCESS/FAILURE classifier
自动 task success rate
自动 hardware test
```

同时保持不动：

```text
raw v29
PointCloud scientific contract
Fingertip geometry contract
Tactile calibration/unit contract
causal observation alignment
run_generation fencing
scheduler
IK
SafetyGate
teleop C/D semantics
```

---

# 11. 推荐实施顺序

```text
Phase 1 — 必须 correctness
A1 EEF attrs propagation
A2 EEF rot6d validation
A4 focused tests

Phase 2 — 小型 runtime 修复
B1 C/D-H batch semantics
B2 Recorder START terminal error fail-closed

Phase 3 — docs
D stale active docs only

Phase 4 — software qualification
E raw → processed → Zarr → Policy contract

Phase 5 — performance decision
C0 benchmark
→ no bottleneck: stop
→ bottleneck: C1 minimal copy optimization
→ still bottleneck: separately review C2

Phase 6 — only if paper needs it
real v19/v12 rebuild + evidence

Phase 7 — hardware
2-episode manual smoke
```

这样优先保证：

```text
正确
→ 能跑
→ 能验证
→ 有必要才优化
```

而不是：

```text
先把工程做得“完整”
→ 再寻找研究价值
```

---

# 12. 推荐 commit 划分

保持小而容易 review：

```text
dexmani_real
1. fix: complete EEF processed-to-Zarr contract
2. fix: tighten policy operator and recorder start boundaries
3. docs: sync current policy/dataset finalization status
```

如实测需要性能优化：

```text
dexmani_policy
4. perf: avoid redundant deployment chunk materialization
```

数据 rebuild 不作为源码 commit 的一部分；只在验证完成后更新 evidence/manifest。

---

# 13. 最终验收标准

本轮代码收尾完成应满足：

```text
[Correctness]
EEF attrs processed → Zarr 不丢失
EEF invalid rot6d 被 owning validator 拒绝
Policy contract parser 接受 Real-produced EEF Zarr

[Runtime]
C/D 在 Policy deployment 中是真正 no-op
S/Q/ESC 的原有 safety 语义不变
Recorder terminal START error 不进入无限重试循环
Recorder ACK 前绝不进入 physical RUNNING

[Architecture]
artifact pinning 保持
runtime inference_steps 保持
persistent multi-episode 保持
fresh H per episode 保持
finalization 后计数保持
raw v29 保持
offline task outcome 保持

[Efficiency]
完成一次真实目标 Policy 的 validation-overhead benchmark
只有实测显著时才修改 hot path

[Scope]
没有伪造 EEF/Tactile model support
没有新 generic framework
没有为了历史 artifact 版本整齐做无研究价值工作
```

---

# 14. 一句话决策原则

> **这个仓库的目标不是证明 deployment infrastructure 可以无限扩展，而是用最少、最可信的代码支撑论文中的真实机器人实验。Correctness 和 safety 直接修；实验会用到的 contract 做闭环；性能先测再优化；暂时没有研究假设的 modality/model 扩展不做。**
