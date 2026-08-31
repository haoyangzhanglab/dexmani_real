# DexMani Real 策略部署解耦实施方案

> 版本：2026-08-29 review-optimized revision
>
> 状态：R0–R4 已完成；H4 仍须单独 review 和明确授权。
>
> 唯一操作者入口：`examples/run_policy.py`

入口要求操作者显式提供 inference `--device`，避免部署无意回落到默认 CPU。当前 frozen
reference 的离线基准与后续真机验证统一使用 `--device cuda:0`。

## 1. Review 结论

当前方向正确：训练与仿真保持在 `dexmani_policy`，真实机器人部署适配集中到
`dexmani_real`。但撤回 Policy 原型后，现有 Real 加载链并未真正独立，必须先修复以下问题：

| 优先级 | 已核实问题 | 处理结论 |
|---|---|---|
| P0 | `deployment/preflight.py` 仍导入已撤回的 Policy `load_deployment_checkpoint_stream` | 在 Real 新增 deployment-v2 stream decoder |
| P0 | `integrations/dexmani_policy.py` 仍导入已撤回的 metadata/data-contract/restore helpers | 用 Real 自己的结构、语义和 strict-restore 校验替换 |
| P0 | artifact receipt 把 `pad_before` 硬编码为 `1` | 改为 `n_obs_steps - 1`；当前 DP3 的 `1` 只是实例值 |
| P1 | Policy provenance 被记录，但 dirty source 尚未统一作为 operational gate | 在 inference ready 之前要求 commit 匹配且工作树干净 |
| P1 | 原方案混合历史 Batch、实机记录与未来任务，存在互相覆盖的要求 | 当前文档只保留下一阶段可执行规格；历史证据引用独立文档 |

最小修复不是把训练框架复制进 Real，也不是恢复 Policy 原型。Real 只实现冻结 deployment
wire format 的读取与真实机器人语义校验；模型结构继续由已提交 Policy agent class 提供。

### 1.1 只读可行性验证

已在无硬件条件下直接检查 frozen v2：

- `torch.load(..., weights_only=True)` 成功，顶层 exact keys 为 `_format/state/weights`；
- state 含 inference/data/train/producer/deployment contracts，weights 只含 model/EMA；
- model 与 EMA 各有 178 个 canonical tensor keys，没有 `module.` 或 `_orig_mod.` prefix；
- embedded config 可在不注册训练 resolver 的情况下完全 resolve；
- clean Policy `7e31d10` 可直接 instantiate `dexmani_policy.agents.core.dp3.DP3Agent`；
- EMA `load_state_dict(strict=True)` 全部匹配，agent dimensions 为 `2/8/19/16/19`；
- restored normalizer keys 精确为 `action/joint_state/point_cloud`。

因此 R0/R1 不需要修改 Policy 或重新生成 reference artifact。

## 2. 权威边界

### 2.1 `dexmani_policy`

本阶段保持 commit `7e31d10e7a31ff3d12df31b8683c9c90b357cbc5` 与干净工作树：

- 不修改 dataset、sampler、replay buffer、checkpoint IO、trainer、workspace 或 eval；
- 不改变已有 sim checkpoint、EMA、normalizer 或 `predict_action` 行为；
- 不要求仿真模型迁移到新 checkpoint 格式；
- Real inference child 只允许导入已提交的 agent class，以及这些 agent 自身的正常依赖；
- 不依赖 Policy 的 deployment、Real Zarr 或机器人安全 helper。

另一台机器上的 Policy 开发完成后，按
[Policy 后续合并清单](dexmani_policy_integration_followup.md) 独立 review。它不能隐式改变当前
frozen artifact 或本阶段 Real loader 的含义。

### 2.2 `dexmani_real`

Real 是以下行为的唯一 owner：

- experiment selector、sidecar、文件 identity、SHA-256 与 producer provenance；
- deployment-v2 payload 的受限反序列化和 exact-schema validation；
- inference config、data contract、train params 与 runtime 的交叉校验；
- model/EMA 选择、state dict strict load 和 normalizer 完整性检查；
- observation adaptation、prediction decoding、SafetyGate、publication、receipt 与硬件生命周期。

### 2.3 frozen artifact

当前 reference 是只读外部输入，不因撤回 Policy 源码而删除或重写：

```text
experiment:
  /home/zhanghaoyang/Desktop/dexmani_policy/experiments/dp3/pick_place_toy/2026-08-28_13-59_42
checkpoint:
  epoch=1126-step=00080000-deployment-v2.pt
sha256:
  b174bd483b64090cd3f5dbe0a5bfadd10998f5d27d43fc9aca06efb82242484c
producer commit:
  7e31d10e7a31ff3d12df31b8683c9c90b357cbc5
```

本阶段不新增 exporter，不修改 experiment selector。未来若需要生成新 artifact，作为单独的
离线工具与写入授权处理，不能混入运行时修复。

## 3. 不可变数据与模型合同

### 3.1 observation window

通用训练 sampler 合同为：

```text
pad_before = n_obs_steps - 1
pad_after  = n_action_steps - 1
padding_semantics = repeat_edge
```

当前 DP3 为 `n_obs_steps=2`、`n_action_steps=8`，因此 artifact 实例记录 `1/7`。

padding 只描述训练序列边缘。Real runtime 按 B 后必须收集 `n_obs_steps` 个真实、不同、因果有效的
observation；不得用 reset-home、ARMED 或上一 generation 的 feedback 填充。

### 3.2 action window

```text
action_key             = action
action_dim             = 19
horizon                = 16
required_action_steps  = horizon - (n_obs_steps - 1) = 15
control_dt_s           = 0.0625
```

adapter 消费完整 15-step actionable future。它不 retime、不补发过期 action，也不改变模型数值。

### 3.3 model、EMA 与 normalizer

- `eval.use_ema=true` 时 EMA weights 必须存在；不允许退回 raw model；
- artifact weights 必须已经使用 canonical state-dict keys；Real 对 `_orig_mod.`、非预期 DDP
  prefix 或 schema 外对象直接拒绝，不在部署时猜测修复；
- `agent.load_state_dict(..., strict=True)` 是唯一 restore；
- restore 后必须存在 `action`、`joint_state`、`point_cloud` normalizer；
- 每个 `scale/offset` 必须是一维、finite、维度匹配，且 scale 非零；
- 不重新拟合、不 clip、不替换 identity normalizer。

### 3.4 Real 数据语义

artifact 中的 frozen data contract 必须与 Real runtime 精确匹配：task、schema/version、control
period、observation/action alignment、point-cloud frame/count/features/pipeline identity、joint/action
维度和 padding 公式。

部署只消费这份 hash-bound evidence，不在运行时打开训练 Zarr，也不修改 Policy dataset。

## 4. 目标加载链

### 4.1 `--print-config`

```text
parse CLI
→ standard-library artifact resolver
→ validate selector/sidecar/allocation/provenance
→ render canonical receipt
→ exit
```

约束：不导入 torch、Policy、camera、robot SDK 或 lifecycle。

### 4.2 `--preflight-only`

```text
resolve immutable artifact
→ spawn isolated inference child
→ open checkpoint with held no-follow directory/file descriptors
→ verify file identity and full SHA-256
→ verify Policy package origin, producer commit and clean source tree
→ Real-owned weights_only deployment-v2 decode
→ exact embedded/sidecar/runtime contract validation
→ instantiate only cfg.agent from resolved embedded config
→ strict restore selected model/EMA state
→ validate normalizer and loaded manifest
→ deterministic synthetic observation prediction
→ recheck file identities and emit receipt
→ exit
```

任何失败都发生在硬件 worker 启动前。不得退回 `latest.pt`、Policy `CheckpointStore.load`
（其 clean baseline 使用 `weights_only=False`）、外部 YAML 或第二次 checkpoint load。

### 4.3 operational shadow/execute

operational path 复用同一个 checked loader。只有 inference ready 后才启动 camera、pointcloud、arm
和 hand workers。B 前仍只下发一次 XHand reset-home；SDK 接受即可，不判断 home tolerance。

R0–R3 已完成离线验证；H4 runbook 继续暂停，直到新的 H2/H3 shadow evidence 通过。

## 5. 最小代码设计

### 5.1 新增 `dexmani_real/deployment/policy_checkpoint.py`

该模块只在 inference child 中使用，职责限定为：

- `LoadedPolicyCheckpoint` frozen dataclass；
- `torch.load(stream, map_location="cpu", weights_only=True)`；
- deployment-v2 payload/state/weights exact keys 与 plain metadata 校验；
- tensor state dict、epoch/global step、inference/data/train/producer contract 校验；
- embedded contract canonical JSON SHA-256；
- 返回 model/EMA tensor mappings，不 import `dexmani_policy`。

不加入 exporter、training checkpoint compatibility、registry、factory 或通用 checkpoint 框架。

### 5.2 修改 `deployment/preflight.py`

- 用 Real decoder 替换 Policy `load_deployment_checkpoint_stream`；
- 保留现有 held-fd、no-follow、identity recheck、full hash 和单次 deserialize；
- package provenance gate 在 Policy import 之前完成；
- checkpoint 对象只通过内存传给 adapter，不再次按路径打开。

### 5.3 修改 `integrations/dexmani_policy.py`

- 删除对 Policy checkpoint/data/eval deployment helper 的 import；
- 只接受 `LoadedPolicyCheckpoint`；
- 自己完成 embedded contract 与 runtime cross-check；
- embedded config 必须是 fully resolved plain mapping，不允许 `${...}` interpolation；
- `_target_` 只允许 `dexmani_policy.agents.*`；只 instantiate `agent`，不 instantiate dataset 或
  `env_runner`；
- 根据 `eval.use_ema` 选择 exact state，直接 strict load；
- 将 `pad_before == 1` 改为 `pad_before == n_obs_steps - 1`；
- 保留现有 15-step prediction、manifest、normalizer 与 finite/shape validation。

### 5.4 package provenance gate

对 print、preflight 和 operational path 使用同一身份结构：

- print 只显示 artifact producer，不导入 Policy；
- preflight/operational 要求 installed Policy commit 等于 producer commit；
- editable checkout 必须 `dirty=false`；
- import 前后 Python tree hash 必须一致；
- origin 不能位于 experiment 目录内。

不为 dirty checkout 增加 execute override。

## 6. 分阶段实施与验收

### R0 — Real-owned decoder

修改：新增 `deployment/policy_checkpoint.py` 和纯离线测试。

验收：

- synthetic v2 正常解析；
- extra/missing keys、非 plain metadata、NaN/Inf、错误 tensor mapping、缺 EMA 均拒绝；
- `weights_only=True`，单次 deserialize；
- decoder import 不加载 Policy 或硬件模块。

状态：已完成。新增 decoder 只接受 `weights_only=True` 的 deployment-v2 exact schema。

### R1 — adapter strict restore

修改：`integrations/dexmani_policy.py`。

验收：

- 不再 import 四个已撤回 Policy helper；
- `n_obs_steps=2 → pad_before=1` 通过；`n_obs_steps=3 → pad_before=2` 通过；错误值拒绝；
- EMA 缺失、normalizer 缺失/维度错/scale=0、state key mismatch 均拒绝；
- test-double agent strict restore 和 15×19 prediction 通过。

状态：已完成。adapter 不再导入已撤回的 Policy deployment helper。

### R2 — preflight 与 provenance

修改：`deployment/preflight.py`、必要的 run identity/CLI tests。

验收：

- clean Policy `7e31d10` 与 reference producer 匹配；
- dirty/mismatched Policy、checkpoint hash mismatch、TOCTOU identity change 均在硬件前拒绝；
- `--print-config` 保持无 torch/Policy/hardware import；
- `--preflight-only` 完成一次实际 reference load 和 synthetic prediction。

状态：已完成。reference artifact 在 clean Policy `7e31d10` 上通过 SHA、EMA strict restore 与
15×19 synthetic prediction。

### R3 — 全量离线关闭

验收命令：

```bash
python -m compileall -q dexmani_real examples
python -m unittest discover -s tests -p 'test_*.py'
git diff --check
```

另需对 frozen reference 分别执行 `--print-config` 与 `--preflight-only`，保存实际 checkpoint、
sidecar、runtime、Real source 和 Policy source identities。不得运行 operational entry。

状态：已完成。全量离线套件为 212 tests，compileall、Black、isort 与 `git diff --check` 均通过。

### R4 — shadow 恢复

R0–R3 review 后，已在 `c9c3454` 取得新的限定 H2/H3 shadow evidence：120.007 s bounded stop、
1912/1912 endpoint shadow validation、zero coupled writes、arm servo calls=0 和所有 worker clean
shutdown。冻结 receipt 见
[`deployment_reference_h2h3_shadow_2026-08-29_c9c3454.json`](deployment_reference_h2h3_shadow_2026-08-29_c9c3454.json)；
旧 reference 仅保留作回归对比。

H4 必须等新的 shadow 证据通过后再单独 review 和授权。

## 7. 测试矩阵

| Boundary | 必测成功路径 | 必测拒绝路径 |
|---|---|---|
| sidecar | frozen schema v1 | dangling/out-of-tree/noncanonical/hash mismatch |
| v2 decoder | exact plain metadata + tensor states | arbitrary object、extra key、NaN/Inf、bad dtype/shape container |
| padding | `pad_before=n_obs_steps-1` | hardcoded 1 在 `n_obs_steps!=2` 时拒绝 |
| restore | exact EMA/raw selection + strict keys | fallback、missing normalizer、zero/nonfinite scale |
| provenance | clean matching commit/tree | dirty、commit mismatch、origin/tree changed |
| lifecycle | inference ready precedes hardware | load failure produces zero hardware workers |
| prediction | deterministic 15×19 | wrong representation/steps/dim/nonfinite |

示例程序不是测试。全量测试不连接设备；任何硬件验证必须另行授权。

## 8. 仿真兼容性证明

本阶段没有 Policy diff，因此主要证明 Real 没有反向改变仿真路径：

- Policy `git status --short` 为空，HEAD 保持 `7e31d10`；
- 旧 Policy `CheckpointStore`、eval loader、train/sim config 均不被 Real patch；
- Real 不写 `experiments/`、checkpoint、selector、normalizer 或 dataset；
- Real loader 只接受 deployment-v2，不接管 `simple.v1` 的仿真加载；
- future Policy 合并必须先独立证明旧 checkpoint 和 sim eval 兼容，再生成新 artifact version。

## 9. 停止条件

出现任一情况即停止，不用 fallback 绕过：

- 必须修改当前 Policy 训练/eval 代码才能加载 reference；
- artifact 缺失构造 agent、normalizer或 Real 数据语义所需字段；
- frozen v2 不能用 `weights_only=True` 和 exact schema 读取；
- clean Policy commit 与 artifact producer 不一致；
- preflight 需要 import dataset、env runner、`dexmani_sim` 或硬件模块；
- 任何测试为了通过而削弱 hash、freshness、generation、collision 或 worker gate；
- 未取得新授权便进入 shadow/execute。

## 10. 当前状态与下一步

- `dexmani_policy`：干净 `7e31d10`；本机 deployment 原型已撤回；
- `dexmani_real`：artifact resolver、Real-owned decoder、strict restore、runtime 与 shadow/H4 guard
  已完成离线验证，并在 `c9c3454` 完成新的 H2/H3 zero-write shadow；
- 下一步：单独 review H4 one-shot execute 的准入条件；
- 硬件：H2/H3 已完成；H4 未获授权，未运行 execute。

历史证据与现场要求见：

- [部署架构审查](deployment_review.md)
- [当前 H2/H3 frozen reference](deployment_reference_h2h3_shadow_2026-08-29_c9c3454.json)
- [历史 H2/H3 reference](deployment_reference_h2h3_shadow_2026-08-29.json)
- [H4 runbook](policy_h4_execute_runbook.md)
