# DexMani Policy 后续合并清单

> 状态：等待另一台机器上的 `dexmani_policy` 开发完成后再执行。
>
> 本文不授权当前 Real 工作流修改 Policy，也不表示任何 Policy 改动已经完成。

## 1. 目的

未来 Policy 分支可以增加训练数据与 checkpoint provenance，但必须保持已训练仿真策略、旧
checkpoint 和 eval/training 路径兼容。当前部署解耦不等待这批改动：Real 先按
[Real-owned deployment 方案](dexmani_real_policy_deployment_refactor_plan.md) 读取现有 frozen
deployment-v2 artifact。

另一台机器的代码只有在形成独立、干净、可测试的 commit 后才合并。本机不提前恢复、复制或
重做刚撤回的 Policy 原型。

## 2. 不可破坏的 Policy 行为

- 现有 sim dataset、sampler、normalizer 和 agent 行为不变；
- `simple.v1` checkpoint 继续能被现有训练恢复和仿真 eval 读取；
- 旧 checkpoint 缺少新增 provenance 时，Policy 自身仍可加载；Real deployment 可以拒绝；
- 原有 `CheckpointStore`、milestone/latest/best selector 语义不变；
- Policy eval 是否允许 EMA 缺失时回退 raw model，由 Policy 自己保持现状；Real 始终拒绝回退；
- 不引入对 `dexmani_real` 的 Python import；跨仓只通过 persisted data/artifact contract 对接；
- 不修改或覆盖已有 experiment、checkpoint、Zarr、视频和 W&B 生成物。

## 3. 时间窗合同

通用规则是：

```text
pad_before = n_obs_steps - 1
pad_after  = n_action_steps - 1
padding_semantics = repeat_edge
n_obs_steps - 1 + n_action_steps <= horizon
```

当前 DP3 的 `n_obs_steps=2`、`n_action_steps=8`，所以实例是 `pad_before=1`、
`pad_after=7`。不得在通用实现或文档中把 `pad_before` 固定成 `0` 或 `1`。

这项规则只描述训练 sequence sampling，不授权 Real runtime 用 B 前或上一 generation 的反馈补帧。

## 4. 允许的未来增量

若另一台机器上的实现确有需要，可以独立评审以下 additive 能力：

- 从实际打开的 Real Zarr 生成 plain、versioned data provenance；
- 在新 checkpoint 中可选保存 fully resolved inference config 与 data contract；
- 为新 checkpoint/artifact 提供 canonical、可 hash 的只读 metadata；
- 增加接受“已加载 checkpoint object”的 restore helper，同时保持原 path-based eval API；
- 生成新的 deployment artifact version，但不能替换旧 sim checkpoint 格式。

这些能力不应把机器人 lifecycle、SafetyGate、hardware config、point-cloud runtime 或 receipt 写入
Policy。真实机器人特有校验继续由 Real 拥有。

## 5. 合并前 review

### 5.1 工作树与范围

1. 记录 Policy branch、HEAD、base commit 与 `git status --short`；
2. 将 deployment/provenance diff 与同时进行的 agent/training 改动分开 review；
3. 确认没有 checkpoint、dataset 或 experiment 生成物进入 diff；
4. 确认没有为了 Real 部署改变通用 sim 默认值。

### 5.2 schema 兼容性

- 新字段必须 additive，历史 payload 用 `.get(...)` 或明确的 version branch 读取；
- 新 checkpoint 写入的 metadata 必须是 finite plain JSON types；
- Real Zarr 声明与实际 array shape/dtype 一起验证，不能只信 attrs；
- `pad_before` 必须根据 `n_obs_steps` 计算；
- action/state/point-cloud/normalizer dimensions 必须来自实际 resolved config 与 tensors；
- FAAS 不作为新部署 contract 的兼容变体。

### 5.3 restore 兼容性

- 原 Policy eval 入口及其现有调用者不改签名；
- 若新增 loaded-object helper，path-based loader只做一次 load 后委托它；
- raw/EMA 选择、DDP/compile key normalization 与 normalizer restore 必须有旧行为回归；
- Real 不直接复用带宽松 fallback 的 Policy eval helper，除非接口提供明确 strict mode 且完成跨仓 review。

## 6. 必须提供的验证证据

合并请求至少包含：

- Policy 全部既有离线测试；
- 受影响的代表性 smoke test（至少 DP3，公共基类变化时增加不同架构代表）；
- 一个旧 `simple.v1` checkpoint 的 load/eval round trip；
- 一个新 checkpoint metadata round trip；
- `n_obs_steps=2/3` 对应 `pad_before=1/2` 的参数化测试；
- sim dataset 不带 Real attrs 时保持原行为的测试；
- git diff/status、未运行的 GPU/训练验证及原因。

长时间训练、DDP、视频评测和硬件运行不因本清单自动获授权。

## 7. 与 Real 的接入方式

Policy commit 合并后：

1. 固定新的 clean Policy commit；
2. 独立生成新的 artifact/sidecar，不覆盖当前 reference；
3. 为新 artifact 使用新 selector 或明确版本；
4. Real 先扩展纯 artifact/decoder tests，不修改 hardware/control 路径；
5. 完成 `--print-config`、`--preflight-only` 和 recorded observation replay；
6. 通过独立 review 后，再申请新的 shadow 授权。

不能让同一个 selector 在相同名字下静默改变 checkpoint、producer commit 或 contract version。

## 8. 合并停止条件

出现任一情况即停止合并：

- 需要修改或重写已有 sim checkpoint；
- 旧 eval/training 入口无法保持兼容；
- padding 仍被固定为与 `n_obs_steps` 无关的常数；
- provenance 只能依靠运行时猜测、外部未冻结 YAML 或 dirty source；
- Policy 开始拥有 Real hardware、安全或 lifecycle 逻辑；
- 新 artifact 无法与旧 reference 并存；
- 另一台机器的分支尚未形成 clean、可 review commit。
