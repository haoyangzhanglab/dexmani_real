# DexMani Policy 部署推理链路与模型兼容边界

本文描述 `examples/run_policy.py` 在 DexMani Real 中从实验目录解析、checkpoint
恢复、观测编码、模型推理、动作反归一化，到 shadow 验证或 coupled command 发布的
实际代码链路。同时回答两个部署问题：核心推理是否在 GPU 上执行，以及当前 adapter
能否直接部署 `dexmani_policy` 中的其他模型。

审查基线为 DexMani Real `7fd2e96` 和已通过 CUDA preflight 的 DP3 reference
artifact。运行时行为和源码优先于本文；后续边界变化应同步更新本文。

`7fd2e96` 的证据边界目前止于离线 CUDA preflight；旧 Real revision 上的 shadow/H4 receipt
不能替代该 revision 的新真机验证。

## 1. 总览

```text
experiment directory
→ resolve selector + canonical deployment sidecar
→ build immutable artifact/runtime projection
→ spawn inference worker
→ no-follow open + identity checks + checkpoint SHA-256
→ torch.load(weights_only=True, map_location="cpu")
→ validate checkpoint schema/data/provenance contracts
→ Hydra instantiate cfg.agent
→ strict model/EMA + normalizer restore
→ agent.to(operator device) + eval + warmup
→ B changes ARMED to RUNNING
→ build causal arm/hand/point-cloud history
→ encode and copy tensors to model device
→ observation normalization
→ model encoder + diffusion/flow inference
→ action unnormalization
→ select full future action interval
→ decode PolicyPrediction
→ stamp logical-grid targets and mask expired prefix
→ policy_plan_ring
→ coordinator endpoint scheduling
→ runtime/feedback/generation/limits/delta/collision validation
→ shadow: validate only, zero coupled writes
→ execute/task: coupled_cmd_ring → arm/hand workers → coupled ACK
```

模型输出只是 proposal。inference worker 不能写 `coupled_cmd_ring`，coordinator 是 learned
policy 唯一的 robot-command producer。

## 2. Artifact 解析与不可变配置

入口 `examples/run_policy.py:main()` 首先执行：

```python
artifact = resolve_policy_artifact(args.experiment_dir)
runtime = resolve_runtime_config(yaml_path=args.config)
projection = resolve_policy_runtime_config(
    artifact=artifact,
    runtime_config=runtime,
    yaml_path=args.deployment_config,
    device=args.device,
    inference_seed=args.inference_seed,
    execution_mode=args.execution_mode,
    hand_acknowledged=args.hand,
    h4_execute_bounds=h4_execute_bounds,
    task_execute_bounds=task_execute_bounds,
)
```

`resolve_policy_artifact()`：

1. 以 directory fd 打开 experiment 与 `checkpoints/`；
2. 优先解析 `deployment_latest.pt`，只允许一层 selector；
3. 读取对应 canonical `.deployment.json`；
4. 校验文件类型、basename、大小、sidecar schema 和 lstat identity；
5. 固定 checkpoint、sidecar、producer 和 allocation contract；
6. 此阶段不加载模型，也不对大 checkpoint 做完整 hash。

`resolve_policy_runtime_config()` 将 ownership 固定为：

- artifact 拥有 checkpoint、task、action、observation horizon、modalities 和 point count；
- Real runtime 拥有 control period、freshness、skew、command deadline 和 safety limits；
- operator 通过 CLI 明确拥有 device、seed、execution mode 和物理执行 bounds；
- runtime 实现固定为 Real-owned `DexManiPolicyRuntime`，YAML 不能重定向 loader。

物理模式还要求干净且可识别的 DexMani Real revision，并要求 CLI 提供与 sidecar 一致的
checkpoint SHA-256。

## 3. Checkpoint 的单次安全加载

operational lifecycle 在 spawn inference child 中调用：

```text
inference_loop
→ _load_inference_runtime
→ load_verified_policy_runtime
→ _load_verified_policy_runtime
```

加载边界执行以下顺序：

1. 根据解析阶段保存的 directory/file identity 重新打开 artifact；
2. checkpoint 使用 `O_NOFOLLOW` 打开，拒绝未授权的 symlink 跳转；
3. 对已经打开的同一个 fd 流式计算 SHA-256；
4. 与 sidecar index 中的 SHA-256 精确比较；
5. 在导入 Policy 前检查 package origin、commit、dirty 和 source-tree SHA；
6. 将同一个 fd rewind 后交给 Real-owned checkpoint decoder；
7. restore 后再次检查 selector、sidecar、checkpoint 与目录 identity，防止 TOCTOU 替换。

decoder 只执行：

```python
torch.load(stream, map_location="cpu", weights_only=True)
```

接受的 deployment-v2 payload 必须精确为：

```text
payload
├── _format = "dexmani.deployment.v2"
├── state
│   ├── epoch
│   ├── global_step
│   ├── train_params
│   ├── inference_config
│   ├── data_contract
│   ├── producer
│   └── deployment_contract
└── weights
    ├── model
    └── ema_model
```

metadata 只能包含普通、有限、可审计的数据；state dict 必须非空且全部为 tensor，并拒绝
`module.`、`_orig_mod.` 等非 canonical key。加载失败直接使 inference worker 启动失败，
不存在 fake policy 或 path-based fallback。

## 4. Agent 恢复与启动门

`DexManiPolicyRuntime.load_loaded_checkpoint()`：

1. 只允许 `dexmani_policy.agents.*` 下的 embedded Hydra target；
2. 校验 embedded inference config 已完全 resolve；
3. 精确比较 sidecar、checkpoint metadata、training data contract 和 producer；
4. 要求 Real Policy Zarr v5、`obs[t]_before_action[t]`、camera-source state alignment；
5. 要求 `pad_before = n_obs_steps - 1`、`pad_after = n_action_steps - 1`；
6. 要求 `use_aux_ee = false`；
7. 用 deployment seed 初始化 Python、NumPy 和 Torch RNG；
8. 只实例化 `cfg.agent`，不构建 dataset 或 env runner；
9. 根据 embedded `eval.use_ema` 选择 EMA 或 raw model state；
10. `agent.load_state_dict(..., strict=True)`；
11. `agent.to(device)`、`agent.eval()`；
12. 校验 manifest 和 action/joint-state/point-cloud normalizer 的 key、维度、finite scale、
    non-zero scale 与 finite offset；
13. 保存 embedded `eval.denoise_steps`。

随后在 inference child 内执行 5 次 warmup。最后 3 次耗时必须小于 artifact 剩余动作窗口，
否则 subsystem 不会 ready。warmup 保存并恢复 deployment RNG state，不消耗 rollout 的第一组
随机序列。

## 5. B 后的因果观测

ARMED 状态只维持 worker readiness，不做策略推理。operator 按 B 后 safety state 进入 RUNNING，
inference worker 才从 shared-memory history 构建 `ObservationBatch`。

当前 DP3 reference 的输入为：

```text
arm_qpos history       [2, 7]       rad
hand_qpos history      [2, 12]      rad
point_cloud history    [2, 1024, 6] xyzrgb
```

每帧必须属于当前 run generation，source/publish timestamp 因果有效且未超龄；arm/hand state
必须与所选 point-cloud source time 处于允许 skew 内；点云历史不能跨 camera generation，也不能
用一张旧点云广播或填充缺失历史。

adapter 编码为：

```text
joint_state  [1, n_obs_steps, 19]    arm7 + hand12
point_cloud  [1, n_obs_steps, N, 6] xyz in xArm base + RGB in [0, 1]
```

## 6. 归一化、模型推理和反归一化

`agent.predict_action(obs_dict)` 内部首先调用 checkpoint 恢复的 normalizer：

```text
x_normalized = x_physical * scale + offset
```

DP3 将归一化后的 point cloud 送入 point-cloud encoder，将 19 维 joint state 送入 state MLP，
拼接 observation condition 后由 diffusion decoder 生成完整 horizon：

```text
normalized pred_action [1, horizon, action_dim]
```

Policy agent 在返回前执行：

```text
x_physical = (x_normalized - offset) / scale
```

所以 Real adapter 收到的 `result["pred_action"]` 已是物理动作空间，而不是 `[-1, 1]`
归一化动作。

## 7. 完整未来区间与 PolicyPrediction

adapter 不采用模型返回的短 `control_action` chunk，而从完整 `pred_action` 中选择：

```python
start = n_obs_steps - 1
future = pred_action[:, start:horizon, :control_action_dim]
```

required action count 固定为：

```text
required_action_steps = horizon - (n_obs_steps - 1)
```

当前 reference：

```text
n_obs_steps = 2
horizon = 16
start = 1
future = pred_action[:, 1:16]
output = [15, 19]
```

joint action 解码为：

```text
arm_qpos  [15, 7]   absolute joint targets, rad
hand_qpos [15, 12]  absolute joint targets, rad
```

EE action 则要求 21 维：

```text
ee_pos     [N, 3]
ee_rot6d   [N, 6]
hand_qpos  [N, 12]
```

EE rot6d 必须 finite 且非退化，随后由 coordinator 执行 IK；joint action 不经过 IK。

## 8. 时间戳、plan 和 command 发布

inference worker 用 Real control grid 给无时间的 `PolicyPrediction` 加时间：

```text
target[i] = observation logical step + i * control_dt
earliest usable target = inference finished + command lead
```

已过期前缀被 `valid_mask=0` 屏蔽；全部过期则整个 prediction 丢弃。发布 plan 前再次读取
`run_generation`，generation 已变化时不能把旧预测重标为新 run。

`policy_plan_ring` 携带 plan/observation/generation、观测和推理时间戳、target grid、valid mask
以及 arm/hand 或 EE/hand 数组。coordinator 读取最新 plan，通过 bounded action buffer 选择 due
endpoint，然后执行：

- plan/source/command age 和 deadline；
- current RUNNING state 与 run-generation ownership；
- arm/hand feedback freshness；
- finite、shape、joint limits、per-tick delta；
- self/environment collision；
- hand policy/mechanical envelope；
- 最小 worker delivery window。

输出边界分为三层：

| 边界 | 表示 | 能否产生运动 |
|---|---|---:|
| model output | untimed `PolicyPrediction` | 否 |
| inference output | timed `policy_plan_ring` record | 否 |
| physical output | `coupled_cmd_ring` record | 仅 execute/task |

shadow 只返回 `SHADOW_VALIDATED`，并验证 coupled-command sequence 没有变化。execute/task 才将
同一 action id、generation、target/deadline 原子发布给 arm/hand workers，并等待两个 worker
对同一 coupled endpoint 的 ACK。

## 9. 当前推理是否在 GPU 上

结论：**当前 reference 命令使用 `--device cuda:0` 时，核心模型推理在 GPU 上；部署代码并不
无条件强制 GPU，显式传 `--device cpu` 就会在 CPU 上推理。** `--device` 是必填参数，不存在
静默选择 CPU/GPU 的默认值。

数据与设备边界如下：

| 阶段 | 设备 |
|---|---|
| selector/sidecar/hash/provenance | CPU |
| `torch.load(..., map_location="cpu")` | CPU |
| Hydra agent construction和初始 state restore | CPU |
| `agent.to("cuda:0")` 后的参数、buffer、normalizer | GPU 0 |
| observation NumPy history assembly | CPU |
| `torch.as_tensor(..., device="cuda:0")` | CPU → GPU 0 |
| observation normalization、encoder、diffusion/flow decoder | GPU 0 |
| action unnormalization | GPU 0 |
| `pred_action.detach().cpu().numpy()` | GPU 0 → CPU，同步边界 |
| timing、plan、SafetyGate、IPC、robot workers | CPU |

当前干净 revision 的 shadow/H4 CUDA preflight receipt 均记录 `device="cuda:0"`，并已完成一次
真实 checkpoint restore 与 `[15, 19]` synthetic prediction。因此“当前 reference 核心推理在
GPU”同时有代码路径和实际 CUDA preflight 证据。

需要注意：换命令、换 device 或换 checkpoint 后必须重新看 receipt，不能从 experiment 名称
或 GPU 是否空闲推断实际设备。

## 10. 对其他 dexmani_policy 模型的兼容性

### 10.1 当前 adapter 的硬合同

另一个模型只有同时满足以下条件才可能直接接入：

1. 有 canonical deployment-v2 checkpoint、sidecar、完整 SHA-256 和 producer provenance；
2. training data contract 是 Real Policy Zarr v5；
3. modalities 精确为 `joint_state + point_cloud`；
4. joint state 为 arm7 + hand12，agent state input 维度为 19，point cloud 为
   `N x 6 xyzrgb`；
5. `cfg.agent` 位于 `dexmani_policy.agents.*`；
6. agent 暴露 `n_obs_steps/n_action_steps/action_dim/horizon/control_action_dim`；
7. embedded config 明确提供 adapter 当前读取的 `agent.num_points` 和 `agent.pc_dim`；
8. agent 具备 checkpoint-owned fitted normalizer；
9. `predict_action(obs_dict, denoise_timesteps=...)` 返回完整
   `[B, horizon, action_dim]` 的 `pred_action`；
10. action 是 joint 19 维或 EE+hand 21 维；
11. `use_aux_ee=false`；
12. warmup latency 适配 artifact 的可用未来窗口。

满足模型 API 还不等于 checkpoint 已可部署。每个新 checkpoint 都必须独立通过 artifact
解析、strict restore、normalizer/manifest 校验、CUDA preflight、H2/H3 shadow，之后才可考虑
物理执行。

### 10.2 模型分类

| 模型 | 当前判断 | 主要依据或阻塞项 |
|---|---|---|
| DP3 | 已验证模型兼容 | 当前 reference 已完成 strict restore、CUDA preflight，并在旧 Real revision 上有 shadow/H4 记录；`7fd2e96` 仍需新的 H2/H3 后才能进入 H4 |
| DQ-RISE | 结构上兼容，尚未实证 | point-cloud + joint-state、`num_points/pc_dim`、完整 `pred_action`、19/21 维均符合；adapter 已处理 checkpoint-owned codebook，但仍需专用 deployment-v2 artifact 与 preflight |
| ActionFlow（默认 joint action） | 结构上兼容，尚未实证 | point-cloud contract、BaseAgent-style normalization 和完整 horizon output 符合；需验证 NFE/latency 与 artifact window。当前 config 将 `state_dim` 绑定到 `action_dim`，不能直接推导其 EE-action 变体也兼容 |
| ManiFlow | 结构上兼容，尚未实证 | `joint_state + point_cloud`、显式 `num_points/pc_dim` 和 BaseAgent output 符合；需验证 GPU latency/显存与 checkpoint contract |
| SAT | 结构上兼容，尚未实证 | SAT 内部转置后仍返回 `[B,T,A]` 并反归一化；显式 point-cloud metadata 符合；需专用 strict restore/preflight |
| R3D，`use_aux_ee=false` | 当前不能直接接入，适合小范围适配 | 模态和输出语义基本符合，但当前 R3D embedded agent config 没有顶层 `agent.num_points/agent.pc_dim`，adapter 会在 manifest 构建前拒绝；应由 artifact contract/模型明确字段提供，不能添加隐式 fallback |
| R3D，`use_aux_ee=true` | 不兼容 | Real contract 明确要求 `use_aux_ee=false`，且 28 维 joint+aux-EE action 不属于 19/21 维控制合同 |
| DP（RGB） | 不兼容当前入口 | 需要 `joint_state + rgb`，当前 runtime 只构建 point-cloud observation |
| MoE DP（RGB） | 不兼容当前入口 | RGB encoder 和 RGB normalizer 不受当前 manifest/observation IPC 支持 |
| MultiTask DiT | 不兼容当前入口 | 需要 RGB、task text/task identity 和多任务 normalizer 语义，当前固定 adapter 没有这些边界 |

### 10.3 推荐接入策略

不要把当前 fixed adapter 改回通用 registry/factory，也不要为了让模型加载而添加缺字段 fallback。
建议顺序为：

1. 优先选择 point-cloud 模型；
2. 在 `dexmani_policy` 稳定分支生成 deployment-v2 artifact；
3. 让 artifact 明确携带所有模型所需 shape/semantics，而不是依赖训练 YAML 路径；
4. 在 Real adapter 中只增加被真实模型证明需要的显式字段读取；
5. 为该模型添加 strict restore、normalization round-trip、prediction shape 和 latency 测试；
6. 依次执行 `--print-config`、CUDA `--preflight-only`、H2/H3 shadow；
7. 获得新的明确授权后才执行受限 H4。

对于下一种模型，若能生成合格的 deployment-v2 artifact，最省 Real-side 改动的候选是
ActionFlow、ManiFlow、SAT 或 DQ-RISE。R3D 应先修正显式 point-cloud manifest metadata；
RGB 与 MultiTask 路线需要新的 observation/IPC/manifest 边界，不能视为当前 adapter 的简单
兼容扩展。
