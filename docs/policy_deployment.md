# DexMani Policy 部署与推理

`examples/run_policy.py` 是 learned-policy 部署的唯一操作者入口。本文件是 Real 侧策略部署的
唯一说明：从 artifact 到 robot command 的数据链、支持边界、物理执行门、验证记录，以及未来
`dexmani_policy` 合并条件均以此为准。源码、schema 和运行时配置优先于本文。

> R1 集成基线：Policy handoff 已在 Policy `main`
> `aa4a0a39dd5a69e3a4ad85ea8190d6889610d175` 接受；handoff receipt 中的 representative
> artifact producer 仍是 `fc6b7dfb45748f4187f2e82b5425721ed02b028e`。下文 H2/H3、H4
> 和 task rollout 是 full-future executable semantics 下产生的 pre-R1 历史证据，不能授权
> R1 后的物理执行。物理任务能力仍未验证。

## 1. 状态、范围与所有权

| 项目 | 当前结论 |
|---|---|
| checkpoint restore、normalizer、GPU inference | 已通过 strict restore、CUDA preflight 与真实运行验证。 |
| H2/H3 shadow | 已通过；不发布策略动作。 |
| H4 one-endpoint execute | 已通过一次；不代表 task rollout 授权。 |
| task command path | 两次均完成 331 次 coupled publication/ACK 与 clean shutdown。 |
| 真实 pick/place | 未验证成功；暂停新的 task rollout，先补只读诊断。 |

职责边界保持简单且单向：

| Owner | 负责 | 不负责 |
|---|---|---|
| `dexmani_policy` | agent class、训练/仿真、checkpoint 内嵌模型状态和 normalizer | Real hardware、lifecycle、SafetyGate、IPC、receipt |
| `dexmani_real` | artifact 解析、strict restore、观测适配、命令验证/发布、workers 和物理生命周期 | 修改训练数据、sim checkpoint 或 Policy eval 语义 |
| 操作者 | device/seed/mode/bounds 的显式 CLI 选择、物理现场确认、H/B/S/ESC/e-stop | 绕过 runtime 的 contract 或安全门 |

模型输出始终只是 proposal。inference worker 不能写 `coupled_cmd_ring`；coordinator 是 learned
policy 唯一的 robot-command producer。

## 2. 固定 reference 与 artifact 合同

当前验证使用：

| 字段 | 值 |
|---|---|
| experiment | `/home/zhanghaoyang/Desktop/dexmani_policy/experiments/dp3/pick_place_toy/2026-08-28_13-59_42` |
| checkpoint | `epoch=1126-step=00080000-deployment-v2.pt` |
| checkpoint SHA-256 | `b174bd483b64090cd3f5dbe0a5bfadd10998f5d27d43fc9aca06efb82242484c` |
| Policy producer commit | `7e31d10e7a31ff3d12df31b8683c9c90b357cbc5` |
| action | 19-D absolute joint target：arm 7 + hand 12 |
| observation | 2 steps of arm/hand state and `1024 × 6` xyzrgb point cloud |
| control | `16 Hz`、horizon 16；15-step prediction future、8-step executable control |

当前 artifact 的 sidecar 是 schema v1，因而 v1 parser 不是可删除的历史兼容分支。schema v2
承载 `control_action_dim`、auxiliary action layout 和 RGB payload；它是 RGB/R3D 未来 artifact 的
明确合同，也必须保留。两种 schema 都 fail closed，不猜测缺失字段。

训练窗口与 Real runtime 的含义不同：

```text
pad_before = n_obs_steps - 1
pad_after  = n_action_steps - 1
padding_semantics = repeat_edge
```

当前实例为 `1/7`，但 Real 在 B 后仍必须收集两个不同、因果有效的 observation；不得用 B 前、
上一 generation 或旧点云填充 history。

## 3. 从 checkpoint 到动作的完整链路

```text
experiment directory + sidecar
    → immutable artifact/runtime projection
    → isolated inference worker
    → verified checkpoint decode + strict agent restore
    → causal observation batch
    → GPU prediction / PolicyPrediction
    → timed policy plan
    → coordinator validation and scheduling
    → shadow validation or coupled command publication
    → arm/hand worker acknowledgement
```

### 3.1 入口与不可变投影

`run_policy.py` 解析 CLI 后依次：

1. `resolve_policy_artifact()` 用 directory fd 解析 selector 和 canonical sidecar，固定
   checkpoint、producer、allocation contract 和 lstat identity；此时不反序列化或 hash 大文件。
2. `resolve_runtime_config()` 解析 Real-owned runtime defaults/YAML。
3. `resolve_policy_runtime_config()` 将 artifact、runtime 和 operator-owned device/seed/mode/bounds
   投影成不可变配置。YAML 不能把 runtime loader 重定向到其他实现。

物理模式还要求可识别且干净的 Real source revision，以及 CLI 的 expected checkpoint SHA-256
与 sidecar 精确一致。

### 3.2 单次验证加载与 agent restore

inference child 对同一个 held file descriptor 执行：

1. `O_NOFOLLOW` 打开并复核 experiment、selector、sidecar 和 file identity；
2. 流式 SHA-256，并与 sidecar 精确比较；
3. Policy import 前验证 package origin、producer commit、clean worktree 和 Python tree hash；
4. `torch.load(stream, map_location="cpu", weights_only=True)`；
5. Real-owned decoder 校验 deployment-v2 的 exact payload/state/weights schema、plain metadata 和
   canonical tensor keys；
6. 只实例化 resolved `cfg.agent`，strict restore model 或要求的 EMA 与 checkpoint-owned normalizer；
7. restore 后重新检查持有对象 identity，防止 TOCTOU 替换。

不使用 fake policy、path-based second load 或宽松 key/normalizer fallback。agent 只允许来自
`dexmani_policy.agents.*`；dataset 和 env runner 不会在部署时构造。

### 3.3 GPU 启动与观测

核心模型推理默认在 `cuda:0`：CUDA 不可用或 index 不存在会启动失败，绝不静默回退 CPU；只有
显式 `--device cpu` 才使用 CPU。loader/identity/hash、NumPy history assembly、timing、IPC 和 robot
workers 留在 CPU；`agent.to(device)` 之后，normalization、encoder、diffusion/flow decoder 和 action
unnormalization 都在所选 GPU。validated `control_action.detach().cpu().numpy()` 是回到 CPU 的唯一
动作数据边界。

启动时执行 5 次 synthetic warmup，最后 3 次必须落在 artifact 的 `n_action_steps` executable
window 内；warmup 同时精确验证 Policy 声明的 canonical control slice。随后恢复随机状态，
不消耗 rollout 的第一组 diffusion sample。isolated preflight 也执行一次同样的 contract warmup。

ARMED 只维护 worker readiness。B 使状态进入 RUNNING 后，worker 才从 current generation 的
shared-memory history 构建 observation：

```text
joint_state  [1, n_obs_steps, 19]       arm7 + hand12
point_cloud  [1, n_obs_steps, N, 6]     xArm-base xyz + RGB [0,1]
rgb          [1, n_obs_steps, 3, H, W]  optional, float32 RGB [0,1]
```

每个 state、camera/point-cloud sample 都必须满足 generation、source/publish freshness 与最大 skew
约束。RGB 以 `uint8 [H,W,3]` 存在 camera ring，复制到 device 后才转为 CHW float；resize、crop 和
image normalization 由 checkpoint 的 image processor 决定。point-cloud-only、RGB-only 和联合输入
均必须由 artifact 显式声明，不能由训练路径或缺字段推断。

### 3.4 归一化、预测与动作解码

checkpoint 恢复的 agent 自己处理归一化与反归一化：

```text
x_normalized = x_physical * scale + offset
result       = agent.predict_action(obs_dict)
pred_action  = result["pred_action"]
control_action = result["control_action"]
x_physical   = (x_normalized - offset) / scale
```

因此 Real 接收到的 `pred_action` 已处于物理动作空间，不应再次按 `[-1,1]` clip 或重拟合
normalizer。temporal dimensions 全部来自 artifact：

```text
control_start = n_obs_steps - 1
prediction_future_steps = horizon - (n_obs_steps - 1)
executable_control_steps = n_action_steps
```

adapter 要求完整 `pred_action` 与 executable `control_action` 同时存在。完整 prediction 必须精确满足
`[1,horizon,action_dim]` 且所有维度 finite；control 必须精确满足
`[1,n_action_steps,control_action_dim]` 且 finite。startup qualification 精确验证：

```python
expected_control = pred_action[
    :, control_start:control_start + n_action_steps, :control_action_dim
]
```

正常 inference tick 不重复做 exact-slice synchronization；它仍逐次检查两者的 shape/finite。
Real 只从 `result["control_action"]` 构造 `PolicyPrediction`，不从 `pred_action` 重建 control，
也不要求或读取 `tail`。representative DP3 的 prediction future 是 15 steps，但 executable control
是 8 steps，解码为 `arm_qpos [8,7]` 与 `hand_qpos [8,12]`。

R3D `joint19_ee9` 的完整 `pred_action` 是 28-D，28 个维度均做 shape/finite validation；唯一
executable source 是 19-D `control_action`。EE action 的完整 prediction 同样做 shape/finite validation，
但 rot6d geometry 只对 executable 21-D control 检查：finite 但退化的未执行 prediction tail 不阻止
合法 control，退化的 executable control 必须拒绝。EE control 由 coordinator 做 IK；joint control
不经过 IK。

### 3.5 Plan、SafetyGate 与 worker ACK

inference worker 把无时间的 `PolicyPrediction` 对齐到 Real control grid，屏蔽已过期 prefix；若
没有可用 target，丢弃整个 prediction。它将 plan、observation/generation、timestamp、valid mask 和
action 数组写入 `policy_plan_ring`。

coordinator 从 latest plan 选择 due endpoint，并在 publication 前复核：RUNNING/generation、plan/
feedback freshness、shape/finite/joint limit、per-tick delta、hand envelope、collision、delivery window
和 deadline。只有 execute/task 才原子写入 `coupled_cmd_ring`；shadow 只完成同一套 validation，
保证 zero coupled write。

接触下 raw learned hand target 可能无法物理收敛。为保证 ACK 的语义与真实 IPC target 一致，先对
raw endpoint 完整过 SafetyGate，再相对 fresh hand feedback 以 `0.3 rad/tick` 整形成 actual IPC
endpoint，并再次完整过 SafetyGate。hand ACK 指 SDK 已接受这一 exact IPC endpoint，不声称 raw
learned target 在接触下已物理到达。该流程没有放宽 limits、collision、freshness、generation、expiry
或双 worker ACK。

## 4. 支持边界与新模型接入

直接接入的模型必须同时具备：hash-bound deployment artifact、Real Policy Zarr v5 data contract、
explicit modality metadata、19-D joint state、checkpoint-owned normalizer、完整 `[B,horizon,action_dim]`
的 `pred_action`、精确 `[B,n_action_steps,control_action_dim]` 的 `control_action`、兼容的 action
layout、`dexmani_policy.agents.*` target 和适配 executable window 的 warmup
latency。满足 API 不等于已获物理部署资格；每个新 artifact 都要独立完成 strict restore、CUDA
preflight、H2/H3 shadow 和相应物理授权。

| 模型 | 当前支持 | 还需的独立证据 |
|---|---|---|
| DP3 | 已验证 reference | task 成功仍未验证。 |
| DQ-RISE、ActionFlow、ManiFlow、SAT | 结构上兼容 | 专用 artifact、strict restore、latency 与 shadow。 |
| R3D without auxiliary EE | schema v2 可接入 | 正确的 point-cloud/RGB contract、provenance、latency。 |
| R3D with auxiliary EE | schema v2 `joint19_ee9` 可接入 | 28-D full-output validation；只有 Policy 返回的 19-D `control_action` 进入控制。 |
| DP / MoE DP with RGB | Real input boundary 已支持 | schema v2 RGB payload、image processor、strict restore 与 shadow。 |
| MultiTask DiT | 不支持 | 显式且经训练数据验证的 task/text conditioning contract。 |

不要为“通用兼容”引入 registry、factory 或缺字段 fallback。优先让 artifact 写清真实模型的输入、
输出和语义，Real 只增加已经被具体 artifact 证明需要的显式读取。

### 实验性扩展（当前不启用）

只有明确的模型/任务证据表明需要时，才考虑以下优化；二者均不能绕过 coordinator、SafetyGate、
generation、deadline、collision 或 worker validation。

- **chunk conditioning：** 仅限模型原生支持该输入合同。缓存必须绑定 run generation、observation
  identity、模型版本和原始 target timestamp；generation/observation 不连续、target 已消费或过期时
  必须清空。它只可使用未消费的原始 model prediction，不能读取 command ring 或把执行 ACK 当作
  condition。先用 recorded observation/replay 比较 jerk、过期率和 safety reject，再考虑 shadow。
- **笛卡尔插补：** 仅作为 EE-action 的纯 candidate generator，不能直接调用 SDK。每个实际端点仍须
  经 IK、工作空间、joint/delta limit、碰撞和时间预算检查；非法旋转、IK/碰撞不可判定、超速度或
  过期 waypoint 一律拒绝。只有在测量表明 endpoint 控制确有连续性问题时才提出该项。

## 5. 执行层级与物理门

| mode | 允许的策略写入 | 目的 | 额外门 |
|---|---:|---|---|
| `shadow` | 0 | 验证 prediction、timing 和所有 publication validation | 明确 H2/H3 授权；B 后仍不发布动作。 |
| `execute` / H4 | 1 | 验证一个 coupled physical endpoint | 空场地、独立 H4 授权、一次 H、一次 B、双 ACK。 |
| `task` | 有界，当前 331 | 真正 task rollout | 独立 task 授权；当前暂停，先完成第 7 节诊断。 |

### H4 single-endpoint sequence

每次 H4 都需要当前干净 revision、同 artifact/device 的 H2/H3 evidence、`--print-config` 与
`--preflight-only`。在场地无人、无物体/障碍、e-stop 就绪并获单次授权后：

1. 启动 `--execution-mode execute --hand --device cuda:0`，确认所有 subsystem 为 `ARMED`；
2. 操作者按一次 H：SDK 接受一次 hand home command，然后执行 collision-checked arm canonical
   home。只要求 hand command accepted，不要求 hand feedback 进入 home tolerance；
3. 日志出现 `physical home sequence completed` 且仍为 ARMED 后，操作者按一次 B；
4. bound 固定为一个 coupled endpoint、30 秒和有限 ACK timeout。任何 S/ESC/e-stop、worker/freshness/
   generation/timeout/safety fault 都停止；不自动重试。

H4 receipt 必须同时证明 `completed=true`、`max_published_endpoints=1`、
`coupled_command_writes=1`、`physical_home_completed=1` 和 acknowledged action id。需要可移交
evidence bundle 时，在进程退出后运行无硬件的 `examples/seal_h4_evidence.py`，绑定 runtime receipt、
terminal log 和操作者记录。

获得本次 H4 的明确授权后，受限 profile 为：

```bash
PYTHONPATH=/home/zhanghaoyang/Desktop/dexmani_policy \
PYTHONUNBUFFERED=1 \
DEXMANI_RECEIPT_DIR=/home/zhanghaoyang/.dexmani/receipts \
/home/zhanghaoyang/miniconda3/envs/real_robot/bin/python \
  examples/run_policy.py \
  --experiment-dir /home/zhanghaoyang/Desktop/dexmani_policy/experiments/dp3/pick_place_toy/2026-08-28_13-59_42 \
  --device cuda:0 \
  --execution-mode execute \
  --hand \
  --max-running-seconds 30 \
  --execute-max-published-endpoints 1 \
  --execute-ack-timeout-seconds 2 \
  --execute-expected-checkpoint-sha256 b174bd483b64090cd3f5dbe0a5bfadd10998f5d27d43fc9aca06efb82242484c
```

任何 operational invocation 都连接硬件；本文、历史 receipt、`--print-config` 或 preflight
都不是硬件授权。

## 6. Pre-R1 历史验证记录

本节所有 H2/H3、H4 和 task evidence 均产生于 Real 把完整 prediction future 当作 executable 的旧
语义。它们只保留为历史 transport evidence，不授权 `control_action` R1 语义下的 physical shadow、
H4 或 task；R1 后如需物理运行必须重新独立 review。

| 级别 | 实际结果 | 可复核证据 |
|---|---|---|
| H2/H3 task-scene shadow | `120.033 s`、`1,916/1,916` endpoints 仅 shadow-validated；zero coupled write/arm servo/hand policy SDK send，clean shutdown。 | log `/tmp/dexmani-task-h2h3-shadow.ee0MrE/terminal.log`，SHA-256 `76e08c5033658dc98fb395a9e4160b76e84c36d35cc9b9ff2a03cc36db665051` |
| H4 | 一次 physical home、一个 coupled endpoint 和 paired ACK。 | log SHA-256 `6eed6a15f4a2b0f3e1dace2940a0b08094e683b4e03572082becdf75549485ba`；receipt SHA-256 `5d99c8468f1fbf7df604f51b25f6ad891ce17f53f6af37898b2d7516a4ebc09f` |
| task #1 | 331 endpoint、331 coupled write/ACK、clean shutdown；没有自动证明物理成功。 | log SHA-256 `213174c4e2f655b6311c22d3af89bb726e013994114769d7f9859b502ca0687f`；receipt SHA-256 `f6c3218cb0ba29ec0519ae2928f1d7a05e2d8dfd8d76c929a92fdd296d2c0dbc` |
| task #2 | 同样 331 endpoint/write/ACK；操作者确认未完成抓取放置，未见传输、安全、expiry 或 ACK fault。 | log SHA-256 `277111c4431d74e2054b83e8cb9156bc6942150158f2e1dfb46a847199d9f352`；receipt SHA-256 `2670d9fd68350718d9d16bf8bce07d4db58eb95f813b3bd0ff97600fcd344506` |

完整 runtime receipt 位于 `/home/zhanghaoyang/.dexmani/receipts/`；原始 `/tmp` logs 可能被系统清理。
以上 source commit、路径和 SHA-256 固定了本次审阅输入。较旧 revision 的逐次 reference 已从
工作树移除，但仍可从 Git history 恢复。

## 7. Task 行为：当前停止条件与诊断

第二次 task 的软件完成说明 transport path 正常，但不能区分 scene mismatch、视觉/点云偏差、
arm/hand behavior、未闭合、滑落或放置失败。固定 seed 只固定 diffusion 随机流，不固定真实观测、
物体初始状态或接触结果。因此不得通过放宽 SafetyGate、hand shaping 上限、expiry、ACK 或
publication bound 来“修复”物理任务。

下一次 task 授权前，诊断必须以只读方式补充：

1. B 前的 aligned RGB-D、policy point cloud、camera/state timestamps、calibration identity 和
   scene setup；
2. 每个 endpoint 的 raw physical prediction、shaped IPC endpoint、arm/hand feedback、tactile/contact、
   generation、deadline、publication 与 paired ACK timestamp；
3. 接近、闭合、抬起和放置阶段的有限场景帧；诊断写失败不得阻塞 control loop 或改变命令；
4. 与训练 episode 的 observation/action range、point-cloud frame 和 calibration identity 的离线比较。

完成诊断 review 后，新的 task 授权仍须单独指定任务布置、人员/障碍物、e-stop、设备健康、
device/seed、H/B、331 endpoint、25 秒、ACK timeout、接触/抓取/放置许可、单次无重试与 S/ESC/e-stop
职责。`completed=true` 仅证明 bounded command path；物理 pick/place 必须由操作者或任务成功传感器
单独记录。

## 8. 未来 `dexmani_policy` 合并

另一台机器上的 Policy 分支形成干净、可测试 commit 前，本仓不预先恢复、复制或修改其原型。合并
必须保护所有现有 sim 行为：dataset、sampler、normalizer、agent、`simple.v1` load/eval、
`CheckpointStore` selector 和已有 checkpoint/experiment 均不改变；Policy 不 import `dexmani_real`。

允许的增量仅限于可审计的数据/provenance：versioned Real Zarr provenance、fully resolved inference
config/data contract、canonical hashable artifact metadata、或在保持原 path-based API 的前提下增加
loaded-checkpoint restore helper。不得把 Real lifecycle、SafetyGate、hardware config、point-cloud runtime
或 receipt 迁入 Policy。

合并 review 至少要求：

- branch/HEAD/base/status，且 deployment/provenance diff 与训练改动分离；
- existing Policy tests、DP3 和受影响架构 smoke、旧 `simple.v1` load/eval、新 metadata round trip；
- `n_obs_steps=2/3 → pad_before=1/2` 参数化验证；
- schema 字段 additive、plain finite metadata、真实 array shape/dtype 验证；
- 旧 sim data 缺少 Real attrs 时保持原行为，未提交 checkpoint/dataset/experiment/W&B 生成物。

新的 Policy artifact 必须使用新 selector 或明确版本，不能静默替换当前 reference。Real 先增加纯
artifact/decoder tests，然后按 `--print-config` → CUDA preflight → recorded-observation replay → H2/H3
shadow 的顺序验证；任何物理执行仍需新的独立 review 和授权。出现旧 eval/training 不兼容、padding
再次硬编码、dirty/外部配置猜测 provenance、Policy 拥有 Real 安全逻辑或新旧 artifact 不能共存时，
停止合并。
