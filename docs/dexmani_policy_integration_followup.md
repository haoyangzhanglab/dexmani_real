# DexMani Policy 对接后续开发清单

> 状态：待 `dexmani_policy` 当前开发分支稳定后实施。
>
> 本文只定义跨仓边界、执行顺序和验收标准，不表示这些修改已经落入
> `dexmani_policy`。截至本文编写时，该仓库工作树保持未修改状态。

## 1. 目标与当前阻塞

`dexmani_real` 已把 learned-policy 部署收紧为 fail-closed 边界：模型必须证明训练数据、
模型配置和实时传感器语义一致，才能在任何硬件 worker 启动前通过 inference preflight。
当前 `dexmani_policy` checkpoint 只有权重、优化器状态和部分 `train_params`，还不能提供
以下证据：

- 构造 agent 所需的完整 resolved inference config；
- 训练数据是否来自 Real Policy Zarr v5；
- `task_name`、控制周期和 observation/action 对齐语义；
- 点云 shape、坐标系、颜色来源、处理算法、配置哈希和桌面平面身份；
- episode 起点是否使用完整历史，以及训练窗口是否跨越 source 数据缺口。

因此，当前 checkpoint 会被
[`DexManiPolicyRuntime.load`](../dexmani_real/integrations/dexmani_policy.py)
明确拒绝。拒绝发生在硬件进程启动前，这是预期的安全行为；不要通过默认值、外部模型 YAML
或不完整 schema 猜测来绕过。

## 2. 跨仓所有权

| 合同 | 生产者 | 消费者 | 权威来源 |
|---|---|---|---|
| Policy Zarr schema/attrs | `dexmani_real.data.export` | `dexmani_policy` dataset | `dexmani_real/data/export.py` |
| observation/action 时间语义 | Real processing/export | sampler、agent、real adapter | persisted Zarr attrs |
| checkpoint resolved config | policy training workspace | policy eval、real adapter | training 时已 resolve 的 Hydra config |
| checkpoint data contract | policy dataset/trainer | real adapter | 实际打开的 Zarr 与数组 shape |
| model shape contract | policy agent | real deployment manifest | checkpoint `train_params` + loaded agent |
| realtime 点云语义 | real lifecycle | real adapter | `PointCloudLoopConfig` 的同一 resolved 实例 |

依赖方向保持为：

```text
dexmani_real 持久化格式
        ↓
dexmani_policy 读取、训练、保存 checkpoint
        ↓
dexmani_real 只通过 checkpoint 恢复并校验 agent
```

`dexmani_policy` 不应 import `dexmani_real`。跨仓常量需要按本文列出的精确值实现，并由两侧
contract tests 锁定。

## 3. 必须实现的 Policy Zarr v5 读取合同

### 3.1 Real 数据识别

当 root attrs 满足任一条件时，按 Real 数据处理：

```text
domain == "real"
schema_name == "dexmani-real-policy-zarr"
```

一旦被识别为 Real，以下字段必须全部存在且精确匹配，不允许把缺失值当作旧格式继续读取：

| 字段 | 必需值/类型 |
|---|---|
| `domain` | `"real"` |
| `schema_name` | `"dexmani-real-policy-zarr"` |
| `schema_version` | integer `5`，bool 不合法 |
| `episode_start_policy` | `"full_history"` |
| `obs_alignment` | `"obs[t]_before_action[t]"` |
| `observation_reference` | `"camera_source_monotonic_ns"` |
| `state_alignment` | `"camera_source_aligned_state"` |
| `max_observation_skew_s` | finite positive float，且须与 deployment runtime 一致 |
| `action_semantics` | `"deployment_grid_rate_limited_target"` |
| `arm_max_delta_rad_per_tick` | finite positive float 或 null，且须与 deployment runtime 一致 |
| `hand_max_delta_rad_per_tick` | finite positive float，且须与 deployment runtime 一致 |
| `deployment_equivalent` | boolean `true` |
| `profile` | 当前 real adapter 只接受 `"pointcloud"` 或 `"rgb_pc"` |
| `task_name` | 非空且不为 `"unknown"` |
| `dt` | finite positive float，单位为秒 |

Sim 数据保持现有行为，不应因 Real 合同新增全局强制字段。

### 3.2 `episode_ends` 边界

`ReplayBuffer` 构造时验证：

1. 一维、非空、integer dtype；
2. 每个值大于零且严格递增；
3. 末值等于每个 `/data/<key>` 的第一维；
4. 不把不同 `episode_ends` 段拼成一个 sequence。

Real Zarr v5 的每个 `episode_ends` 段代表一个 source-contiguous 训练 episode。同一 raw episode
删除中间行后可能产生多个段；这是正常输入，不应被合并。

### 3.3 attrs 与实际数组一起保留

`ReplayBuffer.copy_from_path` 需要保留 canonical root attrs，供 dataset 构造 checkpoint data
contract。不要只保留 attrs 而跳过数组检查，也不要信任 attrs 中声明的点数：

```text
point_cloud_num_points = replay_buffer["point_cloud"].shape[1]
point_cloud_feature_dim = replay_buffer["point_cloud"].shape[2]
```

当前实时链路要求 feature dim 为 `6`，列语义是 xArm-base `xyzrgb`，RGB 范围 `[0,1]`。

## 4. SequenceSampler 与 train/validation split

### 4.1 禁止 Real episode 起点左侧 padding

Real v5 的 `episode_start_policy=full_history` 表示训练样本必须已有完整 observation history：

```text
pad_before == 0
```

若调用者给 Real 数据配置 `pad_before != 0`，dataset 应在初始化阶段抛出带修复提示的
`ValueError`。不要静默改写参数，因为 resolved training config 必须忠实记录实际行为。

不要直接把所有 Sim/多任务配置全局改为 `pad_before=0`。建议在真实数据训练配置或命令中显式
设置 `dataset.pad_before=0`；Sim 是否保留起点 padding 由 policy 仓库自己的实验合同决定。

### 4.2 先筛选可采样 episode，再划分验证集

当前流程先随机选择 validation episode，再由 `SequenceSampler` 过滤短 episode。若验证集恰好
选中短 segment，可能得到空 validation sampler，甚至把唯一可训练段排除。

正确顺序：

```python
episode_lengths = np.diff(np.r_[0, episode_ends])
effective_pad_before = min(max(pad_before, 0), horizon - 1)
effective_pad_after = min(max(pad_after, 0), horizon - 1)
min_required = horizon - effective_pad_before - effective_pad_after
eligible = episode_lengths >= min_required

val_mask = get_val_mask(..., candidate_mask=eligible)
train_mask = eligible & ~val_mask
```

约束：

- 没有 eligible episode 时立即报错；
- `get_val_mask` 只能从 `candidate_mask` 中采样；
- 至少保留一个 eligible train episode；
- `max_train_episodes` 只对 eligible train mask 下采样；
- train/validation sampler 都不得原地修改共享 mask。

## 5. Checkpoint 必需字段

### 5.1 `TrainCheckpoint`

在 `dexmani_policy/common/checkpoint_io.py` 为 checkpoint 增加：

```python
inference_config: dict[str, Any] | None = None
data_contract: dict[str, Any] | None = None
```

Policy 自身可继续读取缺少这两个字段的历史 checkpoint，用于仿真评测或恢复旧训练；
`dexmani_real` 必须拒绝它们。不要在 real adapter 中构造兼容默认值。

保存位置仍为 `payload["state"]`，加载历史 checkpoint 时使用 `.get(...)`。新增字段需要加入
checkpoint round-trip smoke test。

### 5.2 resolved inference config

`TrainWorkspace.save_hydra_config` 已经接收完整配置。应在 resolve 后将
`OmegaConf.to_container(cfg, resolve=True)` 的 plain mapping 保存在 workspace，并由 trainer
写入 checkpoint。

要求：

- 保存的是 resolved 值，不包含未解析 interpolation；
- 必须是 plain mapping；
- checkpoint 保存发生在 config snapshot 已建立之后，否则报错；
- real adapter 只 instantiate `cfg.agent`，不得 instantiate dataset 或 sim `env_runner`；
- `eval.use_ema` 和 `eval.denoise_steps` 由该内嵌配置决定，不再接收部署侧第二份模型 YAML。

### 5.3 data contract 必须显式白名单化

由实际训练 dataset 生成 plain-Python data contract，不建议直接无筛选复制所有 Zarr attrs。
Real point-cloud 单任务 checkpoint 至少包含：

```text
domain
schema_name
schema_version
episode_start_policy
obs_alignment
profile
task_name
dt
sensor_modalities
point_cloud_num_points
point_cloud_feature_dim
point_cloud_frame
point_cloud_color_source
point_cloud_policy_id
point_cloud_config_sha256
point_cloud_table_plane_abcd_json
point_cloud_sampling
point_cloud_transform
```

所有 NumPy scalar、tuple 或映射要转换为稳定的 Python `int`、`float`、`str`、`bool`、`list`
和 `dict`。`point_cloud_num_points/feature_dim` 必须来自实际数组 shape，而不是配置声明。

当前 real adapter 不支持 multitask checkpoint。若 trainer 的 dataset 没有唯一 data contract，
应显式标记为不可部署或拒绝生成 deployment checkpoint，不能静默保存 `None` 后让问题推迟到
机器人运行入口。

### 5.4 `train_params` 完整性

保存并在加载时逐项严格比较：

```text
n_obs_steps
n_action_steps
action_dim
horizon
action_key
tcp_dim
hand_dim
use_faas
control_action_dim
```

字段存在时，即使值为 `None` 也要与 loaded agent 的属性比较；不要用“双方非 None 才比较”的
兼容逻辑掩盖 shape/config 漂移。

## 6. Inference restore API 与 EMA

当前 `load_ckpt_for_inference` 会从磁盘读取 checkpoint。real adapter 在构造 agent 前还必须读取
同一 checkpoint 做数据合同校验，因此未来会产生一次重复的大文件读取。

建议在 `dexmani_policy/training/eval_utils.py` 拆为：

```python
def restore_checkpoint_for_inference(agent, checkpoint, use_ema): ...

def load_ckpt_for_inference(agent, store, path, use_ema):
    checkpoint = store.load(path)
    restore_checkpoint_for_inference(agent, checkpoint, use_ema)
```

现有 policy eval 调用保持不变；real adapter 在该 API 合入后改用已读取的 checkpoint 对象。

Policy 仿真评测是否允许“请求 EMA 但缺少 EMA 权重时退回 raw model”可保持现状。真实部署必须
fail closed：内嵌 `eval.use_ema=true` 且 `ema_model_state is None` 时拒绝启动。

## 7. FAAS 与 normalizer 验证

real adapter 输入始终是 native joint state `7+12=19D`，但 FAAS agent 会在 normalizer 前映射为
`7+32=39D`。验证 normalizer 时必须使用模型空间：

```text
普通模型 joint_state normalizer: 19D
FAAS 模型 joint_state normalizer: 7 + faas_mapper.MAPPED_JOINT_DIM = 39D
action normalizer: agent.action_dim
point_cloud normalizer: 6D
control_action: inverse FAAS 后的 native 19D 或 EE 21D
```

不要用 `hand_dim` 推导 FAAS mapped dim；当前配置里的 `hand_dim=12` 表示 native XHand 维度。

## 8. 建议的实现顺序

### 阶段 A：锁定 policy 当前基线

1. 等待正在进行的 policy 改动合并或 rebase；
2. `git status --short` 确认工作树归属；
3. 记录现有 smoke、训练和评测命令结果；
4. 不同时重构 agent、dataset hierarchy 或 checkpoint 格式。

### 阶段 B：数据读取与 sampler

修改：

- `dexmani_policy/datasets/replay_buffer.py`
- `dexmani_policy/datasets/base_dataset.py`
- `dexmani_policy/datasets/sampler.py`
- 真实数据训练所用配置

先完成 Zarr v5、episode ends、full-history 和 eligible split tests，再进入 checkpoint 工作。

### 阶段 C：checkpoint provenance

修改：

- `dexmani_policy/common/checkpoint_io.py`
- `dexmani_policy/training/workspace.py`
- `dexmani_policy/training/trainer.py`
- `dexmani_policy/smoke_test.py`

确保单进程、DDP main process、resume 和 milestone/latest/best 保存路径都携带相同合同。

### 阶段 D：严格 inference restore

修改：

- `dexmani_policy/training/eval_utils.py`
- 必要的 policy eval tests
- `dexmani_real/integrations/dexmani_policy.py` 中一次读取优化

先保持现有 public loader 可用，再增加已读取 checkpoint 的 restore 函数。

### 阶段 E：生成新数据与 checkpoint

1. 使用 `dexmani_real` 重新生成 deployment-equivalent processed HDF5 v10；
2. 导出全新的 Policy Zarr v5 目标，不覆盖旧 Zarr；
3. 用 `pad_before=0` 训练单任务 point-cloud policy；
4. 生成包含 resolved config/data contract 的新 checkpoint；
5. 只做离线 real adapter preflight，确认通过后再进入受控硬件准入流程。

旧 Zarr/旧 checkpoint 不原地升级，因为缺失的 source continuity、训练配置和点云 provenance
无法从权重可靠推断。

## 9. 必需测试矩阵

### 数据合同

- Sim Zarr 无 Real attrs 时保持现有读取行为；
- Real v5 完整 attrs 可读取；
- 任意非 v5、缺字段、错误 `obs_alignment`、错误 `state_alignment`、错误 `episode_start_policy` 全部拒绝；
- 空、非整数、非递增、末值不匹配的 `episode_ends` 全部拒绝；
- point cloud 声明与实际 `[T,N,C]` shape 不一致时拒绝。

### sampler

- 一个 source gap 的两侧永不出现在同一 sample；
- Real `pad_before=1` 拒绝，`pad_before=0` 可用；
- short segment 不进入 train 或 validation candidate；
- 只有一个 eligible segment 时保留为 train，不产生空训练集；
- `pad_after` 行为与现有 action-tail 训练语义一致。

### checkpoint

- inference config/data contract save-load round-trip；
- 历史 checkpoint 仍可被 policy 自身按既有用途读取；
- real adapter 对历史 checkpoint 明确拒绝；
- `train_params` 的 `None`、FAAS、EE action 和 control dim 漂移均被发现；
- `use_ema=true` 且无 EMA 时 real preflight 拒绝；
- DQ-RISE 不重新读取外部 codebook path，checkpoint buffer 是权威状态。

### 跨仓离线验收

至少覆盖：

```bash
# dexmani_policy
python -m compileall -q dexmani_policy
python dexmani_policy/smoke_test.py dp3

# dexmani_real（不启动硬件）
python -m compileall -q dexmani_real examples tests
python -m unittest discover -s tests -p 'test_*.py'
python examples/run_policy.py \
  --runtime dexmani_real.deployment.fake:FakePolicyRuntime \
  --task-name <task> \
  --print-config
```

还应增加一个只加载 checkpoint、构造 agent、核对 manifest/data contract 后退出的离线 preflight
测试或 CLI。它不得创建 `RuntimeChannels`、camera、arm 或 hand worker。

## 10. 环境前置检查

本次审查时，本机 `base`、`real_robot`、`sim` 环境均无法 import `hydra`，而 policy
`requirements.txt` 声明 `hydra-core==1.2.0`、`pyproject.toml` 声明 `hydra-core>=1.3`。
实施前需要：

1. 由 policy 仓库确定唯一依赖权威和支持版本；
2. 在实际部署 Python 环境安装 policy 及其锁定依赖；
3. 验证 `hydra`、`omegaconf`、`torch`、`dexmani_policy` 可 import；
4. 验证 CUDA/device 选择，但不连接机器人；
5. 把环境 smoke 纳入部署说明，不能依赖开发机偶然的 `PYTHONPATH`。

## 11. 禁止的捷径

- 不接受部署侧第二份 model YAML；
- 不为缺失 data contract 填默认 Real 值；
- 不接受非 v5 的 Policy Zarr；
- 不把 source gap 两侧拼成连续训练窗口；
- 不用左侧重复帧模拟真实 observation history；
- 不在真实部署中把缺失 EMA 静默降级为 raw weights；
- 不因测试方便而放宽 task、dt、shape、frame、配置哈希或 table identity；
- 不让 model loader import/instantiate dataset、sim env runner 或硬件 SDK。

## 12. Definition of Done

只有同时满足以下条件，`dexmani_policy` 对接才算完成：

- policy 仓库的 Sim 既有流程没有非预期语义变化；
- Real Zarr v5 和 full-history/segment 边界有自动化测试；
- 新单任务 checkpoint 自描述且无需外部模型 YAML；
- real adapter 对正确 checkpoint 离线通过，对每类错误合同明确拒绝；
- FAAS、joint action、EE action 的 normalizer/control shape 均有覆盖；
- checkpoint 只读取一次，或有测量证明重复读取可接受；
- 两仓 compile、测试和 diff check 通过；
- 文档记录实际命令、依赖环境和仍未进行的硬件验证；
- 真实硬件测试仍须单独获得授权，并满足
  [`deployment_review.md`](deployment_review.md) 的准入条件。
