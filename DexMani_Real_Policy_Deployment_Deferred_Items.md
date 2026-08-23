# DexMani Real Policy Deployment — 推迟项 / 设计债记录

> 本文档记录 learned-policy 部署重构（Phase 4b–7）在对抗审查中识别出、但**刻意推迟**的问题。
> 每条给出：问题机制、影响评估（安全 vs 质量 vs 调参）、推迟原因、修复方向、以及**要落地所需的条件**。
> 配套文档：`DexMani_Real_Policy_Deployment_Refactor_Plan.md`（原方案）；审查与修复结论见 memory `deployment-refactor-status`。

推迟判定原则（来自 CLAUDE.md / AGENTS.md）：

- **安全类**必须立即修复（fail-closed），本表**不含**任何安全缺陷——安全门、generation 失效、collision、
  delta 拒绝、manifest 启动校验均已在审查中闭合。
- 下表均为**控制质量 / 调参 / 跨仓**类，需要真机数据、设计决策或另一仓库改动才能定案，贸然改动反而可能
  引入错误或过度抽象，故记录而非现在就改。

---

## 1. §6 timestamp-grid 重采样：frame-count 对齐引入跨模态时滞

- **涉及**：`dexmani_real/integrations/dexmani_policy.py` `_encode`（`min(Ta,Th)` + `[-n_obs:]`）。
- **严重度**：控制质量（非安全，机器不会因此失控）。

### 机制

当前 `_encode` 用**帧数**对齐 arm/hand 两条反馈流：

```python
t = min(arm.shape[0], hand.shape[0])
arm = arm[-n_obs:]; hand = hand[-n_obs:]
joint = np.concatenate([arm, hand], axis=-1)  # [n_obs, 19]
```

`arm_state_ring` 与 `hand_state_ring` 的发布速率是**独立可配**的（`arm_loop_hz` / `hand_loop_hz`）。
以 `n_obs_steps=2` 为例：

| 流 | 假设速率 | `[-2:]` 覆盖的窗口 |
|---|---|---|
| arm | 50 Hz | 最近 ~40 ms（`t-40ms`、`t-20ms`） |
| hand | 15 Hz | 最近 ~133 ms（`t-133ms`、`t-67ms`） |

于是拼接出的「当前」帧实际是 `[arm@t-20ms, hand@t-67ms]`——hand 分量比 arm 分量**旧 ~47 ms**，且
行 0 把 `arm@t-40ms` 与 `hand@t-133ms` 拼在一起。这个联合状态**在物理上从未同时存在过**。

### 影响

模型是在**同步采样**的 `(arm_t, hand_t)` 上训练的。推理时喂入的联合向量存在系统性跨模态时滞，且帧间距
与训练控制栅格不一致。对接触密集的操纵任务，这是**静默的控制质量退化**——无报错、无 metric、无 fail-closed。
第一轮验证 agent 同时指出：锚点行（最后一帧）总是「最新 arm」配「最新 hand」，所以时滞是**有界**的，
不是灾难性的；但帧间距不一致是真实存在的 train/deploy 差异。

### 推迟原因

完整的 §6 时间戳栅格重采样是**实质性的设计改动**，且影响是控制质量而非安全。`FrameWindow` 已逐帧携带
`source_monotonic_ns`，数据具备，缺的是采样逻辑与 skew 阈值——阈值需要真机时序数据才能定。

### 修复方向

1. 把 arm/hand/点云历史重采样到统一栅格：对 `T=n_obs_steps` 个栅格点
   `t_k = anchor - (n_obs-1-k)*step_dt`，逐流取「最近的因果帧」（`source_ns <= t_k`），必要时插值。
2. 加跨模态 skew gate：若 `max(|arm 锚点时间 - hand 锚点时间|, |joint 锚点 - 点云锚点|) > 界` 则
   fail-closed 或 log-and-drop（§6 的 "cross modal skew" 检查）。
3. 兜底（较粗）：在 manifest 校验中拒绝 `arm_loop_hz != hand_loop_hz` 的配置——但过严，仅作应急。

### 落地所需

真机时序数据：arm/hand 实际发布速率、skew 分布、不同速率配置下的控制表现。据此定 skew 阈值并验证重采样。

---

## 2. delta cap 与 coalesced catch-up 冲突：误杀健康 run

- **涉及**：`dexmani_real/deployment/coordinator.py` `_select_due_step`（合并 overdue 步）
  与 `dexmani_real/policy/safety_gate.py`（`abs(arm_end - arm_start) > max_arm_delta_rad`）。
- **严重度**：调参 / 语义（fail-closed，机器停在原地，但会错误中止 run）。

### 机制

`safety_gate` 的逐 tick delta 上限默认 `arm_max_delta_rad_per_tick = 8°`（`config/defaults.py`，广播到全部 7 关节）、
`hand_max_delta_rad_per_tick = 0.1 rad`。它量的是**测量反馈 → 合并后端点**的跳变。

而 `_select_due_step`（§7 "stale prefix rejection" 的意图）会把过期的 N 个计划步**合并成一个端点**（返回最新 due 步）。
于是：

- 计划采纳后若推理耗时 ~150 ms（dt=62.5 ms，约 2–3 个 tick），首端点是「当前位姿 + 2~3 个模型步」的跳变；
- 模型单步 ~5–8°，合并跳变 ~12–16°，超过 8° cap → `ARM_DELTA_LIMIT` → `GATE_REJECTED` →
  `_end_policy_run(abort=True)` 直接中止 run；
- 同理，B 之后**第一帧**从任意起停位姿到首目标的大位移也会触发。

也就是说：模型每步输出都在限内，但**合并追赶**让它看起来超限，健康 run 在普通时序下被误杀。

### 影响

Fail-closed（无危险运动，机器人保持 hold），但会**错误中止正常部署**，需要人工重按 B，破坏「可反复 START/STOP」目标。

### 推迟原因

这是调参 + 语义的二选一问题，需要真机数据才能定：
- 8°/tick 到底该不该这么紧？（IK 自己的 per-joint 跳变限是 30–40°/tick，见 `planning/types.py`。）
- delta 该量「目标 vs 测量」还是「相邻计划步」？
- 单次 delta 越界该 abort 还是 hold/丢一步继续？

### 修复方向（任选或组合）

1. 按合并步数缩放 delta 预算：`max_delta * coalesced_steps`。
2. 对 adopt/promote 后的首次选择（追赶）豁免 delta 检查。
3. 改按「相邻计划步」量 delta（而非目标 vs 测量）。
4. delta 越界改为 drop-and-continue（hold 一 tick），而不是 abort。
5. 重调 cap：per-joint（对齐 IK 跳变限）而非广播标量。

### 落地所需

真机运动数据：模型单步输出量级、真实推理延迟下的合并频率、以及 8°/tick 是否真会导致关节超速。
据此定 cap 值与 abort/hold 语义。

---

## 3. `_decode` 时间戳 k=1 vs k=0：每块一步滞后

- **涉及**：`dexmani_real/integrations/dexmani_policy.py` `_decode`（`steps = arange(1, n+1)`）。
- **严重度**：控制质量（一步相位滞后，非 hang）。

### 机制

`_decode` 把 `control_action[0]` 排到 `anchor + 1*step_dt_ns`（k=1 起）。而模型的**对齐约定**是
`predict_action_from_cond` 里 `start = n_obs_steps - 1` 切片，即 `control_action[0]` 落在「最后一帧观测的
地平线位置」= 锚点时刻 `t` 本身。按此约定 `control_action[0]` 应在 `anchor + 0*dt`（k=0）。

于是每条 chunk 的首命令目标整体晚一个控制 tick（~62.5 ms）。

### 影响

每块边界一个 tick 的恒定相位滞后；`replan_stride_steps=8` 时逐计划重复，可能累积成轻微滞后的控制器。
**不是 hang**。且 §7 的 stale-prefix rejection 会部分掩盖它——推理延迟（~100 ms > 1 tick）通常使
`control_action[0]` 在被采纳时已过期并被合并，所以有效 k 主要由延迟决定，而非这个 off-by-one。

### 推迟原因

方案 §7 只写了 `anchor + k*dt`，**没有钉死 k**。k=0 还是 k=1 取决于「首命令应在观测时刻执行」还是
「提前一 tick 前瞻」。真机时序才能判断这一步是否真的影响控制。改动本身很小（`arange(0, n)` 或
`arange(1, n+1)`），缺的是决策与验证。

### 修复方向

1. 若 `control_action[0]` 应在锚点执行 → `steps = arange(0, n)`。
2. 若一 tick 前瞻是有意的（「target the future」）→ 在注释与 manifest 中显式记录，不再含糊。

### 落地所需

一个明确的 k 语义决策 + 真机控制表现 A/B（k=0 vs k=1）对比。

---

## 4. `first_command_timeout_s=5s` 对冷 CUDA / CPU 偏短

- **涉及**：`dexmani_real/deployment/config.py` `first_command_timeout_s`（默认 5.0 s）；
  `coordinator.py` 从 `run_started_ns`（RUNNING 转换时刻）起计。
- **严重度**：调参（冷启动下会误中止，需覆盖配置才能 boot）。

### 机制

模型在 ARMED 阶段已 `load()`（实例化完成），但**首次 forward**（CUDA context 初始化、cuDNN autotune、
首个 diffusion denoise）在冷设备上可能 > 5 s。而 first-command watchdog 从 RUNNING 开始计到首条命令发布，
冷启动慢 → 误判「模型没产出」→ 中止一个其实健康的部署。

### 影响

Fail-closed（安全），但冷环境下需要人工调大 `first_command_timeout_s` 才能跑起来。

### 推迟原因

5 s 是启发式默认。正确的默认值取决于模型 + 硬件的实际首推理耗时（模型相关、设备相关），需真机量测。

### 修复方向

1. 把计时起点从 `run_started_ns` 改为「首个成功观测组装 / 首个 predict 进入」（worker 在模型 forward 前
   先信号就绪，只对 forward 本身计时）。
2. 或调大默认并/或在 config 注释中说明调参。

### 落地所需

真机首推理耗时的实测分布（CPU / 冷 GPU / 热 GPU）。

---

## 5. 点云契约不在 checkpoint 里（需改 `dexmani_policy`）

- **涉及**：`dexmani_real/integrations/dexmani_policy.py` `_cfg_select(cfg, "agent.num_points")` 等；
  `dexmani_policy/common/checkpoint_io.py` `build_train_params`（**未**存 num_points/pc_dim/sensor_modalities）。
- **严重度**：控制质量（跨仓，静默精度损失）。

### 机制

manifest 的点云契约（`point_cloud_num_points` / `pc_dim` / `sensor_modalities`）**只从模型 config.yaml 读**，
不与 checkpoint 对账。而 `dexmani_policy` 的 `build_train_params`（checkpoint 内的 `train_params`）只存
`n_obs_steps/n_action_steps/action_dim/horizon/action_key/tcp_dim/use_faas/hand_dim/control_action_dim`，
**没有** num_points/pc_dim/sensor_modalities。

于是：用 num_points=2048 + FPS+random 采样训练的 checkpoint，若 operator 指向 `agent.num_points=1024` 的
config.yaml，`load` 会通过（点云编码器权重与 N 无关，`load_state_dict(strict=True)` 照样过），模型被静默
喂入 1024 个确定性点，去跑按 2048 个 FPS 采样点训练的权重——正是方案 §14.3.3 点名的 train/deploy 预处理
不一致，直到首次 predict 才可能以 N 形状错误或纯质量下降暴露，而非 load 时报错。

### 影响

静默的策略质量损失。manifest 的 fail-closed 契约在此**没有覆盖到**（因为它无数据可校验）。

### 推迟原因

修复在**另一个仓库**（`dexmani_policy`）：要改 `build_train_params` 存入这些字段，并**重训/重存 checkpoint**
才能让旧 checkpoint 也带字段。跨仓改动 + 数据迁移。

### 修复方向

1. 在 `dexmani_policy/build_train_params` 增加 `num_points`、`pc_dim`、`sensor_modalities`（及 `state_dim`）。
2. 在 `dexmani_real` 的 `load` 中对账 `checkpoint.train_params.num_points == manifest.point_cloud_num_points`
   等，checkpoint 缺字段时 fail-closed。

### 落地所需

`dexmani_policy` 仓库的 `build_train_params` 改动 + 现有 checkpoint 的兼容/重存策略。

---

## 6. `use_ema=True` 静默回退到 raw 权重

- **涉及**：`dexmani_policy/training/eval_utils.py` `load_ckpt_for_inference`（EMA 缺失时仅打印 WARNING）；
  `dexmani_real/integrations/dexmani_policy.py` `use_ema = getattr(config, "use_ema", True)`。
- **严重度**：控制质量（跨仓，静默替换）。

### 机制

checkpoint 无 `ema_model_state` 时，`load_ckpt_for_inference` 静默回退到 raw 权重，只打黄色 WARNING。
manifest 不记录任何 EMA 信息。于是 `use_ema=True` 对上「无 EMA 保存」的 checkpoint，会以 raw 模型冒充 EMA
模型运行——一次静默的权重/行为替换。

### 影响

operator 以为自己跑的是 EMA 模型，实际是 raw 权重；无报错、无记录。

### 推迟原因

回退逻辑在 `dexmani_policy`（跨仓）。本仓可做的拦截（见下）是个小改动，但放在「是否 fail-closed」的决策点上
一并处理。

### 修复方向

1. 在 `dexmani_real` 的 `load` 中，`load_ckpt_for_inference` 后检查
   `store.load(ckpt).ema_model_state is None`，若 `use_ema=True` 却缺 EMA 则 raise（fail-closed）。
2. 或在 manifest 中记录 EMA 存在性并显著提示。

### 落地所需

确认「缺 EMA 应视为错误还是可容忍回退」的产品决策；若 fail-closed，本仓即可实现。

---

## 优先级建议

| 项 | 类别 | 依赖 | 建议顺序 |
|---|---|---|---|
| 5 点云契约进 checkpoint | 跨仓 + 数据 | dexmani_policy 改动 | 先做（最接近「静默错误」） |
| 6 use_ema fail-closed | 跨仓（本仓可拦截） | 决策 | 可立即本仓实现 |
| 2 delta vs 合并追赶 | 调参 | 真机运动数据 | 真机验证后 |
| 1 timestamp-grid 重采样 | 控制质量 | 真机时序数据 | 真机验证后 |
| 3 k=1 vs k=0 | 控制质量 | 决策 + A/B | 与 1 一起 |
| 4 first_command_timeout | 调参 | 真机首推理耗时 | 顺手 |

原则：任何一项在真机首跑前**不会**引入安全问题（它们都不在 fail-closed 路径上）；但要避免把它们
误记为「已完成」。真机首跑验收清单仍见 memory `deployment-refactor-status` 末尾。
