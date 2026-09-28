# Codex 任务书：修正同步策略 rollout 的控制周期语义并完成验证

日期：2026-09-28  
仓库：haoyangzhanglab/dexmani_real  
代码基线（任务书加入前）：2452c223c417fc84b9642b9c49055cf52ab45659  
执行基线：以 Codex 启动时的当前 HEAD / 工作树为准；不得 checkout、reset 或回退到上述旧 SHA；必须保留该 SHA 之后的任务书提交以及用户已有修改。  
关联实验：maniflow/pick_place_toy/2026-09-28_04-48-18_42  
关联分析：PICK_PLACE_TOY_ROLLOUT_ANALYSIS_20260928.md

## 1. 任务目标

本任务只修正 dexmani_real 当前同步推理部署路径中已经确认的控制周期语义错误，并通过纯离线测试、诊断分析和受控真机实验验证修正结果。

核心意图：

1. 对齐社区常见同步控制循环语义：一次控制 tick 的 observation、必要的 policy inference、action realization、publish 都属于同一个 control period 的工作预算。
2. 保留现有 chunked synchronous inference：动作队列为空时同步推理，逐控制步消费动作。
3. 保留当前 Mode 6 + set_servo_angle(..., wait=False) 的 xArm joint online trajectory planning 控制方式。
4. 不通过新机制掩盖问题：不引入 RTC、异步推理、未来动作定时队列、跨 chunk blending、EMA smoothing、额外插值、自动补偿或动作索引偏移。
5. 证明修正只改变错误的时间节拍，不改变模型输入 / 动作索引的定义、policy 调用与 action selection 语义、安全授权和 Raw evidence 语义。对于“同一保存输入 + 同一 RNG 状态”的离线回归，模型输出与所选动作必须一致；真机闭环中由于观测采样时刻和机器人轨迹会随 pacing 修正而变化，后续模型输出不要求与旧 rollout 数值相同。
6. 真机实验只用于验证时间链路和实际运动结果；未获得真机数据前，不宣称机器人抖动已经修复。

工程优先级遵循 AGENTS.md：

hardware safety > scientific correctness > data traceability / reproducibility > research iteration speed > code simplicity > generic extensibility。

## 2. 已确认的当前问题

### 2.1 当前 runner 把控制周期锚定在 action publish 之后

当前 PolicyRunner.step() 在动作发布后执行：

    self.next_step_ns = stamp + int(self.policy_info.control_dt_s * 1e9)

这使 control_dt_s 实际表示：

    本次 action 已发布
        + 再等待一个完整 control_dt
        + 下一轮 observation / inference / action

因此，当 action queue 为空且本轮发生同步推理时，推理耗时被额外加入相邻控制周期，并继续推迟后续 observation / query 时刻。

当前 8-step 实验已经观测到：

- control_dt_s = 62.5 ms；
- inference median ≈ 28.34 ms；
- internal publication interval median ≈ 63.56 ms；
- chunk-boundary publication interval median ≈ 92.39 ms；
- query observation interval 相比 nominal 0.5 s 多约 36.7 ms。

这里的错误不是“同步推理”，而是“完成本轮工作以后又从 publish time 重新等待完整 dt”。

### 2.2 社区同步循环的目标语义

参考：

- Hugging Face LeRobot 的 CycleTimer / synchronous control loop；
- xArm-Developer/lerobot_robot_ufactory 的 uf_lerobot_eval.py；
- LeRobot DiffusionPolicy.select_action() 的 observation queue + action queue 行为。

应采用以下语义：

    tick start
        ↓
    read observation
        ↓
    if action queue empty:
        synchronous inference
        ↓
    select one action
        ↓
    realize / publish
        ↓
    only wait for the remaining time in this tick budget
        ↓
    next tick

即 observation、inference、publish 均计入同一个 dt。

### 2.3 修正后的预期行为必须准确理解

该修正不会让 chunk boundary 的第一条新动作必然恢复为 62.5 ms 间隔。

若：

- dt = 62.5 ms；
- boundary tick inference = 20 ms；
- 其他处理耗时忽略；

则标准同步循环应表现为：

- boundary 前一普通动作 → 新 chunk 第一动作：82.5 ms；
- 新 chunk 第一动作 → 下一动作：42.5 ms；
- 两个间隔之和：125 ms = 2 × dt；
- 两次 8-step query start 间隔：500 ms。

也就是说，推理耗时产生的局部长间隔由下一轮较短等待吸收，不应永久膨胀后续控制周期。

严禁把验收标准错误写成“boundary interval 必须等于 62.5 ms”。

## 3. 明确非目标

本任务禁止顺手实现以下内容：

- RTC / Real-Time Chunking；
- asynchronous inference；
- inference prefetch；
- timed plan / apply_at_ns / future command scheduler；
- 修改 n_obs_steps、horizon、n_action_steps 的训练语义；
- 把 action slice 起点从 n_obs_steps - 1 改成其他值；
- 跳过“迟到”的预测 action；
- chunk overlap blending；
- EMA / low-pass smoothing；
- jerk-limited trajectory generator；
- 16 Hz policy action 人为插值到 30/60 Hz；
- 更换 xArm Mode 6；
- 改用 set_servo_angle_j；
- 修改速度 / 加速度参数作为本次修复的一部分；
- 为了测试方便修改 Raw rollout；
- 为历史行为增加兼容分支或新 framework。

若测试发现上述机制可能有研究价值，只记录为后续问题，不在本任务实现。

## 4. 代码修改范围

### 4.1 必改：dexmani_real/deployment/runner.py

采用最小修改。

PolicyRunner.step() 已在 gate 前读取：

    now = time.monotonic_ns()

并通过：

    if now < self.next_step_ns:
        return

因此，当该 gate 通过、step() 真正进入一个 control tick 时，直接复用这个已经判定到期的 `now` 作为本轮起点，不要再额外读取一次时钟：

    tick_start_ns = now
    self.next_step_ns = tick_start_ns + dt_ns

这样 tick 的 deadline 与 gate 使用同一个时间样本，避免不必要的二次时钟读取和语义漂移。

随后执行现有：

- observation read；
- history append；
- queue-empty synchronous inference；
- action_queue.extend；
- action_queue.popleft；
- ActionRealizer；
- publish_command；
- recorder。

删除或停止使用当前以下“从工作完成时重新计时”的逻辑：

1. 成功 publish 后：

    self.next_step_ns = stamp + dt_ns

2. IK 非成功路径中：

    self.next_step_ns = time.monotonic_ns() + dt_ns

原则：

- 同一 control tick 只设置一次下一 tick 时间；
- 本轮耗时小于 dt：下一轮等待剩余预算；
- 本轮耗时超过 dt：下一次 step 可尽快进入，但每次 step 仍最多消费一个 action；
- 不做 while catch-up；
- 不一次补发多个历史 action；
- 下一 tick 从它自己的真实 tick start 重新计算预算，不形成 backlog burst。

不要为了实现该语义引入新的 scheduler class。当前 runner 内直接、清晰实现即可。

### 4.2 必改：inference_ms 统计口径

当前 runner 已记录：

    inference_end = time.monotonic_ns()

但 summary 使用的 inference_ms 在额外诊断 / recorder 工作之后才取 time.monotonic_ns()，会把模型调用结束后的处理时间混入 inference latency。

修改为严格使用：

    inference_ms = (inference_end - infer_start) / 1e6

要求：

- diagnostics 中 infer_start_ns / infer_end_ns 与 summary 的 inference_ms 口径一致；
- 这里表示 deployment model call latency，不宣称是纯 GPU kernel time；
- 开启 diagnostics 不得改变该统计定义。

### 4.3 原则上不改：动作与观测语义

必须保持：

- action_queue 为空才同步推理；
- 每个 control tick 更新 ObservationHistory；
- n_obs_steps = 2 的模型输入语义不变；
- control_action 对应完整预测从 n_obs_steps - 1 开始；
- 当前实验 8-step 的执行索引仍为 1...8；
- inference 结束后的新 feedback 仍只用于执行侧 realization / telemetry，不反填到已经完成推理的 observation；
- joint action 仍是 absolute joint target；
- ActionRealizer / projection 只保留现有安全与等价角行为。

任何改变这些语义的 diff 都视为任务越界。

### 4.4 原则上不改：xArm / XHand worker

本任务第一阶段不修改：

- arm_worker.py；
- hand_worker.py；
- commands.py 的 transport contract；
- ROBOT_COMMAND_DTYPE；
- worker loop_hz；
- Mode 6 参数。

原因：已有 2026-09-28 基线没有发现未知 sequence 丢失；当前首要任务是先修正同步 loop pacing。

但修正后会出现一个需要验证的回归风险：

    boundary long interval
        +
    immediate post-boundary shorter interval

当 post-boundary publication interval 接近或小于 30 Hz worker 的 33.3 ms poll period 时，latest-only transport 可能存在覆盖风险。

因此必须通过 sequence evidence 验证，而不是预先重写 transport。

若出现 accepted publication 没有对应 SDK send 且不是明确 revoke / timeout：

- 立即将本任务标记为未通过；
- 保存证据；
- 不用提高动作频率、重复发送、睡眠补偿或新队列机制掩盖；
- 单独提出后续 transport 修正任务。

## 5. 纯离线测试要求

所有自动测试必须无真实硬件副作用。

优先修改 tests/test_rollout_diagnostics.py 中现有同步 runner 测试，而不是新建大规模测试框架。

### 5.1 更新现有 timing test

现有 test_runner_keeps_targets_and_exposes_boundary_delay 已注入：

- dt = 62.5 ms；
- n_action_steps = 8；
- fake inference = 20 ms；
- 16 publications；
- 2 inference calls。

修正后必须验证：

1. inference calls == 2；
2. publications == 16；
3. 在该确定性 fake-observation / fake-model 回归中，action 内容与修改前相同；
4. pred_index 仍为：

       1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8

5. 第二次 query 的 infer_start - 第一次 query 的 infer_start == 500 ms；
6. boundary interval：

       stamps[8] - stamps[7] == 82.5 ms

7. boundary 后恢复 interval：

       stamps[9] - stamps[8] == 42.5 ms

8. 两者之和：

       stamps[9] - stamps[7] == 125 ms

9. 普通 steady-state interval 保持 62.5 ms；
10. 在确定性 fake-clock 测试中，若 diagnostics stub 本身不推进 fake clock，diagnostics on/off 必须得到完全相同的 target 序列和发布时间序列；不要把这一断言外推为真实运行中诊断开销必须为零。

注意：不要把第 6 项错误改成 62.5 ms。

### 5.2 新增 inference accounting test

构造 fake model：

- predict() 固定推进 fake clock，例如 20 ms；
- diagnostics.record 或 recorder check 再额外推进 fake clock，例如 7 ms。

断言：

- stats.inference_ms 只记录 20 ms；
- infer_end_ns - infer_start_ns == 20 ms；
- 额外诊断耗时不污染 inference latency；
- diagnostics on/off 不改变 control_action。

### 5.3 新增 overrun 行为测试

构造 inference > dt，例如 80 ms。

要求：

- 单次 PolicyRunner.step() 最多发布一个 action；
- 不存在 while catch-up；
- 下一次外层 step 可以立即开始；
- 下一 tick 重新从自己的 tick_start 计算 next_step；
- 不累计多个“欠下的”控制 tick；
- action 顺序不变；
- 不跳 action。

该测试验证同步 loop 在超预算时“无额外 sleep，但不批量补发”的标准行为。

### 5.4 保留并继续通过现有安全测试

至少覆盖：

- prediction diagnostics 不额外采样；
- stop / revoke 后 prediction 不得继续发送；
- publication rejection；
- SDK exception；
- diagnostic overflow；
- exact run_id / sequence matching；
- action slice / overlap logic；
- n_action_steps override；
- Raw / diagnostics 不被测试修改。

## 6. 静态与离线验证命令

禁止连接真实设备。

最低要求：

    python -m unittest discover -s tests -v
    python -m compileall -q dexmani_real examples
    ruff format --check dexmani_real examples tests
    ruff check dexmani_real examples tests
    git diff --check

若全仓 ruff 仍命中任务前已经存在的问题：

- 不顺手修复；
- 明确列出“existing / unrelated”；
- 对本次修改文件执行 focused lint 并确保通过。

完成后必须人工检查：

    git diff --stat
    git diff

确保只包含预期文件。

## 7. 诊断分析要求

为了让真机实验能直接验证修复，不要求重构 diagnostics；允许对 dexmani_real/deployment/diagnostic_analysis.py 增加最小的派生指标，但不能改变 Raw 或 trace 格式。

建议新增：

### 7.1 post-boundary interval

对于每个完整 chunk boundary：

    post_boundary_interval =
        publish(new_chunk_step_1) - publish(new_chunk_step_0)

仅在 new chunk 至少包含 step 0 和 step 1 时统计。

### 7.2 boundary recovery pair span

    boundary_pair_span =
        publish(new_chunk_step_1) - publish(old_chunk_last_step)

其含义是：

    boundary interval + immediate post-boundary interval

对 dt = 62.5 ms 的健康同步 loop，且 inference 未超过一个周期时，期望接近：

    2 × dt = 125 ms

旧代码通常接近：

    2 × dt + inference

该指标比只看 boundary interval 更能直接验证 pacing 修复。

### 7.3 query drift

继续保留并重点报告：

    observed query interval - n_action_steps × dt

修正目标是让 inference latency 不再系统性累加到 query cadence。

新增分析必须：

- 只读取已有 diagnostics；
- 不修改 Raw；
- 对缺少完整后继 step 的 boundary 显式跳过；
- 不跨 run / episode / timeout boundary 拼接。

## 8. 真机实验任务

Codex 不得自行连接或驱动真实设备。

只有操作者明确授权并现场执行后，才能分析新 session。

### 8.1 实验配置

保持与 2026-09-28 8-step 基线一致：

- experiment：maniflow/pick_place_toy/2026-09-28_04-48-18_42
- checkpoint：epoch=0566-step=00060000-milestone=100pct.pt
- weights：EMA
- inference_steps：4
- seed：0
- horizon：16
- n_obs_steps：2
- n_action_steps：8
- action_mode：joint
- control_dt_s：0.0625
- diagnostics：on
- max duration：30 s
- device：cuda:0

本任务禁止使用 12-step 作为主验收，因为它改变了 replan interval，会混淆 pacing 修复效果。

### 8.2 Phase H1：单次机制验证

先运行 1 次 30 s 真机实验。

必须检查：

1. diagnostics 三份 trace complete=true；
2. dropped=0；
3. accepted publication 与 arm / hand SDK evidence 可按 (run_id, sequence) 精确关联；
4. 除明确 stop / timeout revoke 外，不存在 unexplained missing sequence；
5. query observation interval 不再系统性包含一次 inference latency；
6. boundary interval 仍允许较长；
7. immediate post-boundary interval 应相应缩短；
8. boundary_pair_span 应接近 2 × dt，而不是 2 × dt + inference；
9. effective action publication rate 应接近 nominal 16 Hz；
10. 没有新的安全、SDK、stale observation 或 worker fault。

只有 H1 通过，才进入重复任务实验。

### 8.3 Phase H2：重复 pick_place_toy

建议至少 3 次 30 s、固定 checkpoint / seed / scene protocol 的重复实验。

记录：

- inference latency；
- query interval / query drift；
- internal / boundary / post-boundary publication interval；
- boundary_pair_span；
- arm / hand SDK send interval；
- accepted publication 与 SDK send 完整性；
- model boundary target revision；
- measured qpos / qvel；
- task stage：approach / grasp / lift / transport / place；
- 是否出现肉眼可见周期性顿挫或回拉。

重要：

- 模型自身约 3° 的跨 query target revision 可能仍存在；
- 该数值不是本任务必须降低的指标；
- 若 target revision 不变但时间 cadence 修正，说明代码修复生效而模型问题仍需单独研究；
- 不允许为了让视频更顺临时加入 smoothing。

## 9. 真机验收标准

### 9.1 代码修复通过

必须同时满足：

- query cadence 不再由 inference latency 系统性膨胀；
- boundary_pair_span 回到约 2 × dt 的同步 loop 时间预算；
- action 索引、policy/action selection 语义、观察历史定义未改变；同一保存输入与同一 RNG 下输出一致；
- 没有 unexplained publication → SDK sequence 丢失；
- safety / revoke 行为未退化；
- diagnostics 完整；
- 代码和测试通过。

### 9.2 不应错误使用的验收条件

以下不能作为本任务“代码修复是否正确”的硬性条件：

- chunk boundary interval 必须是 62.5 ms；
- boundary target jump 必须消失；
- ManiFlow revision 必须下降；
- pick_place_toy 成功率必须立即提高；
- measured qvel 不得有任何方向改变。

这些涉及同步推理固有局部延迟、模型重新规划和任务策略本身。

### 9.3 必须失败并停止推进的情况

若出现任一项：

- accepted command 无 SDK evidence，且不是明确 revoke；
- action index 改变；
- diagnostics on/off 输出动作不同；
- stop 后仍有新命令穿过 SDK；
- Raw 被修改；
- worker / hardware fault 增加；
- timing 修复依赖 smoothing / async / RTC 才能通过；
- 新代码一次 step 发布多个积压动作；

则本任务不通过。

## 10. 实验后文档更新

只有获得真实新 session 后，才更新 PICK_PLACE_TOY_ROLLOUT_ANALYSIS_20260928.md。

更新内容必须区分：

1. 已确认代码错误：control period 从 publish-completion 锚定改为 tick-start budget；
2. 修复后的时间指标；
3. worker transport 是否完整；
4. 实际运动是否改善；
5. 仍然存在的模型 prediction revision；
6. 尚未证明的因果关系。

不得把：

    pacing 修复完成

写成：

    模型抖动根因已修复

除非额外证据真正支持。

## 11. Codex 执行顺序

严格按以下顺序执行：

1. 重新读取 AGENTS.md、本任务书、runner.py、tests/test_rollout_diagnostics.py；
2. 确认当前 HEAD 与工作树现状；不得 checkout/reset 到任务书中的历史代码基线；保留用户已有修改，并确认自 2452c223 之后是否存在与本任务相关的代码变化；
3. 本任务书是用户明确要求放在仓库根目录的执行文档；任务期间不要删除、移动或重命名它；
4. 只实现 runner timing + inference accounting 的最小代码修正；
5. 更新 / 新增 focused offline tests；
6. 运行 focused tests；
7. 运行全套无硬件 offline validation；
8. 检查 git diff；
9. 如需要，仅增加诊断派生指标与对应测试；
10. 再次运行 offline validation；
11. 输出代码修正摘要、测试证据、未验证风险；
12. 停止，不连接真实硬件；
13. 等操作者执行 H1/H2 并提供 session 后，再做真机数据分析；
14. 有真机证据后更新 PICK_PLACE_TOY_ROLLOUT_ANALYSIS_20260928.md。

## 12. 最终交付物

代码阶段：

- runner.py 的最小同步 pacing 修正；
- inference latency 统计修正；
- focused unit tests；
- 必要时 diagnostic_analysis.py 的最小派生指标；
- offline validation 结果；
- 清晰 git diff。

实验阶段：

- H1 单次机制验证报告；
- H2 重复实验汇总；
- 新旧 timing 指标对照；
- sequence 完整性证据；
- residual jitter 的归因边界；
- 更新后的 PICK_PLACE_TOY_ROLLOUT_ANALYSIS_20260928.md。

任务完成的标准不是“代码看起来更复杂或更先进”，而是：

> dexmani_real 的同步 policy rollout 与社区标准同步控制循环保持一致，control period 语义正确、行为可复现、证据可追溯，并且没有通过额外控制机制改变实验系统本身。
