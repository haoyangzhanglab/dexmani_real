# Codex 任务书：修正 Policy Rollout Command Transport 丢序列问题

日期：2026-09-28  
仓库：haoyangzhanglab/dexmani_real  
前置任务：CODEX_SYNC_ROLLOUT_TIMING_FIX_TASK_20260928.md  
问题来源：同步 pacing 修正后的 deterministic transport regression

## 1. 任务背景与目标

上一阶段已经修正同步 policy rollout 的 control-period pacing：

- observation / synchronous inference / action realization / publish 均计入同一个 control period；
- dt=62.5 ms、boundary inference=20 ms 时：
  - boundary interval = 82.5 ms；
  - immediate post-boundary interval = 42.5 ms；
  - recovery pair span = 125 ms；
  - 8-step query interval = 500 ms；
- policy/action selection、action index、stop/revoke 等离线回归通过。

该 pacing 修正必须保留，不得回滚。

新发现的 blocker：

当前 command transport 是：

    Policy / Teleop / Replay producer
              ↓
    robot_command_ring(maxlen=1)
              ↓
       worker read_latest()
              ↓
      30 Hz worker polling
              ↓
             SDK

在 deterministic fake-clock 复现中：

    seq 9  publish @ 540.000 ms
    seq 10 publish @ 562.500 ms
    worker next poll ≈ 566.667 ms

由于 robot_command_ring 是 maxlen=1 的 latest-only mailbox，worker 在下一次 poll 时只能看到 seq 10，seq 9 从未进入 fake SDK callback。启动 sequence 1 也可发生同类覆盖。

因此当前问题不是 policy 模型、Mode 6 或 synchronous inference，而是：

> policy rollout 要求每个已接受 action 都能被 arm / hand worker 捕获，而现有 latest-only + periodic polling transport 只保证“读取最新值”，不保证逐 sequence delivery。

本任务目标是：

1. 保留已经修正的标准同步 policy pacing；
2. 保留现有 latest-target mailbox 作为共享 command payload；
3. 对 **policy rollout** 增加标准的 producer/consumer “wakeup + capture acknowledgement”；
4. 在下一条 policy command 可以发布前，确保上一条 command 已被所有相关 worker 复制到 worker-local owned memory；
5. 消除 30 Hz polling phase 对 policy command delivery correctness 的影响；
6. 不通过 FIFO backlog replay、提高 polling 频率或恢复错误 pacing 来掩盖问题；
7. 保留其他 workflow（teleop / replay / calibration）现有 publish_command 默认语义，除非测试证明存在必须同步修正的共享 bug。

工程优先级继续遵循 AGENTS.md：

hardware safety
> scientific correctness
> data traceability / reproducibility
> research iteration speed
> code simplicity
> generic extensibility。

## 2. 当前事实与必须保持的边界

### 2.1 当前 mailbox 是有意的 latest-value transport

RuntimeChannels 当前将：

    robot_command_ring

定义为 latest current-run target mailbox，且 maxlen=1。

SharedMemoryRingBuffer.read_latest() 明确允许 latest reader 跳过被覆盖的 frame。

因此不能简单把这个结构解释成“原本应该是 lossless queue”。

本任务只解决：

> policy rollout 在同步 action chunk 执行中需要逐 command capture guarantee，而 latest-only periodic polling 无法提供该 guarantee。

### 2.2 publish_command 是共享入口

publish_command 当前被多个 workflow 使用，包括：

- deployment / policy rollout；
- teleop；
- replay；
- keyboard control；
- camera calibration motion。

因此禁止直接把 publish_command 全局改成“所有调用都阻塞直到 worker 消费”。

默认调用必须继续保持现有非阻塞 latest-target publication 语义。

只有 policy rollout 显式请求 delivery guarantee。

### 2.3 “capture acknowledgement” 的精确定义

本任务中的 captured 表示：

> worker 已经通过 read_latest() 得到一个经过 seqlock 验证的完整 command，并已经将 RobotCommand payload 复制到该 worker 的本地 Python / NumPy owned memory。

capture acknowledgement 不表示：

- SDK 已返回；
- robot 已开始运动；
- robot 已到达目标；
- arm / hand 已同步完成运动。

只要 worker 已完成本地复制，之后 mailbox 被下一 sequence 覆盖就不会导致该 worker 丢失上一 command。

### 2.4 必须保持的已有行为

不得改变：

- synchronous inference；
- action_queue 为空才进行 policy inference；
- n_obs_steps / horizon / n_action_steps；
- action slice 起点 n_obs_steps - 1；
- Mode 6；
- set_servo_angle(..., wait=False)；
- XHand command representation；
- joint target / frame / unit；
- motion authority / run_id fencing；
- stop / quit / e-stop / fault 的最终安全检查；
- Raw data；
- 前一任务完成的 pacing 修正；
- inference accounting 修正。

## 3. 禁止方案

本任务禁止：

- 回滚到 publish_end + dt 的旧 pacing；
- RTC；
- asynchronous policy inference；
- policy prefetch；
- timed action plan / apply_at_ns；
- trajectory scheduler；
- 未来 action FIFO replay；
- 将 robot_command_ring 改成“大 ring 然后 worker 批量 drain”；
- worker 落后后 back-to-back 补发 stale commands；
- 仅把 worker loop_hz 提高到 60 / 100 / 200 Hz 作为 correctness fix；
- 重复发送同一个 command；
- action interpolation；
- chunk blending / EMA smoothing；
- 修改 Mode 6 / velocity / acceleration；
- 为此次 transport 修复增加新的 daemon / broker process；
- 让多个线程并发调用同一个 hardware SDK；
- 全局改变 teleop / replay 的 publish blocking 语义。

## 4. 推荐的最小 transport contract

### 4.1 保留 robot_command_ring

继续使用一个 coupled RobotCommand payload：

    run_id
    arm_present / arm_qpos
    hand_present / hand_qpos

sequence 继续来自 robot_command_ring.write(frame)。

不要新增第二份 command payload schema。

### 4.2 RuntimeChannels 增加 policy delivery handshake primitives

在 RuntimeChannels 中增加每个 actuator 的轻量同步原语：

- arm_command_wakeup: multiprocessing.Event
- hand_command_wakeup: multiprocessing.Event
- arm_command_captured_sequence: shared uint64 Value
- hand_command_captured_sequence: shared uint64 Value
- arm_command_captured_event: multiprocessing.Event
- hand_command_captured_event: multiprocessing.Event

要求：

- wakeup event 用于把 policy command 从 worker 的普通 30 Hz polling wait 中提前唤醒；
- captured_sequence 是事实来源；
- captured_event 只用于高效唤醒 publisher，不能单独作为正确性证据；
- 所有 sequence 比较必须以 captured_sequence 为准；
- Event / Value 使用 multiprocessing context 创建并可随 spawn 正确传递；它们不是 SharedMemoryRingBuffer / Queue，不要为其伪造 close()/unlink()；现有 RuntimeChannels.close() 只继续显式清理由本仓库实际拥有且需要 close/unlink 的 ring / queue 资源；
- spawn / shutdown 语义保持可验证；
- 不为此新增 registry 或抽象层。

### 4.3 publish_command 增加显式的可选 delivery guarantee

保持现有调用：

    publish_command(shared, target, ...)

默认语义不变。

增加一个明确的 opt-in 参数，例如：

    delivery_timeout_s: float | None = None

当 delivery_timeout_s is None：

- 保持旧语义；
- write mailbox 后立即返回 publication stamp；
- 不等待 capture；
- 不要求 worker 被 wakeup event 提前唤醒。

当 delivery_timeout_s 为正数：

1. 在 motion_lock 下验证当前 authority；
2. 为 target 中 present 的 actuator 准备 capture wait：
   - 清理对应旧 captured_event；
3. write robot_command_ring，获得 sequence；
4. 保存 publication stamp；
5. 对 target 中 present 的 actuator set 对应 command_wakeup；
6. 释放 motion_lock；
7. 等待每个相关 worker 的 captured_sequence == sequence；
8. captured_event 只是 wait hint，每次醒来都重新检查 sequence；
9. 只有所有相关 worker 都 captured 后才返回 publication stamp。

必须避免：

- 在等待 capture 时持有 motion_lock；
- 因 captured_event 残留而误判成功；
- 使用 captured_sequence >= sequence 把“跳过当前 sequence、捕获了后一个 sequence”误判为成功。

若 captured_sequence > expected sequence：

- 这是 transport invariant violation；
- 必须 fail loudly，不能当作成功。

### 4.4 policy rollout 显式要求 delivery guarantee

PolicyRunner 的真实执行路径：

    publish_command(...)

改为显式请求 delivery guarantee。

delivery_timeout_s 使用：

    policy_info.control_dt_s

理由：

- policy 下一 control tick 的 nominal budget 就是一个 control_dt；
- 若相关 worker 在一个完整 control period 内仍无法把 command 复制到本地，则该 transport 已不能可靠支撑当前 policy cadence；
- 不应继续积累或回放 stale command。

注意：

- timeout 不是 robot-motion deadline；
- timeout 只证明 transport capture 未及时完成；
- 不得据此声称 robot 没有运动。

dry-run / execute=False 不需要 delivery wait。

其他 workflow 默认不传该参数，保持原 publication 语义。

## 5. worker 修改：policy command 可立即唤醒，feedback 仍维持原频率

### 5.1 目标

当前 arm / hand worker 将 command admission 和 feedback update 都量化到约 30 Hz loop。

修正后：

- 普通 feedback / health polling 仍保持 config.loop_hz；
- 非 policy、未请求 delivery guarantee 的 publisher 仍按原 periodic latest-read 行为被消费：每个 ordinary worker feedback/poll tick 仍必须检查 mailbox；
- policy publication 会 set 对应 wakeup event，使 worker 不必等待下一个 nominal 30 Hz poll 才读取 command；
- hardware SDK 仍由原 worker 单线程独占。

### 5.2 不允许通过第二 SDK thread 实现

不能：

    command receiver thread → SDK
    feedback thread         → same SDK

SDK IO 必须继续在 worker 主执行线程串行完成。

### 5.3 worker loop 使用普通 event-driven wait

可将当前“每轮末尾固定 LoopRate.wait()”改为小型、直接的 event-driven loop：

- 维护下一 feedback deadline；
- ordinary 30 Hz tick 到期时继续执行原有 mailbox poll 与 feedback / validity / error logic；尽量保持现有相对顺序：arm 为 command admission → feedback，hand 为 feedback → command admission；
- feedback deadline 之间，用 command_wakeup.wait(timeout=remaining_to_feedback)；
- policy command 到达时 wakeup 立即返回；
- explicit wake path 不额外强制一次 feedback read，避免 command arrival 把 feedback rate 提高；
- clear 自己的 wakeup event；
- read_latest()；
- 若 sequence != last_sequence：
  1. 将 command payload 保存在本地变量；
  2. 更新 last_sequence；
  3. 写 captured_sequence = sequence；
  4. set captured_event；
  5. 然后执行原有 target validation、authority check、SDK call、diagnostics；
- 完成 command 后重新进入同一循环；
- 不因为 command wakeup 而重置 feedback cadence；
- feedback loop missed full slot 时可像现有 LoopRate 一样 re-anchor，禁止 catch-up burst。

要求保持：

- e-stop / revoke / fault 检查频率不劣于当前约 30 Hz fallback；
- home request 最大响应延迟不劣于当前约一个 worker period；
- command wakeup 不能推迟 stop / revoke safety fence；
- arm stopped → enter_mode6 → authority recheck 逻辑保留；
- hand state failure timeout、passive / hold_current 行为保留。

实现应直接、局部；不要为两份 worker 引入大型 scheduler framework。

若存在一个很小的纯逻辑 helper 能避免两边 deadline 算法产生不同语义，可以使用；不要升级为通用 runtime subsystem。

## 6. capture 与 safety 的顺序

正确顺序必须是：

    verified read_latest()
        ↓
    copy command to worker-local owned memory
        ↓
    mark captured sequence
        ↓
    target validation
        ↓
    command_may_cross_sdk(...)
        ↓
    SDK call

capture ack 只说明 command 不会因 mailbox overwrite 丢失。

它绝不能替代 SDK 前的最终 authority check。

因此：

- stop / revoke 可以发生在 captured 之后、SDK 之前；
- 此时 worker 必须按现有逻辑 trace_unsent(..., reason="revoked")；
- publisher 已经得到 capture acknowledgement 不意味着该 command 必须穿过 SDK；
- 这是正确的 safety behavior，不算 transport loss。

## 7. delivery wait 的停止与失败语义

### 7.1 authority 在等待期间被撤销

如果 publication 已经成功写入 mailbox，但等待 capture 期间 run 被 stop / quit / e-stop / fault 撤销：

- 不要把它误报成 command delivery timeout；
- 立即结束 delivery wait；
- 保持 publication evidence 为 accepted；
- worker 后续若读取该 command，仍由 run_id / safety fence 拒发并记录；
- 不等待满整个 timeout。

### 7.2 authority 有效但 capture 超时

若：

- delivery_timeout_s 已到；
- 当前 run authority 仍然有效；
- 任一 required worker 的 captured_sequence != sequence；

则必须 fail closed。

推荐定义一个小的专用异常，例如：

    CommandDeliveryTimeout

错误信息必须包含：

- expected sequence；
- arm captured sequence；
- hand captured sequence；
- 哪些 actuator required；
- timeout_s。

不要：

- 自动重发；
- 提高 worker Hz；
- 跳过该 command；
- 继续发布下一 command；
- 清空证据后继续。

Policy rollout 应按现有 policy failure / shutdown 路径停止。

## 8. diagnostics 与证据

### 8.1 保留现有 publication / SDK sequence correlation

必须继续能用：

    (run_id, sequence)

精确关联：

- policy publication；
- arm command；
- hand command。

### 8.2 建议增加 capture timestamp

为了验证 transport 修复，建议 worker command diagnostics 增加：

    capture_ns

表示 verified read_latest + local copy 完成后的主机 monotonic 时间。

要求：

- sent 与 explicitly unsent command 都记录 capture_ns；
- 不能把 capture_ns 命名为 execute / motion / arrival time；
- diagnostics group 字段保持一致；
- diagnostics off 不改变控制语义。

这样可派生：

    publish_to_capture_ms = capture_ns - publish_end_ns
    capture_to_sdk_ms     = sdk_start_ns - capture_ns

但 capture timestamp 不是本任务 correctness 的唯一依据；sequence 完整性仍是首要验收条件。

## 9. 必须加入的纯离线测试

全部测试不得连接真实硬件。

### 9.1 保留 pacing regression

上一任务的 deterministic timing regression 必须继续通过：

- boundary interval = 82.5 ms；
- post-boundary interval = 42.5 ms；
- recovery pair span = 125 ms；
- query interval = 500 ms；
- action index 不变。

transport fix 不得回滚 pacing。

### 9.2 将当前 transport reproduction 改为 acceptance test

保留当前已经复现问题的场景：

- policy dt = 62.5 ms；
- inference = 40 ms；
- worker nominal feedback/poll cadence = 30 Hz；
- seq 9 @ 540 ms；
- seq 10 @ 562.5 ms；
- 原实现 next poll ≈ 566.667 ms。

修正后要求：

- arm capture sequence 顺序包含 9, 10；
- hand capture sequence 顺序包含 9, 10；
- fake arm SDK callback 顺序包含 9, 10；
- fake hand SDK callback 顺序包含 9, 10；
- 每个 sequence 最多一次；
- 不允许 seq 9 被 seq 10 覆盖；
- 不允许为了通过测试把 seq 9/10 在一次 worker iteration 中 batch-drain / burst send。

### 9.3 启动 sequence 1 regression

当前复现中 sequence 1 也可能被下一 publication 覆盖。

新增或保留显式断言：

- first accepted policy publication 必须被 required workers capture；
- first command 不因 worker 初始 phase 被覆盖；
- start / HOME / B 现有 gate 保持不变。

### 9.4 capture acknowledgement test

构造 fake worker：

- publisher delivery_timeout_s > 0；
- worker 延迟一个确定时间后 read + capture；
- publish_command 必须等待 capture 后返回；
- 返回值仍是原 publication stamp，而不是 capture time；
- captured_event 提前 / 残留时，如果 captured_sequence 不匹配，不能误判成功。

### 9.5 sequence skip invariant test

构造：

    expected = 9
    captured_sequence = 10

要求：

- 不视为成功；
- 抛出明确 transport invariant error；
- 不继续发布后续 command。

### 9.6 delivery timeout test

authority 保持 RUNNING，但 required worker 永不 capture：

- 在 bounded timeout 后失败；
- 不发布下一 action；
- 不自动重发；
- 不 burst；
- 错误中包含 sequence 和 missing actuator。

### 9.7 revoke-during-wait test

publication 后、capture 前撤权：

- wait 尽快退出；
- 不误报 timeout；
- 后续 worker 如读取 command，SDK 前 authority check 拒发；
- stop / revoke regression 保持通过。

### 9.8 partial-target test

对：

- arm-only RobotCommand；
- hand-only RobotCommand；

若启用 delivery guarantee，只等待 present actuator。

不等待 absent actuator。

### 9.9 default behavior compatibility

对未传 delivery_timeout_s 的现有 publish_command callers：

- publication 仍非阻塞；
- 不要求 capture ack；
- teleop / replay / calibration 的 existing unit tests 不应因本任务被改写成 policy blocking semantics。

### 9.10 feedback cadence isolation

使用 fake clock / fake SDK 验证：

- policy command wakeup 可在两个 30 Hz feedback deadline 之间被处理；
- command wakeup 不应导致 feedback read 每次都跟着加速；
- 无 command wakeup 时，feedback cadence 与原 config.loop_hz 一致；
- missed feedback slot 不 batch catch-up。

## 10. 代码修改范围

预期只修改与 transport 直接相关的文件：

- dexmani_real/ipc/channels.py
- dexmani_real/robot/commands.py
- dexmani_real/robot/arm_worker.py
- dexmani_real/robot/hand_worker.py
- dexmani_real/deployment/runner.py
- dexmani_real/runtime/diagnostics.py（仅如增加 capture_ns）
- dexmani_real/deployment/diagnostic_analysis.py（仅如消费 capture_ns）
- focused tests

原则上不修改：

- dexmani_policy；
- model / dataset；
- action representation；
- xArm driver Mode 6 API；
- XHand driver command semantics；
- planning / IK；
- Raw / processed dataset。

若当前 local worktree 中上一 pacing 任务的三个文件尚未 commit：

- 必须保留这些修改；
- 不 reset / checkout / revert；
- transport diff 在其基础上继续；
- 最终报告中分别列出 pacing change 与 transport change。

## 11. Offline validation

至少运行：

    python -m unittest discover -s tests -v
    python -m compileall -q dexmani_real examples
    ruff format --check dexmani_real examples tests
    ruff check dexmani_real examples tests
    git diff --check
    git diff --stat
    git diff

若全仓仍只有已知 existing/unrelated lint：

- planner.py E402
- xhand.py E731
- tag_optimizer.py E741

不顺手修改；只确认本次 touched files focused lint 通过。

## 12. 真机准入规则

在本 transport 任务 offline acceptance 全部通过前：

> 禁止进入前一任务定义的 H1 / H2 真机实验。

完成 offline transport 修复后，也不得由 Codex 自行连接硬件。

只有操作者明确授权后，才使用原 8-step、同 checkpoint、同 seed 配置执行 H1。

H1 transport 验收必须新增：

1. 所有 accepted policy publication 都能在 required worker 找到同 sequence capture / SDK 或明确 unsent evidence；
2. 无 unexplained missing sequence；
3. publication → capture latency 有界；
4. capture → SDK latency 有界且可解释；
5. pacing 指标仍满足上一任务语义；
6. 不出现 stale backlog burst；
7. stop / timeout boundary 不发送旧 run command；
8. arm / hand safety behavior 无退化。

只有 H1 通过，才进入 H2 重复实验。

## 13. Codex 执行顺序

1. 读取 AGENTS.md；
2. 读取前一 pacing 任务书；
3. 读取本任务书；
4. 检查当前 HEAD / git status / git diff；
5. 保留当前 local pacing 修改，不 reset；
6. 重新读取 channels.py / ring.py / commands.py / arm_worker.py / hand_worker.py / runner.py；
7. 先复跑当前 transport reproduction，确认旧实现确实失败；
8. 实现最小 policy-specific wakeup + capture acknowledgement；
9. 不改变 non-policy publish 默认语义；
10. 加入 focused transport tests；
11. 确认旧 reproduction 转为通过；
12. 确认 pacing regression 仍通过；
13. 运行完整 offline validation；
14. 检查 diff，删除范围外修改；
15. 输出：
    - root cause；
    - transport contract；
    - touched files；
    - tests；
    - timing / sequence evidence；
    - remaining risks；
    - 真机未验证声明；
16. 停止，不连接硬件。

## 14. 最终完成标准

本任务完成不是“worker 更快”，而是：

> 在标准同步 policy pacing 下，每一个被 policy rollout 接受的 coupled RobotCommand，在下一条 policy command 允许覆盖 shared mailbox 前，已经被所有相关 hardware worker 捕获为本地 owned command；worker 随后仍通过原有 run_id / safety fence 决定该 command 是否允许穿过 SDK。

同时必须满足：

- 无 FIFO backlog replay；
- 无 stale command burst；
- 无 action skip；
- 无 global teleop/replay blocking semantic change；
- 无 Mode 6 改动；
- 无 pacing 回滚；
- 无 Raw 修改；
- 无未授权真机操作。
