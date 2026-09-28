# Codex 任务书：按社区标准 FIFO 修正 Policy Command Transport

日期：2026-09-28  
仓库：haoyangzhanglab/dexmani_real  
前置任务：CODEX_SYNC_ROLLOUT_TIMING_FIX_TASK_20260928.md  
问题来源：同步 pacing 修正后的 deterministic transport regression

## 1. 任务目标

上一阶段已经正确修复 synchronous policy rollout 的 control-period pacing；该修正必须保留，不得回滚。

当前已确定性复现新的 transport correctness bug：

    seq 9  publish @ 540.000 ms
    seq 10 publish @ 562.500 ms
    worker next nominal poll ≈ 566.667 ms

当前 robot_command_ring 是 maxlen=1 的 latest-value mailbox，worker 以约 30 Hz read_latest()。因此 seq 9 可在 worker 读取前被 seq 10 覆盖；启动 sequence 1 也可发生同类覆盖。

本任务不设计新的 acknowledgement protocol。目标是采用参考项目中已经成熟的标准分层：

> state / observation 使用 latest ring；action / command 使用 FIFO queue；hardware worker 收到 command 后立即处理，而不是周期性读取 latest target。

参考实现：

1. xArm-Developer/lerobot_robot_ufactory：
   synchronous eval loop 直接调用 robot.send_action()；joint control 内直接调用 Mode 6 + set_servo_angle(..., wait=False)，不存在 policy → latest mailbox → periodic worker 的中间覆盖窗口。
2. Universal-Control/ManiUniCon：
   state 使用 SharedMemoryRingBuffer；robot action 使用 SharedMemoryQueue FIFO。Policy write_action() 入队，Robot process read_all_action() 消费 action queue。
3. real-stanford/diffusion_policy：
   observation 使用 SharedMemoryRingBuffer；robot controller command 使用 SharedMemoryQueue FIFO。RealEnv.exec_actions() 将 actions/timestamps 入队给独立 RTDEInterpolationController。

dexmani_real 已经选择 hardware SDK 独立 worker，因此本任务采用与 ManiUniCon / Diffusion Policy 同类的 **FIFO command transport**，而不是 UFACTORY LeRobot 的同进程直接 SDK 调用。

## 2. 物理意义

当前 bug 不是“少了一条日志”，而是某个 policy action 可能根本没有进入硬件 SDK。

对 joint Mode 6：

    q_cmd[k-1]
        ↓
    q_cmd[k] 被 mailbox 覆盖，SDK 从未收到
        ↓
    q_cmd[k+1] 到达
        ↓
    Mode 6 直接从当前运动状态在线重规划到 q_cmd[k+1]

Mode 6 会维持其内部速度/加速度过渡连续，但无法恢复已经丢失的 policy waypoint。因此可能造成：

- 参考轨迹路径改变；
- 目标步幅突然变大；
- 运动相位改变；
- 重规划方向变化时更明显的减速 / 反向；
- command timing 与训练动作语义不一致。

对 dexmani_real 更严重的一点是 action 是 coupled 19D：

    7D arm + 12D hand

arm / hand 是独立 worker，轮询相位不同。latest-only transport 允许：

    arm 执行 seq k
    hand 执行 seq k+1

或反过来。

这会把模型在同一逻辑时刻预测的 arm-hand coordination 撕裂成跨时刻组合。在 grasp / contact / lift 阶段可能表现为：

- 手指提前或滞后闭合；
- arm 提前抬升；
- 接触尚未建立便搬运；
- 抓取力形成时 arm 已移动；
- 视觉上出现顿挫、回拉或抓取失败。

因此 command transport 必须保证 FIFO 顺序，不允许 accepted command 被后续 command 静默覆盖。

## 3. 社区参考方案

### 3.1 UFACTORY LeRobot：同步直接发送

核心结构：

    observe
      ↓
    predict/select action
      ↓
    robot.send_action(action)
      ↓
    xArm set_servo_angle(..., wait=False)
      ↓
    wait remaining loop budget

它没有独立 command mailbox，所以 selected action 与 SDK call 之间不存在 periodic latest-read 覆盖。

本项目由于保留 hardware worker / SDK ownership，不直接复制“policy process 持有 SDK”这一拓扑。

### 3.2 ManiUniCon：state ring + action FIFO

ManiUniCon 明确区分：

    robot state → SharedMemoryRingBuffer
    robot action → SharedMemoryQueue

Policy：

    for action in actions:
        shared_storage.write_action(action)

Robot process：

    actions = shared_storage.read_all_action()

action queue 是 FIFO，不会像 latest ring 一样静默覆盖未消费 action。

其默认配置还将 robot action buffer 配为有界 queue。

### 3.3 Diffusion Policy：observation FILO + command FIFO

Diffusion Policy README 明确：

- SharedMemoryRingBuffer 用于获取最新 observation；
- SharedMemoryQueue 用于 RTDEInterpolationController command。

其独立 robot controller 通过 FIFO command queue 收到 SERVOL / SCHEDULE_WAYPOINT 等命令。

因此本任务遵循同一原则：

    latest state ≠ lossless command

不要继续用 state-ring 语义承载 command delivery。

## 4. 必须保留的已有语义

不得改变：

- synchronous inference；
- action_queue 为空才做下一次 policy inference；
- 上一任务已经修正的 tick-start pacing；
- n_obs_steps / horizon / n_action_steps；
- action slice 起点 n_obs_steps - 1；
- joint action 为 absolute target；
- Mode 6；
- set_servo_angle(..., wait=False)；
- XHand action representation；
- run_id / safety-state authority；
- SDK 前最终 authority recheck；
- stop / quit / e-stop / fault；
- Raw / processed data；
- inference accounting。

## 5. 禁止方案

本任务禁止：

- latest mailbox + per-command capture ACK 自定义协议；
- RTC；
- asynchronous policy inference；
- policy prefetch；
- timed plan / apply_at_ns；
- 新 trajectory scheduler；
- chunk blending / EMA smoothing；
- action interpolation；
- 提高 worker polling rate 来“降低覆盖概率”；
- 回滚正确的 synchronous pacing；
- 修改 Mode 6；
- 修改 velocity / acceleration；
- worker 落后后 batch burst 发送历史 commands；
- 静默 drop FIFO command；
- 为此次修复增加 broker / daemon process；
- 多线程并发调用同一个 hardware SDK。

## 6. 目标架构

采用标准 producer → FIFO → consumer：

    Policy / Teleop / Replay / Calibration
                │
                │ publish_command()
                ▼
        ┌─────────────────┐
        │ global sequence │
        └─────────────────┘
             │       │
             ▼       ▼
       arm command  hand command
          FIFO           FIFO
             │             │
             ▼             ▼
       arm worker      hand worker
             │             │
             ▼             ▼
        xArm SDK        XHand SDK

state / observation 通道继续使用现有 ring buffers。

不再使用 robot_command_ring 作为 actuator command transport。

## 7. RuntimeChannels 修改

### 7.1 command queues

在 RuntimeChannels 中建立：

- arm_command_q
- hand_command_q

使用 multiprocessing context 创建的标准 FIFO Queue。

每个 queue 必须有有限 maxsize，避免无限积压。

不要因为 payload 很小再实现新的 custom shared-memory queue；当前 command 只有少量 float，在 16–30 Hz 下 multiprocessing.Queue 的序列化成本不是瓶颈。优先 correctness 和简单性。

推荐将 queue 容量作为 RuntimeChannelsConfig 的明确小整数配置，默认值保持保守，例如 8；若 Codex 能从现有最大 producer cadence / SDK latency 给出更好的有界值，可使用经测试证明的值，但不要使用无界 queue。

### 7.2 global command sequence

新增一个由 motion_lock 保护的 monotonic command sequence counter。

publish_command 每接受一次 logical RobotCommand：

    sequence += 1

同一个 coupled command 的 arm / hand queue item 必须携带完全相同的：

- sequence；
- run_id；
- 对应 actuator target。

sequence 用于：

- arm / hand diagnostics correlation；
- 检测 duplicate / reordering / gap；
- stop/revoke evidence。

不要分别为 arm / hand 生成不同 sequence。

### 7.3 删除 command ring transport dependency

robot_command_ring 不再承担 action command delivery。

如果确认仓库中没有其他独立用途，应：

- 从 RuntimeChannels 资源中删除 robot_command_ring；
- 删除 commands.py 的 read_robot_command() latest-ring path；
- 更新相关 tests / comments / cleanup。

不要保留两套 command transport 形成 legacy ambiguity。

如果确有任务范围外用途必须暂时保留，必须在最终报告中说明；policy/teleop/replay 的正常 actuator command path 仍必须唯一走 FIFO，不允许同一 command 双写 ring + queue。

## 8. publish_command 标准 FIFO 语义

publish_command 继续作为共享 publication 入口。

在 motion_lock 内：

1. 验证 run_id / SafetyState authority；
2. 生成全局 sequence；
3. 构造 arm / hand queue item；
4. 对 target 中 present 的 actuator enqueue；
5. 只有所有 required enqueue 都成功，publication 才算 accepted；
6. publication timestamp 在成功 enqueue 后记录；
7. release motion_lock。

禁止：

- 等待 SDK completion；
- 每 command ACK；
- waiting for consumer capture；
- enqueue 后等待 worker；
- silent queue overflow；
- partial coupled acceptance。

### 8.1 bounded queue full

若 required queue 已满：

- publication 必须 fail closed；
- 不允许继续下一 policy action；
- 不静默覆盖最旧 command；
- 不 drop newest command 后假装 accepted；
- 不扩大 queue 后自动继续。

若 coupled command 的一个 actuator 已 enqueue、另一个 enqueue 失败：

- 在释放 motion_lock 前撤销当前 motion authority / latch appropriate failure；
- 已成功入队的旧 run item 后续必须被 worker 的 run_id / authority fence 拒绝；
- publication 不得记录为完整 accepted coupled command。

实现应最小化这种 partial enqueue 窗口，并为其加入 deterministic unit test。

## 9. worker 标准 queue consumer

### 9.1 command delivery 不再由 30 Hz polling 决定

当前 worker：

    periodic tick
       ↓
    read_latest command
       ↓
    maybe SDK

改为标准 queue-driven worker：

- command queue 到达时立即可被 consumer 获取；
- feedback 仍按 config.loop_hz 定期读取；
- command receipt 与 feedback cadence 分离；
- 仍在同一个 worker 主线程串行调用 hardware SDK。

推荐直接使用：

    queue.get(timeout=remaining_to_next_feedback_deadline)

语义：

- 若 command 先到：立即处理 command；
- 若 timeout：执行一次 feedback/update tick；
- 处理完 command 后继续等待；
- command arrival 不强制额外 feedback read；
- feedback deadline 不因为 command arrival 被永久平移。

这是普通 producer/consumer event loop，不新增自定义 scheduler。

### 9.2 command processing

每个 worker queue item 到达后：

1. 检查 sequence 严格单调；
2. 验证 target shape / finite / physical limits；
3. 检查 run_id / authority；
4. 必要时执行现有 mode recovery；
5. SDK call；
6. diagnostics 记录同一 sequence。

若收到：

    sequence <= last_sequence

视为 duplicate / reordering invariant violation，fail loudly。

若 sequence > last_sequence + 1：

不能简单认为 transport 丢失，因为该 actuator 可能在中间 coupled command 中 absent。

因此 gap 检查必须结合该 actuator 是否 required；最简单做法是每个 actuator queue 只包含该 actuator required 的 commands，并保证 queue 本身 FIFO，不额外要求全局 sequence 连续。

### 9.3 不 burst backlog

worker 每取到一条 command 就正常完成该 command 的 SDK path，再取下一条。

不要调用 get_all() 后在同一个时刻循环 burst 发送全部 backlog。

正常情况下 queue 应接近空，因为 producer cadence 远低于 worker 可处理的 command IO latency。

若 diagnostics 表明持续 backlog / enqueue-to-SDK latency 增长，视为系统无法维持 cadence，不通过真机验收；不要通过 stale burst 掩盖。

## 10. arm / hand coupled action

同一个 RobotCommand 可能包含 arm + hand。

publication 必须给两个 worker：

    same run_id
    same sequence

两个 worker independently consume their FIFO。

不要求两个 SDK 同一 CPU 纳秒调用，但必须：

- 不丢相同 logical sequence；
- 不乱序；
- 不将 arm seq k 与 hand seq k+1 误认为一个 coupled action；
- diagnostics 可以按 (run_id, sequence) 精确关联。

stop / revoke 可导致某个已入队 command 在 SDK 前被明确拒发；这属于 safety behavior，不是 silent transport loss。

## 11. diagnostics

继续保留：

- publication run_id / sequence；
- SDK run_id / sequence；
- target；
- SDK start/end/status。

建议 queue item 同时携带 publication timestamp，或在 diagnostics 中增加：

    enqueue_ns

worker 可记录：

    dequeue_ns

以派生：

    enqueue_to_dequeue_ms
    dequeue_to_sdk_ms

这些名称必须准确：

- enqueue ≠ execute；
- dequeue ≠ motion start；
- sdk_start ≠ robot arrival。

不要为了 diagnostics 改变 transport timing。

## 12. 必须加入的纯离线测试

所有测试不得连接硬件。

### 12.1 pacing regression 必须继续通过

保持上一任务：

- boundary = 82.5 ms；
- post-boundary = 42.5 ms；
- pair span = 125 ms；
- 8-step query interval = 500 ms；
- action index 不变。

### 12.2 原 overwrite reproduction 转为 FIFO acceptance

沿用原场景：

    seq 9 @ 540 ms
    seq 10 @ 562.5 ms
    old periodic poll ≈ 566.667 ms

修复后必须验证：

- arm fake SDK 顺序收到 9, 10；
- hand fake SDK 顺序收到 9, 10；
- 每个 required sequence 恰好一次；
- seq 9 不被 seq 10 覆盖；
- 不依赖提高 worker Hz；
- 不依赖 ACK handshake。

### 12.3 startup seq 1

验证 first accepted command 不因 worker initial phase 丢失。

### 12.4 queue order

连续 publish N 个 command：

- consumer 顺序严格等于 producer；
- 不 duplicate；
- 不 re-order；
- 不 latest-overwrite。

### 12.5 queue full

构造 stalled consumer 使 bounded queue 满：

- publication fail closed；
- 不 silent drop；
- 不 overwrite；
- 不继续 policy execution。

### 12.6 partial coupled enqueue failure

构造 arm enqueue 成功、hand enqueue 失败：

- logical coupled publication 不算 accepted；
- motion authority 在 worker 可穿过 SDK 前被撤销；
- arm queued item 后续被明确 rejected/unsent；
- 不形成 half-coupled physical action。

### 12.7 revoke with queued commands

command 已 enqueue、尚未 SDK 时 revoke：

- worker dequeue 后必须在 SDK fence 拒发；
- 不因 FIFO “保证 delivery”而强制执行旧 command；
- old run queue item 不穿过 SDK。

### 12.8 event-driven command latency

fake clock：

- command 在两个 feedback deadlines 之间到达；
- worker 不必等到下一 30 Hz feedback tick才处理；
- command arrival 不额外制造一次 feedback read；
- feedback cadence 仍保持 nominal frequency。

### 12.9 non-policy callers

teleop / replay / calibration 继续通过同一个 publish_command FIFO path。

确认：

- 不要求 policy-specific ACK；
- 不引入 caller-specific hidden transport；
- existing tests 适配 FIFO 后仍保持各 workflow 的 action / safety semantics。

## 13. 代码修改范围

预期主要修改：

- dexmani_real/ipc/channels.py
- dexmani_real/robot/commands.py
- dexmani_real/robot/arm_worker.py
- dexmani_real/robot/hand_worker.py
- dexmani_real/runtime/diagnostics.py（如增加 enqueue/dequeue timing）
- dexmani_real/deployment/diagnostic_analysis.py（如消费新增 timing）
- tests

runner.py 原则上只保留上一阶段 pacing 修正，不应为了 FIFO transport 再增加 ACK / wait / scheduler。

原则上不修改：

- dexmani_policy；
- model / dataset；
- action representation；
- xArm Mode 6 driver semantics；
- XHand driver semantics；
- planning / IK；
- Raw / processed data。

## 14. Offline validation

最低要求：

    python -m unittest discover -s tests -v
    python -m compileall -q dexmani_real examples
    ruff format --check dexmani_real examples tests
    ruff check dexmani_real examples tests
    git diff --check
    git diff --stat
    git diff

若全仓仍只有既有：

- planner.py E402
- xhand.py E731
- tag_optimizer.py E741

不顺手修复。

## 15. 真机准入

在 FIFO transport offline acceptance 全部通过前：

> 禁止进入 H1 / H2。

Codex 不得连接真实硬件。

操作者后续执行 H1 时必须检查：

1. every accepted publication 都有 required actuator 的 SDK 或明确 unsent evidence；
2. 无 unexplained missing sequence；
3. arm / hand 同一 coupled sequence 可对应；
4. enqueue → dequeue → SDK latency 无持续积累；
5. command queue 无 backlog 增长；
6. pacing 仍符合上一任务；
7. stop / timeout 后 queued old-run commands 被拒发；
8. Mode 6 / XHand safety behavior 无退化。

## 16. Codex 执行顺序

1. 阅读 AGENTS.md；
2. 阅读前一 pacing 任务书；
3. 阅读本任务书；
4. 检查当前 HEAD / git status / git diff；
5. 保留上一阶段尚未 commit 的 pacing 修改；
6. 读取参考项目的相关实现，至少确认：
   - UFACTORY LeRobot direct send_action；
   - ManiUniCon SharedMemoryQueue action path；
   - Diffusion Policy SharedMemoryQueue command path；
7. 复跑当前 overwrite reproduction，确认旧 transport 会失败；
8. 将 command transport 改为标准 FIFO queue；
9. 删除 policy-specific ACK / wakeup 自定义方案，不实现该协议；
10. 修改 arm / hand worker 为 queue-driven command consumption + periodic feedback；
11. 加入 focused FIFO / safety tests；
12. 确认 overwrite reproduction 转为通过；
13. 确认 pacing regression 仍通过；
14. 运行完整 offline validation；
15. 检查最终 diff，只保留本任务必要修改；
16. 输出：
    - physical root cause；
    - 与参考项目的架构映射；
    - FIFO transport contract；
    - touched files；
    - tests；
    - sequence evidence；
    - queue backlog / failure behavior；
    - remaining risks；
    - H1 是否具备 offline 准入条件；
17. 停止，不连接硬件。

## 17. 最终完成标准

最终架构必须符合参考项目的共同原则：

    observation/state:
        latest ring / newest sample semantics

    action/command:
        FIFO queue / preserve command order semantics

而不是：

    command:
        latest-only mailbox + periodic polling

任务完成必须满足：

- accepted action 不被后续 action 静默覆盖；
- no custom per-command ACK protocol；
- no pacing rollback；
- no RTC / async policy inference；
- no stale backlog burst；
- no Mode 6 change；
- no Raw modification；
- no unauthorized hardware execution。
