# 键盘遥操作卡顿、抖动与命令拒绝故障复盘

## 1. 范围与结论

本文记录 `examples/keyboard_teleop.py` 调试期间出现的以下现象：

- hand-home 命令超时或被 worker 丢弃；
- 键盘输入触发 `RUNNING`，但机械臂没有动作；
- 控制循环从目标 33.3 ms 延长到约 60–62 ms；
- 命令被 `required actuator worker did not prepare the coupled command` 拒绝；
- 命令被 `arm per-tick delta limit violation` 拒绝；
- 机械臂表现为启停、卡顿、抖动，旋转时还可能出现动作变形。

结论：问题不是单一“步长过小”，也没有证据表明 8° 是 xArm 控制器的硬件限制。已核实的主要原因是：

1. 中间实现把 actuator prepare/ACK 等待放入 30 Hz 控制热路径，直接破坏控制周期；
2. 通用 arm delta 门禁曾被错误接入键盘路径，并以滞后的实测关节位置约束新的 IK endpoint；
3. 对完整 IK 结果逐关节裁剪会破坏其笛卡尔语义，不能作为键盘轨迹整形方法；
4. 旧的独立 arm/hand command ring 与分离的 state/generation 更新不能表达一个可原子撤销的命令所有权，导致命令丢弃、ACK 归属不清和跨执行器帧不一致风险。

当前修复采用非阻塞 coupled-command latest-wins 发布、原子 motion permit/ticket、按生产者区分的整形策略，以及 worker 端仅用于异常跳变的 20° fail-closed 兜底。软件合同已离线验证；是否完全消除物理抖动仍须受控实机测试确认。

## 2. 现场证据

| 日志现象 | 可以确认的事实 | 不能单独推出的结论 |
| --- | --- | --- |
| `hand home ... was not acknowledged within 1.000s` | 发布者没有收到该 action 的精确 worker ACK | 不能据此认定 XHand 硬件故障 |
| `ARMED → RUNNING` 后不移动 | 操作输入和 safety transition 已发生，但 actuator 没有接受可执行 endpoint | 仅凭这段日志不能区分 generation、运输或 SDK 层问题 |
| `required actuator worker did not prepare...`，同周期约 60–62 ms | 发布路径等待执行器准备，等待时间超过 30 Hz 的 33.3 ms 预算 | 不是简单提高控制频率即可解决 |
| `arm per-tick delta limit violation` | 候选在到达 arm SDK 前被软件门禁拒绝 | 不是 xArm 固件主动拒绝，也不是碰撞检查拒绝 |
| 拒绝后 `blocked until keys change` | 相同按键保持期间不再产生连续 endpoint | 机械臂启停必然会表现为命令流不连续，但物理抖动幅度仍需测量 |
| `Workspace boundary: x⁺0.715` | 虚拟目标到达配置的 workspace 边界 | 这是预期边界提示，不是本次抖动根因 |

日志中 `actual≈62 ms` 与当时约 60 ms 的 prepare 等待上限一致，并与命令拒绝同周期出现。这是热路径阻塞的直接证据。

## 3. 根因分析

### 3.1 控制热路径等待 worker

键盘循环目标频率是 30 Hz，每周期预算约 33.3 ms。中间实现要求 arm/hand worker 先 prepare，再允许 coupled command 成立；任一 worker 未及时响应时，主循环等待并拒绝该命令。

这种协议与 latest-wins 遥操作不匹配：

- worker 调度、SDK 状态读取和串口通信都可能超过一个控制周期；
- 等待占用了下一帧的计算和发布时间；
- timeout 后又将按键标记为 blocked，使一次瞬时延迟变成持续命令静默；
- arm 和 hand 的运行频率及负载不同，最慢 worker 决定整个遥操作延迟。

因此，60 ms 左右的循环超时和间歇停顿是该设计的结构性结果，而不是参数微调问题。

### 3.2 delta 门禁的 owner 和参考状态错误

8° 默认值在检查的旧提交 `b0a2603`、`26e6ff9` 中已经存在于 `PolicyParams.arm_max_delta_rad_per_tick`。它是策略/VR 命令连续性参数，不是 xArm SDK、Mode 6 或机械臂额定能力给出的统一限制。旧提交中的键盘 safety gate 没有启用该限制。

现场中间版本曾从 keyboard gate 读取不存在的 `ArmParams.max_command_delta_rad_per_tick`，随后又出现 `arm per-tick delta limit violation`，证明策略级限制被错误扩展到了键盘共享路径。

更关键的问题是比较参考：

```text
错误：new_ik_qpos - measured_qpos
正确的命令连续性：new_command_qpos - previous_accepted_command_qpos
几何与碰撞检查：measured_qpos -> new_command_qpos
```

实测位置天然滞后于已发送目标。连续按键时，虚拟 EEF 目标可领先实测位置若干帧；即使相邻命令平滑，`new_ik_qpos - measured_qpos` 也会不断增大并误触发门禁。拒绝后命令流中断，松键或换键又从新实测状态重建 anchor，于是形成“运动—拒绝—停顿—重新起步”的锯齿行为。

### 3.3 对 IK 结果做 joint clip 会造成动作变形

IK 输出 `q_ik` 是七个关节共同满足目标位姿的解。逐关节执行

```text
q_cmd = q_prev + clip(q_ik - q_prev, -limit, +limit)
```

后，`q_cmd` 通常不再对应原笛卡尔目标。特别是在腕部旋转、接近奇异位形或 IK 分支变化时，不同关节被不同比例裁剪，EEF 位置和姿态会耦合偏移。这解释了为什么放宽限制能减少拒绝，却不能从机制上消除“动作变形”。

日志能够证明 delta 拒绝和命令间断；动作变形与 joint clip 的因果关系符合运动学机制，但其物理幅度没有在日志中测量，仍需实机轨迹数据确认。

### 3.4 旧命令运输和撤销边界不完整

旧实现分别发布 arm 与 hand record，并分别由两个 worker 消费。即使两个 record 使用同一 `action_id`，也不能保证它们同时成为各自通道的最新值。state transition 与 generation 推进也不是同一个原子操作。

这会产生三类问题：

- arm 读取第 *n* 条命令时，hand 可能已经读取第 *n+1* 条；
- generation 在发布或消费期间变化时，worker 会丢弃命令，但发布者难以判断是哪条命令失去所有权；
- home timeout 或旧调用者取消时，可能误伤已经覆盖它的新命令。

这些问题与早期 hand-home ACK 超时、stale-generation 丢弃和“上层已接受但执行器未准备”的日志一致。

## 4. 最终修复方案

### 4.1 控制热路径改为非阻塞 latest-wins

一个 `COUPLED_COMMAND_DTYPE` record 同时携带 arm/hand target、generation、action ID 和 delivery window。发布者在短 `motion_lock` 临界区内：

1. 复核当前 motion permit 与 candidate generation；
2. 写入完整 record；
3. 将 ring 返回的 sequence 标记为 active；
4. 立即返回，不等待 worker。

`(run_generation, ring_sequence)` 构成 ownership ticket。新 record 会覆盖旧 ticket；`action_id` 只用于审计和精确 ACK，不参与 ownership 判定。

### 4.2 motion lifecycle 与命令撤销原子化

`begin_motion`、`revoke_motion`、coupled publication 和定向取消共用 `motion_lock`。停止、FAULT、e-stop 或新一轮运动都会推进 generation 并清空 active ticket。worker 在校验前后及 SDK 调用前复核 ticket；已经被覆盖或撤销的 snapshot 既不能运动，也不能因其内容异常而错误锁存 fault。

该机制提供软件侧的命令 fencing，但锁不会跨越硬件 IO，因此不构成硬实时屏障或安全额定停止。

### 4.3 按命令源分配连续性策略

| 命令源 | 当前 arm 连续性策略 | 原因 |
| --- | --- | --- |
| 键盘遥操作 | 发布完整 IK endpoint；不启用通用 arm delta clip/reject | 保留目标位姿语义，交给 30 Hz 目标生成和 Mode 6 跟踪 |
| VR 遥操作 | producer 以“上一条命令 → 新 IK”执行 8°/tick joint shaping | VR 输入连续，整形属于 VR proposal owner；后续安全门不重复裁剪 |
| learned policy | safety gate 以“上一条已发布命令 → 新 endpoint”执行默认 8°/tick reject，绝不 clip | 模型动作是提案；越界意味着 endpoint 合同无效，不能静默改写 |
| arm worker | 对所有来源执行默认 20°/accepted-command 的 command-to-command 异常跳变兜底，触发即 fail closed | 只防 IPC/IK 分支异常，不承担正常控制速率整形；latest-wins 允许中间 endpoint 被覆盖，因此它不是严格的 per-tick 速度限制 |

8° 和 20° 的职责不同：前者是特定 producer 的连续性合同，后者是硬件边界的异常检测阈值。二者都不是 xArm 控制器声明的统一“每周期最大运动角”。

### 4.4 键盘目标只限制 EEF lead，不裁剪关节解

键盘仍使用 8 mm 平移步长和 0.03 rad 旋转步长。虚拟目标相对实测 EEF 的领先量限制为 5 帧，即平移范数最多 40 mm、旋转最多 0.15 rad；workspace 另保留 5 mm command margin。

该限制作用于 IK 之前的完整 EEF 目标，保持目标方向和位姿语义。IK 解成功后原样进入安全门，不再逐关节修改。

### 4.5 ACK 退出实时热路径

正常遥操作发布不等待 ACK。只有 home、校准等明确需要“SDK 已接受精确 endpoint”语义的低频流程才等待 `accepted_target_action_id`。等待期间若 ticket 被覆盖则立即返回 superseded；timeout/失败只在该 ticket 仍为当前 owner 时定向取消，不能撤销较新的命令。

ACK 只表示 worker/SDK 接受目标，不表示关节已经物理到位，也不表示 arm 与 hand 同时到位。

## 5. 修复后的数据流

```text
键盘/VR/策略 producer
  -> 生成完整 endpoint（各自拥有连续性策略）
  -> SafetyGate：合同、限位、workspace、collision、可选 command delta
  -> coupled record 非阻塞发布 + active ticket
  -> arm/hand worker 各自读取同一逻辑 record
  -> generation、时效、ticket、限位和异常跳变复核
  -> 各自 SDK
```

该设计保证 IPC record 一致和旧命令不可迟到执行，但两个 worker 仍独立调度，因此不承诺物理同步。

## 6. 对 VR 遥操作和策略推理的影响

- coupled record、atomic revoke 和 worker fencing 是共享基础设施修复，键盘、VR、回放和策略部署都受益；
- 删除的是键盘路径错误启用的通用 arm delta，不是全局删除安全检查；
- VR 仍保留 producer-side 8° command shaping，因此不会因键盘修复而失去原有平滑策略；
- learned-policy 仍以 8° command-to-command 规则 reject 整个 endpoint，并在 rejection 时终止/静默该 action chunk endpoint，而不是发布变形动作；
- 20° worker fallback 对所有来源生效，可阻止异常 IK 分支或损坏 record 直接跨越 SDK 边界；
- 非阻塞发布避免推理/coordinator 因等待 actuator ACK 而破坏调度，但 latest-wins 可能覆盖未消费的旧 endpoint，这是实时遥操作的明确取舍。

## 7. 验证状态

已完成且不连接硬件的验证：

- coupled record 的 shared-memory round trip；
- 新 record 覆盖旧 ticket；
- `RUNNING → ARMED/FAULT` 使旧 ticket 失效；
- timeout 取消不会撤销较新 ticket；
- superseded 的异常 snapshot 不会调用 arm SDK 或锁存 fault；
- keyboard 发布完整 IK solution，不执行 arm delta clip；
- learned-policy delta 使用上一条命令，而 workspace/collision 使用实测反馈；
- arm/hand 共用 generation 和 delivery-window 合同；
- 默认 worker discontinuity fallback 为 20°。

当前离线结果为 21 项回归测试通过，`compileall`、定向 `mypy`、Black、isort 和 `git diff --check` 通过。

尚未完成的物理闭环：

1. 受控低速环境下记录每周期 target、SDK accepted target、measured qpos 和 tracking error；
2. 验证长按平移、方向反转、腕部旋转、松键/再按和 workspace 边界；
3. 确认键盘日志不再出现 prepare rejection、arm delta rejection 或持续 60 ms loop overrun；
4. 分别验证 VR 8° shaping 和 learned-policy 8° rejection，不以键盘结果替代；
5. 验证 e-stop、worker 退出和 generation revoke 后没有后续 SDK command。

在这些实机检查完成前，应表述为“软件根因已修复并完成离线验证”，不能声称物理抖动已经完全关闭。

## 8. 复发诊断

若再次出现卡顿，按以下顺序分类：

1. `Control loop over budget`：先定位同周期是否存在等待、IK/collision 超时或阻塞 IO；
2. `GateRejectCode`：区分 joint limit、workspace、collision 和 policy command delta，不要统一归因于“步长”；
3. ticket/generation：确认 record 是否被新命令覆盖或被 motion revoke；
4. worker validation：区分 stale/expired（正常丢弃）和 fault-class contract violation；
5. SDK/硬件：只有软件 command 已被 ACK 且 target 连续时，才进一步调查控制器、网络、负载和机械因素。

不要通过删除 generation、freshness、collision、limit 或 final SDK-boundary checks 来消除日志；这会隐藏故障而不是修复故障。

## 9. 相关实现与测试

- [`teleop/keyboard_session.py`](../dexmani_real/teleop/keyboard_session.py)：键盘目标、EEF lead、IK 与运行状态。
- [`teleop/action_proposal.py`](../dexmani_real/teleop/action_proposal.py)：VR producer-side arm shaping。
- [`control/safety_gate.py`](../dexmani_real/control/safety_gate.py)：几何检查与可选 command-delta reject。
- [`control/publication.py`](../dexmani_real/control/publication.py)：candidate 校验、coupled publication 与 ACK。
- [`runtime/safety.py`](../dexmani_real/runtime/safety.py)：motion permit、generation、ticket 与撤销。
- [`robot/command_validation.py`](../dexmani_real/robot/command_validation.py)：worker 共用的时效、限位和异常跳变合同。
- [`robot/arm_worker.py`](../dexmani_real/robot/arm_worker.py)、[`robot/hand_worker.py`](../dexmani_real/robot/hand_worker.py)：最终 actuator 边界。
- [`tests/test_keyboard_arm_limits.py`](../tests/test_keyboard_arm_limits.py)、[`tests/test_safety_gate_command_delta.py`](../tests/test_safety_gate_command_delta.py)、[`tests/test_coupled_command_publication.py`](../tests/test_coupled_command_publication.py)、[`tests/test_worker_command_validation.py`](../tests/test_worker_command_validation.py)：离线回归合同。

## 10. 2026-08-26 松键回撤事件

### 10.1 症状与现场证据

在前述连续运动问题修复后，键盘遥操作仍出现另一种独立现象：按键期间运动连续，
但松键后机械臂会小幅反向回撤。发生回撤的版本在确认松键后额外发布一条
`release_stop` 关节目标，并等待该目标速度收敛后才执行 `RUNNING → ARMED`。

诊断日志证明这条停止目标本身就是不连续来源。例如：

```text
action_id=45  release_stop  cmd_step=7.18deg  endpoint_delta_to_measured=1.67deg
action_id=406 release_stop  cmd_step=8.14deg  endpoint_delta_to_measured=1.96deg
```

`endpoint_delta_to_measured` 较小只能说明停止目标靠近当时的实测关节位置；
它没有说明该目标靠近上一条已经发送给 Mode 6 的目标。运动时正常命令会领先实测状态，
因此把停止目标重建为 measured qpos 等价于用一个滞后目标覆盖最后运动目标，机械臂自然会
向后运动。日志中的 `release_stop` 命令步长显著大于正常连续命令，是回撤与该逻辑之间的
直接证据。

### 10.2 需要保持的停止语义

键盘松键不是 e-stop，也不需要构造新的制动轨迹。正确语义是：

1. 短暂确认输入确实已经松开；
2. 确认最后一条正常运动 action 已成功跨过 arm SDK 边界；
3. 不再发布任何新 endpoint；
4. 执行 `RUNNING → ARMED`，撤销继续发布运动命令的权限；
5. 让 xArm Mode 6 保持并完成最后一个已经接受的 endpoint。

因此，日志中的

```text
safety: revoked motion RUNNING(2) → ARMED(1)
```

是正常的软件撤权，不代表故障、急停或动作被回滚。只有 `FAULT`、e-stop、worker/SDK
错误等日志才表示异常停止。

### 10.3 最终修复

- 删除键盘松键路径生成和发布 `release_stop` 目标的逻辑，也不再等待该目标的 qvel
  收敛；
- 连续两个输入采样均为空才确认松键，桥接终端按键事件的瞬时空隙；
- 最多等待 0.15 s，直到 arm state 的 `last_cmd_seq` 追平最后正常 action ID；
- ACK 成功后直接撤权并保留最后 Mode 6 endpoint；ACK 超时则 fail closed 撤权并告警；
- arm worker 以 coupled-command ring sequence 去重，同一个 record 最多调用一次
  `servo()`；该 at-most-once 边界同时覆盖键盘、VR、回放和策略命令；
- 没有降低键盘目标步长、控制频率或机械臂速度。

修复后预期松键日志为：

```text
Keyboard release: final action_id=N accepted; leaving its Mode 6 endpoint unchanged
safety: revoked motion RUNNING(2) → ARMED(1)
```

不应再次出现：

```text
Keyboard release stop published ...
Keyboard release stop ... settled ...
```

### 10.4 真机复验结果

2026-08-26 的后续真机运行确认：

- 每次松键时 `latest=published/SDK-accepted` 最终均追平，最后 endpoint 没有被新目标覆盖；
- 共发布 422 条正常 arm endpoint，arm SDK 实际调用 420 次；两次
  `obs_gap_max=2` 表示 30 Hz producer/consumer 相位未同步时各合并了一个中间
  latest-wins endpoint，最终目标没有丢失；
- `duplicate_skips=179` 表示 worker 重读相同 ring sequence 时被去重，不是丢失
  179 条命令；
- 最大正常 command-to-command 步长为 2.90°，旧 `release_stop` 导致的 7–8°
  跳变消失；
- 最大 observed qvel 为 76.5°/s，IK 最慢约 3 ms，控制循环没有 overrun 或 missed
  slot；
- 操作者确认松键回撤已经消失，回零和 shutdown 均正常完成。

离线回归为 42 项测试通过，`compileall`、isort 和 `git diff --check` 通过；
真实运动效果以上述真机日志和操作者观察为准。

### 10.5 剩余观察项

这些项目不是本次松键回撤的根因，但后续应单独处理或持续监控：

1. 当前 collision 启动摘要为 `table=disabled`、`0 environment pairs`，只提供
   自碰撞和 workspace 保护；存在桌面或固定障碍物时必须确认这是有意配置；
2. 键盘与 arm worker 同为 30 Hz 时可能偶发合并一个中间 latest-wins endpoint。
   若物理上仍可感知，可在不改变运动速度的前提下评估更高的 worker 检查频率；
3. 当前 `cmd_step_max` 是 published-to-published 指标；发生 action gap 时，它不能直接
   表示相邻 SDK-accepted endpoint 的实际步长，必要时应在 arm worker 增加该指标；
4. arm home 的上层完成判断依据位置/速度收敛，而 worker 还负责 dwell 和恢复 Mode 6。
   更严格的合同应由 worker 在 Mode 6 恢复后发布明确的 home completion ID。

### 10.6 防止复发

- 不要在普通松键路径把 measured qpos 重新发布为“停止目标”；
- 不要用 `qvel == 0` 代替“最后正常 endpoint 已跨过 SDK 边界”；
- 比较停止行为时必须同时记录 previous commanded target、SDK-accepted target 和
  measured qpos，不能只比较新目标与 measured qpos；
- latest-wins 允许未消费的中间 endpoint 被覆盖，但必须保证最后 endpoint 被接受，
  且 worker 对每个 ring sequence 最多执行一次；
- 若需要真正的快速制动或紧急停止，应走显式 stop/e-stop 安全路径，不能复用普通松键
  语义。
