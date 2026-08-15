# XHand 瞬时 error_state 异常记录与成因分析（2026-08-15）

> 文档日期：2026-08-16<br>
> 仓库基线提交：`c298505`（"0815 temp 2345"）<br>
> 异常发生：2026-08-15 23:42:58（`collect_teleop.py` 遥操作采集会话）<br>
> 分析对象：会话日志（`dexmani_collect_922700`）、录制 episode `episode_20260815_234248`（532 帧）、`dexmani_real/robot/{xhand,hand_process}.py`、本机 SDK 源码 `/usr/local/xhand_controller/`<br>
> 分析方式：录制轨迹帧级回放 + 代码静态追踪 + SDK C++ 源码核查 + 多智能体对抗式 fact-check<br>
> 安全声明：本文仅为离线事后分析，未打开 XHand 端口、未向真实硬件发送命令

---

## 目录

1. [结论摘要](#1-结论摘要)
2. [异常现象（原始日志）](#2-异常现象原始日志)
3. [事实还原（日志 ↔ 录制帧级对齐）](#3-事实还原日志--录制帧级对齐)
4. [根因分析](#4-根因分析)
5. [fact-check 裁决](#5-fact-check-裁决)
6. [关键机制与 SDK 源码证据](#6-关键机制与-sdk-源码证据)
7. [待真机确认项](#7-待真机确认项)
8. [建议](#8-建议)

---

# 1. 结论摘要

XHand **没有断连**。23:42:58 发生的是**一次约 0.1s 的瞬时 board 级故障**：某个板错误寄存器
（`commboard_err` / `jointboard_err` / `tipboard_err` 之一）置位 → `error_state=True` →
teleop 进入 hand_feedback 静默暂停 → 故障自愈 → 恢复。

**直接触发与"拇指堵转抽大电流"强相关，是最可能的原因**，但**"是关节板 + 是过流保护"这层因果
无法从现有 episode 数据或 SDK 源码证实**：

- **已证实**：未断连；拇指堵转（命令 57°→77°、实测卡 42°）；J0 峰值电流 -1543mA；
  故障瞬时自愈；EtherCAT 线程缺 RT 优先级（EPERM）。
- **可信但未证实**：过流触发故障（相关性最强）。
- **已撤回（过度断言）**：`过流 → jointboard_err` 的因果映射；`tor_max=300mA 是检测阈值而非钳位`；
  `500ms` 时序匹配。

---

# 2. 异常现象（原始日志）

会话关键行（`dexmani_collect_922700`，23:42:16 → 23:43:36）：

```text
[23:42:16] vr_loop: connected to HTS port=8000
EcatUpdateThread: Operation not permitted                      ← EtherCAT 周期线程 RT 调度失败（EPERM）
[23:42:17] XHand ready: SDK=1.4.6 ...
[23:42:46] safety: DISARMED(0) → ARMED(1)
[23:42:48] safety: ARMED(1) → RUNNING(2)
[23:42:49] Control loop over budget: actual=763.7ms target=62.5ms   ← 首个完整周期一次性开销
[23:42:58] hand_loop: hand error_state — clear_local_error() (1/5 consecutive)
[23:42:58] Hand feedback unhealthy — pausing motion: hand reported a hardware error
[23:42:58] teleop_loop: entered hand_feedback command quiescence (run=3)
[23:42:58] Hand feedback recovered after 0.1s — waiting for fresh re-anchor
[23:43:22] C: 暂停遥操作
[23:43:25] H: return_home
[23:43:27] 录制已保存: .../episode_20260815_234248 (532 帧)
[23:43:33] XHand: EtherCAT slave position is unknown; skipping explicit INIT request and using close/watchdog   ← 退出清理期，非会话中故障
[23:43:36] shutdown: ... hand=graceful:0 ... supervisor_normal=True
```

用户感知的"突然断连"，对应日志里 `Hand feedback unhealthy — pausing motion` 这一安全暂停。

---

# 3. 事实还原（日志 ↔ 录制帧级对齐）

录制帧率 16 Hz（帧间隔名义 62.5ms）。帧 150 处的 0.167s source 时间缺口是全场唯一 >100ms 的 gap，
对应 23:42:58 的故障+暂停+重锚点。

| 时刻 | 帧 | 事实 |
|---|---|---|
| 23:42:48 | 0 | B 开始遥操作+录制 |
| 23:42:49 | ~16 | 首次 `Control loop over budget: 763.7ms`（首帧 NLopt/IK 一次性开销） |
| 23:42:5x | 100 | J0 电流 +346mA（一次正方向 <500ms 抖动，与本次故障无关） |
| 23:42:5x | 138 | J0 电流 -470mA（本次堵转事件首次超过 300mA） |
| 23:42:5x | 142–149 | J0 电流 -1351→**-1543mA** 峰值；J4 -1053、J6 -1121、J3 -811、J7 -731、J8 -858；触觉 4/5 指尖接触 |
| 23:42:58 | **150** | **0.167s 缺口** = 故障 + 暂停 + 重锚点 |
| 23:42:58 | 151 | 命令 J0 从 66.9° 重锚到 43.8°（卸载堵转拇指）→ 电流骤降 |
| 23:42:58 | 153 | J0 电流 -8mA（电机驱动被切断后恢复） |

**堵转证据**（帧 140–150）：

| 量 | 数值 |
|---|---|
| `action_hand_joint[:,0]` | 57.48° → 峰值 76.86°（帧 144） |
| `hand_qpos[:,0]` | 卡在 ~42°（41.73°→42.65°） |
| command − measured（度） | 15.75° → 34.29° |
| `hand_tactile_contact`（帧 142–150） | `[1,1,1,1,0]`（四指接触，抓物中） |

**电流证据**（J0，mA）：`137=-298, 138=-470, 139=-578, 140=-652, 141=-707, 142=-1351,
143=-1541, 144=-1543, 145=-1543, 146=-1541, 147=-1536, 148=-1536, 149=-1530, 150=-1515,
151=-1469, 152=-1383, 153=-8`。

**重要**：录制中 `hand_error_state` 全 False、三个 `*board_err` 寄存器全 0、`hand_connected`
全 True。原因：故障锁存极短（<1 个 30Hz hand-loop tick、<62.5ms），未落到 16Hz 采样网格上，
且正好落在暂停重锚点（帧 150 的 gap）里——**因此离线轨迹无法判断是哪块板、哪个 bit 报错**。

---

# 4. 根因分析

## 4.1 已证实的事实链

1. **未断连**：日志走的是 `error_state` 分支（`hand_process.py:434`，`get_state()` 成功、总线在线），
   不是 `get_state failed` 断连分支（`:408`）。
2. **error_state 来源**：`xhand.py:911-915` ——
   `self.error_state = bool(np.any(commboard_err) or np.any(jointboard_err) or np.any(tipboard_err))`，
   即板错误寄存器的 OR，与连接标志无关。
3. **自愈**：`clear_local_error()`（`xhand.py:867`）只清本地锁存、不发硬件调用；日志
   "recovered after 0.1s" 说明**固件侧**过流/故障解除、寄存器自动归零。
4. **拇指堵转 + 高电流**：见第 3 节数据。

## 4.2 最可能（但未证实）的触发

拇指被抓握物体挡住、命令持续弯曲（57°→77°）而实测卡在 42°，电机堵转、电流飙到 -1543mA，
随后（约 0.1s 内）板错误寄存器置位。过流是相关性最强的候选触发源。

## 4.3 已撤回的过度断言

以下三点在本轮 fact-check 中被证伪或判定为不可证实，**不应作为结论引用**：

1. **`过流保护 JOINT_ERROR_CURRENT_PROCTED → jointboard_err` 的因果映射**：SDK 源码显示这是两条
   互不相通的通道（详见第 6 节）。只能确定"某个板错误寄存器置位"，无法确定是关节板、还是过流。
2. **`tor_max=300mA 是检测阈值而非电流钳位`**：host 确实不做钳位，但"firmware 也不钳位"无法从
   host 源码证明；厂商文档反而把 `tor_max` 标为"力矩上限"、作为力控模式替代（电流上限）。
3. **`~500ms` 时序匹配**：精确时长见下，>300mA→故障为 766.5ms、>1000mA→故障为 499.96ms；
   固件过流阈值并未证明等于 300mA。

---

# 5. fact-check 裁决

6 个独立 skeptic 的对抗式证伪结果：

| 主张 | 裁决 |
|---|---|
| XHand 未断连，是 ~0.1s 安全暂停 | ✅ CONFIRMED |
| 拇指堵转（命令 57°→77°、实测卡 42°、4 指接触） | ✅ CONFIRMED |
| 过流 1543mA 持续 ~500ms 匹配固件 "exceeding 500ms" | ⚠️ PLAUSIBLE（峰值对，时序/阈值自相矛盾） |
| 过流 `JOINT_ERROR_CURRENT_PROCTED` → `jointboard_err` → 暂停 | ❌ UNVERIFIABLE（因果第一跳无源码证据） |
| `tor_max=300mA` 是阈值不是钳位 | ❌ UNVERIFIABLE（厂商文档反指"上限"） |
| `EcatUpdateThread: Operation not permitted` = 缺 RT 权限 | ✅ CONFIRMED |

**时序精确值**（按 source_monotonic_ns 差值，非名义 62.5ms）：

| 事件 | 至故障帧 150 的时长 |
|---|---|
| 首次 \|J0 电流\|>300mA（帧 138） | 766.5ms |
| 首次 \|J0 电流\|>1000mA（帧 142） | 499.96ms |
| 帧 142 → 151（gap 结束） | 666.7ms |

仅 >1000mA 锚点 ≈500ms；若固件阈值真是 300mA，500ms 去抖应在帧 145-146 触发、比实际早 ~266ms。
数据更吻合一个 **~1000–1350mA 的固件阈值**（3.3–4.5× `tor_max`），但该阈值值在本机不可得。

---

# 6. 关键机制与 SDK 源码证据

## 6.1 EtherCAT 周期线程缺 RT 优先级

`/usr/local/xhand_controller/src/ethercat_communication.cpp:154-158`：

```cpp
struct sched_param param;
param.__sched_priority = 95;
if (pthread_setschedparam(ethercat_update_thread_->native_handle(),
                          SCHED_FIFO, &param)) {
    perror("EcatUpdateThread");   // 输出 "EcatUpdateThread: Operation not permitted"
}
```

`SCHED_FIFO` 需要 `root`/`CAP_SYS_NICE` 或 `RLIMIT_RTPRIO>0`；缺失时内核返回 EPERM。周期循环
（`:722,730-731`）用 `clock_nanosleep(TIMER_ABSTIME)` 1ms 网格 + `ec_receive_processdata`。
被抢占导致漏周期时，真正暴露的通信错误是 **`Unexpected wkc` → `PDO_COMMUNICATION_ERROR`**
（`:266-270,288-292`），**不是** `commboard_err`——后者是 WKC 完整时才读到的从站错误寄存器（`:747`）。

## 6.2 `jointboard_err` 是不透明 PDO 位域

`/usr/local/xhand_controller/include/data_type.hpp:82-84`：`commboard_err`/`jonitboard_err`/`tipboard_err`
均为 16-bit 位域，注释仅 `Subindex5-7 / res2-res4`。SDK 内 `ethercat_communication.cpp:746-751`、
`serial_communication.cpp:721-726` 直接透传，**全库无任何代码解析其 bit**——位含义只存在于手部固件。

## 6.3 过流错误码是独立通道

`JOINT_ERROR_CURRENT_PROCTED`(=1501035) 在 `error_manager.cpp:42,124-125` 的宿主侧 ErrorManager 表里，
消息为 `"Current exceeds the set threshold, triggering overcurrent warning (exceeding 500ms)"`。
该表唯一消费者 `get_error_info()` 只由两条非 PDO 通道填充（`ethercat_communication.cpp:949-960`）：

- IAP 固件升级 ackResult；
- `CMD_ERROR` 帧的 2 字节 code（`case CMD_ERROR: error_code = frame_data[0] + (frame_data[1]<<8)`）。

**从不来自 `jointboard_err` 寄存器**。即：过流可能走 `CMD_ERROR` 通道上报，而非 `jointboard_err` PDO 位域。

## 6.4 `tor_max` 语义

`ethercat_communication.cpp:788`：`pdo_output->tor_max = command.finger_command[...].tor_max;` —— 仅透传。
host 侧无任何钳位逻辑（`grep min/max/clamp` 仅此一处）。但 `data_type.hpp:134` 字段注释为 **"力矩上限"**，
厂商 README 将 `tor_max`（mA）描述为力控模式的替代 = 电流/力矩上限。钳位还是阈值，取决于 MCU 固件，
不在本仓库可证实范围内。

---

# 7. 待真机确认项

1. **哪块板、哪个 bit 报错**：本次故障帧短于采样网格、未录下 `*board_err` 寄存器值。需在 hand worker
   内做 30Hz 逐 tick 极值/历史缓冲后复现。
2. **`tor_max=300mA` 是否真正限流**：堵转为何能到 1543mA（>5×）——是未生效、钳位漏电、还是该模式下
   钳位不启用，需真机或固件文档确认。
3. **固件过流阈值**：`"exceeding 500ms"` 的"设定阈值"具体数值（~1000mA？），本机不可得。
4. **过流上报通道**：过流究竟是 `jointboard_err` PDO 位域，还是 `CMD_ERROR` 帧（第 6.3 节两条通道）。

---

# 8. 建议

1. **诊断增强（优先）**：在 `hand_process.py` 读态循环内，对 `commboard_err`/`jointboard_err`/
   `tipboard_err` 做逐 tick 极值/滑动历史缓冲（30Hz），而非只靠 16Hz 采样网格——否则离线永远无法
   定位"哪块板、哪个 bit"。
2. **tor_max 语义核查**：不要基于"阈值非钳位"的假设去调 `tor_max`。先真机确认它作为电流上限是否
   生效、堵转为何超限，再决定调值或改限流策略。
3. **环境加固**：给 Python 解释器加 `cap_sys_nice`（并复核 `cap_net_raw`），消除
   `EcatUpdateThread: Operation not permitted`，让 EtherCAT 周期线程获得 RT 优先级。
4. **通信故障口径修正**：后续若出现通信类故障，先查 `Unexpected wkc` / `PDO_COMMUNICATION_ERROR`，
   而非默认归因到 `commboard_err`。

---

## 附：本次结论的证据规则

- 判定为 CONFIRMED 的，均有多条独立证据（录制数据 + 代码行 + SDK 源码）交叉印证。
- 判定为 PLAUSIBLE / UNVERIFIABLE 的，明确标注证据缺口，不把"相关性"表述为"因果"。
- 已撤回的过度断言单独列出，避免后续误引用。
