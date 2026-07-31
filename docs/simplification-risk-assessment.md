# DexMani 简化改进 — 逐项风险评估

**日期:** 2026-07-31  
**基于:** `docs/dexmani-vs-lerobot-ufactory-analysis.md`

---

## 快速总览

| 改进 | 风险等级 | 减行数 | 是否建议执行 | 前置条件 |
|------|---------|--------|------------|---------|
| P0 删除 arm 进程隔离 | **中** | -1000 | ✅ **建议执行** | 需 7 个入口点改造 |
| P1a 删除 delta clamp 代码 | **极低** | -50 | ✅ 立即执行 | 无（已禁用） |
| P1b 删除固件版本检查 | **极低** | -25 | ✅ 立即执行 | 无 |
| P1c 简化可恢复错误处理 | **中高** | -80 | ⚠️ 需审慎 | 需真机验证 3 种错误码 |
| P1d 简化 hold_position | **中** | -30 | ⚠️ 需验证 | 需确认 Mode 6 固件行为 |
| P1e 移除宏命令 RPC | **高** | -200 | ❌ 暂不建议 | robot_rpc.py 依赖 |
| P1f 简化模式漂移恢复 | **低** | -20 | ✅ 建议执行 | 保留核心检查 |
| P2 首帧 wait=True 同步 | **中低** | +10/-30 | ✅ 建议执行 | 需改 startup 流程 |
| P3 手 SHM seqlock→Lock | **中高** | -200 | ❌ 暂不建议 | 需先消除手进程隔离 |
| P4 AsyncEpisodeSaver | **极低** | +70 | ✅ 建议执行 | 无 |
| P5 提取 continuous_rotvec | **极低** | +0 | ✅ 建议执行 | 无 |
| P6 contextvars 替代全局变量 | **低** | +20 | ✅ 可选 | 无 |

---

## P0: 删除 Arm 进程隔离

**风险等级: 中**  
**影响范围: 1000 行删除，7 个入口点需修改**

### 当前架构

```
Entry Point
  └─ make_arm_servo(cfg)
       └─ ArmProcessConfig → ArmSHMFaçade → ArmControlProcess
            └─ fork() → _arm_child_main() → ArmInnerLoop._run()
                                                 └─ XArmAPI(ip)
       └─ ArmInnerLoopSHMAdapter(facade, estop_event)  ← 返回给入口点
```

`make_arm_servo()` 始终走 **子进程** 路径——7 个入口点全部通过 `make_arm_servo()` 创建 arm servo。

### 失败模式分析

**F1: 主线程崩溃导致 Mode 6 断连，臂掉落**

- **当前保护:** 子进程中 XArmAPI 独立存活。主进程崩溃 → 子进程变孤儿但继续运行 → Mode 6 固件持续收命令 → 不掉臂。
- **简化后:** 主线程 = SDK 连接线程。主线程崩溃 → TCP 断开 → Mode 6 固件 **自动保持最后位置**（这是 Mode 6 的设计行为）。
- **风险判断:** **无新增风险**。Mode 6 的断连保持是固件级保证，lerobot_ufactory 已在数百小时运行中验证。实际上当前架构有一个更微妙的故障模式：子进程崩溃时主进程无从得知（daemon=True），臂在无命令状态下依赖 Mode 6 固件保持——与简化后的行为完全相同。

**F2: SDK 调用阻塞主线程，控制回路卡顿**

- **当前保护:** SDK 调用在子进程，主线程仅通过 SHM 通信，不受 SDK 阻塞影响。
- **简化后:** SDK 调用在主线程（`get_joint_states()` 和 `set_servo_angle()` 都是同步调用）。
- **风险判断:** **低风险**。`get_joint_states()` 延迟 ~1-3ms，`set_servo_angle(wait=False)` 延迟 ~1ms。在 16Hz（62.5ms 周期）下，3ms 的 SDK 调用占 5% 周期，完全可接受。lerobot_ufactory 在 60Hz（16.7ms 周期）下运行无误。
- **唯一例外:** SDK 偶尔在异常状态下阻塞更久（~50ms+）。当前子进程架构下这不会影响主线程，简化后会影响。但这种情况只发生在 arm 硬件异常时（此时无论如何都应该 estop），且 lerobot_ufactory 从未报告过此问题。

**F3: estop 事件跨进程传递失效**

- **当前保护:** `multiprocessing.Event` 跨进程传递，子进程 ≤1 tick 内执行 `set_state(4)`。
- **简化后:** `threading.Event` 同进程传递，内环线程 ≤1 tick 内执行 `set_state(4)`。
- **风险判断:** **无新增风险**。线程间 Event 比进程间 Event 更快、更可靠（无序列化开销）。

**F4: 入口点改造引入 bug**

- **7 个入口点** 全部需要将 `arm_inner = make_arm_servo(cfg)` 改为 `arm_inner = ArmInnerLoop(ip, cfg)`，并删除 `do_return_home()` 中的进程重启逻辑。
- **风险判断:** **中风险**——改造量大但逻辑简单（删除包装层，直接实例化）。每个入口点的改造是机械性的。
- **缓解:** 改造后跑一轮冒烟测试（连接 → teleop 5 秒 → estop → disconnect），7 个入口点逐一验收。

### 额外收益

除了减少 1000 行代码，还消除了以下已在注释中记录的故障模式：
- **启动竞争:** `wait_ready()` 返回后 SHM 可能无帧 → `_startup_grace_remaining` 宽限期（arm_process.py:436,481,500-506）
- **撕裂读回退:** torn read → `last_good` 缓存 → 可能返回旧数据（arm_process.py:531-535）
- **错误状态捏造:** stale → `_fabricate_error_state` → validate_action 误触发 estop（arm_process.py:515-528）
- **重启竞争:** `ensure_running()` 中 ring 重建 + 等待（arm_process.py:450-482）

这些都是 **进程隔离自身引入的问题**，消除进程隔离即消除这些 bug。

### 建议

✅ **执行。** 这是收益/风险比最高的改动。先在一个入口点（如 `keyboard_teleop_real.py`）试点，真机验证一个 session 后推广到其他 6 个入口点。

---

## P1a: 删除 delta clamp 代码

**风险等级: 极低**  
**影响范围: inner_loop.py ~50 行**

### 当前状态

```python
# inner_loop.py:76
max_joint_delta: float = 0.0  # Disabled
```

配置中已设为 0.0（禁用），且注释解释："The inner-loop clamp was a second, looser backstop that never fired in normal teleop"。

代码路径：
- `_send_target()` 中不再执行 per-step delta clamp
- `_last_sent_target` 仍被更新（用于 tracking error 计算和首次目标种子）

### 风险分析

删除以下内容完全安全：
- `max_joint_delta` 配置字段
- `_last_sent_target` 的 delta clamp 用法（保留其 tracking error 用途）
- 相关注释和文档

**唯一的残余依赖:** `_last_sent_target` 在 `_monitor()` 中用于计算 tracking error（line 628-629），在 `_bootstrap_state()` 中用于种子首次目标（line 398）。这两个用途**与 delta clamp 无关**，需保留。

### 建议

✅ **立即执行。** 纯代码清理，零行为变更。

---

## P1b: 删除固件版本检查

**风险等级: 极低**  
**影响范围: inner_loop.py ~25 行（`_init_mode` 中的 `get_version` 块）**

### 当前状态

```python
# inner_loop.py:830-847
code, ver_str = arm.get_version()
# Parse "v1.18.4" → major, minor → check >= 1.10.0
```

Mode 6 需要固件 >= 1.10.0，该版本于 **2020 年** 发布。当前使用的 xArm7 固件版本为 v1.18.x 或更高。

### 风险分析

- 任何能运行 Mode 6 的 xArm7 控制器固件都 >= 1.10.0
- 即使固件低于 1.10.0，`set_mode(6)` 会直接失败（SDK 返回错误码），比版本字符串解析更可靠
- lerobot_ufactory 不做此检查

### 建议

✅ **立即执行。** 固件能力由 SDK 调用结果检验，不由版本字符串检验。

---

## P1c: 简化可恢复错误处理

**风险等级: 中高**  
**影响范围: inner_loop.py ~80 行**

### 当前状态

三种可恢复错误码各有独立的恢复逻辑：

| 错误码 | 含义 | 当前处理 |
|--------|------|---------|
| 22 | Self-Collision | `_recover_mode()` → clean_error + clean_warn + set_state(0) + set_mode(6) |
| 24 | Speed Exceeds Limit | 同上 |
| 31 | C31 Collision Current | 同上 |

`_handle_arm_error()` 和 `_send_target()` 中都有错误分类逻辑，部分重复。

### 失败模式分析

**F1: 简化后无法区分可恢复 vs 致命错误**

- **最坏情况:** 将致命错误（如 error 1 Joint Overcurrent）误分类为可恢复 → 内环反复尝试 recover → 可能损坏硬件。
- **缓解:** 保留白名单机制（`_RECOVERABLE_ERRORS`），但将恢复逻辑简化为统一的 2 步（clean_error + reinit_mode），去掉 `_handle_arm_error` 和 `_send_target` 中的重复分类。
- **风险:** 中等。错误码分类逻辑是正确的，不应改动；但恢复路径可以统一。

**F2: 恢复失败循环**

- **当前保护:** 恢复失败 → `_error_state = True` → 内环停止。
- **简化后:** 需保留此保护。
- **风险:** 无新增风险（保留失败即停逻辑）。

### 具体简化方案

将分散在 `_handle_arm_error`（line 416-445）和 `_send_target`（line 682-735）中的错误处理统一为：

```python
# 在 _run() 主循环中，send_target 之后
if code != 0:
    err_code = _get_controller_error(arm)
    if err_code in _RECOVERABLE_ERRORS:
        logger.warning("Recoverable error %d, re-initializing Mode 6", err_code)
        self._recover_mode(arm)
        self._arm_target = None  # drop current target
        continue
    else:
        logger.error("Fatal error %d, stopping inner loop", err_code)
        self._error_state = True
        break
```

删除 `_handle_arm_error` 方法（line 416-445），将逻辑内联到 `_run()` 中。

### 建议

⚠️ **谨慎执行。** 在真机上故意触发 3 种可恢复错误（肘部自碰撞、快速大范围移动触发 overspeed、轻推臂触发 C31），验证简化后的恢复逻辑是否正确。建议在非采集 session 中测试。

---

## P1d: 简化 hold_position

**风险等级: 中**  
**影响范围: inner_loop.py ~30 行**

### 当前状态

```python
def _hold_position(self, arm):
    code, states = arm.get_joint_states(is_radian=True, num=3)
    hold = states[0][:7]
    arm.set_servo_angle(angle=hold.tolist(), speed=..., wait=False)
    self._last_sent_target = hold.copy()  # for tracking error baseline
    self._last_sent_cmd = hold.copy()    # for "sent" stream
```

Hold 时读取当前关节角，重新发送为命令。lerobot_ufactory 的等价物：**不发送任何命令**（Mode 6 固件自动保持最后收到的目标位置）。

### 失败模式分析

**F1: Mode 6 固件在没有新命令时是否真的保持位置？**

- **固件行为:** Mode 6 的 trajectory planner 持续追踪最后收到的目标。如果不再收到新命令，planner 完成当前 trajectory 后将臂保持在最后目标位置。这是设计行为。
- **但:** 如果最后目标是一个 moving target（例如 `set_servo_angle` 发了一个正在运动中的目标），planner 会在目标到达后停止。而 `_hold_position` 的做法是读取**实际**位置并重新发送，确保 "hold current position" 而非 "hold last target"。
- **差距:** 在正常情况下（tracking error < 2°），"last target" ≈ "current position"，两者等效。但在高 tracking error 时（如刚经历了一次大幅运动），last target 可能与 current position 差 5-10°。
- **风险评估:** **低概率，低影响**。hold 通常发生在遥操作暂停或目标超时时，此时臂已经静止，tracking error 接近 0。只有在异常情况下（运动中突然 hold）才有差异，而此时的正确行为本身就是模糊的（是该停在目标位置还是当前位置？）。

**F2: 去掉 hold_position 后动力学会在 hold 期间不更新**

- **当前:** `_hold_position` 同时更新 qvel/tau（line 805-815），确保 hold 期间 validate_action 的力矩门有数据。
- **简化后:** hold 期间不调用 `get_joint_states()`，qvel/tau 停留在最后值。
- **缓解:** 在主循环的 `_read_and_update_state()` 中保留状态读取，hold 期间仍可读取（但当前 hold 分支用 `continue` 跳过了 `_read_and_update_state`）。
- **风险:** **低**。力矩门已禁用，速度 NaN 检查不需要实时数据。

### 建议

⚠️ **可执行，需验证。** 将 hold 行为从 "读当前角度重发" 改为 "不发命令"，真机验证 hold 期间臂是否静止。如果验证通过，删除 `_hold_position` 方法。

---

## P1e: 移除宏命令 RPC 系统

**风险等级: 高**  
**影响范围: inner_loop.py ~200 行 + robot_rpc.py**

### 当前状态

`exec_macro()` 支持 6 种命令码：
- `CLEAR_ERROR` — 模式安全（内环运行中可用）
- `EMERGENCY_STOP` — set_state(4)
- `EXEC_WAYPOINTS` — Mode 1 路径执行
- `RESET_BLOCKING` — Mode 0 阻塞复位
- `REINIT_MODE6` — estop 后重建 Mode 6

调用路径：`robot_rpc.py` → SHM command ring → 某处读取并调用 `exec_macro()`。

### 失败模式分析

**F1: RPC 功能有实际用户**

- `robot_rpc.py` 被 hand_process.py 等引用
- 如果 RPC 是 hand 归位/恢复的关键路径，删除会导致手操作断裂
- **风险: 高**。需先确认 RPC 的实际使用场景。

**F2: REINIT_MODE6 是 estop 后唯一恢复路径**

- estop → `set_state(4)` → arm 停在 state 4 → 需要 `REINIT_MODE6` 重新使能 motion + 切 Mode 6
- 如果删除 RPC，需要替代的恢复路径
- **缓解:** lerobot_ufactory 的恢复只需 `clean_error() + motion_enable() + set_mode(6) + set_state(0)`，4 行。不需要 RPC 框架。

### 建议

❌ **暂不执行。** RPC 系统与其他模块（hand、robot_rpc）耦合较深，在未充分理解调用关系前不应删除。替代方案：先简化 `exec_macro` 的实现（减少重复的状态管理代码），保留接口不变。

---

## P1f: 简化模式漂移恢复

**风险等级: 低**  
**影响范围: inner_loop.py ~20 行**

### 当前状态

`_monitor()` 中检查 `arm.mode != 6` → `_recover_mode()`。`_recover_mode()` 有 7 步状态转换（line 742-782）。

### 简化方案

保留模式检查，但将 `_recover_mode` 简化为 3 行：
```python
if arm.mode != 6:
    arm.clean_error(); arm.set_state(0)
    arm.set_mode(6); arm.set_state(0)
```

去掉注释中提到的 3 次重试循环（已简化为单次尝试），去掉冗余的 `set_mode(0)` 中间状态。

### 风险

**低。** 模式漂移是偶发事件（通常由碰撞恢复触发），简化后的恢复路径与 lerobot_ufactory 的 `configure()` 完全一致。

### 建议

✅ **执行。**

---

## P2: 首帧 wait=True 同步

**风险等级: 中低**  
**影响范围: 入口点 startup 流程**

### 当前 DexMani 方案

`_bootstrap_state()` 读取当前关节角 → 种子 `_last_sent_target` → 第一个目标与当前角度接近 → 无跳跃。

### lerobot_ufactory 方案

首帧 `wait=True`（切 mode 0 阻塞）→ 臂运动到目标 → 后续 `wait=False`（切 mode 6 非阻塞）。

### 失败模式分析

**F1: 首帧目标远离当前位置，wait=True 长时间阻塞**

- **场景:** 上一 session 异常退出，臂未归位。下一 session 启动时，start_joints 可能远离当前位置（如臂在桌面附近，start_joints 在直立位置）。
- **当前保护:** `_bootstrap_state` 种子后，第一个目标（home_qpos）通过内环的 speed 限制逐步到达，不阻塞。
- **wait=True 方案:** 臂以 Mode 0 的速度运动到目标，**阻塞主线程直到到达**。如果距离远，可能耗时 5-10 秒，期间无法响应 estop（阻塞在 SDK 调用中）。
- **风险:** **中等**。但可通过以下方式缓解：
  - 先用非阻塞 `set_servo_angle(home_qpos, speed=slow, wait=False)` 让 Mode 6 慢慢走
  - 或保留当前的 `_bootstrap_state` 种子方案作为首选的慢速归位

**F2: Mode 切换瞬态**

- lerobot_ufactory 的 `send_action` 中已有 mode-switch-guard（line 359-366）：`wait_==False and mode != 6` 时才 `set_mode(6)`。但这个检查**每次 send_action 都执行**，有 `time.sleep(0.1)` 开销。
- **风险:** **低**。Mode 切换是成熟操作。

### 建议

⚠️ **不采用纯 wait=True 方案。** 保留当前的 `_bootstrap_state` 种子作为首帧慢速归位机制，它的行为更好（非阻塞 + 限速）。但可以简化种子的设置方式：直接从 SDK 读取而非通过 bootstrap_state 的多步流程。

---

## P3: 手 SHM seqlock → Lock

**风险等级: 中高**  
**前置条件: 手进程隔离消除**

### 当前状态

手进程隔离（hand_process.py 1135 行）存在的原因是 **XHand SDK 不够稳定**——手子进程可能崩溃（board error、串口超时等），需要独立进程隔离故障。

### 风险分析

**F1: 手 SDK 崩溃影响主线程**

- **当前保护:** 手在独立进程，崩溃 → SHM 检测到 disconnected → 自动恢复。
- **简化后:** 手在主线程 → SDK 崩溃可能抛异常 → 整个控制回路中断。
- **风险:** **高**。XHand SDK 的不稳定性是有历史记录的（多个 memory 文件记录了手 board error、commboard error 等）。

**结论:** 在 XHand SDK 稳定性得到根本改善之前，**不应消除手进程隔离**。因此 seqlock SHM 的需求依然存在。

### 简化替代方案

在保留手进程隔离的前提下：
- 手的 SHM 通信可以简化为 **单生产者单消费者** 的简单 ring buffer（不需要 seqlock odd/even 协议——因为手只有一个 state writer 和一个 target reader）
- 但 arm SHM 如果已消除，总的 SHM 基础设施复杂度已经大幅下降

### 建议

❌ **暂不执行。** 等 XHand SDK 稳定性改善后再考虑。在此之前，仅消除 arm SHM 已足够。

---

## P4: AsyncEpisodeSaver

**风险等级: 极低**  
**影响范围: 纯新增代码**

### 失败模式分析

**F1: 后台线程写 HDF5 失败，episode 静默丢失**

- **lerobot_ufactory 的保护:** `_raise_if_failed()` 在下次 submit 或 wait_idle 时检查 `_exception` 并抛出（`uf_lerobot_record.py:187-189`）。
- **缓解:** 移植此模式。
- **风险:** **极低**。这是一个纯加法，不影响现有录制路径。可以先作为可选功能（`--async_save` flag）上线。

**F2: 主线程在 saver 完成前退出，episode 数据不完整**

- **lerobot_ufactory 的保护:** `close()` 方法先 `queue.join()` 等待所有任务完成，再发 `_STOP` 哨兵，再 `queue.join()` 确保哨兵被处理（line 143-148）。
- **缓解:** 在 `disconnect()` 前调用 `saver.close()`。
- **风险:** **极低**。

### 建议

✅ **立即执行。** 这是最安全的改进，无行为变更，纯增量。

---

## P5: 提取 continuous_rotvec 到 pose_utils

**风险等级: 极低**  

### 风险分析

无。这是纯代码搬运，不改变行为。当前 `continuous_rotvec` 在 `uf_lerobot_eval.py:44-56` 中定义，DexMani 已有等价逻辑散落在各入口点中。

### 建议

✅ **立即执行。**

---

## P6: contextvars 替代全局变量

**风险等级: 低**  

### 风险分析

**F1: contextvars 在多线程下的默认值共享**

- lerobot_ufactory 的 `context.py` 中 `default={}` 在所有线程间共享同一个 dict → 并发写可能丢失条目。
- **缓解:** 用 `ContextVar` 的默认工厂函数或在 `connect()` 时 `set()` 新 dict。
- **风险:** **低**。DexMani 的控制回路是单线程的（除了内环），不存在并发写 contextvars 的场景。

### 建议

✅ **可选执行。** 适合在重构 teleop 注册逻辑时一并引入，不急。

---

## 总结: 建议执行路线图

### 第一轮（本周，低风险快速收益）

| 步骤 | 改动 | 风险 |
|------|------|------|
| 1 | 删除 delta clamp 代码（P1a） | 极低 |
| 2 | 删除固件版本检查（P1b） | 极低 |
| 3 | 添加 AsyncEpisodeSaver（P4） | 极低 |
| 4 | 提取 continuous_rotvec（P5） | 极低 |
| 5 | 简化模式漂移恢复（P1f） | 低 |
| **合计** | **-95 / +70 行** | |

### 第二轮（下周，需真机验证）

| 步骤 | 改动 | 风险 |
|------|------|------|
| 6 | 简化 hold_position（P1d） | 中 |
| 7 | 简化首帧同步（P2，保留 bootstrap_state 种子） | 中低 |
| **合计** | **-60 行** | |

### 第三轮（下下周，需逐个入口点验收）

| 步骤 | 改动 | 风险 |
|------|------|------|
| 8 | 试点消除 arm 进程隔离（P0，1 个入口点） | 中 |
| 9 | 推广到全部 7 个入口点 | 中 |
| 10 | 简化可恢复错误处理（P1c） | 中高 |
| **合计** | **-1080 行** | |

### 待定（需更多分析）

| 步骤 | 改动 | 风险 |
|------|------|------|
| ? | 移除宏命令 RPC（P1e） | 高 |
| ? | 手 SHM 简化（P3） | 中高 |

---

## 不改动清单（保留的安全特性）

即使执行上述全部简化，以下 DexMani 独有的安全特性**一个也不应删除**：

1. **validate_action 的显式 `(ok, reason)` 返回** — 比 lerobot_ufactory 的静默跳过安全
2. **NaN guard** — lerobot_ufactory 缺失
3. **跟踪误差监控** — lerobot_ufactory 缺失
4. **Mode 漂移检测** — 简化为 3 行但保留
5. **碰撞安全归位路径规划** — lerobot_ufactory 盲归
6. **手进程隔离** — XHand SDK 稳定性需求
7. **外环命令限速** (`ARM_CMD_MAX_STEP_RAD`) — 与内环 delta clamp 无关的保护层
