# DexMani Real 简化方案 — 执行文档（下次会话专用）

> 本文件是「未实施项目」的执行清单。Phase 1 已全部完成并提交；本文档只描述**尚未
> 实施**的工作，按优先级排序，每项给出目标、触及文件、定位命令、步骤、验收标准与
> 风险。对照原始计划 `~/.claude/plans/sprightly-weaving-crab.md`。
>
> 生成日期：2026-08-13。分支：`simplify-phase1`。工作树 clean。

---

## 0. 当前状态快照（已完成的 Phase 1）

`simplify-phase1` 分支基于 `c57ae4d`（0813 temp0.0），其上 14 个提交。Phase 1 的
机械重构（1.1–1.4）已全部完成；随后一次 ultracode 代码审查（11 agents / 17 发现 /
6 项对抗验证）抓到并修复了 2 个「零行为变化」回归。当前 tip `7c908f6`，工作树 clean。

| 里程碑 | 提交 | 内容 |
|---|---|---|
| 1.1 死代码 | `14d609e` | 删速度包络配置、`action_validity_s`、`eef_delta_bounds` 钩子、`dt_s` 透传 |
| 1.2 校验去重 | `269f3f9` | 新增 `utils/limits.py::validate_hand_limit_nesting`，收敛 3× 重复限位嵌套 |
| 1.3 配置拍平 | `88da0b7` `89437ba` `b9fb351` | 删 `TeleopConfig._ALIASES`/`__getattr__`，全部 `cfg.` → `runtime.<section>.<field>` |
| 1.4 函数拆分 A | `a96025d` | 提 6 个纯闭包为模块级函数 |
| 1.4 函数拆分 B | `75d6793` `de3fb4a` | `TeleopLoopState` context + 6 个有状态闭包；arm_loop 2 个 homing 闭包 |
| harness | `1b991f5` `c86ab25` `7ce4813` `870988e` | headless harness + 抓出并修复 `_hold_sent_at_s` 非局部 bug |
| 执行文档 | `136f4e8` | 本文件 |
| 审查修复 | `7c908f6` | 修复 homing 返回类型 + 手部校验误删（见下） |

**验证方式**（无自动化测试套件，沿用以下流程）：

```bash
conda run -n real_robot python -m compileall -q dexmani_real examples
conda run -n real_robot python -m pytest tests/ -q        # 14 tests, no hardware
rg -n "<被删符号>" dexmani_real                            # 零残留
```

**审查抓出的 2 个回归（均已修复，`7c908f6`）**：

1. `_execute_mode0_milestones_impl`（arm_loop）失败路径返回裸 `HomeResult`、成功
   路径返回 `(HomeResult, current)` 元组，wrapper 的元组解包在任何多里程碑 homing
   失败时抛 `TypeError` → 已统一为一致元组返回，并新增 `fail_servo_code` 故障注入
   + `test_home_multi_milestone_failure_returns_result` 回归测试。
2. `_sanitize_hand_command` 被 Phase 1.2 误删了控制器侧**手部关节限位**（优雅 hold
   退化为 SafetyGate 粘滞 fault）与**命令间增量**检查（该 bound 的唯一强制点；否则
   臂动而手被 worker 丢弃）→ 已恢复两检查与调用点，docstring 标注「Do not delete
   this check as "redundant"」。

**关键既有 bug（harness 抓出，已修）**：`_enter_measured_hold` 缺 `nonlocal
_hold_sent_at_s`（提交 `7ce4813`），导致所有实测保持 0.75s 超时锁 fault 且
`_complete_reanchor` 永不触发。相对基线 `c57ae4d`，这是**唯一**的净行为变更
（`7c908f6` 是恢复基线行为，非新增变更），需真机优先验证（见 §1）。

---

## 1. 真机校验清单（终止门，最高优先级）

> 这是信任 Phase 1（尤其 `7ce4813` 与 `7c908f6`）前的必经门槛。**需要真实
> xArm7/XHand 与固件，且需硬件授权**；CLAUDE.md 规定「Do not run examples/
> without explicit hardware authorization」。未授权前不执行。

按优先级：

1. **实测保持 apply + 重建锚**（`7ce4813` 回归）——teleop 后 PAUSE/STOP/vr_stale：
   hold 必须 log `applied`（而非 `delivery timed out`），且重建锚后恢复运动。
2. **Mode-6 跟踪**——快腕旋无 tracking-error 告警尖峰；固件是速度/加速度兜底
   （C22/C24/C31）。
3. **碰撞恢复**——C24 → 有界实测保持恢复；C22/C31 → 粘滞 fault 安全停。
4. **Homing 收敛**——return_home 经 Mode-0 里程碑收敛到规范 home，位置+速度静置，
   恢复 Mode 6。
5. **Homing 失败路径**（`7c908f6` 回归）——return_home 在真实失败（SDK 拒绝 /
   C22/C24/C31 打断 / 收敛或整体超时）时返回失败 `HomeResult`（`success=False` +
   原因），worker 不崩溃；可用 `examples/keyboard_teleop.py` 或碰撞触发验证。
6. **手部接触/回程/稳态**——接触/回程/稳态误差是合法结果（非 freshness fault）；
   触觉复位；hand-home 里程碑 ACK。
7. **手部命令校验**（`7c908f6` 回归）——越界手部命令应**优雅 hold**（`retarget_ok
   =False`，臂+手同时保持）而非 SafetyGate 粘滞 fault；命令间增量越界应臂+手同时
   hold（而非臂继续动、手被 worker 丢弃）。
8. **录制**——一整条 v16 episode 仍可经 `visualize_episode.py` 与
   `replay_episode.py --dry-run` 读回。

验收：8 项全过才算 Phase 1 可信任；任一项失败，优先怀疑 `7ce4813`（保持）、
`7c908f6`（homing 返回 / 手部校验）与 Tier B 提取（用 harness 对拍）。详细清单
也在 `tests/README.md` 末尾。

---

## 2. Phase 2.1 — seqlock 去重（低风险，热路径，需逐处核对写序）

**目标**：`SharedMemoryRingBuffer` 与 `CameraRingBuffer` 各自重实现奇偶写 + 双读
校验，抽共享 `SeqlockSlot` 助手，约 -200 行，**零行为变化**。

**触及文件**：`dexmani_real/shm/ring_buffer.py`

**定位**：

```bash
rg -n "class (SharedMemoryRingBuffer|CameraRingBuffer)|def (write|read_latest|get_last_k|read_sequence|_write_idx_view)" dexmani_real/shm/ring_buffer.py
```

当前关键位置：`SharedMemoryRingBuffer`（class 62，write 167，read_latest 201，
get_last_k 218）；`CameraRingBuffer`（class 340，write 454，read_latest 577，
read_sequence 714）。

**步骤**：

1. 逐处抄录两个类的写序（odd marker → payload → even marker → write_idx）与双读
   校验逻辑，确认语义**逐字节一致**后再合并。
2. 新增 `SeqlockSlot` 承载「单槽奇偶写 + 读校验」；两个 ring 只保留各自的内存
   布局与 `get_last_k`/`read_sequence` 语义差异。
3. `compileall` + 全量 harness（现有 14 测试已覆盖两类的读路径）+ focused diff。

**风险/门槛**：这是最安全关键的 IPC 热路径。合并后必须逐处核对写序与双读校验；
任一差异即停止，不得为过测试而放宽。

**验收**：14 harness 全绿 + focused diff 确认零行为变化。

---

## 3. Phase 2.2 — 共享内存表面精简（可选，中等风险）

**目标**：7 心跳 → 1 结构化数组；7 ready 事件 → 1 位掩码；若干元数据字段 → 1
JSON blob。收益中等、调用面广，**默认不并入主线**（「安全优先」），或后置 Phase 3。

**触及文件**：

- `dexmani_real/shm/shared_storage.py`（心跳声明 245-251，分配 416-424；ready 声明
  253-259，分配 426-432；`close()` 记账 446+）
- `dexmani_real/runtime/supervisor.py`（迭代 `heartbeat_fields` 47-125，ready 消费 141+）
- 各 worker 的心跳写点（`rg -n "heartbeat_s" dexmani_real/`）

**定位**：

```bash
rg -n "heartbeat_s|_ready\b|metadata" dexmani_real/shm/shared_storage.py
rg -n "heartbeat_fields|wait_subsystem_ready|_ready" dexmani_real/runtime/supervisor.py
rg -n "\.heartbeat_s\b|\._ready\b" dexmani_real/robot dexmani_real/sensor dexmani_real/teleop dexmani_real/recording
```

**步骤**：

1. 先列全每个心跳/ready/metadata 字段的**所有读写点**（写点 + supervisor 迭代点 +
   `close()` 记账），确认无遗漏。
2. 逐字段迁移，每次一个字段族（心跳→结构化数组；ready→位掩码；metadata→JSON），
   单独 commit。
3. 每步 `compileall` + harness + focused diff。

**风险**：`close()` 记账与 supervisor 迭代读法会改；跨进程 dtype 布局变化需与
`utils/schema.py` 契约一致。**列为可选**，若不确定则不并入主线。

**验收**：14 harness 全绿 + `close()` 记账无泄漏（进程退出无 shm 残留）。

**实施结论（2026-08-13）**：心跳→结构化数组、ready→位掩码已实施并提交（零行为
变化，19 harness 全绿）。**metadata→JSON 已评估并判定不实施**：相机元数据 6 字段
现占 ~2288 B，合并为单 JSON blob 需把 2047 B 的 profile 作为转义字符串内嵌，blob
需 ~4.5–8 KB——比被替换字段更占内存，且给录制路径增加 JSON 序列化/解析开销，并引入
「截断即不可解析」的损坏模式；设备身份字段由不同 worker 写、各自已是单 JSON 字符串，
`camera_observation_required` 是控制标志而非元数据。该合并并非净简化，违反 §6
「零行为变化」约束，故按「安全优先 / 若不确定则不并入主线」不予实施。

---

## 4. Phase 3 — 可选 / 需真机或 profiling 先行的项（本次不展开）

以下 4 项在原始计划中列为「需真机或 profiling 才做」，按依赖排序，**不要在没有
前置数据时启动**：

1. **recorder → 线程**（降级项）：仅当先做 **16Hz 控制环抖动 profiling**、确认
   h5py fsync + 视频编码线程并入 teleop 进程不产生抖动时才做。recorder 进程隔离
   fsync/编码是**有意为之**（memory `h5py flush→fsync→USB starvation`），非过度工程。
2. **录制模型 → ManiUniCon `timestamp_accumulator` per-producer `.npz`**：用户已选
   **暂不迁移**。
3. **配置框架 → hydra**（ManiUniCon `main.py` + `configs/*.yaml` + `_target_`）。
4. **arm/hand worker 内「SDK I/O」与「编排」拆层**：ManiUniCon 式 `RobotInterface`
   + `Robot.run`（薄 driver / 编排在循环里）。

---

## 5. 关键文件清单（剩余工作会触及）

- Phase 2.1：`shm/ring_buffer.py`
- Phase 2.2：`shm/shared_storage.py`、`runtime/supervisor.py`、各 worker 心跳写点
- Phase 3：`recording/io_process.py`、`recording/episode_recorder.py`、
  `policy/`、`config/`、`robot/arm_loop.py`、`robot/hand_process.py`

---

## 6. 全局约束（下次会话必须遵守）

- **零行为变化**：任何剩余项都是机械重构，不得改变 HDF5 v16 数据契约或运行时语义。
- **不跑真机**：`examples/` 入口需硬件授权；真机校验（§1）是显式授权后才做的独立
  关卡。
- **验证三件套**：`compileall` + 符号 `rg` + focused diff；Phase 2 加 harness 全绿 +
  既有 episode 的 `visualize_episode`/`replay --dry-run` 兼容检查。
- **每里程碑独立 commit**，保持可回滚；任一语义差异即停止，不为过测试放宽检查。
- **「冗余」判定以行为为准**：审查证明「存在另一处类似检查」≠「可删」——`_sanitize_hand_command`
  的手部限位/增量检查看似被 SafetyGate / worker 覆盖，实为「优雅 hold vs 粘滞 fault」与
  「唯一强制点」的语义差异。Phase 2 的 seqlock/内存表面合并同样必须先证明行为等价。
