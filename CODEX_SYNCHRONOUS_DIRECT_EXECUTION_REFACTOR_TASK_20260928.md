# Codex CLI 任务书：DexMani 同步直接执行的完整重构

日期：2026-09-28  
仓库：`haoyangzhanglab/dexmani_real`  
任务书编写时核查的代码 HEAD：`01feb5e30b4641f37c2128b68eb64dce288f3680`（`main`）  
其父提交：`6d199495ef024ab97889bf2e32b466cf7e2bd882`（前置清理）  
方案依据：最终 V3，同步直调、薄组合、单一设备 I/O 所有者。本文完整承载实施要求，无需访问聊天附件。  
授权范围：实现代码、无硬件离线验证、最终自审；**不包含任何设备连接、运动或现场实验授权。**

> 从当前实际 HEAD 和工作树继续，不 checkout/reset 到上面的历史 SHA。本任务不是再次修 pacing，也不是 FIFO/ACK transport 修补。只实施下文固定的同步直接执行方案。本文是用户本次明确要求放在根目录的临时执行文件；不要以 AGENTS.md 的长期文档约定为由删除、移动或改写本文。

## 0. 给执行代理的直接指令

先完整阅读当前 `AGENTS.md` 和本文，再核对当前源码与工作树。按 R0–R5 顺序完成实现、纯离线验证和自审，不止于输出设计。保留用户已有改动，不恢复已删除文档、tests 或 rollout 统计，不重开架构选型讨论。

执行中若发现代码已实现某项，验证并保留，不重复实现。若存在确定的代码冲突、缺依赖或安全准入阻断，完成不受影响的离线工作，精确报告未完成部分；不要绕过校验、删掉断言或擅自上真机“验证”。默认不自动 commit/push、不修改分支、不改变 Codex 权限或安装/升级实验环境。无需重新创建一份任务书。

## 1. 最新版本校准：已完成与待完成分开

### 1.1 已完成，不重做

在核查 HEAD 中：

- 三份旧文件 `CODEX_POLICY_COMMAND_TRANSPORT_FIX_TASK_20260928.md`、`CODEX_SYNC_ROLLOUT_TIMING_FIX_TASK_20260928.md`、`PICK_PLACE_TOY_ROLLOUT_ANALYSIS_20260928.md` 已删除。
- 旧 `tests/`、`runtime/diagnostics.py`、`deployment/diagnostic_analysis.py`、诊断分析/probe CLI、`RolloutStats`、`--diagnostics` 及其调用链已清理。
- `PolicyRunner.step()` 已使用 gate 通过的 `now` 设置 `next_step_ns = now + dt_ns`，不是旧的 `publish_stamp + dt_ns`。
- 模型仍走一次 `model.predict()`，本地 deque 逐步消费；正常 `AsyncEpisodeRecorder` 和 `run_config.yaml` 保留。
- session 仍启动 policy、arm、hand 进程；正常动作仍经过 `publish_command()` 和旧 command ring。**清理完成不等于直接执行迁移完成。**

不要再实现旧 `inference_ms` 统计修补：统计对象已经删除。不要恢复旧测试以便跑出历史的“20/20”。

### 1.2 `01feb5e3` 新增规则必须保留

相对 V3 引用的 `6d199495`，最新提交修改了以下七个文件。它们不是待回滚的噪声：

| 文件 | 最新有效行为与约束 |
|---|---|
| `AGENTS.md` | Raw immutability 从成功发布 Raw 后开始。技术无效、尚未发布的 Teleop capture 可以整体 discard；该规则不扩展为丢弃所有失败 rollout。 |
| `robot/action.py` | `policy.hand_enabled=True` 时必须有 hand intent；False 时不得携带 hand intent。不能删掉校验来迁就新接口。 |
| `calibration/camera/extrinsics.py` | 相机姿态先校验 unit quaternion，再处理容差内归一化；不退回对任意非单位输入静默归一化。 |
| `calibration/camera/solver.py` | 标定位置/四元数按实际 float 精度保存，不重新 round 到六位。 |
| `config/hardware.py` | handbase quaternion 使用严格单位四元数校验。 |
| `utils/geometry.py` | 旋转正交性以最大绝对元素残差及 determinant error 校验，不退回旧 Frobenius 判定。 |
| `examples/calibrate_vr_heading.py` | 元数据为 `calibrated_at_utc`，使用 aware UTC datetime。 |

上述数值、标定和数据准入修正不属于本任务算法修改范围。入口确有 hand-enabled 配置与目标不匹配时，修正显式配置/调用关系，不能补零 hand、临时切 flag 或删除门槛。保留未启用 hand 入口的物理场景假设，不悄悄连接 XHand。

### 1.3 开始时的版本核对

仅执行只读检查，记录实际结果：

```bash
git rev-parse --show-toplevel
git rev-parse HEAD
git status --short
git diff --stat
git diff --cached --stat
git log -5 --oneline
```

若当前 HEAD 晚于本文基线，先阅读相关提交差异；若分叉，不强行重置。本文基线只是核查证据，不是要求 checkout 的版本。对工作树原有修改建立清单，并在最终报告中区分本任务新增改动。不要使用 `git reset --hard`、`git clean -fd`、强推或自动 stash。

## 2. 唯一目标与参考边界

**一个同步控制循环 + 一个薄组合 `DexManiRobot` + 现有设备驱动。**

```text
Policy/model + 本地 action deque ──┐
VR/HTS → 现有映射与 retargeting ────┼→ 本轮动作/意图
Replay / Keyboard / Calibration ──┘        │
                                          ▼
                              现有 ActionRealizer（需要时）
                                          │
                                    RobotCommand
                                          │
                                          ▼
                                薄 DexManiRobot
                                 /             \
                           现有 XArm7       现有 XHand
                                │                │
                         SDK / Mode 6       SDK / 手部控制器

相机/点云/VR进程 → 保留的 sensor channels → 本轮 ObservationRow
设备本地读数 ───────────────────────────→ history / 安全 / 正常录制
```

参考只按职责映射，不整体复制：

| 参考 | 采用 | 不采用 |
|---|---|---|
| UFACTORY LeRobot | `observe → select/predict → direct send → remaining wait` | 特有首动作对齐、默认速度、示例平滑。 |
| LeFranX | 薄组合 arm/hand 对象、独立输入来源、顺序发送 | Franka server/Ruckig、stub fallback、零动作兜底、未实现却返回成功的 stop。 |
| ManiUniCon | observation/action/robot/state/reset 的职责分离 | 整套多进程执行器、ready 握手、action FIFO、target_timestamp、插值和时间重排。 |

ManiUniCon 的 xArm6 路径是 Mode 1 + `set_servo_angle_j()`，不是本项目 Mode 6。不要复制其 200 Hz 控制循环或另一套 FK→裁剪→IK。[M1–M4]

本方案消除的是“已选动作在应用 mailbox/30 Hz 消费层中被覆盖，未进入 SDK”。不要求固件独立执行完每个 endpoint，不保证两设备物理同时到达，不承诺消除模型重新预测引起的回拉。没有必要取消 SDK 内部 report thread、控制器在线规划或传感器 latest buffer。

## 3. 范围边界与不做事项

必须迁移 policy、VR teleop、keyboard、replay、camera calibration 的正常设备调用到相同的直接 I/O 实现；不是只修 policy 后永久留下两套设备执行路径。

禁止新增：

- 应用 command mailbox/FIFO、逐动作 capture ACK、跨段 ready event、future-action list、`apply_at_ns` 或目标时间调度器；
- RTC、异步推理、prefetch、额外动作插值、低通/EMA、延迟补偿、按墙钟跳 action 前缀；
- 第二个应用 SDK 调用线程、broker/server、自制 watchdog、Mode 1/Ruckig；
- Robot/Adapter 多层套壳、ABC/registry、通用 ActionSource/executor、LeRobot/Draccus 配置迁移；
- 旧文档/tests/rollout sidecar、每步耗时列表、p95/query-drift 统计、完整预测 dump；
- Raw 重写、checkpoint/模型训练修改、相机标定改参、参考项目 feature schema 替换。

保留既有 teleop 显式 EMA/映射配置；“不新增平滑”不是授权删除示教原有处理。保留 recorder 内部有界 FIFO 和传感器 rings；禁令针对应用动作中转，不针对所有 queue/event。

## 4. 薄 Robot 与单一 I/O 所有权

建议只新增 `dexmani_real/robot/robot.py` 中一个具体 `DexManiRobot`。接口可沿用更合适的现有命名，不为凑接口增加包装：

```python
class DexManiRobot:
    def connect(self): ...
    def read_state(self): ...
    def send_action(self, command): ...
    def stop(self): ...
    def close(self): ...
```

### 4.1 职责

- 组合现有 `XArm7` 和按配置启用的 `XHand`，负责连接、有效状态读取、全目标预检查、顺序发送、停止/关闭。
- `RobotCommand` 已包含 `run_id`；沿用一个权威 epoch 来源，不再同时传入可不一致的第二个 run_id。内部安全上下文/权限检查可以注入现有对象或小回调，不建立权限服务。
- 模型/历史/deque/节拍归 runner；HOME 规划、实测收敛归已有 homing 模块；VR 映射归 teleop；录像归 recorder。
- 不让 Robot 反向持有 policy、planner、Teleoperator 或 recorder；不在 `send_action()` 内重新求 IK。

### 4.2 所有权

同一 session 的模型同步调用、设备 read/send/home/stop/close 在一个应用 I/O 所有者线程中串行进行。键盘监听只发请求/撤权，不调用设备。已连接 SDK 不跨 spawn 传递。传感器进程、SDK 内部线程和 recorder writer 可以保留。

普通构造/import/配置解析不得连接硬件。必要硬件依赖延迟到启用该设备的真实路径导入，尤其 arm-only 入口不能因为模块顶层导入 XHand 而强迫安装或连接手部；缺失必需 SDK 必须报错，不自动 stub。

partial connect 时独立清理所有已创建/已连接设备。不能只检查 Composite 整体 `is_connected`，漏掉仍然连接的 arm。保留既有驱动有界连接过程，不顺手改协议、参数或重试规则。

### 4.3 运行参数

保留 xArm Mode 6、`set_servo_angle(..., wait=False)`、XHand 命令和返回值规则；不在每 tick 清错/切模式/重连。HOME/停止后模式恢复只能在现有授权条件下发生，恢复后再次检查权限。

`execute=False` 不是自动获得无硬件副作用的保证：先核对其当前语义，绝不能把它当作在实验机上随意运行验证的许可证。离线验证必须显式注入 fake drivers，不连接任何设备。

## 5. 状态、观测和模型数值契约

### 5.1 本地状态替换旧状态 IPC

把旧 workers 中仍必要的状态 shape/finite、控制器错误、XHand board 状态、aggregate/dense 触觉有效性、校准与反馈失败处理迁到本地读取职责，之后才能删除 worker。

保留 `ObservationRow` 和需要的 NumPy dtype。状态 ring 删除不等于状态字段可以删除。复用现有 sensor lookup 和 RGB/点云同 camera-sequence 条件，不重算点云、不替换标定。

主机每次真实 SDK 读取的完成时间只能标作 host read-completion；有硬件来源时间则保留，没有则不虚构。重复读取缓存不能刷新来源时间后重新取得 freshness。合法静止状态数值相同不代表重复采样。

本地运行态按真实 control tick 读 robot state；空闲和 HOME 仍须主动服务状态、请求和故障检查，不能依赖已删除 worker 更新反馈。保留已有空闲/HOME 读取需求，不忙等。

### 5.2 每 tick 一条 policy history

`policy_row` 是本轮进入 history 的不可变完整快照。先检查 None/有效性再访问字段。

阻塞推理后先检查授权、故障、quit 和运行预算；仍可继续时，沿当前实现需要补读 `execution_state`，只读取 robot，不复制新 RGB/点云、不重算整份观测、不追加 history。安全或 IK 使用新读数，不改写模型已经看到的旧输入。

freshness、history gap、必需模态准入按当前语义迁移。不放宽阈值掩盖慢循环，不以新读数“证明”过去模型输入变新；不新增无依据的固定预测超时或 29/40 ms 补偿。已失去权限/现有预算的预测不继续执行。

### 5.3 唯一模型桥与 ActionRealizer

继续调用当前 `dexmani_policy` 部署桥；模型加载/warmup 一次，episode reset 沿原语义，一次 query 一次采样。只保留一份 history 和一个本地 action deque；deque 空时才推理，每正常 tick 至多取一个动作。

不改变 `n_obs_steps`、`horizon`、`n_action_steps`、control_dt、EMA 权重选择、归一化和 `n_obs_steps-1` 切片。8-step 案例仍为完整预测索引 1…8；不要二次切片、每 chunk reset seed 或按 wall-clock 跳项。

joint 路径只做当前关节投影；EEF 路径只做当前 workspace/IK。硬件前重复检查 shape/finite/limits 是必要安全检查，但不重复数值变换。保留最新 hand-enabled 契约，校准单位、19D 排列、电流 mA 和未知力/effort 单位不改。

连续相同数值的已选动作仍是两个控制步，不新增相等目标去重。仅对同一保存输入和同一 RNG 要求输出一致；不要求修改后的真实闭环复现旧轨迹每个数值。

## 6. 周期和顺序：继续使用正确的同步预算

现有 tick-start gate 可直接保留，不为了伪代码重新造 scheduler：

```python
# 仅描述职责；按实际 API 实现。不是可直接执行的设备脚本。
now = monotonic_ns()
if now < next_step_ns:
    return
next_step_ns = now + dt_ns
check_authority_services_and_budget()
policy_row = read_current_observation()
history.append(policy_row)
queried = not action_queue
if queried:
    prediction = model.predict(build_policy_observation(history))
    check_authority_services_and_budget()
    validate_prediction(prediction)
    action_queue.extend(prediction)
execution_state = robot_only_recheck_if_needed(policy_row, after_query=queried)
selected = action_queue.popleft()
command = realize_selected_action(selected, execution_state)
result = robot.send_action(command)
record_normal_frame(policy_row, command, result)
# 外层仅等待剩余预算；实际 SDK 与正常 recorder 入队时间都计入。
```

- 所有相对时长使用同主机 monotonic clock；wall clock 用于日期/目录。设备时钟不得未经映射直接相减。
- `wait=False` 不等于 SDK 调用耗时为零。读、推理、发送、正常记录工作均占本轮预算。
- 超预算则减少/取消剩余等待；下一 tick 从实际起点计时，每 tick 最多一个动作，无 catch-up while、无额外发送后完整 dt。
- 不保证所有相邻 SDK 调用间隔至少 dt；一次超期后的短相邻间隔不自动等于批量补发。
- 等待可按现有停止事件/短有界睡眠中断，不引入长期 busy-spin 或每 tick 打印。

### 6.1 精确 fake-clock 验收

dt=62.5 ms、每 8 tick 仅查询耗时20 ms、其他耗时为零时：

| 指标 | 正确结果 |
|---|---:|
| 旧段最后目标 → 新段第一目标 | 82.5 ms |
| 新段第一目标 → 第二目标 | 42.5 ms |
| 两间隔总和 | 125 ms |
| 相邻 query start | 500 ms |

增加 read/send/record 开销后按实际工作预算推导，不能硬凑上述数字。使用40 ms查询复现原目标1、9、10场景，新直调链应按值和顺序各调用一次；原30 Hz消费者不再存在。无需给生产代码恢复全局 sequence 或统计侧车。

## 7. 完整动作发送、返回值与异常

普通动作 `send_action()` 的权限仍为 RUNNING；HOME 使用已有 ARMED 权限路径；stop/close 不能因为已经撤权就拒绝执行。

1. 先验证所有 present targets 的 shape、finite、限位及相应设备可用性。任一失败，不开始任何设备发送。
2. 检查 epoch/安全状态/请求/运行预算；必要的 mode 恢复后再检查。
3. 固定 arm→hand，各 SDK 调用前都重新检查。arm 明确失败时不继续发 hand。
4. 保存真实返回结果。一个小的本地结果对象/携带部分结果的异常足够，不新增结果协议框架。
5. 正常路径每设备调用一次；不为“不确定”自动重发、不握手、不等待物理到位。

`XHandSendStatus` 当前是 `ACCEPTED / CRC_UNCONFIRMED / REJECTED`。CRC_UNCONFIRMED 的既有继续规则不能在这次所有权迁移中自动改成确认成功或一律故障；保留其不确定标记和原处理策略。既有 stop 的不确定响应也不得被当作确认完成。API 抛异常时可能已发送，必须区分“未调用”与“调用后结果未知”。

arm 已发而 hand 未发/失败/被撤权，属于部分发送，不可回滚 arm。保留事实并进入相应终止路径。两个 SDK 成功也不表示物理同步或接触建立。

previous-command、teleop 接受参考只根据已知实际发送结果及既有不确定性策略更新；不得把尚未发送目标写成已执行参考。失败终止后清除缓存，不继续沿错误参考运行。

## 8. HOME、操作者、监督和停止收尾

### 8.1 HOME 不简化为发送 home pose

迁移现有 `arm_homing.py`、`hand_homing.py`、`home.py`，删除的只是 worker 请求/结果传输：

- 保留启用 hand 时先手后臂的现有流程；hand-disabled 分支不连接手，并保留规划假设。
- 保留环境/整条回零路径碰撞检查、周期关节等价角对齐、fresh stationary arm、post-home hand 状态约束。
- 手部 HOME 仍检查后续真实读数和连续收敛样本（当前是3次），不得对同一份缓存连数3次。
- xArm HOME 保留 Mode0、各段时限、位置/速度收敛、dwell、Mode6恢复后的验证及 abort。
- 移除 request/result queue timeout 时，只删除纯传输参数，不能顺便删物理收敛/行程时限。

键盘监听线程只设置请求/撤权。H 的规划和设备调用交给 I/O 所有者。H期间/同批旧B无效，H成功仍为ARMED，必须新的B；S/Q/ESC和故障优先，晚返回的H不得覆盖新S。保留 C/D 在各 workflow 的真实含义，不把 policy 的no-op套到teleop。

### 8.2 一个生命周期，先停止再保存

沿用四状态与run_id，不新建第二套FSM。S/Q/故障/预算到期后：

```text
先撤权/清理待发权限
    ↓
I/O所有者获得控制后，先请求停止每个实际连接设备
    ↓
清action deque、history、previous command及episode临时状态
    ↓
按workflow规则保存或丢弃尚未发布capture，drain recorder
    ↓
关闭设备/模型，停止并确认传感器进程退出，再释放它们仍使用的资源
```

episode停止不必断开设备；session退出才完整关闭。意外异常、RecorderError、启动失败、HOME异常都覆盖。停止的一侧失败不能跳过另一侧；原始错误与清理错误分别保留，非零失败不能最后写成成功/DISARMED掩盖。

XHand沿用足够新实测位置hold、不可用时passive的规则，不用旧endpoint或全零张开目标代替。xArm当前stop是best-effort，不宣称已经验证静止。

### 8.3 监督迁移

session不再启动policy→arm/hand三段动作进程链。模型和robot由本地session/runner组织，仍需的相机/点云/VR进程保留。

supervisor只监督实际存在的子进程；在控制边界、长调用返回后、空闲和有界等待中检查服务状态，不再等待不存在的arm/hand/policy ready。仍活跃的sensor/recorder引用必须退出后才能释放共享资源。不要取消故障上报或用已删除worker的exitcode证明物理停止。

## 9. 正常录制与最新 Raw 准入规则

### 9.1 两类失败不能混淆

- 已发布 Raw：只读，不修改、删帧或重写termination。
- 尚未发布的Teleop capture：保留当前 `mark_discard()`、control/publication/cadence failure导致整段不发布的规则。这不是“删除已发布失败证据”。
- policy rollout：按现有失败/终止语义保存可保存的事实；不能套用Teleop规则把所有抓取失败/部分发送rollout删掉。存储损坏不能伪造成功保存。

`publication` 改为direct dispatch后，明确原Teleop技术准入如何映射到调用结果。不要仅改函数名而丢失discard路径。不要把历史普通循环间隔替换为全局理想时间网格以凑cadence。

### 9.2 最小数据调整

继续用 `AsyncEpisodeRecorder`、`EpisodeFrame`、`run_config.yaml`、既有时间/来源/终止元数据；每帧只提交一次且owned数据不可变。

记录模型历史使用的 `policy_row` 和实际目标；推理后的 `execution_state` 不冒充原模型输入。当前runner在查询后会重新读取整行并用于执行/记录，迁移时显式区分这两者并说明新记录的观察/动作配对口径，不能声称完全没有记录语义变化。

新session必须标识直接SDK dispatch路径，区分selected、已调用、SDK接受/不确定、实际到达。部分发送时分别表达arm/hand，不能把未尝试一侧填成完整19D成功动作，也不能用NaN抹掉已尝试但结果未知的事实。

当前 `recording/storage/schema.py` 会拒绝未定义的额外HDF5字段：优先用现有目标字段和最小的普通结果/终止元数据表达。若确实必须扩展存储，同步检查writer、loader、validate、export，给新契约明确标识并做读写回归；不得仅加字段让读取器报错，也不为旧数据编造SDK结果。禁止重建diagnostics数据库或完整预测侧车。

## 10. 安全与反馈：离线完成不等于准入现场

单I/O线程在推理、SDK或HOME阻塞时不能同时发另一停止调用。监听回调能先撤权，物理停止仍要设备路径支持。`max_running_s`不是硬实时watchdog；强杀进程不是设备已停止证明。

- 每个设备调用前和长操作后复查权限/预算；长操作、保存和SDK调用不持有motion_lock。
- 后来的标志不能撤回已跨过检查、进入SDK的在途调用。测试验证“已观察到撤权后无新的未授权调用”，不要写成不可能保证的纳秒级原子边界。
- 删除worker后不承诺独立于模型的30 Hz Python触觉读取。保留SDK内部接收机制，但不能假设它自动满足XHand采样需求。
- 现场必须另行确认xArm和XHand各自停止覆盖、阻塞容忍边界、反馈/触觉需求和实测总工作预算。未确认只交付离线实现，不运动。
- 若必须模型阻塞期间持续Python采集且无已验证途径，列为现场准入阻断；不得悄悄加SDK并发线程、放宽freshness或把能力宣称保持不变。

这类现场待确认项不要求代理为了等用户而放弃其余可安全完成的离线实现，也不授权代理发明另一套架构替代本文。

## 11. 文件级迁移清单

| 文件/模块 | 必要修改 |
|---|---|
| `robot/robot.py`（建议新增） | 唯一薄组合及最少本地发送结果；保留驱动，lazy import启用设备。 |
| `robot/commands.py` | 保留纯数据，移除旧publish/read transport；不能藏进新send_action。 |
| `robot/action.py`、`robot/projection.py`、`command_validation.py` | 保留数值和最新hand_enabled规则；只调整真实发送参考/所有权接口。 |
| `robot/arm_worker.py`、`hand_worker.py` | 迁走反馈有效性/停止/模式职责后删除动作polling workers。 |
| `robot/drivers/xarm7.py`、`xhand.py` | 保留协议、模式、单位、状态；必要lazy import和过时worker注释最小调整。 |
| `robot/arm_homing.py`、`hand_homing.py`、`home.py` | 本地调用替换HOME队列，保留全部物理规划/收敛/abort。 |
| `deployment/runner.py`、`session.py`、`operator.py` | 本地模型/robot；保留deque/history/pacing；长调用后先复查，stop早于save。 |
| `runtime/observation.py` | 本地状态+现有sensor channels；policy_row与execution_state分开，单条history。 |
| `teleop/control/controller.py`、`teleop/session.py`、`keyboard_session.py` | 复用robot I/O，保留显式EMA、C/D、参考更新、technical-invalid discard和自身节拍。 |
| `replay/replayer.py`、`replay/session.py` | 直接发送但保留原回放时间、准入和停止规则，不补发历史backlog。 |
| `calibration/camera/motion.py`、`session.py`及其他实际caller | 迁移设备所有权，不动本轮新增的标定数值修正。 |
| `ipc/channels.py`、`schema.py`、`config/*` | 删除无使用者的command/state/HOME IPC、ready/queue配置；保留sensor IPC和本地dtype。删除loop_hz前核查其是否仍派生freshness/其他参数，不意外改阈值。 |
| `runtime/supervisor.py`、`processes.py` | 只管理真实子进程，明确本地失败/关闭语义。 |
| `recording/*`、必要`dataset/*` | 仅最小dispatch/partial结果适配；保留Teleop与rollout准入差别。 |
| CLI、README | 描述实际单路径及采样/停止边界，保留正常入口；无legacy/FIFO/ACK切换选项。 |

这是调用关系清单，不是要求无差别修改每个文件。使用当前源码搜索补齐所有引用，包括arm-only、小工具和配置默认值。最终不能永久保留两套硬件路径或同名wrapper伪装迁移。

## 12. 实施步骤（R0–R5）

### R0：只读基线与引用地图

完成第1节检查；阅读当前源码，不读已删除旧任务书。搜索所有producer→realizer→command→SDK、HOME、状态ring、supervisor、recording consumer。确认最新七文件修正仍在。检查参考固定源码所需片段即可，不克隆/安装整套框架。

### R1：薄Robot、读取与发送（纯离线）

显式fake drivers验证连接所有权、部分连接清理、hand配置、全目标预检查、arm→hand、返回状态、停止/关闭；先验证实际函数，不只写一个与生产代码脱离的演示模型。

### R2：policy单一路径

接入runner/session/operator，本地读写和HOME流程闭环。保留当前pacing而非再次“修正”；检查停止先于保存、模型队列不双份、无隐藏mailbox代理。

### R3：其余workflow迁移

逐一迁移teleop、keyboard、replay、camera calibration及检索到的caller。同一发送实现，不同输入/周期。每完成一类做对应fake检查，尤其未发布Teleop丢弃准入和arm-only配置。

### R4：引用删除与正常录制闭环

最后一个caller迁移后删除旧worker、command ring、HOME transport及失效配置。保留必要sensor资源。做临时Raw写入→读取/验证→需要的派生consumer smoke；不改任何历史数据。不恢复tests和统计模块。

### R5：全集成离线检查与最终差异自审

验证矩阵全部执行，审查相对任务启动工作树的新增diff，确保没有覆盖最新数值修正。阶段间可使用临时分支内未迁移代码帮助实现，但最终交付不得有双路径/legacy开关。没有用户另行要求，不自动commit/push。

## 13. 仓库外最低验证矩阵

临时脚本/日志放在权限允许的系统临时目录或仓库外工作目录；输出确切路径与命令。不要恢复 `tests/`，不要写空测试后报告通过，也不要执行测试发现0项并称完整验收。

| 检查 | 必须有的断言/证据 |
|---|---|
| 版本回归 | 第1.2节全部最新校验仍在；四元数错误被拒，标定精度/UTC未回退，hand配置缺/多intent均失败。 |
| 所有权/依赖 | 无第二SDK owner；构造/--help无连接；缺SDK不stub；arm-only不导入/连接未启用手的硬件路径。 |
| 发送预检查 | hand非法时arm也未发送；各有效目标原值与顺序保持。 |
| 原覆盖场景 | 启动目标及原第9/10目标按值分别直调一次，不依赖30Hz消费者相位或新sequence协议。 |
| 重复目标 | 两个数值相同的正常tick不被去重。 |
| 时序/预算 | 零附加开销82.5/42.5/125/500 ms；注入read、SDK、record耗时后按总预算；80ms查询超期无批量追赶。 |
| 模型/历史 | 一次query一次采样；同输入同RNG同输出；每tick一行history；robot-only补读不追加、不复制大观测；episode reset正确。 |
| 几何与配置 | joint不新增FK→IK，EEF只走已有实现；最新hand_enabled与物理设备配置一致。 |
| None/来源/触觉 | 先检查存在性；缓存读取不刷新采样时间；aggregate/dense/校准/CRC区分保留，mA不是Nm。 |
| 发送结果 | arm失败不发hand；arm已发hand失败/撤权可见；CRC_UNCONFIRMED不改成确认或自动重发。 |
| 撤权/预算 | 模型阻塞中、Mode恢复后、两个SDK之间、HOME中注入撤权；返回后无新未授权调用。 |
| HOME与事件 | 未收敛不授权；连续新样本才计数；H/B同批与H期间B无效；新S不能被旧H覆盖。 |
| 停止/异常 | stop先于save/drain；arm stop/close异常仍尝试hand；部分连接清理；原始异常不被finally覆盖。 |
| 录制准入 | Teleop技术无效capture整体不发布；不能外推为丢弃所有失败rollout；历史Raw不变。 |
| 录制读写 | 本轮policy_row、真实target、partial/unknown/termination可解释；新增元数据/字段能被实际loader/validator读取。 |
| 生命周期 | model warmup一次；每episode缓存隔离；停止不自动重连/重启；sensor仍活跃时不unlink。 |
| 全caller | policy/teleop/keyboard/replay/calibration均走相同Robot；旧接口/配置引用为零或明确的仍有效非动作用途。 |

用fake SDK/显式patch保护设备边界，关键测试必须经过实际新Robot/runner/生命周期函数。对缺失policy可选依赖报告阻断，不能拿fake model替代真实桥回归后宣称同RNG集成已通过。

最终运行已有环境中的无硬件检查：

```bash
python -m compileall -q dexmani_real examples
ruff format --check dexmani_real examples
ruff check dexmani_real examples
git diff --check
git diff --stat
git diff --cached --stat
```

对全仓既有lint和任务新增问题分开说明，用启动前基线作证；不要机械抄写历史“10项existing”。修改文件必须通过focused检查；不顺手全仓format。缺工具就报告，不擅自pip升级。普通`--help`也要先审查导入/入口是否有硬件副作用。

全文检查工作树diff、staged diff及新文件；通过`git grep`/`rg`核查旧`publish_command`、`read_robot_command`、`robot_command_ring`、`run_arm_worker`、`run_hand_worker`、HOME queues和无用readiness。本文的文字引用不计为运行代码残留；不能因查到本文而删任务书。

## 14. 完成判据与最终报告

报告至少包含：实际启动HEAD/工作树基线、最新修正保留情况、各R阶段状态、文件改动原因、唯一调用链、实际验证命令/数量/结果、失败或skip原因、临时证据位置、数据契约变化、剩余现场准入事项。

**离线完成**需要所有active callers完成迁移、旧动作路径删除、必要验证真实执行、最新数值与数据规则不退化。只删除文件、只通过compile、只完成policy一路都不是完整重构。

**现场准入**另行评估停止覆盖、阻塞边界、采样需求和实测时延。本任务结束必须停在离线阶段；不安排自动硬件运行，不承诺抖动/抓取已经解决。

若有未解决项，交付准确的部分完成状态及具体证据，不删检查追求绿色；不要为未解决现场条件增加另一套运行架构。

## 15. 固定版本参考（只读、定点核查）

引用用于定位设计依据，不代表已执行参考代码。它们不覆盖当前项目的数值、安全与Raw规则。不要因参考无法联网而恢复已废弃方案。

- 最新差异：[01feb5e3 commit](https://github.com/haoyangzhanglab/dexmani_real/commit/01feb5e30b4641f37c2128b68eb64dce288f3680)
- 当前规则：[AGENTS.md](https://github.com/haoyangzhanglab/dexmani_real/blob/01feb5e30b4641f37c2128b68eb64dce288f3680/AGENTS.md)
- [U1] [UFACTORY同步评估循环](https://github.com/xArm-Developer/lerobot_robot_ufactory/blob/5fe6fb150bfa71ad3ca8a0b780c89ff8d4033a82/src/lerobot_robot_ufactory/scripts/uf_lerobot_eval.py)
- [U2] [UFACTORY joint Mode6设备接口](https://github.com/xArm-Developer/lerobot_robot_ufactory/blob/5fe6fb150bfa71ad3ca8a0b780c89ff8d4033a82/src/lerobot_robot_ufactory/robots/uf_robot/uf_robot.py)
- [L1] [LeFranX薄组合Robot](https://github.com/wengmister/LeFranX/blob/a39906e6629f39490950fe8bd20f4f992ed74fd7/src/lerobot/robots/franka_fer_xhand/franka_fer_xhand.py)
- [M1] [ManiUniCon Robot及执行/复位分层](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/maniunicon/core/robot.py)
- [M2] [ManiUniCon策略/观测/动作wrapper与同步握手](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/maniunicon/policies/torch_model.py)
- [M3] [ManiUniCon xArm6 Mode1接口](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/maniunicon/robot_interface/xarm6_robotiq.py)
- [M4] [ManiUniCon默认频率配置](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/configs/default.yaml)

## 16. 用户启动用短 Prompt

下面是用户启动新Codex任务时的提示，不是让执行中的Codex递归调用自己。不要resume旧FIFO/ACK任务。先确保本地已有本文件和最新用户修改，使用当前CLI支持的workspace写权限，不绕过sandbox/规则。

```text
请先完整阅读 AGENTS.md 和 CODEX_SYNCHRONOUS_DIRECT_EXECUTION_REFACTOR_TASK_20260928.md。
以当前HEAD和工作树为基线，核对并保留01feb5e3后的所有相关修正，按任务书R0–R5完成同步直接执行重构、仓库外离线验证和最终自审。
固定采用薄组合Robot、现有驱动、同步循环；不恢复旧tests/统计，不引入FIFO/ACK/RTC/异步推理，不改变模型、Mode6、标定或Teleop Raw准入规则。
不回退、不覆盖用户修改，不自动commit/push，不连接任何硬件。完成离线实现后报告实际验证证据、未完成项及现场准入边界并停止。
```

> 最终意图：不是再给旧transport打补丁，而是将正常动作统一为一条直接设备调用链，同时保留最新源码的数值校验、真实数据准入和必要安全职责。
