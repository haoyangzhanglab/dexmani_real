# DexMani Real 策略部署、进程管理与数据同步深度审查

> 文档状态：只读架构审查与整改设计，不代表代码已经修复
>
> 审查基线：`167f15a5f76b798ea5a90e44fe3e478eecc266d2`
>
> ManiUniCon 对照基线：`85c6f2e32ecf9f2bed62d202b058c39623444686`
>
> 审查日期：2026-08-09
>
> 硬件边界：未连接 xArm、XHand、Quest 或 RealSense，未执行遥操作、回放、回零和标定入口

## 阅读导航

- 第1–4节：目标、证据标准、当前架构和结论摘要；
- 第5节：PD-01至PD-20的完整发现、成因、影响与修复要求；
- 第6–10节：深层根因、目标架构、IPC协议、schema迁移和ManiUniCon对照；
- 第11–12节：分批整改依赖与自动化验收矩阵；
- 第13–15节：实机验证边界、离线复现记录和最终放行条件；
- 第16–22节：围绕 ManiUniCon 核心机制的二次深审、MU-01至MU-16、移植边界和增量验收条件。

## 1. 文档目的

本文面向 DexMani Real 后续部署学习策略、VLA、diffusion policy、action-chunk policy、时序策略和多模态策略的需求，审查现有五进程架构在以下方面是否具备可安全扩展的基础：

- 模型加载、预热、就绪和故障隔离；
- multiprocessing 启动方式、进程监督和确定性停机；
- 多模态 observation 的因果快照、时间域和历史窗口；
- 策略输出的语义、时效、背压和 arm-hand 协调；
- 安全检查能否覆盖所有新策略，而不是只覆盖 VR teleop；
- 录制数据能否准确重建“策略看到了什么、产生了什么、硬件接受了什么”；
- 当前 schema v13、SharedStorage 和五进程边界应如何兼容演进。

本文不是一次通用机器人框架重构提案。默认继续保留：

1. xArm7、XHand、Quest、L515 的目标硬件组合；
2. Main、camera、VR、policy、arm、hand 的现有进程职责，其中 main 是薄监督进程；
3. arm ordered/bounded queue 和 hand latest-wins ring；
4. xArm Mode 6 固件平滑，不增加 arm-side interpolation；
5. policy 进程拥有录制时钟和 EpisodeRecorder；
6. sticky fault、seqlock 和硬件 SDK 进程隔离。

若部署的模型无法在五进程约束内实现可终止的推理隔离，是否增加独立 inference process 必须作为显式架构决策，而不能通过 daemon policy 内部偷偷创建子进程。

## 2. 审查范围与证据标准

### 2.1 主要源码范围

- `examples/real/vr_teleop_hand_record.py`
- `dexmani_real/policy/vr_teleop_policy.py`
- `dexmani_real/policy/loop_timing.py`
- `dexmani_real/shm/shared_storage.py`
- `dexmani_real/shm/ring_buffer.py`
- `dexmani_real/shm/robot_ring.py`
- `dexmani_real/robot/arm_loop.py`
- `dexmani_real/robot/hand_process.py`
- `dexmani_real/robot/types.py`
- `dexmani_real/sensor/camera_process.py`
- `dexmani_real/sensor/realsense.py`
- `dexmani_real/sensor/vr_receiver_process.py`
- `dexmani_real/recording/episode_recorder.py`
- `dexmani_real/recording/timestamp_buffer.py`
- `tests/`

### 2.2 对照范围

ManiUniCon 仅用于对照以下机制：

- 显式 `spawn`；
- Hydra 组件实例化和 SDK/模型对象的进程所有权；
- BasePolicy、observation wrapper 和 action wrapper；
- policy/robot ready 双向握手；
- observation horizon、共享内存 ring/queue 和多相机融合；
- action target timestamp、action chunk、real-time chunking 和 reset；
- 多进程 TimestampAlignedBuffer 录制和 episode 落盘。

ManiUniCon 不是必须复制的安全标准。其 robot-side interpolation、通用 Robot 生命周期和部分无超时同步不能直接用于 DexMani Real。

### 2.3 严重级别

| 等级 | 定义 |
|---|---|
| P0 | 可能使未经统一安全检查的策略输出进入硬件，或使危险旧动作在错误生命周期中执行；部署前必须阻断 |
| P1 | 可能造成故障漏报、停机不确定、动作过期、跨执行器错配、观测严重失配或数据语义损坏 |
| P2 | 限制模型能力、实时性能、可观测性、可复现性或部署工程质量 |
| P3 | 主要影响维护成本、文档一致性和开发体验 |

### 2.4 置信度标记

- **离线复现**：使用纯 Python、mock、共享内存或 HDF5 重现；
- **静态确认**：源码调用链可以确定行为，但实际固件/调度后果仍需实机或压力验证；
- **条件性风险**：当前 VR policy 未必触发，但按现有接口接入学习策略时会触发；
- **待实机确认**：不能从离线结果推断真实制动距离、固件行为或碰撞灵敏度。

### 2.5 与前序闭环审查的关系

本文聚焦策略部署运行时，不替代 `docs/xarm7_closed_loop_deep_review_167f15a.md` 中对硬件闭环的审查。以下前序问题是策略部署的上游依赖，必须并行修复：

- live replay绕过碰撞和workspace检查；
- Mode 6进入、恢复与后置状态确认；
- 软件DISARMED与控制器state 4的一致性；
- ESC、shutdown和阻塞producer的停止优先级；
- arm反馈读取失败时的source timestamp和故障升级；
- C24恢复后重发原故障目标；
- hand命令失败升级和qpos stale判据；
- `--no-hand` readiness依赖；
- schema v13的future-fill、camera freshness、RGB内部缺帧和过滤sidecar问题。

策略运行时不能用新的抽象掩盖这些硬件和数据问题。特别是ActionSafetyGate、action TTL和ObservationSnapshot只有在控制器生命周期与source freshness可信的前提下才能构成完整闭环。

## 3. 当前架构及其对策略部署的真实含义

### 3.1 当前五进程数据流

```text
camera ───────────────┐
VR ──────────────────┤
arm state ───────────┤
hand/tactile state ──┤
                      v
                 VR policy
              mapping + IK + safety
                 │          │
                 │          └── EpisodeRecorder
                 v
       arm queue       hand ring
          │                │
          v                v
       arm worker       hand worker
```

现有 policy 并不是一个可插拔模型接口，而是一个 1940 行的完整应用进程。它同时拥有：

- 键盘和音频交互；
- VR mapping；
- IK 和 collision planner；
- arm/hand 命令生成；
- 安全状态转换；
- 录制启停与 HDF5/视频写入；
- 传感器读取和状态打印；
- heartbeat。

因此，“把 VR mapping 换成 Torch 模型”并不是局部替换。模型推理会进入当前所有实时职责所在的同一控制线程。

### 3.2 当前 IPC 语义

| 通道 | 当前语义 | 优点 | 策略部署缺口 |
|---|---|---|---|
| arm state ring | seqlock、latest/get_last_k | 不传播 torn frame | helper 丢弃 ring seq/write time；失败帧 source freshness 有缺陷 |
| hand state ring | seqlock、latest/get_last_k | 固定结构、低延迟 | 无统一 age gate；与 arm 状态不成快照 |
| tactile ring | 稀疏、contact-only | 减少大块复制 | 可无限 forward-fill，缺少 age/seq/valid |
| camera ring | 大帧 seqlock、latest | 能保护 RGB/depth/PC 一致写入 | latest 一次复制全部模态；helper 丢弃 seq/write time；无历史 API |
| VR ring | seqlock、latest/get_last_k | 保留 remote/source/local recv 信息 | source clock 与 host clock未建立明确关系 |
| arm action queue | ordered、`maxsize=2` | 有意限制积压 | 无 observation ID、target time、TTL、policy epoch |
| hand command ring | latest-wins | 适合高频手指目标 | payload只有 qpos，无共同 action ID、时间和 ACK |

### 3.3 当前时间域

系统同时存在至少以下时间语义：

| 时间 | 来源 | 时间域 |
|---|---|---|
| arm/hand state `timestamp` | worker `time.monotonic()` | host monotonic seconds |
| ring slot timestamp | `time.monotonic_ns()` | host monotonic ns |
| VR `local_recv_ns` | receiver `time.monotonic_ns()` | host monotonic ns |
| VR `source_ts_ns` | Quest/HTS event | remote/source domain |
| camera header `timestamp` | RealSense `get_timestamp()` | device domain |
| `CameraFrame.host_time` | `time.time()` | host wall clock，且未进入 ring header |
| episode grid timestamp | TimestampAlignedBuffer synthetic grid | policy recording grid |
| action `created_monotonic_s` | 入队前 `time.monotonic()` | host monotonic seconds |

只有 host monotonic 数据可以直接跨进程比较。device、remote、wall-clock 和 synthetic grid 不能在缺少转换关系时直接相减。

## 4. 执行摘要

本专项审查确认：

- **1 项条件性 P0**：新策略没有强制统一安全网关；直接写底层 IPC 会绕过 VR policy 内嵌的 collision/workspace 检查。
- **12 组 P1**：包括 fork/daemon 限制、policy ready 缺失、推理阻塞实时协调、异常误报正常退出、卡死进程无法确认终止、无因果 observation、camera跨时钟域、无 action TTL、arm-hand 动作错配、chunk scheduler 缺失和不安全热重启。
- **7 组 P2**：包括策略接口缺失、历史窗口不足、选择性相机读取缺失、模型资源治理、训练部署元数据、配置解析和测试覆盖。

当前可安全支持的是“单步、低延迟、同线程、明确使用现有 VR 安全路径的策略”。当前不应直接放行：

- 任意直接写 `arm_action_q` 的学习策略；
- action chunk 整批入队；
- policy 热重启或热切换；
- 在当前 daemon policy 中启用多进程 DataLoader；
- 未经 warmup/输出验证就转 ARMED；
- 将 `action_arm_joint_sent` 当作硬件已执行标签；
- 将 latest 多模态读数组合视作同步 observation。

围绕 ManiUniCon 核心机制的第二轮逐行复审又确认了 16 项移植相关发现（MU-01至MU-16）。它们不改变上面的 PD-01至PD-20 计数，而是进一步解释为什么不能把参考实现的 `spawn + Hydra + Event + chunk queue` 当成完整部署方案。新增结论中，最关键的是：参考实现的 synchronized 握手会用“读取新 chunk 之前”的旧 `action` 判断执行完成；全部 chunk 迟到时会把最后一个旧预测重新赋予未来时间；history 不足时会静默复制最旧帧；多相机融合会丢失各相机时间戳；分布式录制没有 episode 提交屏障。这些机制若原样移植，会破坏 DexMani Real 已有的 hold-on-failure、seqlock、sticky fault、bounded queue 和单时钟录制不变量。

### 4.1 发现索引

| ID | 级别 | 发现 | 证据状态 |
|---|---|---|---|
| PD-01 | P0 | 缺少所有策略必须经过的统一动作安全网关 | 条件性风险、静态确认 |
| PD-02 | P1 | 默认fork与GPU/线程运行库不兼容 | 当前行为离线确认 |
| PD-03 | P1 | daemon policy阻止多进程推理和数据加载 | 离线复现 |
| PD-04 | P1 | 没有模型级policy ready和能力握手 | 静态确认 |
| PD-05 | P1 | 推理与实时协调、急停输入和录制共享线程 | 静态确认 |
| PD-06 | P1 | policy异常会被误报为正常Q退出 | 静态确认 |
| PD-07 | P1 | 卡死policy无法被确认终止，IPC可能提前关闭 | 离线复现 |
| PD-08 | P1 | 没有因果一致的多模态ObservationSnapshot | 静态确认 |
| PD-09 | P1 | camera时间戳不在可直接对齐的host monotonic域 | 静态确认 |
| PD-10 | P1 | action缺少observation因果链和TTL | 离线复现 |
| PD-11 | P1 | arm与hand不是一个可追踪的联合动作 | 离线复现 |
| PD-12 | P1 | action chunk没有安全调度边界 | 条件性风险、静态确认 |
| PD-13 | P1 | policy热重启/热切换没有epoch和排空协议 | 条件性风险、静态确认 |
| PD-14 | P2 | 缺少通用Policy、Observation和Action接口 | 静态确认 |
| PD-15 | P2 | 历史窗口容量不足且没有跨模态历史对齐 | 静态确认 |
| PD-16 | P2 | camera读取总是复制RGB、depth和point cloud | 静态确认 |
| PD-17 | P2 | 缺少模型资源治理和实时准入 | 静态确认 |
| PD-18 | P2 | 训练—部署闭环缺少模型和精确输入语义 | 静态确认 |
| PD-19 | P2 | 配置解析不适合策略插件和可复现实例 | 静态确认、部分离线复现 |
| PD-20 | P2 | 现有测试未覆盖策略部署故障模型 | 静态确认 |

## 5. 详细发现

## PD-01 — P0：缺少所有策略必须经过的统一动作安全网关

### 状态

条件性风险，静态确认。现有 canonical VR 路径具有安全检查，但未来策略接入没有结构性强制。

### 证据

VR policy 在自身内部完成：

- IK delta clamp；
- joint hard limit；
- arm connected；
- workspace segment；
- arm-hand transition collision。

对应代码：`dexmani_real/policy/vr_teleop_policy.py:1275-1303`。

通过检查后，policy 调用 `_safe_arm_queue_put()`：

- `dexmani_real/policy/vr_teleop_policy.py:1353-1357`。

但底层接口对其他生产者公开：

- `make_arm_action()`：`dexmani_real/shm/shared_storage.py:362-386`；
- `arm_action_q`：`dexmani_real/shm/shared_storage.py:175`。

arm worker 只做有限值、等价关节带和硬限位检查：

- `dexmani_real/robot/arm_loop.py:381-405`。

它不执行 planner 级 workspace segment、self collision 或 arm-hand transition collision。现有 live replay 已证明这种旁路是现实风险，而不是纯接口理论。

### 触发场景

新增模型代码执行以下任何一种做法：

```python
shared.arm_action_q.put(make_arm_action(shared, model_qpos))
```

或：

```python
write_hand_cmd(shared, model_hand_qpos)
```

### 影响

- 模型输出可能满足关节硬限位，但中间路径穿过自碰撞或桌面；
- arm 与 hand 分别合法，但组合 transition 发生碰撞；
- EEF action 转关节后离开 workspace；
- 旧模型、错误 normalization 或错误坐标系输出直接进入硬件边界。

### 根因

安全契约属于具体 `vr_teleop_policy.py`，而不是属于所有 policy output 的公共边界。

### 修复要求

引入强制、不可旁路的 `ActionSafetyGate`：

```text
PolicyOutput
→ ActionAdapter
→ ActionSafetyGate
→ CommandScheduler
→ arm/hand IPC
```

模型 backend 不应获得 SharedStorage 的写权限。所有 entry point 也应逐步迁移到同一安全发布接口。

### 验证

- 模型输出 NaN、错误shape、超限、碰撞、离开workspace：queue/ring无写入；
- SafetyGate异常：发布 coordinated hold，达到阈值后FAULT；
- 尝试从 plugin直接取得 raw queue：接口层不可用或显式拒绝；
- 对 replay、keyboard、calibration 做生产者清单测试，确保没有遗漏旁路。

## PD-02 — P1：默认 fork 与未来 GPU/线程运行库不兼容

### 状态

离线确认当前 effective start method 为 `fork`。

### 证据

canonical 入口直接使用：

- `import multiprocessing as mp`；
- `mp.Process(...)`；

但未调用 `set_start_method()` 或显式 context：

- `examples/real/vr_teleop_hand_record.py:17-18`；
- `examples/real/vr_teleop_hand_record.py:130-140`。

离线输出：

```text
configured_start None
effective_start fork
```

### 影响

如果父进程在 fork 前导入或初始化 Torch、CUDA、OpenMP、MKL 或其他后台线程运行库，子进程可能继承：

- 不可复用的 CUDA context；
- 已锁定但不存在持有线程的 mutex；
- 不安全的线程池状态；
- 模型或显存对象的写时复制视图。

问题常表现为启动偶发挂起，而不是稳定异常，难以通过常规单测发现。

### 根因

当前入口是在纯 NumPy/SciPy/SDK 五进程环境中形成的，没有把模型运行时作为进程启动约束。

### 修复要求

- 在创建 SharedStorage、Queue、Value、Event 之前选择统一 `spawn` context；
- 所有 multiprocessing primitive 从同一 context 创建；
- 目标函数保持模块顶层可导入；
- main 只传递配置和可重建的IPC描述，不传模型实例；
- 模型和CUDA只能在policy子进程中初始化。

### 对照

ManiUniCon 在 `main.py:160` 显式使用 `spawn`，这一机制可借鉴，但其余进程监督语义不能直接照搬。

## PD-03 — P1：daemon policy 阻止多进程推理和数据加载

### 状态

离线复现。

### 证据

全部 canonical worker 使用 `daemon=True`：

- `examples/real/vr_teleop_hand_record.py:131-138`。

dummy daemon worker 创建子进程得到：

```text
AssertionError: daemonic processes are not allowed to have children
```

### 影响

以下常见模型能力不可直接使用：

- PyTorch DataLoader `num_workers > 0`；
- 独立图像预处理 worker；
- 模型server/client子进程；
- 部分编译、采样或异步环境组件。

### 修复要求

- policy 改为非 daemon，并由 main 完整监督和回收；
- 初期五进程方案要求模型 `num_workers=0`；
- 如需要强故障隔离，显式引入第六 inference process；
- 不依赖 daemon 语义作为停机保障，停机必须有确认协议。

## PD-04 — P1：没有模型级 policy ready 和能力握手

### 状态

静态确认。

### 证据

SharedStorage ready events 只有：

- arm；
- hand；
- camera；
- VR。

见 `dexmani_real/shm/shared_storage.py:193-196`。

main 等这些事件后立即 ARMED：

- `examples/real/vr_teleop_hand_record.py:148-176`。

policy 自己又硬编码等待所有四个设备：

- `dexmani_real/policy/vr_teleop_policy.py:471-483`。

### 影响

- 模型还在加载/compile/warmup，系统已经显示 ARMED；
- 模型初始化超过 policy heartbeat timeout 时被误判为运行期hang；
- 仅需要 proprioception 的策略仍被VR/camera readiness阻塞；
- `--no-hand` 之类可选能力无法正确贯穿依赖图；
- checkpoint、normalizer或输出shape错误直到运行阶段才暴露。

### 根因

ready 表达“设备producer已经启动”，没有表达“所选策略及其必需输入已经可生成安全动作”。

### 修复要求

新增：

```text
policy_loaded
policy_warmed_up
policy_ready
policy_manifest_hash
required_sensor_mask
```

`policy_ready` 只在以下检查全部通过后设置：

1. checkpoint与normalizer hash验证；
2. 模型device/dtype就绪；
3. dummy warmup成功；
4. 输出shape、finite和action representation合格；
5. required sensor均有新鲜有效帧；
6. SafetyGate、scheduler和recorder初始化完成；
7. 旧queue/chunk已清空；
8. 当前 policy epoch 已发布。

Main 必须保持 DISARMED，直到 policy ready。

## PD-05 — P1：推理与实时协调、急停输入和录制共享一个线程

### 状态

静态确认。

### 证据

policy main loop 在同一线程依次执行：

- heartbeat；
- RateManager；
- recorder poll；
- keyboard；
- 多ring读取；
- mapping/retarget/IK/collision；
- queue/ring写入；
- recorder add_frame。

入口：`dexmani_real/policy/vr_teleop_policy.py:568-1401`。

### 影响

一次慢推理或C/CUDA hang会同时暂停：

- policy heartbeat；
- ESC/Q键处理；
- sensor freshness判断；
- 新hold动作；
- recording时钟和写入；
- 模型deadline统计。

在模型执行时间接近或超过 62.5 ms 的16 Hz周期时，系统没有明确的迟到策略。当前 RateManager 会重锚，但没有把“动作迟到”变成动作失效。

### 修复要求

五进程内的最低方案：

```text
PolicyCoordinator thread
  - 16 Hz deadline
  - IPC唯一读写者
  - safety/hold/fault
  - recorder owner
  - heartbeat owner
       ↕ bounded single-slot mailbox
Inference execution unit
  - immutable snapshot in
  - raw output out
  - no SharedStorage write access
```

mailbox 必须是单槽或严格有界，旧 observation 应被覆盖或显式丢弃，不能形成推理 backlog。

如果 backend 无法被线程安全地超时和回收，必须使用独立进程；线程不能解决永久卡死的C/CUDA调用。

## PD-06 — P1：policy异常会被误报为正常Q退出

### 状态

静态确认。

### 证据

policy `finally` 无论正常还是异常都会：

```python
shared.is_running.value = False
```

见 `dexmani_real/policy/vr_teleop_policy.py:1403-1408`。

supervisor 首先检查 `is_running=False`，并固定记录为：

```text
is_running=False (Q key)
normal_exit=True
```

见 `dexmani_real/shm/shared_storage.py:879-885`。

### 影响

模型 OOM、tensor shape错误、checkpoint错误、CUDA异常和预处理崩溃可能被事故记录为正常退出，削弱故障统计和自动化放行。

### 修复要求

分离控制信号：

```text
shutdown_requested      Main-owned
quit_requested          user clean exit
estop_request           emergency
fault_latched           sticky
fault_owner             policy/arm/hand/camera/vr/main
fault_code              structured enum
fault_message           bounded text or logged correlation ID
```

supervisor 优先级应为：

```text
estop/fault
→ worker death/exit code
→ heartbeat/progress timeout
→ explicit quit
→ main shutdown
```

## PD-07 — P1：卡死policy无法被确认终止，IPC可能提前关闭

### 状态

离线复现。

### 证据

`shutdown_processes()`：

1. join 5秒；
2. `terminate()`；
3. join 1秒；
4. 不检查第二次join后是否仍存活；
5. 立即 `shared.close()`。

见 `dexmani_real/shm/shared_storage.py:516-533`。

忽略 SIGTERM 的 dummy policy 复现结果：

```text
shutdown_elapsed_s 6.01
alive_after_shutdown True
exitcode None
```

但函数已经打印 `stuck-policy=term` 并关闭共享内存。

### 影响

- 卡死的模型进程继续访问已关闭/已unlink的IPC；
- 父进程误认为停机完成；
- 新session可能在旧policy仍存活时启动；
- HDF5线程、CUDA context或临时文件状态不可确定。

### 修复要求

```text
cooperative stop
→ bounded join
→ SIGTERM
→ bounded join + is_alive recheck
→ SIGKILL
→ final join + confirmed exitcode
→ close queues/rings
→ unlink shared memory
```

不能确认退出时必须将session标记为故障，并阻止同prefix的新session自动启动。

## PD-08 — P1：没有因果一致的多模态ObservationSnapshot

### 状态

静态确认。

### 证据

当前每个policy tick顺序读取：

```text
arm → VR → camera → hand → tactile
```

见 `dexmani_real/policy/vr_teleop_policy.py:879-918`。

这些都是各自的 `latest`，不共享 anchor timestamp，也不保证相互接近。

公共helper还丢弃ring元数据：

- arm：`dexmani_real/shm/shared_storage.py:353-359`；
- hand：`dexmani_real/shm/shared_storage.py:389-395`；
- camera：`dexmani_real/policy/vr_teleop_policy.py:1472-1480`。

### 当前freshness不对称

| 模态 | 当前检查 |
|---|---|
| arm | `timestamp` age ≤ 0.5 s、finite、connected |
| VR | `local_recv_ns` 与 stale threshold |
| hand | connected/error/qpos_stale，缺统一age |
| tactile | contact-only sparse ring，缓存可无限复用 |
| camera | 无可靠source freshness gate，heartbeat不能证明有新帧 |

### 影响

- 视觉可能对应旧机械臂姿态；
- hand state与arm state来自不同transition阶段；
- observation window混入future/过旧帧；
- 训练数据表面同一行，实际cross-modal skew不可恢复；
- model regression难以区分网络问题还是同步问题。

### 修复要求

引入不可变快照：

```text
ObservationSnapshot
  observation_id
  policy_epoch
  anchor_monotonic_ns
  arm: payload, source_ns, publish_ns, seq, age_ns, valid
  hand: ...
  tactile: ...
  camera: device_ns, host_capture_ns, publish_ns, seq, age_ns, valid
  vr: source_ns, local_recv_ns, publish_ns, seq, age_ns, valid
  max_cross_modal_skew_ns
  required_sensor_valid_mask
```

选帧必须 causal：对每个模态选择 `timestamp <= anchor` 的最新帧。缺失时标 invalid/hold，不得使用 anchor 之后的future sample回填。

## PD-09 — P1：camera时间戳不在可直接对齐的host monotonic域

### 状态

静态确认。

### 证据

RealSense reader生成：

- device timestamp：`depth_frame.get_timestamp() * 1e-3`；
- `host_time = time.time()`。

见 `dexmani_real/sensor/realsense.py:481-523`。

camera header只保存device timestamp：

- `dexmani_real/shm/ring_buffer.py:559-574`。

ring slot本身有host monotonic write timestamp，但policy camera helper丢弃它。

### 影响

camera header timestamp不能直接与arm/hand monotonic timestamp相减；device重启、clock reset或wall-clock调整会破坏推断。

### 修复要求

camera producer应同时发布：

- `device_capture_ts`；
- `host_receive_monotonic_ns`；
- `ring_publish_monotonic_ns`；
- `frame_number`；
- `camera_generation`。

在线跨模态对齐使用host monotonic；device timestamp用于相机内部帧间隔和丢帧诊断。

## PD-10 — P1：action缺少observation因果链和TTL

### 状态

离线复现。

### 证据

arm action只包含：

- `qpos`；
- `command_seq`；
- `created_monotonic_s`；
- `is_hold`。

见 `dexmani_real/shm/shared_storage.py:362-386`。

`created_monotonic_s` 是推理结束后、准备入队时生成，不能表示observation产生时间或inference age。

arm metadata parser只规范化时间，不检查最大年龄：

- `dexmani_real/robot/arm_loop.py:64-85`。

离线构造60秒前的action，parser仍返回原seq和timestamp，没有拒绝。

### 影响

- 慢推理结果基于旧场景但仍可执行；
- queue backlog中的旧命令仍合法；
- policy restart后旧epoch动作无法识别；
- action chunk迟到项不能被丢弃；
- 录制只能测queue latency，不能测observation-to-action latency。

### 修复要求

动作envelope至少包含：

```text
policy_epoch
action_id
observation_id
observation_anchor_ns
inference_started_ns
inference_finished_ns
created_ns
target_ns
valid_until_ns
representation
coordinate_frame
units
arm_target
hand_target
is_hold
```

arm和hand worker都必须拒绝：

- epoch不一致；
- action ID倒退或异常重复；
- `now > valid_until_ns`；
- observation age超限；
- representation/frame/shape不匹配。

对单步16 Hz策略，TTL应从observation anchor计算，而不是从queue put计算。具体阈值必须由实测p99 latency和控制周期确定。

## PD-11 — P1：arm与hand不是一个可追踪的联合动作

### 状态

离线复现。

### 证据

arm是ordered queue，hand是latest-wins ring。hand payload只有：

```text
qpos_cmd(12)
```

见 `dexmani_real/shm/shared_storage.py:142-146`。

离线写入两组动作后观察到：

```text
arm queue下一条仍为第1组
hand ring已经为第2组
```

hand ring的sequence只是该ring独立写序号，不是跨执行器共同action ID。

### 触发场景

- arm queue出现短时backlog；
- xArm进入C24 recovery；
- arm SDK调用变慢；
- policy产生chunk或burst；
- hand bus仍正常接受latest目标。

### 影响

- arm执行旧目标，hand执行新目标；
- policy做过的联合collision transition不再对应实际组合；
- 数据集无法知道同一行arm/hand命令是否属于同一动作版本；
- 无法测量arm-hand apply skew。

### 修复要求

- arm和hand命令携带共同 `action_id/policy_epoch/target_ns`；
- worker分别回显 received、SDK accepted 和 applied/inferred seq；
- recorder记录两侧ACK；
- 超过允许apply skew时进入coordinated hold；
- arm worker拒绝后，系统必须阻止对应hand动作继续被视为成功联合动作。

不要求两个SDK绝对同时执行，但必须可关联、可测量、可限界。

## PD-12 — P1：action chunk没有安全调度边界

### 状态

条件性风险，静态确认。

### 当前约束

arm queue `maxsize=2` 是有意的安全背压，不应扩大。hand ring只保留latest。

### 错误接入方式

- 把整个action chunk依次塞入arm queue；
- 为容纳chunk扩大queue；
- 在arm worker中增加插值；
- 每次新推理直接覆盖正在执行chunk但不处理时间重叠。

### 影响

- queue长度变成不可见执行延迟；
- 旧chunk在新observation之后继续执行；
- arm/hand chunk位置错配；
- arm-side插值与Mode 6固件平滑叠加。

### 修复要求

chunk保留在policy-side本地scheduler：

```text
timestamped model chunk
→ discard expired prefix
→ reconcile overlap with previous chunk
→ at each control grid select one target
→ revalidate against latest measured state
→ publish one paired action
```

每个元素包含目标时间和TTL。arm queue继续保持2；不增加arm-side interpolation。

## PD-13 — P1：policy热重启/热切换没有epoch和排空协议

### 状态

条件性风险，静态确认。

### 证据

- ready events设置后不会表示generation变化；
- arm queue可能保留旧动作；
- hand ring保留最后目标；
- shared command seq跨policy，但没有policy identity；
- model recurrent state/chunk没有统一reset hook；
- 当前supervisor采用整体fail-stop，不支持单worker重启。

### 影响

若未来为了提高可用性只重启policy，新策略可能读取旧ring帧、继承旧hand目标或让arm继续消费旧queue动作。

### 修复要求

在完成以下协议前禁止live hot restart：

```text
transition to DISARMED + controller stop confirmed
→ increment policy_epoch
→ drain arm queue/home result queue
→ publish measured arm/hand hold
→ clear local chunks/recurrent state
→ reload/warmup
→ validate fresh producer generations
→ policy_ready
→ explicit ARMED
```

FAULT期间不得自动热重启策略并恢复运动。

## PD-14 — P2：缺少通用Policy、Observation和Action接口

### 状态

静态确认。

### 证据

`dexmani_real/policy/__init__.py` 只导出 `PolicyConfig` 和 `policy_loop`。现有 `RobotState`/`RobotAction` 主要服务录制，且 `RobotState.timestamp` 在组装时使用 `time.perf_counter()`，不是observation anchor：

- `dexmani_real/policy/vr_teleop_policy.py:1679-1818`。

### 修复建议

最小接口：

```python
class PolicyBackend:
    def load(self) -> PolicyManifest: ...
    def warmup(self, observation_spec: ObservationSpec) -> None: ...
    def reset(self, reason: ResetReason) -> None: ...
    def infer(self, observation: ObservationSnapshot) -> PolicyOutput: ...
    def close(self) -> None: ...
```

相关类型：

```text
ObservationSpec
PolicyManifest
ObservationSnapshot
PolicyOutput
SafeJointAction
CommandAck
ResetReason
```

backend只能接收immutable snapshot并返回raw output，不能拥有SharedStorage或硬件对象。

## PD-15 — P2：历史窗口容量不足且没有跨模态历史对齐

### 状态

静态确认。

默认ring容量：

| 流 | maxlen | 约30 Hz覆盖 |
|---|---:|---:|
| camera | 5 | 167 ms |
| arm state | 8 | 267 ms |
| hand state | 8 | 267 ms |
| tactile | 8 | 267 ms，但稀疏写入 |
| VR | 8 | 取决于Quest发送率 |

定义：`dexmani_real/shm/shared_storage.py:45-50`。

对于0.5–2秒 observation horizon 不足。robot ring虽有 `get_last_k()`，camera没有对等历史接口，各流也没有按共同anchor进行nearest/causal选择。

### 修复建议

- `required_capacity = ceil(obs_horizon_s * source_hz) + jitter_margin`；
- startup时验证ring容量满足所选ObservationSpec；
- camera增加seqlock验证的history读取；
- policy侧按anchor构建固定长度window；
- 每个历史位置保存valid/age/source seq；
- 窗口不足时pad策略必须显式，例如repeat-first并标invalid，不能静默future-fill。

## PD-16 — P2：camera读取总是复制RGB、depth和point cloud

### 状态

静态确认。

`CameraRingBuffer.read_latest()` 无论消费者需要什么，都会复制：

- RGB；
- depth；
- fixed-size point cloud。

见 `dexmani_real/shm/ring_buffer.py:416-519`。

在16 Hz policy中，即使未录制或策略只需要RGB，也会产生额外内存带宽和分配。writer竞争时，reader在最终seqlock检查失败前已经完成大块复制。

### 修复建议

- ObservationSpec声明所需模态；
- 提供经过同一seqlock前后验证的选择性复制；
- 不向模型返回共享内存零拷贝view，以免producer覆盖；
- 记录camera copy/preprocess/H2D各阶段p95/p99。

## PD-17 — P2：缺少模型资源治理和实时准入

### 状态

静态确认。

当前没有统一配置或观测：

- CPU affinity/nice；
- Torch intra/inter-op线程数；
- CUDA device和显存预算；
- camera pointcloud GPU与policy GPU的所有权；
- inference p50/p95/p99/max；
- preprocessing、H2D、safety和enqueue耗时；
- deadline miss、expired action、dropped observation；
- GPU OOM、compile和warmup阶段。

### 修复建议

策略manifest声明：

```text
device
dtype
max_memory_mb
control_hz
max_inference_ms
observation_horizon
action_horizon
required_sensors
threading policy
```

放行不应只看平均值。至少要求在录制开启、camera/PC开启和背景系统负载下测量p99控制周期与action age。

## PD-18 — P2：训练—部署闭环缺少模型和精确输入语义

### 状态

静态确认。

schema v13已经保存相机K、extrinsics、depth scale和部分teleop配置，这是应保留的优点。但学习策略还需要：

- policy名称、版本和commit；
- checkpoint内容hash；
- normalization statistics hash；
- ObservationSpec/ActionSpec版本；
- resize/crop/color order/depth unit；
- model device/dtype；
- policy epoch；
- observation seq/age/skew/valid；
- raw model output；
- adapted joint candidate；
- safety-filtered command；
- arm/hand received/accepted/applied ACK。

当前 `action_arm_joint_sent` 仍是入队意图，不是硬件执行证明。

### 修复建议

使用兼容添加字段，不改变旧v13数据集含义。读者对新字段可选读取，旧episode保持可读。建议下一schema版本明确：

```text
observation_*       策略实际快照
policy_output_*     raw model output
action_safe_*       SafetyGate输出
command_*           发布到worker的联合动作
arm_ack_*           xArm SDK接受/时序
hand_ack_*          XHand SDK接受/时序
```

不建议保存归一化后的完整大图tensor作为唯一证据；应保存raw传感器数据和可复现preprocess manifest。必要时可额外保存低成本的输入hash或抽样tensor用于一致性测试。

## PD-19 — P2：配置解析不适合策略插件和可复现实例

### 状态

静态确认，部分已离线复现。

当前配置：

- JSON原地修改module singleton；
- 不重新运行dataclass `__post_init__`；
- nested dataclass不支持JSON；
- CLI默认值可能静默覆盖JSON；
- 没有policy backend、checkpoint、normalizer和observation/action spec配置。

非法 `control_hz=-1` 可通过JSON loader写入singleton。

### 修复建议

- 解析为新的不可变配置对象；
- 完整校验后一次性发布；
- 固定优先级 `explicit CLI > config file > defaults`；
- 启动打印resolved config及hash；
- 录制同一resolved manifest；
- 不让policy plugin从全局singleton读取隐式参数。

可借鉴Hydra式实例化边界，但不要求引入Hydra本身；关键是显式配置和接口，不是配置框架品牌。

## PD-20 — P2：现有测试未覆盖策略部署故障模型

### 状态

静态确认。

现有90个测试覆盖IK、collision、homing helper、timestamp buffer、录制字段和部分teleop响应，但未覆盖：

- spawn下的完整五进程启动；
- daemon/DataLoader限制；
- policy加载和warmup失败；
- inference timeout/hang/OOM；
- stuck process的SIGTERM→SIGKILL升级；
- causal multi-modal snapshot；
- action TTL和policy epoch；
- arm-hand共同action ID；
- action chunk过期和重叠；
- policy异常必须报告FAULT；
- exact model input与episode round-trip。

现有测试全部通过与本文问题并不矛盾。

## 6. 深层根因分析

### 6.1 安全边界与具体policy实现耦合

现有系统的安全性主要来自 VR policy内部的细致检查，而不是来自所有动作生产者必须遵守的协议。只要新增一种producer，就必须人工复制这些检查。live replay已经展示了复制遗漏的实际后果。

根本修复不是为每个模型再写一份collision gate，而是把SafetyGate提升为policy runtime的强制公共阶段。

### 6.2 process health、model health和device health混用

单个heartbeat只能说明某段代码最近写过时间值，不能同时证明：

- process仍被调度；
- 模型最近一次推理成功；
- SDK连接正常；
- 传感器产生新数据；
- command被硬件接受。

需要至少分为：

| 健康层 | 证明内容 | 示例信号 |
|---|---|---|
| process | 协调循环仍被调度 | coordinator heartbeat、pid、exitcode |
| inference | 模型在推进且输出有效 | inference seq、start/end、deadline miss |
| IPC | producer/consumer在交换新帧 | ring seq、queue age、generation |
| device | 硬件/传感器真实健康 | connected、SDK code、source freshness |
| safety | 动作通过安全网关 | reject reason、safe action ID |

### 6.3 “同一policy tick”被误当作“同一物理时刻”

顺序latest read只保证读取操作发生在同一循环，不能保证数据采样发生在同一时刻。schema grid也只保证行索引规则，不证明各模态物理同步。

必须建立anchor、因果选帧和age/skew，而不是依赖循环顺序。

### 6.4 动作阶段语义不足

当前存在：

```text
generated
→ queued
→ received
→ SDK returned success
→ hardware feedback follows
```

但数据命名容易把queued称作sent，并且hand缺少后三阶段关联。学习策略部署需要将阶段显式化，否则无法分析模型质量、系统延迟和硬件跟踪误差各自贡献。

### 6.5 控制线程承担了慢路径

录制I/O、键盘等待、规划、未来模型推理都在policy线程内。当前16 Hz VR IK尚能工作，不代表相同结构适合50–300 ms的VLA或diffusion推理。

策略部署必须先定义deadline和迟到行为，再选择线程/进程结构。

### 6.6 缺少session/policy generation

共享内存名称、ready event、queue和ring只描述对象，不描述“这条数据属于哪次policy实例”。整体fail-stop掩盖了该问题；一旦引入热重启、模型切换或自动恢复，旧状态就会跨生命周期泄漏。

## 7. 推荐目标架构

### 7.1 保持五进程的首选结构

```text
                         Main Supervisor
             spawn / readiness / fault / shutdown
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
 camera/VR producers      arm/hand workers       Policy Process
       │                      ▲                      │
       └────── SharedStorage ─┴──────────────────────┘
                                                     │
                                           ObservationAssembler
                                                     │
                                           immutable causal snapshot
                                                     │
                                             PolicyBackend
                                                     │ raw output
                                             ActionAdapter
                                                     │ joint candidate
                                         Mandatory SafetyGate
                                                     │ safe paired action
                                          Timestamped Scheduler
                                             │               │
                                      arm queue          hand ring
                                             │               │
                                             └──── ACKs ──────┘
                                                     │
                                               EpisodeRecorder
```

### 7.2 PolicyCoordinator职责

PolicyCoordinator是policy进程内唯一允许访问SharedStorage的组件：

- 按16 Hz产生anchor；
- 构建ObservationSnapshot；
- 更新process heartbeat；
- 检查inference progress；
- 处理键盘/外部控制信号；
- 执行ActionAdapter和SafetyGate；
- 调度chunk并发布一组paired action；
- 记录snapshot/output/safe action/ACK；
- 在缺帧、迟到、reject时发布hold或FAULT。

### 7.3 PolicyBackend职责

PolicyBackend只负责：

- load；
- warmup；
- reset；
- infer；
- close。

它不得：

- 写SharedStorage；
- 转换SafetyState；
- 调用SDK；
- 直接控制EpisodeRecorder；
- 自行决定忽略invalid observation；
- 隐式使用module singleton参数。

### 7.4 ActionSafetyGate职责

按顺序建议：

1. policy epoch、action ID和TTL；
2. representation、frame、shape、dtype、finite；
3. observation required-valid和age/skew；
4. action adapter转换；
5. arm/hand hard limits；
6. delta/velocity/acceleration envelope；
7. workspace segment；
8. self collision和arm-hand transition collision；
9. connected/error/safety state复检；
10. paired command封装。

任何异常都必须fail closed。

### 7.5 Inference执行模式

#### 模式A：同步单步模型

适用条件：在最坏负载下，完整preprocess+infer+postprocess p99显著小于控制周期，并且调用可可靠返回。

优点：简单、snapshot和output天然一一对应。

缺点：模型慢调用会阻塞Coordinator。

#### 模式B：policy内有界推理线程

Coordinator持续16 Hz，inference thread通过单槽mailbox工作。

要求：

- 只保留最新待推理snapshot；
- 每个output携带observation ID；
- output过期即丢弃；
- heartbeat由Coordinator维护；
- 有独立inference progress timestamp。

缺点：无法强制终止永久卡死的C/CUDA调用。

#### 模式C：独立inference process

适用于VLA、大模型、编译模型或需要强故障隔离的backend。该模式会改变五进程架构，必须显式批准并增加：

- 独立heartbeat/readiness；
- bounded observation/output IPC；
- GPU生命周期；
- fault传播；
- shutdown/kill确认。

### 7.6 迟到策略

迟到不能只记录warning。建议：

| 条件 | 行为 |
|---|---|
| 本tick无新output，但最后safe target仍在短hold窗口 | 重发/保持最后safe hold |
| output对应旧observation且已过TTL | 丢弃，不发布 |
| 连续deadline miss达到阈值 | RUNNING→ARMED或FAULT，取决于配置和硬件状态 |
| inference无progress超过hard timeout | sticky FAULT |
| required sensor stale | coordinated hold；持续超时后FAULT |
| arm/hand ACK skew超限 | coordinated hold并记录reject reason |

具体阈值必须来自实机延迟测量，不能仅凭静态分析确定。

## 8. 推荐IPC协议

### 8.1 ObservationSnapshot示意

```python
@dataclass(frozen=True)
class ObservationSnapshot:
    observation_id: int
    policy_epoch: int
    anchor_monotonic_ns: int
    arm: TimedSample
    hand: TimedSample | None
    tactile: TimedSample | None
    camera: TimedSample | None
    vr: TimedSample | None
    max_skew_ns: int
    required_valid_mask: int
```

`TimedSample`至少含：

```text
payload
source_timestamp
host_receive_timestamp
publish_timestamp
sequence
generation
age
validity/fill reason
```

### 8.2 JointActionEnvelope示意

```python
@dataclass(frozen=True)
class JointActionEnvelope:
    policy_epoch: int
    action_id: int
    observation_id: int
    observation_anchor_ns: int
    inference_started_ns: int
    inference_finished_ns: int
    target_ns: int
    valid_until_ns: int
    arm_qpos: np.ndarray
    hand_qpos: np.ndarray | None
    is_hold: bool
```

跨进程结构仍应实现为NumPy dtype/scalar或其他显式结构，不传递任意可变Python对象图。示意dataclass用于文档和边界验证，不意味着直接把对象放入SharedStorage。

### 8.3 ACK语义

```text
GENERATED
→ SAFETY_ACCEPTED
→ QUEUED/PUBLISHED
→ WORKER_RECEIVED
→ SDK_ACCEPTED
→ FEEDBACK_TRACKING
→ CONVERGED（仅需要操作级完成时）
```

普通Mode 6命令不需要逐条CONVERGED，但至少需要arm/hand各自的WORKER_RECEIVED和SDK_ACCEPTED语义。

## 9. 数据与schema迁移建议

### 9.1 兼容原则

- 不改变schema v13现有字段含义；
- 新字段采用additive schema升级；
- reader对新字段可选读取；
- old episode保持可读；
- 不把旧 `action_arm_joint_sent` 重新解释为SDK accepted；
- schema marker和工具链同步更新。

### 9.2 建议新增字段组

#### Observation provenance

```text
observation_id
policy_epoch
observation_anchor_ns
arm_source_ns / arm_publish_ns / arm_seq / arm_age_s
hand_source_ns / hand_publish_ns / hand_seq / hand_age_s
camera_device_ts / camera_receive_ns / camera_publish_ns / camera_seq / camera_age_s
vr_source_ns / vr_receive_ns / vr_publish_ns / vr_seq / vr_age_s
cross_modal_skew_s
observation_valid_mask
```

#### Policy telemetry

```text
policy_output_valid
policy_preprocess_time_ms
policy_inference_time_ms
policy_postprocess_time_ms
policy_deadline_miss
policy_output_age_s
policy_output_raw_*（按representation）
```

#### Command/ACK

```text
action_id
action_target_ns
action_valid_until_ns
action_safe_arm_joint
action_safe_hand_joint
arm_received_action_id
arm_sdk_accepted_action_id
arm_sdk_accepted_ns
hand_received_action_id
hand_sdk_accepted_action_id
hand_sdk_accepted_ns
arm_hand_apply_skew_s
```

### 9.3 元数据manifest

```text
policy_name
policy_version
policy_git_commit
checkpoint_sha256
normalizer_sha256
observation_spec_version
action_spec_version
preprocess_manifest_json
device
dtype
control_hz
inference_mode
resolved_config_hash
```

## 10. ManiUniCon对照结论

### 10.1 可借鉴

| 机制 | 借鉴价值 |
|---|---|
| 显式spawn | 避免fork继承Torch/CUDA线程状态 |
| BasePolicy | 把模型生命周期从业务入口中分离 |
| obs wrapper | 明确模型输入转换边界 |
| act wrapper | 明确模型输出representation和时间化 |
| policy_ready | 模型生成动作后再进入执行阶段 |
| observation horizon | 原生支持时序模型 |
| target timestamp | 支持chunk和迟到动作处理 |

### 10.2 不可照搬

- robot-side pose interpolation：DexMani Real使用xArm Mode 6固件平滑，双重插值可能造成overshoot/C24；
- 无超时的policy/robot Event握手：任一侧死亡可能永久等待；
- 以通用Robot对象弱化xArm/XHand进程隔离；
- 用通用框架替换sticky error和安全状态机；
- 把TimestampAlignedBuffer的future backfill继续扩散到在线observation。

### 10.3 DexMani Real应保留的优势

- seqlock torn-read防御；
- ordered/bounded arm queue；
- latest-wins hand ring；
- sticky error latch；
- monotonic heartbeat；
- policy单时钟录制所有权；
- IK candidate、等价带、delta clamp和hold-on-failure；
- arm-hand联合碰撞检查；
- C22/C31立即FAULT；
- SDK只在相应worker中使用；
- 不增加arm-side interpolation。

## 11. 修复批次与依赖

### Batch 0：部署禁令与接口封口

处理：PD-01、PD-12、PD-13。

- 明确禁止模型直接写raw queue/ring；
- 暂不允许live policy hot restart；
- 暂不允许action chunk整批入队；
- live replay继续保持禁用，直到全路径预检修复。

退出条件：所有动作生产者清单完成，新增policy模板无法取得底层写接口。

### Batch 1：进程生命周期

处理：PD-02至PD-07。

1. 统一spawn context；
2. policy改为非daemon；
3. policy ready和required sensor capability；
4. 明确fault/quit/shutdown信号；
5. 分级terminate/kill并确认退出；
6. SharedStorage在所有子进程退出后才close/unlink；
7. create失败事务回滚与session instance lock。

退出条件：fake model load失败、hang、SIGTERM ignore、worker death、normal quit均得到正确状态和退出码。

### Batch 2：ObservationSnapshot与时钟

处理：PD-08、PD-09、PD-15、PD-16。

1. producer保留source/receive/publish timestamp和seq；
2. camera加入host monotonic capture/receive时间；
3. read helper不再丢弃ring metadata；
4. causal snapshot builder；
5. configurable history capacity；
6. modality-selective camera copy；
7. age/skew/valid统一门控。

退出条件：离线多速率producer测试证明无future sample、skew计算正确、缺帧行为显式。

### Batch 3：动作协议与联合调度

处理：PD-10至PD-13。

1. policy epoch/action ID/observation ID；
2. target time和TTL；
3. arm/hand共同动作封装；
4. worker ACK；
5. policy-side chunk scheduler；
6. expired action拒绝；
7. coordinated hold。

退出条件：60秒旧动作、旧epoch动作、迟到chunk和arm-hand错配均被拒绝或限界。

### Batch 4：Policy Runtime抽象

处理：PD-05、PD-14、PD-17、PD-19。

1. PolicyBackend；
2. ObservationSpec/ActionSpec；
3. ActionAdapter和mandatory SafetyGate；
4. 同步/线程/独立进程三种推理模式；
5. resolved config和manifest；
6. inference/resource metrics。

退出条件：至少一个dummy learned policy通过完整生命周期、deadline、fault和safe action测试。

### Batch 5：数据闭环

处理：PD-18、PD-20。

1. additive schema；
2. exact snapshot/output/safe command/ACK记录；
3. reader/quality/replay兼容；
4. model manifest；
5. episode round-trip和旧格式读取；
6. training-deployment parity测试。

退出条件：离线重新运行preprocess能重建相同模型输入，所有时间/valid/action语义可解释。

## 12. 自动化测试矩阵

### 12.1 进程与启动

| 场景 | 预期 |
|---|---|
| spawn完整启动，所有backend lazy import | 无fork依赖，所有worker ready |
| checkpoint不存在 | 保持DISARMED，policy_ready不置位，structured fault |
| warmup输出NaN/错误shape | 保持DISARMED |
| required camera缺失 | 视觉策略不ready；非视觉策略可按capability继续 |
| policy init超过普通heartbeat timeout | startup phase有独立deadline，不误判运行期hang |
| inference Python异常 | FAULT，不是normal Q |
| inference OOM | FAULT，记录owner/code，安全停机 |
| inference永久hang | coordinator hold/FAULT，main最终SIGKILL并确认退出 |
| worker忽略SIGTERM | shutdown不得在worker存活时关闭IPC |
| normal Q | normal exit，fault=false |
| ESC与Q同时发生 | estop优先 |

### 12.2 Observation同步

| 场景 | 预期 |
|---|---|
| arm 30 Hz、camera 30 Hz、policy 16 Hz异相 | 每个snapshot选择anchor之前最近帧 |
| producer在读取中写新帧 | 无torn frame，不把future frame放入当前snapshot |
| camera停流但process heartbeat继续 | camera age超限、snapshot invalid |
| hand状态旧、arm状态新 | required-valid失败或明确skew reject |
| tactile最后一次contact后长时间无写入 | age超限，不无限当作当前force |
| device timestamp reset | host monotonic对齐仍正确，device discontinuity被记录 |
| history不足 | 按spec显式pad并标invalid，或拒绝ready |
| ring wraparound | `get_last_k()`返回可验证的oldest-first子集 |

### 12.3 动作协议

| 场景 | 预期 |
|---|---|
| `valid_until`已过 | worker拒绝，不调用SDK |
| observation age超限 | SafetyGate拒绝 |
| 旧policy epoch | worker拒绝 |
| action ID倒退 | worker拒绝并记录 |
| arm queue满 | 有界失败，coordinated hold/FAULT，不阻塞ESC |
| hand已到N+1、arm仍为N | 检测apply skew并hold |
| arm SDK reject、hand accept | 联合动作标失败，不记为成功样本 |
| chunk前半段过期 | 丢弃过期prefix，只验证未来元素 |
| 新chunk与旧chunk重叠 | 按明确规则替换/融合并保存决策 |
| SafetyGate异常 | fail closed，queue无新运动目标 |

### 12.4 数据闭环

| 场景 | 预期 |
|---|---|
| snapshot→preprocess→record→read→preprocess | 输入hash一致 |
| queued成功、arm SDK失败 | queued与accepted字段不同 |
| arm/hand ACK时间不同 | 保存可计算skew |
| invalid camera row | quality和replay默认拒绝/过滤 |
| schema v13 episode | 新reader保持可读 |
| schema新版本 | old reader按约定失败或忽略可选字段，不静默错读 |
| ENOSPC | 控制保持安全，临时数据可诊断/恢复，不立即删除 |

### 12.5 性能准入

在以下组合下测量至少10–30分钟：

- recording off/on；
- RGB-only、RGB-D、pointcloud；
- CPU/GPU inference；
- 正常系统负载和受控背景负载；
- 模型warm和cold路径。

必须报告：

```text
snapshot build p50/p95/p99/max
preprocess p50/p95/p99/max
inference p50/p95/p99/max
safety p50/p95/p99/max
enqueue p50/p95/p99/max
observation age p95/p99
camera/arm/hand skew p95/p99
arm/hand apply skew p95/p99
deadline miss count
expired action count
dropped observation count
queue-full count
```

阈值应根据最终模型和实机测量确定，但任何放行标准都不能只使用平均值。

## 13. 最小实机验收清单

仅在全部离线测试通过后执行。要求工作区清空、低速度/低加速度、专人守住物理急停。

1. **启动和warmup**
   - 上电后保持DISARMED；
   - 模型加载/warmup期间控制器不得因policy ready误判进入运行；
   - policy ready后仍需人工ARMED。

2. **单步低幅策略**
   - 仅使用小关节变化；
   - 验证observation age、inference和ACK时间；
   - 验证arm-hand共同action ID。

3. **人工制造推理迟到**
   - 在mock/delay backend中逐步增加延迟；
   - 确认expired output不执行；
   - 确认hold/FAULT时序。

4. **传感器停流**
   - 在静止状态中断camera或VR；
   - 确认source freshness失败，而不是仅依靠heartbeat；
   - 不允许模型继续使用无限旧帧。

5. **软件ESC**
   - 使用毫米/小角度安全动作；
   - 验证即使inference thread繁忙，Coordinator仍处理ESC；
   - 验证控制器停止状态得到确认。

6. **进程故障**
   - 只使用受控dummy backend模拟异常，不主动制造硬件碰撞；
   - 验证policy exit被记录为FAULT；
   - 验证全部进程退出后才unlink IPC。

7. **chunk策略**
   - 只使用小幅、预验证轨迹；
   - 人工注入迟到和chunk重叠；
   - 验证arm queue不扩容、不出现旧chunk尾部执行。

## 14. 本次离线验证记录

### 14.1 基线

```text
compileall: PASS
pytest tests: 90 passed in 1.68s
```

### 14.2 进程模型

```text
configured_start None
effective_start fork
daemon child spawn:
  AssertionError: daemonic processes are not allowed to have children
```

### 14.3 旧动作

构造 `created_monotonic_s = now - 60s` 的arm action，metadata parser仍接受原sequence和timestamp，没有TTL拒绝。

### 14.4 arm-hand动作版本分离

连续发布两组arm/hand动作后：

```text
arm queue下一条：第1组
hand latest ring：第2组
```

证明两侧当前无法表示同一联合动作版本。

### 14.5 卡死进程停机

dummy worker忽略SIGTERM：

```text
shutdown_elapsed_s 6.01
alive_after_shutdown True
exitcode None
```

测试结束后仅对本次创建的dummy进程执行kill并确认退出；未影响仓库或硬件进程。

### 14.6 历史窗口

按30 Hz估算：

```text
camera maxlen=5  → 166.7 ms
arm maxlen=8     → 266.7 ms
hand maxlen=8    → 266.7 ms
```

不足以原生支持常见0.5–2秒model observation horizon。

## 15. 最终判定

### 当前可以继续

- 保留并强化现有canonical VR teleop；
- 开发纯离线PolicyBackend和ObservationSpec；
- 建立fake producer/fake SDK/fake model测试；
- 设计additive schema和兼容reader；
- 离线benchmark模型preprocess/inference；
- 使用dry-run验证模型输出和SafetyGate。

### 当前不应放行

- 学习策略直接写 `arm_action_q`/`hand_cmd_ring`；
- live replay；
- action chunk整批入队或扩大arm queue；
- arm-side插值；
- daemon policy内多进程DataLoader；
- 未warmup/未校验模型进入ARMED；
- live policy热重启；
- 把latest多流读取称作已同步observation；
- 把queued action称作硬件executed；
- 在未确认子进程退出时关闭SharedStorage。

### 建议策略部署放行门槛

```text
Mandatory SafetyGate完成
AND spawn/non-daemon/confirmed shutdown完成
AND policy_ready + required sensor capability完成
AND causal ObservationSnapshot完成
AND policy epoch/action ID/TTL/paired ACK完成
AND action chunk scheduler完成（如模型需要）
AND schema与quality/replay端到端兼容
AND fake model/fake SDK故障矩阵全部PASS
AND 最小实机验收全部记录PASS
```

在这些条件满足前，DexMani Real应被视为“安全性较强的VR teleop与数据采集系统”，而不是“可直接插入任意学习策略的通用部署运行时”。后续整改的核心不是扩大抽象层级，而是把当前已经存在的安全机制、时间语义和录制所有权提升为所有策略都无法绕过的统一契约。

## 16. ManiUniCon 核心机制二次深审范围

### 16.1 对照基线与方法

本轮对照使用 ManiUniCon commit：

```text
85c6f2e32ecf9f2bed62d202b058c39623444686
```

对照仓库在审查时无本地修改。本轮没有运行 ManiUniCon 主程序、相机、机器人接口或任何可能连接硬件的入口，只进行了：

- 源码调用链和配置解析路径核查；
- 不导入硬件包的纯 Python 公式复现；
- 共享内存算法、Event 状态机和录制生命周期的静态状态推演；
- 多相机 payload 大小的离线计算；
- 可选依赖导入边界的离线导入检查。

本轮重点回答的不是“ManiUniCon 有哪些类”，而是以下五个部署问题：

1. `spawn` 之后，模型、SDK 和可变运行时对象究竟在哪个进程创建和销毁；
2. `robot_ready/policy_ready` 是否真的表示动作执行完成，以及任一方死亡时是否能退出等待；
3. `observation_horizon` 是否等于时间对齐后的历史，而不只是各流独立的 latest K；
4. action chunk 的过期、重叠、reset、队列积压和执行确认是否有完整协议；
5. 分布式 TimestampAlignedBuffer 是否能原子地产生同一 episode、同一长度和可解释的有效样本。

### 16.2 ManiUniCon 的真实运行链

```text
Main
  ├─ set_start_method("spawn")
  ├─ Hydra instantiate Robot(robot_interface=...)
  ├─ Hydra instantiate Policy(config objects retained by _recursive_=False)
  ├─ Hydra instantiate Sensor processes
  └─ start sensors → policy → robot

Policy child
  ├─ instantiate model / obs_wrapper / act_wrapper
  ├─ read latest-K state and latest-K fused camera independently
  ├─ model inference
  ├─ action wrapper assigns wall-clock target timestamps
  └─ push the whole surviving chunk into SharedMemoryQueue

Robot child
  ├─ state receiver thread → state ring
  ├─ action loop → drain-all queue
  ├─ optional robot-side interpolation
  └─ interface.send_action()

Camera sensor child
  ├─ spawn one process per physical camera
  ├─ read each camera's latest frame
  ├─ stack frames and replace per-camera timestamps with their mean
  └─ publish fused multi-camera ring

Recording
  policy toggles shared start_time/is_recording
  ├─ robot state thread owns state buffer
  ├─ robot action loop owns action buffer
  └─ sensor process owns camera buffer
      each process independently observes stop and dumps one NPZ
```

这条链路说明 ManiUniCon 的“统一”主要位于配置和接口表面；进程监督、时间语义、动作安全和 episode 提交仍然是分散的。DexMani Real 可以借鉴其插件边界，但不能把这些分散语义一起带入。

### 16.3 “借鉴”的三个等级

| 等级 | 含义 | 示例 |
|---|---|---|
| 原则可借鉴 | 设计意图正确，可在 Dex 契约下重新实现 | spawn、子进程加载模型、obs/action adapter、history、target time |
| 改造后借鉴 | 概念有价值，但参考代码的状态机或数据语义不满足 Dex 安全要求 | policy ready、chunk、reset、Hydra 插件、多相机历史 |
| 明确不移植 | 会削弱 Dex 的安全、时序或数据不变量 | robot-side interpolation、无超时 Event 锁步、256 深度动作队列、分布式录制、用最旧帧静默补 history |

## 17. ManiUniCon 二次深审结论索引

本轮新增 16 项对照发现：12 项 P1、4 项 P2。严重级别表示“若把该机制用于 Dex 策略部署，或让当前 Dex 缺口继续存在”的风险，而不是对参考仓库做发布评级。

| ID | 级别 | 核心发现 | 性质 | 与既有发现关系 |
|---|---|---|---|---|
| MU-01 | P1 | spawn 正确，但 Robot/SDK 对象仍在主进程递归构造，所有权边界不安全 | 移植陷阱 | 加深 PD-02、PD-17 |
| MU-02 | P1 | BasePolicy 把模型 backend、Process 生命周期和 raw SharedStorage 权限绑在一起 | 架构缺口 | 加深 PD-01、PD-14 |
| MU-03 | P1 | 双 Event 锁步无超时，且“执行完成”使用读取新 chunk 前的旧 action 判断 | 参考反例 | 加深 PD-04、PD-05、PD-12 |
| MU-04 | P1 | 主进程不监督 worker，ready/stop 可无限等待，失败也可能被标 ready | 参考反例 | 加深 PD-06、PD-07 |
| MU-05 | P1 | observation horizon 只是独立 latest K，启动不足时静默复制最旧帧 | 同步反例 | 加深 PD-08、PD-15 |
| MU-06 | P1 | 多相机融合丢失逐相机时间戳，以均值掩盖 stale 和 skew | 数据质量反例 | 加深 PD-08、PD-09 |
| MU-07 | P1 | 控制时间使用 wall clock，并在消费侧临时换算到 monotonic | 调度反例 | 加深 PD-09、PD-10 |
| MU-08 | P1 | chunk 全部迟到时重新定时最后一个旧预测，而不是 hold | 安全反例 | 加深 PD-10、PD-12 |
| MU-09 | P1 | 256 深度队列、drain-all 和“确保全部执行”不具备过期/替换协议 | 调度反例 | 加深 PD-10至PD-13 |
| MU-10 | P1 | reset 是可丢失的共享 Event，queue、模型历史和 chunk 历史没有原子换代 | 生命周期反例 | 加深 PD-13 |
| MU-11 | P1 | 分布式录制没有 stop barrier、原子提交或关机 flush 保证 | 数据质量反例 | 加深 PD-18 |
| MU-12 | P1 | TimestampAlignedBuffer 用未来到达样本回填过去网格且没有 valid mask | 数据质量反例 | 加深 PD-18及前序 schema 问题 |
| MU-13 | P2 | lock-free ring 依赖速率/拷贝时限假设，并静默 padding；不等价于 seqlock | IPC 移植边界 | 支持保留 Dex 优势 |
| MU-14 | P2 | wrapper/config 是弱约束 duck typing，关键参数未传播或未使用 | 工程缺口 | 加深 PD-14、PD-19 |
| MU-15 | P2 | 可选依赖被顶层 eager import，动态 eval/remote code 缺少准入治理 | 隔离与供应链缺口 | 加深 PD-17至PD-19 |
| MU-16 | P2 | 大帧复制、busy-spin、逐 tick print 和线程池设置没有统一实时预算 | 性能缺口 | 加深 PD-05、PD-16、PD-17 |

## 18. 进程、插件和监督机制详细发现

## MU-01 — P1：spawn 不能替代正确的对象所有权

### 状态

静态确认。ManiUniCon 使用 `spawn` 的方向正确，但其 Robot 对象构造方式不能作为 Dex SDK 隔离模板。

### 证据

参考入口在 `../ManiUniCon/main.py:160` 调用：

```python
mp.set_start_method("spawn")
```

这是可借鉴的部分。Torch/VLA policy 通过 `_recursive_=False` 保留模型配置，并在 policy 子进程 `run()` 中实例化模型和 wrapper：

- `../ManiUniCon/main.py:44-49`；
- `../ManiUniCon/maniunicon/policies/torch_model.py:126-136`；
- `../ManiUniCon/maniunicon/policies/torch_model_vla.py:126-136`。

但 Robot 使用默认递归 Hydra 实例化：

- `../ManiUniCon/main.py:40-42`；
- `../ManiUniCon/configs/robot/xarm6.yaml:1-5`。

这会在主进程创建 `XArm6RobotiqInterface`，并在解析目标模块时导入顶层 xArm/Robotiq SDK：

- `../ManiUniCon/maniunicon/robot_interface/xarm6_robotiq.py:10-18`；
- `../ManiUniCon/maniunicon/robot_interface/xarm6_robotiq.py:24-44`。

随后整个 interface 对象随 Robot Process 被 pickle 给子进程。停机时主进程还先调用 `self.robot.disconnect()`：

- `../ManiUniCon/main.py:120-129`。

而实际硬件连接是在 Robot 子进程 `run()` 中建立，并在子进程 cleanup 中断开：

- `../ManiUniCon/maniunicon/core/robot.py:253-264`；
- `../ManiUniCon/maniunicon/core/robot.py:468-472`。

因此主进程的 `disconnect()` 操作的是 spawn 前的父进程副本；它不能作为卡死子进程的硬件断连后备。

### 成因

把“配置可实例化”误当成“实例可以跨进程传递”。`spawn` 只避免继承父进程线程状态，不会自动赋予 SDK 对象单一所有者，也不会让父、子对象共享连接状态。

### 对 Dex 的影响

Dex 当前在 `dexmani_real/robot/arm_loop.py:139-142` 的 arm worker 内导入并创建 `XArmAPI`，XHand 也由 hand worker 延迟导入。这一边界必须保留。后续 Hydra 或其他插件系统只能传递纯配置/工厂标识，不能让 main 实例化 vendor interface、CUDA model 或持有 live SDK 对象。

### 修复与验收

- 使用 `mp.get_context("spawn")` 构造全部 Process/Queue/Value/Event；
- main 只创建 `RobotWorkerSpec`、`PolicySpec` 等可序列化配置；
- vendor SDK 模块和对象只在对应 worker `run()` 内导入/构造；
- 测试记录模块 import PID、对象构造 PID、connect PID 和 disconnect PID，要求 SDK 四项只出现在同一 worker；
- policy model、normalizer、CUDA context同样只在 policy/inference owner 内创建。

## MU-02 — P1：BasePolicy 不应同时是模型接口、Process 和 IPC 写权限主体

### 状态

静态确认。参考实现的 BasePolicy 有助于统一启动形式，但不是 Dex 所需的安全策略接口。

### 证据

`BasePolicy` 直接继承 `mp.Process`，构造时持有完整 `SharedStorage` 和 reset Event：

- `../ManiUniCon/maniunicon/core/policy.py:7-20`。

具体 Torch/VLA policy 直接从 SharedStorage 读取 observation，并把 action 写入底层 queue：

- `../ManiUniCon/maniunicon/policies/torch_model.py:232-249`；
- `../ManiUniCon/maniunicon/policies/torch_model.py:300-301`。

`stop()` 同时修改全局 `is_running` 并无超时 `join()`：

- `../ManiUniCon/maniunicon/core/policy.py:26-29`。

### 成因

参考抽象统一的是“进程外形”，不是以下四个应独立验证的契约：

```text
model backend: ObservationTensor → RawPolicyOutput
observation adapter: ObservationSnapshot → ObservationTensor
action adapter: RawPolicyOutput → TypedActionCandidate
runtime coordinator: deadline/safety/scheduler/IPC/recording
```

当 backend 自己持有 SharedStorage 时，任何模型插件都可以绕过统一 snapshot、SafetyGate、TTL、coordinated hold 和 recorder。

### 对 Dex 的影响

Dex 不应复制 `class LearnedPolicy(mp.Process)` 作为插件协议。正确方向是组合：policy worker 拥有 Coordinator，Coordinator 调用无 IPC 权限的 `PolicyBackend`。backend 可以在当前 policy 进程或独立 inference 进程运行，但它只接收不可变输入并返回 raw output。

### 修复与验收

- `PolicyBackend` 不导入 `SharedStorage`，不接受 queue/ring；
- `ObservationAdapter` 和 `ActionAdapter` 是可离线测试的纯边界；
- 只有 Coordinator 可调用 SafetyGate 和 CommandScheduler；
- 用恶意 dummy plugin 尝试写 queue，验证它拿不到 raw IPC；
- backend 单测无需启动 multiprocessing 或硬件依赖。

## MU-03 — P1：双 Event synchronized 握手既可能永久等待，也可能提前宣告执行完成

### 状态

静态状态机确认。该问题不是对 Event API 的泛化担忧，而是参考实现当前控制流中的具体顺序错误。

### 证据

SharedStorage 定义 `robot_ready` 和 `policy_ready`，并把 `robot_ready` 初始置位：

- `../ManiUniCon/maniunicon/utils/shared_memory/shared_storage.py:185-196`。

policy 无超时等待 robot，再在写完 chunk 后置 policy ready：

- `../ManiUniCon/maniunicon/policies/torch_model.py:184-190`；
- `../ManiUniCon/maniunicon/policies/torch_model.py:300-307`。

robot 无超时等待 policy：

- `../ManiUniCon/maniunicon/core/robot.py:301-309`。

更关键的是，Robot 每轮按以下顺序运行：

1. 从旧 interpolator/旧 `executing_actions` 计算局部变量 `action`：`robot.py:311-343`；
2. 才从 queue 读取并安装新 chunk：`robot.py:367-439`；
3. 用步骤1的旧 `action` 判断步骤2的新 chunk是否完成：`robot.py:441-454`。

首个 chunk 到达时，步骤1通常得到 `action is None`。步骤2虽然刚把新动作装入 scheduler，步骤3仍会因 `action is None` 立即 `robot_ready.set()` 并把 `is_executing_actions=False`。下一轮 Robot 又会在处理刚安装的动作前等待下一次 `policy_ready`。因此这个 ready 不等于“新 chunk 已执行完成”。

### 影响

- policy 或 robot 任一方退出时，另一方可永久阻塞在 `Event.wait()`；
- ready 可在动作真正执行前置位；
- 评测结果可能把“推理—执行锁步”误认为已经建立；
- reset/error 通过 set/clear Event 试图解锁时存在丢信号和世代混淆；
- 若把此机制移到 Dex，arm servo/hold 更新可能被 policy 推理锁死。

### 对 Dex 的结论

Dex 不应实现 policy 与 arm worker 的强锁步。arm worker 必须持续以自身周期运行，消费“当前有效的下一动作”或保持 hold；policy inference 可以慢，但不能阻塞安全执行线程。

需要同步评测时，应同步“observation ID → action ID → worker received → SDK accepted/final applied”，而不是同步两个无世代 Event。

### 验证

- policy 在 `robot_ready.wait()` 前/后死亡，main 必须检测并结束 session；
- robot 在 `policy_ready.wait()` 前/后死亡，policy/Coordinator 必须在有界时间 hold/FAULT；
- ACK 只能在对应 `action_id` 被 SDK 接受后产生；
- 最后一个 chunk item 的完成 ACK 不能由读取 chunk 前的局部状态触发；
- 所有等待同时检查 shutdown、fault、epoch，并有 monotonic deadline。

## MU-04 — P1：参考主进程不具备可借鉴的监督和确定性停机语义

### 状态

静态确认。

### 证据

参考 main 启动后只执行无限 sleep，不检查 `is_alive()`、exit code、heartbeat 或 progress：

- `../ManiUniCon/main.py:172-177`。

BasePolicy、BaseSensor 和 Robot 的 `stop()` 都执行无超时 `join()`：

- `../ManiUniCon/maniunicon/core/policy.py:26-29`；
- `../ManiUniCon/maniunicon/core/sensor.py:25-28`；
- `../ManiUniCon/maniunicon/core/robot.py:474-477`。

RealSense 子相机 `start_wait()` 也是无超时 Event wait，并且 finally 在初始化失败时仍设置 ready：

- `../ManiUniCon/maniunicon/sensors/realsense.py:74-86`；
- `../ManiUniCon/maniunicon/sensors/realsense.py:225-237`。

`RobotControlSystem.stop()` 可能从 except 和 finally 重复调用：

- `../ManiUniCon/main.py:180-187`。

### 成因

ready 只被当成流程栅栏，未携带成功/失败/世代；stop 只被当成正常 cooperative join，未覆盖 Python hang、CUDA hang、SDK 阻塞和子进程提前死亡。

### 对 Dex 的结论

Dex 已有 heartbeat、process monitor、sticky error 和 readiness timeout，不能为了统一接口退回参考 main 的 sleep-loop。需要修复的是 PD-06/PD-07 指出的故障分类和最终 kill 确认，而不是替换掉现有 supervisor。

### 修复与验收

- ready 必须是 `{phase, success, generation, detail}`，或至少分离 success/failure Event；
- startup deadline、runtime heartbeat deadline、inference progress deadline分别配置；
- stop 按 cooperative → terminate → kill → confirmed exit 执行；
- worker failure不得通过“置 ready 防止等待”伪装成启动成功；
- shutdown 必须幂等，但第二次调用不能掩盖第一次失败。

## 19. Observation、相机和时钟机制详细发现

## MU-05 — P1：observation horizon 不等于时间对齐历史

### 状态

静态确认，并发现参考 policy 的启动顺序空值错误。

### 证据

Torch/VLA policy 先独立读取 state latest K 和 fused camera latest K：

- `../ManiUniCon/maniunicon/policies/torch_model.py:232-235`；
- `../ManiUniCon/maniunicon/policies/torch_model_vla.py:234-237`。

随后在检查 `state is None` 之前先访问 `state.timestamp[-1]`：

- `../ManiUniCon/maniunicon/policies/torch_model.py:236-244`；
- `../ManiUniCon/maniunicon/policies/torch_model_vla.py:238-247`。

因此自动激活或启动竞态下，无 state 会触发 `AttributeError`，而不是得到“not ready”。

共享 ring 在历史不足时不是返回较短历史或 valid mask，而是把最旧的真实项复制到开头，直到长度恰好等于 K：

- `../ManiUniCon/maniunicon/utils/shared_memory/shared_memory_ring_buffer.py:201-258`，特别是 `:219-235`。

obs wrapper 拼接 state 并处理图像，但不读取或核对 state/camera timestamp：

- `../ManiUniCon/maniunicon/customize/obs_wrapper/ppt_rgb_wrapper.py:58-89`；
- `../ManiUniCon/maniunicon/customize/obs_wrapper/spatialvla_rgb_wrapper.py:69-100`。

默认 `get_max_k=3`：

- `../ManiUniCon/configs/default.yaml:1-5`。

### 成因

“长度对齐”被当成“时间对齐”。两个独立 ring 的第 `i` 项可能来自不同物理时刻；silent padding 又让模型无法区分真实历史与启动复制。

### 对 Dex 的结论

Dex 应借鉴 history 作为一等能力，但实现必须遵守：

```text
anchor = host monotonic timestamp
for each requested history slot t_i:
    select newest sample with source/publish time <= t_i
    return payload + seq + age + valid + pad_reason
never select a future sample
```

现有 Dex `get_last_k()` “可能少于 K、oldest-first、seqlock verified”的语义优于参考实现，必须保留。padding 如果由模型要求，应在 ObservationAdapter 中显式进行并携带 mask，不能改变底层 ring 契约。

### 验证

- 0、1、K-1、K 条历史分别测试；
- state 30 Hz、camera 15 Hz 的第 i 项不得按数组索引盲配；
- startup padding 必须有 `valid=False/pad_reason=startup`；
- state/camera 任一为 None 不得先解引用；
- 训练 preprocessing 必须消费同一 mask/padding 规则。

## MU-06 — P1：多相机融合用均值时间掩盖逐相机 stale 和 skew

### 状态

静态确认。

### 证据

每个 RealSense 子进程原本发布：

- device capture timestamp；
- host wall receive timestamp；
- step index；
- 通用 timestamp。

见 `../ManiUniCon/maniunicon/sensors/realsense.py:169-207` 和 `shared_storage.py:114-125`。

聚合进程逐相机读取各自最新帧：

- `../ManiUniCon/maniunicon/sensors/realsense.py:391-404`。

融合时只收集通用 `data.timestamp`，然后把多相机 timestamp 设为均值：

- `../ManiUniCon/maniunicon/sensors/realsense.py:459-492`。

最终 `MultiCameraData` 只保留一个 timestamp，不保留每台相机的 capture/receive/step：

- `../ManiUniCon/maniunicon/utils/shared_memory/shared_storage.py:128-138`；
- `../ManiUniCon/maniunicon/utils/shared_memory/shared_storage.py:509-544`。

### 触发场景

camera A 正常更新，camera B 停在旧帧。聚合进程仍能每轮读到 B 的 last frame，然后与 A 堆叠。均值时间既不是 A 的时间，也不是 B 的时间；随着 A 更新，它仍会变化，使 fused stream 看起来“在前进”，但 B 已 stale。

### 影响

- 视觉策略使用跨相机不同物理时刻的图像；
- mean timestamp 无法恢复 `max(camera_ts)-min(camera_ts)`；
- heartbeat/聚合更新时间不能证明每个 source 在出新帧；
- episode 中无法区分相机停流、重复帧和正常同步帧。

### 对 Dex 的结论

Dex 当前是单 L515，但未来多相机/外部视觉部署也不应采用 fused mean timestamp。camera snapshot 必须逐 source 保存 device capture、host receive monotonic、publish、frame number、generation、valid，并显式计算 skew。多相机 stack 只能在所有 required source 通过 freshness/skew gate 后构造。

### 验证

- 冻结一台 fake camera 的 frame number，聚合 heartbeat继续：snapshot 必须 invalid；
- 两相机注入 100 ms skew：不得只保存均值；
- device clock reset：host monotonic freshness仍正确；
- 录制保存逐相机 timestamp/seq/valid，quality tool能定位具体坏相机。

## MU-07 — P1：wall-clock target timestamp 不能作为实时控制时基

### 状态

静态确认。

### 证据

参考 RobotState 普遍用 `time.time()`：例如 xArm interface 在：

- `../ManiUniCon/maniunicon/robot_interface/xarm6_robotiq.py:177-188`。

相机 receive timestamp 也使用 `time.time()`：

- `../ManiUniCon/maniunicon/sensors/realsense.py:169-205`。

action wrapper 用 state wall timestamp 生成整个 chunk，并用当前 `time.time()` 判断过期：

- `../ManiUniCon/maniunicon/customize/act_wrapper/chunk_wrapper.py:45-70`。

Robot 消费时再以当下的两个时钟差把 wall target 转换为 monotonic：

- `../ManiUniCon/maniunicon/core/robot.py:426-435`。

synchronized 模式还会把所有 target 相对当前 wall time重写：

- `../ManiUniCon/maniunicon/core/robot.py:397-403`。

### 成因

同一个 `timestamp` 同时承担 observation source time、动作版本和执行 deadline。wall clock 适合人类日志，不适合超时、age 和 target scheduling；NTP/PTP 校时、手工改时或虚拟化时钟跳变都会改变其差值。

### 对 Dex 的结论

Dex 已将 heartbeat、arm/hand state 和 action creation 放在 host monotonic 域，这是应保留的优势。未来 target/TTL/ACK 全部使用 `monotonic_ns`；wall time只作为 episode 元数据和跨机器相关信息，不能参与本机安全判定。

### 验证

- monkeypatch/模拟 wall clock 向前或向后跳变，动作 age/TTL/target顺序不变；
- monotonic deadline严格递增；
- source device/remote clock不直接与 host monotonic相减；
- record 同时保留 wall anchor 和 monotonic offsets，但消费者知道各自域。

## MU-13 — P2：参考 lock-free ring 的安全模型不等于 Dex seqlock

### 状态

静态确认，属于不可照搬边界。

### 证据

参考 ring 根据 `put_desired_frequency * get_time_budget` 计算额外槽位，假定 recent K 在最大拷贝时间内不会被覆盖：

- `../ManiUniCon/maniunicon/utils/shared_memory/shared_memory_ring_buffer.py:42-54`。

writer 在复用槽位过早时主动 sleep：

- `../ManiUniCon/maniunicon/utils/shared_memory/shared_memory_ring_buffer.py:125-166`。

reader 在复制完成后检查总耗时，超预算则抛错，但没有读前/读后同一 slot sequence 验证：

- `../ManiUniCon/maniunicon/utils/shared_memory/shared_memory_ring_buffer.py:177-199`；
- `../ManiUniCon/maniunicon/utils/shared_memory/shared_memory_ring_buffer.py:201-258`。

此外，SharedMemoryQueue 的 producer 流程是“load write counter → 写 slot → counter add”，没有多 producer reservation：

- `../ManiUniCon/maniunicon/utils/shared_memory/shared_memory_queue.py:88-107`。

### 影响与边界

- ring 正确性依赖单 writer、可信最大写频率和可信拷贝预算；
- 系统受压时 writer 可能在数据发布路径 sleep；
- K 不足的 silent padding混淆数据完整性；
- queue 若被多个 action producer共享，两个 producer可能选择同一 slot；
- 这些约束适合明确的参考拓扑，但不适合 Dex 的多入口 arm command 生态。

### 对 Dex 的结论

保留 Dex seqlock、verified read、may-return-fewer-than-K 和 `mp.Queue(maxsize=2)`。不要为追求“lock-free”替换它们。若未来优化 camera ring，应以 sequence 验证和选择性模态复制为前提，而不是以固定时间预算证明不会覆盖。

### 验证

- writer 在 reader 复制中 wrap：读结果要么完整旧帧、完整新帧，要么明确失败，不能返回混合帧；
- 多 producer queue 测试 action ID不重号、不覆盖；
- 系统受压超过预计 copy budget 时 fail closed，并有指标；
- `get_last_k()` 仍返回实际 verified 数量，不静默伪造历史。

## 20. Chunk、动作调度、reset 和配置契约详细发现

## MU-08 — P1：全部 chunk 迟到时“重定时最后一个旧预测”违背 hold-on-failure

### 状态

源码确认并纯离线复现。

### 证据

ActionChunkWrapper 先以 observation timestamp 生成：

```text
target[i] = observation_timestamp + i * dt
```

非 synchronized 模式只保留 `target > now + action_exec_latency` 的动作：

- `../ManiUniCon/maniunicon/customize/act_wrapper/chunk_wrapper.py:45-59`。

如果一个都不剩，代码注释是“exceeded time budget, still do something”，随后选择最后一个预测并把它重新安排到未来全局网格：

- `../ManiUniCon/maniunicon/customize/act_wrapper/chunk_wrapper.py:61-70`。

同样逻辑存在于 SpatialVLA/RoboVLMs wrapper：

- `../ManiUniCon/maniunicon/customize/act_wrapper/spatialvla_eepose_wrapper.py:72-82`；
- `../ManiUniCon/maniunicon/customize/act_wrapper/robovlms_eepose_wrapper.py:72-82`。

离线复现使用 observation time 10.0 s、当前时间 10.35 s、3步 chunk、dt=0.1 s。原 targets 为 `[10.0, 10.1, 10.2]`，全部过期；输出为：

```text
No new actions !!! Check model inference time
stale_fallback_count 1
target 10.4
last_pred 3.0
```

即旧 chunk 的最后一个动作被赋予新的未来 deadline。

### 成因

把“控制不能空缺”解释成“必须执行某个模型动作”，而不是“执行器必须持续得到安全 hold”。这会把时间有效性检查变成重放授权。

### 对 Dex 的结论

Dex 的 late/invalid/no-output 统一语义必须是：

```text
expired output → discard
no valid future action → coordinated hold
consecutive deadline miss over threshold → FAULT or operator-confirmed degraded mode
```

不能给旧预测刷新 `valid_until`，也不能把 chunk 最后一项当作默认 hold，除非它经过独立的“当前状态 hold”生成和安全验证。

### 验证

- chunk 全迟到：raw output被记录，但 command queue无模型运动项；
- scheduler生成当前 arm/hand coordinated hold；
- 迟到前缀被丢弃，未来后缀保持原 action ID/target，不重写因果时间；
- deadline miss counter和故障阈值可观测。

## MU-09 — P1：大动作队列和 drain-all 不是 chunk scheduler

### 状态

静态确认。

### 证据

参考默认 action buffer size 为 256：

- `../ManiUniCon/configs/default.yaml:31-34`。

policy 把 wrapper 返回的全部动作逐个写入 queue：

- `../ManiUniCon/maniunicon/policies/torch_model.py:300-301`。

Robot 每 tick `get_all()`，一次取走当前全部动作：

- `../ManiUniCon/maniunicon/core/robot.py:367-368`；
- `../ManiUniCon/maniunicon/utils/shared_memory/shared_memory_queue.py:140-149`。

synchronized 模式明确重写每个 target，“to make sure all actions get executed”：

- `../ManiUniCon/maniunicon/core/robot.py:397-403`。

RobotAction 只有 `timestamp/target_timestamp`，没有 policy epoch、observation ID、action ID、valid-until 或 replace group：

- `../ManiUniCon/maniunicon/utils/shared_memory/shared_storage.py:91-107`。

另一个配置断点是：top-level policy 配置有 `synchronized`，但 action wrapper 配置没有把它传下去；wrapper 默认仍是 `False`：

- `../ManiUniCon/configs/policy/spatialvla.yaml:10,28-33`；
- `../ManiUniCon/maniunicon/customize/act_wrapper/spatialvla_eepose_wrapper.py:15-35`。

因此 policy、wrapper 和 robot 对“synchronized”的理解可能不一致。

### 影响

- 新旧 chunk 重叠时没有明确 replace/merge 规则；
- 旧 chunk 尾部可在新 observation 已到达后继续有效；
- 大 queue 隐藏 policy/robot 不匹配，扩大最坏停止距离；
- drain-all 把瞬时 backlog整体推入 interpolator/executing list；
- retime-to-execute-all 会取消过期保护。

### 对 Dex 的结论

保持 arm queue `maxsize=2`。chunk 应保存在 policy-side scheduler 中，而不是扩大 IPC queue。scheduler 每个控制 tick只发布下一项或一个有界小窗口，并以 observation ID、epoch、target、TTL、overlap policy和安全验证管理它。

### 验证

- 两个重叠 chunk 注入不同 observation ID，替换规则确定且可记录；
- queue 永远不因 action horizon扩大；
- reset/FAULT 后旧 epoch chunk无法重新进入 queue；
- 过期动作即使仍在 queue也由 worker二次拒绝；
- 配置中的 synchronized/async mode在 policy、adapter、scheduler、worker只有一个解析结果。

## MU-10 — P1：reset Event 没有 generation，可能被错过且未清空全部时序状态

### 状态

静态确认。

### 证据

main 键盘线程只执行 `reset_event.set()`：

- `../ManiUniCon/main.py:70-79`。

Robot 处理 reset 后由自身 `clear()`：

- `../ManiUniCon/maniunicon/core/robot.py:281-299`。

policy 只有在轮询时恰好看到 Event 才会 drain action queue和 de-activate：

- `../ManiUniCon/maniunicon/policies/torch_model.py:173-177`。

TorchModelPolicy reset 分支没有调用 model reset；VLAPolicy 调用了 `model.reset()`，但没有清空 `prev_raw_actions/prev_raw_timestamps`：

- `../ManiUniCon/maniunicon/policies/torch_model.py:173-177`；
- `../ManiUniCon/maniunicon/policies/torch_model_vla.py:174-179`；
- `../ManiUniCon/maniunicon/policies/torch_model_vla.py:75-77,253-281`。

### 触发场景

Robot 比 policy 更快看到 Event，执行 reset、sleep 后 clear；此时 policy 可能从未看到 reset。其模型历史、real-time chunk overlap、activation 状态和 queue drain就不保证与物理回零同步。

### 成因

Event 只表达当前电平，不表达“第 N 次 reset 已由哪些参与者确认”。由某一个消费者 clear 的广播 Event天生不适合多参与者事务。

### 对 Dex 的结论

policy restart、home/reset和模式切换都必须用 generation/epoch，而不是单个 clearable Event：

```text
requested_epoch
arm_ack_epoch
hand_ack_epoch
policy_ack_epoch
scheduler_epoch
```

新动作带 epoch；worker只接受当前 epoch。换代时先禁止新运动、排空/作废旧动作、生成 hold，再由各组件确认。

### 验证

- reset 在 policy 推理中、queue 满、worker wait、录制中分别触发；
- 每个参与者最终 ACK 同一 epoch；
- prev observation/action history和model cache全部按 spec reset；
- 旧 epoch queue/ring内容存在也不能被执行；
- 连续两次 reset不合并成一次不可区分事件。

## MU-14 — P2：配置和 wrapper 缺少可机器验证的跨组件契约

### 状态

静态确认。

### 证据

Torch/VLA policy 接受并保存 `infer_latency`，但运行链中没有使用它做 deadline、target 或准入判断：

- `../ManiUniCon/maniunicon/policies/torch_model.py:49-71`；
- `../ManiUniCon/maniunicon/policies/torch_model_vla.py:49-71`。

实际迟到筛选使用 wrapper 自己的 `action_exec_latency` 默认值；policy 的 `synchronized` 又未自动传给 wrapper。`dt`、horizon、hist action、steps per inference 分散在 model、policy 和 wrapper 配置中，主要依靠注释说明“should match”。

RobotAction 的 Pydantic model允许任意 NumPy type，除 control mode 外没有 shape/finite/unit/frame 约束：

- `../ManiUniCon/maniunicon/utils/shared_memory/shared_storage.py:91-112`。

基础 `RobotInterface.validate_action()` 通过 clip 修改输入并无条件返回 True，不检查 finite、shape、连续性或碰撞：

- `../ManiUniCon/maniunicon/robot_interface/base.py:89-112`。

### 成因

配置复用依赖字段命名和人工同步，缺少 resolved spec 的等式约束；wrapper 直接产出硬件 RobotAction，使表示转换、安全验证和调度时间化混在一起。

### 对 Dex 的结论

引入可验证 spec，而不是复制 loose Hydra dict：

```text
ObservationSpec
  modalities, shapes, dtypes, units, frames, horizon_s, sample_times, padding

RawActionSpec
  representation, shape, dtype, normalization, horizon, dt

SafeCommandSpec
  arm/hand dimensions, units, target clock, TTL, epoch, limits
```

所有派生字段在启动时一次解析并断言：`model_horizon == adapter_horizon`、`chunk_dt == scheduler_dt`、`required_history <= ring_capacity`、`representation` 有唯一 adapter。

### 验证

- 故意配置 horizon/dt/shape/num_cams不一致，必须在 DISARMED 启动阶段失败；
- NaN/Inf、错误 quaternion order、degree/radian错误由 adapter/gate拒绝；
- unused config在 lint/startup validation中报错；
- resolved config hash进入 episode manifest。

## 21. 录制、依赖隔离和实时性能详细发现

## MU-11 — P1：分布式录制没有 episode stop barrier 和原子提交

### 状态

静态确认。

### 证据

policy 通过 shared flag 发布 wall-clock start time 和 dt：

- `../ManiUniCon/maniunicon/policies/torch_model.py:192-230`；
- `../ManiUniCon/maniunicon/utils/shared_memory/shared_storage.py:571-579`。

Robot state thread、Robot action loop和 camera sensor各自在第一次看到 `is_recording=True` 时创建自己的 TimestampAlignedBuffer，在看到 False 时各自 dump：

- `../ManiUniCon/maniunicon/core/robot.py:186-204`；
- `../ManiUniCon/maniunicon/core/robot.py:369-395`；
- `../ManiUniCon/maniunicon/sensors/realsense.py:406-424`。

dump 直接 `np.savez(final_path)`，没有 temp/commit manifest：

- `../ManiUniCon/maniunicon/utils/timestamp_accumulator.py:207-212`。

worker 循环若在 `is_recording=True` 时因全局 shutdown退出，不会进入 `else` dump 分支；Robot cleanup和 sensor finally也没有强制 flush 当前 buffer：

- `../ManiUniCon/maniunicon/core/robot.py:468-472`；
- `../ManiUniCon/maniunicon/sensors/realsense.py:437-446`。

### 竞态分析

各 producer 轮询频率不同。policy 将 `is_recording` 短暂置 False 后很快开始下一 episode 时，慢 producer 可能没有观察到中间 False：

```text
policy: episode A stop(False) ── set new dir ── episode B start(True)
sensor: ─────────────────────── only observes True ──────────────
```

此时 sensor 可以继续沿用 A 的 buffer/start time，却最终把它写到 B 的目录。即使通常人工按键间隔足够长，协议本身仍没有防止该结果。

### 对 Dex 的结论

Dex policy 单进程拥有 EpisodeRecorder 和 recording grid 是明确优势，不能改成每个 producer独立落盘。producer只发布带 source metadata 的帧，policy/recorder决定属于哪个 episode/grid。episode 完成必须 temp-dir → flush/fsync/close → manifest/quality → atomic rename。

### 验证

- stop 后立即 start：旧 episode数据不能进入新目录；
- shutdown/FAULT/ENOSPC发生在 recording=True：得到明确 aborted temp episode，而不是静默丢失；
- camera/arm/hand长度、valid和source count在commit前核对；
- recorder是唯一 final-path writer；
- 删除 episode必须等待 recorder确认没有活跃 writer。

## MU-12 — P1：参考 TimestampAlignedBuffer 把未来样本写入过去网格且没有有效位

### 状态

源码确认并纯离线复现。

### 证据

`get_accumulate_timestamp_idxs()` 对晚到样本计算跨过的所有 global index，并把同一个 local sample复制到每个 index：

- `../ManiUniCon/maniunicon/utils/timestamp_accumulator.py:27-46`。

TimestampAlignedBuffer 随后把同一 `data` 和同一 source timestamp写入全部这些位置：

- `../ManiUniCon/maniunicon/utils/timestamp_accumulator.py:128-188`。

它没有 `flag_sample_valid`、source index或 fill reason。参考 Robot 源码还明确注释 overwrite buffer存在“entries might remain unfilled”问题：

- `../ManiUniCon/maniunicon/core/robot.py:186-196`。

离线调用：

```python
get_accumulate_timestamp_idxs(
    timestamps=0.35,
    start_time=0.0,
    dt=0.1,
    next_global_idx=0,
)
```

结果：

```text
local_idxs  [0, 0, 0, 0]
global_idxs [0, 1, 2, 3]
next        4
```

0.35 s 才到达的样本被写入 0.0、0.1、0.2、0.3 s 四个网格位置。这是 future-fill，不是 causal forward-fill。

### 对 Dex 的结论

Dex 当前 TimestampAlignedBuffer 已为真正 source slot设置 `flag_sample_valid=True`，回填 slot为 False：`dexmani_real/recording/timestamp_buffer.py:160-194`。这是明显改进，但 payload仍会出现在 invalid gap slot，所有 reader、quality、training export 和 replay必须强制尊重 valid/fill reason；不能只因数组有值就把它当真实观测。

长期方案应区分：

```text
source_sample_valid
fill_valid_for_model
fill_direction
source_index
source_age
```

是否允许 causal forward-fill应按模态配置；任何 future-fill默认非法。

### 验证

- 0.35 s 首样本绝不能成为 0.0–0.2 s 的 valid source；
- source index和age可追踪；
- replay默认拒绝 invalid action/state row；
- training export必须显式选择 drop/mask/causal-fill策略；
-旧 schema没有 valid信息时不得静默宣称数据完全对齐。

## MU-15 — P2：插件化表面下仍有 eager import 和动态代码准入缺口

### 状态

静态确认，并做了无硬件导入检查。

### 证据

ManiUniCon 包顶层主动导入 core、robot_interface、utils：

- `../ManiUniCon/maniunicon/__init__.py:1-8`。

子包又继续 eager import Robot、dummy/meshcat和 meshcat utilities：

- `../ManiUniCon/maniunicon/core/__init__.py:1-5`；
- `../ManiUniCon/maniunicon/robot_interface/__init__.py:1-29`；
- `../ManiUniCon/maniunicon/utils/__init__.py:1-6`。

在当前 `real_robot` 环境中，仅尝试正常导入 `maniunicon.utils.timestamp_accumulator`，就因无关的可选 MeshCat 缺失而失败：

```text
ModuleNotFoundError: This example requires MeshCat.
It can be installed e.g. by `conda install meshcat-python`
```

此外，入口注册 `OmegaConf` 的 `eval` resolver：

- `../ManiUniCon/main.py:152-160`。

部分 VLA model 使用 `trust_remote_code=True`：

- `../ManiUniCon/maniunicon/customize/policy_model/spatialvla_model.py:62-67`。

### 成因

配置可选不等于依赖可选。Python 包初始化链会在选择具体 plugin前导入不相关后端；任意 `_target_`、`eval` 和 remote code又扩大了配置到代码执行的边界。

### 对 Dex 的结论

- canonical main离线 import不能要求 Torch、Transformers、MeshCat、RealSense或 vendor SDK全装齐；
- backend模块只在被选择的 policy child内导入；
- plugin target使用 allowlist/registry，不接受任意 import path；
- 数学派生用结构化解析或显式 Python config，不用通用 `eval`；
- live robot默认禁止远程下载和未固定 revision的 remote code；
- checkpoint、normalizer、processor和自定义代码全部记录 hash/revision。

### 验证

- 在只安装 canonical VR依赖的环境中 import main/SharedStorage成功；
- 选择 policy A不导入 policy B/C的依赖；
- 缺少可选 backend只使该 backend启动失败，不使通用工具导入失败；
- 未 allowlist target、未固定 remote revision或 hash不匹配时保持 DISARMED。

## MU-16 — P2：参考实时路径的内存带宽和调度开销没有统一预算

### 状态

静态确认和离线容量计算。实际 p99 仍需目标机器 benchmark。

### 证据

policy 的 `precise_wait()` 默认保留最后 1 ms busy-spin：

- `../ManiUniCon/maniunicon/policies/torch_model.py:16-25`。

每次 inference 都同步打印耗时：

- `../ManiUniCon/maniunicon/policies/torch_model.py:287-290`。

policy 每轮从 shared ring复制完整的 K 帧 multi-camera colors、depths、intrinsics和transforms，然后 wrapper再 resize/转换到 GPU：

- `../ManiUniCon/maniunicon/policies/torch_model.py:232-249`；
- `../ManiUniCon/maniunicon/customize/obs_wrapper/ppt_rgb_wrapper.py:66-89`。

四台 1280×720 camera 的配置见：

- `../ManiUniCon/configs/sensors/realsense_franka_fr3.yaml:6-75`。

仅 RGB uint8 + depth float32，不计 intrinsics/transforms和额外临时数组：

```text
one multi-camera frame = 24.609 MiB
K=3                   = 73.828 MiB / policy read
horizon=1 @ 30 Hz     ≈ 0.721 GiB/s shared→local copy
horizon=3 @ 30 Hz     ≈ 2.163 GiB/s shared→local copy
```

RealSense aggregator还把 threadpool和 OpenCV线程数设为16：

- `../ManiUniCon/maniunicon/sensors/realsense.py:367-371`。

### 影响

- camera copy、resize、GPU transfer和 inference竞争内存带宽；
- busy-spin、同步 print和大线程池引入调度抖动；
- 增加 horizon会线性放大大帧复制，而不一定增加独立信息；
- 平均 inference time正常时，p99 snapshot/build仍可能越过 action deadline。

### 对 Dex 的结论

Dex 已在 PD-16识别选择性相机读取缺口。进一步要求是把“历史元数据”和“大 payload”分离：先读取小型 header/seq/timestamp决定需要哪些帧和模态，再只复制 RGB、depth或 pointcloud中的必需部分。GPU staging、pinned memory、resize和模型推理必须在独立 benchmark中测量，不能仅从控制 Hz推断实时性。

### 验证

- 对 RGB-only、RGB-D、pointcloud分别报告 bytes/tick、copy p50/p95/p99；
- history 只增加 timestamp查询时不复制无用 payload；
- recording on/off与 inference on/off交叉测试；
- 禁止 control-loop逐 tick print，改用计数器/限频结构化日志；
- CPU affinity/thread count需要基准证明，不能由 backend自行设为全局固定值。

## 22. ManiUniCon 机制的最终移植决策与增量整改

### 22.1 可采纳、需改造和禁止照搬清单

| ManiUniCon 机制 | Dex 决策 | Dex 版本的必要变化 |
|---|---|---|
| 显式 spawn | 采纳原则 | 使用统一显式 context；所有 primitive同 context；对象在 owner child内构造 |
| Hydra `_target_` 配置组合 | 改造后采纳 | allowlist registry、resolved schema、无通用 eval、配置/hash入 manifest |
| policy child加载模型 | 采纳原则 | load→warmup→output validation→policy_ready，期间保持 DISARMED |
| BasePolicy | 只借鉴命名/职责意图 | backend与 Process/Coordinator解耦，backend无 SharedStorage写权限 |
| obs wrapper | 改造后采纳 | 输入 immutable ObservationSnapshot；纯转换；保存 preprocessing manifest |
| act wrapper | 改造后采纳 | 只产出 TypedActionCandidate；不得直接产出可执行 RobotAction |
| observation horizon | 改造后采纳 | causal time query、valid mask、source age、显式 padding |
| policy_ready | 改造后采纳 | phase/generation/capability/timeout；不是单 Event |
| robot_ready/policy_ready锁步 | 禁止照搬 | servo持续运行；用 action ACK和deadline做异步协议 |
| target timestamp | 改造后采纳 | host monotonic ns、TTL、epoch、action ID；wall只做日志 |
| action chunk | 改造后采纳 | policy-side有界 scheduler、过期丢弃、overlap规则、逐项SafetyGate |
| 256动作queue + drain-all | 禁止照搬 | 保持 arm `maxsize=2`，只发布近期有效项 |
| late chunk最后项重定时 | 禁止照搬 | expired→discard→coordinated hold |
| robot-side interpolation | 禁止照搬 | 保持 xArm Mode 6固件平滑，避免双重插值 |
| reset Event | 禁止照搬 | generation/epoch + 多参与者 ACK + old action invalidation |
| lock-free timing-budget ring | 禁止替换 Dex ring | 保留 seqlock/verified read/may-return-fewer-than-K |
| 多相机 mean timestamp | 禁止照搬 | 逐相机 capture/receive/publish/seq/valid + skew gate |
| 分布式 recorder | 禁止照搬 | 保持 policy单时钟所有权和 episode原子提交 |
| TimestampAlignedBuffer future-fill | 禁止照搬 | causal fill、valid/source_index/age、reader强制尊重 |

### 22.2 对原整改批次的增量要求

#### Batch 1：进程生命周期补充

在 PD-02至PD-07基础上增加 MU-01至MU-04、MU-15：

1. 配置对象和 live runtime对象严格分离；
2. vendor SDK/model只在 owner child构造；
3. policy backend不继承 Process；
4. 所有 ready/wait带 phase、generation、deadline和死亡传播；
5. optional dependency import测试成为 CI准入；
6. 插件 target、remote code和 checkpoint建立 allowlist/hash策略。

回滚条件：任何 backend使 main导入新增硬件/模型依赖，或 shutdown无法确认 owner process退出。

#### Batch 2：ObservationSnapshot 补充

在 PD-08、PD-09、PD-15、PD-16基础上增加 MU-05至MU-07、MU-13、MU-16：

1. history query从“latest K”升级为“按 monotonic target time的causal K”；
2. padding在 adapter层执行并携带 mask；
3. camera header先行，小元数据选择后再复制 payload；
4. 多相机保留逐 source metadata，禁止 mean timestamp替代；
5. 所有 observation time、age、skew使用同一 host monotonic域；
6. 以 bytes/tick和p99 latency验收，不只以 Hz验收。

回滚条件：实现需要改变 seqlock为基于时间预算的无验证 ring，或 history容量扩展导致控制/录制p99不可接受。

#### Batch 3：动作协议与 chunk 补充

在 PD-10至PD-13基础上增加 MU-03、MU-08至MU-10、MU-14：

1. 禁止 Event锁步控制 arm servo；
2. late/no-output统一 coordinated hold；
3. chunk保存在 policy scheduler，不扩大 arm queue；
4. overlap/replace/drop决策写入 telemetry；
5. reset/home/restart统一使用 epoch；
6. action adapter spec在启动时验证 horizon、dt、shape、unit、frame；
7. worker以 TTL/epoch/action ID二次拒绝旧动作。

回滚条件：任何实现为了“确保全部动作执行”重写过期 target，或要求 arm-side interpolation。

#### Batch 5：数据闭环补充

在 PD-18基础上增加 MU-11、MU-12：

1. recorder继续作为唯一 episode owner；
2. source sample、model fill和action validity分离；
3. stop/commit有 barrier 和每流统计；
4. aborted episode保留可诊断 temp状态；
5. quality/replay/training export默认拒绝或mask invalid/future-filled row。

回滚条件：producer自行写最终 episode文件，或新 reader把有 payload 的 invalid slot默认为有效。

### 22.3 新增离线验收矩阵

| 场景 | 必须结果 | 覆盖发现 |
|---|---|---|
| main环境缺少非所选策略依赖 | canonical import成功 | MU-01、MU-15 |
| spawn后记录构造/connect PID | SDK和model只在owner child | MU-01 |
| dummy backend尝试访问raw IPC | 无权限/接口不存在 | MU-02 |
| policy死于robot等待期间 | 有界FAULT，arm持续hold，进程可回收 | MU-03、MU-04 |
| 首个chunk到达 | ACK不能在首项执行前置位 | MU-03 |
| state/camera为空 | 保持not-ready，不先解引用 | MU-05 |
| history只有1项而请求3项 | 返回1项或2个显式invalid pad | MU-05、MU-13 |
| 两相机一新一停流 | 指定相机invalid，不能以mean timestamp通过 | MU-06 |
| wall clock前后跳变 | TTL/target/age不变 | MU-07 |
| 整个chunk已过期 | 丢弃并发布当前hold，不重定时旧预测 | MU-08 |
| 新旧chunk重叠 | 按policy记录replace/drop，queue仍有界 | MU-09 |
| reset发生在推理中 | epoch增加，所有历史/queue/ACK换代 | MU-10 |
| stop后立即start下一episode | 数据不串集，旧commit完成后新start | MU-11 |
| 首样本晚到3.5个grid | 过去slot不得标source-valid | MU-12 |
| ring在高负载下wrap | 无torn frame，返回实际verified数量 | MU-13 |
| horizon/dt/synchronized配置冲突 | DISARMED启动失败并指出字段 | MU-14 |
| 四相机history benchmark | bytes/tick及copy/build p99满足门槛 | MU-16 |

### 22.4 本轮新增离线核查记录

```text
ManiUniCon source commit:
  85c6f2e32ecf9f2bed62d202b058c39623444686

Timestamp accumulator gap reproduction:
  local_idxs  = [0, 0, 0, 0]
  global_idxs = [0, 1, 2, 3]
  next         = 4

All-stale chunk reproduction:
  original targets = [10.0, 10.1, 10.2]
  now              = 10.35
  output count     = 1
  rewritten target = 10.4
  selected action  = last prediction

Optional-dependency isolation check:
  importing maniunicon.utils.timestamp_accumulator through the normal package
  failed because the package eagerly imported optional MeshCat utilities

Four-camera payload calculation (1280x720, RGB uint8 + depth float32):
  24.609 MiB/frame
  73.828 MiB/K=3 read
  0.721 GiB/s at horizon=1, 30 Hz
  2.163 GiB/s at horizon=3, 30 Hz
```

这些复现均未初始化相机、机器人 SDK、CUDA model或网络连接。

### 22.5 综合判定

ManiUniCon 对 DexMani Real 最有价值的不是其具体 Robot loop，而是它证明了后续策略部署确实需要以下一等概念：

- 可配置的 model/observation/action组件；
- policy 子进程内模型加载；
- observation history；
- action chunk与target timestamp；
- 明确的策略就绪阶段。

但第二轮深审也证明，这些概念的参考实现不足以承载 Dex 的安全语义：history没有因果对齐，ready不是可靠 ACK，wall target不是安全 deadline，chunk迟到会被重新授权，reset没有世代，distributed recording没有事务，通用 Robot对象破坏 SDK owner边界。

因此最终路线应是“借鉴能力模型，重建安全协议”：

```text
spawn + child-local factory
→ phase-aware policy readiness
→ causal ObservationSnapshot
→ pure model backend / typed adapters
→ mandatory SafetyGate
→ monotonic epoch/action/TTL chunk scheduler
→ bounded arm queue + latest hand ring + paired ACK
→ policy-owned atomic episode recorder
```

任何为了快速接入 ManiUniCon policy而绕过这条链路的兼容层，都只能用于离线 dry-run，不能取得 live arm/hand IPC写权限。只有 MU-01至MU-16 对应的离线矩阵和原 PD-01至PD-20 的放行门槛同时满足后，DexMani Real 才具备安全部署 action-chunk、VLA 和多模态时序策略的基础。
