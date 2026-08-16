# Hand Retarget 当前控制合同与实现说明

> 最近静态审阅：2026-08-17
>
> 实现基线：当前工作树；源码与运行配置始终是最终事实来源
>
> 适用范围：Quest 右手关键点写入共享内存后，到 XHand 命令跨越 SDK 边界之前
>
> 安全说明：本文只描述静态实现；没有连接、探测或驱动真实硬件

## 1. 本文的用途与边界

本文是 hand retarget 的**当前实现合同**，回答以下问题：

- 输入观测如何进入求解器；
- TAG 与 DexPilot 分别优化什么；
- 求解结果经过哪些整形、验证和发布边界；
- 各类时间状态在什么时候推进；
- 失败时机械臂、手和记录分别发生什么；
- 当前实现没有提供哪些碰撞、接触和可复现性保证。

本文不把未来设计写成当前能力。所有尚未实现的建议集中在第 14 节，并明确标记为 backlog。

源码和运行配置始终是最终事实来源。本文刻意不复制所有默认值、依赖版本和 dtype 字段，以减少实现变化后的文档漂移。

## 2. 当前合同摘要

当前链路为：

```text
Quest HTS 右手关键点
    │
    ▼
Unity 坐标系 → FLU 坐标系
    │
    ▼
VR_FRAME_DTYPE / 历史可验证 VR ring
    │
    ▼
16 Hz 因果观测选择与新鲜度检查
    │
    ├── 陈旧、跨 generation 或未完成重锚：停止发布或 hold
    │
    ├── 同一 VR ring sequence：复用上次 solve，不重复推进 solver state
    │
    ▼
21×3 shape / finite / 几何退化检查
    │
    ▼
掌心局部坐标系 → MANO 轴约定 → 自适应小指补偿
    │
    ├── TAG：两阶段 Pinocchio + NLopt，默认后端
    └── DexPilot：外部 dex-retargeting，可选后端
    │
    ▼
startup smoothstep ramp
    │
    ▼
operational command-box clip
    │
    ▼
结构 / finite / operational / mechanical 复验
    │
    ▼
ActionCandidate → SafetyGate → HAND_COMMAND_DTYPE
    │
    ▼
latest-wins hand command ring
    │
    ▼
30 Hz hand worker：generation / 过期 / 范围复验
    │
    ▼
XHand driver：最终范围检查，原样发送 endpoint
```

最重要的当前事实是：

1. 正常控制只消费**右手 21 个关键点**；手腕四元数服务于机械臂映射，不参与手指 retarget。
2. TAG 是默认后端；DexPilot 是兼容保留的可选后端。
3. TAG 当前**没有输出级 EMA**。它的连续性来自 Stage 1 时间正则、pinch activation EMA 和启动 ramp。
4. DexPilot 默认保留外部库内部 LPFilter；仓库包装层的 output EMA 当前为直通。
5. operational command bounds 当前是**发布前逐关节投影**，不是求解器边界，也不是整条拒绝条件。
6. 应用侧 command-to-command delta clamp（`hand.max_delta_rad`）已移除；手部速度保护仅由 EtherCAT 固件 PID 与电流限位承担。
7. 结构错误、NaN/Inf、机械范围错误和 IPC/lifecycle 错误仍然 fail closed，不会被裁成合法动作。
8. solver、ramp、published command 和 SDK accepted command 有不同的状态推进时机，不能统称为“按已发布命令推进”。
9. teleop 先完成本周期 hand solve、ramp、clip 和 sanitizer，再把该 shaped hand endpoint 放入机械臂 IK/collision model。
10. tactile 进入反馈与记录，但当前不闭环调节捏合。

## 3. 所有权、时钟和数据流

### 3.1 进程所有权

| 层 | 当前所有权 |
|---|---|
| VR worker | 接收 HTS、Unity→FLU、基础 shape/finite 校验、写 VR ring |
| teleop | 控制网格、因果观测、retarget、ramp、command shaping、候选发布和记录选择 |
| hand worker | 消费固定 dtype 命令、生命周期复验、拥有 XHand driver |
| XHand driver | 拥有 SDK、最终命令边界检查、设备发送和反馈读取 |
| RecorderIO | 序列化、校验并发布 teleop 已选择的固定网格样本 |
| main/lifecycle | readiness、监督、safety state、generation 边界和 shutdown |

因此：

- worker 不根据 VR 重新求手型；
- RecorderIO 不按到达时间选择动作；
- teleop、main 和 recorder 不持有 XHand SDK；
- hand retarget 的控制决策不下放到设备进程。

主要调用路径：

| 职责 | Source of truth |
|---|---|
| VR 固定协议 | [utils/schema.py](../dexmani_real/utils/schema.py) |
| 因果观测 | [shm/causal_reader.py](../dexmani_real/shm/causal_reader.py) |
| teleop 决策 | [teleop/loop.py](../dexmani_real/teleop/loop.py) |
| 手命令整形 | [teleop/hand_control.py](../dexmani_real/teleop/hand_control.py) |
| retarget 包装 | [teleop/hand_retarget.py](../dexmani_real/teleop/hand_retarget.py) |
| TAG 求解器 | [teleop/tag_retargeting/optimizer.py](../dexmani_real/teleop/tag_retargeting/optimizer.py) |
| TAG 梯度 | [teleop/tag_retargeting/pin_grad.py](../dexmani_real/teleop/tag_retargeting/pin_grad.py) |
| 发布与 worker 校验 | [policy/safety.py](../dexmani_real/policy/safety.py) |
| hand worker | [robot/hand_process.py](../dexmani_real/robot/hand_process.py) |
| XHand driver | [robot/xhand.py](../dexmani_real/robot/xhand.py) |

### 3.2 三个默认频率

| 时钟 | 默认频率 | 作用 |
|---|---:|---|
| coordinator | 64 Hz | 事件、健康检查和细粒度协调 |
| control / record grid | 16 Hz | arm + hand 动作决策与 episode 对齐 |
| hand worker | 30 Hz | 读取最新命令、发送 XHand、读取反馈 |

频率定义在 [config/defaults.py](../dexmani_real/config/defaults.py)，运行时由 [config/runtime.py](../dexmani_real/config/runtime.py) 解析和校验。

### 3.3 两类 ring 语义

VR ring 保存可回看的观测历史。控制网格只选择锚点之前已经发布的最新观测，使用本机 monotonic time 保证因果关系。

hand command ring 是 latest-wins：

- teleop 发布 endpoint；
- worker 只消费尚未处理的最新序号；
- 中间 endpoint 可以被跳过；
- 命令不会作为轨迹队列逐条回放。

这适合实时 servo endpoint，但意味着 worker 可能跳过中间命令，因此仍必须在设备边界复验。

### 3.4 generation 与命令寿命

`run_generation` 标识逻辑控制时代。begin、pause、home 和反馈故障等命令静默边界会使旧命令失效；worker 在 SDK 边界前拒绝旧 generation。camera stall 会废弃当前 episode，但不会自行推进 generation；下一次显式 BEGIN 才推进。

命令同时带 monotonic 有效期。当前 hand worker delivery window 由 [policy/safety.py](../dexmani_real/policy/safety.py) 编码为 target time 后 300 ms。

generation 与有效期解决不同问题：

- generation 拒绝逻辑时代错误；
- valid-until 拒绝同一时代内过晚到达的命令。

不要把 worker delivery window 与 ActionCandidate 自身的 policy validity window 混为同一个期限。

## 4. 状态推进合同

当前链路没有一个覆盖所有状态的事务提交点。各状态的推进时机如下：

| 状态 | 所有者 | 当前推进时机 | reset / 失效时机 |
|---|---|---|---|
| hand retarget observation cache | teleop | 新 VR ring sequence 首次进入 retarget 时 | command quiescence / reanchor |
| TAG `last_qpos` | TAG optimizer | Stage 1 或 Stage 2 成功返回时 | retargeter reset |
| TAG `pinch_factors` | TAG optimizer | Stage 1 成功后、Stage 2 判断前 | retargeter reset |
| DexPilot warm start / internal LP | dex-retargeting | 外部 `retarget()` 成功调用时 | wrapper reset |
| wrapper output EMA | DexPilot wrapper | 仅当 alpha < 1 时；当前默认直通 | wrapper reset |
| hand ramp step | teleop | 进入 ramp shaping 分支时 | reanchor、ramp 完成或显式清理 |
| `ctx.prev_hand_qpos` | teleop | 候选发布成功后 | home / reanchor seed |
| worker command-ring cursor | hand worker | 读取一个新 hand command ring sequence 时 | worker 生命周期 |
| driver `last_qpos_cmd` | XHand driver | SDK send 成功后 | connect / home 初始化 |

由此得到四个必须区分的结论：

1. 非法 landmarks 在调用求解器前被拒绝，不推进 solver 状态。
2. 同一 VR ring sequence 即使被多个 16 Hz 控制网格选中，也只调用一次有状态 retargeter；后续网格复用 solve 结果，ramp 仍按控制网格推进。
3. 合法 landmarks 得到 solver 结果后，即使后续整形、SafetyGate 或发布失败，solver temporal state 也可能已经推进。
4. `ctx.prev_hand_qpos` 和 driver `last_qpos_cmd` 分别表示最后发布成功和最后设备调用接受的目标，不等同于最新 solver 候选。

cache key 是 shared VR ring 的 verified `ring_sequence`。成功和失败都会占用该 sequence；成功命中返回 solved endpoint 的副本并保持 `retarget_ok=True`，失败命中则返回调用时最新的 `ctx.prev_hand_qpos`。每个控制 tick 仍创建新的 `action_id`，所以复用 observation 不等于重复 hand command ring publication。

当前 ramp step 也在发布前推进。若 ramp 期间 retarget 返回 hold，或后续发布失败，该控制帧仍可能消耗一个 ramp step。分析启动阶段时不能假设 ramp 帧数等于成功发布数。

## 5. 输入、坐标系和小指补偿

### 5.1 右手输入合同

[VR_FRAME_DTYPE](../dexmani_real/utils/schema.py) 保存：

- wrist position；
- wrist quaternion；
- 21×3 landmarks；
- source/local monotonic 时间与 sequence；
- side 和 head 元数据。

当前 VR publication 由右手 frame 驱动，teleop retargeter 也固定使用 right-hand 资产。schema 中保留 side 字段不代表系统已经实现左右手对称控制。

schema 没有逐关键点 confidence、tracked bit 或遮挡标记，因此当前质量 gate 只能依据整帧新鲜度和几何一致性。

### 5.2 几何 gate

`validate_landmarks()` 在任何 temporal solver state 之前检查：

1. shape 必须为 21×3；
2. 全部值必须 finite；
3. wrist→index MCP 和 wrist→pinky MCP 至少 1 cm；
4. 两条掌向量夹角正弦至少 0.1；
5. 五条 landmark chain 上最短相邻骨段至少 2 mm。

尚未覆盖：

- 最大手掌或骨段尺度；
- 跨帧瞬移、冻结和速度；
- 逐点追踪置信度；
- chirality；
- wrist/index/middle 三点 SVD 的直接 condition-number gate。

几何 gate 使用 index/pinky，而实际 palm SVD 使用 wrist/index/middle。这是当前边界，不能把前者理解为后者的完整数值稳定性证明。

### 5.3 掌心局部坐标系

掌心 frame 使用 wrist、index MCP 和 middle MCP：

- 三点中心化后做 SVD，取得掌面法向；
- `wrist - middle MCP` 构造纵向轴；
- Gram–Schmidt 正交化；
- index/middle 横向参考统一法向符号；
- 返回列向量基 `[x, normal, z]`。

源码中表达式 `wrist - middle` 的方向是从 middle 指向 wrist。理解实现时应以表达式为准。

右手 operator→MANO 旋转为：

$$
R_{operator\to mano}=
\begin{bmatrix}
0&0&-1\\
-1&0&0\\
0&1&0
\end{bmatrix}
$$

其 determinant 为 +1，是 proper rotation。Unity→FLU 的 chirality 转换已经在 VR producer 完成，retarget 不应再次镜像。

两种后端最终都使用相对位置或差向量，因此全局手腕平移不会进入手关节优化；空间手腕运动由机械臂映射处理。

### 5.4 自适应小指补偿

公共预处理依据 pinky MCP→TIP 距离，在 3–10 cm 区间把 scale 从 1.2 线性提升到 2.2，并用原始 MCP→PIP→DIP→TIP 骨段逐段重建。

重建使用未修改的原始骨段方向，输入数组不会原地修改。TAG 和 DexPilot 后续主要消费 wrist 与 fingertips，因此中间小指点的直接作用是构造一致的最终 pinky tip。

该补偿会把 MCP→TIP 距离噪声同时带入 scale 和目标长度；在阈值附近还存在分段函数斜率变化。它是当前经验补偿，不是用户手型标定。

## 6. TAG 后端

### 6.1 模型与关节顺序

TAG 使用 [xhand_right.urdf](../assets/robots/xhand/xhand_right.urdf)，通过 Pinocchio FreeFlyer 模型计算 FK 和 Jacobian：

- generalized configuration 为 7 维 FreeFlyer + 12 维手关节；
- FreeFlyer 固定为零平移和单位四元数；
- 优化变量只有 12 个手关节；
- Jacobian 丢弃前 6 个 floating-base velocity columns。

Pinocchio model order 与 XHand SDK order 不同。跨进程手关节向量的 canonical SDK 名称与顺序由 [utils/schema.py](../dexmani_real/utils/schema.py) 定义；TAG 按运行时模型名称构造映射，并使用逆置换完成 measured SDK qpos→optimizer warm start。DexPilot YAML 的名称顺序在加载时必须与该常量精确一致，否则初始化 fail closed。

固定的 model→SDK index 关系当前为：

```text
[9, 10, 11, 0, 1, 2, 3, 4, 7, 8, 5, 6]
```

规划侧对应关系以 [planning/constants.py](../dexmani_real/planning/constants.py) 为准。

### 6.2 当前优化边界

TAG 当前使用独立 URDF 的机械 joint limits 作为 NLopt box bounds，不再与 operational command floor 取交集。

这样 measured warm start 可以位于 operator-set command floor 之下，例如设备对 5° 命令实际落在略低位置时，不会仅因反馈误差被投影到更严格的 optimizer box。

代价是 solver raw 结果可能位于 operational command floor 之外；teleop 随后把最终发布目标投影到 command box。因此：

- solver 的 FK/loss 对应 raw 姿态；
- 设备实际收到的是 shaped command；
- 两者在 bound saturation 时可能持续不同。

### 6.3 五指目标

TAG 使用 thumb/index/middle/ring/pinky 五个 fingertip，相对 wrist 居中，旋转到 URDF frame，然后按每指 robot/human length ratio 和全局 boost 缩放。

目标是整条 wrist-to-tip 向量的标量缩放，包括掌宽方向 offset；当前没有把掌宽、掌长和指骨长度拆成独立的人体标定参数。

### 6.4 Stage 1：指尖位置与时间正则

Stage 1 最小化：

$$
L_1(q)=
\sum_{i=0}^{4}\|p_i(q)-t_i\|^2
+\lambda_s\|q-q_{prev}\|^2
$$

当前默认 `smooth_weight` 为 0.02，solver 为 NLopt L-BFGS，最大 evaluation 80。

解析梯度为：

$$
\nabla L_1=
2\sum_i J_i^T(p_i-t_i)
+2\lambda_s(q-q_{prev})
$$

Stage 1 抛异常或返回非法数组时，本次 retarget 失败；调用者保持上一条手命令。

### 6.5 pinch activation

Stage 1 成功后，TAG 在**未做 robot/human length scaling**的 fingertip targets 上计算 thumb 与其余四指距离：

- 距离 ≥30 mm：目标 activation 为 0；
- 距离 ≤8 mm：目标 activation 为 1；
- 中间线性变化；
- activation 使用 alpha 0.4 的 EMA；
- 最大 activation 小于 0.01 时跳过 Stage 2。

这是连续权重，不是有独立 enter/exit threshold 的迟滞状态机。

### 6.6 Stage 2：捏合精修

Stage 2 从 Stage 1 解开始，优化：

$$
L_2(q)=
w_a\|q-q_{s1}\|^2
+w_t\|q-q_{prev}\|^2
+\sum_{i=1}^{4}w_p a_i^2
\|p_i(q)-p_{thumb}(q)\|^2
$$

当前默认权重为：

- Stage 1 anchor：1.0；
- temporal：0.8；
- pinch base：2000。

相对 Jacobian 使用 `J_i - J_thumb`。Stage 2 使用 SLSQP，最大 evaluation 100；失败时回退 Stage 1，不使整次 retarget 失败。

Stage 2 的目标是指尖 frame origin 的零距离，不包含：

- 指腹几何或非零接触间距；
- 接触法向；
- tactile；
- 力或力矩目标；
- 手内自碰项。

多个 activation 同时为正时，多根手指会同时被吸引到 thumb。该目标更准确地称为视觉指尖闭合强化，而不是物理接触优化。

### 6.7 TAG 输出和 reset

当前 TAG 成功路径为：

```text
optimizer model-order q
    → SDK order
    → last_raw_qpos
    → 返回 teleop
```

当前没有 TAG output EMA。`reset(measured_sdk_qpos)` 会：

- 清空 `last_raw_qpos`；
- SDK order 映射到 model order；
- 把 warm start 裁入 URDF mechanical bounds；
- 清空 pinch activation 和 Stage 1 cache。

## 7. DexPilot 后端

DexPilot 使用：

- 与 TAG 共用的机械范围 [xhand_right.urdf](../assets/robots/xhand/xhand_right.urdf)；
- [xhand_right_dexpilot.yml](../assets/retargeting/xhand_right_dexpilot.yml)；
- 10 条 fingertip-to-fingertip vector；
- 5 条 wrist-to-fingertip vector。

公共 palm/MANO 和 pinky 预处理完成后，外部 dex-retargeting 构造 reference vector graph，并以 SLSQP、鲁棒几何误差和时间正则求解。

DexPilot 的 optimizer 边界主要来自机械范围 `xhand_right.urdf` 和外部库处理，与 TAG 一样不把 operational command floor 混入求解器。仓库只在发布前对最终 shaped command 应用 operational clip 和独立复验。

当前滤波合同：

| 层 | 当前默认 |
|---|---:|
| dex-retargeting internal LPFilter | alpha 0.6 |
| 仓库 wrapper output EMA | alpha 1.0，直通 |

因此 DexPilot 当前不是文档历史版本中的“两层持续 EMA”。它仍然有外部库内部 LPFilter。

`reset()` 会清除外部 optimizer warm start、internal LPFilter、projected flags 和 wrapper EMA state，并尝试用 measured SDK qpos 重建内部顺序的 warm start。

### 7.1 两后端语义差异

| 维度 | TAG | DexPilot |
|---|---|---|
| 默认状态 | 默认 | 可选 |
| 目标 | 5 条 wrist-to-tip | 15 条 reference vectors |
| pinch | 独立 Stage 2 | 外部 projected-vector 机制 |
| 持续平滑 | 时间正则 + activation EMA | 外部 internal LPFilter |
| wrapper output EMA | 无 | 当前直通 |
| optimizer bounds 来源 | mechanical URDF | mechanical URDF / 外部实现 |
| Stage 2 回退 | 有 | 由外部实现决定 |
| raw 可观测性 | 明确的 solver SDK-order 输出 | 当前只得到外部 retarget 返回值 |
| 可审计性 | 目标和梯度在仓库内 | 部分语义依赖安装版本 |

不要用同一个“raw”或“low-pass alpha”术语假设两个后端处于相同处理阶段。

## 8. 命令整形、验证和发布

### 8.1 五阶段术语

本文统一使用：

```text
observed → solved → shaped → published → accepted / measured
```

- observed：控制网格选中的 VR landmarks；
- solved：retargeter 返回的 SDK-order 候选；
- shaped：ramp 和 command-box clip 后的 endpoint；
- published：通过候选验证并写入 hand command ring 的 endpoint；
- accepted：XHand SDK send 成功后 driver 保存的 endpoint；
- measured：设备反馈 qpos。

### 8.2 retarget failure

`_compute_hand_command()` 在以下情况返回上一条 published hand command，并令 `retarget_ok=False`：

- hand 不可用；
- retargeter 不存在；
- landmarks 缺失或非法；
- VR ring sequence 缺失或非法；
- backend 抛异常或返回 `None`；
- 输出不是 finite `(12,)`。

这类失败不是结构非法命令；它产生一个合法 hold endpoint。

因果 reader 可能在相邻控制网格返回同一个 VR ring sequence。teleop 对每个 sequence 最多调用一次有状态 backend：成功结果被缓存供后续网格复用，失败结果也不会在同一观测上重试。该缓存只约束 solver；startup ramp 仍按 16 Hz 控制网格推进。

### 8.3 startup ramp

默认 ramp duration 为 0.5 s，在 16 Hz 下通过四舍五入得到 8 个 shaping step。第 k 步使用 smoothstep：

$$
u=\frac{k+1}{N},\qquad
w=u^2(3-2u)
$$

$$
q_{ramp}=q_{start}+w(q_{live}-q_{start})
$$

`q_live` 在新 VR observation 成功 solve 时可以变化；重复选择同一 ring sequence 时保持为缓存值。因此这不是固定终点的离线轨迹，但也不会因同一观测被重复选中而反复滤波。最后一个 configured step 到达该控制帧使用的 live target。

当前 ramp step 在发布前推进，不保证每一步都成为 published command。

### 8.4 command shaping

ramp 后依次执行：

1. 对 operational command lower/upper bounds 做 `np.clip`；
2. 调用 sanitizer 复验 shape、finite、limit nesting、operational 和 mechanical。

需要区分两种语义：

- operational clip 是正常、确定性的策略整形；
- 结构、finite、mechanical envelope 或配置合同错误触发 reject/hold。

operational clip 会改变 solver 手型，尤其在 distal command floor 附近。当前 `retarget_ok` 仍表示 solver 是否成功，不表示 solved endpoint 被原样发布。

### 8.5 SafetyGate 与发布边界

shaped endpoint 进入 `ActionCandidate`，再通过统一发布边界：

- generation；
- shape 和 finite；
- arm/hand joint limits；
- workspace 等候选级检查；
- hand mechanical envelope；
- runtime state 和反馈新鲜度。

通过后，hand endpoint 编码为固定 `HAND_COMMAND_DTYPE` 并写 latest-wins ring。控制路径不使用 JSON、动态对象或 ACK/apply 事务。

## 9. 失败耦合语义

| Arm IK | Retarget | shaped hand | 当前结果 |
|---|---|---|---|
| 成功 | 成功 | 合法 | 发布新 arm + shaped hand |
| 成功 | 失败 | 上一 hand 合法 | arm 可继续，hand hold，记录 retarget failure |
| 成功 | 成功 | sanitizer 非法 | coupled action 不发布，发布 arm hold 路径 |
| 失败 | 成功 | 合法 | arm hold，允许 hand-only 发布 |
| 失败 | 失败 | 上一 hand 合法 | arm hold + hand hold |
| 任意 | 任意 | IPC/lifecycle 非法 | 不跨越共享命令边界 |

重要区别：

- `retarget_ok=False` 不等于 hand endpoint 非法；
- `retarget_ok=True` 不等于 endpoint 未被 ramp 或 clip 修改；
- arm IK 失败不必冻结手；
- hand sanitizer 失败会阻止一个不一致的 coupled arm/hand 发布。

## 10. hand worker 与 XHand driver

hand worker 读取最新未处理 hand command ring sequence，并在 SDK 边界前检查：

- 固定 dtype、shape 和 finite；
- operational、mechanical 和 rated envelope 的嵌套与 endpoint；
- generation；
- valid-until；
- safety state、error state 和 e-stop。

worker 只在允许的 safety state 下发送。persistent send/read/board faults 进入共享 fault 路径；`error_state` 是 sticky 的系统错误标志。

worker 的本地 consumed cursor 只表示 hand command ring publication；反馈字段 `HAND_STATE_DTYPE.last_cmd_seq` 表示最后一次 `XHand.send_action()` 成功对应的 `action_id`。两者名称相近但不属于同一序号空间。

XHand driver 再次检查 shape、finite 和范围。检查通过后 endpoint 原样写入 SDK command；应用层不在 driver 中插值。

只有 SDK send 成功，driver 才更新 `last_qpos_cmd`，其值为设备调用接受的上一目标，不是 measured feedback，也不是 teleop 最新 solved target。

## 11. 碰撞与 tactile 边界

### 11.1 teleop 当前碰撞手姿态

teleop 先计算本周期 hand target，并完成 ramp、operational clip 和 sanitizer，然后调用：

```text
planner.set_hand_qpos(hand_cmd)
```

因此机械臂 IK/collision 使用：

$$
collision(q_{arm}^{candidate,t}, q_{hand}^{shaped,t})
$$

如果本周期 retarget 失败，`hand_cmd` 是上一条 published hand endpoint；否则它是本周期 shaped endpoint。arm IK 失败时，合法 hand-only endpoint 仍可独立发布。

当前联合模型保留 arm–hand 活跃碰撞对，但 hand–hand 自碰对被禁用。hand retarget 目标本身也没有碰撞项。

[planning/collision_model.py](../dexmani_real/planning/collision_model.py) 已提供 arm/hand transition envelope 检查，并由 replay dense preflight 使用；teleop 16 Hz 在线路径当前没有调用它。

由于 arm queue 和 hand latest-wins ring 独立、执行频率和固件动态不同，未来在线检查若只沿同一个 interpolation alpha 检查“同步轨迹”，不能完整覆盖异步可达组合。

### 11.2 tactile

XHand tactile/contact 数据进入反馈和 episode，但当前不会：

- 改变 pinch activation；
- 调整 Stage 2 权重；
- 在接触后停止闭合；
- 形成力或力矩闭环。

所以视觉闭合、物理接触和稳定抓取是三件不同的事情。

## 12. 记录、raw/final 和 replay

HDF5 schema v16 保存与 hand retarget 相关的主要信息：

| 层 | 主要字段语义 |
|---|---|
| observed | 网格对齐的原始 VR landmarks |
| solved diagnostic | 正常 active frame 的 `action_hand_joint_raw` |
| published selection | 最终 `action_hand_joint` |
| measured | `hand_qpos` 与 hand fingertip FK |
| status | retarget flag、frame status、held/safety 标志 |
| timing | hand retarget wrapper elapsed time |

当前 raw 语义：

- TAG：solver 输出，SDK order，ramp 和 command shaping 之前；
- DexPilot：外部 retargeter 返回值，已经包含 external internal LPFilter；wrapper EMA 当前默认直通；
- retarget failure：raw helper 回退为当前 hold endpoint。
- held / safety-fallback frame：当前 held recorder path 没有接收被拒绝的 solver raw，字段会退回 hold action。

因此两个后端的 raw 不能作为完全相同的优化阶段直接比较，也不能依靠 held frame 的 raw 恢复被拒绝候选。

`action_hand_joint` 表示 teleop 最终选择并记录的 endpoint。它可能与 raw 不同，原因包括：

- startup ramp；
- operational clip；
- normal active path 上的 retarget failure hold。

当前没有单独记录每关节 clip/saturation bitmask；held path 也不保留 rejected solver raw。因此只有正常 active frame 能通过 raw/final 差值和 frame flags 间接解释 shaping。

replay 使用记录的最终 hand action，不从 VR landmarks 重新执行 retarget。这复现的是采集时选择的 endpoint，而不是在当前依赖版本下重新解释人手动作。

## 13. 可复现性和实时效率边界

### 13.1 provenance

[examples/collect_teleop.py](../examples/collect_teleop.py) 当前记录联合机器人 URDF、SRDF 和 calibration 等资源哈希，但没有完整覆盖：

- hand retarget 共用的独立 `xhand_right.urdf`；
- DexPilot YAML；
- dex-retargeting、Pinocchio、NLopt 版本；
- solver result code、loss、evaluation count；
- pinch activation、bound saturation 和 warm-start clipping。

因此最终 action replay 有定义；只依赖 episode metadata 从 landmarks 位级重算相同 action，目前没有严格保证。

### 13.2 实时预算

16 Hz 控制周期名义预算为 62.5 ms。当前记录了单帧 retarget elapsed time；复用同一 VR observation 的控制帧只计 cache lookup，不代表执行了一次 solver。仓库合同没有把以下指标固化为回归门槛：

- Stage 1 / Stage 2 分项耗时；
- P50/P95/P99；
- maximum evaluation hit rate；
- control deadline miss；
- Stage 2 activation 与耗时的关系；
- 日志洪泛对控制周期的影响。

单个 synthetic case 成功不能证明真实追踪分布下的实时性。

## 14. 后续实现 backlog（当前尚未实现）

本节只描述建议，不代表当前代码具有这些能力。

### P1：正确性和可观测性

1. 将 `solver_ok`、`command_shaped`、`published`、`accepted` 和 `measured` 分层记录或形成明确派生规则。
2. 增加 per-joint operational clip bitmask、raw-to-final 差值和 joint-at-bound 指标。
3. 明确 solver temporal state 已推进但 publish 失败时的恢复政策：接受分叉、显式 reset，或设计 propose/commit 接口。
4. 将实际选择后端的 URDF、YAML 和关键原生依赖版本纳入 episode provenance。
5. 记录 Stage 1/2 result code、loss、evaluation count、Stage 2 是否运行/回退和 pinch activation。

### P1：离线回归和效率

至少覆盖：

- landmark 退化拒绝且 solver state 不推进；
- 同一 VR ring sequence 的成功、失败和异常分支都只调用一次 backend，cache reset 后才允许重新调用；
- 合法 solve 后 publish failure 的 temporal-state 语义；
- palm basis 正交性、determinant 和 conditioning；
- pinky scale 的阈值、同比重建和噪声响应；
- SDK/model order round trip；
- TAG Stage 1/2 梯度；
- reset 后首帧连续性；
- operational clip 和机械范围 reject；
- latest-wins 跳帧相对 last-accepted 的跳变幅度；
- P50/P95/P99 solver 与完整 policy tick 延迟。

### P2：输入质量

1. 若 HTS 可提供，向固定 schema 增加 tracked/confidence；schema 变化必须协调所有 producer、consumer 和 HDF5 版本。
2. 增加最大尺度、跨帧跳变、冻结和 chirality 检查。
3. 直接检查 wrist/index/middle SVD conditioning。
4. 评估用户掌宽、掌长和各指长度标定，避免只依赖固定 wrist-to-tip ratio。

### P2：算法和接触质量

1. 评估中间关节、fingertip orientation 或 finger-pad frame 是否改善弱可观测关节和接触姿态。
2. 评估非零接触距离与多指同时 activation 的冲突。
3. tactile 若进入控制，先定义传感器失效、接触终止、sticky fault、记录和 replay 语义。
4. 不允许 tactile 绕过 operational/mechanical command boundary。

### P2：碰撞语义

1. 基准测试在线 arm/hand endpoint 或 transition-envelope 检查成本。
2. 若增加在线检查，应覆盖独立执行的保守组合，而不只检查同步插值。
3. 审计 hand–hand collision 全禁用的依据；若继续禁用，明确由机械耦合、command bounds 还是 firmware 承担约束。
4. 任何新增规划检查仍由 teleop/planning 所有，不下放到 hand worker，也不增加应用侧 arm 插值。

## 15. 修改检查清单

### 15.1 修改关键点或坐标系

- VR SDK 原始约定；
- Unity→FLU；
- `VR_FRAME_DTYPE`；
- causal reader；
- palm basis 与 MANO transform；
- backend target indices；
- 左右手 chirality；
- raw landmarks 记录和离线工具。

### 15.2 修改关节顺序或 shape

- runtime bounds；
- SDK/model mapping；
- DexPilot YAML joint names；
- planning constants 和 collision-model order；
- hand command/state dtype；
- recorder、reader、visualization 和 replay。

持久化 shape 或含义变化不能静默写入 HDF5 v16。

### 15.3 修改滤波、ramp 或 temporal state

- solver state 的推进点；
- VR observation cache 是否保证每个 ring sequence 最多 solve 一次；
- reset 是否清除所有 backend state；
- ramp step 是否按 attempt、publish 或 accept 推进；
- raw 字段阶段；
- latest-wins 跳帧；
- measured/action 相位差；
- replay 仍只消费 final action。

### 15.4 修改 bounds 或 command shaping

从 [config/defaults.py](../dexmani_real/config/defaults.py) 和 [config/runtime.py](../dexmani_real/config/runtime.py) 开始，再审计：

- optimizer bounds；
- teleop clip；
- sanitizer；
- SafetyGate/publication preflight；
- hand worker；
- XHand driver；
- home；
- replay preflight；
- metadata 和本文档。

## 16. 历史语义说明

旧审计基线 `d97c436` 曾具有以下行为：

- TAG 和 DexPilot wrapper output EMA 默认 alpha 0.5；
- TAG optimizer bounds 为 URDF 与 operational bounds 的交集；
- hand `max_delta_rad` 默认 0.20 rad；
- teleop 对 operational/delta 违规采用整条拒绝而非正常 clip；
- hand command delivery lifetime 的表述约为 0.5 s。

这些历史行为解释了旧 episode、旧日志和旧讨论中的术语，但不应再用于描述当前运行合同。

## 17. 总结

当前 hand retarget 的核心不是一个孤立优化器，而是：

```text
因果观测
→ verified VR ring sequence 去重 / solve cache
→ 几何 gate
→ backend solve
→ deterministic shaping
→ candidate validation
→ latest-wins publication
→ worker/driver boundary
→ fixed-grid recording
```

TAG 提供仓库内可审计的五指位置匹配和条件式捏合精修；DexPilot 提供外部 vector-graph 路径。当前输出机制已经简化为：TAG 不再增加 output EMA，DexPilot wrapper 默认直通，启动阶段使用 ramp，operational command floor 通过发布前 clip 实现，应用侧 command-to-command delta clamp 已移除。

需要长期保持清晰的不是某个默认数字，而是以下边界：

- observed、solved、shaped、published、accepted 和 measured 不同；
- solver state、ramp state、published state 和 driver state 有不同推进点；
- retarget success、command shaping、safety rejection 和 lifecycle invalidation 是不同事件；
- 视觉指尖闭合不等于物理接触、碰撞安全或抓取稳定；
- replay final action 有定义，从 landmarks 重算相同 action 仍受资产和依赖 provenance 限制。
