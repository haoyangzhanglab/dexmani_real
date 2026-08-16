# 机器人遥操作 Episode 数据清洗与策略学习数据构建参考规范

## 1. 目的与适用范围

本文面向通过**机器人遥操作采集 Episode 示范轨迹，并用于行为克隆、ACT、Diffusion Policy、VLA 等策略学习**的数据工程系统。

本文重点讨论：

- 数据处理原则；
- 数据层级划分；
- Episode 清洗与质量控制；
- 停止、抖动、漂移等问题的处理思想；
- 时间与 Action 语义管理；
- 有效训练样本的生成机制。

本文不规定具体：

- 文件格式；
- 滤波算法；
- 运动阈值；
- 采样频率；
- 坐标表示；
- 机器人控制接口。

这些实现应根据具体硬件、控制方式和训练算法决定。

---

# 2. 核心目标

遥操作数据处理的目标不是得到一条“更平滑”的轨迹，而是得到：

> **时间可信、物理语义明确、任务行为可信、质量状态可知、训练采样可控的数据。**

可以进一步概括为：

```text
语义正确
   ↓
时间正确
   ↓
质量可知
   ↓
采样可控
   ↓
训练视图可派生
```

其中前三项属于数据基础设施，后两项属于策略学习接口。

---

# 3. 五条核心原则

## 3.1 Raw 数据不可逆

原始采集数据应尽量忠实保存，不因为当前训练需求而覆盖或重写。

Raw 层至少应保留：

- 原始图像或视频；
- 机器人状态；
- 遥操作原始输入；
- 实际控制命令；
- 原始时间戳；
- 标定信息；
- Episode 元数据。

推荐的数据关系：

```text
Raw
 │
 ▼
Canonical Dataset
 │
 ▼
Training View
```

而不是：

```text
Raw
 │
 ▼
反复修改
 │
 ▼
唯一 cleaned dataset
```

原则：

> **Raw 用于保留事实，Derived 数据用于表达当前解释。**

---

## 3.2 时间正确优先于轨迹平滑

对于模仿学习，严重的 Observation-Action 时间错位通常比轻微噪声更危险。

需要首先明确：

```text
什么时候看到 observation
        ↓
什么时候产生 action
        ↓
什么时候发送 command
        ↓
机器人什么时候开始响应
```

因此处理优先级应为：

```text
Timestamp / Latency
        ↓
Action semantics
        ↓
Quality analysis
        ↓
Smoothing / Filtering
```

不能依赖后处理滤波补偿错误的时间关系。

---

## 3.3 Action 必须是显式接口

遥操作系统内部可能同时存在：

```text
Human / Leader / VR
        ↓
teleop_raw_action
        ↓
mapping / scaling
        ↓
policy_target_action
        ↓
constraint / IK / safety
        ↓
robot_command
```

这些信号不能笼统地都称为 `action`。

至少在 Raw 数据中，应尽量区分：

```text
teleop_raw_action
policy_target_action
robot_command
```

训练使用的 Action 应满足：

> **Policy 在训练时学习输出什么，部署时就应该被要求输出什么。**

---

## 3.4 优先改变“使用方式”，而不是修改“事实”

质量问题推荐遵循：

```text
Detect
  ↓
Annotate
  ↓
Decide
  ↓
Filter / Downsample / Reweight
```

只有在物理意义明确且能够保证时序一致的情况下，才考虑：

```text
Repair / Transform
```

因此通常优先：

```text
标记无效区间
降低采样概率
限制采样范围
```

而不是：

```text
删除 Frame
修改 Action
重写原始轨迹
```

---

## 3.5 小动作不等于噪声

Manipulation 中大量关键行为本身就是微动作：

- 接触后的姿态调整；
- 抓取前对准；
- 插孔；
- 柔性物体操作；
- 精细旋转；
- 微小夹爪调整。

因此：

```text
small action ≠ invalid action
```

仅依据 Action 幅值不足以判断数据质量。

---

# 4. 推荐的数据层级

数据系统建议分为三层。

## 4.1 L0：Raw Dataset

目标：

> 保存采集事实。

包含：

```text
sensor streams
robot state
teleoperation input
robot command
timestamps
calibration
episode metadata
```

Raw 层不承担模型训练需求。

---

## 4.2 L1：Canonical Episode Dataset

目标：

> 将设备相关数据转换成稳定、统一、可学习的数据语义。

典型逻辑结构：

```text
Episode
 ├── metadata
 ├── observation[t]
 ├── action[t]
 ├── timestamp[t]
 └── quality[t]
```

这一层需要解决：

- Episode 边界；
- 时间对齐；
- 单位统一；
- 坐标系统一；
- Action 定义；
- 数据有效性；
- 质量标注。

Canonical Dataset 应尽量与具体模型无关。

---

## 4.3 L2：Training View

目标：

> 根据具体策略构造训练样本。

例如：

```text
ACT
obs_t
  ↓
action[t : t+K]
```

```text
Diffusion Policy
obs[t-H : t]
      ↓
action[t : t+K]
```

```text
VLA
image history
+ proprio
+ language
      ↓
action chunk
```

以下操作通常属于 Training View：

- normalization；
- history window；
- action chunk；
- padding；
- image resize；
- augmentation；
- modality dropout。

这些操作不应写死在 Canonical Dataset 中。

---

# 5. 推荐的总体 Pipeline

```text
Teleoperation Recording
          │
          ▼
① Raw Integrity Check
          │
          ▼
② Episode Segmentation
          │
          ▼
③ Temporal Alignment
          │
          ▼
④ Semantic Canonicalization
          │
          ▼
⑤ Quality Analysis
          │
          ▼
⑥ Quality Annotation
          │
          ▼
⑦ Sampling Index Generation
          │
          ▼
⑧ Canonical Dataset
          │
          ▼
⑨ Dataset Split + Statistics
          │
          ▼
⑩ Training View Construction
          │
          ▼
Policy Training
```

整个流程可以进一步归纳为四个模块：

```text
Semantic Standardization
          ↓
Quality Analysis
          ↓
Sampling Policy
          ↓
Training Transformation
```

这四个模块应尽量解耦。

---

# 6. Episode 与时间语义

## 6.1 Episode 应对应真实任务过程

建议区分：

```text
reset
  ↓
operator preparation
  ↓
task start
  ↓
demonstration
  ↓
success / failure / abort
  ↓
task end
```

Reset、等待和采集准备阶段不应该依赖固定帧数猜测，而应尽可能由明确事件定义。

---

## 6.2 Episode 状态应显式保存

建议至少支持：

```text
success
failure
aborted
invalid
unknown
```

失败轨迹不必从 Canonical Dataset 中删除。

是否参与训练属于：

```text
Sampling / Training Policy
```

而不是：

```text
Raw Data Policy
```

---

## 6.3 时间对齐必须显式

首先建立统一时间轴：

```text
t0, t1, t2, ... , tN
```

然后把不同数据源映射到该时间轴。

典型策略可以包括：

```text
Image
→ nearest valid frame

Continuous robot state
→ nearest / interpolation

Discrete state
→ zero-order hold

Control command
→ 根据实际 command timestamp 对齐
```

建议同时保留：

```text
source_timestamp
aligned_timestamp
alignment_error
```

原则：

> **允许时间误差存在，但不能让时间误差不可见。**

---

# 7. 质量处理的统一机制

停止、抖动、漂移不应该分别设计三个互不相关的“清洗算法”。

推荐使用统一框架：

```text
Signal / Segment
      │
      ▼
Detect
      │
      ▼
Classify
      │
      ▼
Annotate
      │
      ▼
Sampling Decision
      │
      └────必要时────► Repair
```

其中：

**Detect**

判断是否存在异常模式。

**Classify**

判断它更可能属于：

```text
task behavior
operator behavior
sensor noise
teleop system artifact
controller artifact
invalid data
```

**Annotate**

记录事实或评分。

**Sampling Decision**

决定：

```text
keep
downsample
reweight
exclude
```

**Repair**

只处理少数物理意义明确的问题。

---

# 8. Stop / Idle 的处理

## 8.1 Idle 不等于无效

停止可能代表：

### 有意义停止

例如：

- 接触稳定；
- 抓取完成；
- 等待物体响应；
- Task completion；
- 本身就应该保持静止。

### 低价值停止

例如：

- 操作者思考；
- 调整遥操作设备；
- 与采集任务无关的等待；
- Episode 开始前长时间静止；
- Episode 完成后的停留。

因此不能使用：

```text
velocity ≈ 0
      ↓
delete
```

这样的单一规则。

---

## 8.2 使用持续状态，而不是单帧阈值

建议构造抽象的：

```text
motion_score(t)
```

它可以综合：

```text
state change
action change
gripper event
task-relevant signal
```

当低运动状态持续达到一定时间后，才认为进入：

```text
persistent idle
```

判断重点是：

```text
幅值
+
持续时间
+
上下文
```

而不是单独的 Action magnitude。

---

## 8.3 不建议删除后重新拼接时间序列

原始数据：

```text
motion A
   ↓
long idle
   ↓
motion B
```

如果删除 idle：

```text
motion A
   ↓
motion B
```

就人为制造了不存在的时间邻接关系。

对需要 temporal context 或 action horizon 的策略，这种处理尤其危险。

推荐：

```text
Episode 保持原结构
        │
        ▼
生成 Valid Sampling Range
```

---

## 8.4 推荐通过采样控制 Idle 比例

例如：

```text
有效运动            高概率
短暂停顿            正常概率
长时间 Idle         低概率
确定无效 Idle       不采样
```

核心思想：

> **控制训练分布，而不是粗暴改变轨迹本身。**

---

# 9. Jitter 的处理

## 9.1 Jitter 首先是系统诊断问题

常见来源：

```text
VR tracking noise
encoder noise
human tremor
network jitter
teleop mapping
controller instability
```

因此应先判断抖动出现在哪一层：

```text
teleoperator
    ↓
mapping
    ↓
command
    ↓
robot state
```

这比直接对最终数据做低通更重要。

---

## 9.2 默认不建议离线覆盖式平滑

简单低通可能造成：

- 相位延迟；
- Observation-Action 错位；
- 急停被钝化；
- 微动作被消除；
- 接触事件被污染。

因此默认推荐：

```text
raw signal
    ↓
jitter metrics / annotation
    ↓
sampling / QC decision
```

而不是：

```text
raw signal
    ↓
low-pass
    ↓
overwrite
```

---

## 9.3 如果必须滤波，优先在控制接口完成

理想结构：

```text
teleop_raw
    ↓
明确的 input processor
    ↓
policy_target
    ↓
robot controller
```

同时记录：

```text
raw input
processed target
actual command
```

这样既能改善机器人控制质量，又不会丢失原始信息。

---

# 10. Drift 的处理

Drift 通常指持续、缓慢累积的非预期变化。

例如：

```text
0
0.2 mm
0.4 mm
0.7 mm
1.1 mm
...
```

可能来源于：

- 标定误差；
- Tracking drift；
- Sensor bias；
- Mapping 积分误差；
- Operator 无意识移动。

---

## 10.1 Drift 优先检测和归因

推荐：

```text
detect
  ↓
flag
  ↓
analyze source
```

而不是：

```text
small motion
  ↓
force to zero
```

因为小幅持续动作也可能是任务行为。

---

## 10.2 系统性 Drift 应优先修采集系统

如果多个 Episode 中出现：

```text
方向一致
速度相似
长期累积
```

更可能是：

```text
calibration
tracking
mapping
controller
```

等系统问题。

原则：

> **能够在数据源解决的问题，不应长期依赖离线数据修复。**

---

# 11. 硬异常与软异常

数据问题可以简化成三类。

| 类别 | 示例 | 默认处理 |
|---|---|---|
| **硬异常** | NaN、严重掉帧、Timestamp 错乱、损坏视频、Action 语义未知 | Exclude |
| **软异常** | 长 Idle、轻微抖动、疑似 Drift、Operator hesitation | Annotate + Downsample / Reweight |
| **正常复杂行为** | 急停、微动作、接触调整、短暂停顿 | Keep |

这一区分非常重要：

> **轨迹不平滑，不等于轨迹无效。**

---

# 12. Quality Annotation

推荐质量信息与物理数据分离。

例如：

```yaml
quality:
  timing_valid: true
  observation_valid: true
  action_valid: true

  motion_state: moving
  idle_state: false

  jitter_level: low
  drift_suspected: false

  task_phase: manipulation

  sampling_valid: true
```

不需要一开始设计大量复杂 score。

对于多数项目，一个简单可靠的：

```text
valid
motion_state
idle
jitter_flag
drift_flag
```

往往已经足够。

只有当这些指标确实影响训练决策时，再增加连续评分。

原则：

> **不要为了“数据质量体系完整”而制造无实际用途的指标。**

---

# 13. Valid Sampling Range

序列模型中，一个 Frame 有效并不代表它能够成为有效训练样本。

假设：

```text
history length = H
action horizon = K
```

则以 `t` 为训练起点时，通常需要保证：

```text
[t-H, t+K]
```

整个窗口满足训练要求。

因此需要区分：

```text
frame_valid
```

和：

```text
sample_start_valid
```

推荐离线生成：

```text
valid_sampling_ranges
```

例如：

```text
Episode 001

[20, 150]
[215, 380]
```

训练阶段只需要：

```text
choose episode
      ↓
choose valid range
      ↓
sample t
```

而无需重复执行复杂 QC。

这是整个 Pipeline 中非常重要的效率机制。

---

# 14. Dataset Split 与统计量

Dataset split 至少应以：

```text
Episode
```

为单位。

如果重点评估泛化能力，则应进一步考虑：

```text
scene
object
operator
collection session
robot configuration
```

避免高度相似的连续数据同时出现在 Train 与 Validation 中。

Normalization 统计量只能使用：

```text
Train Split
```

计算。

流程：

```text
Canonical Dataset
      ↓
Train / Val / Test Split
      ↓
Train Statistics
      ↓
Training Normalization
```

Normalization 属于训练变换，而不是数据清洗。

---

# 15. Training View

Canonical Dataset 应保存物理数据：

```text
observation
action
timestamp
task
quality
```

Training View 再根据模型产生：

```text
history window
action chunk
normalization
padding mask
image augmentation
modality dropout
```

这样同一份数据可以服务：

```text
BC
ACT
Diffusion Policy
VLA
World Model
```

而无需建立多个互不兼容的数据集版本。

---

# 16. 最小可用清洗机制

对于一个新项目，不建议一开始实现复杂的数据质量系统。

一个高收益、低复杂度的最小版本只需要：

```text
1. Episode boundary check

2. Timestamp / FPS check

3. Observation-Action alignment

4. Action semantics validation

5. NaN / missing / corrupted data detection

6. Persistent idle detection

7. Valid sampling range generation

8. Episode-level success / invalid metadata

9. Train / Val split

10. Train statistics
```

这一版本已经可以解决大多数高风险数据问题。

之后再根据实际现象增加：

```text
jitter analysis
drift analysis
task-phase annotation
sampling reweighting
```

这比一开始设计复杂的“自动轨迹清洗算法”更加稳健。

---

# 17. 数据处理优先级

## P0：语义正确性

```text
Episode boundary
Timestamp
Observation-Action alignment
Action semantics
Units
Coordinate frame
Calibration
```

错误会直接产生错误监督。

---

## P1：数据有效性

```text
Corrupted data
Missing frame
Timing anomaly
Invalid Episode
Persistent Idle
Valid Sampling Range
```

主要决定哪些样本能够可靠使用。

---

## P2：数据分布优化

```text
Idle downsampling
Jitter annotation
Drift annotation
Task-phase balancing
Sampling reweighting
```

主要改善训练数据分布。

---

## P3：模型特定变换

```text
Normalization
History
Action chunk
Augmentation
Padding
```

属于 Policy Adapter。

推荐始终遵循：

```text
P0 → P1 → P2 → P3
```

不要在 P0 尚未可靠时投入大量精力设计高级滤波算法。

---

# 18. 应避免的处理方式

### 18.1 小 Action 直接删除

```text
|action| < threshold
      ↓
delete
```

会误删精细 manipulation 行为。

---

### 18.2 Idle 删除后重新拼接轨迹

会制造虚假的时间连续性。

---

### 18.3 对所有 Action 统一低通

可能导致 Action latency 和接触行为失真。

---

### 18.4 覆盖 Raw 数据

使后续无法重新验证清洗策略。

---

### 18.5 用“平滑程度”定义 Demonstration 质量

好的 manipulation demonstration 完全可能包含：

```text
急停
方向突变
接触
短暂停顿
微动作
```

这些并不是天然的数据问题。

---

### 18.6 过度设计 Quality Score

质量指标只有在最终能够影响：

```text
filtering
sampling
analysis
```

时才有价值。

没有实际决策用途的复杂指标会增加系统维护成本。

---

# 19. 推荐的软件抽象

从工程实现角度，可以将整个系统保持为五个相对独立的组件：

```text
Recorder
   │
   ▼
Canonicalizer
   │
   ▼
Quality Analyzer
   │
   ▼
Sampling Index Builder
   │
   ▼
Policy Dataset Adapter
```

职责分别为：

```text
Recorder
→ 保留事实

Canonicalizer
→ 建立统一语义

Quality Analyzer
→ 描述质量状态

Sampling Index Builder
→ 决定哪些位置值得训练

Policy Dataset Adapter
→ 构造模型需要的数据
```

这种拆分具有几个优点：

- 数据清洗策略可以独立迭代；
- 更换 Policy 不需要重新清洗数据；
- 更换采样策略不需要重新生成视频或状态数据；
- 可以方便进行数据处理 Ablation；
- 容易追踪训练结果来自哪一版数据规则。

---

# 20. 版本与可追溯性

建议记录：

```text
dataset_version
schema_version
processing_version
calibration_version
sampling_policy_version
statistics_version
```

并尽量保证：

```text
Raw Dataset
+
Processing Config
+
Processing Code Version
=
Canonical Dataset
```

从而使数据处理具有：

```text
reproducibility
traceability
comparability
```

---

# 21. 最终原则总结

整个数据系统可以浓缩为以下逻辑：

```text
保留原始事实
      ↓
建立正确的时间与物理语义
      ↓
检测数据质量状态
      ↓
尽量不修改原始轨迹
      ↓
通过 Sampling 控制数据使用方式
      ↓
根据 Policy 动态构造 Training View
```

其中最关键的判断准则是：

> **数据清洗不是让轨迹变得更漂亮，而是避免模型学习到错误的因果、时间和动作关系。**

对于 Stop、Jitter、Drift：

```text
Stop
→ 判断是否为持续且低价值的 Idle
→ 优先调整采样

Jitter
→ 优先定位信号来源
→ 默认保留 Raw，必要时在控制接口处理

Drift
→ 优先检测和归因
→ 系统性问题优先修采集系统
```

因此，一个简洁、高效的数据清洗系统，不需要大量复杂的轨迹修复算法。

真正高价值的机制是：

```text
Timestamp correctness
+
Action semantics
+
Persistent Idle Detection
+
Quality Annotation
+
Valid Sampling Range
+
Training-time Transformation
```

这六项构成了机器人遥操作数据进入策略学习之前最值得优先建设的数据基础设施。