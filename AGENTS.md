# AGENTS.md

DexMani Real 是个人 PhD 真实机器人灵巧操作研究仓库。本文定义代码代理在本仓库中的长期工作约束。

默认使用**中文**进行分析、计划、文档和结果说明；代码标识符与必要技术术语保留英文。

## Scope

`dexmani_real` 负责：

- real robot / sensor integration；
- calibration；
- teleoperation 与 data collection；
- Raw experiment data；
- offline processing；
- learned policy 的真实机器人 deployment 与 evaluation；
- 与真实执行相关的 safety boundary。

模型结构、通用 Dataset、训练流程与算法实验原则上属于 `dexmani_policy`。

## Priorities

按以下顺序做工程取舍：

```text
hardware safety
> scientific correctness
> data traceability / reproducibility
> research iteration speed
> code simplicity
> generic extensibility
```

这是研究代码。没有当前实验需求时，不为潜在未来场景增加 framework、registry、schema system、migration layer 或兼容分支。

## Before Editing

开始修改前：

1. 明确当前研究 / 实验目标；
2. 检查相关入口、配置与调用链；
3. 追踪 producer → transform → consumer → side effect；
4. 区分当前有效行为与 legacy implementation；
5. 保留与任务无关的现有修改。

具体行为以当前源码和 resolved configuration 为准，不根据旧文档或命名猜测。

## Hardware Safety

除非用户明确授权，不执行任何可能连接或改变真实设备状态的命令，包括：

- xArm / XHand motion、home、teleoperation；
- replay、policy rollout；
- RealSense / VR / HTS 实时连接与采集；
- hardware diagnostics；
- calibration write。

如果无法确定一个命令是否有硬件副作用，按“有”处理。

普通 import、配置解析和纯离线数据处理应保持无硬件副作用。离线 compile、lint、smoke test 或仿真结果不能作为真机安全证明。

涉及 motion、limits、collision、emergency stop、sensor freshness、resource shutdown 的改动必须保守处理，并在结果中说明真机验证状态。

## Data

### Raw

Raw 是实验 evidence。Raw immutability 从 capture 成功发布为 Raw 后开始。

对已发布 Raw：

- 保持 immutable，不原地修改；
- 不伪造缺失 observation / action；
- 不为了训练方便删除失败证据或重写 termination；
- demonstration 和 rollout 都保留真实实验边界；
- processed data 从 Raw 派生，训练筛选不回写 Raw。

停止、晚帧、控制失败和 technical-invalid capture 保留已采前缀及结束原因；
仅操作者显式重录/丢弃才删除当前 capture。writer/编码/发布失败保留 staging，
不能将损坏文件报告为成功 Raw；已发布 Raw 仍不可变。
采集保留全部已接入物理模态，公共导出保留全部 13 字段，下游按模型所需字段加载。
辅助缺测在 Raw 中保留 NaN/有效性与 dispatch，不冒充有效零值。导出拒绝不完整 episode，不删除或改写对应 Raw。

### Processed Data

Canonical / processed dataset 是可重建训练缓存。

- processing 必须明确、可追溯；
- 公共导出要求每段全部 13 字段完整且浮点数值有限，不完整 episode 整段拒绝并记录原因，原 Raw 保留；实际模型输入仍检查其所需数值；
- 不静默补模态、删坏帧、插值或改变时间语义，除非这是明确研究设计；
- 处理逻辑实质变化时优先从 Raw 重新导出；
- 不默认维护旧缓存 migration / compatibility。

## Calibration

camera extrinsics、desk plane、hand mount、VR alignment 等是**当前实验状态**，不是跨实验 ABI。

因此：

- 物理设置变化后重新标定或确认；
- 不将一次历史标定值硬编码为长期契约；
- 不用历史标定数值判断新的实验是否“兼容”；
- 训练 artifact 只保存真正与训练 numerical representation 相关的信息；
- 桌面、安装/运动学使用代码当前配置，不要求采集快照或历史一致性准入；
- 触觉导出直接使用采集保存值，不重复扣偏置；
- 相机保留数据对应的内外参和 depth scale，缺外参不阻止保存物理观测；
- 真机运行使用当前实验环境的有效标定。

## Architecture

保持真实责任边界清楚：

```text
hardware / sensors
      ↓
runtime / teleop / recording
      ↓
Raw
      ↓
dataset processing
      ↓
dexmani_policy
      ↓
deployment
      ↓
real robot
```

优先：

- 删除 obsolete path；
- 合并重复逻辑；
- 缩小跨模块 / 跨仓库接口；
- 在真实 hardware、process、data boundary 处保留必要抽象。

避免：

- 单一实现上的多层 wrapper；
- 重复 semantic dictionaries；
- 无需求的 registry / capability framework；
- 为历史兼容长期复杂化当前研究路径；
- 把一次性 task plan 写成永久架构。

## Changing Runtime or Deployment

修改 `robot/`、`sensor/`、`teleop/`、`runtime/`、`replay/`、`deployment/` 时，检查是否改变：

- device connection timing；
- motion authorization / revocation；
- pause / stop / fault / emergency behavior；
- joint target、frame、unit 或 timing semantics；
- stale command / stale observation handling；
- recorded data 与实际执行之间的对应关系。

不要在没有明确研究动机时加入额外 smoothing、fallback、auto-recovery 或控制补偿；这些机制会改变实验系统本身。

## Changing Data

修改 `recording/`、`dataset/` 或 export workflow 时，确认：

- 数据来自哪个实验时刻；
- frame / unit / ordering 是否变化；
- missing data 如何表示；
- action 表示 command、published target 还是 measured state；
- Raw 是否仍不可变；
- processed data 是否仍可由 Raw 重建。

不要为了让训练代码通过而把无效真实数据填成看似正常的值。

## Code Style

- 代码保持 direct、explicit、readable；
- comments 解释 robotics / math / frame / unit / safety rationale，而不是复述控制流；
- 没有明确收益时不增加依赖；
- 简单 helper 不升级为 subsystem；
- 与当前任务无关的重构不要顺手进行。

## Documentation

- `README.md`：项目定位、workflow、quick start、主要入口。
- `AGENTS.md`：代码代理的长期工作规则。
- `CLAUDE.md`：Claude 的精简入口，不重复整份 AGENTS。
- 临时 task / migration / acceptance notes 不作为长期根目录文档；用户明确指定的本次根目录任务书保留。

README 不维护容易过时的类清单、字段级 runtime contract、状态机细节或历史重构说明。

## Offline Validation

默认只运行无硬件副作用的检查：

```bash
python -m compileall -q dexmani_real examples
git diff --check
```

本仓库不使用 Ruff 或 pytest。不要运行或重新引入这两个工具，也不要生成 `.ruff_cache/` 或 `.pytest_cache/`。

根据改动补充 focused pure-logic checks，例如 geometry、FK / IK、config、dataset processing 或 observation construction。

缺少可选依赖时说明限制，不擅自升级真实实验环境。

## Completion

完成任务前确认：

- 修改直接服务当前研究目标；
- 没有无必要的新抽象；
- 没有 dead code、stale docs 或 legacy alias；
- 没有意外改变 data / action / frame / timing semantics；
- 真机未验证的部分已明确说明；
- final diff 只包含预期修改。
