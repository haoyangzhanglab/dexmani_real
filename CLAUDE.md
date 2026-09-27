# CLAUDE.md — Claude 工作入口

本文件是 Claude 在 DexMani Real 中工作的入口。开始修改前先阅读 [README.md](README.md) 理解研究定位，再阅读 [AGENTS.md](AGENTS.md)；AGENTS 是本仓库关于安全、科研数据与代码修改的共享长期规则。

默认使用**中文**进行分析、计划、文档和结果说明。

## 工作目标

DexMani Real 是个人 PhD 真实机器人灵巧操作研究仓库。你的任务是帮助研究者更可靠地完成实验和验证研究假设，而不是把代码库扩展成通用机器人平台。

优先级始终是：

```text
真实硬件安全
> 实验正确性与科学可信度
> 数据可追溯 / 可复现
> 研究迭代效率
> 代码清晰
> 通用工程化
```

## 开始任务时

1. 从用户的研究目标出发理解需求，不要只根据当前实现做局部修补。
2. 检查相关入口、配置和真实数据 / 控制链，确认实际行为。
3. 区分当前实验需要的核心逻辑与历史遗留机制。
4. 优先采用最小、直接、可解释的修改。
5. 跨 `dexmani_real` / `dexmani_policy` 时，先明确两个仓库各自应拥有的责任。

## 必须遵守的边界

### 不擅自操作真实硬件

没有用户明确授权时，不运行任何可能连接设备、运动机器人、采集实时传感器、写标定、回放轨迹或执行 policy rollout 的命令。

如果无法确定是否有硬件副作用，按有副作用处理。

### 不修改实验事实

Raw 数据是实验事实，不为了让训练或结果“更漂亮”而改写、补齐或删除失败证据。训练缓存属于可重建派生物，应从 Raw 和明确的处理配置生成。

### 不把标定写成 ABI

camera extrinsics、desk plane、hand mount、VR alignment 等描述当前实验现场。它们应在新的真实实验中重新标定或确认，而不是作为跨实验永久契约去比较历史数值。

### 不做无需求的平台化

没有当前研究需求时，不新增通用 schema、registry、capability system、版本兼容框架、migration layer 或多层 wrapper。旧机制已经失效时，优先删除。

### 保持跨仓库职责清楚

`dexmani_real` 负责真实硬件、真实数据、现场标定和真机 deployment boundary；`dexmani_policy` 负责模型、训练与通用策略逻辑。不要复制另一侧的内部实现来“方便”。

## 修改方式

对代码任务：

- 先追踪入口到副作用；
- 对安全或数据语义敏感的改动说明影响；
- 只修改与任务相关的文件；
- 优先删除重复 / legacy 逻辑；
- 保持接口简单；
- 用当前源码和 resolved config 作为实现事实。

对文档任务：

- README 只写长期稳定的研究信息；
- AGENTS 写代理协作原则；
- CLAUDE 只保留 Claude 特有的执行入口；
- 临时 task plan、验收过程和实现历史不要沉积到长期文档。

## 验证与汇报

默认执行离线、无硬件副作用的检查。可优先使用：

```bash
python -m compileall -q dexmani_real examples
ruff format --check dexmani_real examples
ruff check dexmani_real examples
git diff --check
```

再根据改动补充针对性的纯逻辑验证。

任务完成时，向用户清楚说明：

- 改了什么；
- 为什么这样改；
- 做了哪些离线验证；
- 哪些真机行为没有验证；
- 是否存在需要研究者现场确认的实验假设。

不要把 compile、lint、unit smoke 或仿真通过描述成真机安全验证。
