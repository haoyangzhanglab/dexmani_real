# CLAUDE.md

Claude 在本仓库工作前应先阅读 [README.md](README.md) 和 [AGENTS.md](AGENTS.md)。

默认使用**中文**分析、计划和汇报。代码标识符与必要技术术语保留英文。

## Repository Context

DexMani Real 是个人 PhD 的真实机器人灵巧操作研究代码：

```text
calibration
→ teleoperation / recording
→ Raw data
→ offline processing
→ dexmani_policy
→ real-robot deployment
```

目标是支撑可靠的真实实验，不是建设通用机器人 framework。

## Working Rules

1. 从研究目标出发，再看实现。
2. 修改前追踪相关 entry point、config、data/control flow 和 side effect。
3. 优先最小、直接、可解释的实现；obsolete logic 优先删除。
4. `dexmani_real` 保持 real-world boundary，模型和训练逻辑留在 `dexmani_policy`。
5. Raw 是实验 evidence；processed dataset 是可重建训练缓存。
6. camera extrinsics、desk plane、hand mount、VR alignment 是当前实验标定，不是永久 ABI。
7. 不为潜在未来需求新增 schema / registry / migration / compatibility framework。

## Hardware

没有用户明确授权时，不执行：

- robot motion / home / teleoperation；
- replay / policy rollout；
- real sensor acquisition；
- hardware diagnostics；
- calibration write。

不确定是否有硬件副作用时，按有副作用处理。

离线检查通过不代表真机验证通过。

## Validation

优先运行与改动匹配的 offline checks：

```bash
python -m compileall -q dexmani_real examples
ruff format --check dexmani_real examples
ruff check dexmani_real examples
git diff --check
```

不要为了检查而擅自修改实验机依赖环境。

## Final Report

完成后简要说明：

- changed；
- rationale；
- offline validation；
- 未执行的 hardware validation；
- 仍需研究者现场确认的实验假设。

其余规则以 [AGENTS.md](AGENTS.md) 为准，避免在本文件重复维护。
