# 在本地用 Claude Code 执行

本目录只提供任务材料，尚未实施生产代码。默认流程：取任务分支 → 建本地实现分支 → 激活现有项目环境 → 启动交互式 Claude Code → 审核本地 commits。无需先合并任务 PR；不要在任务文档分支或 main 上直接开发。

## 1. 文件用途

- `TASKBOOK.md`：唯一执行任务书，含 T0–T7、V01–V22、14 个 examples、数据保护和提交合同。
- `PROMPT.md`：启动及跨会话恢复实施用的完整指令；会先识别已有进度。
- `REVIEW_PROMPT.md`：实施结束后的独立只读审查指令。
- `RUNBOOK.md`：本文。不要把本目录复制进根 CLAUDE.md/AGENTS.md，也不需要新增 hooks、skills、agents 或权限配置。

## 2. 获取任务并创建实现分支

在现有 `dexmani_real` 仓库根目录操作。先确认 `git remote get-url origin` 指向 `haoyangzhanglab/dexmani_real`（或明确包含任务分支的已确认 remote），不要自动修改 remote。以下以 Bash/Zsh 和 `origin` 为例。

工作树干净、实现分支尚不存在时：

```bash
if [ -n "$(git status --porcelain)" ]; then
  printf '%s\n' '工作树或暂存区有修改：请先人工处理，或使用下方独立 worktree 路线；不要自动清理。'
else
  git fetch origin docs/claude-runtime-refactor-v2 &&
  git switch --no-track -c refactor/research-runtime-v2 FETCH_HEAD
fi
```

`-c` 在同名实现分支已存在时会停止，不会覆盖它。恢复任务时使用已有正确分支，不改成 `-C` 或 reset。执行分支从任务提交开始，Claude 会记录实际 IMPLEMENTATION_BASE 并核对与审查代码基线的差异；不得自动回滚较新的实现。若要同时纳入之后的 main 变化，先由维护者检查并正常合并/解决冲突，再启动；不自动重写用户历史。

原工作树需保留时，可在人工确认目标路径/新分支名未占用后使用独立 worktree：

```bash
git fetch origin docs/claude-runtime-refactor-v2 &&
git worktree add -b refactor/research-runtime-v2 ../dexmani_real-runtime-v2 FETCH_HEAD
```

成功后进入新 worktree。不要自动删除冲突目录。新 worktree 不含原目录 ignored 的 raw/环境/产物；把真实 raw 的绝对路径作为只读输入告知 Claude，不复制全数据集、不临时改写 raw、不要误把“新目录没有数据”当成原数据丢失。检查相邻 `dexmani_policy` 路径和 Python editable install 实际指向；验证必须作用当前 worktree，而不是旧 checkout。

## 3. 启动实施

先由你激活已有的项目 Python/Conda 环境并确认 `python --version`、`claude --version`。这里不猜环境名，也不自动安装/升级机器人 SDK。停止其他正在占用本工作树的代码写入任务；本次执行不需要连接机器人。

在正确实现分支的根目录运行：

```bash
claude --permission-mode default "$(cat .claude/tasks/research-runtime-refactor/PROMPT.md)"
```

这是交互模式，可处理正常工具权限确认。默认不使用 `-p`、危险权限跳过、全 Bash 白名单或无人值守循环。显式 `default` 避免继承其他起始模式，但已有 allow rules/hooks/组织策略仍生效；启动前人工核对本机配置。任何权限模式都不是机器人安全隔离。

在已隔离、确认没有可触达真机的开发环境，需要减少逐文件编辑确认时，可以将上面的模式改成 `acceptEdits`。它只改变权限处理，不授权真机、不保证所有 shell 命令都自动批准。本任务不要求修改任何 `.claude/settings*.json`。

若真实 raw 不在默认目录，可在会话初始消息中追加：

```text
本地 raw 位于 <实际绝对路径>，仅允许只读盘点/preflight；已有 derived/checkpoint 位于 <实际绝对路径>，不得覆盖。当前 Python 环境为 <实际环境>。不要执行全量转换或硬件程序。
```

将尖括号替换为真实信息；没有对应目录则明确说未提供，不编造。任务书允许在数据/依赖/容差条件缺失时继续其他安全代码子项，但不能把受阻验证写成已通过。

## 4. 执行过程与恢复

Claude 应先读取任务和本地事实，简短列出执行顺序，然后逐阶段实施/离线验证/本地提交，不停在方案描述。实现期间仍可询问真正缺少的条件或权限；不要通过一键放行所有 Bash 解决。

每阶段的本地证据位置：

```bash
git rev-parse --git-path research-runtime-refactor/STATUS.md
```

此文件及相关日志不提交。它记录 IMPLEMENTATION_BASE、阶段、文件、验证、commit、阻塞、下一步；未完成任务不会靠聊天“记住”。

中断后先确认所在分支/工作树没有被其他人改变。在同一目录选择原会话：

```bash
claude --permission-mode default --resume
```

也可开新会话重跑第 3 节的 PROMPT；它会核实 STATUS/Git 后恢复，不能盲目重复已完成迁移或修改。不要无条件 `--continue` 到可能属于另一任务的最新会话。保持同一个主写入者。

## 5. 最终只读审查与修复

实施阶段结束后启动独立只读审查：

```bash
claude --permission-mode plan "$(cat .claude/tasks/research-runtime-refactor/REVIEW_PROMPT.md)"
```

审查核对实际 diff/源码/验证证据，不因为 plan mode 就将未经检查的脚本视为安全。审查输出不是自动真机认证。

若有有依据的 findings，回原实施会话输入：

```text
按刚才独立审查中的具体 findings 逐项核实并最小修复，继续遵守 TASKBOOK；不要借机新增框架或重新引入已删 gate。对缺证据的项先核对，不盲目改。更新相关 tests/docs/comments，运行受影响离线验证，显式 stage 后创建后续修复 commit；不 amend、不 push。最后更新 STATUS 和完整完成矩阵。
```

## 6. 验证、提交与交付检查

生产测试需先由 Claude 检查 imports/执行路径。基本离线命令为：

```bash
python -m compileall -q dexmani_real examples
python -m unittest discover -s tests -p 'test_*.py' -v
git diff --check
git status --short
```

它们不是全部验收，也不代表本任务材料已经运行过项目测试。必须同时覆盖 TASKBOOK 的 V01–V22 和 EX01–EX14。缺依赖的 skip、baseline failure、未测硬件/数据分别报告；用 `tee` 时保留原退出码。不要批量执行 examples/--help，尤其 `xhand_control_example.py` 当前会进入硬件流程。

提交后还要核对 IMPLEMENTATION_BASE→HEAD 的完整 diff；只查看干净工作树没有意义。只 stage 明确代码/必要文档/测试文件，不上传 raw、checkpoint、calibration 私有产物、机器日志或凭证。不绕过 hooks/改 author，不擅自 push/merge。

最终要求 Claude 给出实际本地 commit SHA、变更与删除机制、14 入口适配结论、命令和 pass/fail/skip、原数据不变及兼容结果、最终 Git 状态和具体条件项。手工 review 后再决定是否 push 或真机验收；这两者都不在默认启动指令的授权范围内。

## 7. CLI 依据与材料生命周期

上面的交互初始 prompt、`--permission-mode default/acceptEdits/plan` 与 `--resume` 用法于 2026-09-20 核对 Claude Code 官方文档：

- CLI reference: https://code.claude.com/docs/en/cli-reference
- Permissions: https://code.claude.com/docs/en/permissions
- Best practices: https://code.claude.com/docs/en/best-practices

本机版本/组织策略可能不同，以实际 CLI 错误和已安装文档为准；不要为绕过不支持选项自动升级或关闭权限。任务完成后由维护者决定归档本目录；一次性状态/测试日志不进入永久 README/AGENTS。
